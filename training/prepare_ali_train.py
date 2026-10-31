"""Turn the AliMeeting training archives into training data, without unpacking them.

Reads Train_Ali_far.tar.gz and Train_Ali_near.tar.gz as streams, one file at a time, and
keeps only what training needs:
    <out>/far/<meeting>.wav          channel 0 of the table microphone array (int16)
    <out>/near/<meeting>_<SPK>.wav   each person's headset
    <out>/ref/<meeting>.stm          who said what when (from the human TextGrids)
So 96 GB of archives become about 35 GB, and nothing else is ever written to disk.
Files are written by their base name only, so an archive can't write anywhere else.

    python training/prepare_ali_train.py data/alimeeting/Train_Ali_near.tar.gz data/ali_train
    python training/prepare_ali_train.py data/alimeeting/Train_Ali_far.tar.gz data/ali_train
Safe to rerun: finished files are skipped.
"""

from __future__ import annotations

import argparse
import io
import sys
import tarfile
import time
from pathlib import Path

# The TextGrid reader is shared with the benchmark's AliMeeting converter.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
from make_alimeeting import clean, read_textgrid  # noqa: E402

SR = 16_000


def textgrid_to_stm(raw: bytes, session: str, tmp: Path) -> str:
    tmp.write_bytes(raw)
    lines = []
    for tier, start, end, text in sorted(read_textgrid(tmp), key=lambda x: x[1]):
        words = clean(text)
        if words and end > start:
            speaker = tier.split("_")[-1] or tier
            lines.append(f"{session} 1 {speaker} {start:.3f} {end:.3f} {words}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("archive", type=Path, help="Train_Ali_far.tar.gz or Train_Ali_near.tar.gz")
    ap.add_argument("out", type=Path)
    args = ap.parse_args()

    import soundfile as sf

    far = "_far" in args.archive.name
    for sub in ("far", "near", "ref"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    tmp = args.out / ".textgrid.tmp"
    done = skipped = 0
    t0 = time.time()
    with tarfile.open(args.archive, "r|gz") as tar:  # stream: one member at a time
        for member in tar:
            if not member.isfile():
                continue
            name = Path(member.name).name  # base name only
            stem = Path(name).stem
            if name.lower().endswith(".textgrid"):
                if not far:
                    continue  # the far-field TextGrids cover everyone in the meeting
                target = args.out / "ref" / f"{stem}.stm"
                if not target.exists():
                    source = tar.extractfile(member)
                    assert source is not None
                    target.write_text(textgrid_to_stm(source.read(), stem, tmp), encoding="utf-8")
                continue
            if not name.lower().endswith(".wav"):
                continue
            if far:  # R0003_M0046_MS002.wav -> R0003_M0046.wav
                session = "_".join(stem.split("_")[:2])
                target = args.out / "far" / f"{session}.wav"
            else:  # R0003_M0046_F_SPK0093.wav -> R0003_M0046_SPK0093.wav
                parts = stem.split("_")
                target = args.out / "near" / f"{parts[0]}_{parts[1]}_{parts[-1]}.wav"
            if target.exists():
                skipped += 1
                continue
            source = tar.extractfile(member)
            assert source is not None
            audio, sr = sf.read(io.BytesIO(source.read()), dtype="int16", always_2d=True)
            if sr != SR:
                raise SystemExit(f"{name} is {sr} Hz; expected {SR}")
            partial = target.with_suffix(".part.wav")
            sf.write(partial, audio[:, 0], SR, subtype="PCM_16")
            partial.replace(target)  # only complete files get the final name
            done += 1
            if done % 10 == 0:
                print(f"  {done} files ({(time.time() - t0) / 60:.0f} min)", flush=True)
    tmp.unlink(missing_ok=True)
    print(f"Done: {done} written, {skipped} already there, in {args.out}")


if __name__ == "__main__":
    main()
