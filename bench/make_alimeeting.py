"""Turn AliMeeting (real Mandarin meetings, OpenSLR 119, CC-BY-SA-4.0) into test recordings.

AliMeeting has 2-4 people per meeting, 15-30 minutes each, recorded in real rooms with an
8-channel microphone array in the middle of the table, and lots of overlapping speech.
We use one array channel (the standard "single distant microphone" setup, like a phone on
the table) and the human transcripts in the TextGrid files.

Download (from Alibaba Cloud, Shanghai) and unpack:
    mkdir -p data/alimeeting && cd data/alimeeting
    wget -c https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Eval_Ali.tar.gz
    wget -c https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Test_Ali.tar.gz
    tar xzf Eval_Ali.tar.gz && tar xzf Test_Ali.tar.gz

Then:
    python bench/make_alimeeting.py data/alimeeting/Eval_Ali --out data/ali_eval
    python bench/make_alimeeting.py data/alimeeting/Test_Ali --out data/ali_test

Writes the layout bench/run_bench.py expects: <out>/audio/<id>.wav, <out>/ref/<id>.rttm/.stm
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

SR = 16_000
# Transcript markers that are not words: <sil>, [laugh], etc.
_MARKERS = re.compile(r"<[^>]*>|\[[^\]]*\]")


def read_textgrid(path: Path) -> list[tuple[str, float, float, str]]:
    """Parse a long-format Praat TextGrid: (tier name, start, end, text) per interval."""
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-16", "gbk"):
        try:
            lines = raw.decode(enc).splitlines()
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Can't decode {path}")

    out = []
    tier, in_interval = "", False
    start = end = 0.0
    for line in lines:
        s = line.strip()
        if s.startswith("item ["):
            in_interval = False
        elif s.startswith("name =") and not in_interval:
            tier = s.split("=", 1)[1].strip().strip('"')
        elif s.startswith("intervals ["):
            in_interval = True
        elif in_interval and s.startswith("xmin ="):
            start = float(s.split("=", 1)[1])
        elif in_interval and s.startswith("xmax ="):
            end = float(s.split("=", 1)[1])
        elif in_interval and s.startswith("text ="):
            text = s.split("=", 1)[1].strip()
            if text.startswith('"') and text.endswith('"') and len(text) >= 2:
                text = text[1:-1]
            out.append((tier, start, end, text.replace('""', '"')))
    return out


def clean(text: str) -> str:
    return " ".join(_MARKERS.sub(" ", text).split())


def overlap_share(segments: list[tuple[str, float, float, str]]) -> float:
    """Share of speech time where two or more people talk at once."""
    events = sorted([(s, 1) for _, s, _e, _t in segments] + [(e, -1) for _, _s, e, _t in segments])
    speech = over = 0.0
    active, prev = 0, 0.0
    for t, delta in events:
        if active >= 1:
            speech += t - prev
        if active >= 2:
            over += t - prev
        active += delta
        prev = t
    return over / speech if speech else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", type=Path, help="Unpacked folder, e.g. data/alimeeting/Eval_Ali")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--channel", type=int, default=0, help="Which array channel to use")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N meetings")
    args = ap.parse_args()

    import soundfile as sf

    far = next(iter(sorted(args.src.glob("*_far"))), None) or args.src
    grids = sorted((far / "textgrid_dir").glob("*.TextGrid"))[: args.limit]
    if not grids:
        raise SystemExit(f"No TextGrid files in {far / 'textgrid_dir'}")
    (args.out / "audio").mkdir(parents=True, exist_ok=True)
    (args.out / "ref").mkdir(parents=True, exist_ok=True)

    total_minutes = 0.0
    for grid in grids:
        session = grid.stem
        wavs = sorted((far / "audio_dir").glob(f"{session}*.wav"))
        if not wavs:
            print(f"  {session}: no audio, skipped")
            continue
        audio, sr = sf.read(wavs[0], dtype="float32", always_2d=True)
        if sr != SR:
            raise SystemExit(f"{wavs[0]} is {sr} Hz; expected {SR}")
        sf.write(args.out / "audio" / f"{session}.wav", audio[:, args.channel], SR)

        segments = []
        for tier, start, end, text in read_textgrid(grid):
            words = clean(text)
            if words and end > start:
                speaker = tier.split("_")[-1] or tier  # e.g. R8001_M8004_N_SPK8001 -> SPK8001
                segments.append((speaker, start, end, words))
        segments.sort(key=lambda x: x[1])
        with open(args.out / "ref" / f"{session}.rttm", "w", encoding="utf-8") as f:
            for spk, s, e, _ in segments:
                f.write(f"SPEAKER {session} 1 {s:.3f} {e - s:.3f} <NA> <NA> {spk} <NA> <NA>\n")
        with open(args.out / "ref" / f"{session}.stm", "w", encoding="utf-8") as f:
            for spk, s, e, words in segments:
                f.write(f"{session} 1 {spk} {s:.3f} {e:.3f} {words}\n")

        minutes = len(audio) / SR / 60
        total_minutes += minutes
        n = len({s[0] for s in segments})
        print(
            f"  {session}: {minutes:.1f} min, {n} speakers, {len(segments)} utterances, "
            f"{overlap_share(segments):.0%} of speech overlapped"
        )
    print(f"Done: {len(grids)} meetings, {total_minutes / 60:.1f} hours in {args.out}")


if __name__ == "__main__":
    main()
