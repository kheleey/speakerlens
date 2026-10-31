"""Make voice samples for benchmark files, from clips that are NOT in the file.

For each file in a dataset built by make_ascend.py, every reference line is matched back to
its ASCEND clip (same text and length) to find who each speaker label really is (session +
speaker). The voice sample for that label is made from the same person's other clips, the
longest first, until it reaches --seconds. Clips used anywhere in the file are never used,
so the sample adds no test audio, just like a real person recording a sample beforehand.

Writes <dataset>/voices/<file id>/<label>.wav, for `run_bench.py --voices-root`.

Usage:
    python bench/make_voices.py --src data/ascend/main --split train \\
        data/dev/ascend_meetings data/dev/ascend_real data/heldout_real
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_ascend import SR, Utt, load_split  # noqa: E402


def norm(text: str) -> str:
    return " ".join(text.split())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("datasets", type=Path, nargs="+")
    ap.add_argument("--src", type=Path, default=Path("data/ascend/main"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--seconds", type=float, default=15.0, help="Length of each voice sample")
    args = ap.parse_args()

    import soundfile as sf

    print(f"Loading {args.split} split...", flush=True)
    utts = load_split(args.src, args.split)
    index: dict[str, list[int]] = defaultdict(list)
    by_person: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, u in enumerate(utts):
        index[norm(u.text)].append(i)
        by_person[(u.session, u.speaker)].append(i)

    for dataset in args.datasets:
        print(f"\n{dataset}")
        for stm in sorted((dataset / "ref").glob("*.stm")):
            fid = stm.stem
            used: set[int] = set()
            votes: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
            missing = 0
            for line in stm.read_text(encoding="utf-8").splitlines():
                p = line.split(maxsplit=5)
                if len(p) < 5 or line.startswith(";;"):
                    continue
                text = p[5] if len(p) == 6 else ""
                seconds = float(p[4]) - float(p[3])
                found = [
                    i
                    for i in index.get(norm(text), [])
                    if abs(len(utts[i].audio) / SR - seconds) < 0.01
                ]
                if not found:
                    missing += 1
                used.update(found)
                for i in found:
                    votes[p[2]][(utts[i].session, utts[i].speaker)] += 1

            notes = []
            for label, counter in sorted(votes.items()):
                person, _n = counter.most_common(1)[0]
                pool = [i for i in by_person[person] if i not in used]
                pool.sort(key=lambda i: -len(utts[i].audio))
                pieces: list[np.ndarray] = []
                total = 0.0
                gap = np.zeros(int(0.2 * SR), dtype=np.float32)
                for i in pool:
                    if total >= args.seconds:
                        break
                    clip: Utt = utts[i]
                    pieces += [clip.audio, gap]
                    total += len(clip.audio) / SR
                if total < 3:
                    notes.append(f"{label}: only {total:.1f} s available, skipped")
                    continue
                out = dataset / "voices" / fid / f"{label}.wav"
                out.parent.mkdir(parents=True, exist_ok=True)
                sf.write(out, np.concatenate(pieces), SR)
                notes.append(f"{label} {total:.0f}s")
            if missing:
                notes.append(f"({missing} reference lines not matched)")
            print(f"  {fid}: {', '.join(notes)}")


if __name__ == "__main__":
    main()
