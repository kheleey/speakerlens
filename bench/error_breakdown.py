"""Split a run's errors into speech recognition and who-spoke-when.

1. Oracle run (GPU): the pipeline's speech recognition on the reference speaker segments,
   with no diarization or speaker check. Its tcpWER is the recognition error alone; the
   difference to a normal run is the cost of who-spoke-when. Oracle turns are also grouped
   by how much someone else talks at the same time (clean / some / heavy overlap).
   Extraction is used for overlapped turns as in the pipeline; --no-extract turns it off.

2. Breakdown of any finished run (CPU): substitutions, deletions and insertions, and how
   much reference speech got no output from anyone.

3. Code-switching split of any finished run: the share of reference words recognized (by
   any speaker), for Mandarin and English words in mostly-Mandarin and mostly-English lines.

Parts 2 and 3 need only the run's <id>.stm files, so other systems are scored the same way.

    python bench/error_breakdown.py data/ali_eval --run bench/results
    CUDA_VISIBLE_DEVICES=2 python bench/error_breakdown.py data/ali_eval --oracle
    # several GPUs: --oracle --shard 1/4 ... 4/4 in parallel, then --oracle --score-only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

from speakerlens.diarization.base import DiarizationResult, DiarizedSegment, Diarizer
from speakerlens.export import to_stm
from speakerlens.metrics import edit_distance, tokenize
from speakerlens.types import Transcript

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_bench import normalize_stm  # noqa: E402

BUCKETS = (("clean", 0.0, 0.1), ("some overlap", 0.1, 0.5), ("heavy overlap", 0.5, 1.01))


def read_stm(path: Path) -> list[tuple[str, float, float, str]]:
    """(speaker, start, end, text) per reference line."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.split(maxsplit=5)
        if len(p) >= 5 and not line.startswith(";;"):
            out.append((p[2], float(p[3]), float(p[4]), p[5] if len(p) == 6 else ""))
    return sorted(out, key=lambda x: x[1])


def ref_overlaps(segments: list[tuple[str, float, float, str]]) -> list[tuple[float, float]]:
    """Stretches where two different reference speakers talk at once."""
    out = []
    for i, (s1, a1, b1, _) in enumerate(segments):
        for s2, a2, b2, _ in segments[i + 1 :]:
            if s1 != s2 and min(b1, b2) > max(a1, a2):
                out.append((max(a1, a2), min(b1, b2)))
    return out


def merged_segments(
    segments: list[tuple[str, float, float, str]], gap: float = 0.5, longest: float = 30.0
) -> list[DiarizedSegment]:
    """Each reference speaker's lines merged into stretches (gaps up to `gap` s, at most
    `longest` s), so overlapping speakers each keep whole turns."""
    out = []
    for spk in sorted({s for s, *_ in segments}):
        cur: list | None = None
        for _, a, b, _t in sorted((x for x in segments if x[0] == spk), key=lambda x: x[1]):
            if cur and a - cur[1] <= gap and b - cur[0] <= longest:
                cur[1] = max(cur[1], b)
            else:
                if cur:
                    out.append(DiarizedSegment(speaker=spk, start=cur[0], end=cur[1]))
                cur = [a, b]
        if cur:
            out.append(DiarizedSegment(speaker=spk, start=cur[0], end=cur[1]))
    return sorted(out, key=lambda x: x.start)


class OracleDiarizer(Diarizer):
    """Returns the reference segments: perfect who-spoke-when, overlaps included."""

    model = "oracle"

    def __init__(self, segments: list[tuple[str, float, float, str]]) -> None:
        self.segments = segments

    def result(self) -> DiarizationResult:
        return DiarizationResult(
            segments=merged_segments(self.segments), overlaps=ref_overlaps(self.segments)
        )

    def diarize(self, audio, sample_rate, **kwargs):  # type: ignore[no-untyped-def]
        return self.result()


def overlap_share(spk: str, start: float, end: float, ref: list) -> float:
    """Share of [start, end] during which a different reference speaker is talking."""
    spans = sorted(
        (max(a, start), min(b, end)) for s, a, b, _ in ref if s != spk and b > start and a < end
    )
    covered, cur_a, cur_b = 0.0, None, None
    for a, b in spans:
        if cur_b is None or a > cur_b:
            covered += (cur_b - cur_a) if cur_b is not None else 0.0
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    covered += (cur_b - cur_a) if cur_b is not None else 0.0
    return covered / max(end - start, 1e-6)


def turn_errors(transcript: Transcript, ref: list) -> list[dict]:
    """For oracle turns: reference text of the segments each turn was built from, the
    token errors, and the turn's overlap share. Reference lines no turn covers (dropped as
    too short) count as fully missed."""
    used: set[int] = set()
    rows = []
    for t in transcript.turns:
        mine = [
            i
            for i, (s, a, b, _) in enumerate(ref)
            if s == t.speaker and t.start - 0.01 <= (a + b) / 2 <= t.end + 0.01 and i not in used
        ]
        used.update(mine)
        r = [tok for i in mine for tok in tokenize(ref[i][3])]
        h = tokenize(t.text)
        rows.append(
            {
                "n": len(r),
                "errors": edit_distance(r, h),
                "share": overlap_share(t.speaker, t.start, t.end, ref),
            }
        )
    for i, (s, a, b, text) in enumerate(ref):
        if i not in used and (n := len(tokenize(text))):
            rows.append({"n": n, "errors": n, "share": overlap_share(s, a, b, ref)})
    return rows


def missed_tokens(ref: list, hyp: list) -> tuple[int, int]:
    """(missed, total) reference tokens; missed = in lines where the run output nothing,
    from anyone. `ref` and `hyp` are STM lines (speaker, start, end, text)."""
    total = missed = 0
    for _s, a, b, text in ref:
        n = len(tokenize(text))
        total += n
        heard = sum(max(0.0, min(b, e) - max(a, s)) for _x, s, e, t in hyp if t.strip())
        if heard < 0.2 * (b - a):
            missed += n
    return missed, total


_HAN = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
# An English word carries about as much speech as 1.5 Chinese characters (syllables); used
# only to decide which language a line is mostly in.
EN_WEIGHT = 1.5
CS_GROUPS = (
    ("zh", "zh", "Mandarin words, mostly-Mandarin lines"),
    ("zh", "en", "English inside Mandarin"),
    ("en", "en", "English words, mostly-English lines"),
    ("en", "zh", "Mandarin inside English"),
)


def word_language(token: str) -> str:
    return "zh" if _HAN.fullmatch(token) else "en"


def codeswitch_breakdown(ref: list, hyp: list) -> dict[tuple[str, str], list[int]]:
    """(main language of the line, language of the word) -> [recognized, total] reference
    words. Reference and system words are each put in time order (by line start) and
    aligned as two sequences, ignoring who said them; a reference word counts as recognized
    when the alignment matches it to the same word in the output. Extra words in the output
    aren't counted here (they are in tcORC-WER)."""
    ref_words: list[tuple[str, tuple[str, str]]] = []
    for _s, _a, _b, text in sorted(ref, key=lambda x: x[1]):
        tokens = tokenize(text)
        if not tokens:
            continue
        zh = sum(1 for t in tokens if word_language(t) == "zh")
        main = "zh" if zh >= EN_WEIGHT * (len(tokens) - zh) else "en"
        ref_words += [(t, (main, word_language(t))) for t in tokens]
    hyp_tokens = [t for _s, _a, _b, text in sorted(hyp, key=lambda x: x[1]) for t in tokenize(text)]
    matched = [False] * len(ref_words)
    blocks = SequenceMatcher(None, [w for w, _ in ref_words], hyp_tokens, autojunk=False)
    for block in blocks.get_matching_blocks():
        for i in range(block.a, block.a + block.size):
            matched[i] = True
    out: dict[tuple[str, str], list[int]] = {}
    for (_w, group), ok in zip(ref_words, matched, strict=True):
        cell = out.setdefault(group, [0, 0])
        cell[0] += ok
        cell[1] += 1
    return out


def print_codeswitch(results: dict[tuple[str, str], list[int]]) -> None:
    total = max(sum(n for _, n in results.values()), 1)
    if not any(lang == "en" for (_m, lang) in results):
        return  # no English at all: nothing to split
    print("\n  Recognized words by language (any speaker; extra words not counted):")
    print(f"  {'':40s} {'share of words':>14s} {'recognized':>11s}")
    for main, lang, name in CS_GROUPS:
        ok, n = results.get((main, lang), [0, 0])
        if n:
            print(f"  {name:40s} {n / total:14.1%} {ok / n:11.1%}  ({n} words)")


def counts(ref_dir: Path, run_dir: Path, ids: list[str], work: Path) -> dict:
    """tcpWER and tcORC-WER with their substitution / deletion / insertion parts."""
    import meeteval

    ref_path, hyp_path = work / "ref.norm.stm", work / "hyp.norm.stm"
    ref_path.write_text(
        "".join(normalize_stm((ref_dir / f"{i}.stm").read_text()) for i in ids), encoding="utf-8"
    )
    hyp_path.write_text(
        "".join(normalize_stm((run_dir / f"{i}.stm").read_text()) for i in ids), encoding="utf-8"
    )
    out = {}
    for metric in ("tcpwer", "tcorcwer"):
        er = meeteval.wer.combine_error_rates(
            getattr(meeteval.wer, metric)(
                reference=str(ref_path), hypothesis=str(hyp_path), collar=5
            )
        )
        n = max(er.length, 1)
        out[metric] = {
            "rate": er.error_rate,
            "substitutions": er.substitutions / n,
            "deletions": er.deletions / n,
            "insertions": er.insertions / n,
        }
    return out


def print_counts(label: str, c: dict, missed: float) -> None:
    print(f"\n{label}")
    print(f"  {'':28s} {'total':>7s} {'misheard':>9s} {'missed':>7s} {'extra':>6s}")
    for metric, name in (
        ("tcpwer", "tcpWER (right person)"),
        ("tcorcwer", "tcORC-WER (any person)"),
    ):
        m = c[metric]
        print(
            f"  {name:28s} {m['rate']:7.3f} {m['substitutions']:9.3f} "
            f"{m['deletions']:7.3f} {m['insertions']:6.3f}"
        )
    print(f"  Reference speech with no output from anyone: {missed:.1%} of words")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--run", type=Path, help="A finished run_bench.py output folder to break down")
    ap.add_argument("--oracle", action="store_true", help="Recognize the true segments (GPU)")
    ap.add_argument("--out", type=Path, default=Path("bench/oracle"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default=None, help="e.g. 1/4, with --oracle")
    ap.add_argument("--score-only", action="store_true", help="With --oracle: score saved runs")
    ap.add_argument("--lang", action="append", default=None)
    ap.add_argument(
        "--extract",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="With --oracle: speaker extraction for overlapped turns (default: as configured)",
    )
    ap.add_argument("--tse-model", type=Path, default=None, help="Overrides TSE_MODEL")
    args = ap.parse_args()
    if not args.run and not args.oracle:
        ap.error("give --run DIR, --oracle, or both")

    ref_dir = args.dataset / "ref"
    ids = sorted(p.stem for p in (args.dataset / "audio").glob("*.wav"))[: args.limit]
    refs = {i: read_stm(ref_dir / f"{i}.stm") for i in ids}

    if args.run:
        run_dir = args.run / args.dataset.name
        if not run_dir.is_dir():
            run_dir = args.run
        done = [i for i in ids if (run_dir / f"{i}.stm").exists()]
        hyps = {i: read_stm(run_dir / f"{i}.stm") for i in done}
        pairs = [missed_tokens(refs[i], hyps[i]) for i in done]
        missed = sum(m for m, _ in pairs) / max(sum(t for _, t in pairs), 1)
        print_counts(
            f"{run_dir} ({len(done)} files)", counts(ref_dir, run_dir, done, run_dir), missed
        )
        split: dict[tuple[str, str], list[int]] = {}
        for i in done:
            for group, (ok, n) in codeswitch_breakdown(refs[i], hyps[i]).items():
                cell = split.setdefault(group, [0, 0])
                cell[0] += ok
                cell[1] += n
        print_codeswitch(split)

    if args.oracle:
        out = args.out / args.dataset.name
        out.mkdir(parents=True, exist_ok=True)
        todo = ids
        if args.shard:
            k, n = (int(x) for x in args.shard.split("/"))
            todo = ids[k - 1 :: n]
        if not args.score_only:
            from speakerlens.config import load_settings
            from speakerlens.pipeline import Pipeline, PipelineOptions

            if args.extract is not None:
                os.environ["SPEAKER_EXTRACTION"] = "on" if args.extract else "off"
            if args.tse_model is not None:
                os.environ["TSE_MODEL"] = str(args.tse_model)
            settings = load_settings()
            options = PipelineOptions(
                languages=args.lang or ["zh", "en"],
                summarize=False,
                reassign_margin=None,  # recognition only: keep the true speakers
                use_cache=False,
            )
            pipeline = Pipeline.from_settings(settings, options)
            for j, fid in enumerate(todo, 1):
                print(f"[{j}/{len(todo)}] {fid} (oracle segments)")
                pipeline.diarizer = OracleDiarizer(refs[fid])
                path = args.dataset / "audio" / f"{fid}.wav"
                transcript = pipeline.run(path, options)
                (out / f"{fid}.stm").write_text(to_stm(transcript, fid), encoding="utf-8")
                (out / f"{fid}.json").write_text(transcript.model_dump_json(indent=2))
            if args.shard:
                print("Shard done. When all shards finish, run again with --oracle --score-only.")
                return

        done = [i for i in ids if (out / f"{i}.json").exists()]
        if not done:
            raise SystemExit(f"No finished meetings in {out}: check the shard logs for errors.")
        transcripts = {
            i: Transcript.model_validate_json((out / f"{i}.json").read_text()) for i in done
        }
        print_counts(
            f"ORACLE: recognition on the reference segments ({len(done)} files)",
            counts(ref_dir, out, done, out),
            0.0,
        )
        rows = [r for i in done for r in turn_errors(transcripts[i], refs[i])]
        all_err = max(sum(r["errors"] for r in rows), 1)
        all_n = max(sum(r["n"] for r in rows), 1)
        print(f"\n  Recognition by overlap (oracle turns; overall {all_err / all_n:.3f}):")
        print(f"  {'':16s} {'share of speech':>15s} {'error rate':>11s} {'share of errors':>16s}")
        summary = {}
        for name, lo, hi in BUCKETS:
            b = [r for r in rows if lo <= r["share"] < hi]
            n, e = sum(r["n"] for r in b), sum(r["errors"] for r in b)
            summary[name] = {"tokens": n, "errors": e}
            print(f"  {name:16s} {n / all_n:15.0%} {e / max(n, 1):11.3f} {e / all_err:16.0%}")
        (out / "breakdown.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
