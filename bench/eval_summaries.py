"""Check per-speaker summaries against the reference transcript.

For each file:
1. Summarize every speaker with the configured LLM (DeepSeek by default), from the
   reference transcript (true words, true speakers) and from the system run's transcript.
   Reference summaries are cached in --out and reused for every run compared.
2. A judge (the same LLM) checks each system point against the true transcript:
     supported    that person really said or clearly expressed it
     other        someone else said it: credited to the wrong person
     unsupported  nobody said it, or it misrepresents what was said
3. Coverage: how many of each person's reference points the system's summary also has.

System speakers are matched to real ones by shared speaking time. Points about a system
speaker who matches nobody count as "other".

A run is read from <run>/<set>/<id>.json (speakerlens) or, with --stm or when there is no
JSON, from <id>.stm, so other systems' transcripts get the same summarizer and judge.

Usage (needs DEEPSEEK_API_KEY in .env; a few LLM calls per file):
    python bench/eval_summaries.py data/ali_eval bench/results --limit 4
    python bench/eval_summaries.py data/ali_eval bench/vibevoice --stm --limit 4
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from speakerlens.config import load_settings
from speakerlens.llm import JSONChat, LLMClient
from speakerlens.summarize import (
    CATEGORIES,
    PROMPT_VERSION,
    summarize,
    transcript_lines,
)
from speakerlens.types import Speaker, Summary, Transcript, Turn

JUDGE_SYSTEM = """You check claims about what people said in a conversation, against its true
transcript. Each transcript line is: [turn id] speaker (time): text.
For EACH claim, decide:
- "supported": that speaker really said or clearly expressed it.
- "other": it was said or expressed by a different speaker, not the one named.
- "unsupported": nobody in the transcript said it, or it misrepresents what was said.
Be strict about WHO said it. When a claim says someone proposed, agreed with or disagreed
with another person, check the order in the transcript: who raised it first and who
responded. A reversed or invented agreement or disagreement is "unsupported".
Claims may be in a different language than the transcript.
Reply with JSON only: {"verdicts": [{"id": "C1", "verdict": "supported"}]}"""

COVER_SYSTEM = """You compare two summaries of the same people in a conversation.
For each REFERENCE point about a speaker, decide whether the CANDIDATE points about that
same speaker express the same idea. It counts as covered when a candidate says it in other
words or another language, more generally or more specifically, or as part of a longer
point. It is not covered when the idea is missing or a candidate says the opposite.
Reply with JSON only: {"verdicts": [{"id": "R1", "covered": true}]}"""


def reference_transcript(dataset: Path, fid: str) -> Transcript:
    return stm_transcript(dataset / "ref" / f"{fid}.stm", fid)


def stm_transcript(path: Path, fid: str) -> Transcript:
    turns = []
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.split(maxsplit=5)
        if len(p) == 6 and not line.startswith(";;"):
            turns.append((float(p[3]), float(p[4]), p[2], p[5]))
    turns.sort()
    return Transcript(
        source=fid,
        duration=max((e for _, e, _, _ in turns), default=0.0),
        languages=["zh", "en"],
        speakers=[Speaker(id=s) for s in sorted({t[2] for t in turns})],
        turns=[
            Turn(id=f"T{i:04d}", speaker=spk, start=s, end=e, text=text)
            for i, (s, e, spk, text) in enumerate(turns, 1)
        ],
    )


def match_speakers(system: Transcript, ref: Transcript) -> dict[str, str]:
    """System speaker id -> reference speaker id, by most shared speaking time."""
    from scipy.optimize import linear_sum_assignment

    sys_ids = sorted({t.speaker for t in system.turns})
    ref_ids = sorted({t.speaker for t in ref.turns})
    if not sys_ids or not ref_ids:
        return {}
    shared = np.zeros((len(sys_ids), len(ref_ids)))
    for a in system.turns:
        for b in ref.turns:
            t = min(a.end, b.end) - max(a.start, b.start)
            if t > 0:
                shared[sys_ids.index(a.speaker), ref_ids.index(b.speaker)] += t
    rows, cols = linear_sum_assignment(-shared)
    return {sys_ids[r]: ref_ids[c] for r, c in zip(rows, cols, strict=True) if shared[r, c] > 0}


def points(summary: Summary) -> list[tuple[str, str]]:
    """(speaker id, point text) for every point in every category."""
    out = []
    for spk in summary.speakers:
        for cat in CATEGORIES:
            out += [(spk.speaker, p.text) for p in getattr(spk, cat)]
    return out


def judge_claims(
    llm: JSONChat, ref: Transcript, claims: list[tuple[str, str]], max_chars: int = 120_000
) -> list[str]:
    """Verdict per claim: supported / other / unsupported, or "unjudged" if the judge
    skipped it."""
    if not claims:
        return []
    transcript = "\n".join(transcript_lines(ref))[:max_chars]
    listing = "\n".join(f"C{i} [about {spk}]: {text}" for i, (spk, text) in enumerate(claims, 1))
    data = llm.chat_json(
        JUDGE_SYSTEM, f"Transcript:\n{transcript}\n\nClaims:\n{listing}", temperature=0.0
    )
    verdicts = {str(v.get("id")): str(v.get("verdict", "")).lower() for v in _verdicts(data)}
    allowed = {"supported", "other", "unsupported"}
    out = []
    for i in range(1, len(claims) + 1):
        v = verdicts.get(f"C{i}", "")
        out.append(v if v in allowed else "unjudged")
    return out


def judge_coverage(
    llm: JSONChat, ref_points: list[tuple[str, str]], candidates: list[tuple[str, str]]
) -> list[bool | None]:
    """For each reference point (in the given order): covered by the candidate points about
    the same speaker? None if the judge skipped it. One call per speaker, at temperature 0."""
    out: list[bool | None] = [None] * len(ref_points)
    for spk in sorted({s for s, _ in ref_points}):
        mine = [(i, t) for i, (s, t) in enumerate(ref_points) if s == spk]
        theirs = [t for s, t in candidates if s == spk]
        lines = [f"Speaker {spk}", "Reference points:"]
        lines += [f"R{n}: {t}" for n, (_i, t) in enumerate(mine, 1)]
        lines.append("Candidate points:")
        lines += [f"- {t}" for t in theirs] or ["(none)"]
        if not theirs:
            for i, _t in mine:
                out[i] = False
            continue
        data = llm.chat_json(COVER_SYSTEM, "\n".join(lines), temperature=0.0)
        covered = {str(v.get("id")): v.get("covered") for v in _verdicts(data)}
        for n, (i, _t) in enumerate(mine, 1):
            v = covered.get(f"R{n}")
            out[i] = v if isinstance(v, bool) else None
    return out


def _verdicts(data: dict[str, Any]) -> list[dict[str, Any]]:
    v = data.get("verdicts", [])
    return [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []


def rename(text: str, renames: list[tuple[str, str]]) -> str:
    """Replace speaker names, longest first (so SPEAKER_10 isn't hit by SPEAKER_1)."""
    for old, new in sorted(renames, key=lambda kv: -len(kv[0])):
        text = text.replace(old, new)
    return text


def cached_summary(path: Path, transcript: Transcript, llm: JSONChat, language: str) -> Summary:
    """The saved summary at `path`, or a new one, saved there. Saved summaries made with
    other summary instructions are refused, so runs are never compared across versions. A
    summary is saved only when every part of it was summarized."""
    if path.exists():
        summary = Summary.model_validate_json(path.read_text())
        if summary.prompt_version != PROMPT_VERSION:
            raise SystemExit(
                f"{path} was made with summary instructions v{summary.prompt_version}, not "
                f"v{PROMPT_VERSION}. Use another --out folder."
            )
        return summary
    summary = summarize(transcript, llm, language=language, strict=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary.model_dump_json(indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dataset", type=Path)
    ap.add_argument("run", type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--language", default="en", help="Language the summaries are written in")
    ap.add_argument("--out", type=Path, default=Path("bench/summary_eval"))
    ap.add_argument(
        "--stm", action="store_true", help="Read the run's <id>.stm files instead of <id>.json"
    )
    ap.add_argument(
        "--ceiling",
        action="store_true",
        help="Also summarize the reference a second time and report how much one reference "
        "summary covers of the other: the best coverage to expect",
    )
    args = ap.parse_args()

    settings = load_settings()
    llm = LLMClient.from_settings(settings.llm)
    if llm is None:
        raise SystemExit("No LLM configured: set DEEPSEEK_API_KEY in .env")

    name = args.dataset.name
    ids = sorted(p.stem for p in (args.dataset / "audio").glob("*.wav"))
    run_dir = args.run / name
    suffix = ".stm" if args.stm else ".json"
    ids = [
        i for i in ids if (run_dir / f"{i}{suffix}").exists() or (run_dir / f"{i}.stm").exists()
    ][: args.limit]
    verdict_totals: Counter[str] = Counter()
    ceiling_scores: list[float] = []
    covered_total = ref_total = 0
    report = {}

    print(
        f"{'file':24s} {'points':>6s} {'right':>6s} {'wrong':>6s} {'made':>6s} "
        f"{'ref pts':>7s} {'covered':>8s}" + (f" {'ceiling':>8s}" if args.ceiling else "")
    )
    for fid in ids:
        ref = reference_transcript(args.dataset, fid)
        json_path = run_dir / f"{fid}.json"
        if args.stm or not json_path.exists():
            system = stm_transcript(run_dir / f"{fid}.stm", fid)
        else:
            system = Transcript.model_validate_json(json_path.read_text())
        known = {x.id for x in system.speakers}
        system.speakers += [
            Speaker(id=x) for x in sorted({t.speaker for t in system.turns} - known)
        ]
        ref_sum = cached_summary(
            args.out / "reference" / name / f"{fid}.json", ref, llm, args.language
        )
        sys_sum = cached_summary(
            args.out / args.run.name / name / f"{fid}.json", system, llm, args.language
        )
        mapping = match_speakers(system, ref)
        # Points may name people by the system's labels; the judge sees the matched names.
        renames = [(system.speaker_label(k), v) for k, v in mapping.items()]
        claims = [
            (mapping.get(spk, "nobody"), rename(text, renames)) for spk, text in points(sys_sum)
        ]
        verdicts = judge_claims(llm, ref, claims)
        verdicts = [
            "other" if s == "nobody" else v for (s, _), v in zip(claims, verdicts, strict=True)
        ]
        cover = judge_coverage(llm, points(ref_sum), claims)
        for label, summ in (("reference", ref_sum), ("system", sys_sum)):
            if summ.rejected_points:
                reasons = Counter(r.reason.split(":")[0] for r in summ.rejected_points)
                example = summ.rejected_points[0]
                print(
                    f"  note: the {label} summary of {fid} lost {len(summ.rejected_points)} "
                    f"points ({', '.join(f'{k}: {v}' for k, v in reasons.most_common())}); "
                    f"e.g. cited {example.turn_ids[:4]} -> {example.reason[:60]}"
                )
        ceiling = None
        if args.ceiling:
            ref_again = cached_summary(
                args.out / "reference_again" / name / f"{fid}.json",
                ref,
                llm,
                args.language,
            )
            again = judge_coverage(llm, points(ref_sum), points(ref_again))
            ceiling = sum(1 for c in again if c) / max(len(again), 1)
            ceiling_scores.append(ceiling)

        counts = Counter(verdicts)
        verdict_totals.update(counts)
        n_cov = sum(1 for c in cover if c)
        covered_total += n_cov
        ref_total += len(cover)
        n = max(len(verdicts), 1)
        print(
            f"{fid:24s} {len(verdicts):6d} {counts['supported'] / n:6.0%} "
            f"{counts['other'] / n:6.0%} {counts['unsupported'] / n:6.0%} "
            f"{len(cover):7d} {n_cov / max(len(cover), 1):8.0%}"
            + (f" {ceiling:8.0%}" if ceiling is not None else "")
        )
        report[fid] = {
            "speaker_map": mapping,
            "claims": [
                {"speaker": s, "text": t, "verdict": v}
                for (s, t), v in zip(claims, verdicts, strict=True)
            ],
            "reference_points": [
                {"speaker": s, "text": t, "covered": c}
                for (s, t), c in zip(points(ref_sum), cover, strict=True)
            ],
        }

    judged = sum(verdict_totals[k] for k in ("supported", "other", "unsupported")) or 1
    print(
        f"\nPoints about the right person, and true:  {verdict_totals['supported'] / judged:.1%}"
        f"\nPoints credited to the wrong person:      {verdict_totals['other'] / judged:.1%}"
        f"\nPoints nobody said (or misrepresented):   {verdict_totals['unsupported'] / judged:.1%}"
        f"\nReference points also in these summaries: {covered_total / max(ref_total, 1):.1%}"
        + (
            f"\n  ... and in a second reference summary:   "
            f"{sum(ceiling_scores) / len(ceiling_scores):.1%} (the ceiling)"
            if ceiling_scores
            else ""
        )
        + (
            f"\n({verdict_totals['unjudged']} points the judge skipped)"
            if verdict_totals["unjudged"]
            else ""
        )
    )
    out = args.out / args.run.name / name / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Details: {out}")


if __name__ == "__main__":
    main()
