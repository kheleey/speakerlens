"""Benchmark the pipeline: DER (who spoke when) and cpWER (right words, right person).

Expected layout (see make_ascend.py and make_alimeeting.py):

    data/ali_eval/
      audio/R8001_M8004.wav
      ref/R8001_M8004.rttm    # reference speaker turns
      ref/R8001_M8004.stm     # reference speaker-attributed transcript (optional)

Usage:

    pip install -e ".[all,bench]"
    python bench/run_bench.py data/ali_eval --speakers-from-ref --lang zh

Writes hypotheses and results.json into bench/results/<dataset name>/. With several GPUs,
run one copy per GPU with --shard 1/4 ... 4/4, then once more with --score-only.
Text is normalized before scoring (lowercase, no punctuation, one token per Chinese
character), so "cpWER" is cpCER on Chinese and a mixed error rate on mixed speech.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from speakerlens.config import load_settings
from speakerlens.export import to_rttm, to_stm
from speakerlens.metrics import tokenize
from speakerlens.pipeline import Pipeline, PipelineOptions

AUDIO_EXT = {".wav", ".flac", ".mp3", ".m4a"}


def normalize_stm(text: str) -> str:
    """Rewrite STM lines so MeetEval scores tokens the same way for every language."""
    out = []
    for line in text.splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) < 5 or line.startswith(";;"):
            continue
        words = " ".join(tokenize(parts[5])) if len(parts) == 6 else ""
        out.append(" ".join(parts[:5] + [words]))
    return "\n".join(out) + "\n"


def compute_der(
    ref_dir: Path, hyp_dir: Path, file_ids: list[str], skip_overlap: bool = False
) -> dict:
    """DER with no collar. skip_overlap=True leaves out the stretches where the reference
    has two or more people talking at once, so systems that write one speaker at a time
    aren't charged for everyone they can't hear underneath."""
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate

    def load(path: Path) -> Annotation:
        ann = Annotation(uri=path.stem)
        for line in path.read_text(encoding="utf-8").splitlines():
            f = line.split()
            if len(f) >= 8 and f[0] == "SPEAKER":
                start, dur = float(f[3]), float(f[4])
                ann[Segment(start, start + dur)] = f[7]
        return ann

    metric = DiarizationErrorRate(collar=0.0, skip_overlap=skip_overlap)
    per_file = {}
    for fid in file_ids:
        per_file[fid] = float(metric(load(ref_dir / f"{fid}.rttm"), load(hyp_dir / f"{fid}.rttm")))
    return {"overall": float(abs(metric)), "per_file": per_file}


def compute_meeteval(
    ref_dir: Path, hyp_dir: Path, file_ids: list[str], work: Path, metric: str = "cpwer"
) -> dict:
    """Speaker-attributed error rates from MeetEval, on normalized text.

    cpwer     right words for the right person (standard)
    tcpwer    same, and words must be within 5 s of the right time (CHiME-8 standard)
    tcorcwer  right words at the right time, ignoring who said them; compare with
              tcpwer to see how much error comes from speaker mix-ups
    (The time limit is what keeps tcorcwer fast; plain ORC-WER can run for hours.)
    """
    import meeteval

    ref_all = "".join(normalize_stm((ref_dir / f"{f}.stm").read_text()) for f in file_ids)
    hyp_all = "".join(normalize_stm((hyp_dir / f"{f}.stm").read_text()) for f in file_ids)
    ref_path, hyp_path = work / "ref.norm.stm", work / "hyp.norm.stm"
    ref_path.write_text(ref_all, encoding="utf-8")
    hyp_path.write_text(hyp_all, encoding="utf-8")
    kwargs = {"collar": 5} if metric.startswith("tc") else {}
    fn = getattr(meeteval.wer, metric)
    results = fn(reference=str(ref_path), hypothesis=str(hyp_path), **kwargs)
    overall = meeteval.wer.combine_error_rates(results)
    return {
        "overall": float(overall.error_rate),
        "per_file": {k: float(v.error_rate) for k, v in results.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset", type=Path, help="Folder with audio/ and ref/ subfolders")
    parser.add_argument("--lang", action="append", default=None, help="Spoken languages")
    parser.add_argument("--out", type=Path, default=Path("bench/results"))
    parser.add_argument("--limit", type=int, default=None, help="Only the first N files")
    parser.add_argument("--speakers", type=int, default=None, help="Exact speakers per file")
    parser.add_argument(
        "--speakers-from-ref",
        action="store_true",
        help="Give each file its true number of speakers (from the reference), for datasets "
        "where it varies, like AliMeeting",
    )
    parser.add_argument(
        "--asr-profile",
        choices=["auto", "routing", "nano"],
        default=None,
        help="Which speech models to use (default: ASR_PROFILE in .env, else auto)",
    )
    parser.add_argument(
        "--reassign-margin",
        type=float,
        default=None,
        help="Speaker check margin (default: SPEAKER_REASSIGN_MARGIN in .env, else 0.1; 0 = off)",
    )
    parser.add_argument(
        "--voices-root",
        type=Path,
        default=None,
        help="Folder with one voice-sample folder per file (<root>/<file id>/<name>.wav), "
        "e.g. data/dev/ascend_meetings/voices from make_voices.py",
    )
    parser.add_argument(
        "--extract",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force speaker extraction for overlapped speech on or off (default: "
        "SPEAKER_EXTRACTION, else on)",
    )
    parser.add_argument("--tse-model", type=Path, default=None, help="Overrides TSE_MODEL")
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument(
        "--score-only", action="store_true", help="Re-score saved results without re-running"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip files this --out folder already has results for (e.g. after a crash)",
    )
    parser.add_argument(
        "--shard",
        default=None,
        help="Run only part of the files, e.g. 1/4 (run 4 copies on 4 GPUs, then --score-only)",
    )
    args = parser.parse_args()

    if args.asr_profile:
        os.environ["ASR_PROFILE"] = args.asr_profile
    if args.extract is not None:
        os.environ["SPEAKER_EXTRACTION"] = "on" if args.extract else "off"
    if args.tse_model is not None:
        os.environ["TSE_MODEL"] = str(args.tse_model)
    audio_dir, ref_dir = args.dataset / "audio", args.dataset / "ref"
    out = args.out / args.dataset.name
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in audio_dir.iterdir() if p.suffix.lower() in AUDIO_EXT)[: args.limit]
    all_files = files
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        files = files[k - 1 :: n]
        print(f"Shard {k}/{n}: {len(files)} of {len(all_files)} files")
    settings = load_settings()
    margin = settings.reassign_margin if args.reassign_margin is None else args.reassign_margin
    options = PipelineOptions(
        languages=args.lang or ["zh", "en"],
        summarize=False,
        num_speakers=args.speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        reassign_margin=margin or None,
        # Any non-empty value makes from_settings load the voice embedder; set per file below.
        voices_dir=args.voices_root,
    )
    pipeline = None if args.score_only else Pipeline.from_settings(settings, options)

    for i, path in enumerate([] if pipeline is None else files, 1):
        if args.resume and (out / f"{path.stem}.json").exists():
            print(f"[{i}/{len(files)}] {path.name}: already done")
            continue
        print(f"[{i}/{len(files)}] {path.name}")
        if args.speakers_from_ref:
            rttm = ref_dir / f"{path.stem}.rttm"
            options.num_speakers = len(
                {ln.split()[7] for ln in rttm.read_text().splitlines() if ln.startswith("SPEAKER")}
            )
        if args.voices_root:
            voices = args.voices_root / path.stem
            options.voices_dir = voices if voices.is_dir() else None
        transcript = pipeline.run(path, options)
        (out / f"{path.stem}.rttm").write_text(to_rttm(transcript, path.stem), encoding="utf-8")
        (out / f"{path.stem}.stm").write_text(to_stm(transcript, path.stem), encoding="utf-8")
        (out / f"{path.stem}.json").write_text(transcript.model_dump_json(indent=2))

    if args.shard:
        print("Shard done. When every shard has finished, run again with --score-only.")
        return
    ids = [p.stem for p in all_files]
    results: dict = {"dataset": args.dataset.name, "files": len(ids)}
    if all((ref_dir / f"{i}.rttm").exists() for i in ids):
        results["der"] = compute_der(ref_dir, out, ids)
        results["der_noov"] = compute_der(ref_dir, out, ids, skip_overlap=True)
    if all((ref_dir / f"{i}.stm").exists() for i in ids):
        for metric in ("cpwer", "tcpwer", "tcorcwer"):
            results[metric] = compute_meeteval(ref_dir, out, ids, out, metric)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    for key in ("der", "der_noov", "cpwer", "tcpwer", "tcorcwer"):
        if key in results:
            print(f"{key:9s} overall {results[key]['overall']:.3f}")
            for fid, v in results[key]["per_file"].items():
                print(f"          {fid:24s} {v:.3f}")


if __name__ == "__main__":
    main()
