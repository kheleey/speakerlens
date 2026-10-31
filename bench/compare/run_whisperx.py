"""Run WhisperX (Whisper large-v3 + pyannote community-1 diarization) on a benchmark set,
in its standard setup, writing output that bench/run_bench.py --score-only can score.

Runs in its own environment (WhisperX pins torch 2.8): see bench/compare/README.md.

    CUDA_VISIBLE_DEVICES=2 python bench/compare/run_whisperx.py data/ali_eval \\
        --out bench/whisperx --word-out bench/whisperx_words --speakers-from-ref
    python bench/run_bench.py data/ali_eval --out bench/whisperx --score-only   # (speakerlens env)

Default settings: language detected once per file (or --language), words aligned for
timing. Two outputs from the same run:
  --out       each segment under one speaker, as in WhisperX's transcript and subtitles;
  --word-out  segments split wherever the per-word speaker labels change.
The full result is saved as <id>.whisperx.json next to the --out files. Traditional
characters are converted to Simplified before scoring.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    Timer,
    audio_files,
    speakers_in_ref,
    to_simplified,
    turns_by_word_speaker,
    write_run,
)


def segment_turns(segments: list[dict]) -> list[tuple[str, float, float, str]]:
    return [
        (
            str(s.get("speaker", "UNKNOWN")),
            float(s["start"]),
            float(s["end"]),
            str(s.get("text", "")),
        )
        for s in segments
        if s.get("start") is not None and s.get("end") is not None
    ]


def simplified(turns: list[tuple[str, float, float, str]]) -> list[tuple[str, float, float, str]]:
    return [(s, a, b, to_simplified(t)) for s, a, b, t in turns]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--out", type=Path, default=Path("bench/whisperx"))
    ap.add_argument(
        "--word-out", type=Path, default=None, help="Also write per-word-speaker turns here"
    )
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--language", default=None, help="Default: detected once per file")
    ap.add_argument("--speakers", type=int, default=None)
    ap.add_argument("--speakers-from-ref", action="store_true")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import whisperx
    from whisperx.diarize import DiarizationPipeline

    device = "cuda"
    token = os.environ.get("HF_TOKEN")
    asr = whisperx.load_model(args.model, device, compute_type="float16", language=args.language)
    diarizer = DiarizationPipeline(token=token, device=device)
    aligners: dict[str, tuple] = {}
    out = args.out / args.dataset.name
    word_out = args.word_out / args.dataset.name if args.word_out else None

    for path in audio_files(args.dataset, args.shard, args.limit):
        fid, timer = path.stem, Timer()
        audio = whisperx.load_audio(str(path))
        result = asr.transcribe(audio, batch_size=args.batch_size, language=args.language)
        lang = result.get("language") or args.language or "zh"
        aligned = True
        try:
            if lang not in aligners:
                aligners[lang] = whisperx.load_align_model(language_code=lang, device=device)
            model_a, meta = aligners[lang]
            result = whisperx.align(result["segments"], model_a, meta, audio, device)
        except Exception as exc:  # no aligner for this language: segment-level speakers
            aligned = False
            print(f"  {fid}: no word alignment ({exc}); speakers by segment")
        n = speakers_in_ref(args.dataset, fid) if args.speakers_from_ref else args.speakers
        segments = diarizer(audio, num_speakers=n)
        result = whisperx.assign_word_speakers(segments, result)
        turns = segment_turns(result["segments"])
        write_run(out, fid, simplified(turns))
        (out / f"{fid}.whisperx.json").write_text(
            json.dumps(
                {"language": lang, "aligned": aligned, **result}, ensure_ascii=False, default=float
            ),
            encoding="utf-8",
        )
        note = ""
        if word_out is not None:
            word_turns = turns_by_word_speaker(result["segments"])
            write_run(word_out, fid, simplified(word_turns))
            note = f", {len(word_turns)} by word speaker"
        print(f"{fid}: language {lang}, {len(turns)} segments{note}, {timer}", flush=True)


if __name__ == "__main__":
    main()
