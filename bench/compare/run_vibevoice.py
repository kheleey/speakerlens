"""Run Microsoft VibeVoice-ASR (one model for transcription and who-spoke-when, MIT
license) on a benchmark set, writing output bench/run_bench.py --score-only can score.

Runs in its own environment (needs transformers >= 5.3): see bench/compare/README.md.
About 18 GB of weights in bf16: give it two 24 GB GPUs, or one larger GPU.

    CUDA_VISIBLE_DEVICES=2,3 python bench/compare/run_vibevoice.py data/ali_eval \\
        --out bench/vibevoice
    python bench/run_bench.py data/ali_eval --out bench/vibevoice --score-only  # (speakerlens env)

Each recording is transcribed in one pass with the model's default settings. The model
has no option for the number of speakers, and no hotword prompt is given. Traditional
characters are converted to Simplified.

If the output is cut off (for example a turn repeated until the output limit), the complete
turns before that are kept and the log says how far into the recording they reach.
--resume skips recordings already done.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Timer, audio_files, to_simplified, write_run  # noqa: E402

MODEL = "microsoft/VibeVoice-ASR-HF"


def _is_turn(item: object) -> bool:
    return (
        isinstance(item, dict)
        and "Content" in item
        and all(isinstance(item.get(k) or 0.0, (int, float)) for k in ("Start", "End"))
    )


def salvage(raw: str) -> list[dict]:
    """The complete turns at the start of output that doesn't parse as a whole, e.g.
    '[{"Start":0.0,"End":3.2,"Speaker":0,"Content":"..."},{"Start":3.2,"Content":"cut of'.
    Each turn is read as JSON, so braces or quotes inside the text are handled."""
    decoder = json.JSONDecoder()
    i = raw.find("[")
    if i < 0:
        return []
    i += 1
    turns = []
    while True:
        while i < len(raw) and raw[i] in " \t\r\n,":
            i += 1
        try:
            item, i = decoder.raw_decode(raw, i)
        except json.JSONDecodeError:
            return turns  # the turn that was cut off, or the closing ']'
        if _is_turn(item):
            turns.append(item)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--out", type=Path, default=Path("bench/vibevoice"))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument(
        "--chunk",
        type=int,
        default=None,
        help="tokenizer_chunk_size (multiple of 3200; lower it if memory runs out)",
    )
    ap.add_argument("--resume", action="store_true", help="Skip recordings already done")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model)
    model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.bfloat16
    )
    out = args.out / args.dataset.name
    extra = {"tokenizer_chunk_size": args.chunk} if args.chunk else {}

    for path in audio_files(args.dataset, args.shard, args.limit):
        fid, timer = path.stem, Timer()
        if args.resume and (out / f"{fid}.stm").exists():
            print(f"{fid}: already done")
            continue
        inputs = processor.apply_transcription_request(audio=str(path)).to(
            model.device, model.dtype
        )
        with torch.inference_mode():
            ids = model.generate(**inputs, **extra)
        new = ids[:, inputs["input_ids"].shape[1] :]
        note = ""
        try:
            parsed = processor.decode(new, return_format="parsed")[0]
        except (ValueError, json.JSONDecodeError):
            parsed = None
        if not isinstance(parsed, list):  # cut off or malformed: keep the complete turns
            raw = processor.batch_decode(new, skip_special_tokens=True)[0]
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{fid}.raw.txt").write_text(raw, encoding="utf-8")
            parsed = salvage(raw)
            reached = max((float(p.get("End") or 0.0) for p in parsed), default=0.0)
            limit = model.generation_config.max_new_tokens
            why = (
                f"output limit of {limit} tokens reached"
                if limit and new.shape[1] >= limit
                else f"stopped after {new.shape[1]} tokens"
            )
            note = (
                f" (output cut off, {why}: kept {len(parsed)} complete turns, "
                f"up to {reached:.0f} s)"
            )
        turns = [
            (
                f"S{p.get('Speaker', 0)}",
                float(p.get("Start") or 0.0),
                float(p.get("End") or 0.0),
                to_simplified(str(p.get("Content", ""))),
            )
            for p in parsed
            if _is_turn(p)
        ]
        write_run(out, fid, turns)
        print(f"{fid}: {len(turns)} turns, {timer}{note}", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
