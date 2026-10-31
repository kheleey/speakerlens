"""Shared helpers for running other systems on the benchmark sets (no speakerlens import:
these scripts run in the other systems' own environments)."""

from __future__ import annotations

import time
from pathlib import Path

AUDIO_EXT = {".wav", ".flac", ".mp3", ".m4a"}


def audio_files(dataset: Path, shard: str | None = None, limit: int | None = None) -> list[Path]:
    files = sorted(p for p in (dataset / "audio").iterdir() if p.suffix.lower() in AUDIO_EXT)
    files = files[:limit]
    if shard:
        k, n = (int(x) for x in shard.split("/"))
        files = files[k - 1 :: n]
    return files


def speakers_in_ref(dataset: Path, file_id: str) -> int:
    rttm = dataset / "ref" / f"{file_id}.rttm"
    return len({ln.split()[7] for ln in rttm.read_text().splitlines() if ln.startswith("SPEAKER")})


def write_run(out: Path, file_id: str, turns: list[tuple[str, float, float, str]]) -> None:
    """Turns (speaker, start, end, text) as <out>/<id>.stm and .rttm, the layout
    bench/run_bench.py --score-only and bench/error_breakdown.py --run read."""
    out.mkdir(parents=True, exist_ok=True)
    turns = sorted((t for t in turns if t[2] > t[1]), key=lambda t: t[1])
    stm = [
        f"{file_id} 1 {s} {a:.3f} {b:.3f} {' '.join(x.split())}"
        for s, a, b, x in turns
        if x.strip()
    ]
    rttm = [
        f"SPEAKER {file_id} 1 {a:.3f} {b - a:.3f} <NA> <NA> {s} <NA> <NA>" for s, a, b, _ in turns
    ]
    (out / f"{file_id}.stm").write_text("\n".join(stm) + "\n", encoding="utf-8")
    (out / f"{file_id}.rttm").write_text("\n".join(rttm) + "\n", encoding="utf-8")


def turns_by_word_speaker(
    segments: list[dict], default: str = "UNKNOWN"
) -> list[tuple[str, float, float, str]]:
    """Turns (speaker, start, end, text) that follow WhisperX's per-word speakers: a
    segment is split wherever the speaker of its words changes. (Its standard output gives
    the whole segment one speaker, and a Mandarin segment can run for 30 s, since sentences
    are only split at '.', '?' and '!'.)

    Words without timing or without a speaker (no diarized speech under them) stay with
    the word before them. The text keeps the segment's own spacing, so English words
    inside Mandarin, which WhisperX aligns letter by letter, come out whole. Segments
    without word timing stay whole, under the segment's speaker."""
    turns: list[tuple[str, float, float, str]] = []
    for seg in segments:
        text = str(seg.get("text", ""))
        seg_speaker = str(seg.get("speaker") or default)
        words = [w for w in seg.get("words") or [] if str(w.get("word", "")).strip()]
        if not any(w.get("start") is not None for w in words):
            if seg.get("start") is not None and seg.get("end") is not None and text.strip():
                turns.append((seg_speaker, float(seg["start"]), float(seg["end"]), text.strip()))
            continue
        runs: list[dict] = []  # speaker, text pieces, times
        cursor = 0
        speaker: str | None = None
        for w in words:
            token = str(w["word"]).strip()
            at = text.find(token, cursor)
            if at < 0:  # not found as written: keep the word, with a space before it
                piece = " " + token
            else:
                piece, cursor = text[cursor:at] + token, at + len(token)
            new = w.get("speaker")
            if not runs or (new is not None and str(new) != speaker):
                speaker = str(new) if new is not None else (speaker or seg_speaker)
                runs.append({"speaker": speaker, "pieces": [], "times": []})
            runs[-1]["pieces"].append(piece)
            if w.get("start") is not None:
                start = float(w["start"])
                end = float(w["end"]) if w.get("end") is not None else start
                runs[-1]["times"] += [start, end]
        pending = ""  # text of a run with no timed word, given to the next run
        for run in runs:
            body = pending + "".join(run["pieces"])
            if not run["times"]:
                pending = body
                continue
            pending = ""
            turns.append((run["speaker"], min(run["times"]), max(run["times"]), body.strip()))
        if pending.strip() and turns:
            spk, a, b, last = turns[-1]
            turns[-1] = (spk, a, b, (last + pending).strip())
    return turns


def to_simplified(text: str) -> str:
    """Traditional -> Simplified Chinese (the references are Simplified), if zhconv is
    installed; Whisper sometimes writes Mandarin in Traditional characters."""
    try:
        import zhconv
    except ImportError:
        return text
    return str(zhconv.convert(text, "zh-cn"))


class Timer:
    def __init__(self) -> None:
        self.t0 = time.time()

    def __str__(self) -> str:
        return f"{time.time() - self.t0:.0f} s"
