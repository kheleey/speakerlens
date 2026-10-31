"""Write a Transcript as Markdown, subtitles, JSON, or benchmark formats (RTTM, STM)."""

from __future__ import annotations

from pathlib import Path

from speakerlens.types import SummaryPoint, Transcript

FORMATS = ("md", "json", "srt", "vtt", "rttm", "stm")


def timestamp(seconds: float, sep: str = ".", millis: bool = False) -> str:
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    base = f"{h:02d}:{m:02d}:{s:02d}"
    return f"{base}{sep}{ms:03d}" if millis else base


def to_json(transcript: Transcript) -> str:
    return transcript.model_dump_json(indent=2)


def _point_line(transcript: Transcript, point: SummaryPoint) -> str:
    times = []
    for tid in point.turn_ids:
        turn = transcript.turn(tid)
        if turn:
            times.append(timestamp(turn.start))
    who = f" (with {transcript.speaker_label(point.with_speaker)})" if point.with_speaker else ""
    cite = f" [{', '.join(times)}]" if times else ""
    return f"- {point.text}{who}{cite}"


def to_markdown(transcript: Transcript) -> str:
    out = [f"# Transcript: {Path(transcript.source).name}", ""]
    out.append(
        f"Duration {timestamp(transcript.duration)} · Languages: "
        f"{', '.join(transcript.languages)} · Speakers: {len(transcript.speakers)}"
    )
    out.append("")

    summary = transcript.summary
    if summary:
        out += ["## Summary", ""]
        if summary.overview:
            out += [summary.overview, ""]
        if summary.rejected_points:
            out.append(
                f"_{len(summary.rejected_points)} summary point(s) were dropped because their "
                "citations did not match the speaker; see the JSON output._"
            )
            out.append("")
        titles = {
            "main_points": "Main points",
            "agreements": "Agreed with",
            "disagreements": "Disagreed with",
            "open_questions": "Open questions",
        }
        for spk in summary.speakers:
            out += [f"### {transcript.speaker_label(spk.speaker)}", ""]
            for cat, title in titles.items():
                points = getattr(spk, cat)
                if points:
                    out.append(f"**{title}**")
                    out += [_point_line(transcript, p) for p in points]
                    out.append("")

    out += ["## Transcript", ""]
    for t in transcript.turns:
        if not t.text:
            continue
        flags = []
        if t.overlap:
            flags.append("overlap")
        if t.confidence is not None and t.confidence < 0.5:
            flags.append(f"low confidence {t.confidence:.2f}")
        flag = f" _({', '.join(flags)})_" if flags else ""
        lang = f" `{t.language}`" if t.language else ""
        out.append(
            f"**{transcript.speaker_label(t.speaker)}** [{timestamp(t.start)}]{lang}{flag}  "
        )
        out.append(t.text)
        if t.translation and t.translation != t.text:
            out.append(f"> {t.translation}")
        out.append("")
    return "\n".join(out)


def _subtitles(transcript: Transcript, vtt: bool, use_translation: bool) -> str:
    sep = "." if vtt else ","
    out = ["WEBVTT", ""] if vtt else []
    n = 0
    for t in transcript.turns:
        text = t.translation if use_translation and t.translation else t.text
        if not text:
            continue
        n += 1
        if not vtt:
            out.append(str(n))
        start = timestamp(t.start, sep, millis=True)
        end = timestamp(t.end, sep, millis=True)
        out.append(f"{start} --> {end}")
        out.append(f"{transcript.speaker_label(t.speaker)}: {text}")
        out.append("")
    return "\n".join(out)


def to_srt(transcript: Transcript, use_translation: bool = False) -> str:
    return _subtitles(transcript, vtt=False, use_translation=use_translation)


def to_vtt(transcript: Transcript, use_translation: bool = False) -> str:
    return _subtitles(transcript, vtt=True, use_translation=use_translation)


def to_rttm(transcript: Transcript, file_id: str) -> str:
    """Diarization output for DER scoring (pyannote.metrics, dscore)."""
    return (
        "\n".join(
            f"SPEAKER {file_id} 1 {t.start:.3f} {t.duration:.3f} <NA> <NA> {t.speaker} <NA> <NA>"
            for t in transcript.turns
        )
        + "\n"
    )


def to_stm(transcript: Transcript, file_id: str) -> str:
    """Speaker-attributed transcript for cpWER/tcpWER scoring with MeetEval."""
    return (
        "\n".join(
            f"{file_id} 1 {t.speaker} {t.start:.3f} {t.end:.3f} {t.text}"
            for t in transcript.turns
            if t.text
        )
        + "\n"
    )


def write_outputs(
    transcript: Transcript, out_dir: str | Path, stem: str, formats: list[str]
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writers = {
        "md": lambda: to_markdown(transcript),
        "json": lambda: to_json(transcript),
        "srt": lambda: to_srt(transcript),
        "vtt": lambda: to_vtt(transcript),
        "rttm": lambda: to_rttm(transcript, stem),
        "stm": lambda: to_stm(transcript, stem),
    }
    paths = []
    for fmt in formats:
        if fmt not in writers:
            raise ValueError(f"Unknown format {fmt!r}. Choose from: {', '.join(FORMATS)}")
        path = out_dir / f"{stem}.{fmt}"
        path.write_text(writers[fmt](), encoding="utf-8")
        paths.append(path)
        if fmt == "srt" and transcript.target_language:
            translated = out_dir / f"{stem}.{transcript.target_language}.srt"
            translated.write_text(to_srt(transcript, use_translation=True), encoding="utf-8")
            paths.append(translated)
    return paths
