"""Per-speaker summaries that stay attributable.

The LLM must cite turn ids for every point. Only the cited turns that person actually spoke
are kept, and points citing none of them are rejected, so every point is backed by the
person's own words.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from speakerlens.llm import JSONChat, LLMError, LLMUnavailable
from speakerlens.metrics import tokenize
from speakerlens.translate import language_name
from speakerlens.types import RejectedPoint, SpeakerSummary, Summary, SummaryPoint, Transcript

log = logging.getLogger(__name__)

CATEGORIES = ("main_points", "agreements", "disagreements", "open_questions")
RELATIONS = ("agreements", "disagreements")
# Summary languages written in Chinese, Japanese or Korean script: for others, a point that
# is mostly in that script is a pasted transcript line, not a summary.
_CJK_LANGUAGES = {"zh", "ja", "ko", "yue"}
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")
# Bumped when the instructions change, so saved summaries from older ones aren't mixed in.
PROMPT_VERSION = 4

SYSTEM_PROMPT = """You analyze conversations between several people.
You get a transcript where each line is: [turn id] speaker (time): text
Write a summary for EACH speaker, in {language}. Only use what is in the transcript.

Rules:
- Every point must cite the turn ids where THAT speaker said it, e.g. ["T0003", "T0007"].
- Never cite another person's turn as evidence for a speaker's point.
- Credit an idea to whoever raised it FIRST. When one speaker proposes something and others
  go along, it is a main point of the proposer and an agreement of the others (with the
  proposer in "with"), never the other way round.
- "agreements" and "disagreements" are only for responses to something another speaker said
  EARLIER; name that speaker in "with". If others object to this speaker's idea, that is
  their disagreement, not this speaker's.
- Don't add reasons, details or examples taken from another speaker's lines.
- Be complete: include every distinct point each speaker makes (proposals, opinions, facts,
  questions, agreements, disagreements). Don't drop or merge points to keep it short.
- Filler on its own (对, 嗯, 好的, 是的, "right", "OK") is not a point. Count a short "yes"
  as an agreement only when it is clear what was agreed to, and say what.
- Write every point in {language} in your own words; never paste transcript lines.
- Don't repeat a point you already wrote for that speaker.
- Copy numbers, dates, prices and names exactly as they are in the transcript.
- Keep each point to one sentence.

Reply with JSON only, in this shape:
{{
  "overview": "2-3 sentences on what the conversation was about",
  "speakers": [
    {{
      "speaker": "<speaker name exactly as in the transcript>",
      "main_points": [{{"text": "...", "turn_ids": ["T0001"]}}],
      "agreements": [{{"text": "...", "with": "<other speaker>", "turn_ids": ["T0004"]}}],
      "disagreements": [{{"text": "...", "with": "<other speaker>", "turn_ids": ["T0009"]}}],
      "open_questions": [{{"text": "...", "turn_ids": ["T0012"]}}]
    }}
  ]
}}"""


def _fmt_time(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def transcript_lines(transcript: Transcript, use_translation: bool = False) -> list[str]:
    lines = []
    for t in transcript.turns:
        text = (t.translation if use_translation and t.translation else t.text).strip()
        if text:
            label = transcript.speaker_label(t.speaker)
            lines.append(f"[{t.id}] {label} ({_fmt_time(t.start)}): {text}")
    return lines


def _chunks(lines: list[str], max_chars: int) -> list[list[str]]:
    chunks: list[list[str]] = [[]]
    size = 0
    for line in lines:
        if chunks[-1] and size + len(line) > max_chars:
            chunks.append([])
            size = 0
        chunks[-1].append(line)
        size += len(line) + 1
    return [c for c in chunks if c]


def summarize(
    transcript: Transcript,
    llm: JSONChat,
    language: str = "en",
    use_translation: bool = False,
    max_chars: int = 60_000,
    strict: bool = False,
) -> Summary:
    """Summarize per speaker. Long transcripts are summarized in chunks and the
    attributed points are combined. With `strict`, a chunk the model fails on raises
    instead of being left out."""
    lines = transcript_lines(transcript, use_translation)
    system = SYSTEM_PROMPT.format(language=language_name(language))
    combined = Summary(language=language, prompt_version=PROMPT_VERSION)
    overviews: list[str] = []
    for chunk in _chunks(lines, max_chars):
        _summarize_chunk(chunk, system, transcript, llm, language, combined, overviews, strict)
    combined.overview = " ".join(overviews)
    drop_duplicates(combined)
    return combined


def _summarize_chunk(
    chunk: list[str],
    system: str,
    transcript: Transcript,
    llm: JSONChat,
    language: str,
    combined: Summary,
    overviews: list[str],
    strict: bool = False,
) -> None:
    """Summarize one chunk; if the model fails on it, split it in half and try again."""
    try:
        data = llm.chat_json(system, "\n".join(chunk))
    except LLMUnavailable:
        raise
    except LLMError as exc:
        if len(chunk) < 4:
            if strict:
                raise
            log.warning("Could not summarize a %d-line chunk: %s", len(chunk), exc)
            return
        half = len(chunk) // 2
        for piece in (chunk[:half], chunk[half:]):
            _summarize_chunk(piece, system, transcript, llm, language, combined, overviews, strict)
        return
    part = check_attribution(data, transcript, language)
    if part.overview:
        overviews.append(part.overview)
    _merge(combined, part)


def pasted_line(text: str, language: str) -> bool:
    """A point mostly in Chinese/Japanese/Korean script when the summary language isn't one
    of those: the model copied a transcript line instead of summarizing it. Points that
    just keep a few original words ("suggested a 赠品 with the logo") are fine."""
    if language.split("-")[0].lower() in _CJK_LANGUAGES:
        return False
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and len(_CJK.findall(text)) > 0.5 * len(letters)


def _key(text: str) -> str:
    return re.sub(r"[\W_]+", " ", text.lower()).strip()


def drop_duplicates(summary: Summary) -> int:
    """Remove points repeated for the same speaker (same words, ignoring case and
    punctuation), keeping the first. Returns how many were removed."""
    removed = 0
    for spk in summary.speakers:
        seen: set[str] = set()
        for cat in CATEGORIES:
            kept = []
            for point in getattr(spk, cat):
                key = _key(point.text)
                if key in seen:
                    removed += 1
                    continue
                seen.add(key)
                kept.append(point)
            setattr(spk, cat, kept)
    return removed


def _merge(into: Summary, part: Summary) -> None:
    existing = {s.speaker: s for s in into.speakers}
    for spk in part.speakers:
        if spk.speaker not in existing:
            existing[spk.speaker] = SpeakerSummary(speaker=spk.speaker)
            into.speakers.append(existing[spk.speaker])
        for cat in CATEGORIES:
            getattr(existing[spk.speaker], cat).extend(getattr(spk, cat))
    into.rejected_points.extend(part.rejected_points)


def _resolver(transcript: Transcript) -> dict[str, str]:
    """Map names and ids (case-insensitive) to speaker ids."""
    lookup = {}
    for s in transcript.speakers:
        lookup[s.id.lower()] = s.id
        lookup[s.label.lower()] = s.id
    return lookup


def check_attribution(data: dict[str, Any], transcript: Transcript, language: str) -> Summary:
    owner = {t.id: t.speaker for t in transcript.turns}
    resolve = _resolver(transcript)
    summary = Summary(
        overview=str(data.get("overview", "")).strip(),
        language=language,
        prompt_version=PROMPT_VERSION,
    )

    for entry in data.get("speakers", []):
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("speaker", "")).strip()
        speaker_id = resolve.get(name.lower())
        if speaker_id is None:
            log.warning("Summary mentions unknown speaker %r; skipped.", name)
            continue
        result = SpeakerSummary(speaker=speaker_id)
        for cat in CATEGORIES:
            for item in entry.get(cat, []) or []:
                if not isinstance(item, dict) or not item.get("text"):
                    continue
                text = str(item["text"]).strip()
                turn_ids = [str(x) for x in item.get("turn_ids", []) or []]
                own, reason = own_citations(turn_ids, owner, speaker_id)
                if not reason and pasted_line(text, language):
                    reason = "pasted a transcript line instead of a summary point"
                if reason:
                    summary.rejected_points.append(
                        RejectedPoint(
                            speaker=speaker_id,
                            category=cat,
                            text=text,
                            turn_ids=turn_ids,
                            reason=reason,
                        )
                    )
                    continue
                with_name = item.get("with")
                with_id = resolve.get(str(with_name).lower()) if with_name else None
                if cat in RELATIONS:
                    reason = relation_problem(own, speaker_id, with_id, transcript)
                    if reason:
                        summary.rejected_points.append(
                            RejectedPoint(
                                speaker=speaker_id,
                                category=cat,
                                text=text,
                                turn_ids=turn_ids,
                                reason=reason,
                            )
                        )
                        continue
                getattr(result, cat).append(
                    SummaryPoint(text=text, turn_ids=own, with_speaker=with_id)
                )
        summary.speakers.append(result)
    return summary


def relation_problem(
    own: list[str],
    speaker_id: str,
    with_id: str | None,
    transcript: Transcript,
    window: float = 120.0,
    min_tokens: int = 4,
) -> str | None:
    """Why an agreement/disagreement can't be right, or None. It needs the other person to
    have said something of substance (at least `min_tokens` words) before this speaker's
    last cited turn, at most `window` seconds before their first one. Catches reversed
    relations ("A disagrees with B" when B objected to A) where B had only said "对" or
    nothing before. An unnamed or unknown "with" isn't checked."""
    if with_id is None:
        return None
    if with_id == speaker_id:
        return "agrees or disagrees with themselves"
    turns = {t.id: t for t in transcript.turns}
    mine = [turns[i] for i in own if i in turns]
    if not mine:
        return None
    first = min(t.start for t in mine)
    last = max(t.start for t in mine)
    for t in transcript.turns:
        if (
            t.speaker == with_id
            and t.start < last
            and t.end >= first - window
            and len(tokenize(t.text)) >= min_tokens
        ):
            return None
    return (
        f"responds to nobody: {transcript.speaker_label(with_id)} said nothing of substance "
        "just before"
    )


def own_citations(
    turn_ids: list[str], owner: dict[str, str], speaker_id: str
) -> tuple[list[str], str | None]:
    """The cited turns this speaker actually spoke, or a reason to reject the point when
    there are none."""
    if not turn_ids:
        return [], "no turn cited"
    own = [t for t in turn_ids if owner.get(t) == speaker_id]
    if own:
        return own, None
    unknown = [t for t in turn_ids if t not in owner]
    if len(unknown) == len(turn_ids):
        return [], f"unknown turn ids: {', '.join(unknown)}"
    foreign = [t for t in turn_ids if t in owner]
    return [], f"cites only turns by someone else: {', '.join(foreign)}"
