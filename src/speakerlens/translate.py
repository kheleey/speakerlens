"""Translate every turn into the reader's language with an LLM."""

from __future__ import annotations

import json
import logging

from speakerlens.llm import JSONChat, LLMError, LLMUnavailable
from speakerlens.types import Transcript, Turn

log = logging.getLogger(__name__)

LANGUAGE_NAMES = {
    "en": "English",
    "zh": "Simplified Chinese",
    "zh-tw": "Traditional Chinese",
    "yue": "Cantonese (written)",
    "ja": "Japanese",
    "ko": "Korean",
    "id": "Indonesian",
    "ms": "Malay",
    "vi": "Vietnamese",
    "th": "Thai",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
}

SYSTEM_PROMPT = """You are a professional interpreter translating a multi-person conversation.
Translate each utterance into {target}. Keep the meaning, tone, names and technical terms.
The speech may mix languages; translate all of it. Do not merge, split or skip utterances.
Reply with JSON only, in this shape:
{{"translations": [{{"id": "<utterance id>", "text": "<translation>"}}]}}"""


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code.lower(), code)


def translate_transcript(
    transcript: Transcript, target: str, llm: JSONChat, batch_size: int = 20
) -> int:
    """Fill `turn.translation` in place. Turns already in the target language are copied.

    Returns how many turns could not be translated. A batch that fails is split in half
    and retried, so one bad reply costs at most one turn.
    """
    transcript.target_language = target
    pending = []
    for turn in transcript.turns:
        if not turn.text:
            continue
        if turn.language == target:
            turn.translation = turn.text
        else:
            pending.append(turn)

    system = SYSTEM_PROMPT.format(target=language_name(target))
    failed = 0
    for i in range(0, len(pending), batch_size):
        failed += _translate_batch(transcript, pending[i : i + batch_size], system, llm)
    return failed


def _translate_batch(transcript: Transcript, batch: list[Turn], system: str, llm: JSONChat) -> int:
    items = [
        {"id": t.id, "speaker": transcript.speaker_label(t.speaker), "text": t.text} for t in batch
    ]
    try:
        data = llm.chat_json(system, json.dumps({"utterances": items}, ensure_ascii=False))
    except LLMUnavailable:
        raise
    except LLMError as exc:
        if len(batch) == 1:
            log.warning("Could not translate %s: %s", batch[0].id, exc)
            return 1
        half = len(batch) // 2
        return _translate_batch(transcript, batch[:half], system, llm) + _translate_batch(
            transcript, batch[half:], system, llm
        )

    by_id = {
        str(item.get("id")): str(item.get("text", ""))
        for item in data.get("translations", []) or []
        if isinstance(item, dict)
    }
    missing = 0
    for t in batch:
        if by_id.get(t.id):
            t.translation = by_id[t.id]
        else:
            missing += 1
            log.warning("No translation returned for %s", t.id)
    return missing
