"""Whisper plugins via faster-whisper (MIT): English specialist and multilingual fallback."""

from __future__ import annotations

import math
import threading
from typing import Any

import numpy as np

from speakerlens.plugins.base import (
    ANY_LANGUAGE,
    LanguagePlugin,
    PluginManifest,
    PluginResult,
    TurnContext,
)
from speakerlens.types import Word

# Languages written without spaces between words.
_NO_SPACE_LANGUAGES = {"zh", "ja", "th", "lo", "my", "km", "yue"}

_models: dict[tuple[str, str], Any] = {}
_lock = threading.Lock()


def get_whisper_model(name: str, device: str) -> Any:
    """Load each Whisper model once and share it between plugins and language ID."""
    key = (name, device)
    with _lock:
        if key not in _models:
            from faster_whisper import WhisperModel

            compute_type = "float16" if device == "cuda" else "int8"
            _models[key] = WhisperModel(name, device=device, compute_type=compute_type)
        return _models[key]


def join_text(parts: list[str], language: str | None) -> str:
    sep = "" if language in _NO_SPACE_LANGUAGES else " "
    return sep.join(p.strip() for p in parts if p.strip())


class WhisperPlugin(LanguagePlugin):
    def __init__(
        self,
        plugin_id: str,
        languages: list[str],
        model_name: str,
        device: str = "cpu",
        fallback: bool = False,
        priority: int = 0,
        beam_size: int = 5,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.beam_size = beam_size
        self.manifest = PluginManifest(
            id=plugin_id,
            languages=languages,
            model=f"faster-whisper/{model_name}",
            license="MIT",
            fallback=fallback,
            priority=priority,
            requires=["faster_whisper"],
            description="Whisper via faster-whisper"
            + (" (any language)" if fallback else f" ({', '.join(languages)})"),
        )

    def load(self) -> None:
        get_whisper_model(self.model_name, self.device)

    def _language_for(self, ctx: TurnContext) -> str | None:
        if self.manifest.fallback or ANY_LANGUAGE in self.manifest.languages:
            return ctx.language  # None lets Whisper detect it
        if ctx.language in self.manifest.languages:
            return ctx.language
        return self.manifest.languages[0]

    def transcribe(self, audio: np.ndarray, sample_rate: int, ctx: TurnContext) -> PluginResult:
        model = get_whisper_model(self.model_name, self.device)
        language = self._language_for(ctx)
        segments, info = model.transcribe(
            audio,
            language=language,
            beam_size=self.beam_size,
            word_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=False,  # diarization already removed the silence
        )
        texts: list[str] = []
        words: list[Word] = []
        logprob_sum, logprob_weight = 0.0, 0.0
        for seg in segments:
            texts.append(seg.text)
            weight = max(seg.end - seg.start, 0.01)
            logprob_sum += seg.avg_logprob * weight
            logprob_weight += weight
            for w in seg.words or []:
                words.append(
                    Word(text=w.word.strip(), start=w.start, end=w.end, confidence=w.probability)
                )

        if words:
            raw = float(np.mean([w.confidence for w in words if w.confidence is not None]))
        elif logprob_weight > 0:
            raw = math.exp(logprob_sum / logprob_weight)
        else:
            raw = None
        detected = language or info.language
        return PluginResult(
            text=join_text(texts, detected), words=words, raw_confidence=raw, language=detected
        )
