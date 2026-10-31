"""Spoken-language identification, limited to the languages the user selected."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from speakerlens.plugins.whisper_plugin import get_whisper_model


class LanguageIdentifier(ABC):
    @abstractmethod
    def probs(self, audio: np.ndarray, sample_rate: int, allowed: list[str]) -> dict[str, float]:
        """Probability of each allowed language. Must sum to 1 over `allowed`."""


def restrict(all_probs: dict[str, float], allowed: list[str], top_k: int = 3) -> dict[str, float]:
    """Keep only allowed languages and renormalize. Falls back to uniform if none match.
    An empty `allowed` list means "any language": keep the `top_k` most likely."""
    if not allowed:
        allowed = [k for k, _ in sorted(all_probs.items(), key=lambda kv: -kv[1])[:top_k]]
        if not allowed:
            return {}
    kept = {lang: all_probs.get(lang, 0.0) for lang in allowed}
    total = sum(kept.values())
    if total <= 0:
        return {lang: 1.0 / len(allowed) for lang in allowed}
    return {lang: p / total for lang, p in kept.items()}


class FixedLanguage(LanguageIdentifier):
    """Used when the user selected exactly one language."""

    def __init__(self, language: str) -> None:
        self.language = language

    def probs(self, audio: np.ndarray, sample_rate: int, allowed: list[str]) -> dict[str, float]:
        return {self.language: 1.0}


class WhisperLanguageIdentifier(LanguageIdentifier):
    """Uses Whisper's language detector (99 languages). A small model is plenty."""

    def __init__(self, model_name: str = "small", device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device

    def probs(self, audio: np.ndarray, sample_rate: int, allowed: list[str]) -> dict[str, float]:
        model = get_whisper_model(self.model_name, self.device)
        # transcribe() returns a lazy generator; language detection runs up front,
        # and we never iterate the segments, so no decoding happens.
        _segments, info = model.transcribe(audio[: 30 * sample_rate], beam_size=1)
        all_probs = dict(info.all_language_probs or [(info.language, 1.0)])
        return restrict(all_probs, allowed)
