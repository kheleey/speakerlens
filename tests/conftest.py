"""Fakes so the whole pipeline can be tested without downloading any model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from speakerlens.diarization.base import DiarizationResult, DiarizedSegment, Diarizer
from speakerlens.langid import LanguageIdentifier
from speakerlens.plugins.base import (
    ANY_LANGUAGE,
    LanguagePlugin,
    PluginManifest,
    PluginResult,
    TurnContext,
)
from speakerlens.plugins.registry import PluginRegistry
from speakerlens.types import Word

SR = 16_000


class FakePlugin(LanguagePlugin):
    def __init__(
        self,
        plugin_id: str,
        languages: list[str],
        text: str,
        confidence: float | None = 0.9,
        mixed: bool = False,
        fallback: bool = False,
        priority: int = 0,
    ) -> None:
        self.text = text
        self.confidence = confidence
        self.calls: list[TurnContext] = []
        self.manifest = PluginManifest(
            id=plugin_id,
            languages=languages,
            model="fake",
            license="MIT",
            mixed=mixed,
            fallback=fallback,
            priority=priority,
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int, ctx: TurnContext) -> PluginResult:
        self.calls.append(ctx)
        return PluginResult(
            text=f"{self.text} {ctx.turn_id}",
            words=[Word(text=self.text, start=0.0, end=0.5, confidence=self.confidence)],
            raw_confidence=self.confidence,
            language=ctx.language,
        )


class FakeDiarizer(Diarizer):
    """Alice speaks 0-4 s and 9-12 s, Bob 4.2-9 s, with a short overlap at 8.5-9 s."""

    model = "fake"

    def diarize(self, audio, sample_rate, num_speakers=None, min_speakers=None, max_speakers=None):  # type: ignore[no-untyped-def]
        return DiarizationResult(
            segments=[
                DiarizedSegment(speaker="SPEAKER_00", start=0.0, end=4.0),
                DiarizedSegment(speaker="SPEAKER_01", start=4.2, end=9.0),
                DiarizedSegment(speaker="SPEAKER_00", start=9.0, end=12.0),
            ],
            overlaps=[(8.5, 9.0)],
        )


class FakeLanguageID(LanguageIdentifier):
    """Returns preset probabilities per call, cycling through `script`."""

    def __init__(self, script: list[dict[str, float]]) -> None:
        self.script = script
        self.i = 0

    def probs(self, audio, sample_rate, allowed):  # type: ignore[no-untyped-def]
        out = self.script[self.i % len(self.script)]
        self.i += 1
        return out


class FakeLLM:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str]] = []

    def chat_json(self, system: str, user: str, temperature: float = 0.2) -> dict[str, Any]:
        self.calls.append((system, user))
        return self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]


@pytest.fixture
def registry() -> PluginRegistry:
    reg = PluginRegistry()
    reg.register(FakePlugin("en", ["en"], "hello", 0.9, priority=10), check_available=False)
    reg.register(FakePlugin("zh", ["zh"], "你好", 0.6, priority=10), check_available=False)
    reg.register(
        FakePlugin("mixed", ["zh", "en"], "混合 mixed", 0.7, mixed=True), check_available=False
    )
    reg.register(
        FakePlugin("fallback", [ANY_LANGUAGE], "fallback", 0.5, fallback=True),
        check_available=False,
    )
    return reg


@pytest.fixture
def silence() -> np.ndarray:
    return np.zeros(12 * SR, dtype=np.float32)
