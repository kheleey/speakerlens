"""The language plugin contract.

A plugin wraps one speech-to-text model for one or more languages. The pipeline sends it
one speaker turn at a time and expects text, optional word timings and a raw confidence
score. The calibration layer later turns that raw score into a trustworthy probability,
so plugins never need to agree on what "0.8" means.
"""

from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field

from speakerlens.types import Word

ANY_LANGUAGE = "*"


class PluginManifest(BaseModel):
    id: str  # unique, e.g. "whisper-en"
    languages: list[str]  # ISO 639-1 codes; ["*"] means any language (fallback)
    model: str  # model identifier, shown in `speakerlens plugins`
    license: str  # license of the model weights
    mixed: bool = False  # handles switching between its languages inside one turn
    fallback: bool = False  # used when no specialist plugin exists for a language
    priority: int = 0  # higher wins when two plugins cover the same language
    needs_gpu: bool = False
    requires: list[str] = Field(default_factory=list)  # Python modules that must be importable
    description: str = ""


class PluginResult(BaseModel):
    text: str
    # Word times are in seconds relative to the start of the audio the plugin received.
    words: list[Word] = Field(default_factory=list)
    # The model's own confidence for the whole turn, 0..1, or None if it has none.
    raw_confidence: float | None = None
    language: str | None = None
    # Set by wrapper plugins (e.g. FallbackPlugin) to the id of the model whose text is used.
    used: str | None = None
    # Problems the plugin noticed and repaired, e.g. "loop" for a repetition loop.
    notes: list[str] = Field(default_factory=list)


@dataclass
class TurnContext:
    turn_id: str
    speaker: str
    start: float
    end: float
    language: str | None  # the language the router chose, None for mixed turns


class LanguagePlugin(ABC):
    manifest: PluginManifest

    @property
    def id(self) -> str:
        return self.manifest.id

    def is_available(self) -> bool:
        """True when every module in `manifest.requires` can be imported."""
        return all(importlib.util.find_spec(m) is not None for m in self.manifest.requires)

    def load(self) -> None:  # noqa: B027 - optional hook
        """Optional: load model weights ahead of the first turn."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int, ctx: TurnContext) -> PluginResult:
        """Transcribe one turn of 16 kHz mono float32 audio."""
