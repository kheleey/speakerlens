"""A wrapper plugin: a primary model with a backup for when its output looks broken.

LLM-based speech models such as Fun-ASR-Nano can stop early at a language switch, or get
stuck in a repetition loop. The backup model (SenseVoice) also transcribes every turn of at
least `min_seconds`, and every turn where the primary looped or wrote in the wrong script
(Japanese kana in a Mandarin/English recording). The backup's text is used when the primary
has fewer than `ratio` times as many tokens, more than `max_ratio` times as many plus 10,
looped, or wrote in the wrong script.
"""

from __future__ import annotations

import numpy as np

from speakerlens.metrics import tokenize
from speakerlens.plugins.base import LanguagePlugin, PluginResult, TurnContext


def use_backup(
    n_primary: int, n_backup: int, looped: bool, ratio: float, max_ratio: float = 3.0
) -> bool:
    """The decision rule: does the backup replace the primary's text?"""
    if n_backup == 0:
        return False
    return looped or n_primary < ratio * n_backup or n_primary > max_ratio * n_backup + 10


class FallbackPlugin(LanguagePlugin):
    def __init__(
        self,
        primary: LanguagePlugin,
        backup: LanguagePlugin,
        ratio: float = 0.6,
        min_seconds: float = 1.5,
        max_ratio: float = 3.0,
    ) -> None:
        self.primary = primary
        self.backup = backup
        self.ratio = ratio
        self.max_ratio = max_ratio
        self.min_seconds = min_seconds
        self.manifest = primary.manifest.model_copy(
            update={
                "requires": sorted(set(primary.manifest.requires) | set(backup.manifest.requires)),
                "description": f"{primary.manifest.description}; falls back to "
                f"{backup.manifest.model} when it stops early or loops",
            }
        )

    def is_available(self) -> bool:
        return self.primary.is_available() and self.backup.is_available()

    def load(self) -> None:
        self.primary.load()
        self.backup.load()

    def transcribe(self, audio: np.ndarray, sample_rate: int, ctx: TurnContext) -> PluginResult:
        first = self.primary.transcribe(audio, sample_rate, ctx)
        looped = "loop" in first.notes
        wrong_script = "wrong_script" in first.notes
        if len(audio) / sample_rate < self.min_seconds and not (looped or wrong_script):
            return first
        second = self.backup.transcribe(audio, sample_rate, ctx)
        n_first, n_second = len(tokenize(first.text)), len(tokenize(second.text))
        if wrong_script or use_backup(n_first, n_second, looped, self.ratio, self.max_ratio):
            return second.model_copy(
                update={"language": first.language, "used": self.backup.id, "notes": first.notes}
            )
        return first
