"""SenseVoice-Small via FunASR: Mandarin specialist and mixed Mandarin-English plugin.

Model license: FunASR Model License v1.1 (commercial use allowed with attribution).
SenseVoice reports no confidence score or word timestamps through the FunASR API, so
`raw_confidence` is None and `words` is empty.
"""

from __future__ import annotations

import re
import threading
from collections import Counter
from typing import Any

import numpy as np

from speakerlens.plugins.base import LanguagePlugin, PluginManifest, PluginResult, TurnContext

_models: dict[tuple[str, str, str], Any] = {}
_lock = threading.Lock()

# SenseVoice prefixes each segment with tags: <|zh|><|NEUTRAL|><|Speech|><|withitn|>.
_TAG = re.compile(r"<\|[^|]*\|>")
_LANG_TAG = re.compile(r"<\|(zh|en|yue|ja|ko)\|>")


def clean_sensevoice_text(raw: str) -> tuple[str, str | None]:
    """Remove SenseVoice's language/emotion/event tags. Returns (text, language), where
    language is the language SenseVoice tagged most often, or "a+b" when segments differ."""
    found = Counter(_LANG_TAG.findall(raw))
    text = re.sub(r"\s+", " ", _TAG.sub(" ", raw)).strip()
    # Tags between CJK characters leave a stray space ("今天 天气"); remove it.
    text = re.sub(r"(?<=[㐀-鿿　-〿＀-￯]) (?=[㐀-鿿])", "", text)
    if not found:
        return text, None
    if len(found) == 1:
        return text, next(iter(found))
    return text, "+".join(sorted(found))


def get_sensevoice_model(model: str, hub: str, device: str) -> Any:
    key = (model, hub, device)
    with _lock:
        if key not in _models:
            from funasr import AutoModel

            _models[key] = AutoModel(
                model=model,
                hub=hub,  # "ms" = ModelScope, "hf" = Hugging Face
                device="cuda:0" if device == "cuda" else device,
                disable_update=True,
            )
        return _models[key]


class SenseVoicePlugin(LanguagePlugin):
    def __init__(
        self,
        plugin_id: str,
        languages: list[str],
        model: str,
        hub: str,
        device: str = "cpu",
        language_mode: str = "auto",  # "zh", "en", "yue", "ja", "ko" or "auto"
        mixed: bool = False,
        priority: int = 0,
    ) -> None:
        self.model = model
        self.hub = hub
        self.device = device
        self.language_mode = language_mode
        self.manifest = PluginManifest(
            id=plugin_id,
            languages=languages,
            model=model,
            license="FunASR Model License v1.1",
            mixed=mixed,
            priority=priority,
            requires=["funasr", "torch"],
            description="SenseVoice-Small via FunASR"
            + (" (mixed speech)" if mixed else f" ({', '.join(languages)})"),
        )

    def load(self) -> None:
        get_sensevoice_model(self.model, self.hub, self.device)

    def transcribe(self, audio: np.ndarray, sample_rate: int, ctx: TurnContext) -> PluginResult:
        model = get_sensevoice_model(self.model, self.hub, self.device)
        result = model.generate(
            input=audio.astype(np.float32),
            cache={},
            language=self.language_mode,
            use_itn=True,  # punctuation and number formatting
            batch_size_s=60,
        )
        raw_text = " ".join(str(r.get("text", "")) for r in result or [])
        text, tagged = clean_sensevoice_text(raw_text)
        if self.manifest.mixed:
            language = "+".join(sorted(self.manifest.languages))
        else:
            language = ctx.language or tagged or self.manifest.languages[0]
        return PluginResult(text=text, raw_confidence=None, language=language)
