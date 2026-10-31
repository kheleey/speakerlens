"""Fun-ASR-Nano via FunASR: a speech LLM for Mandarin, English and code-switched speech.

Included in funasr >= 1.4. Needs a GPU for reasonable speed.
Model license: see https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512

- Audio is passed as a torch tensor; FunASR builds no prompt for a numpy array.
- The language stays on auto by default: forcing one can make the model translate the
  other language.
- The model gives no confidence score, so raw_confidence is None.
- Output length is capped by the turn's duration, so a repetition loop stays short.
"""

from __future__ import annotations

import re
import threading
from typing import Any

import numpy as np

from speakerlens.plugins.base import LanguagePlugin, PluginManifest, PluginResult, TurnContext
from speakerlens.textfix import collapse_loops, has_kana, join_spelled_letters

_models: dict[tuple[str, str, str], Any] = {}
_lock = threading.Lock()

# Prompt values Fun-ASR-Nano understands, for when a language is forced.
PROMPT_LANGUAGE = {"zh": "中文", "en": "英文", "ja": "日文"}


def max_new_tokens(seconds: float) -> int:
    """Generous upper bound on output length: fast speech is about 6 Chinese characters or
    4 English words a second, well under 12 LLM tokens a second."""
    return min(512, int(20 + 12 * seconds))


def get_nano_model(model: str, hub: str, device: str) -> Any:
    """The model, loaded once per (model, hub, device)."""
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


def clean_nano_text(text: str) -> str:
    text = re.sub(r"<\|[^|]*\|>", " ", text)  # stray special tokens, if any
    text = re.sub(r"\s+", " ", text).strip()
    # Spaces between Chinese characters carry no meaning; remove them.
    text = re.sub(r"(?<=[㐀-鿿　-〿＀-￯]) (?=[㐀-鿿])", "", text)
    return join_spelled_letters(text)


class FunASRNanoPlugin(LanguagePlugin):
    def __init__(
        self,
        plugin_id: str,
        languages: list[str],
        model: str,
        hub: str,
        device: str = "cpu",
        mixed: bool = False,
        priority: int = 0,
        force_language: bool = False,
    ) -> None:
        self.model = model
        self.hub = hub
        self.device = device
        self.force_language = force_language
        self.manifest = PluginManifest(
            id=plugin_id,
            languages=languages,
            model=model,
            license="see model card (FunAudioLLM)",
            mixed=mixed,
            priority=priority,
            needs_gpu=True,
            requires=["funasr", "torch"],
            description="Fun-ASR-Nano speech LLM, strong on Chinese-English code-switching",
        )

    def load(self) -> None:
        get_nano_model(self.model, self.hub, self.device)

    def _language(self, ctx: TurnContext) -> str | None:
        if not self.force_language or ctx.language is None:
            return None
        return PROMPT_LANGUAGE.get(ctx.language)

    def transcribe(self, audio: np.ndarray, sample_rate: int, ctx: TurnContext) -> PluginResult:
        import torch

        model = get_nano_model(self.model, self.hub, self.device)
        kwargs: dict[str, Any] = {
            "cache": {},
            "batch_size": 1,
            "itn": True,
            "max_length": max_new_tokens(len(audio) / sample_rate),
        }
        language = self._language(ctx)
        if language:
            kwargs["language"] = language
        result = model.generate(
            input=[torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))], **kwargs
        )
        text = clean_nano_text(" ".join(str(r.get("text", "")) for r in result or []))
        text, looped = collapse_loops(text)
        if self.manifest.mixed or ctx.language is None:
            out_language = "+".join(sorted(self.manifest.languages))
        else:
            out_language = ctx.language
        notes = ["loop"] if looped else []
        if "ja" not in self.manifest.languages and has_kana(text):
            notes.append("wrong_script")  # drifted into Japanese, e.g. on a short "嗯"
        return PluginResult(text=text, raw_confidence=None, language=out_language, notes=notes)
