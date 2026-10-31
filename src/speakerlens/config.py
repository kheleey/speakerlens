"""Settings, read from environment variables and a local .env file.

Put secrets (DEEPSEEK_API_KEY, HF_TOKEN) in `.env` at the project root. That file is
git-ignored. See `.env.example` for every option.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv

DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_DEFAULT_MODEL = "deepseek-flash"
DEEPSEEK_DEFAULT_REASONING_EFFORT = "none"
DEEPSEEK_REASONING_EFFORTS = ("none", "low", "high", "max")
DEEPSEEK_DEFAULT_MAX_TOKENS = 16_000
OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"
OLLAMA_DEFAULT_MODEL = "qwen3:8b"


def _mode(name: str) -> str:
    """An auto / on / off setting (1/0, true/false, yes/no also accepted)."""
    value = (_env(name, "auto") or "auto").lower()
    if value in ("1", "true", "yes", "on"):
        return "on"
    if value in ("0", "false", "no", "off"):
        return "off"
    if value == "auto":
        return "auto"
    raise ValueError(f"{name} must be auto, on or off, not {value!r}")


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


@dataclass
class LLMSettings:
    provider: str  # "deepseek", "ollama" or "none"
    base_url: str | None
    api_key: str | None
    model: str | None
    max_tokens: int | None = None
    # Provider-specific fields added to every request body (e.g. DeepSeek's reasoning_effort).
    extra_body: dict[str, Any] | None = None

    @property
    def enabled(self) -> bool:
        return self.provider != "none"

    @property
    def runs_locally(self) -> bool:
        return self.provider == "ollama"


@dataclass
class Settings:
    llm: LLMSettings
    hf_token: str | None
    device: str
    diarization_model: str
    embedding_model: str
    whisper_model: str
    whisper_fallback_model: str
    langid_model: str
    sensevoice_hub: str
    sensevoice_model: str
    nano_model: str
    asr_profile: str
    nano_fallback_ratio: float
    reassign_margin: float  # 0 = speaker check off
    extraction: str  # target speaker extraction for overlapped turns: "auto", "on", "off"
    tse_model: Path | None  # folder with the extraction model (avg_model.pt, config.yaml)
    cache_dir: Path
    calibration_dir: Path


def _llm_settings() -> LLMSettings:
    provider = (_env("LLM_PROVIDER", "auto") or "auto").lower()
    deepseek_key = _env("DEEPSEEK_API_KEY")

    if provider == "auto":
        provider = "deepseek" if deepseek_key else "none"

    if provider == "deepseek":
        if not deepseek_key:
            raise ValueError(
                "LLM_PROVIDER=deepseek but DEEPSEEK_API_KEY is empty. Add it to your .env file."
            )
        effort = (
            _env("DEEPSEEK_REASONING_EFFORT", DEEPSEEK_DEFAULT_REASONING_EFFORT) or "none"
        ).lower()
        if effort not in DEEPSEEK_REASONING_EFFORTS:
            raise ValueError(
                f"DEEPSEEK_REASONING_EFFORT={effort!r}; use one of "
                + ", ".join(DEEPSEEK_REASONING_EFFORTS)
            )
        return LLMSettings(
            provider="deepseek",
            base_url=_env("DEEPSEEK_BASE_URL", DEEPSEEK_DEFAULT_BASE_URL),
            api_key=deepseek_key,
            model=_env("DEEPSEEK_MODEL", DEEPSEEK_DEFAULT_MODEL),
            max_tokens=int(_env("LLM_MAX_TOKENS", str(DEEPSEEK_DEFAULT_MAX_TOKENS)) or 0) or None,
            extra_body={"reasoning_effort": effort},
        )
    if provider == "ollama":
        return LLMSettings(
            provider="ollama",
            base_url=_env("OLLAMA_BASE_URL", OLLAMA_DEFAULT_BASE_URL),
            api_key="ollama",  # Ollama ignores the key but the client needs one
            model=_env("OLLAMA_MODEL", OLLAMA_DEFAULT_MODEL),
            max_tokens=int(_env("LLM_MAX_TOKENS", "0") or 0) or None,
        )
    if provider == "none":
        return LLMSettings(provider="none", base_url=None, api_key=None, model=None)
    raise ValueError(f"Unknown LLM_PROVIDER={provider!r}. Use auto, deepseek, ollama or none.")


ASR_PROFILES = ("auto", "routing", "nano")


def _asr_profile() -> str:
    profile = (_env("ASR_PROFILE", "auto") or "auto").lower()
    if profile not in ASR_PROFILES:
        raise ValueError(f"ASR_PROFILE={profile!r}; use one of {', '.join(ASR_PROFILES)}")
    return profile


def load_settings(env_file: str | Path | None = None) -> Settings:
    """Load settings. Values already in the environment win over the .env file."""
    path = str(env_file) if env_file else find_dotenv(usecwd=True)
    if path:
        load_dotenv(path, override=False)

    hub = (_env("SENSEVOICE_HUB", "ms") or "ms").lower()
    default_sensevoice = "iic/SenseVoiceSmall" if hub == "ms" else "FunAudioLLM/SenseVoiceSmall"
    cache_dir = Path(_env("SPEAKERLENS_CACHE_DIR", "~/.cache/speakerlens") or "").expanduser()

    return Settings(
        llm=_llm_settings(),
        hf_token=_env("HF_TOKEN") or _env("HUGGINGFACE_TOKEN"),
        device=(_env("DEVICE", "auto") or "auto").lower(),
        diarization_model=_env("DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1")
        or "",
        embedding_model=_env("EMBEDDING_MODEL", "pyannote/wespeaker-voxceleb-resnet34-LM") or "",
        whisper_model=_env("WHISPER_MODEL", "large-v3-turbo") or "",
        whisper_fallback_model=_env("WHISPER_FALLBACK_MODEL", "large-v3") or "",
        langid_model=_env("LANGID_WHISPER_MODEL", "small") or "",
        sensevoice_hub=hub,
        sensevoice_model=_env("SENSEVOICE_MODEL", default_sensevoice) or "",
        nano_model=_env("FUNASR_NANO_MODEL", "FunAudioLLM/Fun-ASR-Nano-2512") or "",
        asr_profile=_asr_profile(),
        nano_fallback_ratio=float(_env("NANO_FALLBACK_RATIO", "0.6") or 0),
        reassign_margin=float(_env("SPEAKER_REASSIGN_MARGIN", "0.1") or 0),
        extraction=_mode("SPEAKER_EXTRACTION"),
        tse_model=Path(tse).expanduser() if (tse := _env("TSE_MODEL")) else None,
        cache_dir=cache_dir,
        calibration_dir=Path(
            _env("SPEAKERLENS_CALIBRATION_DIR", str(cache_dir / "calibration")) or ""
        ).expanduser(),
    )


def resolve_device(device: str) -> str:
    """Turn "auto" into "cuda" when an NVIDIA GPU is usable, otherwise "cpu"."""
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
