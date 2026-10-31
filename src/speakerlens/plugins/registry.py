"""Finds plugins: built-in ones plus any installed package that registers under the
`speakerlens.plugins` entry-point group."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

from speakerlens.plugins.base import ANY_LANGUAGE, LanguagePlugin

if TYPE_CHECKING:
    from speakerlens.config import Settings

log = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "speakerlens.plugins"


class PluginRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, LanguagePlugin] = {}
        self.unavailable: dict[str, str] = {}  # plugin id -> reason

    def register(self, plugin: LanguagePlugin, check_available: bool = True) -> None:
        if check_available and not plugin.is_available():
            missing = ", ".join(plugin.manifest.requires)
            self.unavailable[plugin.id] = f"missing Python package(s): {missing}"
            return
        if plugin.id in self._plugins:
            raise ValueError(f"Duplicate plugin id: {plugin.id}")
        self._plugins[plugin.id] = plugin

    def register_all(self, plugins: Iterable[LanguagePlugin]) -> None:
        for plugin in plugins:
            self.register(plugin)

    def get(self, plugin_id: str) -> LanguagePlugin:
        return self._plugins[plugin_id]

    def all(self) -> list[LanguagePlugin]:
        return list(self._plugins.values())

    def specialist_for(self, language: str) -> LanguagePlugin | None:
        """Best single-language (non-mixed, non-fallback) plugin for `language`."""
        candidates = [
            p
            for p in self._plugins.values()
            if not p.manifest.mixed and not p.manifest.fallback and language in p.manifest.languages
        ]
        return max(candidates, key=lambda p: p.manifest.priority, default=None)

    def mixed_for(self, languages: set[str]) -> LanguagePlugin | None:
        """Best mixed-speech plugin that covers every language in `languages`."""
        candidates = [
            p
            for p in self._plugins.values()
            if p.manifest.mixed and languages <= set(p.manifest.languages)
        ]
        return max(candidates, key=lambda p: p.manifest.priority, default=None)

    def fallback(self) -> LanguagePlugin | None:
        candidates = [
            p
            for p in self._plugins.values()
            if p.manifest.fallback or ANY_LANGUAGE in p.manifest.languages
        ]
        return max(candidates, key=lambda p: p.manifest.priority, default=None)

    def for_language(self, language: str) -> LanguagePlugin | None:
        return self.specialist_for(language) or self.fallback()

    def load_entry_points(self, settings: object) -> None:
        """Load third-party plugins. Each entry point is a factory:
        `def create_plugins(settings) -> LanguagePlugin | list[LanguagePlugin]`."""
        for ep in entry_points(group=ENTRY_POINT_GROUP):
            try:
                created = ep.load()(settings)
            except Exception as exc:  # a broken plugin must not break the app
                log.warning("Could not load plugin %s: %s", ep.name, exc)
                self.unavailable[ep.name] = f"failed to load: {exc}"
                continue
            self.register_all(created if isinstance(created, list) else [created])


def resolve_profile(profile: str, device: str) -> str:
    """ "auto" picks Fun-ASR-Nano when a GPU is available, else routing."""
    if profile == "auto":
        return "nano" if device == "cuda" else "routing"
    return profile


def default_registry(settings: Settings) -> PluginRegistry:
    """Built-in plugins for the chosen ASR profile, plus installed third-party ones.

    Profiles (ASR_PROFILE in .env, or --asr-profile):
      auto     nano on a CUDA GPU, routing otherwise (default)
      routing  Whisper for English, SenseVoice for Mandarin and mixed turns
      nano     Fun-ASR-Nano for every Mandarin, English and mixed turn
    Whisper large-v3 is the fallback for any other language in every profile.
    """
    from speakerlens.config import resolve_device
    from speakerlens.plugins.fallback import FallbackPlugin
    from speakerlens.plugins.funasr_nano_plugin import FunASRNanoPlugin
    from speakerlens.plugins.sensevoice_plugin import SenseVoicePlugin
    from speakerlens.plugins.whisper_plugin import WhisperPlugin

    device = resolve_device(settings.device)
    profile = resolve_profile(settings.asr_profile, device)

    def sensevoice(plugin_id: str, languages: list[str], mixed: bool = False) -> LanguagePlugin:
        # Mandarin mode also for mixed turns: "auto" mode tends to decide an English-heavy
        # turn is all English and garble its Chinese words.
        return SenseVoicePlugin(
            plugin_id,
            languages,
            settings.sensevoice_model,
            settings.sensevoice_hub,
            device,
            language_mode="zh",
            mixed=mixed,
            priority=10,
        )

    def nano(plugin_id: str, languages: list[str], mixed: bool = False) -> LanguagePlugin:
        primary = FunASRNanoPlugin(
            plugin_id,
            languages,
            settings.nano_model,
            settings.sensevoice_hub,
            device,
            mixed=mixed,
            priority=20,
        )
        if settings.nano_fallback_ratio <= 0:
            return primary
        backup = sensevoice("sensevoice", languages, mixed=mixed)
        return FallbackPlugin(primary, backup, ratio=settings.nano_fallback_ratio)

    plugins: list[LanguagePlugin]
    if profile == "nano":
        plugins = [
            nano("nano-zh", ["zh"]),
            nano("nano-en", ["en"]),
            nano("nano-mixed", ["zh", "en"], True),
        ]
    else:
        plugins = [
            WhisperPlugin("whisper-en", ["en"], settings.whisper_model, device, priority=10),
            sensevoice("sensevoice-zh", ["zh"]),
            sensevoice("sensevoice-mixed", ["zh", "en"], mixed=True),
        ]
    plugins.append(
        WhisperPlugin(
            "whisper-fallback",
            [ANY_LANGUAGE],
            settings.whisper_fallback_model,
            device,
            fallback=True,
        )
    )
    registry = PluginRegistry()
    registry.register_all(plugins)
    registry.load_entry_points(settings)
    return registry
