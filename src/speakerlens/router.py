"""Decides which plugin(s) transcribe each turn.

Rules, in order:
1. Blend language-ID scores with the speaker's usual languages (their "prior").
2. A mixed plugin covers the top two languages and the top one is below
   `mixed_confident` (0.97) -> the mixed plugin. In code-switched speech, "mostly English"
   turns usually still contain Chinese, and a monolingual model drops it.
3. One language clearly wins -> that language's specialist plugin.
4. No other language is plausible (second < mixed_min) -> the leader's plugin.
5. Two languages are both plausible and a mixed plugin covers them -> mixed plugin.
6. Still unsure -> run the plugins for the top two languages; the pipeline keeps the
   result with the higher calibrated confidence.
7. No specialist for a language -> the fallback plugin.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from speakerlens.plugins.registry import PluginRegistry


class NoPluginError(RuntimeError):
    pass


@dataclass
class RouterConfig:
    confident: float = 0.8  # top language at least this likely -> no second opinion
    # With a mixed plugin available, a monolingual model is only used above this.
    mixed_confident: float = 0.97
    mixed_min: float = 0.2  # second language at least this likely -> consider a mixed plugin
    prior_weight: float = 0.3  # how much the speaker's usual languages count
    short_turn_seconds: float = 1.0  # language ID is unreliable on very short turns...
    short_turn_prior_weight: float = 0.7  # ...so lean on the speaker's prior instead
    min_prior_seconds: float = 10.0  # prior only counts after this much of a speaker's speech


@dataclass
class Candidate:
    plugin_id: str
    language: str | None  # None means "let the plugin decide" (mixed plugins)


@dataclass
class RouteDecision:
    candidates: list[Candidate]
    scores: dict[str, float]
    reason: str


class SpeakerLanguagePrior:
    """Tracks how much each speaker has spoken in each language so far."""

    def __init__(self, min_seconds: float = 10.0) -> None:
        self.min_seconds = min_seconds
        self._seconds: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def update(self, speaker: str, language: str | None, seconds: float) -> None:
        if not language:
            return
        parts = language.split("+")  # mixed turns count for each language
        for part in parts:
            self._seconds[speaker][part] += seconds / len(parts)

    def get(self, speaker: str) -> dict[str, float] | None:
        counts = self._seconds.get(speaker)
        if not counts:
            return None
        total = sum(counts.values())
        if total < self.min_seconds:
            return None
        return {lang: s / total for lang, s in counts.items()}


def blend(lid: dict[str, float], prior: dict[str, float] | None, weight: float) -> dict[str, float]:
    if not prior:
        scores = dict(lid)
    else:
        langs = set(lid) | set(prior)
        scores = {
            lang: (1 - weight) * lid.get(lang, 0.0) + weight * prior.get(lang, 0.0)
            for lang in langs
        }
    total = sum(scores.values())
    return {k: v / total for k, v in scores.items()} if total > 0 else scores


def route(
    lid: dict[str, float],
    registry: PluginRegistry,
    prior: dict[str, float] | None = None,
    duration: float | None = None,
    config: RouterConfig | None = None,
) -> RouteDecision:
    config = config or RouterConfig()
    short = duration is not None and duration < config.short_turn_seconds
    weight = config.short_turn_prior_weight if short else config.prior_weight
    scores = blend(lid, prior, weight)
    if not scores:
        raise NoPluginError("Language ID returned no scores.")

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_lang, top_p = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None

    def single(lang: str) -> Candidate:
        plugin = registry.for_language(lang)
        if plugin is None:
            raise NoPluginError(f"No plugin for language {lang!r} and no fallback installed.")
        return Candidate(plugin.id, lang)

    if second is not None and top_p < config.mixed_confident:
        mixed_first = registry.mixed_for({top_lang, second[0]})
        if mixed_first is not None:
            return RouteDecision(
                [Candidate(mixed_first.id, None)],
                scores,
                f"possibly code-switched: {top_lang}={top_p:.2f} < {config.mixed_confident}",
            )

    if top_p >= config.confident or second is None:
        return RouteDecision([single(top_lang)], scores, f"confident: {top_lang}={top_p:.2f}")

    second_lang, second_p = second
    if second_p < config.mixed_min:
        # Not confident, but nothing else is plausible either: go with the leader.
        return RouteDecision([single(top_lang)], scores, f"leading: {top_lang}={top_p:.2f}")

    mixed = registry.mixed_for({top_lang, second_lang})
    if mixed is not None:
        return RouteDecision(
            [Candidate(mixed.id, None)],
            scores,
            f"mixed: {top_lang}={top_p:.2f}, {second_lang}={second_p:.2f}",
        )

    first, other = single(top_lang), single(second_lang)
    if first.plugin_id == other.plugin_id:
        # Both languages land on the same (fallback) plugin: let the model pick the language.
        return RouteDecision(
            [Candidate(first.plugin_id, None)],
            scores,
            f"unsure between {top_lang} and {second_lang}: {first.plugin_id} detects",
        )
    return RouteDecision(
        [first, other], scores, f"unsure: comparing {first.plugin_id}, {other.plugin_id}"
    )


@dataclass
class TurnOutcome:
    """What the pipeline records for one routed turn (kept for debugging)."""

    turn_id: str
    decision: RouteDecision
    chosen: str
    calibrated: dict[str, float | None] = field(default_factory=dict)
