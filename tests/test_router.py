from speakerlens.router import RouterConfig, SpeakerLanguagePrior, blend, route


def test_confident_language_goes_to_specialist(registry):
    d = route({"en": 0.99, "zh": 0.01}, registry)
    assert [c.plugin_id for c in d.candidates] == ["en"]
    assert d.candidates[0].language == "en"


def test_mostly_english_goes_to_mixed_plugin_when_one_exists(registry):
    # 95% English in code-switched speech usually still has Chinese words in it.
    d = route({"en": 0.95, "zh": 0.05}, registry)
    assert [c.plugin_id for c in d.candidates] == ["mixed"]
    assert d.candidates[0].language is None


def test_two_likely_languages_use_mixed_plugin(registry):
    d = route({"zh": 0.55, "en": 0.45}, registry)
    assert [c.plugin_id for c in d.candidates] == ["mixed"]
    assert d.candidates[0].language is None


def test_unsure_without_mixed_plugin_compares_top_two(registry):
    d = route({"zh": 0.6, "ja": 0.4}, registry)
    assert [c.plugin_id for c in d.candidates] == ["zh", "fallback"]


def test_unknown_language_uses_fallback(registry):
    d = route({"fr": 1.0}, registry)
    assert [c.plugin_id for c in d.candidates] == ["fallback"]
    assert d.candidates[0].language == "fr"


def test_short_turn_leans_on_speaker_prior(registry):
    # Language ID says English, but this speaker almost always speaks Mandarin.
    prior = {"zh": 0.95, "en": 0.05}
    long_turn = route({"en": 0.9, "zh": 0.1}, registry, prior, duration=5.0)
    short_turn = route({"en": 0.9, "zh": 0.1}, registry, prior, duration=0.5)
    assert long_turn.scores["en"] > long_turn.scores["zh"]
    assert short_turn.scores["zh"] > short_turn.scores["en"]


def test_blend_normalizes():
    scores = blend({"en": 0.8, "zh": 0.2}, {"zh": 1.0}, 0.5)
    assert abs(sum(scores.values()) - 1.0) < 1e-9
    assert scores["zh"] > scores["en"]


def test_prior_needs_enough_speech():
    prior = SpeakerLanguagePrior(min_seconds=10)
    prior.update("A", "zh", 5)
    assert prior.get("A") is None
    prior.update("A", "en+zh", 10)
    p = prior.get("A")
    assert p is not None and abs(p["zh"] - 10 / 15) < 1e-9


def test_threshold_is_configurable(registry):
    config = RouterConfig(confident=0.7, mixed_confident=0.7)
    d = route({"en": 0.75, "zh": 0.25}, registry, config=config)
    assert [c.plugin_id for c in d.candidates] == ["en"]
