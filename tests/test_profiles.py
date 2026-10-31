"""ASR profiles and the Fun-ASR-Nano plugin, with fake funasr/torch modules (no downloads)."""

import sys
import types

import numpy as np
import pytest

from speakerlens.config import load_settings
from speakerlens.plugins import funasr_nano_plugin as nano_mod
from speakerlens.plugins.base import LanguagePlugin, TurnContext
from speakerlens.plugins.funasr_nano_plugin import FunASRNanoPlugin, clean_nano_text
from speakerlens.plugins.registry import default_registry


@pytest.fixture
def env(monkeypatch, tmp_path):
    for key in ("ASR_PROFILE", "DEEPSEEK_API_KEY", "LLM_PROVIDER", "NANO_FALLBACK_RATIO"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.chdir(tmp_path)
    # Pretend every plugin's libraries are installed.
    monkeypatch.setattr(LanguagePlugin, "is_available", lambda self: True)


def registry_for(profile, monkeypatch):
    monkeypatch.setenv("ASR_PROFILE", profile)
    return default_registry(load_settings())


def test_profiles_pick_the_right_plugins(env, monkeypatch):
    def ids(profile):
        return {p.id for p in registry_for(profile, monkeypatch).all()}

    assert ids("routing") == {"whisper-en", "sensevoice-zh", "sensevoice-mixed", "whisper-fallback"}
    assert ids("nano") == {"nano-zh", "nano-en", "nano-mixed", "whisper-fallback"}


def test_routing_in_each_profile(env, monkeypatch):
    reg = registry_for("nano", monkeypatch)
    assert reg.specialist_for("zh").id == "nano-zh"
    assert reg.specialist_for("en").id == "nano-en"
    assert reg.mixed_for({"zh", "en"}).id == "nano-mixed"
    assert reg.for_language("fr").id == "whisper-fallback"
    reg = registry_for("routing", monkeypatch)
    assert reg.specialist_for("zh").id == "sensevoice-zh"
    assert reg.specialist_for("en").id == "whisper-en"
    assert reg.mixed_for({"zh", "en"}).id == "sensevoice-mixed"


def test_mixed_sensevoice_uses_mandarin_mode(env, monkeypatch):
    plugin = registry_for("routing", monkeypatch).get("sensevoice-mixed")
    assert plugin.language_mode == "zh"


def test_unknown_profile_is_an_error(env, monkeypatch):
    monkeypatch.setenv("ASR_PROFILE", "fastest")
    with pytest.raises(ValueError, match="ASR_PROFILE"):
        load_settings()


class FakeTensor:
    def __init__(self, array):
        self.array = array


class FakeAutoModel:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.calls = []

    def generate(self, input, **kwargs):
        self.calls.append((input, kwargs))
        return [{"key": "k", "text": "我们 今天 讨论 the project plan。"}]


@pytest.fixture
def fake_funasr(monkeypatch):
    funasr = types.ModuleType("funasr")
    funasr.AutoModel = FakeAutoModel
    torch = types.ModuleType("torch")
    torch.from_numpy = FakeTensor
    monkeypatch.setitem(sys.modules, "funasr", funasr)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(nano_mod, "_models", {})


def test_nano_plugin_passes_a_tensor_and_cleans_text(fake_funasr):
    plugin = FunASRNanoPlugin("nano-mixed", ["zh", "en"], "m", "ms", "cpu", mixed=True)
    res = plugin.transcribe(
        np.zeros(16000, dtype=np.float32), 16000, TurnContext("T1", "A", 0, 1, None)
    )
    assert res.text == "我们今天讨论 the project plan。"
    assert res.language == "en+zh"
    assert res.raw_confidence is None
    model = nano_mod._models[("m", "ms", "cpu")]
    sent_input, sent_kwargs = model.calls[0]
    assert isinstance(sent_input, list) and isinstance(sent_input[0], FakeTensor)
    assert "language" not in sent_kwargs  # auto by default: forcing can make it translate
    assert sent_kwargs["itn"] is True and model.init_kwargs["hub"] == "ms"
    assert sent_kwargs["max_length"] == 32  # 1 s of audio: 20 + 12 tokens, not 512


def test_nano_can_force_language(fake_funasr):
    plugin = FunASRNanoPlugin("nano-zh", ["zh"], "m", "ms", "cpu", force_language=True)
    plugin.transcribe(np.zeros(1600, dtype=np.float32), 16000, TurnContext("T1", "A", 0, 1, "zh"))
    assert nano_mod._models[("m", "ms", "cpu")].calls[0][1]["language"] == "中文"


def test_clean_nano_text():
    assert clean_nano_text("  你 好 <|en|> world ") == "你好 world"
    assert clean_nano_text("to have a job in a i") == "to have a job in AI"
    assert clean_nano_text("虽然是在U S T哦") == "虽然是在UST哦"
    assert clean_nano_text("Am I a good fit") == "Am I a good fit"  # mixed case: real words
    assert clean_nano_text("I I love Li An") == "I I love Li An"  # a stutter, not "II"


def test_nano_reports_and_collapses_loops(fake_funasr, monkeypatch):
    looping = "And I think if, " + ", ".join(["if I was just", "ifI was just"] * 30)
    monkeypatch.setattr(FakeAutoModel, "generate", lambda self, input, **kw: [{"text": looping}])
    plugin = FunASRNanoPlugin("nano-mixed", ["zh", "en"], "m", "ms", "cpu", mixed=True)
    res = plugin.transcribe(
        np.zeros(32000, dtype=np.float32), 16000, TurnContext("T1", "A", 0, 2, None)
    )
    assert res.notes == ["loop"]
    assert res.text.count("was just") <= 2


from speakerlens.plugins.base import PluginManifest, PluginResult  # noqa: E402
from speakerlens.plugins.fallback import FallbackPlugin  # noqa: E402


class Fixed(LanguagePlugin):
    def __init__(self, plugin_id, text, notes=()):
        self.text = text
        self.notes = list(notes)
        self.calls = 0
        self.manifest = PluginManifest(
            id=plugin_id, languages=["zh", "en"], model=plugin_id, license="MIT", mixed=True
        )

    def transcribe(self, audio, sample_rate, ctx):
        self.calls += 1
        return PluginResult(text=self.text, language="en+zh", notes=self.notes)


def ctx():
    return TurnContext("T1", "A", 0, 3, None)


def test_fallback_replaces_a_truncated_transcript():
    primary = Fixed("nano", "足球比较难一点，因为刚才你说。")
    backup = Fixed(
        "sensevoice", "足球比较难一点因为刚才你说 the football pitch is a lot bigger 所以会更累"
    )
    res = FallbackPlugin(primary, backup, ratio=0.6).transcribe(np.zeros(16000 * 3), 16000, ctx())
    assert res.used == "sensevoice" and "football" in res.text


def test_fallback_keeps_a_complete_transcript():
    primary = Fixed("nano", "我们今天讨论 the project plan")
    backup = Fixed("sensevoice", "我们今天讨论the project plan吧")
    res = FallbackPlugin(primary, backup, ratio=0.6).transcribe(np.zeros(16000 * 3), 16000, ctx())
    assert res.used is None and res.text == "我们今天讨论 the project plan"


def test_fallback_skips_short_turns():
    primary, backup = Fixed("nano", "嗯"), Fixed("sensevoice", "嗯 对 对 对 对")
    res = FallbackPlugin(primary, backup).transcribe(np.zeros(16000), 16000, ctx())
    assert res.text == "嗯" and backup.calls == 0


def test_fallback_replaces_a_loop_even_on_short_turns():
    primary = Fixed("nano", "And I think if, if I was just", notes=["loop"])
    backup = Fixed("sensevoice", "And I think if 如果就是没有")
    res = FallbackPlugin(primary, backup).transcribe(np.zeros(16000), 16000, ctx())
    assert res.used == "sensevoice" and res.notes == ["loop"]


def test_fallback_replaces_a_runaway_transcript():
    primary = Fixed("nano", " ".join(["we should go"] * 12))
    backup = Fixed("sensevoice", "we should go")
    res = FallbackPlugin(primary, backup).transcribe(np.zeros(16000 * 3), 16000, ctx())
    assert res.used == "sensevoice"


def test_auto_profile_and_fallback_wiring(env, monkeypatch):
    monkeypatch.setenv("ASR_PROFILE", "auto")
    monkeypatch.setenv("DEVICE", "cuda")
    reg = default_registry(load_settings())
    plugin = reg.get("nano-mixed")
    assert isinstance(plugin, FallbackPlugin) and plugin.ratio == 0.6
    monkeypatch.setenv("DEVICE", "cpu")
    assert "sensevoice-mixed" in {p.id for p in default_registry(load_settings()).all()}
    monkeypatch.setenv("DEVICE", "cuda")
    monkeypatch.setenv("NANO_FALLBACK_RATIO", "0")
    assert isinstance(default_registry(load_settings()).get("nano-mixed"), FunASRNanoPlugin)


def test_fallback_replaces_japanese_even_on_short_turns():
    primary = Fixed("nano", "うん、はいきな。", notes=["wrong_script"])
    backup = Fixed("sensevoice", "嗯，hiking啊")
    res = FallbackPlugin(primary, backup).transcribe(np.zeros(16000), 16000, ctx())
    assert res.used == "sensevoice" and res.text == "嗯，hiking啊"
    # Even an empty backup beats Japanese nobody spoke.
    res = FallbackPlugin(primary, Fixed("sensevoice", "")).transcribe(np.zeros(16000), 16000, ctx())
    assert res.text == ""


def test_nano_flags_japanese_unless_japanese_was_chosen(fake_funasr, monkeypatch):
    monkeypatch.setattr(
        FakeAutoModel, "generate", lambda self, input, **kw: [{"key": "k", "text": "ええ。"}]
    )
    turn = TurnContext("T1", "A", 0, 1, None)
    audio = np.zeros(16000, dtype=np.float32)
    zh_en = FunASRNanoPlugin("nano-mixed", ["zh", "en"], "m", "ms", "cpu", mixed=True)
    assert zh_en.transcribe(audio, 16000, turn).notes == ["wrong_script"]
    ja = FunASRNanoPlugin("nano-ja", ["ja"], "m", "ms", "cpu")
    assert ja.transcribe(audio, 16000, turn).notes == []


def test_shouting_case_is_lowered():
    from speakerlens.textfix import has_kana, tame_caps

    assert tame_caps("大家 OK, SO WE NEED A PLAN TODAY。") == "大家 OK, so we need a plan today。"
    assert (
        tame_caps("WHO'S BOOKING IT? I'M NOT SURE, ASK HR")
        == "who's booking it? I'm not sure, ask HR"
    )
    assert tame_caps("the GPU is busy, ask HR") == "the GPU is busy, ask HR"  # one acronym: kept
    assert tame_caps("我们的STAND UP改成IN PERSON") == "我们的stand up改成in person"
    assert has_kana("うんうん。") and not has_kana("嗯嗯。 OK")
