import pytest

from speakerlens.config import load_settings

KEYS = [
    "LLM_PROVIDER",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_MODEL",
    "DEEPSEEK_REASONING_EFFORT",
    "LLM_MAX_TOKENS",
    "OLLAMA_MODEL",
    "SENSEVOICE_HUB",
    "SPEAKER_EXTRACTION",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)  # so no real .env is picked up


def test_auto_uses_deepseek_when_key_present(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("DEEPSEEK_API_KEY=sk-test-1234\n")
    s = load_settings(env)
    assert s.llm.provider == "deepseek"
    assert s.llm.base_url == "https://api.deepseek.com"
    # deepseek-chat was retired on 2026-07-24; deepseek-flash replaced it.
    assert s.llm.model == "deepseek-flash"
    assert s.llm.api_key == "sk-test-1234"
    assert s.llm.extra_body == {"reasoning_effort": "none"}  # thinking off by default
    assert s.llm.max_tokens == 16_000


def test_auto_without_key_disables_llm(tmp_path):
    s = load_settings(tmp_path / "missing.env")
    assert s.llm.provider == "none" and not s.llm.enabled


def test_ollama_runs_locally(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    s = load_settings()
    assert s.llm.runs_locally and s.llm.base_url.endswith("/v1")


def test_deepseek_without_key_is_an_error(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        load_settings()


def test_sensevoice_hub_picks_model_id(monkeypatch):
    monkeypatch.setenv("SENSEVOICE_HUB", "hf")
    assert load_settings().sensevoice_model == "FunAudioLLM/SenseVoiceSmall"


def test_reasoning_effort_is_validated(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
    monkeypatch.setenv("DEEPSEEK_REASONING_EFFORT", "LOW")
    assert load_settings().llm.extra_body == {"reasoning_effort": "low"}
    monkeypatch.setenv("DEEPSEEK_REASONING_EFFORT", "medium-ish")
    with pytest.raises(ValueError, match="DEEPSEEK_REASONING_EFFORT"):
        load_settings()


def test_ollama_sends_no_deepseek_options(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    s = load_settings()
    assert s.llm.extra_body is None and s.llm.max_tokens is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, "auto"), ("auto", "auto"), ("1", "on"), ("on", "on"), ("0", "off"), ("False", "off")],
)
def test_extraction_setting_values(monkeypatch, tmp_path, value, expected):
    monkeypatch.delenv("SPEAKER_EXTRACTION", raising=False)
    if value is not None:
        monkeypatch.setenv("SPEAKER_EXTRACTION", value)
    assert load_settings(tmp_path / "missing.env").extraction == expected


def test_bad_extraction_setting_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setenv("SPEAKER_EXTRACTION", "maybe")
    with pytest.raises(ValueError, match="auto, on or off"):
        load_settings(tmp_path / "missing.env")
