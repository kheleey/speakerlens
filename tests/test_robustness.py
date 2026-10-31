"""Failure handling: LLM errors, long monologues, routing edge cases, SenseVoice tags."""

from types import SimpleNamespace

import numpy as np
import openai
import pytest

from speakerlens.diarization.base import (
    DiarizationResult,
    DiarizedSegment,
    segments_to_turns,
    split_points,
)
from speakerlens.llm import (
    LLMClient,
    LLMError,
    LLMTruncated,
    LLMUnavailable,
    parse_json_content,
)
from speakerlens.pipeline import Pipeline, PipelineOptions
from speakerlens.plugins.sensevoice_plugin import clean_sensevoice_text
from speakerlens.router import route
from speakerlens.summarize import summarize
from speakerlens.translate import translate_transcript
from speakerlens.types import Speaker, Transcript, Turn

from .conftest import SR, FakeDiarizer, FakeLanguageID


def reply(content: str | None, finish_reason: str = "stop"):
    message = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish_reason, message=message)])


class StubCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_client(monkeypatch, outcomes, **kwargs) -> tuple[LLMClient, StubCompletions]:
    monkeypatch.setattr("speakerlens.llm.time.sleep", lambda s: None)
    client = LLMClient("http://localhost:1/v1", "key", "deepseek-flash", **kwargs)
    stub = StubCompletions(outcomes)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=stub))
    return client, stub


def api_error(cls, status: int):
    # Only the attributes the SDK's error classes read; avoids depending on its HTTP library.
    request = SimpleNamespace(method="POST", url="http://localhost:1")
    response = SimpleNamespace(request=request, status_code=status, headers={})
    return cls(message="nope", response=response, body=None)


def test_request_carries_deepseek_options(monkeypatch):
    client, stub = make_client(
        monkeypatch,
        [reply('{"ok": true}')],
        max_tokens=16_000,
        extra_body={"reasoning_effort": "none"},
    )
    assert client.chat_json("Reply in JSON.", "hi") == {"ok": True}
    sent = stub.calls[0]
    assert sent["model"] == "deepseek-flash"
    assert sent["response_format"] == {"type": "json_object"}
    assert sent["max_tokens"] == 16_000
    assert sent["extra_body"] == {"reasoning_effort": "none"}


def test_bad_key_fails_fast_without_retrying(monkeypatch):
    client, stub = make_client(monkeypatch, [api_error(openai.AuthenticationError, 401)] * 3)
    with pytest.raises(LLMError, match="API key"):
        client.chat_json("JSON", "hi")
    assert len(stub.calls) == 1


def test_unknown_model_fails_fast(monkeypatch):
    client, stub = make_client(monkeypatch, [api_error(openai.NotFoundError, 404)] * 3)
    with pytest.raises(LLMError, match="deepseek-flash"):
        client.chat_json("JSON", "hi")
    assert len(stub.calls) == 1


def test_empty_balance_fails_fast(monkeypatch):
    client, stub = make_client(monkeypatch, [api_error(openai.APIStatusError, 402)] * 3)
    with pytest.raises(LLMUnavailable, match="balance"):
        client.chat_json("JSON", "hi")
    assert len(stub.calls) == 1


def test_empty_and_broken_replies_are_retried(monkeypatch):
    client, stub = make_client(monkeypatch, [reply(""), reply("not json"), reply('{"a": 1}')])
    assert client.chat_json("JSON", "hi") == {"a": 1}
    assert len(stub.calls) == 3


def test_cut_off_reply_is_reported_not_retried(monkeypatch):
    client, stub = make_client(monkeypatch, [reply('{"a": ', finish_reason="length")])
    with pytest.raises(LLMTruncated):
        client.chat_json("JSON", "hi")
    assert len(stub.calls) == 1


def test_think_blocks_are_ignored():
    assert parse_json_content('<think>use {"x": 0}?</think>\n{"x": 1}') == {"x": 1}


def transcript_with(n: int) -> Transcript:
    turns = [
        Turn(id=f"T{i:04d}", speaker="A", start=i, end=i + 1, text=f"句子{i}", language="zh")
        for i in range(1, n + 1)
    ]
    return Transcript(
        source="x.wav", duration=n + 1, languages=["zh"], speakers=[Speaker(id="A")], turns=turns
    )


class SplittingLLM:
    """Fails on any batch with more than `limit` items, like a reply hitting a token limit."""

    def __init__(self, limit: int, bad_ids: frozenset[str] = frozenset()) -> None:
        self.limit = limit
        self.bad_ids = bad_ids
        self.calls = 0

    def chat_json(self, system, user, temperature=0.2):
        import json

        self.calls += 1
        items = json.loads(user)["utterances"]
        if len(items) > self.limit or any(i["id"] in self.bad_ids for i in items):
            raise LLMTruncated("too long")
        return {"translations": [{"id": i["id"], "text": f"EN {i['id']}"} for i in items]}


def test_translation_splits_batches_that_fail():
    t = transcript_with(20)
    failed = translate_transcript(t, "en", SplittingLLM(limit=6))
    assert failed == 0
    assert all(turn.translation == f"EN {turn.id}" for turn in t.turns)


def test_one_untranslatable_turn_does_not_sink_the_rest():
    t = transcript_with(8)
    failed = translate_transcript(t, "en", SplittingLLM(limit=50, bad_ids=frozenset({"T0003"})))
    assert failed == 1
    assert t.turns[2].translation is None
    assert sum(turn.translation is not None for turn in t.turns) == 7


class FailingLLM:
    def chat_json(self, system, user, temperature=0.2):
        raise LLMError("service down")


def test_summary_failure_returns_empty_summary_not_crash():
    summary = summarize(transcript_with(3), FailingLLM())
    assert summary.speakers == []


def test_strict_summary_raises_instead_of_leaving_parts_out():
    with pytest.raises(LLMError):
        summarize(transcript_with(3), FailingLLM(), strict=True)


class UnavailableLLM:
    def __init__(self) -> None:
        self.calls = 0

    def chat_json(self, system, user, temperature=0.2):
        self.calls += 1
        raise LLMUnavailable("no balance")


def test_unavailable_llm_stops_summaries_and_translation_after_one_call():
    llm = UnavailableLLM()
    with pytest.raises(LLMUnavailable):
        summarize(transcript_with(40), llm)
    assert llm.calls == 1
    llm = UnavailableLLM()
    with pytest.raises(LLMUnavailable):
        translate_transcript(transcript_with(40), "en", llm)
    assert llm.calls == 1


def test_pipeline_keeps_transcript_when_llm_is_down(registry):
    pipeline = Pipeline(
        registry=registry,
        diarizer=FakeDiarizer(),
        language_id=FakeLanguageID([{"en": 1.0}]),
        llm=FailingLLM(),
    )
    audio = np.zeros(12 * SR, dtype=np.float32)
    transcript = pipeline.run("x.wav", PipelineOptions(translate_to="zh"), audio=audio)
    assert len(transcript.turns) == 3 and all(t.text for t in transcript.turns)
    assert transcript.summary is not None and transcript.summary.speakers == []


def test_monologue_is_split_at_quiet_moments():
    rng = np.random.default_rng(1)
    audio = (rng.standard_normal(SR * 70) * 0.1).astype(np.float32)
    audio[int(26.0 * SR) : int(26.3 * SR)] = 0  # a pause at 26 s
    audio[int(51.0 * SR) : int(51.3 * SR)] = 0  # and at 51 s
    pieces = split_points(audio, SR, 0.0, 70.0, max_len=30)
    assert [round(b) for _, b in pieces[:-1]] == [26, 51]
    assert all(b - a <= 30 for a, b in pieces)
    assert pieces[0][0] == 0.0 and pieces[-1][1] == 70.0

    result = DiarizationResult(segments=[DiarizedSegment(speaker="A", start=0.0, end=70.0)])
    turns = segments_to_turns(result, audio=audio, sample_rate=SR)
    assert len(turns) == 3 and [t.id for t in turns] == ["T0001", "T0002", "T0003"]


def test_short_turns_are_not_split():
    audio = np.zeros(SR * 20, dtype=np.float32)
    assert split_points(audio, SR, 2.0, 20.0) == [(2.0, 20.0)]


def test_leader_is_used_when_nothing_else_is_plausible(registry):
    d = route({"en": 0.7, "fr": 0.15, "de": 0.15}, registry)  # no en+fr mixed plugin
    assert [c.plugin_id for c in d.candidates] == ["en"]


def test_two_languages_on_the_same_fallback_let_the_model_decide(registry):
    d = route({"ja": 0.55, "ko": 0.45}, registry)
    assert len(d.candidates) == 1
    assert d.candidates[0].plugin_id == "fallback" and d.candidates[0].language is None


def test_sensevoice_tags_are_removed_without_emoji():
    raw = (
        "<|zh|><|HAPPY|><|Speech|><|withitn|>今天天气很好。"
        "<|en|><|NEUTRAL|><|Applause|><|withitn|>Great!"
    )
    text, language = clean_sensevoice_text(raw)
    assert text == "今天天气很好。 Great!"
    assert language == "en+zh"


def test_sensevoice_single_language_tag():
    text, language = clean_sensevoice_text("<|zh|><|NEUTRAL|><|Speech|><|woitn|>我们开会")
    assert (text, language) == ("我们开会", "zh")
