import json

from speakerlens.calibration import CalibrationStore, Calibrator
from speakerlens.export import to_markdown, to_srt, to_stm, write_outputs
from speakerlens.llm import parse_json_content
from speakerlens.pipeline import Pipeline, PipelineOptions

from .conftest import FakeDiarizer, FakeLanguageID, FakeLLM


def summary_response():
    return {
        "overview": "A short chat.",
        "speakers": [
            {"speaker": "SPEAKER_00", "main_points": [{"text": "Greets", "turn_ids": ["T0001"]}]},
            {"speaker": "SPEAKER_01", "main_points": [{"text": "Replies", "turn_ids": ["T0002"]}]},
        ],
    }


def test_end_to_end_with_fakes(registry, silence, tmp_path):
    lid = FakeLanguageID(
        [{"en": 0.99, "zh": 0.01}, {"zh": 0.99, "en": 0.01}, {"zh": 0.5, "en": 0.5}]
    )
    llm = FakeLLM(
        [
            {
                "translations": [
                    {"id": "T0002", "text": "hello (translated)"},
                    {"id": "T0003", "text": "mixed (translated)"},
                ]
            },
            summary_response(),
        ]
    )
    pipeline = Pipeline(registry=registry, diarizer=FakeDiarizer(), language_id=lid, llm=llm)
    options = PipelineOptions(languages=["zh", "en"], translate_to="en")
    transcript = pipeline.run("meeting.wav", options, audio=silence)

    assert [t.plugin for t in transcript.turns] == ["en", "zh", "mixed"]
    assert [t.language for t in transcript.turns] == ["en", "zh", "en+zh"]
    assert transcript.turns[1].overlap is True
    assert transcript.turns[0].translation == transcript.turns[0].text  # already English
    assert transcript.turns[1].translation == "hello (translated)"
    # Word times are moved from turn-relative to recording-relative.
    assert transcript.turns[1].words[0].start == transcript.turns[1].start
    assert transcript.summary is not None
    assert [s.speaker for s in transcript.summary.speakers] == ["SPEAKER_00", "SPEAKER_01"]

    paths = write_outputs(transcript, tmp_path, "meeting", ["md", "json", "srt", "stm"])
    names = sorted(p.name for p in paths)
    assert names == ["meeting.en.srt", "meeting.json", "meeting.md", "meeting.srt", "meeting.stm"]
    assert json.loads((tmp_path / "meeting.json").read_text())["turns"][0]["id"] == "T0001"


def test_calibration_decides_between_two_plugins(registry, silence, tmp_path):
    # zh vs ja is unsure and there's no zh+ja mixed plugin, so two plugins run.
    lid = FakeLanguageID([{"zh": 0.6, "ja": 0.4}])
    # Raw scores: zh plugin 0.6, fallback 0.5. Calibration says the fallback is more
    # trustworthy than the zh plugin at those raw values, so the fallback should win.
    Calibrator(plugin_id="zh", method="isotonic", xs=[0.0, 1.0], ys=[0.0, 0.5]).save(tmp_path)
    Calibrator(plugin_id="fallback", method="isotonic", xs=[0.0, 1.0], ys=[0.0, 0.9]).save(tmp_path)
    pipeline = Pipeline(
        registry=registry,
        diarizer=FakeDiarizer(),
        language_id=lid,
        calibration=CalibrationStore(tmp_path),
    )
    transcript = pipeline.run("x.wav", PipelineOptions(summarize=False), audio=silence)
    assert all(t.plugin == "fallback" for t in transcript.turns)
    assert transcript.turns[0].language == "ja"
    assert pipeline.last_outcomes[0].calibrated == {"zh": 0.3, "fallback": 0.45}


def test_no_llm_means_no_translation_or_summary(registry, silence):
    lid = FakeLanguageID([{"en": 1.0}])
    pipeline = Pipeline(registry=registry, diarizer=FakeDiarizer(), language_id=lid, llm=None)
    transcript = pipeline.run("x.wav", PipelineOptions(translate_to="zh"), audio=silence)
    assert transcript.summary is None
    assert all(t.translation is None for t in transcript.turns)


def test_exports(registry, silence):
    lid = FakeLanguageID([{"en": 1.0}])
    pipeline = Pipeline(registry=registry, diarizer=FakeDiarizer(), language_id=lid)
    transcript = pipeline.run("x.wav", PipelineOptions(summarize=False), audio=silence)
    transcript.speakers[0].name = "Alice"
    srt = to_srt(transcript)
    assert srt.startswith("1\n00:00:00,000 --> 00:00:04,000\nAlice: hello T0001")
    md = to_markdown(transcript)
    assert "**Alice** [00:00:00]" in md and "_(overlap)_" in md
    assert to_stm(transcript, "x").splitlines()[0] == "x 1 SPEAKER_00 0.000 4.000 hello T0001"


def test_parse_json_content_handles_fences():
    assert parse_json_content('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_content('Sure! {"a": 2} Hope this helps.') == {"a": 2}
