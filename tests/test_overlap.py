"""Overlapped speech: turns, hidden speech, and dropping hidden speech that only repeats the
main speaker."""

import numpy as np
import pytest

from speakerlens.diarization.base import (
    DiarizationResult,
    DiarizedSegment,
    hidden_turns,
    segments_to_turns,
)
from speakerlens.pipeline import PipelineOptions, is_echo
from speakerlens.types import Turn

from .conftest import SR
from .test_extraction import FakeExtractor, run


def seg(spk, a, b):
    return DiarizedSegment(speaker=spk, start=a, end=b)


def test_segments_do_not_merge_across_someone_else():
    result = DiarizationResult(segments=[seg("A", 0, 5), seg("B", 5.1, 5.4), seg("A", 5.5, 8)])
    assert [t.speaker for t in segments_to_turns(result)] == ["A", "B", "A"]


def test_hidden_turns_merge_and_drop_tiny_bits():
    result = DiarizationResult(
        segments=[seg("A", 0, 10)],
        hidden=[seg("B", 2, 3), seg("B", 3.3, 4), seg("B", 8, 8.3)],
    )
    turns = hidden_turns(result)
    assert [(t.id, t.speaker, t.start, t.end, t.overlap) for t in turns] == [
        ("H0001", "B", 2, 4, True)  # 2-3 and 3.3-4 merged; 0.3 s at 8 s dropped
    ]


def test_pyannote_output_gives_hidden_speech():
    pytest.importorskip("pyannote.core")
    pytest.importorskip("torch")
    from pyannote.core import Annotation, Segment

    from speakerlens.diarization.pyannote_diarizer import PyannoteDiarizer

    regular, exclusive = Annotation(), Annotation()
    regular[Segment(0, 10), "1"] = "A"
    regular[Segment(6, 9), "2"] = "B"
    regular[Segment(12, 14), "3"] = "B"
    exclusive[Segment(0, 10), "1"] = "A"
    exclusive[Segment(12, 14), "3"] = "B"

    class Output:
        speaker_diarization = regular
        exclusive_speaker_diarization = exclusive

    d = PyannoteDiarizer()
    d._pipeline = lambda file, **kw: Output()
    result = d.diarize(np.zeros(16000 * 15, np.float32), 16000)
    assert [(h.speaker, h.start, h.end) for h in result.hidden] == [("B", 6, 9)]
    assert [s.speaker for s in result.segments] == ["A", "B"]  # exclusive view unchanged


def test_echo_check_ignores_scattered_common_characters():
    alice = Turn(
        id="T1",
        speaker="A",
        start=0,
        end=20,
        text="我觉得这个预算是我们今年最大的问题因为成本一直在涨",
    )
    bob = Turn(id="H1", speaker="B", start=5, end=7, text="我不是这么想的")  # shares 我/是/的 only
    leaked = Turn(id="H2", speaker="B", start=5, end=7, text="这个预算是我们今年最大的问题")
    assert not is_echo(bob, [alice])
    assert is_echo(leaked, [alice])


class PassThrough(FakeExtractor):
    """An extraction that fails to remove the main speaker: returns the mixture."""

    def extract(self, clip, prepared):
        self.calls += 1
        return clip


def test_hidden_speech_that_repeats_the_main_speaker_is_dropped(registry):
    transcript, _ = run(registry, PassThrough())
    assert [t.id for t in transcript.turns] == ["T0001", "T0002"]  # H0001 repeated Alice


def test_hidden_speech_needs_an_extractor(registry):
    transcript, _ = run(registry, None)
    assert [t.id for t in transcript.turns] == ["T0001", "T0002"]


def test_little_hidden_speech_is_not_transcribed(registry):
    from speakerlens.pipeline import Pipeline

    from .conftest import FakeLanguageID
    from .test_extraction import AliceUnderBob, Recorder

    registry.register(Recorder(), check_available=False)
    pipeline = Pipeline(
        registry=registry,
        diarizer=AliceUnderBob(),  # 2 s of hidden speech
        language_id=FakeLanguageID([{"en": 0.99, "zh": 0.01}]),
        extractor=FakeExtractor(),
    )
    mixture = np.full(200 * SR, 0.5, dtype=np.float32)  # in 200 s: 1%, under the 2% minimum
    mixture[6 * SR :] = -0.5
    transcript = pipeline.run("m.wav", PipelineOptions(summarize=False), audio=mixture)
    assert [t.id for t in transcript.turns] == ["T0001", "T0002"]
