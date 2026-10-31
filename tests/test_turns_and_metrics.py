from speakerlens.diarization.base import DiarizationResult, DiarizedSegment, segments_to_turns
from speakerlens.langid import restrict
from speakerlens.metrics import error_rate, tokenize


def test_segments_merge_and_overlap_flag():
    result = DiarizationResult(
        segments=[
            DiarizedSegment(speaker="A", start=0.0, end=2.0),
            DiarizedSegment(speaker="A", start=2.3, end=4.0),  # small gap: merged
            DiarizedSegment(speaker="B", start=4.0, end=6.0),
            DiarizedSegment(speaker="B", start=6.0, end=6.1),  # merged into B's turn
            DiarizedSegment(speaker="A", start=8.0, end=8.1),  # too short: dropped
        ],
        overlaps=[(5.0, 5.5)],
    )
    turns = segments_to_turns(result)
    assert [(t.speaker, t.start, t.end) for t in turns] == [("A", 0.0, 4.0), ("B", 4.0, 6.1)]
    assert [t.overlap for t in turns] == [False, True]
    assert [t.id for t in turns] == ["T0001", "T0002"]


def test_long_turns_are_split_at_segment_boundaries():
    result = DiarizationResult(
        segments=[DiarizedSegment(speaker="A", start=i * 10.0, end=i * 10.0 + 10) for i in range(5)]
    )
    turns = segments_to_turns(result, max_turn=30)
    assert all(t.duration <= 30 for t in turns)
    assert len(turns) == 2


def test_mixed_tokenization():
    assert tokenize("我们用 Python 写 code!") == ["我", "们", "用", "python", "写", "code"]


def test_error_rate():
    assert error_rate("hello world", "hello world") == 0.0
    assert error_rate("你好世界", "你好") == 0.5
    assert error_rate("", "") == 0.0


def test_restrict_to_allowed_languages():
    probs = restrict({"en": 0.5, "zh": 0.3, "ja": 0.2}, ["zh", "en"])
    assert abs(probs["en"] - 0.625) < 1e-9 and set(probs) == {"en", "zh"}
    assert set(restrict({"en": 0.5, "zh": 0.3, "ja": 0.2, "ko": 0.0}, [])) == {"en", "zh", "ja"}
    assert restrict({"fr": 1.0}, ["zh", "en"]) == {"zh": 0.5, "en": 0.5}
