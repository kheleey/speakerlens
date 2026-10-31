"""The comparison scripts in bench/compare (pure Python, no models): reading VibeVoice-ASR
output that was cut off, and turning WhisperX's per-word speakers into turns."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench" / "compare"))

from common import turns_by_word_speaker  # noqa: E402
from run_vibevoice import salvage  # noqa: E402


def zh_segment(chars, speakers, start=0.0, step=0.3, seg_speaker="A"):
    """A Mandarin segment as WhisperX aligns it: one 'word' per character, spaces dropped."""
    text = "".join(chars)
    words, t = [], start
    for ch, spk in zip(chars, speakers, strict=True):
        if ch == " ":
            continue
        w = {"word": ch, "start": t, "end": t + step}
        if spk is not None:
            w["speaker"] = spk
        words.append(w)
        t += step
    return {"text": text, "start": start, "end": t, "speaker": seg_speaker, "words": words}


def test_segment_is_split_where_the_word_speaker_changes():
    seg = zh_segment(list("我们去吧好的"), ["A", "A", "A", "A", "B", "B"])
    turns = turns_by_word_speaker([seg])
    assert [(s, x) for s, _a, _b, x in turns] == [("A", "我们去吧"), ("B", "好的")]
    assert turns[0][1] == 0.0 and abs(turns[1][1] - 1.2) < 1e-9  # B starts at its first word


def test_english_inside_mandarin_comes_out_whole():
    chars = list("我想去 hiking 吧")
    seg = zh_segment(chars, ["A"] * len(chars))
    assert turns_by_word_speaker([seg])[0][3] == "我想去 hiking 吧"


def test_english_segment_keeps_its_spaces():
    seg = {
        "text": " I think so. Yes.",
        "start": 0.0,
        "end": 2.0,
        "speaker": "A",
        "words": [
            {"word": "I", "start": 0.0, "end": 0.2, "speaker": "A"},
            {"word": "think", "start": 0.2, "end": 0.6, "speaker": "A"},
            {"word": "so.", "start": 0.6, "end": 1.0, "speaker": "A"},
            {"word": "Yes.", "start": 1.2, "end": 2.0, "speaker": "B"},
        ],
    }
    assert [(s, x) for s, _a, _b, x in turns_by_word_speaker([seg])] == [
        ("A", "I think so."),
        ("B", "Yes."),
    ]


def test_words_without_speaker_or_timing_stay_with_the_word_before():
    seg = zh_segment(list("对对好"), ["B", None, "A"], seg_speaker="A")
    del seg["words"][1]["start"]  # an unalignable word: no timing, no speaker
    turns = turns_by_word_speaker([seg])
    assert [(s, x) for s, _a, _b, x in turns] == [("B", "对对"), ("A", "好")]


def test_segments_without_word_timing_stay_whole():
    seg = {"text": "没有对齐", "start": 1.0, "end": 3.0, "speaker": "C", "words": []}
    assert turns_by_word_speaker([seg]) == [("C", 1.0, 3.0, "没有对齐")]


TURN_1 = '{"Start":0.0,"End":3.2,"Speaker":0,"Content":"我们开始吧 {笑}"}'
TURN_2 = '{"Start":3.2,"End":7.5,"Speaker":1,"Content":"好的，\\"AI\\" 那个 [laugh]"}'


def test_complete_turns_are_kept_when_the_last_one_is_cut_off():
    raw = f'assistant\n[{TURN_1},{TURN_2},{{"Start":7.5,"End":9.0,"Speaker":0,"Content":"对对对对'
    turns = salvage(raw)
    assert [t["Content"] for t in turns] == ["我们开始吧 {笑}", '好的，"AI" 那个 [laugh]']
    assert turns[-1]["End"] == 7.5


def test_whole_output_reads_the_same():
    assert len(salvage(f"assistant\n[{TURN_1}, {TURN_2}]\n")) == 2


def test_cut_off_inside_the_numbers():
    assert len(salvage(f'[{TURN_1},{{"Start":7.')) == 1


def test_not_json_at_all():
    assert salvage("assistant\nsorry, no transcript") == []
    assert salvage("") == []


def test_turns_without_text_or_with_bad_times_are_skipped():
    raw = f'[{{"Start":1,"End":2,"Speaker":0}},{{"Start":"x","End":2,"Content":"a"}},{TURN_2}'
    assert [t["Speaker"] for t in salvage(raw)] == [1]
