"""Reading other systems' STM transcripts in bench/eval_summaries.py (no LLM calls)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))

from eval_summaries import match_speakers, stm_transcript  # noqa: E402


def test_stm_transcript_orders_turns_and_numbers_them(tmp_path):
    stm = tmp_path / "m1.stm"
    stm.write_text(
        "m1 1 S1 4.000 6.000 我也同意\n;; comment\nm1 1 S0 0.000 3.500 我们 去 杭州 hiking 吧\n",
        encoding="utf-8",
    )
    t = stm_transcript(stm, "m1")
    assert [x.id for x in t.turns] == ["T0001", "T0002"]
    assert [x.speaker for x in t.turns] == ["S0", "S1"]
    assert t.turns[0].text == "我们 去 杭州 hiking 吧"
    assert {s.id for s in t.speakers} == {"S0", "S1"}
    assert t.duration == 6.0


def test_system_speakers_are_matched_by_shared_time(tmp_path):
    pytest.importorskip("scipy")
    (tmp_path / "ref.stm").write_text("m 1 alice 0 5 a\nm 1 bob 5 10 b\n", encoding="utf-8")
    (tmp_path / "hyp.stm").write_text("m 1 S1 0 4.5 a\nm 1 S0 5.2 10 b\n", encoding="utf-8")
    ref = stm_transcript(tmp_path / "ref.stm", "m")
    hyp = stm_transcript(tmp_path / "hyp.stm", "m")
    assert match_speakers(hyp, ref) == {"S1": "alice", "S0": "bob"}
