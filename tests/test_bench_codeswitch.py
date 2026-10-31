"""The code-switching split in bench/error_breakdown.py (pure Python, no models)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))

from error_breakdown import codeswitch_breakdown, missed_tokens  # noqa: E402


def line(text, start, spk="A"):
    return (spk, start, start + 2, text)


def test_words_are_split_by_line_language_and_word_language():
    ref = [line("我们去 hiking 吧", 0), line("I think 我们 should go", 3, "B")]
    hyp = [line("我们去 fighting 吧", 0), line("I think women should go", 3, "B")]
    split = codeswitch_breakdown(ref, hyp)
    assert split[("zh", "zh")] == [4, 4]  # 我 们 去 吧
    assert split[("zh", "en")] == [0, 1]  # hiking -> fighting: English inside Mandarin
    assert split[("en", "en")] == [4, 4]  # i think should go
    assert split[("en", "zh")] == [0, 2]  # 我们 -> women: Mandarin inside English


def test_speaker_labels_do_not_matter():
    ref = [line("hello there", 0, "A")]
    hyp = [line("hello there", 0, "SPEAKER_07")]
    assert codeswitch_breakdown(ref, hyp)[("en", "en")] == [2, 2]


def test_missed_lines_need_output_from_someone_at_that_time():
    ref = [line("你好", 0), line("再见", 10)]
    hyp = [line("你好", 0.5)]
    assert missed_tokens(ref, hyp) == (2, 4)  # the second line had no output at all
