from speakerlens.summarize import check_attribution, summarize, transcript_lines
from speakerlens.types import Speaker, Transcript, Turn

from .conftest import FakeLLM


def make_transcript() -> Transcript:
    return Transcript(
        source="meeting.m4a",
        duration=20,
        languages=["zh", "en"],
        speakers=[Speaker(id="SPEAKER_00", name="Alice"), Speaker(id="SPEAKER_01")],
        turns=[
            Turn(id="T0001", speaker="SPEAKER_00", start=0, end=5, text="We should ship Friday."),
            Turn(id="T0002", speaker="SPEAKER_01", start=5, end=9, text="Too risky, tests fail."),
            Turn(id="T0003", speaker="SPEAKER_00", start=9, end=12, text="Then Monday."),
        ],
    )


def test_points_citing_someone_else_are_rejected():
    data = {
        "overview": "Release timing.",
        "speakers": [
            {
                "speaker": "Alice",
                "main_points": [
                    {"text": "Wants to ship soon", "turn_ids": ["T0001", "T0003"]},
                    {"text": "Says tests fail", "turn_ids": ["T0002"]},  # that was Bob
                ],
                "disagreements": [
                    # cites Bob's objection too: kept, citing only Alice's own turn
                    {"text": "Moved date", "with": "SPEAKER_01", "turn_ids": ["T0002", "T0003"]}
                ],
            },
            {
                "speaker": "speaker_01",  # case-insensitive id works too
                "main_points": [{"text": "Release is risky", "turn_ids": ["T0002"]}],
                "open_questions": [{"text": "No citation", "turn_ids": []}],
            },
            {"speaker": "Charlie", "main_points": [{"text": "x", "turn_ids": ["T0001"]}]},
        ],
    }
    summary = check_attribution(data, make_transcript(), "en")
    alice, bob = summary.speakers
    assert alice.speaker == "SPEAKER_00"
    assert [p.text for p in alice.main_points] == ["Wants to ship soon"]
    assert alice.disagreements[0].with_speaker == "SPEAKER_01"
    assert alice.disagreements[0].turn_ids == ["T0003"]
    assert [p.text for p in bob.main_points] == ["Release is risky"]
    reasons = sorted(r.reason for r in summary.rejected_points)
    assert reasons == ["cites only turns by someone else: T0002", "no turn cited"]


def test_transcript_lines_use_names():
    lines = transcript_lines(make_transcript())
    assert lines[0] == "[T0001] Alice (00:00:00): We should ship Friday."
    assert lines[1].startswith("[T0002] SPEAKER_01")


def test_long_transcripts_are_summarized_in_chunks():
    llm = FakeLLM(
        [
            {
                "overview": "Part.",
                "speakers": [
                    {"speaker": "Alice", "main_points": [{"text": "p", "turn_ids": ["T0001"]}]}
                ],
            }
        ]
    )
    summary = summarize(make_transcript(), llm, max_chars=40)
    assert len(llm.calls) == 3
    # Each chunk returned the same point; repeats of a point are kept once.
    assert [p.text for p in summary.speakers[0].main_points] == ["p"]


def _talk():
    from speakerlens.types import Speaker, Transcript, Turn

    turns = [
        Turn(id="T0001", speaker="A", start=0, end=5, text="我觉得面试放在五月五号到十号"),
        Turn(id="T0002", speaker="B", start=5, end=6, text="对"),
        Turn(id="T0003", speaker="B", start=6, end=9, text="可以 五月五号 开始 面试 挺好"),
        Turn(id="T0004", speaker="A", start=300, end=305, text="还有 培训 也要 安排 一下"),
    ]
    return Transcript(
        source="m",
        duration=310,
        languages=["zh"],
        speakers=[Speaker(id="A"), Speaker(id="B")],
        turns=turns,
    )


def test_agreement_with_someone_who_spoke_just_before_is_kept():
    from speakerlens.summarize import relation_problem

    assert relation_problem(["T0003"], "B", "A", _talk()) is None


def test_reversed_agreement_is_caught():
    # "A agrees with B" citing A's proposal: B had said nothing of substance before it.
    from speakerlens.summarize import relation_problem

    assert "responds to nobody" in relation_problem(["T0001"], "A", "B", _talk())
    # ...nor 5 minutes later: B's lines are too long ago to be what A responds to.
    assert relation_problem(["T0004"], "A", "B", _talk()) is not None


def test_relation_check_skips_unknown_partner_and_flags_self():
    from speakerlens.summarize import relation_problem

    assert relation_problem(["T0001"], "A", None, _talk()) is None
    assert relation_problem(["T0001"], "A", "A", _talk()) == "agrees or disagrees with themselves"


def test_check_attribution_rejects_reversed_relations_and_keeps_main_points():
    from speakerlens.summarize import PROMPT_VERSION, check_attribution

    data = {
        "speakers": [
            {
                "speaker": "A",
                "main_points": [{"text": "Proposes interviews May 5-10", "turn_ids": ["T0001"]}],
                "agreements": [
                    {"text": "Agrees with B on May 5-10", "with": "B", "turn_ids": ["T0001"]}
                ],
            },
            {
                "speaker": "B",
                "agreements": [
                    {"text": "Agrees with A on May 5-10", "with": "A", "turn_ids": ["T0003"]}
                ],
            },
        ]
    }
    summary = check_attribution(data, _talk(), "en")
    a, b = summary.speakers
    assert [p.text for p in a.main_points] == ["Proposes interviews May 5-10"]
    assert a.agreements == [] and len(b.agreements) == 1
    assert [r.text for r in summary.rejected_points] == ["Agrees with B on May 5-10"]
    assert summary.prompt_version == PROMPT_VERSION


def test_instructions_say_who_proposed_first():
    from speakerlens.summarize import SYSTEM_PROMPT

    assert "raised it FIRST" in SYSTEM_PROMPT and "EARLIER" in SYSTEM_PROMPT


def test_summaries_record_the_instructions_version():
    from speakerlens.summarize import PROMPT_VERSION, summarize

    class OneShot:
        def chat_json(self, system, user, **kwargs):
            return {"overview": "x", "speakers": []}

    assert summarize(_talk(), OneShot()).prompt_version == PROMPT_VERSION


def test_pasted_transcript_lines_are_rejected_unless_summarizing_in_chinese():
    from speakerlens.summarize import pasted_line

    line = "毕竟用户都是这全国人民，全国人民用起来会比较舒服了一些"
    assert pasted_line(line, "en")
    assert not pasted_line(line, "zh")
    assert not pasted_line("Proposed that 赠品 should carry the brand logo.", "en")


def test_duplicate_points_are_dropped_per_speaker():
    from speakerlens.summarize import drop_duplicates
    from speakerlens.types import SpeakerSummary, Summary, SummaryPoint

    point = "Agrees with B that the teacher might not accept gifts."
    summary = Summary(
        speakers=[
            SpeakerSummary(
                speaker="A",
                agreements=[SummaryPoint(text=point), SummaryPoint(text=point.upper() + "!")],
                main_points=[SummaryPoint(text="Wants flowers.")],
            ),
            SpeakerSummary(speaker="B", agreements=[SummaryPoint(text=point)]),
        ]
    )
    assert drop_duplicates(summary) == 1
    assert len(summary.speakers[0].agreements) == 1 and len(summary.speakers[1].agreements) == 1


def test_check_attribution_rejects_a_pasted_line():
    from speakerlens.summarize import check_attribution

    data = {
        "speakers": [
            {
                "speaker": "A",
                "main_points": [
                    {"text": "我觉得面试放在五月五号到十号", "turn_ids": ["T0001"]},
                    {"text": "Proposes interviews on May 5-10.", "turn_ids": ["T0001"]},
                ],
            }
        ]
    }
    summary = check_attribution(data, _talk(), "en")
    assert [p.text for p in summary.speakers[0].main_points] == ["Proposes interviews on May 5-10."]
    assert summary.rejected_points[0].reason.startswith("pasted a transcript line")
