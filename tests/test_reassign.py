import numpy as np
import pytest

from speakerlens.diarization.base import DiarizationResult, DiarizedSegment, Diarizer
from speakerlens.diarization.enrollment import SpeakerEmbedder
from speakerlens.diarization.reassign import reassign_turns, turn_embeddings
from speakerlens.pipeline import Pipeline, PipelineOptions
from speakerlens.types import Turn

from .conftest import SR, FakeLanguageID

E1, E2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])


def turn(tid, speaker, start, end, overlap=False):
    return Turn(id=tid, speaker=speaker, start=start, end=end, overlap=overlap)


def test_a_turn_moves_to_the_voice_it_matches():
    turns = [
        turn("T1", "A", 0, 3),
        turn("T2", "B", 3, 6),
        turn("T3", "A", 6, 9),  # labeled A, but it's B's voice
        turn("T4", "B", 9, 12),
        turn("T5", "A", 12, 15),
    ]
    emb = {"T1": E1, "T2": E2, "T3": E2, "T4": E2, "T5": E1}
    moves = reassign_turns(turns, emb, margin=0.1)
    assert [t.speaker for t in turns] == ["A", "B", "B", "B", "A"]
    assert [(m.turn_id, m.old, m.new) for m in moves] == [("T3", "A", "B")]


def test_nothing_moves_without_a_clear_margin():
    turns = [turn("T1", "A", 0, 3), turn("T2", "B", 3, 6), turn("T3", "A", 6, 9)]
    near = np.array([0.6, 0.8]) / np.linalg.norm([0.6, 0.8])
    emb = {"T1": np.array([0.8, 0.6]), "T2": near, "T3": np.array([0.75, 0.66])}
    emb = {k: v / np.linalg.norm(v) for k, v in emb.items()}
    assert reassign_turns(turns, emb, margin=0.3) == []
    assert [t.speaker for t in turns] == ["A", "B", "A"]


class VoiceByLevel(SpeakerEmbedder):
    """Audio filled with 0.1 is voice 1, with 0.2 voice 2."""

    def __init__(self):
        self.calls = 0

    def embed(self, audio, sample_rate):
        self.calls += 1
        return E1 if float(np.mean(audio)) < 0.15 else E2


def test_short_and_overlapping_turns_are_not_embedded():
    audio = np.full(10 * SR, 0.1, dtype=np.float32)
    turns = [turn("T1", "A", 0, 0.5), turn("T2", "A", 1, 4, overlap=True), turn("T3", "A", 4, 8)]
    emb = turn_embeddings(audio, SR, turns, VoiceByLevel())
    assert list(emb) == ["T3"]


class FourTurns(Diarizer):
    model = "fake"

    def diarize(self, audio, sample_rate, num_speakers=None, min_speakers=None, max_speakers=None):  # type: ignore[no-untyped-def]
        spans = [("A", 0.0, 3.0), ("B", 3.5, 6.5), ("A", 7.0, 10.0), ("B", 10.5, 13.0)]
        return DiarizationResult(
            segments=[DiarizedSegment(speaker=s, start=a, end=b) for s, a, b in spans]
        )


def test_pipeline_reassigns_when_asked(registry):
    audio = np.full(13 * SR, 0.2, dtype=np.float32)
    audio[: 3 * SR] = 0.1  # only the first turn is voice 1
    lid = FakeLanguageID([{"en": 0.99, "zh": 0.01}])

    def run(margin):
        pipeline = Pipeline(
            registry=registry, diarizer=FourTurns(), language_id=lid, embedder=VoiceByLevel()
        )
        options = PipelineOptions(summarize=False, reassign_margin=margin, use_cache=False)
        return [t.speaker for t in pipeline.run("m.wav", options, audio=audio).turns]

    assert run(None) == ["A", "B", "A", "B"]
    assert run(0.1) == ["A", "B", "B", "B"]


def test_overlapping_turns_count_only_when_long():
    from speakerlens.diarization.reassign import usable

    assert not usable(turn("T1", "A", 0, 3, overlap=True))
    assert usable(turn("T1", "A", 0, 3, overlap=True), overlap_min_seconds=2.0)
    assert not usable(turn("T1", "A", 0, 1.5, overlap=True), overlap_min_seconds=2.0)
    assert usable(turn("T2", "A", 0, 0.6), min_seconds=0.5)


def test_regroup_fixes_a_file_where_two_people_were_mixed():
    pytest.importorskip("scipy")
    from speakerlens.diarization.reassign import recluster

    # Diarization put voices 1 and 2 under both labels; the regroup splits them cleanly
    # and keeps the label that already held most of each voice.
    speakers = ["A", "A", "A", "B", "B", "B", "A", "B"]
    voices = [E1, E1, E2, E2, E2, E1, E1, E2]
    turns = [turn(f"T{i}", s, 3 * i, 3 * i + 2.5) for i, s in enumerate(speakers)]
    emb = {t.id: v for t, v in zip(turns, voices, strict=True)}
    changed = recluster(turns, emb, n_speakers=2)
    assert changed == 2
    assert [t.speaker for t in turns] == ["A" if v is E1 else "B" for v in voices]


def test_speaker_check_is_on_by_default(monkeypatch, tmp_path):
    from speakerlens.config import load_settings

    monkeypatch.delenv("SPEAKER_REASSIGN_MARGIN", raising=False)
    monkeypatch.chdir(tmp_path)
    assert load_settings().reassign_margin == 0.1
    monkeypatch.setenv("SPEAKER_REASSIGN_MARGIN", "0")
    assert load_settings().reassign_margin == 0


def test_voice_samples_split_a_merged_speaker():
    from speakerlens.diarization.reassign import anchors_from_voices

    # Diarization put Bob's one turn under A; only the voice samples know there are two people.
    turns = [turn(f"T{i}", "A", 3 * i, 3 * i + 2.5) for i in range(4)]
    emb = {"T0": E1, "T1": E1, "T2": E2, "T3": E1}
    anchors = anchors_from_voices(turns, emb, {"Alice": E1, "Bob": E2})
    assert set(anchors) == {"A", "VOICE_Bob"}
    reassign_turns(turns, emb, margin=0.1, anchors=anchors)
    assert [t.speaker for t in turns] == ["A", "A", "VOICE_Bob", "A"]


def test_pipeline_names_speakers_found_from_voice_samples(registry, monkeypatch, tmp_path):
    from speakerlens.diarization import enrollment
    from speakerlens.pipeline import VoiceBank

    audio = np.full(15 * SR, 0.1, dtype=np.float32)
    audio[8 * SR : 11 * SR] = 0.2  # the third turn is someone else, labeled A by diarization

    class OneVoiceDiarizer(Diarizer):
        model = "fake"

        def diarize(
            self, audio, sample_rate, num_speakers=None, min_speakers=None, max_speakers=None
        ):  # type: ignore[no-untyped-def]
            return DiarizationResult(
                segments=[DiarizedSegment(speaker="A", start=a, end=a + 3) for a in (0, 4, 8, 12)]
            )

    monkeypatch.setattr(
        VoiceBank, "from_dir", classmethod(lambda cls, *a, **k: cls({"Alice": E1, "Bob": E2}))
    )
    assert enrollment.VoiceBank is VoiceBank
    pipeline = Pipeline(
        registry=registry,
        diarizer=OneVoiceDiarizer(),
        language_id=FakeLanguageID([{"en": 0.99, "zh": 0.01}]),
        embedder=VoiceByLevel(),
    )
    options = PipelineOptions(
        summarize=False, reassign_margin=0.1, voices_dir=tmp_path, use_cache=False
    )
    transcript = pipeline.run("m.wav", options, audio=audio)
    names = {s.id: s.name for s in transcript.speakers}
    assert [names[t.speaker] for t in transcript.turns] == ["Alice", "Alice", "Bob", "Alice"]


def test_names_use_the_best_overall_assignment():
    pytest.importorskip("scipy")
    from speakerlens.diarization.reassign import name_speakers

    x, y = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    a = np.array([0.95, 0.9, 0.0])  # sounds a bit more like X than Y...
    b = np.array([0.7, 0.2, 0.68])  # ...but B sounds like X and not at all like Y
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    turns = [turn("T1", "A", 0, 3), turn("T2", "B", 3, 6)]
    names = name_speakers(turns, {"T1": a, "T2": b}, {"X": x, "Y": y})
    # Best-pair-first would give A=X and leave B unnamed (B-Y is too dissimilar).
    assert {k: v[0] for k, v in names.items()} == {"A": "Y", "B": "X"}
