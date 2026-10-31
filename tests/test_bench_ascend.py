"""Simulated meetings in bench/make_ascend.py (no dataset needed)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))

from make_ascend import SR, Utt, build_meetings  # noqa: E402


def clips(session, speaker, n):
    return [Utt(session, speaker, "", f"{speaker} {i}", np.zeros(SR), None, i) for i in range(n)]


def test_a_person_recorded_in_two_rounds_appears_once_per_meeting():
    utts = []
    for spk in ("spk1", "spk2", "spk3", "spk4"):
        utts += clips("1", spk, 6) + clips("2", spk, 8)
    for seed in range(20):
        meetings = build_meetings(utts, 1, 4, 0.5, 0.1, seed)
        placed = next(iter(meetings.values()))
        voice_of = {}
        for p in placed:
            voice_of.setdefault(p.speaker, set()).add(p.utt.speaker)
        people = [next(iter(v)) for v in voice_of.values()]
        assert all(len(v) == 1 for v in voice_of.values())
        assert len(set(people)) == len(people) == 4
        assert {p.utt.session for p in placed if p.utt.speaker == "spk1"} == {"2"}
