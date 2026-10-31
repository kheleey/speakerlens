"""Diarization interface and turn building."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel, Field

from speakerlens.types import Turn


class DiarizedSegment(BaseModel):
    speaker: str
    start: float
    end: float


class DiarizationResult(BaseModel):
    # Exclusive segments: at most one speaker at any moment (easier to transcribe).
    segments: list[DiarizedSegment]
    # Regions where two or more people spoke at once.
    overlaps: list[tuple[float, float]] = Field(default_factory=list)
    # Speech the exclusive view hides: a speaker who talks while someone else holds the
    # floor (from the overlap-aware view). Transcribed from that speaker's extracted voice.
    hidden: list[DiarizedSegment] = Field(default_factory=list)

    @property
    def speakers(self) -> list[str]:
        return sorted({s.speaker for s in self.segments})


class Diarizer(ABC):
    @abstractmethod
    def diarize(
        self,
        audio: np.ndarray,
        sample_rate: int,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> DiarizationResult: ...


def segments_to_turns(
    result: DiarizationResult,
    merge_gap: float = 0.5,
    max_turn: float = 30.0,
    min_turn: float = 0.3,
    min_overlap: float = 0.2,
    audio: np.ndarray | None = None,
    sample_rate: int = 16_000,
) -> list[Turn]:
    """Merge consecutive segments of the same speaker into turns.

    Segments are merged when the gap is at most `merge_gap` seconds and the merged turn
    stays under `max_turn` seconds (long turns hurt language ID and translation).
    If `audio` is given, single segments longer than `max_turn` (monologues) are split at
    the quietest moment before each `max_turn` boundary. Turns shorter than `min_turn`
    are dropped; turns that overlap someone else for at least `min_overlap` seconds are
    flagged.
    """
    # Segments are exclusive (one speaker at a time): each merges with the previous one only.
    groups = [list(result.segments)]
    merged: list[list] = []  # [speaker, start, end]
    for group in groups:
        mine: list[list] = []
        for seg in sorted(group, key=lambda x: x.start):
            if (
                mine
                and mine[-1][0] == seg.speaker
                and seg.start - mine[-1][2] <= merge_gap
                and seg.end - mine[-1][1] <= max_turn
            ):
                mine[-1][2] = max(mine[-1][2], seg.end)
            else:
                mine.append([seg.speaker, seg.start, seg.end])
        merged += mine
    merged.sort(key=lambda m: m[1])

    if audio is not None:
        pieces = []
        for speaker, start, end in merged:
            for a, b in split_points(audio, sample_rate, start, end, max_turn):
                pieces.append([speaker, a, b])
        merged = pieces

    turns: list[Turn] = []
    for speaker, start, end in sorted(merged, key=lambda m: m[1]):
        if end - start < min_turn:
            continue
        overlap = sum(
            max(0.0, min(end, o_end) - max(start, o_start)) for o_start, o_end in result.overlaps
        )
        turns.append(
            Turn(
                id=f"T{len(turns) + 1:04d}",
                speaker=speaker,
                start=round(start, 3),
                end=round(end, 3),
                overlap=overlap >= min_overlap,
            )
        )
    return turns


def hidden_turns(
    result: DiarizationResult, merge_gap: float = 0.5, min_turn: float = 0.5
) -> list[Turn]:
    """Turns for the hidden speech (someone talking under someone else), merged per speaker,
    with ids H0001, H0002, ... so they never clash with the regular turns."""
    merged: list[list] = []
    for spk in sorted({h.speaker for h in result.hidden}):
        mine: list[list] = []
        for h in sorted((h for h in result.hidden if h.speaker == spk), key=lambda h: h.start):
            if mine and h.start - mine[-1][2] <= merge_gap:
                mine[-1][2] = max(mine[-1][2], h.end)
            else:
                mine.append([spk, h.start, h.end])
        merged += mine
    kept = sorted((m for m in merged if m[2] - m[1] >= min_turn), key=lambda m: m[1])
    return [
        Turn(id=f"H{i:04d}", speaker=spk, start=round(a, 3), end=round(b, 3), overlap=True)
        for i, (spk, a, b) in enumerate(kept, 1)
    ]


def split_points(
    audio: np.ndarray,
    sample_rate: int,
    start: float,
    end: float,
    max_len: float = 30.0,
    search: float = 8.0,
    frame: float = 0.1,
) -> list[tuple[float, float]]:
    """Split [start, end] into pieces of at most `max_len` seconds, cutting at the quietest
    `frame`-long moment within the last `search` seconds before each limit."""
    pieces: list[tuple[float, float]] = []
    cur = start
    hop = int(frame * sample_rate)
    while end - cur > max_len:
        lo = int((cur + max_len - search) * sample_rate)
        hi = int((cur + max_len) * sample_rate)
        window = audio[lo:hi]
        n = len(window) // hop
        if n == 0:
            cut = cur + max_len
        else:
            energy = (window[: n * hop].reshape(n, hop) ** 2).mean(axis=1)
            # Ties (e.g. silence) go to the latest frame, keeping pieces long.
            best = n - 1 - int(np.argmin(energy[::-1]))
            cut = (lo + best * hop + hop // 2) / sample_rate
        pieces.append((cur, cut))
        cur = cut
    pieces.append((cur, end))
    return pieces
