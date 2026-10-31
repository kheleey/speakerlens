"""Core data types shared by every pipeline stage.

Everything here serializes to JSON, so each stage's output can be saved, cached and
re-loaded independently.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Word(BaseModel):
    text: str
    start: float  # seconds from the start of the recording
    end: float
    confidence: float | None = None


class Turn(BaseModel):
    """One stretch of speech by one speaker."""

    id: str  # "T0001", used by summaries to cite where a point was said
    speaker: str  # diarization label, e.g. "SPEAKER_00"
    start: float
    end: float
    overlap: bool = False  # someone else was talking at the same time
    language: str | None = None  # "en", "zh", or "en+zh" for mixed speech
    language_confidence: float | None = None
    plugin: str | None = None  # id of the plugin that produced the text
    text: str = ""
    words: list[Word] = Field(default_factory=list)
    raw_confidence: float | None = None  # as reported by the plugin
    confidence: float | None = None  # after calibration: probability the turn is right
    translation: str | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


class Speaker(BaseModel):
    id: str  # diarization label
    name: str | None = None  # filled in by voice enrollment
    match_score: float | None = None  # cosine similarity to the enrolled voice

    @property
    def label(self) -> str:
        return self.name or self.id


class SummaryPoint(BaseModel):
    text: str
    turn_ids: list[str] = Field(default_factory=list)
    with_speaker: str | None = None  # for agreements and disagreements


class SpeakerSummary(BaseModel):
    speaker: str  # speaker id
    main_points: list[SummaryPoint] = Field(default_factory=list)
    agreements: list[SummaryPoint] = Field(default_factory=list)
    disagreements: list[SummaryPoint] = Field(default_factory=list)
    open_questions: list[SummaryPoint] = Field(default_factory=list)


class RejectedPoint(BaseModel):
    speaker: str
    category: str
    text: str
    turn_ids: list[str]
    reason: str


class Summary(BaseModel):
    overview: str = ""
    language: str = "en"
    prompt_version: int = 1  # summarize.PROMPT_VERSION it was made with
    speakers: list[SpeakerSummary] = Field(default_factory=list)
    rejected_points: list[RejectedPoint] = Field(default_factory=list)


class Transcript(BaseModel):
    source: str
    duration: float
    languages: list[str]
    speakers: list[Speaker]
    turns: list[Turn]
    target_language: str | None = None
    summary: Summary | None = None
    created_with: str = "speakerlens 0.1.0"

    def speaker_label(self, speaker_id: str) -> str:
        for speaker in self.speakers:
            if speaker.id == speaker_id:
                return speaker.label
        return speaker_id

    def turn(self, turn_id: str) -> Turn | None:
        for t in self.turns:
            if t.id == turn_id:
                return t
        return None
