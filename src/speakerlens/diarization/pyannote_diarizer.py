"""Diarization with pyannote community-1 (model license: CC-BY-4.0, gated download).

Before first use: accept the model's terms on Hugging Face
(https://huggingface.co/pyannote/speaker-diarization-community-1) and put a read token
in HF_TOKEN in your .env file.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from speakerlens.diarization.base import DiarizationResult, DiarizedSegment, Diarizer


def _from_pretrained(cls: Any, name: str, token: str | None) -> Any:
    try:
        return cls.from_pretrained(name, token=token)  # pyannote.audio 4.x
    except TypeError:
        return cls.from_pretrained(name, use_auth_token=token)  # pyannote.audio 3.x


class PyannoteDiarizer(Diarizer):
    def __init__(
        self,
        model: str = "pyannote/speaker-diarization-community-1",
        hf_token: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.hf_token = hf_token
        self.device = device
        self._pipeline: Any = None

    def _load(self) -> Any:
        if self._pipeline is None:
            import torch
            from pyannote.audio import Pipeline

            pipeline = _from_pretrained(Pipeline, self.model, self.hf_token)
            if pipeline is None:
                raise RuntimeError(
                    f"Could not load {self.model}. Accept its terms on Hugging Face and set "
                    "HF_TOKEN in your .env file."
                )
            pipeline.to(torch.device(self.device))
            self._pipeline = pipeline
        return self._pipeline

    def diarize(
        self,
        audio: np.ndarray,
        sample_rate: int,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> DiarizationResult:
        import torch

        pipeline = self._load()
        kwargs = {
            k: v
            for k, v in {
                "num_speakers": num_speakers,
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
            }.items()
            if v is not None
        }
        waveform = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)
        output = pipeline({"waveform": waveform, "sample_rate": sample_rate}, **kwargs)

        # pyannote.audio 4 returns an object with both views; 3.x returns an Annotation.
        regular = getattr(output, "speaker_diarization", output)
        exclusive = getattr(output, "exclusive_speaker_diarization", None)
        if exclusive is None:
            exclusive = regular

        segments = [
            DiarizedSegment(speaker=str(label), start=float(seg.start), end=float(seg.end))
            for seg, _track, label in exclusive.itertracks(yield_label=True)
        ]
        overlaps = [(float(s.start), float(s.end)) for s in regular.get_overlap()]
        hidden = []
        if exclusive is not regular:
            for label in regular.labels():
                rest = regular.label_timeline(label).extrude(exclusive.label_timeline(label))
                hidden += [
                    DiarizedSegment(speaker=str(label), start=float(seg.start), end=float(seg.end))
                    for seg in rest
                    if seg.duration >= 0.2
                ]
        return DiarizationResult(segments=segments, overlaps=overlaps, hidden=hidden)
