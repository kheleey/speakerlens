"""Voice enrollment: name speakers using short voice samples.

Put one audio file per person in a folder, named after them:

    voices/
      Alice.m4a
      Bob.wav

Each diarized speaker is matched to the most similar enrolled voice (cosine similarity
of speaker embeddings). Speakers below the threshold keep their SPEAKER_xx label.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from speakerlens.audio import SAMPLE_RATE, file_hash, load_audio, slice_audio
from speakerlens.types import Turn

log = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".mp4", ".webm"}


class SpeakerEmbedder(ABC):
    @abstractmethod
    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray: ...


class PyannoteEmbedder(SpeakerEmbedder):
    """WeSpeaker ResNet34 embeddings (the same embedding model community-1 uses)."""

    def __init__(
        self,
        model: str = "pyannote/wespeaker-voxceleb-resnet34-LM",
        hf_token: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.hf_token = hf_token
        self.device = device
        self._inference: Any = None

    def _load(self) -> Any:
        if self._inference is None:
            import torch
            from pyannote.audio import Inference, Model

            try:
                model = Model.from_pretrained(self.model, token=self.hf_token)
            except TypeError:
                model = Model.from_pretrained(self.model, use_auth_token=self.hf_token)
            self._inference = Inference(model, window="whole", device=torch.device(self.device))
        return self._inference

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        import torch

        inference = self._load()
        waveform = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)
        vector = np.asarray(inference({"waveform": waveform, "sample_rate": sample_rate}))
        return vector.reshape(-1).astype(np.float32)


def normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


def speaker_audio(
    audio: np.ndarray, turns: list[Turn], speaker: str, max_seconds: float = 30.0
) -> np.ndarray:
    """Concatenate a speaker's longest non-overlapping turns, up to `max_seconds`."""
    own = sorted(
        (t for t in turns if t.speaker == speaker and not t.overlap),
        key=lambda t: t.duration,
        reverse=True,
    )
    pieces, total = [], 0.0
    for t in own:
        if total >= max_seconds:
            break
        end = min(t.end, t.start + (max_seconds - total))
        pieces.append(slice_audio(audio, t.start, end))
        total += end - t.start
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)


class VoiceBank:
    def __init__(self, voices: dict[str, np.ndarray]) -> None:
        self.voices = {name: normalize(v) for name, v in voices.items()}

    @classmethod
    def from_dir(
        cls, directory: str | Path, embedder: SpeakerEmbedder, cache_dir: str | Path | None = None
    ) -> VoiceBank:
        directory = Path(directory)
        cache = Path(cache_dir) / "voices" if cache_dir else None
        voices: dict[str, np.ndarray] = {}
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            cached = cache / f"{file_hash(path)}.npy" if cache else None
            if cached and cached.exists():
                voices[path.stem] = np.load(cached)
                continue
            vector = embedder.embed(load_audio(path), SAMPLE_RATE)
            if cached:
                cached.parent.mkdir(parents=True, exist_ok=True)
                np.save(cached, vector)
            voices[path.stem] = vector
        if not voices:
            log.warning("No voice samples found in %s", directory)
        return cls(voices)

    def match(
        self, speakers: dict[str, np.ndarray], threshold: float = 0.5
    ) -> dict[str, tuple[str, float]]:
        """Greedy one-to-one matching: speaker id -> (name, cosine similarity)."""
        pairs = []
        for spk, vec in speakers.items():
            if vec.size == 0:
                continue
            v = normalize(vec)
            for name, ref in self.voices.items():
                pairs.append((float(np.dot(v, ref)), spk, name))
        pairs.sort(reverse=True)
        matched: dict[str, tuple[str, float]] = {}
        used_names: set[str] = set()
        for score, spk, name in pairs:
            if score < threshold:
                break
            if spk in matched or name in used_names:
                continue
            matched[spk] = (name, score)
            used_names.add(name)
        return matched
