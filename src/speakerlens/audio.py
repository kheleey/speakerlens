"""Audio loading. Every stage works on 16 kHz mono float32 numpy arrays."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000


class AudioError(RuntimeError):
    pass


def load_audio(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any audio or video file with ffmpeg into mono float32 at `sample_rate`."""
    if shutil.which("ffmpeg") is None:
        raise AudioError("ffmpeg is not installed. On Ubuntu/Debian: sudo apt install ffmpeg")
    path = Path(path)
    if not path.exists():
        raise AudioError(f"File not found: {path}")
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        str(path),
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg could not decode {path}:\n{proc.stderr.decode(errors='ignore')}")
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def slice_audio(
    audio: np.ndarray, start: float, end: float, sample_rate: int = SAMPLE_RATE
) -> np.ndarray:
    lo = max(0, int(start * sample_rate))
    hi = min(len(audio), int(end * sample_rate))
    return audio[lo:hi]


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(audio: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(audio).tobytes()).hexdigest()
