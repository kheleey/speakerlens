"""Caches slow stage results (like diarization) by audio hash and parameters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class StageCache:
    def __init__(self, directory: str | Path | None, enabled: bool = True) -> None:
        self.directory = Path(directory) / "stages" if directory else None
        self.enabled = enabled and self.directory is not None

    def _path(self, stage: str, audio_hash: str, params: dict[str, Any]) -> Path:
        assert self.directory is not None
        key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
        return self.directory / stage / f"{audio_hash[:32]}-{key}.json"

    def get(self, stage: str, audio_hash: str, params: dict[str, Any]) -> str | None:
        if not self.enabled:
            return None
        path = self._path(stage, audio_hash, params)
        return path.read_text(encoding="utf-8") if path.exists() else None

    def put(self, stage: str, audio_hash: str, params: dict[str, Any], payload: str) -> None:
        if not self.enabled:
            return
        path = self._path(stage, audio_hash, params)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
