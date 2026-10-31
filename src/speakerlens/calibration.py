"""Confidence calibration.

Each model reports confidence on its own scale, so "0.8" from Whisper and "0.8" from
another model mean different things. For each plugin we fit a monotonic mapping
(isotonic regression) from raw confidence to "probability this turn is right", using a
small labeled calibration set. After calibration, scores from different plugins can be
compared directly, which is what the router's "run two plugins, keep the better one"
rule relies on.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

MIN_SAMPLES = 50


def pool_adjacent_violators(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Isotonic (non-decreasing) fit of y on x. Returns block centers and block values."""
    order = np.argsort(x, kind="stable")
    xs, ys = x[order].astype(float), y[order].astype(float)
    # Each block: [sum_y, count, sum_x]
    blocks: list[list[float]] = []
    for xi, yi in zip(xs, ys, strict=True):
        blocks.append([yi, 1.0, xi])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            sy, n, sx = blocks.pop()
            blocks[-1][0] += sy
            blocks[-1][1] += n
            blocks[-1][2] += sx
    centers = np.array([b[2] / b[1] for b in blocks])
    values = np.array([b[0] / b[1] for b in blocks])
    return centers, values


def expected_calibration_error(
    confidence: list[float] | np.ndarray, correct: list[bool] | np.ndarray, n_bins: int = 10
) -> float:
    """Average gap between confidence and accuracy, weighted by how many turns fall in
    each confidence bin. 0 means perfectly calibrated."""
    conf = np.asarray(confidence, dtype=float)
    hit = np.asarray(correct, dtype=float)
    if len(conf) == 0:
        return 0.0
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(conf[mask].mean() - hit[mask].mean())
    return float(ece)


class Calibrator(BaseModel):
    plugin_id: str
    method: Literal["identity", "isotonic"] = "identity"
    xs: list[float] = Field(default_factory=list)
    ys: list[float] = Field(default_factory=list)
    n_samples: int = 0
    ece_before: float | None = None
    ece_after: float | None = None

    def apply(self, raw: float | None) -> float | None:
        if raw is None:
            return None
        if self.method == "identity" or not self.xs:
            return float(np.clip(raw, 0.0, 1.0))
        return float(np.clip(np.interp(raw, self.xs, self.ys), 0.0, 1.0))

    @classmethod
    def fit(cls, plugin_id: str, raw: list[float], correct: list[bool]) -> Calibrator:
        x = np.asarray(raw, dtype=float)
        y = np.asarray(correct, dtype=float)
        if len(x) < MIN_SAMPLES:
            log.warning(
                "Only %d calibration samples for %s; aim for at least %d.",
                len(x),
                plugin_id,
                MIN_SAMPLES,
            )
        centers, values = pool_adjacent_violators(x, y)
        calibrator = cls(
            plugin_id=plugin_id,
            method="isotonic",
            xs=centers.tolist(),
            ys=values.tolist(),
            n_samples=len(x),
        )
        calibrator.ece_before = expected_calibration_error(np.clip(x, 0, 1), y)
        calibrator.ece_after = expected_calibration_error(
            [calibrator.apply(v) or 0.0 for v in x], y
        )
        return calibrator

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.plugin_id}.json"
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path


class CalibrationStore:
    """Loads saved calibrators; plugins without one get the identity mapping."""

    def __init__(self, directory: str | Path | None = None) -> None:
        self.directory = Path(directory) if directory else None
        self._cache: dict[str, Calibrator] = {}

    def get(self, plugin_id: str) -> Calibrator:
        if plugin_id not in self._cache:
            self._cache[plugin_id] = self._load(plugin_id)
        return self._cache[plugin_id]

    def _load(self, plugin_id: str) -> Calibrator:
        if self.directory:
            path = self.directory / f"{plugin_id}.json"
            if path.exists():
                return Calibrator.model_validate(json.loads(path.read_text(encoding="utf-8")))
        return Calibrator(plugin_id=plugin_id)
