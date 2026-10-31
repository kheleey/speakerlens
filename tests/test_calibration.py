import numpy as np

from speakerlens.calibration import (
    CalibrationStore,
    Calibrator,
    expected_calibration_error,
    pool_adjacent_violators,
)


def test_pav_is_monotonic():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    y = np.array([0, 1, 0, 1, 1])
    _, values = pool_adjacent_violators(x, y)
    assert all(a <= b for a, b in zip(values, values[1:], strict=False))


def test_calibration_fixes_an_overconfident_model():
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.6, 1.0, 2000)
    # The model claims 60-100% but is actually right about half as often.
    correct = rng.uniform(size=raw.size) < (raw - 0.5)
    cal = Calibrator.fit("overconfident", raw.tolist(), correct.tolist())
    assert cal.ece_before is not None and cal.ece_after is not None
    assert cal.ece_after < cal.ece_before / 3
    assert cal.apply(0.95) < 0.6


def test_identity_when_no_calibrator(tmp_path):
    store = CalibrationStore(tmp_path)
    assert store.get("unknown").apply(0.42) == 0.42
    assert store.get("unknown").apply(None) is None


def test_save_and_load_roundtrip(tmp_path):
    cal = Calibrator.fit("p", [0.2, 0.4, 0.6, 0.8] * 20, [False, False, True, True] * 20)
    cal.save(tmp_path)
    loaded = CalibrationStore(tmp_path).get("p")
    assert loaded.method == "isotonic"
    assert loaded.apply(0.8) == cal.apply(0.8)


def test_ece_of_perfect_calibration_is_zero():
    assert expected_calibration_error([1.0, 1.0, 0.0], [True, True, False]) == 0.0
