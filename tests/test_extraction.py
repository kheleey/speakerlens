import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from speakerlens import extraction
from speakerlens.diarization.base import DiarizationResult, DiarizedSegment, Diarizer
from speakerlens.pipeline import Pipeline, PipelineOptions, voice_samples
from speakerlens.plugins.base import LanguagePlugin, PluginManifest, PluginResult
from speakerlens.types import Turn

from .conftest import SR, FakeLanguageID


def seg(spk, a, b):
    return DiarizedSegment(speaker=spk, start=a, end=b)


class AliceUnderBob(Diarizer):
    """Alice holds the floor 0-6 s with Bob talking under her at 2-4 s (hidden speech);
    Bob has the floor alone 6-9 s."""

    model = "fake"

    def diarize(self, audio, sample_rate, **kwargs):  # type: ignore[no-untyped-def]
        return DiarizationResult(
            segments=[seg("A", 0, 6), seg("B", 6, 9)],
            overlaps=[(2, 4)],
            hidden=[seg("B", 2, 4)],
        )


class FakeExtractor:
    """'Extracts' a constant signal whose sign is the sampled voice's sign (the mixture is
    +0.5 while Alice talks and -0.5 while Bob does)."""

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    def prepare(self, sample):
        return float(np.sign(np.mean(sample)))

    def extract(self, clip, prepared):
        self.calls += 1
        if self.fail:
            raise RuntimeError("out of memory")
        return np.full(len(clip), 0.3 * prepared, np.float32)


class Recorder(LanguagePlugin):
    def __init__(self):
        self.means: list[float] = []
        self.manifest = PluginManifest(
            id="rec", languages=["en"], model="fake", license="MIT", priority=100
        )

    def transcribe(self, audio, sample_rate, ctx):
        m = round(float(np.mean(audio)), 2)
        self.means.append(m)
        text = "alpha beta gamma delta" if m > 0 else "one two three four"
        return PluginResult(text=text, raw_confidence=0.9, language="en")


def run(registry, extractor, options=None):
    recorder = Recorder()
    registry.register(recorder, check_available=False)
    pipeline = Pipeline(
        registry=registry,
        diarizer=AliceUnderBob(),
        language_id=FakeLanguageID([{"en": 0.99, "zh": 0.01}]),
        extractor=extractor,
    )
    mixture = np.full(10 * SR, 0.5, dtype=np.float32)
    mixture[6 * SR :] = -0.5
    transcript = pipeline.run("m.wav", options or PipelineOptions(summarize=False), audio=mixture)
    return transcript, recorder.means


def test_overlapped_and_hidden_speech_use_the_extracted_voice(registry):
    transcript, means = run(registry, FakeExtractor())
    # Transcribed in order T0001 (Alice, overlapped), T0002 (Bob, alone), H0001 (Bob under
    # Alice). Extracted audio is peak-normalized to 0.9; Bob's clean turn stays the mixture.
    assert means == [0.9, -0.5, -0.9]
    assert [(t.id, t.speaker) for t in transcript.turns] == [
        ("T0001", "A"),
        ("H0001", "B"),
        ("T0002", "B"),
    ]


def test_a_failing_extractor_falls_back_to_the_original_audio(registry, caplog):
    extractor = FakeExtractor(fail=True)
    transcript, means = run(registry, extractor)
    assert extractor.calls == 1  # gave up after the first failure
    assert means[:2] == [0.5, -0.5]  # mixture for the regular turns
    assert [t.id for t in transcript.turns] == ["T0001", "T0002"]  # hidden: just an echo
    assert "Speaker extraction failed" in caplog.text


def turn(i, spk, a, b, overlap=False):
    return Turn(id=f"T{i}", speaker=spk, start=a, end=b, overlap=overlap)


def test_voice_samples_prefer_clean_turns_then_fall_back():
    audio = np.arange(20 * SR, dtype=np.float32)
    turns = [
        turn(1, "A", 0, 2),  # clean
        turn(2, "A", 2, 6, overlap=True),
        turn(3, "A", 6, 9),  # clean, and longer: 5 s of clean speech in all
        turn(4, "B", 9, 11, overlap=True),  # B never talks alone
        turn(5, "B", 11, 11.2),  # too short to use
    ]
    samples = voice_samples(audio, turns)
    a, a_clean = samples["A"]
    assert a_clean and len(a) == 5 * SR and a[0] == 6 * SR  # longest clean turn first
    b, b_clean = samples["B"]
    assert not b_clean and len(b) == 2 * SR


def test_a_voice_file_is_used_as_the_sample():
    voice = np.ones(30 * SR, np.float32)
    samples = voice_samples(np.zeros(SR, np.float32), [turn(1, "A", 0, 1)], {"A": voice})
    sample, clean = samples["A"]
    assert clean and len(sample) == 20 * SR  # cut to 20 s


def test_extraction_setting(monkeypatch, tmp_path):
    from speakerlens.config import load_settings
    from speakerlens.extraction import TargetExtractor
    from speakerlens.pipeline import make_extractor

    model = tmp_path / "wesep"
    model.mkdir()
    for key in ("SPEAKER_EXTRACTION", "TSE_MODEL", "SPEAKERLENS_CACHE_DIR"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)

    def build(mode, tse=None):
        monkeypatch.setenv("SPEAKER_EXTRACTION", mode)
        if tse:
            monkeypatch.setenv("TSE_MODEL", str(tse))
        return make_extractor(load_settings(tmp_path / "none.env"), "cpu")

    default = build("auto")  # no TSE_MODEL: downloads into the cache on first use
    assert isinstance(default, TargetExtractor) and default.download
    assert default.model_dir.name == "wesep_bsrnn"
    assert build("off") is None
    assert build("auto", model) is None  # a TSE_MODEL folder without the model files
    (model / "avg_model.pt").write_bytes(b"x")
    (model / "config.yaml").write_text("x: 1")
    mine = build("auto", model)
    assert isinstance(mine, TargetExtractor) and not mine.download  # loads lazily
    with pytest.raises(ValueError):
        build("sometimes", model)


def _archive(tmp_path, members):
    import io
    import tarfile

    path = tmp_path / "model.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path.as_uri()


def test_download_takes_only_the_model_files(tmp_path, monkeypatch):
    from speakerlens.extraction import download_model, model_ready

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    url = _archive(
        tmp_path,
        {
            "bsrnn/avg_model.pt": b"weights",
            "bsrnn/config.yaml": b"model: {}",
            "../escape.txt": b"nope",
            "bsrnn/README": b"hi",
        },
    )
    target = tmp_path / "cache" / "wesep"
    assert download_model(target, url) == target and model_ready(target)
    assert sorted(p.name for p in target.iterdir()) == ["avg_model.pt", "config.yaml"]
    assert (target / "avg_model.pt").read_bytes() == b"weights"
    assert not (tmp_path / "cache" / "escape.txt").exists()
    assert not (tmp_path / "escape.txt").exists()


def test_download_refuses_offline_and_incomplete_archives(tmp_path, monkeypatch):
    from speakerlens.extraction import download_model

    url = _archive(tmp_path, {"avg_model.pt": b"weights"})  # no config.yaml
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(RuntimeError, match="offline"):
        download_model(tmp_path / "a", url)
    monkeypatch.delenv("HF_HUB_OFFLINE")
    with pytest.raises(RuntimeError, match="config.yaml"):
        download_model(tmp_path / "b", url)
    assert not (tmp_path / "b" / "avg_model.pt").exists()  # nothing half-installed


def test_vendored_model_code_is_complete_and_committed():
    """The BSRNN model and every vendored module it imports exist and aren't git-ignored, so
    a fresh clone can run speaker extraction."""
    vendor = Path(extraction.VENDOR)
    imports = re.compile(r"^\s*(?:from|import) ((?:wesep|wespeaker)[\w.]*)", re.MULTILINE)
    found, missing, todo = set(), [], ["wesep.models.bsrnn"]
    while todo:
        name = todo.pop()
        base = vendor.joinpath(*name.split("."))
        options = (base.with_suffix(".py"), base / "__init__.py")
        path = next((p for p in options if p.exists()), None)
        if path is None:
            missing.append(name)
        elif path not in found:
            found.add(path)
            todo += imports.findall(path.read_text())
            todo += [name.rsplit(".", k)[0] for k in range(1, name.count(".") + 1)]
    assert not missing, missing

    root = vendor.parents[2]
    if subprocess.run(["git", "-C", str(root), "rev-parse"], capture_output=True).returncode:
        pytest.skip("not a git checkout")
    check = ["git", "-C", str(root), "check-ignore", *map(str, sorted(found))]
    ignored = subprocess.run(check, capture_output=True, text=True).stdout.split()
    assert not ignored, f"git ignores vendored code: {ignored}"
