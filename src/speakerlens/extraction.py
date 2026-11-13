"""Target speaker extraction (TSE): pull one person's voice out of overlapped speech.

Given a voice sample of a person, the model keeps that person's voice and suppresses the
others. It is applied only to turns where someone else talks at the same time.

Model: WeSep BSRNN + ECAPA, trained on VoxCeleb1 mixtures (code in _vendor/, Apache-2.0;
checkpoint CC-BY-4.0). It is downloaded from ModelScope on first use (250 MB) into the
cache. TSE_MODEL can point to a local folder holding avg_model.pt and config.yaml instead.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from speakerlens.audio import SAMPLE_RATE

log = logging.getLogger(__name__)

VENDOR = Path(__file__).resolve().parent / "_vendor"
MODEL_URL = (
    "https://www.modelscope.cn/datasets/wenet/wesep_pretrained_models/resolve/master/"
    "bsrnn_ecapa_vox1.tar.gz"
)
MODEL_FILES = ("avg_model.pt", "config.yaml")


def model_ready(model_dir: str | Path | None) -> bool:
    """True when `model_dir` holds a checkpoint (avg_model.pt) and its config.yaml."""
    if not model_dir:
        return False
    d = Path(model_dir).expanduser()
    return (d / "avg_model.pt").is_file() and (d / "config.yaml").is_file()


def download_model(model_dir: str | Path, url: str = MODEL_URL) -> Path:
    """Download and unpack the checkpoint into `model_dir`, unless it's already there.
    Only the two model files are taken from the archive, by name, so nothing can be
    written anywhere else. Offline (HF_HUB_OFFLINE=1 or SPEAKERLENS_OFFLINE=1), it never
    downloads."""
    import os
    import tarfile
    import tempfile
    import urllib.request

    model_dir = Path(model_dir).expanduser()
    if model_ready(model_dir):
        return model_dir
    if os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("SPEAKERLENS_OFFLINE") == "1":
        raise RuntimeError(
            f"the extraction model isn't in {model_dir} and downloads are off (offline mode); "
            "run once online, or set TSE_MODEL"
        )
    model_dir.mkdir(parents=True, exist_ok=True)
    log.info("Downloading the speaker extraction model (250 MB, once) to %s", model_dir)
    with tempfile.TemporaryDirectory(dir=model_dir) as tmp:
        archive = Path(tmp) / "model.tar.gz"
        urllib.request.urlretrieve(url, archive)  # noqa: S310 - fixed https URL
        with tarfile.open(archive) as tar:
            found = {}
            for member in tar.getmembers():
                name = Path(member.name).name
                if member.isfile() and name in MODEL_FILES and name not in found:
                    found[name] = member
            missing = set(MODEL_FILES) - set(found)
            if missing:
                raise RuntimeError(f"the downloaded archive has no {', '.join(sorted(missing))}")
            for name, member in found.items():
                source = tar.extractfile(member)
                assert source is not None
                (Path(tmp) / name).write_bytes(source.read())
        for name in MODEL_FILES:  # move into place only once both are complete
            (Path(tmp) / name).replace(model_dir / name)
    return model_dir


def _import_vendor() -> None:
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))


class TargetExtractor:
    """Loads lazily on first use, so building a pipeline stays cheap."""

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "cpu",
        window: float = 10.0,
        fade: float = 0.5,
        download: bool = False,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser()
        self.device = device
        self.download = download  # fetch the model on first use if it isn't there
        self.window = window
        self.fade = fade
        self._model: Any = None

    @property
    def model(self) -> Any:
        if self._model is None:
            import torch
            import yaml

            if self.download:
                download_model(self.model_dir)
            _import_vendor()
            from wesep.models.bsrnn import BSRNN

            configs = yaml.safe_load((self.model_dir / "config.yaml").read_text())
            if configs["model"]["tse_model"] != "BSRNN":
                raise RuntimeError(f"Only BSRNN is supported, not {configs['model']['tse_model']}")
            args = dict(configs["model_args"]["tse_model"])
            args["spk_model_init"] = False  # weights come from the checkpoint
            model = BSRNN(**args)
            states = torch.load(
                self.model_dir / "avg_model.pt", map_location="cpu", weights_only=True
            )
            state = (
                states["models"][0] if isinstance(states, dict) and "models" in states else states
            )
            model.load_state_dict(state)
            self._model = model.eval().to(self.device)
        return self._model

    def prepare(self, sample: np.ndarray) -> Any:
        """The voice sample as the model takes it: raw audio, or (for checkpoints trained with
        `spk_feat: true`, like bsrnn_ecapa_vox1) 80-band filterbank features with mean
        normalization, computed like WeSep's own extractor (cli/extractor.py)."""
        import torch

        wav = torch.from_numpy(np.ascontiguousarray(sample, np.float32)).unsqueeze(0)
        if getattr(self.model, "spk_feat", False):
            import torchaudio.compliance.kaldi as kaldi

            feat = kaldi.fbank(
                wav, num_mel_bins=80, frame_length=25, frame_shift=10, sample_frequency=SAMPLE_RATE
            )
            wav = (feat - feat.mean(0)).unsqueeze(0)  # (1, frames, 80)
        return wav.to(self.device)

    def extract(self, clip: np.ndarray, prepared: Any) -> np.ndarray:
        """The sampled person's voice in `clip`, in `window`-second pieces joined with short
        crossfades (long clips don't fit the model at once)."""
        import torch

        model = self.model
        n, win = len(clip), int(self.window * SAMPLE_RATE)
        nf = int(self.fade * SAMPLE_RATE)
        hop = win - nf
        out = np.zeros(n, np.float32)
        weight = np.zeros(n, np.float32)
        ramp = np.ones(win, np.float32)
        ramp[:nf] = np.linspace(0, 1, nf, dtype=np.float32)
        ramp[-nf:] = np.linspace(1, 0, nf, dtype=np.float32)
        with torch.no_grad():
            for start in range(0, max(n - nf, 1), hop):
                piece = clip[start : start + win]
                short = len(piece)
                if short < SAMPLE_RATE // 2:  # too short for the model: pad
                    piece = np.pad(piece, (0, SAMPLE_RATE // 2 - short))
                mix = torch.from_numpy(np.ascontiguousarray(piece, np.float32)).unsqueeze(0)
                y = model(mix.to(self.device), prepared)
                y = (y[0] if isinstance(y, (list, tuple)) else y).squeeze(0).cpu().numpy()
                m = min(short, len(y), n - start)
                w = ramp[:m] if m == win else np.ones(m, np.float32)
                out[start : start + m] += y[:m] * w
                weight[start : start + m] += w
        return out / np.maximum(weight, 1e-6)
