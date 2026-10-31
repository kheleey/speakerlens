"""Fine-tune the speaker extraction model on real meetings (AliMeeting Train).

The released model is trained on simulated mixtures. AliMeeting's training set has real
recordings of both sides: the table microphone and each person's headset. Training examples,
made on the fly:
    mixture     4 s of the table microphone, while the target person talks
    target      the same 4 s of that person's headset, aligned to the table mic
    voice       4-8 s of the person's table-mic speech with nobody else talking, from
                elsewhere in the meeting, as the pipeline builds voice samples
Loss: negative SI-SDR. The speaker encoder stays frozen by default.

Run (prepare the data first with training/prepare_ali_train.py; --init is the released
model, which speakerlens downloads to ~/.cache/speakerlens/models/wesep_bsrnn on first use):
    INIT=~/.cache/speakerlens/models/wesep_bsrnn
    python training/train_tse.py data/ali_train --init $INIT --index-only
    CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun --nproc_per_node 6 training/train_tse.py \\
        data/ali_train --init $INIT --out models/tse_ft_run
Writes <out>/avg_model.pt + config.yaml (the best checkpoint by held-out SI-SDR) and a
snapshot every 5000 steps (<out>/step_N/); each folder is usable as TSE_MODEL. The model in
the README is step_20000 with these defaults.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np

SR = 16_000
MARGIN = 4000  # samples (0.25 s) searched each way when lining up headset and table mic
MIN_QUALITY = 12.0  # alignment peak quality needed to use an example (no match scores ~6)


def clean_parts(
    segments: list[tuple[str, float, float, str]], speaker: str
) -> list[tuple[float, float]]:
    """The parts of `speaker`'s speech where nobody else talks."""
    others = sorted((a, b) for spk, a, b, _ in segments if spk != speaker)
    out = []
    for spk, a, b, _ in segments:
        if spk != speaker:
            continue
        pieces = [(a, b)]
        for oa, ob in others:
            if ob <= a or oa >= b:
                continue
            pieces = [
                part
                for x, y in pieces
                for part in ((x, min(y, oa)), (max(x, ob), y))
                if part[1] - part[0] > 0
            ]
        out += pieces
    return out


def read_stm(path: Path) -> list[tuple[str, float, float, str]]:
    """(speaker, start, end, text) per line."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.split(maxsplit=5)
        if len(p) >= 5 and not line.startswith(";;"):
            out.append((p[2], float(p[3]), float(p[4]), p[5] if len(p) == 6 else ""))
    return sorted(out, key=lambda x: x[1])


def estimate_lag(near: np.ndarray, far: np.ndarray, max_lag: int = 1600) -> int:
    """Samples by which the table mic lags the headset (cross-correlation)."""
    n = min(len(near), len(far))
    a, b = near[:n] - near[:n].mean(), far[:n] - far[:n].mean()
    size = 1 << (2 * n - 1).bit_length()
    xc = np.fft.irfft(np.fft.rfft(b, size) * np.conj(np.fft.rfft(a, size)), size)
    lags = np.concatenate([xc[: max_lag + 1], xc[-max_lag:]])
    best = int(np.argmax(np.abs(lags)))
    return best if best <= max_lag else best - len(lags)


def align(mix: np.ndarray, ext: np.ndarray, n_lags: int) -> tuple[int, float]:
    """Where `mix` best lines up inside the longer `ext`: (offset into ext, peak quality).
    GCC-PHAT cross-correlation (sharp peaks even with room echo); quality is the peak
    over the median correlation, so a clear match scores high and no match scores ~1-3."""
    size = 1 << (len(mix) + len(ext)).bit_length()
    cross = np.fft.rfft(ext, size) * np.conj(np.fft.rfft(mix, size))
    cc = np.abs(np.fft.irfft(cross / (np.abs(cross) + 1e-12), size)[:n_lags])
    best = int(np.argmax(cc))
    return best, float(cc[best] / (np.median(cc) + 1e-12))


def build_index(data: Path) -> list[dict]:
    """Meetings with table mic, headsets and transcript; per person: their stretches,
    clean parts and headset lag. Cached in <data>/index.json."""
    import soundfile as sf

    cache = data / "index.json"
    if cache.exists():
        index = json.loads(cache.read_text())
        if index:
            return index
        cache.unlink()  # empty: written before the data was prepared
    index = []
    for stm in sorted((data / "ref").glob("*.stm")):
        session = stm.stem
        far_path = data / "far" / f"{session}.wav"
        if not far_path.exists():
            continue
        segments = read_stm(stm)
        people = {}
        for spk in sorted({s for s, *_ in segments}):
            near_path = data / "near" / f"{session}_{spk}.wav"
            clean = [p for p in clean_parts(segments, spk) if p[1] - p[0] >= 1.0]
            if not near_path.exists() or not clean:
                continue
            a, b = max(clean, key=lambda p: p[1] - p[0])
            a, b = a, min(b, a + 10.0)
            near, _ = sf.read(near_path, start=int(a * SR), stop=int(b * SR), dtype="float32")
            far, _ = sf.read(far_path, start=int(a * SR), stop=int(b * SR), dtype="float32")
            people[spk] = {
                "near": str(near_path),
                "lag": estimate_lag(near, far),
                "turns": [(st, en) for s, st, en, _ in segments if s == spk and en - st >= 1.0],
                "clean": clean,
            }
        if people:
            index.append(
                {
                    "session": session,
                    "far": str(far_path),
                    "people": people,
                    "segments": [(s, st, en) for s, st, en, _ in segments],
                }
            )
        print(f"  indexed {session}: {len(people)} people", flush=True)
    if not index:
        raise SystemExit(
            f"No meetings with table mic, headsets and transcript in {data}: has "
            "training/prepare_ali_train.py finished for both archives?"
        )
    cache.write_text(json.dumps(index))
    return index


def check_alignment(index: list[dict], seconds: float, tries: int = 300) -> None:
    """How many random windows line up clearly, and the shifts found."""
    import soundfile as sf

    rng = random.Random(7)
    n = int(seconds * SR)
    qualities, shifts = [], []
    while len(qualities) < tries:
        meeting = rng.choice(index)
        spk = rng.choice(sorted(meeting["people"]))
        person = meeting["people"][spk]
        st, en = rng.choice(person["turns"])
        a = max(0.0, rng.uniform(st, en) - seconds / 2)
        t0 = int(a * SR) - person["lag"] - MARGIN
        if t0 < 0:
            continue
        mix, _ = sf.read(meeting["far"], start=int(a * SR), frames=n, dtype="float32")
        ext, _ = sf.read(person["near"], start=t0, frames=n + 2 * MARGIN, dtype="float32")
        if len(mix) < n or len(ext) < n + 2 * MARGIN:
            continue
        offset, quality = align(mix, ext, 2 * MARGIN + 1)
        qualities.append(quality)
        if quality >= MIN_QUALITY:
            shifts.append(int(a * SR) - (t0 + offset))  # table-mic sample - headset sample
    q = np.array(qualities)
    print(
        f"{len(q)} windows: {np.mean(q >= MIN_QUALITY):.0%} line up clearly "
        f"(quality >= {MIN_QUALITY:g}); median quality {np.median(q):.1f}"
    )
    if shifts:
        s = np.sort(shifts)
        print(
            f"shift (samples) 10% / 50% / 90%: {s[len(s) // 10]} / {s[len(s) // 2]} / "
            f"{s[9 * len(s) // 10]}"
        )


def fbank(wave: np.ndarray):  # type: ignore[no-untyped-def]
    """Voice features exactly as speakerlens.extraction.TargetExtractor.prepare makes them."""
    import torch
    import torchaudio.compliance.kaldi as kaldi

    feat = kaldi.fbank(
        torch.from_numpy(wave).unsqueeze(0),
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        sample_frequency=SR,
    )
    return feat - feat.mean(0)


def read_pieces(
    path: str, spans: list, want: float, rng: random.Random, shortest: float = 0.5
) -> list[np.ndarray]:
    """Random pieces of `spans` from `path`, about `want` seconds in all. Pieces under
    `shortest` seconds, or past the end of the file, are skipped."""
    import soundfile as sf

    spans = list(spans)
    rng.shuffle(spans)
    pieces, got = [], 0.0
    for x, y in spans:
        if got >= want:
            break
        y = min(y, x + want - got)
        if y - x < shortest:
            continue
        piece, _ = sf.read(path, start=int(x * SR), stop=int(y * SR), dtype="float32")
        if len(piece) >= shortest * SR:
            pieces.append(piece)
            got += len(piece) / SR
    return pieces


def make_example(  # type: ignore[no-untyped-def]
    meeting: dict,
    rng: random.Random,
    seconds: float,
    voice_seconds: float,
    contaminate: float = 0.0,
):
    """One (mixture, target, voice features) example, or None to draw again."""
    import soundfile as sf

    spk = rng.choice(sorted(meeting["people"]))
    person = meeting["people"][spk]
    st, en = rng.choice(person["turns"])
    centre = rng.uniform(st, en)
    a = max(0.0, centre - seconds / 2)
    b = a + seconds
    talking = sum(max(0.0, min(b, e) - max(a, s)) for x, s, e in meeting["segments"] if x == spk)
    if talking < 1.0:
        return None
    n = int(seconds * SR)
    mix, _ = sf.read(meeting["far"], start=int(a * SR), frames=n, dtype="float32")
    # The headset and the table mic aren't sample-synchronized and drift apart, so line up
    # each window on its own: search +-MARGIN around the person's rough shift.
    t0 = int(a * SR) - person["lag"] - MARGIN
    if t0 < 0:
        return None
    ext, _ = sf.read(person["near"], start=t0, frames=n + 2 * MARGIN, dtype="float32")
    if len(mix) < n or len(ext) < n + 2 * MARGIN or float(np.abs(ext).max()) < 1e-3:
        return None
    offset, quality = align(mix, ext, 2 * MARGIN + 1)
    if quality < MIN_QUALITY:
        return None  # no clear match: skip rather than train on a misaligned target
    target = ext[offset : offset + n]
    pool = [(x, y) for x, y in person["clean"] if y < a - 1.0 or x > b + 1.0]
    if not pool:
        return None
    want = rng.uniform(voice_seconds / 2, voice_seconds)
    pieces = read_pieces(meeting["far"], pool, want, rng)
    if rng.random() < contaminate:
        # Like the pipeline's voice samples, which sometimes include a turn the speaker
        # labels gave the wrong person: part of the sample is someone else.
        others = [p for k, p in meeting["people"].items() if k != spk and p["clean"]]
        if others:
            share = rng.uniform(0.2, 0.4)
            pieces += read_pieces(meeting["far"], rng.choice(others)["clean"], want * share, rng)
            rng.shuffle(pieces)
    if sum(len(x) for x in pieces) < SR:  # under 1 s of usable voice
        return None
    voice = np.concatenate(pieces)
    reps = int(np.ceil(voice_seconds * SR / max(len(voice), 1)))
    voice = np.tile(voice, reps)[: int(voice_seconds * SR)]  # fixed length for batching
    return mix, target, fbank(voice)


def batches(  # type: ignore[no-untyped-def]
    index: list[dict],
    batch: int,
    seconds: float,
    voice_seconds: float,
    seed: int,
    contaminate: float = 0.0,
):
    import torch

    rng = random.Random(seed)
    weights = [sum(len(p["turns"]) for p in m["people"].values()) for m in index]
    bad = 0
    while True:
        out = []
        while len(out) < batch:
            try:
                ex = make_example(
                    rng.choices(index, weights)[0], rng, seconds, voice_seconds, contaminate
                )
            except Exception as exc:  # one odd recording must not stop hours of training
                bad += 1
                if bad <= 5:
                    print(f"  skipped an example ({type(exc).__name__}: {exc})", flush=True)
                continue
            if ex is not None:
                out.append(ex)
        yield (
            torch.from_numpy(np.stack([e[0] for e in out])),
            torch.from_numpy(np.stack([e[1] for e in out])),
            torch.stack([e[2] for e in out]),
        )


def si_sdr(est, ref, eps: float = 1e-8):  # type: ignore[no-untyped-def]
    est = est - est.mean(-1, keepdim=True)
    ref = ref - ref.mean(-1, keepdim=True)
    scale = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + eps)
    target = scale * ref
    noise = est - target
    return 10 * (target.pow(2).sum(-1) + eps).log10() - 10 * (noise.pow(2).sum(-1) + eps).log10()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("data", type=Path)
    ap.add_argument("--init", type=Path, required=True, help="Starting model folder")
    ap.add_argument("--out", type=Path, default=Path("models/tse_ft_run"))
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--batch", type=int, default=4, help="Per GPU")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--voice-seconds", type=float, default=6.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--train-speaker-encoder", action="store_true")
    ap.add_argument("--val-meetings", type=int, default=8, help="Held out from training")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--val-examples", type=int, default=48)
    ap.add_argument(
        "--contaminate",
        type=float,
        default=0.3,
        help="Share of training voice samples that partly come from someone else",
    )
    ap.add_argument(
        "--snapshot-every",
        type=int,
        default=5000,
        help="Also keep the model every N steps as <out>/step_N/ (a ready model folder)",
    )
    ap.add_argument(
        "--index-only", action="store_true", help="Just build <data>/index.json (one process)"
    )
    ap.add_argument(
        "--check-alignment",
        action="store_true",
        help="Report how well headset and table mic line up on 300 random windows, then stop",
    )
    args = ap.parse_args()

    import torch
    import torch.distributed as dist

    from speakerlens.extraction import TargetExtractor

    ddp = "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"
    if ddp:
        dist.init_process_group("nccl" if device.startswith("cuda") else "gloo")
    main_rank = rank == 0

    if main_rank:
        index = build_index(args.data)
        if args.check_alignment:
            check_alignment(index, args.seconds)
            return
        if args.index_only:
            n = sum(len(m["people"]) for m in index)
            print(f"Index: {len(index)} meetings, {n} people, in {args.data / 'index.json'}")
            return
    if ddp:
        dist.barrier()
    index = json.loads((args.data / "index.json").read_text())
    random.Random(0).shuffle(index)
    val, train = index[: args.val_meetings], index[args.val_meetings :]
    if main_rank:
        print(f"{len(train)} meetings for training, {len(val)} held out", flush=True)

    model = TargetExtractor(args.init, device).model
    if not args.train_speaker_encoder:
        for p in model.spk_model.parameters():
            p.requires_grad = False

    def train_mode(m) -> None:  # type: ignore[no-untyped-def]
        m.train()
        if not args.train_speaker_encoder:
            m.spk_model.eval()  # frozen: its normalization statistics must not drift either

    train_mode(model)
    params = [p for p in model.parameters() if p.requires_grad]
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local] if device.startswith("cuda") else None
        )
    net = model.module if ddp else model
    opt = torch.optim.Adam(params, lr=args.lr)
    warm = min(500, args.steps // 10)

    def lr_at(step: int) -> float:
        if step < warm:
            return args.lr * (step + 1) / warm
        progress = (step - warm) / max(1, args.steps - warm)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

    # A fixed held-out set, so scores are comparable across checks.
    val_batches = []
    gen = batches(val, args.batch, args.seconds, args.voice_seconds, seed=1234)
    for _ in range(max(1, args.val_examples // args.batch)):
        val_batches.append(next(gen))

    def evaluate() -> float:
        net.eval()
        gains = []
        with torch.no_grad():
            for mix, tgt, feat in val_batches:
                mix, tgt, feat = mix.to(device), tgt.to(device), feat.to(device)
                est = net(mix, feat)[0]
                gains.append((si_sdr(est, tgt) - si_sdr(mix, tgt)).mean().item())
        train_mode(net)
        return float(np.mean(gains))

    args.out.mkdir(parents=True, exist_ok=True)
    best = evaluate() if main_rank else 0.0
    if main_rank:
        print(f"step 0: held-out SI-SDR improvement {best:.2f} dB (starting model)", flush=True)
        shutil.copy(args.init / "config.yaml", args.out / "config.yaml")
        torch.save({"models": [net.state_dict()]}, args.out / "avg_model.pt")
    data = batches(
        train,
        args.batch,
        args.seconds,
        args.voice_seconds,
        seed=rank + 1,
        contaminate=args.contaminate,
    )
    t0, losses = time.time(), []
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        mix, tgt, feat = (x.to(device) for x in next(data))
        est = model(mix, feat)[0]
        loss = -si_sdr(est, tgt).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        losses.append(loss.item())
        if main_rank and args.snapshot_every and step % args.snapshot_every == 0:
            snap = args.out / f"step_{step}"
            snap.mkdir(parents=True, exist_ok=True)
            shutil.copy(args.init / "config.yaml", snap / "config.yaml")
            torch.save({"models": [net.state_dict()]}, snap / "avg_model.pt")
        if step % args.eval_every == 0 or step == args.steps:
            if ddp:
                dist.barrier()
            if main_rank:
                score = evaluate()
                note = ""
                if score > best:
                    best = score
                    torch.save({"models": [net.state_dict()]}, args.out / "avg_model.pt")
                    note = "  (saved)"
                print(
                    f"step {step}: train loss {np.mean(losses):.2f}, held-out SI-SDR "
                    f"improvement {score:.2f} dB, {(time.time() - t0) / 60:.0f} min{note}",
                    flush=True,
                )
                losses = []
            if ddp:
                dist.barrier()
    if main_rank:
        print(f"Best held-out SI-SDR improvement {best:.2f} dB; model in {args.out}")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
