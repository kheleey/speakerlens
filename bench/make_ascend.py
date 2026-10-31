"""Turn the ASCEND dataset (Mandarin-English code-switching) into test recordings.

ASCEND on Hugging Face stores single-utterance clips. This script rebuilds:

  real      full two-person conversations, utterances placed back at their original
            times when the clip ids contain them (natural turn-taking and overlap)
  meetings  simulated 3-4 person meetings from different people's clips, with some
            deliberate overlap

Each set is written in the layout bench/run_bench.py expects:

    data/ascend_real/audio/<id>.wav   data/ascend_real/ref/<id>.rttm   ref/<id>.stm

Usage:
    python bench/make_ascend.py --src data/ascend/main --split train --out data/dev \
        --max 20 --meetings 10
    python bench/make_ascend.py --inspect      # just show the dataset's columns
"""

from __future__ import annotations

import argparse
import io
import random
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 16_000
# ASCEND clip ids usually end in "_<start>_<duration>", e.g. ses2_spk3_L13101_189.900_5.490
_TIMES = re.compile(r"_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)$")


@dataclass
class Utt:
    session: str
    speaker: str
    topic: str
    text: str
    audio: np.ndarray
    start: float | None  # original start time in the session, if known
    order: int


def decode(audio_bytes: bytes) -> np.ndarray:
    """Decode wav/flac/mp3 bytes to 16 kHz mono float32."""
    try:
        import soundfile as sf

        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
        data = data.mean(axis=1)
        if sr == SR:
            return data
    except Exception:
        pass
    proc = subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            str(SR),
            "pipe:1",
        ],
        input=audio_bytes,
        capture_output=True,
        check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def pick(columns: list[str], *names: str) -> str:
    for name in names:
        if name in columns:
            return name
    raise SystemExit(
        f"None of {names} found. Columns are: {columns}. Run with --inspect and send me the output."
    )


def load_split(src: Path, split: str) -> list[Utt]:
    import pandas as pd

    files = sorted(src.glob(f"{split}-*.parquet"))
    if not files:
        raise SystemExit(f"No {split}-*.parquet files in {src}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    cols = list(df.columns)
    c_audio = pick(cols, "audio")
    c_text = pick(cols, "transcription", "text", "sentence")
    c_sess = pick(cols, "session_id", "session")
    c_spk = pick(cols, "original_speaker_id", "speaker_id", "speaker")
    c_topic = "topic" if "topic" in cols else None
    # Original times are in the clip file name, e.g. .../ses1_spk17_L3818_9.3200_0.6400.wav
    c_times = [c for c in ("path", "id") if c in cols]

    utts = []
    for i, row in enumerate(df.itertuples(index=False)):
        rec = row._asdict()
        audio = rec[c_audio]
        raw = audio.get("bytes") if isinstance(audio, dict) else None
        if not raw:
            continue
        start = None
        for c in c_times:
            m = _TIMES.search(Path(str(rec[c])).stem)
            if m:
                start = float(m.group(1))
                break
        utts.append(
            Utt(
                session=str(rec[c_sess]),
                speaker=f"spk{rec[c_spk]}",
                topic=str(rec[c_topic]) if c_topic else "",
                text=str(rec[c_text]).strip(),
                audio=decode(raw),
                start=start,
                order=i,
            )
        )
    return utts


@dataclass
class Placed:
    speaker: str
    start: float
    utt: Utt


def mix(placed: list[Placed]) -> np.ndarray:
    end = max(p.start + len(p.utt.audio) / SR for p in placed)
    out = np.zeros(int(end * SR) + SR, dtype=np.float32)
    for p in placed:
        i = int(p.start * SR)
        out[i : i + len(p.utt.audio)] += p.utt.audio
    peak = float(np.abs(out).max()) or 1.0
    return out / max(peak, 1.0) * 0.95 if peak > 1.0 else out


def write_set(name: str, placed: list[Placed], root: Path) -> float:
    import soundfile as sf

    (root / "audio").mkdir(parents=True, exist_ok=True)
    (root / "ref").mkdir(parents=True, exist_ok=True)
    audio = mix(placed)
    sf.write(root / "audio" / f"{name}.wav", audio, SR)
    placed = sorted(placed, key=lambda p: p.start)
    with open(root / "ref" / f"{name}.rttm", "w", encoding="utf-8") as f:
        for p in placed:
            dur = len(p.utt.audio) / SR
            f.write(f"SPEAKER {name} 1 {p.start:.3f} {dur:.3f} <NA> <NA> {p.speaker} <NA> <NA>\n")
    with open(root / "ref" / f"{name}.stm", "w", encoding="utf-8") as f:
        for p in placed:
            end = p.start + len(p.utt.audio) / SR
            f.write(f"{name} 1 {p.speaker} {p.start:.3f} {end:.3f} {p.utt.text}\n")
    return len(audio) / SR


def overlap_share(placed: list[Placed]) -> float:
    """Share of speech time where two clips overlap (a sign of broken timestamps)."""
    spans = sorted((p.start, p.start + len(p.utt.audio) / SR) for p in placed)
    total = sum(e - s for s, e in spans) or 1.0
    over, reach = 0.0, 0.0
    for s, e in spans:
        if s < reach:
            over += min(e, reach) - s
        reach = max(reach, e)
    return over / total


def _spans(items: list[Utt]) -> list[tuple[float, float]]:
    return sorted((u.start, u.start + len(u.audio) / SR) for u in items)  # type: ignore[misc]


def _overlap_seconds(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    total, j = 0.0, 0
    for s, e in a:
        while j < len(b) and b[j][1] <= s:
            j += 1
        k = j
        while k < len(b) and b[k][0] < e:
            total += max(0.0, min(e, b[k][1]) - max(s, b[k][0]))
            k += 1
    return total


def find_pairs(
    utts: list[Utt], max_overlap: float = 0.1, min_shared_span: float = 0.5
) -> list[tuple[str, str, str, str, list[Utt]]]:
    """Find ASCEND conversations.

    session_id is a recording round, not a conversation: each speaker has their own
    recording per round. Two speakers form a conversation when they share a round and a
    topic, their recordings cover the same stretch of time, and they take turns (their
    clips barely overlap). Unrelated speakers on the same timeline overlap ~50% of the
    time, partners only a few percent, so the two are easy to tell apart.
    """
    groups: dict[tuple[str, str], dict[str, list[Utt]]] = defaultdict(lambda: defaultdict(list))
    for u in utts:
        if u.start is not None:
            groups[(u.session, u.topic)][u.speaker].append(u)

    found = []
    for (session, topic), by_spk in groups.items():
        spans = {k: _spans(v) for k, v in by_spk.items() if len(v) >= 10}
        candidates = []
        names = sorted(spans)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                sa, sb = spans[a], spans[b]
                shared = min(sa[-1][1], sb[-1][1]) - max(sa[0][0], sb[0][0])
                shorter = min(sa[-1][1] - sa[0][0], sb[-1][1] - sb[0][0]) or 1.0
                if shared / shorter < min_shared_span:
                    continue
                speech = min(sum(e - s for s, e in sa), sum(e - s for s, e in sb)) or 1.0
                ratio = _overlap_seconds(sa, sb) / speech
                if ratio <= max_overlap:
                    candidates.append((ratio, a, b))
        used: set[str] = set()
        for _ratio, a, b in sorted(candidates):
            if a in used or b in used:
                continue
            used |= {a, b}
            found.append((session, topic, a, b, by_spk[a] + by_spk[b]))
    found.sort(key=lambda c: -len(c[4]))
    return found


def build_real(
    utts: list[Utt], max_convs: int, max_minutes: float, max_gap: float = 2.0
) -> dict[str, list[Placed]]:
    pairs = find_pairs(utts)
    paired = {(s, sp) for s, _t, a, b, _u in pairs for sp in (a, b)}
    alone = {(u.session, u.speaker) for u in utts} - paired
    print(f"  found {len(pairs)} two-person conversations; {len(alone)} recordings have no partner")
    out = {}
    for session, _topic, a, b, items in pairs[:max_convs]:
        # Original timing; silences longer than max_gap are shortened to 1 s.
        items = sorted(items, key=lambda u: u.start)  # type: ignore[arg-type, return-value]
        placed: list[Placed] = []
        shift, prev_end = items[0].start, items[0].start
        for u in items:
            gap = u.start - prev_end  # type: ignore[operator]
            if gap > max_gap:
                shift += gap - 1.0
            start = u.start - shift  # type: ignore[operator]
            if start > max_minutes * 60:
                break
            placed.append(Placed(u.speaker, start, u))
            prev_end = max(prev_end, u.start + len(u.audio) / SR)  # type: ignore[operator]
        out[f"ascend_ses{session}_{a}_{b}"] = placed
    return out


def build_meetings(
    utts: list[Utt], count: int, speakers: int, minutes: float, overlap: float, seed: int
) -> dict[str, list[Placed]]:
    rng = random.Random(seed)
    # ASCEND speaker ids name people, and most people were recorded in several rounds
    # (sessions). Each person contributes clips from one round only, so a meeting never has
    # the same voice under two labels.
    rounds: dict[str, dict[str, list[Utt]]] = defaultdict(lambda: defaultdict(list))
    for u in utts:
        rounds[u.speaker][u.session].append(u)
    by_speaker = {spk: max(r.values(), key=len) for spk, r in rounds.items()}
    pool = [k for k, v in by_speaker.items() if len(v) >= 5]
    if len(pool) < speakers:
        raise SystemExit(
            f"Only {len(pool)} people in this split have enough clips for {speakers}-person "
            "meetings; use a larger split or --meetings 0."
        )
    out = {}
    for m in range(count):
        chosen = rng.sample(pool, speakers)
        queues = {k: rng.sample(by_speaker[k], len(by_speaker[k])) for k in chosen}
        labels = {k: f"P{i + 1}" for i, k in enumerate(chosen)}
        placed: list[Placed] = []
        t, last = 0.0, None
        while t < minutes * 60 and any(queues.values()):
            options = [k for k in chosen if queues[k] and k != last] or [
                k for k in chosen if queues[k]
            ]
            k = rng.choice(options)
            u = queues[k].pop()
            if placed and rng.random() < overlap:
                start = max(0.0, t - rng.uniform(0.3, 1.2))  # interrupt the previous speaker
            else:
                start = t + rng.uniform(0.2, 0.8)
            placed.append(Placed(labels[k], start, u))
            t = max(t, start + len(u.audio) / SR)
            last = k
        out[f"ascend_meeting{m + 1:02d}"] = placed
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, default=Path("data/ascend/main"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--max", type=int, default=5, help="Max two-person conversations to build")
    ap.add_argument("--max-minutes", type=float, default=10)
    ap.add_argument("--meetings", type=int, default=3, help="Simulated meetings to build")
    ap.add_argument("--speakers", type=int, default=4)
    ap.add_argument("--meeting-minutes", type=float, default=5)
    ap.add_argument("--overlap", type=float, default=0.15, help="Share of turns that interrupt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()

    if args.inspect:
        import pandas as pd

        f = sorted(args.src.glob(f"{args.split}-*.parquet"))[0]
        df = pd.read_parquet(f)
        print("columns:", list(df.columns), "| rows:", len(df))
        print(df.drop(columns=[c for c in df.columns if c == "audio"]).head(5).to_string())
        return

    print(f"Loading {args.split} split...")
    utts = load_split(args.src, args.split)
    timed = sum(u.start is not None for u in utts)
    print(
        f"{len(utts)} clips, {len({u.session for u in utts})} sessions, "
        f"{timed} with original timestamps"
    )

    for name, placed in build_real(utts, args.max, args.max_minutes).items():
        dur = write_set(name, placed, args.out / "ascend_real")
        n = len({p.speaker for p in placed})
        print(f"  real     {name}: {dur / 60:.1f} min, {n} speakers, {len(placed)} turns")

    if args.meetings:
        built = build_meetings(
            utts, args.meetings, args.speakers, args.meeting_minutes, args.overlap, args.seed
        )
        for name, placed in built.items():
            dur = write_set(name, placed, args.out / "ascend_meetings")
            print(
                f"  meeting  {name}: {dur / 60:.1f} min, {args.speakers} speakers, "
                f"{len(placed)} turns"
            )
    print(f"Done. See {args.out / 'ascend_real'} and {args.out / 'ascend_meetings'}.")


if __name__ == "__main__":
    main()
