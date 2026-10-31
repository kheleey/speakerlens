"""Move turns to the speaker whose voice they actually match.

Diarization decides speakers from short windows, so some turns end up with the wrong
person, especially short ones and in meetings with many speakers. Here each turn gets a
voice embedding, each speaker gets a voiceprint (the average of their longer turns), and a
turn moves to another speaker only when it is clearly closer to that speaker's voiceprint
(by at least `margin` in cosine similarity). Turns too short for a reliable embedding are
never moved, and neither are short turns that overlap someone else. Voiceprints are
recomputed and the check repeated a few times.

`recluster` is the stronger option when the number of speakers is known: it regroups the
longer turns from scratch, for files where diarization mixed several people together.

With voice samples (`--voices`), `anchors_from_voices` replaces guessed voiceprints with the
real ones: each found speaker that matches a sample uses the sample's voiceprint, and each
sample nobody matched becomes a new speaker that turns can move to (for people diarization
merged into someone else).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from speakerlens.audio import slice_audio
from speakerlens.diarization.enrollment import SpeakerEmbedder, VoiceBank, normalize
from speakerlens.types import Turn


@dataclass
class Move:
    turn_id: str
    old: str
    new: str
    gain: float  # similarity to the new speaker minus similarity to the old one


def usable(turn: Turn, min_seconds: float = 1.0, overlap_min_seconds: float | None = None) -> bool:
    """Long enough to embed reliably. A turn that overlaps someone else counts only if it is
    at least `overlap_min_seconds` long (None: never), so the other voice is a small part."""
    if turn.duration < min_seconds:
        return False
    if turn.overlap:
        return overlap_min_seconds is not None and turn.duration >= overlap_min_seconds
    return True


def turn_embeddings(
    audio: np.ndarray,
    sample_rate: int,
    turns: list[Turn],
    embedder: SpeakerEmbedder,
    min_seconds: float = 1.0,
    overlap_min_seconds: float | None = None,
) -> dict[str, np.ndarray]:
    """Normalized embedding per turn id, for turns `usable` says are reliable."""
    out = {}
    for t in turns:
        if not usable(t, min_seconds, overlap_min_seconds):
            continue
        clip = slice_audio(audio, t.start, t.end, sample_rate)
        out[t.id] = normalize(embedder.embed(clip, sample_rate))
    return out


def voiceprints(
    turns: list[Turn], embeddings: dict[str, np.ndarray], min_seconds: float = 2.0
) -> dict[str, np.ndarray]:
    """Duration-weighted mean embedding per speaker, from turns of at least `min_seconds`
    (or all embedded turns, for speakers without any that long)."""
    prints = {}
    for spk in {t.speaker for t in turns}:
        own = [t for t in turns if t.speaker == spk and t.id in embeddings]
        long = [t for t in own if t.duration >= min_seconds] or own
        if long:
            weighted = np.stack([embeddings[t.id] * t.duration for t in long])
            prints[spk] = normalize(weighted.sum(axis=0))
    return prints


def anchors_from_voices(
    turns: list[Turn],
    embeddings: dict[str, np.ndarray],
    voices: dict[str, np.ndarray],
    threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """Speaker label -> fixed voiceprint from a voice sample. Found speakers are matched to
    samples one-to-one (cosine similarity >= threshold); unmatched samples get a new label
    "VOICE_<name>" so turns that sound like them can move there."""
    bank = VoiceBank(voices)
    matched = bank.match(voiceprints(turns, embeddings), threshold)
    anchors = {label: bank.voices[name] for label, (name, _score) in matched.items()}
    taken = {name for name, _score in matched.values()}
    for name, vec in bank.voices.items():
        if name not in taken:
            anchors[f"VOICE_{name}"] = vec
    return anchors


def reassign_turns(
    turns: list[Turn],
    embeddings: dict[str, np.ndarray],
    margin: float = 0.1,
    rounds: int = 3,
    min_print_seconds: float = 2.0,
    anchors: dict[str, np.ndarray] | None = None,
) -> list[Move]:
    """Relabel turns in place; returns the moves made (last move per turn). `anchors` are
    fixed voiceprints (e.g. from voice samples) that override the guessed ones."""
    moves: dict[str, Move] = {}
    original = {t.id: t.speaker for t in turns}
    for _ in range(rounds):
        prints = voiceprints(turns, embeddings, min_print_seconds)
        prints.update(anchors or {})
        if len(prints) < 2:
            break
        changed = False
        for t in turns:
            vec = embeddings.get(t.id)
            if vec is None or t.speaker not in prints:
                continue
            sims = {spk: float(np.dot(vec, p)) for spk, p in prints.items()}
            best = max(sims, key=lambda k: sims[k])
            gain = sims[best] - sims[t.speaker]
            if best != t.speaker and gain >= margin:
                t.speaker = best
                changed = True
                if best == original[t.id]:
                    moves.pop(t.id, None)
                else:
                    moves[t.id] = Move(t.id, original[t.id], best, round(gain, 3))
        if not changed:
            break
    return list(moves.values())


def recluster(
    turns: list[Turn],
    embeddings: dict[str, np.ndarray],
    n_speakers: int,
    min_seconds: float = 2.0,
) -> int:
    """Regroup turns into `n_speakers` voices from scratch, in place; returns turns changed.

    Turns of at least `min_seconds` are clustered (Ward linkage on unit-length embeddings,
    which favors groups of similar size, so one odd turn doesn't become a "speaker"). The
    new groups take over the old speaker labels they share most time with, so labels stay
    stable. Other embedded turns go to the nearest group; turns without an embedding keep
    their label.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.optimize import linear_sum_assignment

    long = [t for t in turns if t.id in embeddings and t.duration >= min_seconds]
    if len(long) <= n_speakers or n_speakers < 2:
        return 0
    x = np.stack([embeddings[t.id] for t in long])
    groups = fcluster(linkage(x, method="ward"), n_speakers, "maxclust")

    old_labels = sorted({t.speaker for t in turns})
    new_ids = sorted(set(groups.tolist()))
    shared = np.zeros((len(new_ids), len(old_labels)))
    for t, g in zip(long, groups, strict=True):
        shared[new_ids.index(g), old_labels.index(t.speaker)] += t.duration
    rows, cols = linear_sum_assignment(-shared)
    name = {new_ids[r]: old_labels[c] for r, c in zip(rows, cols, strict=True)}
    spare = iter(f"SPEAKER_X{i}" for i in range(len(new_ids)))
    for g in new_ids:
        name.setdefault(g, next(spare))  # more groups than old labels

    labels = {t.id: name[g] for t, g in zip(long, groups, strict=True)}
    prints = {}
    for g in new_ids:
        members = [t for t, gg in zip(long, groups, strict=True) if gg == g]
        prints[name[g]] = normalize(
            np.stack([embeddings[t.id] * t.duration for t in members]).sum(axis=0)
        )
    changed = 0
    for t in turns:
        vec = embeddings.get(t.id)
        new = labels.get(t.id)
        if new is None and vec is not None:
            new = max(prints, key=lambda k: float(np.dot(vec, prints[k])))
        if new is not None and new != t.speaker:
            t.speaker = new
            changed += 1
    return changed


def name_speakers(
    turns: list[Turn],
    embeddings: dict[str, np.ndarray],
    voices: dict[str, np.ndarray],
    threshold: float = 0.5,
) -> dict[str, tuple[str, float]]:
    """Speaker label -> (name, similarity), using each speaker's voiceprint from all their
    turns and the best overall one-to-one assignment (so two similar speakers can't swap
    names because one grabbed its best match first). "VOICE_<name>" labels keep their name.
    Pairs below `threshold` stay unnamed."""
    bank = VoiceBank(voices)
    prints = voiceprints(turns, embeddings)
    out: dict[str, tuple[str, float]] = {}
    for label, vec in prints.items():
        name = label.removeprefix("VOICE_")
        if name != label and name in bank.voices:
            out[label] = (name, round(float(np.dot(vec, bank.voices[name])), 3))
    labels = [k for k in prints if k not in out]
    names = [n for n in bank.voices if n not in {v[0] for v in out.values()}]
    if not labels or not names:
        return out
    sims = np.array([[float(np.dot(prints[k], bank.voices[n])) for n in names] for k in labels])
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(-sims)
        pairs: list[tuple[int, int]] = list(zip(rows.tolist(), cols.tolist(), strict=True))
    except ImportError:  # scipy comes with pyannote; fall back to greedy matching
        matched = bank.match({k: prints[k] for k in labels}, threshold)
        pairs = [(labels.index(k), names.index(n)) for k, (n, _s) in matched.items()]
    for r, c in pairs:
        if sims[r, c] >= threshold:
            out[labels[r]] = (names[c], round(float(sims[r, c]), 3))
    return out
