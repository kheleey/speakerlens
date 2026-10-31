"""The full chain: diarize -> speaker check -> extract overlapped speech -> route and
transcribe each turn -> translate -> summarize per speaker."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from speakerlens import __version__
from speakerlens.audio import SAMPLE_RATE, array_hash, load_audio, slice_audio
from speakerlens.cache import StageCache
from speakerlens.calibration import CalibrationStore
from speakerlens.diarization.base import (
    DiarizationResult,
    Diarizer,
    hidden_turns,
    segments_to_turns,
)
from speakerlens.diarization.enrollment import (
    AUDIO_EXTENSIONS,
    SpeakerEmbedder,
    VoiceBank,
    speaker_audio,
)
from speakerlens.diarization.reassign import (
    anchors_from_voices,
    name_speakers,
    reassign_turns,
    recluster,
    turn_embeddings,
)
from speakerlens.langid import LanguageIdentifier
from speakerlens.llm import JSONChat, LLMError
from speakerlens.metrics import tokenize
from speakerlens.plugins.base import PluginResult, TurnContext
from speakerlens.plugins.registry import PluginRegistry
from speakerlens.router import (
    Candidate,
    RouterConfig,
    SpeakerLanguagePrior,
    TurnOutcome,
    route,
)
from speakerlens.summarize import summarize
from speakerlens.textfix import tame_caps
from speakerlens.translate import translate_transcript
from speakerlens.types import Speaker, Transcript, Turn, Word

if TYPE_CHECKING:
    from speakerlens.config import Settings
    from speakerlens.extraction import TargetExtractor

log = logging.getLogger(__name__)

AUTO = "auto"
ProgressFn = Callable[[str, int, int], None]  # (stage, done, total)


# Hidden speech (someone talking under someone else) is transcribed only when it adds up to
# HIDDEN_MIN_SHARE of the recording, in stretches of at least HIDDEN_MIN_TURN seconds. A
# hidden stretch whose text mostly (ECHO_SHARE) repeats the main speaker is dropped.
# EXTRACT_CONTEXT: seconds of audio around each turn given to the extractor.
HIDDEN_MIN_SHARE = 0.02
HIDDEN_MIN_TURN = 0.5
ECHO_SHARE = 0.6
EXTRACT_CONTEXT = 1.0
VOICE_SAMPLE_SECONDS = 20.0


@dataclass
class PipelineOptions:
    languages: list[str] = field(default_factory=lambda: ["zh", "en"])  # or ["auto"]
    translate_to: str | None = None
    summarize: bool = True
    summary_language: str | None = None  # defaults to translate_to, else "en"
    voices_dir: Path | None = None
    voice_threshold: float = 0.5
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    # Move turns whose voice clearly matches another speaker (None = off). See reassign.py.
    reassign_margin: float | None = None
    # With the speaker check on, first regroup turns of 2 s or more into voices from scratch.
    speaker_regroup: bool = True
    use_cache: bool = True

    @property
    def allowed_languages(self) -> list[str]:
        """Empty list = any language."""
        return [] if AUTO in self.languages else list(self.languages)


@dataclass
class Scored:
    candidate: Candidate
    result: PluginResult
    calibrated: float | None


def make_extractor(settings: Settings, device: str) -> TargetExtractor | None:
    """Target speaker extraction for overlapped turns, unless SPEAKER_EXTRACTION=off. Uses
    the model in TSE_MODEL if set, otherwise downloads it into the cache on first use (only
    recordings with overlapped speech need it). If that fails, the run continues without."""
    from speakerlens.extraction import TargetExtractor, model_ready

    if settings.extraction == "off":
        return None
    if settings.tse_model is not None:
        if not model_ready(settings.tse_model):
            log.warning(
                "TSE_MODEL (%s) has no avg_model.pt and config.yaml; continuing without "
                "speaker extraction. See speakerlens/extraction.py for the download.",
                settings.tse_model,
            )
            return None
        return TargetExtractor(settings.tse_model, device)
    return TargetExtractor(settings.cache_dir / "models" / "wesep_bsrnn", device, download=True)


def voice_samples(
    audio: np.ndarray,
    turns: list[Turn],
    voices: dict[str, np.ndarray] | None = None,
    max_seconds: float = 20.0,
    min_clean: float = 3.0,
    min_turn: float = 0.5,
) -> dict[str, tuple[np.ndarray, bool]]:
    """Speaker -> (voice sample, clean?) for target speaker extraction. A voice file from
    --voices is used when given; otherwise the speaker's longest turns with nobody else
    talking, up to `max_seconds`. With under `min_clean` seconds of clean turns, their
    longest turns of any kind (clean=False: the sample also carries other voices)."""
    out: dict[str, tuple[np.ndarray, bool]] = {}
    for spk in sorted({t.speaker for t in turns} | set(voices or {})):
        if voices and spk in voices:
            out[spk] = (voices[spk][: int(max_seconds * SAMPLE_RATE)], True)
            continue
        mine = sorted(
            (t for t in turns if t.speaker == spk and t.duration >= min_turn),
            key=lambda t: -t.duration,
        )
        clean = [t for t in mine if not t.overlap]
        is_clean = sum(t.duration for t in clean) >= min_clean
        pieces, got = [], 0.0
        for t in clean if is_clean else mine:
            if got >= max_seconds:
                break
            pieces.append(slice_audio(audio, t.start, t.end))
            got += t.duration
        if pieces:
            out[spk] = (np.concatenate(pieces), is_clean)
    return out


def _voice_file(directory: Path, name: str) -> Path | None:
    for path in sorted(Path(directory).iterdir()):
        if path.stem == name and path.suffix.lower() in AUDIO_EXTENSIONS:
            return path
    return None


def is_echo(hidden: Turn, turns: list[Turn], share: float = 0.6) -> bool:
    """True when a hidden turn's text mostly repeats a regular turn at the same time: the
    extracted voice still carried the main speaker, not a second person."""
    words = tokenize(hidden.text)
    if not words:
        return True
    # Count only runs of 2+ matching words: single common characters (的, 我, 是) match by
    # chance inside a long turn.
    min_run = 2 if len(words) > 3 else 1
    for t in turns:
        if t.text and t.start < hidden.end and hidden.start < t.end:
            blocks = SequenceMatcher(None, words, tokenize(t.text), autojunk=False)
            repeated = sum(b.size for b in blocks.get_matching_blocks() if b.size >= min_run)
            if repeated >= share * len(words):
                return True
    return False


def _recording(audio: np.ndarray) -> Callable[[Turn], np.ndarray]:
    """Each turn's audio, straight from the recording."""

    def audio_of(turn: Turn) -> np.ndarray:
        return slice_audio(audio, turn.start, turn.end)

    return audio_of


def _pick(results: list[Scored]) -> Scored:
    """Highest calibrated confidence wins; results without a score lose to ones with one."""
    usable = [r for r in results if r.result.text.strip()] or results
    scored = [r for r in usable if r.calibrated is not None]
    return max(scored, key=lambda r: r.calibrated or 0.0) if scored else usable[0]


class Pipeline:
    def __init__(
        self,
        registry: PluginRegistry,
        diarizer: Diarizer,
        language_id: LanguageIdentifier,
        calibration: CalibrationStore | None = None,
        llm: JSONChat | None = None,
        embedder: SpeakerEmbedder | None = None,
        cache: StageCache | None = None,
        router_config: RouterConfig | None = None,
        progress: ProgressFn | None = None,
        cache_dir: Path | None = None,
        extractor: TargetExtractor | None = None,
    ) -> None:
        self.registry = registry
        self.diarizer = diarizer
        # Optional: pulls one person's voice out of overlapped speech (extraction.py).
        self.extractor = extractor
        self.language_id = language_id
        self.calibration = calibration or CalibrationStore()
        self.llm = llm
        self.embedder = embedder
        self.cache = cache or StageCache(None, enabled=False)
        self.router_config = router_config or RouterConfig()
        self.progress = progress or (lambda stage, done, total: None)
        self.cache_dir = cache_dir
        self.last_outcomes: list[TurnOutcome] = []

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        options: PipelineOptions,
        progress: ProgressFn | None = None,
    ) -> Pipeline:
        """Build the real pipeline (pyannote, Whisper, SenseVoice, DeepSeek/Ollama)."""
        from speakerlens.config import resolve_device
        from speakerlens.diarization.enrollment import PyannoteEmbedder
        from speakerlens.diarization.pyannote_diarizer import PyannoteDiarizer
        from speakerlens.langid import FixedLanguage, WhisperLanguageIdentifier
        from speakerlens.llm import LLMClient
        from speakerlens.plugins.registry import default_registry

        device = resolve_device(settings.device)
        allowed = options.allowed_languages
        language_id: LanguageIdentifier = (
            FixedLanguage(allowed[0])
            if len(allowed) == 1
            else WhisperLanguageIdentifier(settings.langid_model, device)
        )
        needs_llm = bool(options.translate_to) or options.summarize
        llm = LLMClient.from_settings(settings.llm) if needs_llm else None
        if needs_llm and llm is None:
            log.warning(
                "No LLM configured (set DEEPSEEK_API_KEY or LLM_PROVIDER=ollama in .env); "
                "skipping translation and summaries."
            )
        return cls(
            registry=default_registry(settings),
            diarizer=PyannoteDiarizer(settings.diarization_model, settings.hf_token, device),
            extractor=make_extractor(settings, device),
            language_id=language_id,
            calibration=CalibrationStore(settings.calibration_dir),
            llm=llm,
            embedder=(
                PyannoteEmbedder(settings.embedding_model, settings.hf_token, device)
                if options.voices_dir or options.reassign_margin is not None
                else None
            ),
            cache=StageCache(settings.cache_dir, enabled=options.use_cache),
            progress=progress,
            cache_dir=settings.cache_dir,
        )

    def _diarize(
        self, audio: np.ndarray, audio_hash: str, options: PipelineOptions
    ) -> DiarizationResult:
        params = {
            "diarizer": type(self.diarizer).__name__,
            "model": getattr(self.diarizer, "model", None),
            "num": options.num_speakers,
            "min": options.min_speakers,
            "max": options.max_speakers,
            "format": 2,  # 2: includes hidden (overlapped) speech
        }
        cached = self.cache.get("diarization", audio_hash, params)
        if cached:
            log.info("Using cached diarization.")
            return DiarizationResult.model_validate_json(cached)
        result = self.diarizer.diarize(
            audio,
            SAMPLE_RATE,
            num_speakers=options.num_speakers,
            min_speakers=options.min_speakers,
            max_speakers=options.max_speakers,
        )
        self.cache.put("diarization", audio_hash, params, result.model_dump_json())
        return result

    def _name_speakers(
        self,
        audio: np.ndarray,
        turns: list[Turn],
        speakers: list[Speaker],
        options: PipelineOptions,
    ) -> None:
        if not options.voices_dir or self.embedder is None:
            return
        bank = VoiceBank.from_dir(options.voices_dir, self.embedder, self.cache_dir)
        # Speakers the speaker check created from a voice sample already know their name.
        for spk in speakers:
            name = spk.id.removeprefix("VOICE_")
            if name != spk.id and name in bank.voices:
                spk.name = name
                bank.voices.pop(name)
        vectors = {}
        for spk in speakers:
            if spk.name:
                continue
            clip = speaker_audio(audio, turns, spk.id)
            if clip.size >= SAMPLE_RATE:  # need at least 1 second
                vectors[spk.id] = self.embedder.embed(clip, SAMPLE_RATE)
        matched = bank.match(vectors, options.voice_threshold)
        for spk in speakers:
            if spk.id in matched:
                name, score = matched[spk.id]
                spk.name, spk.match_score = name, round(score, 3)

    def _check_speakers(
        self,
        audio: np.ndarray,
        turns: list[Turn],
        options: PipelineOptions,
    ) -> dict[str, tuple[str, float]] | None:
        """Regroup and move turns by voice (see diarization/reassign.py). With voice samples,
        also returns the names: label -> (name, similarity)."""
        assert self.embedder is not None and options.reassign_margin is not None
        embeddings = turn_embeddings(
            audio,
            SAMPLE_RATE,
            turns,
            self.embedder,
            min_seconds=0.5,
            overlap_min_seconds=2.0,
        )
        before = {t.id: t.speaker for t in turns}
        if options.speaker_regroup:
            n = options.num_speakers or len(set(before.values()))
            try:
                recluster(turns, embeddings, n)
            except ImportError:  # scipy comes with pyannote; without it, skip the regroup
                log.warning("scipy not installed; skipping the speaker regroup.")
        anchors = None
        if options.voices_dir:
            bank = VoiceBank.from_dir(options.voices_dir, self.embedder, self.cache_dir)
            anchors = anchors_from_voices(turns, embeddings, bank.voices, options.voice_threshold)
        reassign_turns(turns, embeddings, margin=options.reassign_margin, anchors=anchors)
        changed = sum(t.speaker != before[t.id] for t in turns)
        log.info("Speaker check changed %d of %d turns.", changed, len(turns))
        if not options.voices_dir:
            return None
        return name_speakers(turns, embeddings, bank.voices, options.voice_threshold)

    def _extraction_audio(
        self,
        audio: np.ndarray,
        turns: list[Turn],
        speakers: list[Speaker],
        options: PipelineOptions,
    ) -> tuple[Callable[[Turn], np.ndarray], set[str]]:
        """Audio for each turn: overlapped turns and hidden speech (ids starting with "H")
        from the speaker's extracted voice, everything else from the recording. Returns it
        and the speakers with a voice sample. If extraction fails, it warns once and uses
        the recording from then on."""
        assert self.extractor is not None
        extractor = self.extractor
        voices: dict[str, np.ndarray] = {}
        if options.voices_dir:
            for spk in speakers:
                path = _voice_file(options.voices_dir, spk.name) if spk.name else None
                if path is not None:
                    voices[spk.id] = load_audio(path)
        samples = voice_samples(audio, turns, voices, max_seconds=VOICE_SAMPLE_SECONDS)
        weak = sorted(s for s, (_, clean) in samples.items() if not clean)
        if weak:
            log.info(
                "No clean speech for %s: their voice sample includes other voices. A voice "
                "sample (--voices) would help.",
                ", ".join(weak),
            )
        prepared: dict[str, object] = {}
        failed = False
        plain = _recording(audio)

        def audio_of(turn: Turn) -> np.ndarray:
            nonlocal failed
            wanted = turn.overlap or turn.id.startswith("H")
            if not wanted or failed or turn.speaker not in samples:
                return plain(turn)
            try:
                if turn.speaker not in prepared:
                    prepared[turn.speaker] = extractor.prepare(samples[turn.speaker][0])
                a = max(0.0, turn.start - EXTRACT_CONTEXT)
                voice = extractor.extract(
                    slice_audio(audio, a, turn.end + EXTRACT_CONTEXT), prepared[turn.speaker]
                )
                lo = int(round((turn.start - a) * SAMPLE_RATE))
                piece = voice[lo : lo + int(round(turn.duration * SAMPLE_RATE))]
            except Exception as exc:  # never lose the transcript over it
                log.warning(
                    "Speaker extraction failed (%s); using the original audio from here on.", exc
                )
                failed = True
                return plain(turn)
            peak = float(np.abs(piece).max()) if piece.size else 0.0
            return piece * (0.9 / peak) if peak > 0 else piece

        return audio_of, set(samples)

    def _transcribe_turn(
        self, clip: np.ndarray, turn: Turn, prior: SpeakerLanguagePrior, options: PipelineOptions
    ) -> None:
        lid = self.language_id.probs(clip, SAMPLE_RATE, options.allowed_languages)
        decision = route(
            lid, self.registry, prior.get(turn.speaker), turn.duration, self.router_config
        )
        ranked = sorted(decision.scores, key=lambda k: decision.scores[k], reverse=True)

        results: list[Scored] = []
        for cand in decision.candidates:
            plugin = self.registry.get(cand.plugin_id)
            ctx = TurnContext(turn.id, turn.speaker, turn.start, turn.end, cand.language)
            try:
                res = plugin.transcribe(clip, SAMPLE_RATE, ctx)
            except Exception as exc:
                log.warning("Plugin %s failed on %s: %s", cand.plugin_id, turn.id, exc)
                continue
            calibrated = self.calibration.get(cand.plugin_id).apply(res.raw_confidence)
            results.append(Scored(cand, res, calibrated))

        if not results:
            return
        best = _pick(results)
        mixed_label = "+".join(sorted(ranked[:2]))
        turn.language = best.candidate.language or best.result.language or mixed_label
        turn.language_confidence = round(decision.scores.get(ranked[0], 0.0), 3)
        turn.plugin = best.candidate.plugin_id + (
            f">{best.result.used}" if best.result.used else ""
        )
        turn.text = tame_caps(best.result.text.strip())
        turn.raw_confidence = best.result.raw_confidence
        turn.confidence = best.calibrated
        turn.words = [
            Word(
                text=w.text,
                start=round(turn.start + w.start, 3),
                end=round(turn.start + w.end, 3),
                confidence=w.confidence,
            )
            for w in best.result.words
        ]
        prior.update(turn.speaker, turn.language, turn.duration)
        self.last_outcomes.append(
            TurnOutcome(
                turn_id=turn.id,
                decision=decision,
                chosen=best.candidate.plugin_id,
                calibrated={r.candidate.plugin_id: r.calibrated for r in results},
            )
        )

    def run(
        self, source: str | Path, options: PipelineOptions, audio: np.ndarray | None = None
    ) -> Transcript:
        if audio is None:
            self.progress("Loading audio", 0, 1)
            audio = load_audio(source)
            self.progress("Loading audio", 1, 1)
        audio_hash = array_hash(audio)

        self.progress("Diarizing", 0, 1)
        diarization = self._diarize(audio, audio_hash, options)
        turns = segments_to_turns(diarization, audio=audio, sample_rate=SAMPLE_RATE)
        self.progress("Diarizing", 1, 1)
        # Speech hidden under someone else is recovered from the extracted voice, so only
        # with an extractor, and only when there is enough of it to matter.
        hidden: list[Turn] = []
        if self.extractor is not None and diarization.hidden:
            hidden_seconds = sum(h.end - h.start for h in diarization.hidden)
            if hidden_seconds >= HIDDEN_MIN_SHARE * len(audio) / SAMPLE_RATE:
                hidden = hidden_turns(diarization, min_turn=HIDDEN_MIN_TURN)
            else:
                log.info("Little overlapped speech (%.0f s); not transcribed.", hidden_seconds)

        names = None
        if options.reassign_margin is not None and self.embedder is not None:
            self.progress("Checking speakers", 0, 1)
            names = self._check_speakers(audio, turns, options)
            self.progress("Checking speakers", 1, 1)
        found = set(diarization.speakers) | {t.speaker for t in turns}
        speakers = [Speaker(id=s) for s in sorted(found)]

        if names is not None:  # named by the speaker check, from the same voiceprints
            for spk in speakers:
                if spk.id in names:
                    spk.name, spk.match_score = names[spk.id]
        elif options.voices_dir:
            self.progress("Matching voices", 0, 1)
            self._name_speakers(audio, turns, speakers, options)
            self.progress("Matching voices", 1, 1)

        self.last_outcomes = []
        prior = SpeakerLanguagePrior(self.router_config.min_prior_seconds)
        audio_of = _recording(audio)
        if self.extractor is not None:
            audio_of, sampled = self._extraction_audio(audio, turns, speakers, options)
            hidden = [h for h in hidden if h.speaker in sampled]
            if hidden:
                log.info("%d stretches of overlapped speech to transcribe.", len(hidden))
        everything = turns + hidden
        for i, turn in enumerate(everything):
            self.progress("Transcribing", i, len(everything))
            self._transcribe_turn(audio_of(turn), turn, prior, options)
        self.progress("Transcribing", len(everything), len(everything))
        echoes = {h.id for h in hidden if h.text and is_echo(h, turns, ECHO_SHARE)}
        if echoes:
            log.info(
                "Dropped %d overlapped stretches that only repeated the main speaker.", len(echoes)
            )

        kept = sorted(
            (t for t in everything if t.text and t.id not in echoes), key=lambda t: t.start
        )
        speaking = {t.speaker for t in kept}
        transcript = Transcript(
            source=str(source),
            duration=round(len(audio) / SAMPLE_RATE, 3),
            languages=options.languages,
            speakers=[s for s in speakers if s.id in speaking],
            turns=kept,
            created_with=f"speakerlens {__version__}",
        )

        # LLM steps come last and never throw away the transcript if they fail.
        if options.translate_to and self.llm:
            self.progress("Translating", 0, 1)
            try:
                failed = translate_transcript(transcript, options.translate_to, self.llm)
                if failed:
                    log.warning("%d turn(s) could not be translated.", failed)
            except LLMError as exc:
                log.error("Translation failed: %s", exc)
            self.progress("Translating", 1, 1)

        if options.summarize and self.llm and transcript.turns:
            self.progress("Summarizing", 0, 1)
            language = options.summary_language or options.translate_to or "en"
            try:
                transcript.summary = summarize(
                    transcript,
                    self.llm,
                    language=language,
                    use_translation=options.translate_to == language,
                )
            except LLMError as exc:
                log.error("Summary failed: %s", exc)
            self.progress("Summarizing", 1, 1)
        return transcript
