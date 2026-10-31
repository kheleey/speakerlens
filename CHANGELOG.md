# Changelog

## 0.1.0 (2026-10-31)

First version: a recording in, a speaker-labeled, translated transcript with per-speaker
summaries out.

- **Who spoke when:** pyannote community-1 diarization, then a speaker check that regroups
  turns by voice and moves turns that clearly sound like someone else
  (`SPEAKER_REASSIGN_MARGIN`, `--reassign-margin`). Optional voice samples (`--voices`) name
  the speakers and guide the check, including people diarization merged into someone else.
- **Overlapping speech:** turns where someone talks over someone else, and speech hidden
  under someone else, are recognized from that person's voice, pulled out of the mix by
  target speaker extraction (WeSep BSRNN, code bundled in `_vendor/`, Apache-2.0; model
  downloads itself on first use; `SPEAKER_EXTRACTION`, `--extract/--no-extract`,
  `TSE_MODEL` for your own copy). A failure falls back to the original audio.
- **Speech recognition:** language ID over the languages you choose, a router that uses each
  speaker's usual language, and language plugins with calibrated confidence. On a GPU,
  Fun-ASR-Nano for Mandarin, English and mixed speech, with SenseVoice taking over when Nano
  stops early, runs on, loops or answers in Japanese; without a GPU, SenseVoice and Whisper
  (`ASR_PROFILE=auto|nano|routing`). Whisper covers other languages. Third-party plugins
  register through the `speakerlens.plugins` entry point.
- **Text repairs:** repetition loops cut, spelled-out acronyms joined ("a i" -> "AI"),
  all-capital English lowered.
- **Translation and summaries** with DeepSeek (`deepseek-flash`, thinking off by default) or
  Ollama. Every summary point cites its turns; points citing someone else's words are
  rejected; ideas are credited to whoever raised them first and agreements are checked for
  direction; filler, pasted transcript lines and repeated points are dropped. A failed LLM
  batch is split and retried; a rejected key or an empty account balance stops at once. The
  transcript is never lost.
- **Outputs:** Markdown, JSON, SRT/VTT (plus translated SRT), RTTM, STM.
- **Optional AliMeeting-tuned extraction model** attached to the release
  (`tse-alimeeting.tar.gz`, CC-BY-SA-4.0); `training/` rebuilds it.
- **CLI:** `speakerlens transcribe | plugins | licenses | doctor | calibrate`.
- Measured on ASCEND (code-switching) and AliMeeting (real overlapped meetings), and on two
  real test recordings; see the README. Compared with WhisperX and VibeVoice-ASR on the same
  sets with the same scoring (`bench/compare/`), including held-out sets: clearly the best on
  real overlapped meetings (AliMeeting Test tcpWER 0.288 against 0.416-0.441); on clean
  code-switching speech VibeVoice-ASR recognizes more English. Per-person summaries from
  speakerlens's and VibeVoice-ASR's transcripts score about the same. Ideas that were
  measured and not kept (PixIT separation, fine-tuning Fun-ASR-Nano, echo removal, various
  extraction settings) are listed in docs/architecture.md.
- The simulated ASCEND meetings no longer put a person recorded in two sessions into one
  meeting twice.
