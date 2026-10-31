# Architecture

```
recording ──► diarization (pyannote) ──► turns ──► speaker check ──► overlapped turns: extract that person's voice
                                                        ▲                          │
voice samples (optional) ───────────────────────────────┘                          ▼
                     language ID + router ──► plugin(s) ──► calibration ──► translation ──► per-speaker summaries ──► outputs
```

## Stages

| Stage | Module | Input → output |
| --- | --- | --- |
| Load audio | `audio.py` | any file → 16 kHz mono float32 (via ffmpeg) |
| Diarize | `diarization/pyannote_diarizer.py` | audio → exclusive speaker segments, overlaps, hidden speech (cached) |
| Build turns | `diarization/base.py` | segments → turns (merged, max 30 s, overlap-flagged) |
| Speaker check | `diarization/reassign.py` | turns + voice embeddings → turns regrouped and moved by voice |
| Name speakers | `diarization/enrollment.py`, `reassign.py` | voice samples → speaker names |
| Extract voices | `extraction.py` | overlapped turn + voice sample → that person's voice alone |
| Language ID | `langid.py` | turn audio → probabilities over the selected languages |
| Route | `router.py` | probabilities + speaker prior → which plugin(s) to run |
| Transcribe | `plugins/`, `textfix.py` | turn audio → text, words, raw confidence |
| Calibrate | `calibration.py` | raw confidence → probability the turn is right |
| Translate | `translate.py` | turns → translated turns (LLM, batched) |
| Summarize | `summarize.py` | transcript → per-speaker points with turn citations, checked |
| Export | `export.py` | transcript → Markdown, JSON, SRT/VTT, RTTM, STM |

## Routing rules

1. Blend language-ID scores with the speaker's prior (30%, or 70% on turns under 1 s).
   The prior only counts after 10 s of that speaker's speech.
2. A mixed plugin covers the top two languages and the top one is below 0.97 → the mixed
   plugin. On ASCEND, "English" turns routed to Whisper had 41% error because Whisper drops
   the Chinese inside them.
3. Top language ≥ 0.8 → its specialist plugin (or the fallback if none).
4. Second language < 0.2 (nothing else plausible) → the leader's plugin.
5. Second language ≥ 0.2 and a mixed plugin covers both → the mixed plugin.
6. Otherwise run the top two languages' plugins and keep the higher calibrated confidence.
   If both languages map to the same fallback plugin, it runs once and detects the language.

All thresholds live in `RouterConfig`.

Turns longer than 30 s (monologues) are split at the quietest moment before each 30 s
boundary, so language ID, the plugins and summary citations work on manageable pieces.

## Speech models

Which models fill the plugin slots is set by `ASR_PROFILE` (or `--asr-profile`):

| Profile | Mandarin turns | English turns | Mixed turns |
| --- | --- | --- | --- |
| `nano` | Fun-ASR-Nano | Fun-ASR-Nano | Fun-ASR-Nano |
| `routing` | SenseVoice (zh mode) | Whisper turbo | SenseVoice (zh mode) |

`auto` (the default) means `nano` on a CUDA GPU and `routing` otherwise. Whisper large-v3 is
the fallback for any other language in both.

**Repairs for Fun-ASR-Nano.** It is a speech LLM, with typical failures:

- It sometimes ends a turn at a language switch and drops the rest, or runs on. SenseVoice
  also transcribes every turn of 1.5 s or more; its text is used when Nano has fewer than
  60% of its tokens (`NANO_FALLBACK_RATIO`) or more than 3 times plus 10.
- It sometimes gets stuck repeating a phrase. Output length is capped by the turn's
  duration, a phrase repeated 4+ times in a row is cut to one copy, and SenseVoice's text is
  used instead (`textfix.collapse_loops`).
- On short replies it sometimes answers in Japanese. When Japanese wasn't chosen,
  SenseVoice's text is used, even if empty.
- It spells some acronyms out ("a i") and sometimes writes English in capitals; both are
  normalized (`textfix.join_spelled_letters`, `tame_caps`).

Turns that used SenseVoice show up as e.g. `nano-mixed>sensevoice` in the output.

| ASCEND dev, tcpWER / tcORC-WER | Conversations (9) | Meetings (8) |
| --- | --- | --- |
| routing (SenseVoice + Whisper) | 0.222 / 0.182 | 0.325 / 0.167 |
| nano | 0.207 / 0.164 | 0.359 / 0.196 |
| nano + repairs | 0.200 / 0.156 | 0.311 / 0.145 |

On the two held-out conversations: routing 0.227 / 0.178, nano + repairs 0.197 / 0.147.

## Speaker check

After diarization, every turn of 0.5 s or more gets a voice embedding (turns that overlap
someone only if they are 2 s or longer, so the other voice is a small part). Then:

1. **Regroup:** turns of 2 s or more are clustered into the expected number of speakers
   from scratch (Ward linkage), keeping the old labels where they match. This fixes files
   where diarization mixed people together.
2. **Check:** each speaker gets a voiceprint (duration-weighted average of their turns of
   2 s or more), and a turn moves to another speaker when it is closer to that voiceprint by
   at least `SPEAKER_REASSIGN_MARGIN` (default 0.1; 0 turns the whole step off; also
   `--reassign-margin`). Voiceprints are recomputed and the check repeated up to 3 times.

**With voice samples** (`--voices`), step 2 uses the real voiceprints: found speakers
matched to a sample use that sample's voiceprint, and a sample nobody matched becomes a new
speaker (`VOICE_<name>`) that turns can move to, for people diarization merged into someone
else. Speakers are named from the same voiceprints, with the best overall assignment.

| ASCEND, tcpWER / DER | No check | Check | Check + voice samples |
| --- | --- | --- | --- |
| 9 dev conversations | 0.200 / 0.153 | 0.175 / 0.138 | 0.176 / 0.138 |
| 8 dev meetings | 0.311 / 0.206 | 0.248 / 0.169 | 0.203 / 0.151 |
| 2 held-out conversations | 0.197 / 0.132 | 0.178 / 0.119 | 0.181 / 0.121 |

The settings were chosen on the dev sets; the held-out conversations picked the same winner.
The voice samples came from each person's *other* ASCEND clips, never ones in the file.

The dev set has ten simulated meetings, but in two of them (04 and 09) the meeting builder
had put the same person in twice under two labels: ASCEND records most people in several
sessions, and the builder treated each session as a different person. It now takes each
person from one session only. Every meeting number here and in the README leaves those two
meetings out. With them every system scored worse, VibeVoice-ASR more than speakerlens
(tcpWER 0.314 and 0.306), and the steps above ranked the same.

## Overlapping speech

Diarization is *exclusive*: one speaker per moment, so when someone talks over someone
else, the one underneath loses those words. On AliMeeting (real meetings, 25-65% of speech
overlapped in 4-person meetings), 0.22 of the 0.27 DER is such missed speech.

pyannote also produces an overlap-aware view. Speech in that view but not in the exclusive
one is *hidden* speech (`DiarizationResult.hidden`). The pipeline
(`Pipeline._extraction_audio`, `extraction.py`):

1. After the speaker check, each speaker gets a voice sample: their `--voices` file, or
   their longest turns with nobody else talking (up to 20 s). People who never talk alone
   get a sample from their longest turns, which also carries other voices (logged).
2. Regular turns that overlap someone, and hidden speech (extra turns `H0001`, ... for the
   person underneath, when hidden speech is at least 2% of the recording), are recognized
   from that person's voice, pulled out of the mix by a target speaker extraction model,
   with 1 s of context on each side.
3. A hidden turn whose text mostly repeats the main speaker at that moment is dropped
   (`pipeline.is_echo`): the extracted voice still carried the main speaker.
4. Everything else is recognized from the original audio. If extraction fails, a warning
   is logged and the rest of the recording uses the original audio.

With true speaker segments on AliMeeting Eval, extraction cut errors in heavily overlapped
speech from 0.61 to 0.45 and in partly overlapped speech from 0.14 to 0.12, with clean
speech unchanged (0.256 -> 0.200 overall). Full pipeline: AliMeeting Eval tcpWER 0.303 -> 0.278 and DER 0.273 -> 0.224;
Test 0.299 -> 0.288 and 0.266 -> 0.226. ASCEND (little overlap) unchanged.

Model: WeSep BSRNN + ECAPA trained on VoxCeleb1 with on-the-fly mixtures (ModelScope
`wenet/wesep_pretrained_models`, downloaded on first use; code bundled in `_vendor/`).
WeNet's model license policy: a pretrained model follows its training data's license, here
CC-BY-4.0. A version fine-tuned on AliMeeting's training meetings does a little better
(Eval 0.275, Test 0.281; DER 0.219); `training/` rebuilds it (table microphone in, each
person's headset as the answer, lined up per window with GCC-PHAT; voice samples made as
the pipeline makes them, 30% deliberately contaminated; SI-SDR loss, speaker encoder
frozen). Set `TSE_MODEL` to use it.

## Summaries

Each speaker's points come in four kinds: main points, agreements, disagreements and open
questions. The instructions (`summarize.PROMPT_VERSION`) and code checks:

- **Citations.** Every point must cite turn ids; only cited turns that person actually spoke
  are kept, and points with none are rejected (kept in the JSON as `rejected_points`).
- **Who proposed what.** An idea is credited to whoever raised it first; "agrees with X" is
  only kept when X said something of substance shortly before, so agreements can't be
  reversed (the main error a human spot-check found).
- **No filler, no pasted lines, no repeats.** Backchannels aren't points; a point that is a
  transcript line pasted unchanged is rejected (when the summary language isn't Chinese,
  Japanese or Korean); repeated points are dropped.

Checked against the human transcript of AliMeeting Eval with an LLM judge: 93-95% of points
said by that person, 0-0.5% by someone else, 5-7% unsupported or twisted. The instructions
were developed on these meetings, and the same model writes and judges the summaries. In a
manual check of 30 judged points (marked with an LLM assistant, reviewed by the author), the
judge was right on 85%.

## When the LLM fails

Translation and summaries run last. A failed batch is split in half and retried, and if the
LLM is down entirely the transcript is still written; only the translation or summary is
missing. A rejected API key, an empty account balance or an unknown model name stops at
once instead of retrying.

## Why calibration matters

Plugins report confidence on different scales. Calibration maps each plugin's raw score to
the observed probability of being right on labeled data (`speakerlens calibrate`), so "0.8"
means the same thing for every plugin, measured with expected calibration error (ECE). It
matters for routing rule 6. Fun-ASR-Nano and SenseVoice report no confidence, so with the
default GPU setup the rule rarely applies; it is used by plugins that do report one
(Whisper, third-party plugins).

## Measured and not kept

All measured on the same data as above; tcpWER unless noted.

| Idea | Result | Why not kept |
| --- | --- | --- |
| Separating overlapping voices with pyannote PixIT | AliMeeting Eval 0.303 -> 0.299; recognizing overlapped turns from its tracks made them worse (heavy overlap 0.61 -> 0.64) | Extraction does much better (0.278), and PixIT needs fragile extra packages |
| PixIT as the only diarizer | one AliMeeting meeting 0.395 -> 0.677 | Much worse |
| Fine-tuning Fun-ASR-Nano on AliMeeting (audio adaptor, 115k clips) | AliMeeting Eval 0.275 -> 0.290, ASCEND 0.175 -> 0.212 | Worse everywhere, worst on English and mixed speech |
| Extraction settings: 3 s context, 40 s voice samples, "typical" turns as the voice sample, extracting every turn | 0.278-0.281 vs 0.278 | No gain |
| Headset recordings as voice samples on AliMeeting | 0.282 vs 0.278 | No gain |
| Extra audio around turns, other echo and minimum-length settings for hidden speech | 0.298-0.299 vs 0.299 | No gain |
| Removing room echo (WPE, one microphone) | AliMeeting 0.275 -> 0.275 (recognition only), 0.283 (everything); found 0.00-0.08 dB to remove | No gain; WPE needs several microphones |

## Real recordings

Two scripted 3-person Mandarin-English recordings on one phone (mono), no voice samples,
`--speakers 3`, judged against role cards (what each person was meant to argue):

- **Turn-taking discussion in an office:** speakers kept apart and consistent; summaries
  matched the role cards, including the direction of agreements.
- **Debate with planned interruptions:** the person who mostly talked over others had all 3
  of their overlapped lines that mattered attributed to the person underneath (diarization
  gives overlapped stretches to the dominant voice), which put a reversed point into that
  person's summary. Extraction can't fix this: the lines are assigned before it runs.
- **Most common error:** English words inside Mandarin sentences (hiking -> "fighting",
  intern -> "Intel"), a limit of the speech model.

## Not yet built

- Live mode.
- Web viewer.
- Confidence scores and word timestamps for SenseVoice and Fun-ASR-Nano.
