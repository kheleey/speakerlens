# Comparing with other systems

Runs other open-source systems on the same test sets and writes their output in the
layout `bench/run_bench.py --score-only` and `bench/error_breakdown.py --run` read, so
every system is scored by the same code.

| System | What it is | Script |
| --- | --- | --- |
| [WhisperX](https://github.com/m-bain/whisperX) 3.8 | Whisper large-v3 + pyannote community-1 diarization (the same diarizer speakerlens starts from) | `run_whisperx.py` |
| [VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR) | one 8.7B model for transcription and who-spoke-when, MIT license | `run_vibevoice.py` |

Each runs as a typical user would run it: no hints about the test (no word lists), the
true number of speakers given where the system accepts it (speakerlens and WhisperX yes,
VibeVoice-ASR has no such option). speakerlens runs with its default extraction model
(`TSE_MODEL` unset), as a new user gets it. Traditional Chinese characters are converted to
Simplified before scoring.

WhisperX is scored two ways from the same run: `whisperx_auto` / `whisperx_zh` give each
segment one speaker, as WhisperX's own transcript and subtitle output does;
`*_words` split segments wherever its per-word speakers change, the finest labels it gives.
Its full result is kept as `<id>.whisperx.json`. VibeVoice-ASR transcribes each recording in
one pass; when its output is cut off (it can repeat itself until its output limit), the
complete turns are kept and the log says how far they reach.

DER is reported twice: over all speech (`der`), and leaving out the stretches where the
reference has several people talking at once (`der_noov`). Systems that write one speaker
at a time can't get the people talking underneath, so the first charges them for it and the
second shows how good their speaker turns are otherwise.

Each system needs its own environment, because they pin different library versions than
speakerlens (WhisperX: torch 2.8; VibeVoice-ASR: transformers 5.3 or newer):

```bash
conda create -n whisperx python=3.11 -y && conda activate whisperx
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu126
pip install whisperx==3.8.6 zhconv

conda create -n vibevoice python=3.11 -y && conda activate vibevoice
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install "transformers>=5.3.0" accelerate soundfile librosa zhconv
```

WhisperX needs `HF_TOKEN` (pyannote) in the environment, and NLTK's `punkt_tab` for word
timing (`python -c "import nltk; nltk.download('punkt_tab')"`; NLTK refuses an
`~/nltk_data` folder other accounts can write to: `chmod -R go-w ~/nltk_data`). Then, for
example:

```bash
conda activate whisperx
CUDA_VISIBLE_DEVICES=2 python bench/compare/run_whisperx.py data/ali_eval --speakers-from-ref \
    --out bench/whisperx --word-out bench/whisperx_words
conda activate vibevoice
CUDA_VISIBLE_DEVICES=2,3 python bench/compare/run_vibevoice.py data/ali_eval --out bench/vibevoice

conda activate speakerlens   # scoring uses speakerlens's metrics
TSE_MODEL= CUDA_VISIBLE_DEVICES=2 python bench/run_bench.py data/ali_eval --speakers-from-ref \
    --out bench/speakerlens
python bench/run_bench.py data/ali_eval --out bench/whisperx --score-only
python bench/error_breakdown.py data/ali_eval --run bench/whisperx
```

All three accept `--shard k/n` to split a set across GPUs.

Check speakerlens's logs for "Speaker extraction failed". If the default extraction model
can't be downloaded (for example with `HF_HUB_OFFLINE=1` set before its first download),
speakerlens carries on with the original audio for overlapped speech, and the scores no
longer measure the real pipeline.

`run_all.sh` runs all systems on a list of sets from `sets.sh` (WhisperX twice: language
detected, as by default, and Mandarin set), then scores everything with `score_all.sh`. It
downloads the extraction model before running speakerlens, and stops speakerlens if any of
its logs reports a failed extraction.

```bash
# development sets: ASCEND conversations and meetings, AliMeeting Eval
tmux new -d -s compare 'WX_GPU=2 SL_GPUS="3 4 5 6 7" VV_GPUS=5,6 bash bench/compare/run_all.sh > logs/compare.log 2>&1'

# held-out sets, plus speakerlens without the number of speakers (as VibeVoice-ASR runs)
SETS="heldout_real ali_test" BENCH=bench/heldout LOG=logs/heldout \
RUN="speakerlens speakerlens_nospk whisperx vibevoice" bash bench/compare/run_all.sh
```

`RUN` picks systems; `score_all.sh` (same `SETS` and `BENCH`) rescores whatever has run.
Results go to `<BENCH>/<system>/<dataset folder name>`, so sets whose dataset folders share
a name need different `BENCH` folders; both scripts refuse to mix them.

`summaries_all.sh` gives every system's transcript of one set to the same summarizer and
judge (`bench/eval_summaries.py --stm`), one system after another so they share the
reference summaries:

```bash
SET=ali_test BENCH=bench/heldout bash bench/compare/summaries_all.sh
```
