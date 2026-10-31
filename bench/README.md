# Benchmarks

The scripts behind the numbers in the main README. They are not part of the installed tool.

```bash
pip install -e ".[all,bench]"
```

## Getting the data

```bash
# ASCEND (Hugging Face; set HF_ENDPOINT in China)
hf download CAiRE/ASCEND --repo-type dataset --local-dir data/ascend

# AliMeeting Eval and Test (3.5 GB and 8.9 GB; also listed at https://www.openslr.org/119/)
mkdir -p data/alimeeting && cd data/alimeeting
wget -c https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Eval_Ali.tar.gz
wget -c https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Test_Ali.tar.gz
tar xzf Eval_Ali.tar.gz && tar xzf Test_Ali.tar.gz
cd ../..
```

## Test sets

Each dataset folder needs `audio/<id>.wav` plus `ref/<id>.rttm` (who spoke when) and
`ref/<id>.stm` (who said what). Recordings are git-ignored; check each dataset's license
before downloading.

| Script | Builds | Source |
| --- | --- | --- |
| `make_ascend.py` | Mandarin–English conversations (real speaker pairs) and simulated 4-person meetings | [ASCEND](https://huggingface.co/datasets/CAiRE/ASCEND), CC-BY-SA-4.0 |
| `make_voices.py` | voice samples for those files, from each person's *other* clips | ASCEND |
| `make_alimeeting.py` | real Mandarin meetings, one distant microphone, heavy overlap | [AliMeeting](https://www.openslr.org/119/) Eval/Test, CC-BY-SA-4.0 |

## Measuring

| Script | Measures |
| --- | --- |
| `run_bench.py` | the full pipeline: DER (who spoke when; also without overlapped stretches), cpWER, tcpWER, tcORC-WER |
| `error_breakdown.py` | where errors come from: an *oracle* run on the true speaker segments (recognition only, split by how much overlap); misheard / missed / extra words of any run; and recognized words split by language: English inside Mandarin, Mandarin inside English |
| `eval_summaries.py` | per-person summaries against the human transcript, with an LLM judge: said by that person / by someone else / by nobody, and coverage |

Examples:

```bash
# AliMeeting: development (Eval) and held-out (Test) sets
python bench/make_alimeeting.py data/alimeeting/Eval_Ali --out data/ali_eval
python bench/make_alimeeting.py data/alimeeting/Test_Ali --out data/ali_test
python bench/run_bench.py data/ali_eval --speakers-from-ref --out bench/results
python bench/error_breakdown.py data/ali_eval --run bench/results
python bench/eval_summaries.py data/ali_eval bench/results --limit 4

# ASCEND: conversations and simulated meetings from the train split
# (writes data/dev/ascend_real and data/dev/ascend_meetings)
python bench/make_ascend.py --src data/ascend/main --split train --out data/dev \
    --max 20 --meetings 10
python bench/make_voices.py --src data/ascend/main --split train data/dev/ascend_meetings

# Two of the conversations are held out: their speakers appear in no other conversation
mkdir -p data/heldout_real/audio data/heldout_real/ref
for id in ascend_ses4_spk1_spk21 ascend_ses4_spk2_spk24; do
  mv data/dev/ascend_real/audio/$id.wav data/heldout_real/audio/
  mv data/dev/ascend_real/ref/$id.* data/heldout_real/ref/
done

python bench/run_bench.py data/dev/ascend_real --speakers 2
python bench/run_bench.py data/dev/ascend_meetings --speakers 4 \
    --voices-root data/dev/ascend_meetings/voices
```

Held-out sets, never used for any choice: `data/ali_test` (AliMeeting Test) and
`data/heldout_real`. ASCEND's test split has too few people for 4-person meetings with four
different voices, so there are no held-out simulated meetings.

The README's meeting numbers come from 8 of the 10 dev meetings. An earlier version of
`make_ascend.py` took a person recorded in two sessions for two people, which put the same
voice in meetings 04 and 09 twice; those two are left out. The fixed script draws a
different set of meetings, so a rebuilt dev set won't give exactly these numbers.

With several GPUs, run one copy per GPU (`CUDA_VISIBLE_DEVICES=k python bench/run_bench.py
... --shard k/n`), then once more with `--score-only`. `--resume` skips files a crashed run
already finished. Run-to-run differences of about ±0.003 tcpWER are noise.

To compare with other systems (WhisperX, VibeVoice-ASR) on the same sets and scoring,
including their per-person summaries, see [compare/](compare/README.md).
