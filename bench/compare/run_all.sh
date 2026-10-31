#!/usr/bin/env bash
# Run speakerlens, WhisperX (language detected, and Mandarin set) and VibeVoice-ASR on the
# benchmark sets, then score them all with score_all.sh. Needs the speakerlens, whisperx and
# vibevoice conda environments (bench/compare/README.md). From the repo root, inside tmux:
#   WX_GPU=2 SL_GPUS="3 4 5 6 7" VV_GPUS=5,6 bash bench/compare/run_all.sh
# Options (environment variables):
#   SETS         sets from sets.sh (default: the development sets)
#   RUN          systems to run (default: speakerlens whisperx vibevoice); speakerlens_nospk
#                runs speakerlens without the number of speakers
#   BENCH        results folder (default: bench); LOG: log folder (default: logs/compare)
#   MODEL_CACHE  download folder for VibeVoice-ASR (about 17 GB)
set -u
source ~/miniconda3/etc/profile.d/conda.sh
cd "$(dirname "$0")/../.."
source bench/compare/sets.sh
SETS=${SETS:-"ascend_real ascend_meetings ali_eval"}
RUN=${RUN:-"speakerlens whisperx vibevoice"}
BENCH=${BENCH:-bench}
LOG=${LOG:-logs/compare}
WX_GPU=${WX_GPU:-2}
SL_GPUS=${SL_GPUS:-"3 4 5 6 7"}
VV_GPUS=${VV_GPUS:-5,6}
WX_BATCH=${WX_BATCH:-8}
read -ra SET_NAMES <<< "$SETS"
check_sets "${SET_NAMES[@]}" || exit 1
mkdir -p "$LOG"
set -a; source <(grep -E '^(HF_TOKEN|HF_ENDPOINT)=' .env); set +a
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=0
want() { [[ " $RUN " == *" $1 "* ]]; }

run_speakerlens() {  # system name, then "with" or "without" the number of speakers
  local system=$1 count=$2 s data spk k
  read -ra G <<< "$SL_GPUS"
  for s in "${SET_NAMES[@]}"; do
    IFS='|' read -r data spk <<< "$(set_info "$s")"
    [ "$count" = without ] && spk=""
    echo "$(date +%H:%M) $system $s"
    for k in $(seq 1 ${#G[@]}); do
      TSE_MODEL= CUDA_VISIBLE_DEVICES=${G[$((k - 1))]} python bench/run_bench.py $data $spk \
        --out "$BENCH/$system" --shard $k/${#G[@]} --resume \
        > "$LOG/${system}_${s}_$k.log" 2>&1 &
    done
    wait
    if grep -l "Speaker extraction failed" "$LOG/${system}_${s}"_*.log 2>/dev/null; then
      echo "$(date +%H:%M) ERROR: speaker extraction failed in the logs above; stopping $system"
      return 1
    fi
  done
}

if want speakerlens || want speakerlens_nospk; then
  (
    conda activate speakerlens
    # Fetch the default extraction model now: offline mode would block it later and
    # overlapped speech would fall back to the original audio.
    if ! TSE_MODEL= python -c "from speakerlens.config import load_settings; \
from speakerlens.extraction import download_model; \
print('extraction model:', download_model(load_settings().cache_dir / 'models' / 'wesep_bsrnn'))"; then
      echo "$(date +%H:%M) speakerlens: could not get the extraction model; not run"
      exit 1
    fi
    export HF_HUB_OFFLINE=1
    if want speakerlens; then run_speakerlens speakerlens with || exit 1; fi
    if want speakerlens_nospk; then run_speakerlens speakerlens_nospk without || exit 1; fi
  ) &
fi
if want whisperx; then
  (
    conda activate whisperx
    for s in "${SET_NAMES[@]}"; do
      IFS='|' read -r data spk <<< "$(set_info "$s")"
      for mode in auto zh; do
        echo "$(date +%H:%M) whisperx $mode $s"
        lang=$([ $mode = zh ] && echo "--language zh")
        CUDA_VISIBLE_DEVICES=$WX_GPU python bench/compare/run_whisperx.py $data $spk $lang \
          --batch-size $WX_BATCH --out "$BENCH/whisperx_$mode" \
          --word-out "$BENCH/whisperx_${mode}_words" > "$LOG/whisperx_${mode}_$s.log" 2>&1
      done
    done
  ) &
fi
if want vibevoice; then
  (
    conda activate vibevoice
    if [ -n "${MODEL_CACHE:-}" ]; then mkdir -p "$MODEL_CACHE" && export HF_HUB_CACHE="$MODEL_CACHE"; fi
    for s in "${SET_NAMES[@]}"; do
      IFS='|' read -r data spk <<< "$(set_info "$s")"
      echo "$(date +%H:%M) vibevoice $s"
      CUDA_VISIBLE_DEVICES=$VV_GPUS python bench/compare/run_vibevoice.py $data --resume \
        --out "$BENCH/vibevoice" > "$LOG/vibevoice_$s.log" 2>&1
    done
  ) &
fi
wait
echo "$(date +%H:%M) runs done"

conda activate speakerlens
SETS="$SETS" BENCH="$BENCH" bash bench/compare/score_all.sh
