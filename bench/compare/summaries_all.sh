#!/usr/bin/env bash
# Summarize and judge each system's transcripts of one set with the same summarizer and
# judge (bench/eval_summaries.py, reading every system's <id>.stm). Systems run one after
# another so they share one set of reference summaries. From the repo root, in the
# speakerlens environment, with DEEPSEEK_API_KEY in .env:
#   SET=ali_test BENCH=bench/heldout bash bench/compare/summaries_all.sh
set -u
cd "$(dirname "$0")/../.."
source bench/compare/sets.sh
SET=${SET:-ali_eval}
BENCH=${BENCH:-bench}
LOG=${LOG:-logs/compare}
SYSTEMS=${SYSTEMS:-"speakerlens whisperx_zh_words vibevoice"}
IFS='|' read -r data spk <<< "$(set_info "$SET")" || exit 1
mkdir -p "$LOG"
for sys in $SYSTEMS; do
  echo; echo "=== $SET / $sys (summaries)"
  python -u bench/eval_summaries.py "$data" "$BENCH/$sys" --stm --out "$BENCH/summaries" \
    > "$LOG/summaries_${SET}_$sys.log" 2>&1
  tail -n 7 "$LOG/summaries_${SET}_$sys.log"
done
