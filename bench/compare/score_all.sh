#!/usr/bin/env bash
# Score every system on the benchmark sets with the same code. From the repo root, in the
# speakerlens environment:
#   bash bench/compare/score_all.sh
#   SETS="ali_test" BENCH=bench/heldout SYSTEMS="speakerlens vibevoice" bash bench/compare/score_all.sh
# A system is a folder <BENCH>/<system>/<dataset folder name>/ with <id>.stm and <id>.rttm
# per file. Prints DER (all speech, and without overlapped stretches), cpWER, tcpWER,
# tcORC-WER and the share of words recognized by language.
set -u
cd "$(dirname "$0")/../.."
source bench/compare/sets.sh
SETS=${SETS:-"ascend_real ascend_meetings ali_eval"}
BENCH=${BENCH:-bench}
SYSTEMS=${SYSTEMS:-"speakerlens speakerlens_nospk whisperx_auto whisperx_auto_words whisperx_zh whisperx_zh_words vibevoice"}
check_sets $SETS || exit 1
for s in $SETS; do
  IFS='|' read -r data spk <<< "$(set_info "$s")"
  folder=$(basename "$data")
  for sys in $SYSTEMS; do
    [ -d "$BENCH/$sys/$folder" ] || continue
    echo; echo "=== $s / $sys"
    python bench/run_bench.py $data $spk --out "$BENCH/$sys" --score-only 2>/dev/null | grep overall \
      || echo "  (scoring failed: run bench/run_bench.py --score-only without 2>/dev/null)"
    python bench/error_breakdown.py $data --run "$BENCH/$sys" 2>/dev/null | grep -A6 "Recognized words"
  done
done
