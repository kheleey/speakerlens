# Benchmark sets used by run_all.sh and score_all.sh: name -> "dataset folder|speaker option".
# Development sets (settings were chosen on these): ascend_real, ascend_meetings, ali_eval.
# Held-out sets (never used for any choice): heldout_real, ali_test.
set_info() {
  case "$1" in
    ascend_real) echo "data/dev/ascend_real|--speakers 2" ;;
    ascend_meetings) echo "data/dev/ascend_meetings|--speakers 4" ;;
    ali_eval) echo "data/ali_eval|--speakers-from-ref" ;;
    heldout_real) echo "data/heldout_real|--speakers 2" ;;
    ali_test) echo "data/ali_test|--speakers-from-ref" ;;
    *) echo "unknown set: $1" >&2; return 1 ;;
  esac
}

# Results of a set go to <system folder>/<dataset folder name>, so two sets whose folders
# share a name must not be run into the same BENCH folder.
check_sets() {
  local names=() s info
  for s in "$@"; do
    info=$(set_info "$s") || return 1
    names+=("$(basename "${info%%|*}")")
  done
  if [ "$(printf '%s\n' "${names[@]}" | sort | uniq -d)" ]; then
    echo "these sets share a folder name; run them with different BENCH folders: $*" >&2
    return 1
  fi
}
