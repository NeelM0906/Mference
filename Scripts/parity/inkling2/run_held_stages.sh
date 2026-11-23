#!/bin/bash
# run_held_stages.sh: every K0 model stage that waits on "model runs cleared",
# in order, each as its own lock hold; stops at the first nonzero exit (70 is a
# swap-guard trip: stop and report). Run it only after the coordinator clears
# model runs.
#
#   run_held_stages.sh [stage...]     (default: all, in the order below)
#
# Order: the goldens first (not timing; any power), then the baseline timing
# stages shortest first (model-hold.sh --timing: quiet, AC power, builds drained).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=paths.sh
. "$HERE/paths.sh"
B=$INK/ref/llama-build-b12641
STAGES=("$@")
[ ${#STAGES[@]} -gt 0 ] || STAGES=(golden:metal golden:metal-r2 golden:cpu
  base:bench-512 base:text-512 base:bench-4k base:text-4k base:bench-16k base:text-16k)
# Spec v1.3 section 7: every model run starts through the program's
# model-hold.sh (lock + quiet flag, drains running test jobs inside the hold;
# --timing also drains builds and needs AC power). guarded_run.sh keeps its
# 2-second swap sampling inside it.
MODEL_HOLD=${MODEL_HOLD:-$INK/bin/model-hold.sh}
for s in "${STAGES[@]}"; do
  kind=${s%%:*}; stage=${s#*:}
  if [ "$kind" = golden ]; then
    "$MODEL_HOLD" ws0 "golden $stage" -- "$HERE/golden_stage.sh" "$B" "$INK/goldens" "$stage"
  else
    "$MODEL_HOLD" --timing ws0 "baseline $stage" -- "$HERE/baseline_stage.sh" "$B" "$INK/results/k0/baseline" "$stage"
  fi
  rc=$?
  echo "$(date -u +%FT%TZ) $s exit $rc"
  [ $rc = 0 ] || exit $rc
done
