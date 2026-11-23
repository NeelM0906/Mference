#!/bin/bash
# run_held_stages.sh: every K0 model stage that waits on "model runs cleared",
# in order, each as its own lock hold; stops at the first nonzero exit (70 is a
# swap-guard trip: stop and report). Run it only after the coordinator clears
# model runs.
#
#   run_held_stages.sh [stage...]     (default: all, in the order below)
#
# Order: the goldens first (not timing; any power), then the baseline timing
# stages shortest first (--quiet, AC power, <= 1 other swift job).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=paths.sh
. "$HERE/paths.sh"
B=$INK/ref/llama-build-b12641
STAGES=("$@")
[ ${#STAGES[@]} -gt 0 ] || STAGES=(golden:metal golden:metal-r2 golden:cpu
  base:bench-512 base:text-512 base:bench-4k base:text-4k base:bench-16k base:text-16k)
for s in "${STAGES[@]}"; do
  kind=${s%%:*}; stage=${s#*:}
  # preflight fails while any model-free test helper runs; wait for a pass
  until "$LOCK" preflight > /dev/null 2>&1; do sleep 20; done
  if [ "$kind" = golden ]; then
    "$LOCK" run --wait 3600 ws0 "ink golden $stage" -- "$HERE/golden_stage.sh" "$B" "$INK/goldens" "$stage"
  else
    "$LOCK" run --quiet --wait 3600 ws0 "ink baseline $stage" -- "$HERE/baseline_stage.sh" "$B" "$INK/results/k0/baseline" "$stage"
  fi
  rc=$?
  echo "$(date -u +%FT%TZ) $s exit $rc"
  [ $rc = 0 ] || exit $rc
done
