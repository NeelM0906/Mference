#!/bin/bash
# golden_stage.sh: one stage of the K0 weight-loading goldens (greedy
# continuations, top-20 log-probabilities, the 48-visit trace), through
# guarded_run.sh (swap guard; not a timing run, the power source is recorded).
#
#   golden_stage.sh <build-dir> <goldens-dir> <stage>
#
# Stages, each its own lock hold:
#   metal      ink-golden generate, -ngl 99 (the default GPU offload): the goldens
#   metal-r2   the same again in a fresh process: the reference's run-to-run noise
#   cpu        the same with -ngl 0: ink-golden then loads no GPU device, turns
#              op offload and CPU weight repacking off, so every op runs on the
#              CPU from the mapped file: the backend noise
# All three use inputs/generate_prompts.json, n_ctx 4096, n_ubatch 512, 8 threads.
set -uo pipefail
B=${1:?build dir}; G=${2:?goldens dir}; STAGE=${3:?stage}
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/paths.sh"; GGUF=$INKLING_GGUF
case "$STAGE" in
  metal|metal-r2) ngl=99 ;;
  cpu) ngl=0 ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
exec "$HERE/guarded_run.sh" "$G/generate/$STAGE/log" "$GGUF" -- \
  "$B/ink-golden" generate --model "$GGUF" --in "$HERE/inputs/generate_prompts.json" \
  --out "$G/generate/$STAGE" --ngl "$ngl" --threads 8 --n-ctx 4096 --ubatch 512 --top 20
