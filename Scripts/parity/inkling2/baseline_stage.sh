#!/bin/bash
# baseline_stage.sh: one stage of the K0 reference baseline (spec section 9,
# checkpoint C0) on stock upstream llama.cpp, through guarded_run.sh --timing.
#
#   baseline_stage.sh <build-dir> <out-dir> <stage>
#
# Stages (run shortest first, one per `model-hold.sh --timing` hold; stop at the
# first swap growth, exit 70):
#   bench-512    llama-bench -p 512   -n 128 -r 3 (synthetic tokens; prefill and decode tok/s)
#   bench-4k     llama-bench -p 4096  -n 0   -r 3
#   bench-16k    llama-bench -p 16384 -n 0   -r 2
#   text-512     llama-completion on the real-text prompt prompt-512.txt,  128 new tokens (TTFT)
#   text-4k      llama-completion on prompt-4k.txt,  64 new tokens
#   text-16k     llama-completion on prompt-16k.txt, 32 new tokens
# Every stage: the default GPU offload (-ngl 99; NVFP4 products run on the
# CPU in this build), default batch sizes, mmap, no flash-attention override,
# greedy sampling for the text stages.
set -uo pipefail
B=${1:?build dir}; OUT=${2:?out dir}; STAGE=${3:?stage}
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/paths.sh"; GGUF=$INKLING_GGUF
P=$OUT/prompts
case "$STAGE" in
  bench-512) cmd=("$B/llama-bench" -m "$GGUF" -ngl 99 -p 512 -n 128 -r 3 -o md) ;;
  bench-4k)  cmd=("$B/llama-bench" -m "$GGUF" -ngl 99 -p 4096 -n 0 -r 3 -o md) ;;
  bench-16k) cmd=("$B/llama-bench" -m "$GGUF" -ngl 99 -p 16384 -n 0 -r 2 -o md) ;;
  text-512)  cmd=("$B/llama-completion" -m "$GGUF" -ngl 99 -c 4096  -f "$P/prompt-512.txt" -n 128 --temp 0 --seed 7 -no-cnv --no-display-prompt) ;;
  text-4k)   cmd=("$B/llama-completion" -m "$GGUF" -ngl 99 -c 8192  -f "$P/prompt-4k.txt"  -n 64  --temp 0 --seed 7 -no-cnv --no-display-prompt) ;;
  text-16k)  cmd=("$B/llama-completion" -m "$GGUF" -ngl 99 -c 20480 -f "$P/prompt-16k.txt" -n 32  --temp 0 --seed 7 -no-cnv --no-display-prompt) ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
# Coordinator rule (2026-11-23): a timing stage runs with at most one other
# swift build/test job on the machine. The quiet flag (set by model-hold.sh)
# stops new builds; wait up to 30 min for running jobs to drop to <= 1, else
# refuse (exit 71). Nothing is stopped. The counts are recorded for baseline.md.
mkdir -p "$OUT/$STAGE"
swift_jobs() { pgrep -f '/usr/bin/swift-(build|test) ' | wc -l | tr -d ' '; }
n=$(swift_jobs)
for _ in $(seq 1 180); do
  [ "$n" -le 1 ] && break
  sleep 10; n=$(swift_jobs)
done
echo "$(date -u +%FT%TZ) swift build/test jobs at start: $n" | tee -a "$OUT/$STAGE/jobs.log"
[ "$n" -le 1 ] || { echo "REFUSED: $n swift jobs still running after 30 min" | tee -a "$OUT/$STAGE/jobs.log"; exit 71; }
"$HERE/guarded_run.sh" --timing "$OUT/$STAGE" "$GGUF" -- "${cmd[@]}"
rc=$?
echo "$(date -u +%FT%TZ) swift build/test jobs at end: $(swift_jobs)" | tee -a "$OUT/$STAGE/jobs.log"
exit $rc
