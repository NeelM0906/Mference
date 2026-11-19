#!/bin/bash
# run_baselines.sh: the llama.cpp / ggml-metal baselines of
# docs/NATIVE_BENCHMARKS.md. Every step is one command in its own quiet hold
# of the machine lock (lock.sh run --quiet: no build or test starts during it),
# through guarded-run.sh --timing (AC power only, swap guard, peak footprint).
# Steps whose run.json says "valid": true are skipped, so a stopped run
# resumes; a void or older record is re-run.
#
#   LOCK=<lock.sh> BUILD=<build dir> GGUF=<Quick gguf> TOKENS=<golden long-32k golden.json> \
#   GGML_METAL_PATH_RESOURCES=<llama.cpp>/ggml/src/ggml-metal \
#   Scripts/native/bench/run_baselines.sh <out-dir> [step...]
#
# Steps (default: all): llama-bench-pp-tg llama-bench-pp32k llama-bench-tg-depth
# mfe-bench-ttft kbench-quick kbench-deep-iq sched-bench
#
# sched-bench runs SchedulerBenchTests' 10 ms reference loops
# (MFERENCE_NATIVE_BENCH=1). Its test bundle is built first, outside the lock
# (lock.sh build, one call), and the timed run uses --skip-build, so nothing
# builds inside the quiet hold. Run it from the repository's own worktree.
set -uo pipefail
: "${LOCK:?set LOCK}" "${BUILD:?set BUILD}" "${GGUF:?set GGUF}" "${TOKENS:?set TOKENS}" "${GGML_METAL_PATH_RESOURCES:?set GGML_METAL_PATH_RESOURCES}"
export LOCK GGML_METAL_PATH_RESOURCES
OUT=${1:?usage: run_baselines.sh <out-dir> [step...]}; shift
mkdir -p "$OUT"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
GR="$OUT/guarded-run.sh"
cp "$HERE/../guarded-run.sh" "$GR"   # a frozen copy: never edited while it runs
WS=${WS:-ws2}
steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(llama-bench-pp-tg llama-bench-pp32k llama-bench-tg-depth mfe-bench-ttft kbench-quick kbench-deep-iq sched-bench)
REPO=$(cd "$HERE/../../.." && pwd -P)
PLUG=(-Xswiftc -plugin-path -Xswiftc /Library/Developer/CommandLineTools/usr/lib/swift/host/plugins/testing)

# The host and code a result was measured on, written into each step's
# directory when it runs (never a shared file overwritten on resume).
host_info() {
    { echo "utc $(date -u +%FT%TZ)"
      echo "mference $(git -C "$REPO" rev-parse HEAD) dirty=$(git -C "$REPO" status --porcelain | wc -l | tr -d ' ')"
      system_profiler SPHardwareDataType 2>/dev/null | grep -E "Model Name|Chip|Memory"
      system_profiler SPDisplaysDataType 2>/dev/null | grep -E "Chipset Model|Metal (Support|Family)|Total Number of Cores"
      sw_vers; swift --version 2>&1 | head -1; clang --version | head -1
      pmset -g batt | head -1; pmset -g therm 2>/dev/null | grep -v "^$"
      cat "$BUILD/BUILD_INFO.txt" 2>/dev/null; } > "$1/host.txt"
}

# Spec section 10: let running build/test jobs finish before taking a quiet
# hold (guarded-run --timing checks again inside it).
wait_for_no_jobs() {
    while [ "$(pgrep -f '/usr/bin/swift-(build|test) ' | wc -l | tr -d ' ')" != 0 ]; do sleep 30; done
}

run_step() {
    local name=$1; shift
    # done only when the wrapper judged the run valid (exit 0, no swap growth
    # during or by the end); a void or older record is re-run
    if [ -f "$OUT/$name/run.json" ] && grep -q '"valid": true' "$OUT/$name/run.json"; then echo "$name: done"; return 0; fi
    rm -rf "${OUT:?}/$name"
    wait_for_no_jobs
    # a timing run: --quiet so no build or test starts during it (spec section 10)
    "$LOCK" run --quiet --wait 7200 "$WS" "mfe bench $name" -- "$GR" --timing "$OUT/$name" "$GGUF" -- "$@"
    local rc=$?
    [ -d "$OUT/$name" ] && host_info "$OUT/$name"
    echo "$name exit=$rc" >> "$OUT/summary.txt"
    return $rc
}

for s in "${steps[@]}"; do
    case "$s" in
        llama-bench-pp-tg) run_step "$s" "$BUILD/llama-bench" -m "$GGUF" -ngl 99 -p 512,4096 -n 128 -r 5 -o json ;;
        llama-bench-pp32k) run_step "$s" "$BUILD/llama-bench" -m "$GGUF" -ngl 99 -p 32768 -n 0 -r 2 -o json ;;
        llama-bench-tg-depth) run_step "$s" "$BUILD/llama-bench" -m "$GGUF" -ngl 99 -p 0 -n 128 -d 4096,32768 -r 3 -o json ;;
        mfe-bench-ttft) run_step "$s" "$BUILD/mfe-bench" --model "$GGUF" --tokens "$TOKENS" --ctx 512,4096,32768 \
                            --gen 128 --suffix 64 --reps 3 --json "$OUT/$s.json" ;;
        kbench-quick) run_step "$s" "$BUILD/mfe-kbench" --suite dispatch,copy,gemm,attn,gdn --rows 1,2,4,8,16,512 \
                          --ctx 512,4096,32768 --reps 20 --json "$OUT/$s.json" ;;
        kbench-deep-iq) run_step "$s" "$BUILD/mfe-kbench" --suite dispatch,copy,gemm --types iq3_s,iq3_xxs,iq4_xs \
                            --shapes 5120:12288,5120:10240,5120:6144,5120:1024,6144:5120,5120:17408,17408:5120 \
                            --rows 1,2,4,8,16 --reps 20 --json "$OUT/$s.json" ;;
        sched-bench)
            # build the test bundle outside the lock first: one lock.sh build call
            # of the model-free SchedulerBenchTests (without MFERENCE_NATIVE_BENCH
            # they are short), so the timed --skip-build run reuses exactly
            # Scripts/test.sh's flags; at most two build/test jobs machine-wide
            while [ "$(pgrep -f '/usr/bin/swift-(build|test) ' | wc -l | tr -d ' ')" -ge 2 ]; do sleep 30; done
            (cd "$REPO" && "$LOCK" build -- Scripts/test.sh --filter SchedulerBenchTests "${PLUG[@]}") > "$OUT/sched-bench-build.log" 2>&1 \
                || { echo "sched-bench build failed (see sched-bench-build.log)" >&2; exit 1; }
            run_step "$s" env MFERENCE_NATIVE_BENCH=1 "$REPO/Scripts/test.sh" --skip-build --filter SchedulerBenchTests "${PLUG[@]}" ;;
        *) echo "unknown step $s" >&2; exit 2 ;;
    esac || exit $?
done
