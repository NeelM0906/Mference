#!/bin/bash
# guarded-run.sh: one model command for the native engine harness (golden
# capture, llama.cpp baselines, kernel benchmarks), with the program's machine
# rules (scratch/mference-engine-program spec, section 10) applied around it.
#
#   guarded-run.sh [--timing] <log-dir> <gguf> -- <command> [args...]
#
# Run it as the single command of its own lock hold:
#   lock.sh run --wait 3600 ws2 "mfe <purpose>" -- Scripts/native/guarded-run.sh <log-dir> <gguf> -- ...
#
# Power: every run records its power source in run.json. --timing marks a
# timing run (benchmark baselines, kernel benchmarks, L5 gates, any tok/s,
# TTFT or kernel figure that is compared or published): it refuses unless on
# AC Power, so every compared number shares a power source. Correctness,
# parity and capture runs may run on battery (program spec section 10).
# Before: refuses unless `lock.sh preflight`
# passes (no --model: a GGUF is not an install directory), the GGUF exists and
# no other model process runs. During: samples swap every 5 s; any growth
# (the program's swap guard; SWAP_GUARD_MB may only be 0) stops the command
# (INT, then TERM after 30 s) and exits 70; growth seen only at the end also
# exits 70. After: records swap at start and end, the exit code, wall time
# and /usr/bin/time -l's peak memory footprint in <log-dir>/run.json.
# Exit status: the command's; 70 swap guard; 71 a precondition failed.
set -uo pipefail
# LOCK: the machine lock script (the program's scratch/program/lock.sh).
LOCK=${LOCK:?set LOCK to the machine lock script}
SWAP_GUARD_MB=${SWAP_GUARD_MB:-0}
# The program's guard is "any growth": refuse anything looser.
case "$SWAP_GUARD_MB" in 0|0.0) ;; *) echo "guarded-run.sh: SWAP_GUARD_MB must be 0 (any swap growth stops the run)" >&2; exit 71 ;; esac

timing=0
if [ "${1:-}" = "--timing" ]; then timing=1; shift; fi
[ $# -ge 4 ] && [ "$3" = "--" ] || { echo "usage: guarded-run.sh [--timing] <log-dir> <gguf> -- <command> [args...]" >&2; exit 2; }
LOG=$1; GGUF=$2; shift 3
mkdir -p "$LOG"
note() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG/run.log" >&2; }
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/'; }

power=$(pmset -g batt | head -1 | sed -E "s/.*'(.*)'.*/\1/")
if [ $timing = 1 ] && [ "$power" != "AC Power" ]; then note "REFUSED: timing run not on AC power ($power)"; exit 71; fi
note "power source: $power$([ $timing = 1 ] && echo ' (timing run)')"
if [ ! -f "$GGUF" ]; then note "REFUSED: no GGUF at $GGUF"; exit 71; fi
if ! "$LOCK" preflight >> "$LOG/run.log" 2>&1; then note "REFUSED: lock.sh preflight failed (see run.log)"; exit 71; fi
# Processes by exact name (a shell whose command line merely mentions a tool
# must not count). lock.sh preflight matches process names too, so a test
# bundle run by a bare `xctest .../MferencePackageTests.xctest` (process name
# "xctest") escapes both name checks: it is caught here by its command line,
# anchored at the start so a shell naming the bundle does not match.
others=$( { pgrep -lx 'llama-server|llama-cli|llama-bench|mfe-golden|mfe-kbench|mfe-bench|MferenceServer|MferenceCLI'
            pgrep -lf '^[^ ]*xctest .*MferencePackageTests\.xctest'; } || true)
if [ -n "$others" ]; then note "REFUSED: another model process runs: $others"; exit 71; fi

s0=$(swap_mb)
note "swap start: $(sysctl -n vm.swapusage)"
note "memory_pressure: $(memory_pressure -Q 2>/dev/null | tail -1)"
note "cmd: $*"
t0=$(date +%s)
/usr/bin/time -l "$@" > "$LOG/stdout.log" 2> "$LOG/stderr.log" &
wrapper=$!
sleep 1
child=$(pgrep -P "$wrapper" | head -1)
tripped=0
while kill -0 "$wrapper" 2>/dev/null; do
    now=$(swap_mb)
    if awk -v a="$now" -v b="$s0" -v g="$SWAP_GUARD_MB" 'BEGIN { exit !(a - b > g) }'; then
        note "SWAP GUARD: used ${now} MB, start ${s0} MB; stopping"
        target=${child:-$wrapper}
        kill -INT "$target" 2>/dev/null
        for _ in $(seq 1 30); do kill -0 "$target" 2>/dev/null || break; sleep 1; done
        kill -0 "$target" 2>/dev/null && kill -TERM "$target" 2>/dev/null
        tripped=1
        break
    fi
    sleep 5
done
wait "$wrapper"; rc=$?
t1=$(date +%s)
s1=$(swap_mb)
note "swap end: $(sysctl -n vm.swapusage)"
peak=$(grep -E 'peak memory footprint' "$LOG/stderr.log" | awk '{print $1}' | tail -1)
rss=$(grep -E 'maximum resident set size' "$LOG/stderr.log" | awk '{print $1}' | tail -1)
delta=$(awk -v a="$s1" -v b="$s0" 'BEGIN { printf "%.2f", a - b }')
note "exit $rc; wall $((t1 - t0)) s; swap delta ${delta} MB; peak footprint ${peak:-?} B; max RSS ${rss:-?} B"
# Growth seen only at the end trips the guard too. Decide before writing
# run.json, so the record never shows a void run as clean: "exit" is the
# command's, "status" the wrapper's, "valid" true only for status 0.
if [ $tripped = 0 ] && awk -v d="$delta" -v g="$SWAP_GUARD_MB" 'BEGIN { exit !(d > g) }'; then
    note "SWAP GUARD: swap grew by ${delta} MB within the run"
    tripped=1
fi
status=$rc
[ $tripped = 1 ] && status=70
cat > "$LOG/run.json" <<EOF
{"command": "$(echo "$*" | sed 's/"/\\"/g')", "exit": $rc, "status": $status, "valid": $([ $status = 0 ] && echo true || echo false), "power_source": "$power", "timing_run": $([ $timing = 1 ] && echo true || echo false), "wall_s": $((t1 - t0)), "swap_start_mb": $s0, "swap_end_mb": $s1, "swap_delta_mb": $delta, "swap_guard_tripped": $([ $tripped = 1 ] && echo true || echo false), "peak_footprint_bytes": ${peak:-null}, "max_rss_bytes": ${rss:-null}}
EOF
exit $status
