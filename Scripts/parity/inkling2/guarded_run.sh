#!/bin/bash
# guarded_run.sh: one reference model command (llama-bench, ink-golden, ...)
# with the Inkling program's machine rules applied around it (spec section 7).
#
#   guarded_run.sh [--timing] <log-dir> <gguf> -- <command> [args...]
#
# Run it as the single command of its own lock hold, for example
#   $INK/bin/model-hold.sh [--timing] ws0 "<what>" -- guarded_run.sh [--timing] <log-dir> <gguf> -- llama-bench ...
# (spec v1.3: model runs start through model-hold.sh; any lock.sh run hold passes
# the ancestor check below, but model-hold.sh is the program's rule)
#
# Before: refuses (exit 71) unless `lock.sh preflight` passes, the GGUF exists,
# (with --timing) the Mference app is not open,
# no other model process runs and, with --timing, `pmset -g batt` shows AC
# Power. During: samples swap every 2 s; any growth stops the command (TERM,
# then KILL after 30 s) and exits 70. TERM, not INT: the command is started
# with & from this non-interactive script, so it inherits SIGINT ignored. After: records swap at start and end,
# exit code, wall time, power source and /usr/bin/time -l's peak memory
# footprint in <log-dir>/run.json; stdout and stderr in <log-dir>.
set -uo pipefail
# shellcheck source=paths.sh
. "$(cd "$(dirname "$0")" && pwd)/paths.sh"
TIMING=0
if [ "${1:-}" = "--timing" ]; then TIMING=1; shift; fi
[ $# -ge 4 ] && [ "$3" = "--" ] || { echo "usage: guarded_run.sh [--timing] <log-dir> <gguf> -- <command> [args...]" >&2; exit 2; }
LOG=$1; GGUF=$2; shift 3
mkdir -p "$LOG"
note() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG/run.log" >&2; }
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/'; }

# Hard rule (coordinator, 2026-11-23): refuse every command known to read the
# model's tensor data into memory instead of mapping it. llama-gguf's `r` mode
# heap-loads all tensors (examples/gguf/gguf.cpp, gguf_ex_read_1, no_alloc =
# false) and caused the 2026-11-23 swap-guard trip; the quantize/split/hash
# tools read or rewrite whole tensors; --no-mmap copies the weights to heap and
# --mlock wires them.
case "$(basename "$1")" in
  llama-gguf|llama-quantize|llama-gguf-split|llama-gguf-hash|llama-export-lora)
    note "REFUSED: $(basename "$1") reads tensor data into memory (not mmap); not allowed on this host"; exit 71 ;;
esac
# Load modes at b12641: `-lm` / `--load-mode auto|none|mmap|mlock|mmap+mlock|dio`
# (common/arg.cpp, llama-bench); only auto and mmap map the file. Older
# spellings are refused too, as is the LLAMA_ARG_LOAD_MODE environment variable.
if [ -n "${LLAMA_ARG_LOAD_MODE:-}" ]; then note "REFUSED: LLAMA_ARG_LOAD_MODE is set ($LLAMA_ARG_LOAD_MODE)"; exit 71; fi
prev=""
for a in "$@"; do
  case "$a" in
    --no-mmap|-nmp|--mlock|--mmap=0|--mmap=false|--load-mode=*)
      case "$a" in --load-mode=auto|--load-mode=mmap) ;; *)
        note "REFUSED: argument $a loads or wires the weights in memory; not allowed on this host"; exit 71 ;; esac ;;
  esac
  if [ "$prev" = "-lm" ] || [ "$prev" = "--load-mode" ]; then
    # llama-bench also takes a comma-separated list: every entry must map
    IFS=, read -r -a modes <<< "$a"
    for m in "${modes[@]}"; do
      case "$m" in auto|mmap) ;; *) note "REFUSED: load mode '$m' does not map the file; not allowed on this host"; exit 71 ;; esac
    done
  fi
  prev=$a
done

# Spec section 7: a reference tool never runs outside a machine-lock hold.
# Require that the lock's recorded holder is one of this script's ancestors.
holder=$("$LOCK" status 2>/dev/null | sed -n 's/^ *pid=\([0-9][0-9]*\).*/\1/p' | head -1)
p=$$; inside=0
while [ -n "$p" ] && [ "$p" -gt 1 ]; do
  [ "$p" = "${holder:-x}" ] && { inside=1; break; }
  p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
done
[ $inside = 1 ] || { note "REFUSED: not inside a lock.sh run hold (lock holder pid ${holder:-none})"; exit 71; }

power=$(pmset -g batt | head -1)
if [ $TIMING = 1 ] && ! echo "$power" | grep -q "AC Power"; then note "REFUSED: timing run not on AC power ($power)"; exit 71; fi
# Coordinator rule (2026-11-23): no tok/s number while the Mference app is open
# (it may hold a model); try again later by hand.
# by executable path (ps comm), not by command line: another process's
# arguments can mention the app path
if [ $TIMING = 1 ] && ps -axo comm= | grep -q '/Mference.app/Contents/MacOS/'; then note "REFUSED: timing run while the Mference app is open"; exit 71; fi
[ -f "$GGUF" ] || { note "REFUSED: no GGUF at $GGUF"; exit 71; }
if ! "$LOCK" preflight >> "$LOG/run.log" 2>&1; then note "REFUSED: lock.sh preflight failed (see run.log)"; exit 71; fi
# Match executables by full basename (pgrep -x sees names cut to 16 characters);
# any llama.cpp tool, the golden tools and the Mference model binaries count.
others=$(ps -axo pid=,comm= | while read -r pid comm; do
  case "${comm##*/}" in (llama-*|ink-golden|mfe-*|MferenceServer|MferenceCLI|mlx_lm*) echo "$pid ${comm##*/}" ;; esac
done)
[ -z "$others" ] || { note "REFUSED: another model process runs: $others"; exit 71; }

s0=$(swap_mb)
note "power: $power"
note "swap start: $(sysctl -n vm.swapusage)"
note "memory_pressure: $(memory_pressure -Q 2>/dev/null | tail -1)"
note "cmd: $*"
t0=$(date +%s)
/usr/bin/time -l "$@" > "$LOG/stdout.log" 2> "$LOG/stderr.log" &
wrapper=$!
tripped=0
max_swap=$s0
while kill -0 "$wrapper" 2>/dev/null; do
    now=$(swap_mb)
    if awk -v a="$now" -v b="$s0" 'BEGIN { exit !(a > b) }'; then
        note "SWAP GUARD: used ${now} MB, start ${s0} MB; stopping"
        child=$(pgrep -P "$wrapper" | head -1)
        target=${child:-$wrapper}
        # TERM first: an INT sent here would be ignored (see the header)
        kill -TERM "$target" 2>/dev/null
        for _ in $(seq 1 30); do kill -0 "$target" 2>/dev/null || break; sleep 1; done
        # bounded: a process that outlives TERM still holds the lock and its memory
        if kill -0 "$target" 2>/dev/null; then
            note "SWAP GUARD: $target ignored TERM for 30 s; sending KILL"
            kill -KILL "$target" 2>/dev/null
        fi
        sleep 2
        kill -0 "$wrapper" 2>/dev/null && kill -KILL "$wrapper" 2>/dev/null
        tripped=1
        break
    fi
    sleep 2
done
wait "$wrapper"; rc=$?
t1=$(date +%s)
s1=$(swap_mb)
note "swap end: $(sysctl -n vm.swapusage)"
peak=$(awk '/peak memory footprint/ {print $1}' "$LOG/stderr.log" | tail -1)
rss=$(awk '/maximum resident set size/ {print $1}' "$LOG/stderr.log" | tail -1)
delta=$(awk -v a="$s1" -v b="$s0" 'BEGIN { printf "%.2f", a - b }')
note "exit $rc; wall $((t1 - t0)) s; swap delta ${delta} MB; peak footprint ${peak:-?} B; max RSS ${rss:-?} B"
cat > "$LOG/run.json" <<EOF
{"command": "$(echo "$*" | sed 's/"/\\"/g')", "exit": $rc, "wall_s": $((t1 - t0)), "power": "$(echo "$power" | sed 's/"//g')", "swap_start_mb": $s0, "swap_end_mb": $s1, "swap_delta_mb": $delta, "swap_guard_tripped": $([ $tripped = 1 ] && echo true || echo false), "peak_footprint_bytes": ${peak:-null}, "max_rss_bytes": ${rss:-null}}
EOF
[ $tripped = 1 ] && exit 70
if awk -v d="$delta" 'BEGIN { exit !(d > 0) }'; then note "SWAP GUARD: swap grew by ${delta} MB within the run"; exit 70; fi
exit $rc
