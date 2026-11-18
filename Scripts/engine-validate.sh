#!/usr/bin/env bash
# engine-validate.sh: the engine program's section 8 validation matrix.
#
# Runs every product model through MferenceServer one at a time, each run under
# the machine lock, and writes raw outputs plus a summary table to
#   <results-root>/<YYYY-MM-DD>-<short sha>/
#
#   Scripts/engine-validate.sh [options] [phase ...]
#
# Phases (default: models prompts readings identity swap summary):
#   models     /v1/models lists exactly the five product models, each with
#              `engine`; Flash-Next absent (checked in every engine run)
#   prompts    fixed prompt set (short chat, 2K-token document question, tool
#              round-trip, JSON answer) non-streaming and streaming at
#              temperature 0; the two must agree
#   readings   load time, TTFT, prefill and decode tok/s, peak RSS for a 512-
#              and a 4K-token prompt
#   identity   native runs against a `main` server build (--baseline-server),
#              GGUF runs against upstream llama-server (--upstream-llama-server),
#              Deep MTP on against MTP off
#   swap       Quick -> Deep -> Qwen 3.6 -> MiniCPM5 -> Maple -> Quick through
#              one server, release-before-load from its log, peak RSS per step
#   summary    rebuild summary.md from what is on disk
#
# Options:
#   --models <k,k,...>           subset of quick,deep,qwen36,minicpm5,maple
#   --server <path>              MferenceServer under test
#                                (default .build/release/MferenceServer)
#   --server-args "<args>"       extra MferenceServer arguments
#   --baseline-server <path>     MferenceServer built from main (bc99691)
#   --upstream-llama-server <p>  llama-server built from the llama.cpp pin
#   --upstream-args "<args>"     llama-server arguments besides -m/--host/--port
#                                (default: -c 32768 -ngl 99 -np 1 --jinja)
#   --upstream-mtp-args "<args>" added for the Deep MTP-on upstream run
#   --mtp-off-args "<args>"      MferenceServer arguments that turn Deep MTP off
#   --mtp-off-env "<K=V ...>"    environment that turns Deep MTP off
#   --extra '<json>'             merged into every request body
#   --port <n>                   loopback port (default 8091)
#   --runs <n>                   readings per prompt size (default 2)
#   --results-root <dir>         default $PROGRAM/results
#   --out <dir>                  exact results directory (overrides the default)
#   --build                      build the server first (under the lock)
#   --no-release-line            do not require 'model released' log lines
#   --dry-run                    print the plan; run, build and lock nothing
#
# Safety (AGENTS.md, spec section 6): every model run takes the machine lock as
# workstream ws3, runs `lock.sh preflight` first, refuses on battery, refuses
# while any other model process (including llama.cpp tools) exists, keeps the
# server on 127.0.0.1, stops only the server it started, and treats any swap
# growth as a trip: it stops at once and exits 90.

set -euo pipefail

PROGRAM=/Users/zidanemitra/Desktop/Mference/scratch/engine-program
LOCK=${ENGINE_VALIDATE_LOCK:-/Users/zidanemitra/Desktop/Mference/scratch/program/lock.sh}
WS=ws3
SWAP_TRIP=90

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
checker=(python3 "$script_dir/engine_validate.py")

note() { printf '[engine-validate] %s\n' "$*" >&2; }
fail() { note "error: $*"; exit 1; }

# Processes that own a model. The lock's own pattern plus llama.cpp tools,
# which the lock does not know about.
other_model_processes() {
  pgrep -fl '(^|/)(MferenceServer|MferenceCLI|MferencePackageTests|swiftpm-testing-helper|llama-server|llama-cli|llama-bench|llama-speculative)( |$)|mlx_lm|mlx-lm' || true
}

on_ac_power() { pmset -g batt 2>/dev/null | head -1 | grep -q "AC Power"; }

swap_used_mib() { "${checker[@]}" swapusage; }

# ---------------------------------------------------------------------------
# Inner command: one model run, executed by `lock.sh run` while we hold the lock.
#   _run <key> <variant> <run_dir>
# variant: engine | engine-nomtp | main | upstream | upstream-nomtp | swap

run_one() {
  local key=$1 variant=$2 run_dir=$3
  mkdir -p "$run_dir"
  local log="$run_dir/server.log" binary args=() model_path kind model_id
  local env_prefix=()

  model_path=$("${checker[@]}" registry "$key" path)
  kind=$("${checker[@]}" registry "$key" kind)
  model_id=$("${checker[@]}" registry "$key" id)

  # Preflight, inside the lock.
  if [ "$kind" = gturbo ] && [ "$variant" != swap ]; then
    "$LOCK" preflight --model "$model_path" >"$run_dir/preflight.txt" 2>&1 \
      || { cat "$run_dir/preflight.txt" >&2; fail "preflight failed for $key"; }
  else
    "$LOCK" preflight >"$run_dir/preflight.txt" 2>&1 \
      || { cat "$run_dir/preflight.txt" >&2; fail "preflight failed for $key"; }
    if [ "$kind" = gguf ] && [ "$variant" != swap ]; then
      [ -f "$model_path" ] || fail "GGUF missing: $model_path"
    fi
  fi
  on_ac_power || { echo "on battery" >"$run_dir/REFUSED"; fail "not on AC power: report to the coordinator and wait"; }
  local others; others=$(other_model_processes)
  [ -z "$others" ] || fail "model process already running (never terminated by this script):
$others"
  if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    fail "port $PORT is already in use"
  fi

  case "$variant" in
    engine|engine-nomtp|swap)
      binary=$SERVER
      args=(--library --port "$PORT")
      [ -n "$SERVER_ARGS" ] && read -r -a extra_args <<<"$SERVER_ARGS" && args+=("${extra_args[@]}")
      if [ "$variant" = engine-nomtp ]; then
        [ -n "$MTP_OFF_ARGS" ] && read -r -a off <<<"$MTP_OFF_ARGS" && args+=("${off[@]}")
        [ -n "$MTP_OFF_ENV" ] && read -r -a env_prefix <<<"$MTP_OFF_ENV"
      fi
      ;;
    main)
      binary=$BASELINE_SERVER
      args=(--library --port "$PORT")
      [ -n "$SERVER_ARGS" ] && read -r -a extra_args <<<"$SERVER_ARGS" && args+=("${extra_args[@]}")
      ;;
    upstream|upstream-nomtp)
      binary=$UPSTREAM
      read -r -a up <<<"$UPSTREAM_ARGS"
      args=(-m "$model_path" --host 127.0.0.1 --port "$PORT" "${up[@]}")
      if [ "$key" = deep ] && [ "$variant" = upstream ] && [ -n "$UPSTREAM_MTP_ARGS" ]; then
        read -r -a mtp <<<"$UPSTREAM_MTP_ARGS"; args+=("${mtp[@]}")
      fi
      ;;
    *) fail "unknown variant $variant" ;;
  esac

  local swap_before swap_after started
  swap_before=$(swap_used_mib)
  printf '%s\n' "$binary ${args[*]}" >"$run_dir/command.txt"
  started=$(python3 -c 'import time; print(time.time())')
  # /usr/bin/time -l reports the server's peak RSS and footprint on exit.
  env ${env_prefix[@]+"${env_prefix[@]}"} /usr/bin/time -l "$binary" "${args[@]}" \
    >"$run_dir/server.stdout" 2>"$log" &
  # Globals, not locals: the EXIT trap may run after `fail` unwinds.
  time_pid=$!
  server_pid=""
  stopped=0
  stop_server() {
    [ "$stopped" = 1 ] && return
    stopped=1
    [ -n "$server_pid" ] && kill -TERM "$server_pid" 2>/dev/null || true
    wait "$time_pid" 2>/dev/null || true
  }
  trap 'stop_server' EXIT INT TERM HUP

  for _ in $(seq 1 100); do
    server_pid=$(pgrep -P "$time_pid" | head -1 || true)
    [ -n "$server_pid" ] && break
    sleep 0.1
  done
  [ -n "$server_pid" ] || fail "server did not start; see $log"

  # Ready: /health answers 200 (llama-server answers 503 while it loads).
  local ready=0
  for _ in $(seq 1 3600); do
    kill -0 "$server_pid" 2>/dev/null || break
    if curl --silent --fail --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      ready=1; break
    fi
    sleep 0.5
  done
  [ "$ready" = 1 ] || fail "server never became healthy; see $log"
  local ready_at; ready_at=$(python3 -c 'import time; print(time.time())')

  local status=0 rc=0 api_id=$model_id

  if [ "$variant" = swap ]; then
    "${checker[@]}" swap --port "$PORT" --log "$log" --pid "$server_pid" \
      --out "$run_dir/swap.json" $NO_RELEASE_LINE || rc=$?
    status=$rc
  else
    case "$variant" in engine*)
      if want models; then
        "${checker[@]}" models --port "$PORT" --out "$run_dir/models.json" || status=1
        [ -f "$RESULTS/models.json" ] || cp "$run_dir/models.json" "$RESULTS/models.json"
      fi ;;
    esac
    # Load: library mode loads on the first request and logs the duration.
    "${checker[@]}" probe --port "$PORT" --model "$api_id" --out "$run_dir/load-probe.json" \
      || status=1
    if want prompts || want identity; then
      "${checker[@]}" prompts --port "$PORT" --model "$api_id" --out "$run_dir" \
        ${EXTRA:+--extra "$EXTRA"} || status=1
    fi
    if want readings; then
      "${checker[@]}" readings --port "$PORT" --model "$api_id" --pid "$server_pid" \
        --runs "$RUNS" --out "$run_dir/readings.json" ${EXTRA:+--extra "$EXTRA"} || rc=$?
      [ "$rc" = "$SWAP_TRIP" ] && status=$SWAP_TRIP
    fi
  fi

  stop_server
  trap - EXIT INT TERM HUP
  swap_after=$(swap_used_mib)

  python3 - "$run_dir" "$key" "$variant" "$started" "$ready_at" "$swap_before" "$swap_after" "$status" <<'PY'
import json, re, sys
from pathlib import Path
run_dir, key, variant, started, ready_at, before, after, status = sys.argv[1:]
log = (Path(run_dir) / "server.log").read_text(encoding="utf-8", errors="replace")
peaks = {}
for k, pat in (("max_rss_bytes", r"(\d+)\s+maximum resident set size"),
               ("peak_footprint_bytes", r"(\d+)\s+peak memory footprint")):
    m = re.search(pat, log)
    peaks[k] = int(m.group(1)) if m else None
loads = [float(x) for x in re.findall(r"swap finished model=\S+ in ([0-9.]+)s", log)]
def num(s):
    try: return float(s)
    except ValueError: return None
b, a = num(before), num(after)
run = {"key": key, "variant": variant, "status": int(status),
       "load_s": loads[0] if loads else (float(ready_at) - float(started) if variant.startswith("upstream") else None),
       "swap_before_mib": b, "swap_after_mib": a,
       "swap_delta_mib": (a - b) if a is not None and b is not None else None, **peaks}
(Path(run_dir) / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True))
PY
  local delta
  delta=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])).get("swap_delta_mib"); print("1" if d is not None and d > 0 else "0")' "$run_dir/run.json")
  if [ "$delta" = 1 ] || [ "$status" = "$SWAP_TRIP" ]; then
    echo "swap grew during $key/$variant: before=$swap_before after=$swap_after MiB" | tee "$run_dir/SWAP_TRIP" >&2
    exit "$SWAP_TRIP"
  fi
  exit "$status"
}

# ---------------------------------------------------------------------------
# Options

ORIG_ARGS="$*"
PHASES=()
MODELS=""
SERVER="$repo_root/.build/release/MferenceServer"
SERVER_ARGS=""
BASELINE_SERVER=""
UPSTREAM=""
UPSTREAM_ARGS="-c 32768 -ngl 99 -np 1 --jinja"
UPSTREAM_MTP_ARGS=""
MTP_OFF_ARGS=""
MTP_OFF_ENV=""
EXTRA=""
PORT=8091
RUNS=2
RESULTS_ROOT="$PROGRAM/results"
RESULTS=""
BUILD=0
DRY_RUN=0
NO_RELEASE_LINE=""

if [ "${1:-}" = _run ]; then
  shift
  # The outer invocation exported its configuration.
  # shellcheck disable=SC2034
  PHASES_STR=${ENGINE_VALIDATE_PHASES:-}
  want() { case " $PHASES_STR " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
  SERVER=$ENGINE_VALIDATE_SERVER SERVER_ARGS=$ENGINE_VALIDATE_SERVER_ARGS
  BASELINE_SERVER=$ENGINE_VALIDATE_BASELINE UPSTREAM=$ENGINE_VALIDATE_UPSTREAM
  UPSTREAM_ARGS=$ENGINE_VALIDATE_UPSTREAM_ARGS UPSTREAM_MTP_ARGS=$ENGINE_VALIDATE_UPSTREAM_MTP_ARGS
  MTP_OFF_ARGS=$ENGINE_VALIDATE_MTP_OFF_ARGS MTP_OFF_ENV=$ENGINE_VALIDATE_MTP_OFF_ENV
  EXTRA=$ENGINE_VALIDATE_EXTRA PORT=$ENGINE_VALIDATE_PORT RUNS=$ENGINE_VALIDATE_RUNS
  RESULTS=$ENGINE_VALIDATE_RESULTS NO_RELEASE_LINE=$ENGINE_VALIDATE_NO_RELEASE_LINE
  run_one "$@"
fi

while [ $# -gt 0 ]; do
  case "$1" in
    --models) MODELS=$2; shift 2 ;;
    --server) SERVER=$2; shift 2 ;;
    --server-args) SERVER_ARGS=$2; shift 2 ;;
    --baseline-server) BASELINE_SERVER=$2; shift 2 ;;
    --upstream-llama-server) UPSTREAM=$2; shift 2 ;;
    --upstream-args) UPSTREAM_ARGS=$2; shift 2 ;;
    --upstream-mtp-args) UPSTREAM_MTP_ARGS=$2; shift 2 ;;
    --mtp-off-args) MTP_OFF_ARGS=$2; shift 2 ;;
    --mtp-off-env) MTP_OFF_ENV=$2; shift 2 ;;
    --extra) EXTRA=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --runs) RUNS=$2; shift 2 ;;
    --results-root) RESULTS_ROOT=$2; shift 2 ;;
    --out) RESULTS=$2; shift 2 ;;
    --build) BUILD=1; shift ;;
    --no-release-line) NO_RELEASE_LINE=--no-release-line; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,52p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    models|prompts|readings|identity|swap|summary) PHASES+=("$1"); shift ;;
    *) fail "unknown argument: $1 (see --help)" ;;
  esac
done
[ ${#PHASES[@]} -gt 0 ] || PHASES=(models prompts readings identity swap summary)
PHASES_STR=" ${PHASES[*]} "
want() { case "$PHASES_STR" in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
[ -n "$MODELS" ] || MODELS=$("${checker[@]}" registry all)
IFS=, read -r -a MODEL_KEYS <<<"$MODELS"
for key in "${MODEL_KEYS[@]}"; do "${checker[@]}" registry "$key" id >/dev/null || exit 2; done
case "$PORT" in ''|*[!0-9]*) fail "--port must be a number" ;; esac

sha=$(git -C "$repo_root" rev-parse --short=7 HEAD)
[ -n "$RESULTS" ] || RESULTS="$RESULTS_ROOT/$(date +%Y-%m-%d)-$sha"

# The run list: (key variant). Identity needs the reference runs.
RUN_LIST=()
for key in "${MODEL_KEYS[@]}"; do
  kind=$("${checker[@]}" registry "$key" kind)
  RUN_LIST+=("$key engine")
  if [ "$key" = deep ] && { [ -n "$MTP_OFF_ARGS" ] || [ -n "$MTP_OFF_ENV" ]; }; then
    RUN_LIST+=("$key engine-nomtp")
  fi
  if want identity || want readings; then
    if [ "$kind" = gturbo ] && [ -n "$BASELINE_SERVER" ]; then RUN_LIST+=("$key main"); fi
    if [ "$kind" = gguf ] && [ -n "$UPSTREAM" ]; then
      RUN_LIST+=("$key upstream")
      if [ "$key" = deep ] && [ -n "$UPSTREAM_MTP_ARGS" ]; then RUN_LIST+=("$key upstream-nomtp"); fi
    fi
  fi
done

lock_cmd() { printf '%q ' "$LOCK" run --wait 3600 "$WS" "$1" -- "${@:2}"; echo; }

if [ "$DRY_RUN" = 1 ]; then
  echo "results: $RESULTS"
  echo "phases: ${PHASES[*]}"
  [ "$BUILD" = 1 ] && lock_cmd "eng build validate server" "$LOCK" build -- swift build -c release --product MferenceServer
  if want models || want prompts || want readings || want identity; then
    for item in "${RUN_LIST[@]}"; do
      read -r key variant <<<"$item"
      lock_cmd "eng validate $key $variant" "$script_dir/engine-validate.sh" _run "$key" "$variant" "$RESULTS/$key/$variant"
    done
  fi
  want swap && lock_cmd "eng validate swap chain" "$script_dir/engine-validate.sh" _run quick swap "$RESULTS/swap"
  want identity && echo "identity: compare engine runs with main/upstream runs; deep MTP on vs off"
  want summary && echo "summary: $RESULTS/summary.md"
  exit 0
fi

[ -x "$LOCK" ] || fail "lock script not found: $LOCK"
mkdir -p "$RESULTS"

if [ "$BUILD" = 1 ]; then
  (cd "$repo_root" && "$LOCK" run --wait 3600 "$WS" "eng build validate server" -- \
    "$LOCK" build -- swift build -c release --product MferenceServer) \
    || fail "build failed or the quiet flag is set (exit $?); stop and wait"
fi
needs_server=0
for p in models prompts readings identity swap; do want "$p" && needs_server=1; done
if [ "$needs_server" = 1 ]; then
  [ -x "$SERVER" ] || fail "server binary not found: $SERVER (build it, or pass --build)"
fi

python3 - "$RESULTS/meta.json" "$repo_root" "$SERVER" "$BASELINE_SERVER" "$UPSTREAM" "$ORIG_ARGS" "${PHASES[*]}" <<'PY'
import json, os, platform, subprocess, sys, time
out, repo, server, baseline, upstream, argv, phases = sys.argv[1:]
def sh(*cmd):
    try: return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e: return f"unavailable: {e}"
def stamp(p):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(os.path.getmtime(p))) if p and os.path.exists(p) else None
meta = {
    "commit": sh("git", "-C", repo, "rev-parse", "HEAD"),
    "branch": sh("git", "-C", repo, "rev-parse", "--abbrev-ref", "HEAD"),
    "dirty": bool(sh("git", "-C", repo, "status", "--porcelain")),
    "hardware": sh("sysctl", "-n", "machdep.cpu.brand_string"),
    "memory_bytes": sh("sysctl", "-n", "hw.memsize"),
    "macos": sh("sw_vers", "-productVersion") + " " + sh("sw_vers", "-buildVersion"),
    "swift": sh("swift", "--version").splitlines()[0] if sh("swift", "--version") else None,
    "server": server, "server_mtime": stamp(server),
    "baseline_server": baseline or None, "baseline_mtime": stamp(baseline),
    "upstream_llama_server": upstream or None,
    "phases": phases,
    "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(meta, open(out, "w"), indent=2, sort_keys=True)
PY

export ENGINE_VALIDATE_PHASES="$PHASES_STR" ENGINE_VALIDATE_SERVER="$SERVER"
export ENGINE_VALIDATE_SERVER_ARGS="$SERVER_ARGS" ENGINE_VALIDATE_BASELINE="$BASELINE_SERVER"
export ENGINE_VALIDATE_UPSTREAM="$UPSTREAM" ENGINE_VALIDATE_UPSTREAM_ARGS="$UPSTREAM_ARGS"
export ENGINE_VALIDATE_UPSTREAM_MTP_ARGS="$UPSTREAM_MTP_ARGS"
export ENGINE_VALIDATE_MTP_OFF_ARGS="$MTP_OFF_ARGS" ENGINE_VALIDATE_MTP_OFF_ENV="$MTP_OFF_ENV"
export ENGINE_VALIDATE_EXTRA="$EXTRA" ENGINE_VALIDATE_PORT="$PORT" ENGINE_VALIDATE_RUNS="$RUNS"
export ENGINE_VALIDATE_RESULTS="$RESULTS" ENGINE_VALIDATE_NO_RELEASE_LINE="$NO_RELEASE_LINE"

overall=0
locked_run() {
  local purpose=$1; shift
  local rc=0
  "$LOCK" run --wait 3600 "$WS" "$purpose" -- "$script_dir/engine-validate.sh" _run "$@" || rc=$?
  case "$rc" in
    0) ;;
    "$SWAP_TRIP")
      note "SWAP GUARD TRIPPED during $purpose: stopping all model runs; report to the coordinator"
      "${checker[@]}" summary "$RESULTS" >/dev/null || true
      exit "$SWAP_TRIP" ;;
    75) fail "could not take the machine lock for $purpose; stop and wait" ;;
    *) note "$purpose failed (exit $rc); continuing"; overall=1 ;;
  esac
}

if want models || want prompts || want readings || want identity; then
  for item in "${RUN_LIST[@]}"; do
    read -r key variant <<<"$item"
    locked_run "eng validate $key $variant" "$key" "$variant" "$RESULTS/$key/$variant"
  done
fi

if want swap; then
  locked_run "eng validate swap chain" quick swap "$RESULTS/swap"
fi

if want identity; then
  compare() {
    local label=$1 a=$2 b=$3
    if [ -f "$a/prompts.json" ] && [ -f "$b/prompts.json" ]; then
      "${checker[@]}" compare "$a" "$b" --label "$label" --out "$RESULTS/identity-$label.json" \
        || overall=1
    fi
  }
  for key in "${MODEL_KEYS[@]}"; do
    compare "$key-engine-vs-main" "$RESULTS/$key/engine" "$RESULTS/$key/main"
    compare "$key-engine-vs-upstream" "$RESULTS/$key/engine" "$RESULTS/$key/upstream"
  done
  compare deep-mtp-on-vs-off "$RESULTS/deep/engine" "$RESULTS/deep/engine-nomtp"
  compare deep-upstream-mtp-on-vs-off "$RESULTS/deep/upstream" "$RESULTS/deep/upstream-nomtp"
fi

"${checker[@]}" summary "$RESULTS" >/dev/null
note "results: $RESULTS (summary.md)"
exit "$overall"
