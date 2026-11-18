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
#              temperature 0: the two must agree, and the tool round-trip must
#              actually call the tool
#   readings   load time, TTFT, prefill and decode tok/s, peak RSS for a 512-
#              and a 4K-token prompt; with a reference run, the engine must be
#              within 5% of it (main for native, upstream llama.cpp for GGUF)
#   identity   md5 of every output: native vs a `main` server build, GGUF vs
#              upstream llama-server, Deep MTP on vs off. Needs every
#              reference option below; refuses to start without them
#   swap       Quick -> Deep -> Qwen 3.6 -> MiniCPM5 -> Maple -> Quick through
#              one server: the old model's release logged before the new one
#              loads, peak RSS per step below one model's peak
#   summary    rebuild summary.md from what is on disk
#
# Not covered here: G5 through MferenceCLI (the CLI does not use the engine),
# and section 8 item 6 (Open WebUI), which is checked by hand.
#
# Options:
#   --models <k,k,...>           subset of quick,deep,qwen36,minicpm5,maple
#   --server <path>              MferenceServer under test
#                                (default .build/release/MferenceServer); the
#                                results are labelled with the commit of the
#                                checkout that owns its .build directory
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
#   --results-root <dir>         default $PROGRAM/results; an existing
#                                <date>-<sha> directory is refused
#   --out <dir>                  exact results directory (may already exist;
#                                each run directory is cleared before it runs)
#   --build                      build the server first (under the lock)
#   --yield                      pass --yield to lock.sh (runs over ~15 min
#                                that are not on the priority list)
#   --no-release-line            accept swaps without a 'model released' log
#                                line, on the peak-RSS bound alone
#   --dry-run                    print the plan; run, build and lock nothing
#
# Safety (AGENTS.md, spec section 6): every model run takes the machine lock as
# workstream ws3 with --quiet, runs `lock.sh preflight` (with --model for every
# install it loads), refuses on battery or while any other model process
# (llama.cpp tools included) exists, keeps the server on 127.0.0.1 (pass-through
# --host/--bind is rejected), and stops only the server it started. Exit codes:
# 0 pass, 1 a check failed, 90 swap guard tripped (any growth; everything
# stops), 91 a run was refused (everything stops), 75 lock not taken, 130
# interrupted. Report 90 and 91 to the coordinator.

set -euo pipefail

WS=ws3
SWAP_TRIP=90

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
# scratch/ exists only in the main checkout, not in worktrees: find it through
# git's common directory, which every worktree shares.
main_checkout=$(cd -- "$(git -C "$repo_root" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || echo "$repo_root/.git")/.." && pwd)
PROGRAM=${ENGINE_VALIDATE_PROGRAM:-$main_checkout/scratch/engine-program}
LOCK=${ENGINE_VALIDATE_LOCK:-$main_checkout/scratch/program/lock.sh}
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
#
# Exit status: 0 pass, 1 a check failed, 90 swap guard tripped, 91 refused
# (preflight, power, another model process, port), 130 interrupted. The outer
# loop stops on 90, 91 and 130.

REFUSED=91
INTERRUPTED=130

refuse() {
  note "refused: $*"
  if [ -n "${RUN_DIR:-}" ]; then printf '%s\n' "$*" >"$RUN_DIR/REFUSED"; fi
  exit "$REFUSED"
}

# Arguments passed through to a server must not move it off loopback.
check_loopback_args() {
  case " $1 " in
    *" --bind "*|*" --bind="*|*" --host "*|*" --host="*|*0.0.0.0*|*"::"*)
      fail "pass-through arguments may not set --bind/--host: $1" ;;
  esac
}

# Globals, not locals: the EXIT handler runs after `fail`/`exit` unwinds.
RUN_DIR="" RUN_KEY="" RUN_VARIANT="" TIME_PID="" SERVER_PID="" SWAP_BEFORE=""
STARTED="" READY_AT="" RUN_STATUS=0 SERVER_STOPPED=1

stop_server() {
  [ "$SERVER_STOPPED" = 1 ] && return 0
  SERVER_STOPPED=1
  # Only the server this run started: the child of our own /usr/bin/time.
  [ -n "$SERVER_PID" ] || SERVER_PID=$(pgrep -P "$TIME_PID" 2>/dev/null | head -1 || true)
  if [ -n "$SERVER_PID" ]; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
  else
    kill -TERM "$TIME_PID" 2>/dev/null || true
  fi
  local waited=0
  while kill -0 "$TIME_PID" 2>/dev/null && [ "$waited" -lt 120 ]; do
    sleep 0.5; waited=$((waited + 1))
  done
  if kill -0 "$TIME_PID" 2>/dev/null; then
    note "server ${SERVER_PID:-?} ignored TERM for 60 s; sending KILL"
    if [ -n "$SERVER_PID" ]; then kill -KILL "$SERVER_PID" 2>/dev/null || true; fi
  fi
  wait "$TIME_PID" 2>/dev/null || true
}

# Runs on every exit of a model run once the server may exist: stop it,
# measure swap, write run.json, and turn any swap growth into exit 90.
finish_run() {
  local rc=$?
  trap - EXIT INT TERM HUP
  stop_server
  [ "$rc" = 0 ] || RUN_STATUS=$rc
  local swap_after; swap_after=$(swap_used_mib 2>/dev/null || true)
  "${checker[@]}" runrecord "$RUN_DIR" --key "$RUN_KEY" --variant "$RUN_VARIANT" \
    --started "$STARTED" --ready-at "$READY_AT" --swap-before "$SWAP_BEFORE" \
    --swap-after "$swap_after" --status "$RUN_STATUS" || true
  if [ "$RUN_STATUS" = "$SWAP_TRIP" ] || "${checker[@]}" swapgrew "$RUN_DIR/run.json"; then
    echo "swap grew during $RUN_KEY/$RUN_VARIANT: before=$SWAP_BEFORE after=$swap_after MiB" \
      | tee "$RUN_DIR/SWAP_TRIP" >&2
    exit "$SWAP_TRIP"
  fi
  exit "$RUN_STATUS"
}

preflight_model() {
  local key=$1 out=$2 path kind
  path=$("${checker[@]}" registry "$key" path)
  kind=$("${checker[@]}" registry "$key" kind)
  if [ "$kind" = gturbo ]; then
    "$LOCK" preflight --model "$path" >>"$out" 2>&1 \
      || { cat "$out" >&2; refuse "preflight failed for $key"; }
  else
    [ -f "$path" ] || refuse "GGUF missing: $path"
  fi
}

run_one() {
  RUN_KEY=$1 RUN_VARIANT=$2 RUN_DIR=$3
  local key=$1 variant=$2
  # From here on an interrupt exits 130; once the server is launched the EXIT
  # handler below also stops it and records the run.
  trap 'exit $INTERRUPTED' INT TERM HUP
  # A rerun never inherits an earlier run's outputs.
  rm -rf "$RUN_DIR"
  mkdir -p "$RUN_DIR"
  local log="$RUN_DIR/server.log" binary args=() model_path model_id
  local env_prefix=() extra_args=() off=() up=() mtp=()

  model_path=$("${checker[@]}" registry "$key" path)
  model_id=$("${checker[@]}" registry "$key" id)

  # Preflight, inside the lock: general checks, then every model this run loads.
  "$LOCK" preflight >"$RUN_DIR/preflight.txt" 2>&1 \
    || { cat "$RUN_DIR/preflight.txt" >&2; refuse "preflight failed"; }
  if [ "$variant" = swap ]; then
    local chain_key
    for chain_key in $("${checker[@]}" registry chain); do
      preflight_model "$chain_key" "$RUN_DIR/preflight.txt"
    done
  else
    preflight_model "$key" "$RUN_DIR/preflight.txt"
  fi
  on_ac_power || refuse "not on AC power: report to the coordinator and wait"
  local others; others=$(other_model_processes)
  [ -z "$others" ] || refuse "model process already running (never terminated by this script):
$others"
  if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    refuse "port $PORT is already in use"
  fi

  case "$variant" in
    engine|engine-nomtp|swap)
      binary=$SERVER
      args=(--library --port "$PORT")
      if [ -n "$SERVER_ARGS" ]; then read -r -a extra_args <<<"$SERVER_ARGS"; args+=("${extra_args[@]}"); fi
      if [ "$variant" = engine-nomtp ]; then
        if [ -n "$MTP_OFF_ARGS" ]; then read -r -a off <<<"$MTP_OFF_ARGS"; args+=("${off[@]}"); fi
        if [ -n "$MTP_OFF_ENV" ]; then read -r -a env_prefix <<<"$MTP_OFF_ENV"; fi
      fi
      ;;
    main)
      binary=$BASELINE_SERVER
      args=(--library --port "$PORT")
      if [ -n "$SERVER_ARGS" ]; then read -r -a extra_args <<<"$SERVER_ARGS"; args+=("${extra_args[@]}"); fi
      ;;
    upstream|upstream-nomtp)
      binary=$UPSTREAM
      args=(-m "$model_path")
      if [ -n "$UPSTREAM_ARGS" ]; then read -r -a up <<<"$UPSTREAM_ARGS"; args+=("${up[@]}"); fi
      if [ "$key" = deep ] && [ "$variant" = upstream ] && [ -n "$UPSTREAM_MTP_ARGS" ]; then
        read -r -a mtp <<<"$UPSTREAM_MTP_ARGS"; args+=("${mtp[@]}")
      fi
      # Last, so nothing passed through can override the loopback binding.
      args+=(--host 127.0.0.1 --port "$PORT")
      ;;
    *) fail "unknown variant $variant" ;;
  esac

  SWAP_BEFORE=$(swap_used_mib)
  printf '%s\n' "${env_prefix[*]+${env_prefix[*]} }$binary ${args[*]}" >"$RUN_DIR/command.txt"
  STARTED=$(python3 -c 'import time; print(time.time())')
  SERVER_STOPPED=0
  trap finish_run EXIT
  trap 'exit $INTERRUPTED' INT TERM HUP
  # /usr/bin/time -l reports the server's peak RSS and footprint on exit.
  env ${env_prefix[@]+"${env_prefix[@]}"} /usr/bin/time -l "$binary" "${args[@]}" \
    >"$RUN_DIR/server.stdout" 2>"$log" &
  TIME_PID=$!

  local i
  for i in $(seq 1 300); do
    SERVER_PID=$(pgrep -P "$TIME_PID" | head -1 || true)
    [ -n "$SERVER_PID" ] && break
    kill -0 "$TIME_PID" 2>/dev/null || break
    sleep 0.1
  done
  [ -n "$SERVER_PID" ] || fail "server did not start; see $log"

  # Ready: /health answers 200 (llama-server answers 503 while it loads).
  local ready=0
  for i in $(seq 1 3600); do
    kill -0 "$SERVER_PID" 2>/dev/null || break
    if curl --silent --fail --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      ready=1; break
    fi
    sleep 0.5
  done
  [ "$ready" = 1 ] || fail "server never became healthy; see $log"
  READY_AT=$(python3 -c 'import time; print(time.time())')

  local rc=0
  if [ "$variant" = swap ]; then
    local bound=()
    local single; single=$("${checker[@]}" maxrss "$RESULTS" 2>/dev/null || true)
    if [ -n "$single" ]; then bound+=(--max-single-rss-mib "$single"); fi
    if [ -n "$NO_RELEASE_LINE" ]; then
      [ -n "$single" ] || fail "--no-release-line needs the per-model engine runs (their peak RSS) first"
      bound+=(--no-release-line)
    fi
    "${checker[@]}" swap --port "$PORT" --log "$log" --pid "$SERVER_PID" \
      --out "$RUN_DIR/swap.json" ${bound[@]+"${bound[@]}"} || rc=$?
    RUN_STATUS=$rc
  else
    case "$variant" in engine*)
      if want models; then
        "${checker[@]}" models --port "$PORT" --out "$RUN_DIR/models.json" || RUN_STATUS=1
        cp "$RUN_DIR/models.json" "$RESULTS/models.json" 2>/dev/null || RUN_STATUS=1
      fi ;;
    esac
    # Load: library mode loads on the first request and logs the duration.
    "${checker[@]}" probe --port "$PORT" --model "$model_id" --out "$RUN_DIR/load-probe.json" \
      || RUN_STATUS=1
    if want prompts || want identity; then
      "${checker[@]}" prompts --port "$PORT" --model "$model_id" --out "$RUN_DIR" \
        ${EXTRA:+--extra "$EXTRA"} || RUN_STATUS=1
    fi
    if want readings; then
      rc=0
      "${checker[@]}" readings --port "$PORT" --model "$model_id" --pid "$SERVER_PID" \
        --runs "$RUNS" --out "$RUN_DIR/readings.json" ${EXTRA:+--extra "$EXTRA"} || rc=$?
      if [ "$rc" = "$SWAP_TRIP" ]; then RUN_STATUS=$SWAP_TRIP
      elif [ "$rc" != 0 ]; then RUN_STATUS=1; fi
    fi
  fi
  exit "$RUN_STATUS"
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
YIELD=""
NO_RELEASE_LINE=""

if [ "${1:-}" = _run ]; then
  shift
  # The outer invocation exported its configuration.
  PHASES_STR=${ENGINE_VALIDATE_PHASES:-}
  want() { case "$PHASES_STR" in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
  SERVER=$ENGINE_VALIDATE_SERVER SERVER_ARGS=$ENGINE_VALIDATE_SERVER_ARGS
  BASELINE_SERVER=$ENGINE_VALIDATE_BASELINE UPSTREAM=$ENGINE_VALIDATE_UPSTREAM
  UPSTREAM_ARGS=$ENGINE_VALIDATE_UPSTREAM_ARGS UPSTREAM_MTP_ARGS=$ENGINE_VALIDATE_UPSTREAM_MTP_ARGS
  MTP_OFF_ARGS=$ENGINE_VALIDATE_MTP_OFF_ARGS MTP_OFF_ENV=$ENGINE_VALIDATE_MTP_OFF_ENV
  EXTRA=$ENGINE_VALIDATE_EXTRA PORT=$ENGINE_VALIDATE_PORT RUNS=$ENGINE_VALIDATE_RUNS
  RESULTS=$ENGINE_VALIDATE_RESULTS NO_RELEASE_LINE=$ENGINE_VALIDATE_NO_RELEASE_LINE
  run_one "$@"
fi

explicit_out=0
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
    --out) RESULTS=$2; explicit_out=1; shift 2 ;;
    --build) BUILD=1; shift ;;
    --yield) YIELD=--yield; shift ;;
    --no-release-line) NO_RELEASE_LINE=--no-release-line; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,66p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
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
for passthrough in "$SERVER_ARGS" "$UPSTREAM_ARGS" "$UPSTREAM_MTP_ARGS" "$MTP_OFF_ARGS"; do
  check_loopback_args "$passthrough"
done

# Identity is only meaningful with every reference it compares against.
if want identity; then
  missing=()
  for key in "${MODEL_KEYS[@]}"; do
    case "$("${checker[@]}" registry "$key" kind)" in
      gturbo) [ -n "$BASELINE_SERVER" ] || missing+=("--baseline-server (for $key)") ;;
      gguf) [ -n "$UPSTREAM" ] || missing+=("--upstream-llama-server (for $key)") ;;
    esac
    if [ "$key" = deep ]; then
      [ -n "$MTP_OFF_ARGS$MTP_OFF_ENV" ] || missing+=("--mtp-off-args or --mtp-off-env (for deep)")
      [ -n "$UPSTREAM_MTP_ARGS" ] || missing+=("--upstream-mtp-args (for deep)")
    fi
  done
  if [ ${#missing[@]} -gt 0 ]; then
    fail "the identity phase needs: ${missing[*]}; supply them or leave identity out of the phases"
  fi
fi

# Label results with the commit the server under test was built from: the
# checkout that owns its .build directory, else this script's checkout.
server_repo=$repo_root
case "$SERVER" in */.build/*) server_repo=${SERVER%%/.build/*} ;; esac
sha=$(git -C "$server_repo" rev-parse --short=7 HEAD 2>/dev/null || echo unknown)
if [ -z "$RESULTS" ]; then
  RESULTS="$RESULTS_ROOT/$(date +%Y-%m-%d)-$sha"
  if [ "$DRY_RUN" = 0 ] && [ -e "$RESULTS" ]; then
    fail "$RESULTS already exists; pass --out <dir> to reuse a directory explicitly"
  fi
fi

# The run list: (key variant). Identity and the speed bar need the reference runs.
RUN_LIST=()
for key in "${MODEL_KEYS[@]}"; do
  kind=$("${checker[@]}" registry "$key" kind)
  RUN_LIST+=("$key engine")
  if [ "$key" = deep ] && [ -n "$MTP_OFF_ARGS$MTP_OFF_ENV" ]; then
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

LOCK_RUN=("$LOCK" run --quiet ${YIELD:+"$YIELD"} --wait 3600 "$WS")
lock_cmd() { printf '%q ' "${LOCK_RUN[@]}" "$1" -- "${@:2}"; echo; }

if [ "$DRY_RUN" = 1 ]; then
  echo "results: $RESULTS"
  echo "phases: ${PHASES[*]}"
  if [ "$BUILD" = 1 ]; then
    printf '%q ' "$LOCK" run --wait 3600 "$WS" "eng build validate server" -- \
      "$LOCK" build -- swift build -c release --product MferenceServer; echo
  fi
  if want models || want prompts || want readings || want identity; then
    for item in "${RUN_LIST[@]}"; do
      read -r key variant <<<"$item"
      lock_cmd "eng validate $key $variant" "$script_dir/engine-validate.sh" _run "$key" "$variant" "$RESULTS/$key/$variant"
    done
  fi
  if want swap; then
    lock_cmd "eng validate swap chain" "$script_dir/engine-validate.sh" _run quick swap "$RESULTS/swap"
  fi
  if want identity; then echo "identity: engine vs main (native), engine vs upstream (GGUF), Deep MTP on vs off"; fi
  if want readings; then echo "speed bar: engine within 5% of main (native) and upstream (GGUF)"; fi
  if want summary; then echo "summary: $RESULTS/summary.md"; fi
  exit 0
fi

[ -x "$LOCK" ] || fail "lock script not found: $LOCK"
mkdir -p "$RESULTS"

if [ "$BUILD" = 1 ]; then
  (cd "$server_repo" && "$LOCK" run --wait 3600 "$WS" "eng build validate server" -- \
    "$LOCK" build -- swift build -c release --product MferenceServer) \
    || fail "build failed or the quiet flag is set; stop and wait"
fi
needs_server=0
for p in models prompts readings identity swap; do if want "$p"; then needs_server=1; fi; done
if [ "$needs_server" = 1 ]; then
  [ -x "$SERVER" ] || fail "server binary not found: $SERVER (build it, or pass --build)"
fi

# meta.json describes the run that produced the measurements: a summary-only
# rebuild keeps it as it is.
if [ "$needs_server" = 1 ] || [ ! -f "$RESULTS/meta.json" ]; then
  "${checker[@]}" meta "$RESULTS/meta.json" --repo "$server_repo" --server "$SERVER" \
    --baseline "$BASELINE_SERVER" --upstream "$UPSTREAM" --argv "$ORIG_ARGS" --phases "${PHASES[*]}"
fi

export ENGINE_VALIDATE_PHASES="$PHASES_STR" ENGINE_VALIDATE_SERVER="$SERVER"
export ENGINE_VALIDATE_SERVER_ARGS="$SERVER_ARGS" ENGINE_VALIDATE_BASELINE="$BASELINE_SERVER"
export ENGINE_VALIDATE_UPSTREAM="$UPSTREAM" ENGINE_VALIDATE_UPSTREAM_ARGS="$UPSTREAM_ARGS"
export ENGINE_VALIDATE_UPSTREAM_MTP_ARGS="$UPSTREAM_MTP_ARGS"
export ENGINE_VALIDATE_MTP_OFF_ARGS="$MTP_OFF_ARGS" ENGINE_VALIDATE_MTP_OFF_ENV="$MTP_OFF_ENV"
export ENGINE_VALIDATE_EXTRA="$EXTRA" ENGINE_VALIDATE_PORT="$PORT" ENGINE_VALIDATE_RUNS="$RUNS"
export ENGINE_VALIDATE_RESULTS="$RESULTS" ENGINE_VALIDATE_NO_RELEASE_LINE="$NO_RELEASE_LINE"

abort() {
  "${checker[@]}" summary "$RESULTS" >/dev/null 2>&1 || true
  note "$1"
  exit "$2"
}
trap 'abort "interrupted: no further model runs" 130' INT TERM HUP

overall=0
locked_run() {
  local purpose=$1; shift
  local rc=0
  "${LOCK_RUN[@]}" "$purpose" -- "$script_dir/engine-validate.sh" _run "$@" || rc=$?
  case "$rc" in
    0) ;;
    "$SWAP_TRIP") abort "SWAP GUARD TRIPPED during $purpose: all model runs stopped; report to the coordinator" "$SWAP_TRIP" ;;
    "$REFUSED") abort "$purpose refused (preflight, power, another model process or port): stopped; report to the coordinator" "$REFUSED" ;;
    75) abort "could not take the machine lock for $purpose: stopped; wait and rerun" 75 ;;
    130|143|129) abort "$purpose interrupted: no further model runs" 130 ;;
    *) note "$purpose failed (exit $rc); continuing with the next model"; overall=1 ;;
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

# Pairwise checks. A missing side is recorded as a failure, never skipped.
pair() {
  local check=$1 label=$2 a=$3 b=$4
  "${checker[@]}" "$check" "$a" "$b" --label "$label" --out "$RESULTS/$check-$label.json" \
    || overall=1
}
for key in "${MODEL_KEYS[@]}"; do
  kind=$("${checker[@]}" registry "$key" kind)
  if [ "$kind" = gguf ]; then ref=upstream ref_binary=$UPSTREAM
  else ref=main ref_binary=$BASELINE_SERVER; fi
  if want identity; then
    pair identity "$key-engine-vs-$ref" "$RESULTS/$key/engine" "$RESULTS/$key/$ref"
  fi
  if want readings && [ -n "$ref_binary" ]; then
    pair speed "$key-engine-vs-$ref" "$RESULTS/$key/engine" "$RESULTS/$key/$ref"
    if [ "$key" = deep ] && [ -n "$MTP_OFF_ARGS$MTP_OFF_ENV" ] && [ -n "$UPSTREAM_MTP_ARGS" ]; then
      pair speed "deep-nomtp-engine-vs-upstream" "$RESULTS/deep/engine-nomtp" "$RESULTS/deep/upstream-nomtp"
    fi
  fi
done
if want identity && printf '%s\n' "${MODEL_KEYS[@]}" | grep -qx deep; then
  pair identity deep-mtp-on-vs-off "$RESULTS/deep/engine" "$RESULTS/deep/engine-nomtp"
  pair identity deep-upstream-mtp-on-vs-off "$RESULTS/deep/upstream" "$RESULTS/deep/upstream-nomtp"
fi

"${checker[@]}" summary "$RESULTS" >/dev/null
note "results: $RESULTS (summary.md); exit $overall"
exit "$overall"
