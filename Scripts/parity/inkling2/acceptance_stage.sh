#!/bin/bash
# acceptance_stage.sh: one C4 acceptance stage against a real MferenceServer
# --library, started and stopped inside this script, run as the single command
# of one hold:
#
#   $INK/bin/model-hold.sh ws0 "acceptance <stage>" -- \
#     Scripts/parity/inkling2/guarded_run.sh <out>/<stage>/hold <inkling install>/manifest.json -- \
#     Scripts/parity/inkling2/acceptance_stage.sh <server-dir> <out> <stage>
#
# <server-dir> holds a release MferenceServer with its resource bundles beside
# it and a SOURCE.txt (first line: the source commit). The server binds
# 127.0.0.1 only, serves the library's default roots (the Inkling install in
# scratch/, Deep's GGUF in ~/llm-models/gguf) with --max-context 40960 (Inkling
# runs at its 40,960 cap; Deep stays at 32,768, coordinator ruling) and
# --prompt-cache-mode off, and is stopped on every exit: SIGINT, then KILL
# (the server ignores a TERM after INT). Under the swap guard's TERM the server
# gets 20 s before KILL, inside the guard's 30 s before it KILLs this script,
# so the server can never outlive the hold; on a normal end it gets 120 s.
# A server that dies during the stage fails the stage (exit 75). One model is resident at a
# time; the server swaps between them.
#
# Stages (client: agent_acceptance.py, corpus $INK/acceptance/agent-acceptance-v1):
#   load-inkling     first real load of Inkling at 40,960 and its longest prompt:
#                    full tier, block C, c09-30k-cross-doc (Inkling only)
#   swap             one short A case on Inkling, then the same on Deep: the swap
#   slice1-inkling   gate tier block A without tool cases (16 cases), Inkling
#   slice1-deep      the same on Deep
#   gate-inkling     the whole gate tier (A 20, B 16, C 6), Inkling
#   gate-deep        the whole gate tier, Deep
#   gate:<inkling|deep>:<blocks>   the gate tier's blocks given (e.g. gate:inkling:B),
#                    so a long tier can be split into shorter holds
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=paths.sh
. "$HERE/paths.sh"
SRV_DIR=${1:?server dir}; OUT=${2:?out dir}; STAGE=${3:?stage}
INK_ID=inkling-2-mini-ud-q4_k_xl
DEEP_ID=qwen3.8-27b
PORT=${ACCEPTANCE_PORT:-18489}
CORPUS=$INK/acceptance/agent-acceptance-v1
INK_INSTALL=${ACCEPTANCE_INKLING_INSTALL:-$(dirname "$INK")/inkling2mini.gturbo}
DEEP_GGUF=${ACCEPTANCE_DEEP_GGUF:-$HOME/llm-models/gguf/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf}
SLICE1=everyday-hours,everyday-time,everyday-calendar,reasoning-discount,reasoning-order,reasoning-sets,code-slice,code-alias,code-modulo,instruction-json,instruction-exact,instruction-dedupe,synthesis-total,synthesis-missing,synthesis-priority,synthesis-count

# run specs: "<model id> <tier> <blocks> <cases or -> [extra client arguments]"
case "$STAGE" in
  load-inkling)   RUNS=("$INK_ID full C c09-30k-cross-doc --seeds 1 --warmups 0") ;;
  swap)           RUNS=("$INK_ID gate A everyday-hours" "$DEEP_ID gate A everyday-hours") ;;
  slice1-inkling) RUNS=("$INK_ID gate A $SLICE1") ;;
  slice1-deep)    RUNS=("$DEEP_ID gate A $SLICE1") ;;
  gate-inkling)   RUNS=("$INK_ID gate A,B,C -") ;;
  gate-deep)      RUNS=("$DEEP_ID gate A,B,C -") ;;
  gate:inkling:*|gate:deep:*)
    blk=${STAGE##*:}
    [[ "$blk" =~ ^[A-E](,[A-E])*$ ]] || { echo "bad blocks '$blk' in $STAGE (want e.g. A or B,C)" >&2; exit 2; }
    if [[ "$STAGE" == gate:inkling:* ]]; then RUNS=("$INK_ID gate $blk -"); else RUNS=("$DEEP_ID gate $blk -"); fi ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac

D=$OUT/${STAGE//:/-}/run
if [ -e "$D" ]; then echo "REFUSED: $D exists; use a new out dir" >&2; exit 71; fi
mkdir -p "$D"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$D/stage.log"; }
refuse() { log "REFUSED: $*"; exit 71; }

[ -x "$SRV_DIR/MferenceServer" ] || refuse "no MferenceServer in $SRV_DIR"
for b in Mference_Mference.bundle Mference_LlamaCpp.bundle; do
  [ -d "$SRV_DIR/$b" ] || refuse "$b missing beside the server"
done
[ -f "$SRV_DIR/SOURCE.txt" ] || refuse "no SOURCE.txt beside the server"
[ -f "$INK_INSTALL/manifest.json" ] || refuse "no Inkling install at $INK_INSTALL"
[ -f "$DEEP_GGUF" ] || refuse "no Deep GGUF at $DEEP_GGUF"
ENGINE_COMMIT=$(head -1 "$SRV_DIR/SOURCE.txt")
(cd "$CORPUS" && shasum -a 256 -c --quiet MANIFEST.sha256) || refuse "corpus does not match MANIFEST.sha256"
if pgrep -f 'Mference.app/Contents/MacOS/' > /dev/null; then refuse "the Mference app is open"; fi
{ sw_vers; sysctl -n hw.model hw.memsize; echo "server: $ENGINE_COMMIT";
  echo "server sha256: $(shasum -a 256 "$SRV_DIR/MferenceServer" | cut -d' ' -f1)"; } > "$D/machine.txt"
log "stage $STAGE; server $ENGINE_COMMIT; port $PORT"

# Deep's file SHA-256, before the server starts: hashing reads 12 GB (about 40 s,
# inside this hold), and a TERM that arrives during a foreground command waits
# for it, so no server may be running yet. Cached beside the results, keyed by
# size and modification time.
DEEP_SHA=""
if printf '%s\n' "${RUNS[@]}" | grep -q "^$DEEP_ID "; then
  key="$(stat -f '%z %m' "$DEEP_GGUF")"
  cache=$OUT/deep-gguf.sha256
  if [ -f "$cache" ] && [ "$(head -1 "$cache")" = "$key" ]; then
    DEEP_SHA=$(sed -n 2p "$cache")
  else
    log "hashing $(basename "$DEEP_GGUF")"
    DEEP_SHA=$(shasum -a 256 "$DEEP_GGUF" | cut -d' ' -f1)
    printf '%s\n%s\n' "$key" "$DEEP_SHA" > "$cache"
  fi
  log "deep: $(basename "$DEEP_GGUF") sha256 $DEEP_SHA"
fi

SPID=""; CPID=""; BUDGET=120
stop_server() {
  # TERM to the client: a command started with & by a non-interactive shell
  # (as guarded_run.sh starts this script) inherits SIGINT ignored
  [ -n "$CPID" ] && kill -0 "$CPID" 2>/dev/null && kill -TERM "$CPID" 2>/dev/null
  if [ -n "$SPID" ] && kill -0 "$SPID" 2>/dev/null; then
    kill -INT "$SPID" 2>/dev/null
    for _ in $(seq 1 "$BUDGET"); do kill -0 "$SPID" 2>/dev/null || break; sleep 1; done
    if kill -0 "$SPID" 2>/dev/null; then
      log "server still running ${BUDGET} s after INT; KILL"
      kill -KILL "$SPID" 2>/dev/null
    fi
    wait "$SPID" 2>/dev/null
    log "server stopped"
  fi
  grep -E "swap|release|load|unload" "$D/server.log" 2>/dev/null | tail -40 > "$D/server-swaps.log"
}
trap 'stop_server' EXIT
# INT may be ignored on entry; TERM is the swap guard's signal. 20 s keeps the
# whole stop inside the guard's 30 s before its KILL of this script.
trap 'BUDGET=20; log "interrupted"; exit 130' INT TERM

"$SRV_DIR/MferenceServer" --library --port "$PORT" --max-context 40960 --prompt-cache-mode off \
  > "$D/server.log" 2>&1 &
SPID=$!
ready=0
for _ in $(seq 1 900); do
  grep -q "MferenceServer ready" "$D/server.log" && { ready=1; break; }
  kill -0 "$SPID" 2>/dev/null || break
  sleep 1
done
[ $ready = 1 ] || refuse "server not ready (see server.log)"
log "ready: $(grep 'MferenceServer ready' "$D/server.log" | tail -1)"
# provenance: which install or file the server discovered for each model
grep -E "$INK_ID|$DEEP_ID" "$D/server.log" | head -10 | while read -r line; do log "server: $line"; done
curl -sS "http://127.0.0.1:$PORT/v1/models" -o "$D/models.json" || refuse "GET /v1/models failed"
for id in $(echo "${RUNS[@]}" | tr ' ' '\n' | grep -E "^($INK_ID|$DEEP_ID)$" | sort -u); do
  grep -q "\"$id\"" "$D/models.json" || refuse "$id not listed by /v1/models"
done

server_alive() {
  kill -0 "$SPID" 2>/dev/null && return 0
  wait "$SPID" 2>/dev/null; local st=$?
  log "SERVER DIED (exit $st) during the stage; see server.log"
  SPID=""
  return 1
}
n=0; rc=0
for spec in "${RUNS[@]}"; do
  server_alive || { rc=75; break; }
  read -r model tier blocks cases extra <<< "$spec"
  n=$((n + 1))
  prof=$D/profile-$n.json
  if [ "$model" = "$INK_ID" ]; then
    rev="$(basename "$INK_INSTALL") manifest sha256 $(shasum -a 256 "$INK_INSTALL/manifest.json" | cut -d' ' -f1)"
  else
    rev="$(basename "$DEEP_GGUF") $(stat -f %z "$DEEP_GGUF") bytes sha256 $DEEP_SHA"
  fi
  printf '[{"label": "product-default", "model": "%s", "source_revision": "%s"}]\n' "$model" "$rev" > "$prof"
  args=(run --corpus "$CORPUS" --tier "$tier" --blocks "$blocks" --profiles "$prof"
        --engine-commit "$ENGINE_COMMIT" --machine-record "$D/machine.txt" --output "$D/run-$n-$model.jsonl" --port "$PORT")
  [ "$cases" != "-" ] && args+=(--cases "$cases")
  # shellcheck disable=SC2206  # the extra arguments are this script's own literals
  [ -n "${extra:-}" ] && args+=($extra)
  log "run $n: $model $tier $blocks ${cases:0:60}"
  # in the background, so an INT (the swap guard's) runs the trap at once
  python3 -I "$HERE/agent_acceptance.py" "${args[@]}" > "$D/run-$n.log" 2>&1 &
  CPID=$!
  wait "$CPID"
  r=$?
  CPID=""
  log "run $n exit $r"
  [ $r = 0 ] || { rc=$r; break; }
  # a client finishes with transport_error rows when the server is gone: that is a failed stage
  server_alive || { rc=75; break; }
  python3 -I "$HERE/agent_acceptance.py" summarize "$D/run-$n-$model.jsonl" > "$D/run-$n-summary.md" 2>&1 \
    || log "summarize of run $n failed (see run-$n-summary.md)"
done
exit $rc
