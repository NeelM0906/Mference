#!/bin/bash
# g5_server_stage.sh <server-dir> <out-dir> model <family> | library : one G5-server server process, run as
# the single command of guarded_run.sh inside one plain hold (G5-server compares outputs, not speed).
# <server-dir> holds a release MferenceServer with its bundles and a SOURCE.txt; the installs are
# ~/llm-models/{minicpm5,qwen36,maple}.gturbo. Starts MferenceServer on 127.0.0.1
# (--max-context 4096, default prompt cache and verify) with --model <install> or --library, runs
# g5_server.py's cases for the family (or for minicpm5, qwen36, maple in turn), stops the server
# (INT, KILL after 20 s; TERM from the guard handled the same way).
set -uo pipefail
SRV=${1:?server dir}; O=${2:?out dir}; MODE=${3:?model|library}; FAM=${4:-}
HERE=$(cd "$(dirname "$0")" && pwd)
G=$HERE/g5_server.py
PORT=${G5_PORT:-18491}
inst() { echo "$HOME/llm-models/$1.gturbo"; }   # minicpm5, qwen36, maple
pid_of() { case $1 in minicpm5) echo minicpm5-2b-int4g64 ;; qwen36) echo qwen3.6-35b-a3b ;; maple) echo maple-preview-2bit-mlx ;; esac; }
mkdir -p "$O"; tag=$MODE${FAM:+-$FAM}
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$O/$tag.log"; }
SPID=""; CPID=""
stop() {
  [ -n "$CPID" ] && kill -0 "$CPID" 2>/dev/null && kill -TERM "$CPID" 2>/dev/null
  [ -n "$SPID" ] && kill -0 "$SPID" 2>/dev/null || return 0
  kill -INT "$SPID" 2>/dev/null
  for _ in $(seq 1 20); do kill -0 "$SPID" 2>/dev/null || break; sleep 1; done
  kill -0 "$SPID" 2>/dev/null && kill -KILL "$SPID" 2>/dev/null
  wait "$SPID" 2>/dev/null; log "server stopped"
}
trap stop EXIT
trap 'log interrupted; exit 130' INT TERM
if [ "$MODE" = model ]; then args=(--model "$(inst "$FAM")"); fams=("$FAM"); else args=(--library); fams=(minicpm5 qwen36 maple); fi
log "server $(head -1 "$SRV/SOURCE.txt"); args ${args[*]} --max-context 4096; port $PORT"
"$SRV/MferenceServer" "${args[@]}" --port "$PORT" --max-context 4096 > "$O/$tag.server.log" 2>&1 &
SPID=$!
for _ in $(seq 1 900); do grep -q "MferenceServer ready" "$O/$tag.server.log" && break; kill -0 "$SPID" 2>/dev/null || { log "server exited before ready"; exit 75; }; sleep 1; done
grep -q "MferenceServer ready" "$O/$tag.server.log" || { log "not ready"; exit 75; }
log "ready: $(grep 'MferenceServer ready' "$O/$tag.server.log" | tail -1)"
curl -sS "http://127.0.0.1:$PORT/v1/models" > "$O/$tag.models.json" || { log "GET /v1/models failed"; exit 75; }
rc=0
for f in "${fams[@]}"; do
  if [ "$MODE" = model ]; then id=$(python3 -I -c "import json,sys;print(json.load(open(sys.argv[1]))['data'][0]['id'])" "$O/$tag.models.json"); else id=$(pid_of "$f"); fi
  log "cases: $f as $id"
  python3 -I "$G" run --port "$PORT" --model "$id" --family "$f" --out "$O/$tag.jsonl" >> "$O/$tag.log" 2>&1 &
  CPID=$!; wait "$CPID"; r=$?; CPID=""
  log "cases $f exit $r"; [ $r = 0 ] || { rc=$r; break; }
  kill -0 "$SPID" 2>/dev/null || { log "SERVER DIED"; rc=75; break; }
done
exit $rc
