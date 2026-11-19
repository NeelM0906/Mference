#!/bin/bash
# Checkpoint C-A2: a web question answered with sources, on Quick and Deep,
# through the app's own path (ChatViewModel + web tools + bundled SearXNG +
# EngineService.live). LOADS REAL MODELS: only the app coordinator starts it,
# when it schedules the checkpoint around engine work (maintainer rule
# 2026-11-19). App developers never run it.
#
# Usage: App/Scripts/searxng/checkpoint-c-a2.sh [model ...]     (default: quick deep)
#   env OUT=<dir>   results folder (default <checkout>/scratch/app-program/results/checkpoints/c-a2)
#   env LOCK=<path> lock.sh (default <checkout>/scratch/program/lock.sh)
#
# Steps:
#   1. Without the lock (app builds run unlocked): the SearXNG runtime from the
#      verified cache (prepare.sh --offline) and a release MferenceSmoke.
#   2. Per model, one guarded run: `lock.sh run` -> `lock.sh preflight` (GGUF
#      models take no --model) -> MferenceSmoke --web. Models run one after
#      another, never in parallel, chained with && only.
# Exit: 0 every model passed; otherwise MferenceSmoke's code for the first
# model that did not (1 failed answer, 4 swap grew: stop all model runs and
# report, 5 not on AC power, 8 swap unreadable), or a build/lock error.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(cd "$HERE/../.." && pwd)"
CHECKOUT="$(cd "$(dirname "$(git -C "$APP_DIR" rev-parse --path-format=absolute --git-common-dir)")" && pwd)"
LOCK="${LOCK:-$CHECKOUT/scratch/program/lock.sh}"
OUT="${OUT:-$CHECKOUT/scratch/app-program/results/checkpoints/c-a2}"
MODELS=("$@")
[[ ${#MODELS[@]} -gt 0 ]] || MODELS=(quick deep)
COMMIT="$(git -C "$APP_DIR" rev-parse --short HEAD)"

[[ -x "$LOCK" ]] || { echo "lock.sh not found at $LOCK (set LOCK=)" >&2; exit 2; }
mkdir -p "$OUT"

echo "==> C-A2 at $COMMIT: models ${MODELS[*]}; results in $OUT"
echo "==> SearXNG runtime (offline, verified cache)"
"$APP_DIR/Scripts/searxng/prepare.sh" --offline
echo "==> release MferenceSmoke"
swift build -c release --package-path "$APP_DIR" -j 4 --product MferenceSmoke
SMOKE="$(swift build -c release --package-path "$APP_DIR" --show-bin-path)/MferenceSmoke"
RUNTIME="$APP_DIR/Resources/SearXNG/runtime"

for model in "${MODELS[@]}"; do
    echo "==> $model: web question under the lock"
    status=0
    "$LOCK" run --wait 14400 a3 "C-A2 web answer $model" -- bash -c \
        "\"$LOCK\" preflight && MFERENCE_APP_SMOKE=1 MFERENCE_SEARXNG_DIR=\"$RUNTIME\" \"$SMOKE\" --model $model --web --commit $COMMIT --out \"$OUT\"" \
        || status=$?
    if [[ $status -ne 0 ]]; then
        echo "==> $model did not pass (exit $status); stopping. Results: $OUT" >&2
        [[ $status -eq 4 ]] && echo "==> swap grew during the run: stop all model runs and report." >&2
        exit "$status"
    fi
    echo "==> $model passed"
done
echo "==> C-A2 passed on ${MODELS[*]} at $COMMIT. Results: $OUT"
