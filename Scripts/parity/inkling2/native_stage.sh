#!/bin/bash
# native_stage.sh: one stage of the native side of the K0 baseline table:
# MferenceCLI on the real Inkling 2 Mini install, on the same real-text prompts
# and settings as baseline_stage.sh's text-* stages, through
# guarded_run.sh --timing. One `model-hold.sh --timing` hold per stage.
#
#   native_stage.sh <cli-dir> <install-dir> <out-dir> <stage>
#
# <cli-dir> holds a release MferenceCLI with its resource bundles beside it
# (a copied binary without Mference_Mference.bundle traps at launch).
# Stages (shortest first; each mirrors the llama-completion stage of the same
# size: raw prompt, greedy, same context and new-token count):
#   native-512   prompt-512.txt,  --max-context 4096,  --max-new 128
#   native-4k    prompt-4k.txt,   --max-context 8192,  --max-new 64
#   native-16k   prompt-16k.txt,  --max-context 20480, --max-new 32
# The prompts are $INK/results/k0/baseline/prompts/prompt-*.txt, checked
# against the SHA-256 recorded in baseline.md; any difference refuses.
# Default CLI controls otherwise (prefill chunking, expert streaming), except
# `--verify trusted-receipt`: the expert pool is checked against the install
# receipt's sizes, and an invalid receipt refuses the run instead of falling
# back to hashing the pool inside the timed prefill. No warm-up run: unlike
# llama-completion (whose warm-up decode falls in its excluded load time), the
# native prefill includes first-use costs (first expert reads, pipeline
# builds). If the runner's expert plan does not fit the host budget the CLI
# exits with `error: ...`; this script surfaces that line and its exit.
# Build the CLI from a tree with the Inkling 2 runner (inkling/trunk); its
# SOURCE.txt, if present beside it, is recorded.
# native_table.py writes the table from <out-dir>.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

# Internal: guarded_run.sh runs `native_stage.sh --exec <cli> <prompt> args...`
# so its logs and run.json record a short command line; the prompt text is
# read here and the CLI replaces this shell (same pid: /usr/bin/time and the
# swap guard see the CLI itself).
if [ "${1:-}" = "--exec" ]; then
  cli=$2; prompt=$3; shift 3
  exec "$cli" --prompt "$(cat "$prompt")" "$@"
fi

CLI_DIR=${1:?cli dir}; INST=${2:?install dir}; OUT=${3:?out dir}; STAGE=${4:?stage}
. "$HERE/paths.sh"
P=$INK/results/k0/baseline/prompts
case "$STAGE" in
  native-512) f=prompt-512.txt; ctx=4096;  n=128; sha=cdd0e7d21c319815b0fc7dab29c28de8e7a0d5141633abf2eb396122a95443f3 ;;
  native-4k)  f=prompt-4k.txt;  ctx=8192;  n=64;  sha=3e4ab329b2c23a46b9e0cecfa434ffc09be0f4c8c970d3b44e4234cfb008a676 ;;
  native-16k) f=prompt-16k.txt; ctx=20480; n=32;  sha=f4ca3dacc71a6c944ce35efd8e724e36a1113c281413fe44b9bdeb7176bb2f7a ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
D=$OUT/$STAGE
# A rerun starts clean: an earlier attempt's files move to prev-<UTC>/ so a
# refused attempt never leaves the last attempt's numbers in the table.
if [ -d "$D" ] && [ -n "$(ls -A "$D" 2>/dev/null | grep -v '^prev-')" ]; then
  prev=$D/prev-$(date -u +%Y%m%dT%H%M%SZ)
  mkdir -p "$prev"
  for x in "$D"/*; do case "$(basename "$x")" in prev-*) ;; *) mv "$x" "$prev/" ;; esac; done
fi
mkdir -p "$D"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$D/checks.log"; }
refuse() { log "REFUSED: $*"; exit 71; }

CLI=$CLI_DIR/MferenceCLI
[ -x "$CLI" ] || refuse "no MferenceCLI at $CLI"
for b in Mference_Mference.bundle Mference_LlamaCpp.bundle; do
  [ -d "$CLI_DIR/$b" ] || refuse "$b missing beside the CLI (a copied CLI traps without it)"
done
[ -f "$INST/manifest.json" ] || refuse "no install at $INST (manifest.json missing)"
got=$(shasum -a 256 "$P/$f" | cut -d' ' -f1)
[ "$got" = "$sha" ] || refuse "$f SHA-256 $got is not the baseline's $sha"
log "stage $STAGE: $f (SHA-256 $got), max-context $ctx, max-new $n, greedy"
log "cli: $CLI (SHA-256 $(shasum -a 256 "$CLI" | cut -d' ' -f1))"
if [ -f "$CLI_DIR/SOURCE.txt" ]; then log "cli source: $(head -1 "$CLI_DIR/SOURCE.txt")"
else log "cli source: no SOURCE.txt beside the CLI"; fi
log "install: $INST (manifest SHA-256 $(shasum -a 256 "$INST/manifest.json" | cut -d' ' -f1))"
if ps -axo comm= | grep -q '/Mference.app/Contents/MacOS/'; then  # executable path, not command lines
  log "app open check: the Mference app is open"   # guarded_run.sh --timing refuses below
else
  log "app open check: not open"
fi

# Same rule as baseline_stage.sh: at most one other swift build/test job;
# wait up to 30 min, stop nothing, record the counts.
swift_jobs() { pgrep -f '/usr/bin/swift-(build|test) ' | wc -l | tr -d ' '; }
j=$(swift_jobs)
for _ in $(seq 1 180); do
  [ "$j" -le 1 ] && break
  sleep 10; j=$(swift_jobs)
done
echo "$(date -u +%FT%TZ) swift build/test jobs at start: $j" | tee -a "$D/jobs.log"
[ "$j" -le 1 ] || { echo "REFUSED: $j swift jobs still running after 30 min" | tee -a "$D/jobs.log"; exit 71; }

"$HERE/guarded_run.sh" --timing "$D" "$INST/manifest.json" -- \
  "$HERE/native_stage.sh" --exec "$CLI" "$P/$f" \
  --model "$INST" --max-context "$ctx" --max-new "$n" --temperature 0 --seed 7 \
  --verify trusted-receipt
rc=$?
echo "$(date -u +%FT%TZ) swift build/test jobs at end: $(swift_jobs)" | tee -a "$D/jobs.log"
# The stage's exit (70: swap grew, even after a clean CLI exit; 71: refused
# before the CLI ran, reason in run.log) and the CLI's error line, if any.
err=$(grep '^error:' "$D/stderr.log" 2>/dev/null | tail -1)
log "stage exit $rc${err:+; CLI: $err}"
exit $rc
