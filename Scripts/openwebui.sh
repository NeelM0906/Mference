#!/usr/bin/env bash
# Runs Open WebUI against MferenceServer in library mode, both on loopback.
#
# Starts one MferenceServer (127.0.0.1:8080) serving every completed install it
# finds, waits for /health, then starts `open-webui serve` (127.0.0.1:3000)
# pointed at it. Control-C stops both.
#
# See docs/OPEN_WEBUI.md. Open WebUI authentication is disabled here, so
# neither process may be exposed beyond loopback.

set -euo pipefail

server_port=8080
webui_port=3000
max_context=16384
prompt_cache_mode=single-prefix
preload_model=""
library_roots=()
dry_run=0

usage() {
  cat <<'USAGE'
usage: Scripts/openwebui.sh [options]

  --library <dir>        Library root to scan, repeatable. Defaults to the
                         roots the Mac app scans when omitted.
  --model <dir>          Preload this install instead of loading lazily.
  --server-port <port>   MferenceServer port (default 8080).
  --webui-port <port>    Open WebUI port (default 3000).
  --max-context <tokens> Context window for every model (default 16384).
  --prompt-cache-mode <off|single-prefix>
                         Passed through to MferenceServer.
  --dry-run              Print what would run, start nothing, exit 0.
  --help                 Show this message.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --library) library_roots+=("$2"); shift 2 ;;
    --model) preload_model="$2"; shift 2 ;;
    --server-port) server_port="$2"; shift 2 ;;
    --webui-port) webui_port="$2"; shift 2 ;;
    --max-context) max_context="$2"; shift 2 ;;
    --prompt-cache-mode) prompt_cache_mode="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "error: unknown option $1" >&2; usage >&2; exit 2 ;;
  esac
done

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "$script_directory/.." && pwd)"
server_binary="$repository_root/.build/release/MferenceServer"

server_arguments=(
  --port "$server_port"
  --max-context "$max_context"
  --prompt-cache-mode "$prompt_cache_mode"
)
if [[ ${#library_roots[@]} -eq 0 ]]; then
  server_arguments+=(--library)
else
  for root in "${library_roots[@]}"; do
    server_arguments+=(--library "$root")
  done
fi
if [[ -n "$preload_model" ]]; then
  server_arguments+=(--model "$preload_model")
fi

data_directory="$HOME/Library/Application Support/Mference/open-webui"

# Every one of these would otherwise fire an extra generation into a queue that
# runs one generation at a time: a title, tags, follow-ups, an autocomplete, and
# two query rewrites per turn. They seed Open WebUI's persisted defaults, so
# they take effect on a fresh DATA_DIR and are overridden by anything already
# toggled in its admin settings.
webui_environment=(
  "OPENAI_API_BASE_URL=http://127.0.0.1:$server_port/v1"
  "OPENAI_API_KEY=local"
  "ENABLE_OLLAMA_API=false"
  "WEBUI_AUTH=false"
  "ENABLE_TITLE_GENERATION=false"
  "ENABLE_TAGS_GENERATION=false"
  "ENABLE_FOLLOW_UP_GENERATION=false"
  "ENABLE_AUTOCOMPLETE_GENERATION=false"
  "ENABLE_RETRIEVAL_QUERY_GENERATION=false"
  "ENABLE_SEARCH_QUERY_GENERATION=false"
  "DATA_DIR=$data_directory"
)

webui_binary="$(command -v open-webui || true)"
if [[ -z "$webui_binary" && -x "$HOME/.local/bin/open-webui" ]]; then
  webui_binary="$HOME/.local/bin/open-webui"
fi

# Anchored on the executable path so a shell whose command line merely quotes
# these names (a poll loop, an editor, this script's own dry run) does not
# count as a model process.
model_process_pattern='(^|/)(MferenceServer|MferenceCLI|MferenceRepack|MferencePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm)( |$)'

check_no_model_process() {
  # AGENTS.md: one model-owning process at a time. Never terminate one that is
  # already running; refuse to start instead.
  local running
  running="$(pgrep -fl "$model_process_pattern" || true)"
  if [[ -n "$running" ]]; then
    echo "error: another Mference model process is running; refusing to start." >&2
    echo "$running" >&2
    exit 1
  fi
}

if [[ "$dry_run" -eq 1 ]]; then
  echo "would check: pgrep -fl '$model_process_pattern'"
  echo "would run:   $server_binary ${server_arguments[*]}"
  echo "would wait:  http://127.0.0.1:$server_port/health"
  echo "would run:   $webui_binary serve --host 127.0.0.1 --port $webui_port"
  echo "with env:"
  for entry in "${webui_environment[@]}"; do
    echo "  $entry"
  done
  exit 0
fi

if [[ ! -x "$server_binary" ]]; then
  echo "error: $server_binary is missing." >&2
  echo "build it first: swift build -c release --product MferenceServer" >&2
  exit 1
fi
if [[ -z "$webui_binary" ]]; then
  echo "error: open-webui is not on PATH." >&2
  echo "install it: uv tool install --python 3.11 open-webui" >&2
  exit 1
fi

check_no_model_process

server_pid=""
webui_pid=""

stop_children() {
  trap - INT TERM EXIT
  # Only the two processes this script started.
  for pid in "$webui_pid" "$server_pid"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap stop_children INT TERM EXIT

mkdir -p "$data_directory"

echo "starting MferenceServer on 127.0.0.1:$server_port (library mode)"
"$server_binary" "${server_arguments[@]}" &
server_pid=$!

# Library mode opens the port before loading anything, so /health answers
# quickly. A preload with --model has to verify the install first, which is why
# the wait is generous.
health_url="http://127.0.0.1:$server_port/health"
for _ in $(seq 1 600); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "error: MferenceServer exited before it became healthy." >&2
    exit 1
  fi
  if curl --silent --show-error --fail --max-time 2 "$health_url" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if ! curl --silent --show-error --fail --max-time 2 "$health_url" >/dev/null 2>&1; then
  echo "error: MferenceServer did not answer $health_url in time." >&2
  exit 1
fi
echo "MferenceServer is healthy; models:"
curl --silent --show-error "http://127.0.0.1:$server_port/v1/models" || true
echo

echo "starting Open WebUI on http://127.0.0.1:$webui_port"
env "${webui_environment[@]}" "$webui_binary" serve --host 127.0.0.1 --port "$webui_port" &
webui_pid=$!

# Exit when either child does, then stop the other through the trap. macOS
# ships bash 3.2, which has no `wait -n`, so poll both children instead.
while kill -0 "$server_pid" 2>/dev/null && kill -0 "$webui_pid" 2>/dev/null; do
  sleep 1
done
