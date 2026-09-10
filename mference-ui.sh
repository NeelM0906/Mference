#!/usr/bin/env bash
# The Mference UI: Open WebUI in the browser, MferenceServer behind it.
#
#   ./mference-ui.sh                 start both on loopback and open the browser
#   ./mference-ui.sh install gemma4  install a model with MferenceRepack
#   ./mference-ui.sh models          list what the UI's model picker would show
#
# Both halves stay on 127.0.0.1 and Open WebUI's authentication is disabled, so
# neither process may be exposed beyond loopback. See docs/OPEN_WEBUI.md.

set -euo pipefail

# The Open WebUI release this launcher installs and expects. A different
# installed version is reported, never replaced.
OPEN_WEBUI_VERSION=0.11.3

server_port=8080
webui_port=3000
max_context=16384
prompt_cache_mode=single-prefix
preload_model=""
library_roots=()
dry_run=0
command_name=run
install_family=""
repack_arguments=()

# AGENTS.md's model-process check, minus the two names that went away with the
# UI. Anything matching owns a model already, and this script starts
# nothing while one exists — and never terminates one.
model_process_pattern='(^|/)(MferenceServer|MferenceCLI|MferenceRepack|MferencePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm)( |$)'

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$script_directory"
server_binary="$repository_root/.build/release/MferenceServer"
data_directory="$HOME/Library/Application Support/Mference/open-webui"

note() { printf '[mference-ui] %s\n' "$*" >&2; }
fail() { note "error: $*"; exit 1; }

usage() {
  cat <<'USAGE'
usage: ./mference-ui.sh [options]                      start the UI
       ./mference-ui.sh install <family> [repack args] install a model
       ./mference-ui.sh models                         list installed models

  --library <dir>        Library root to scan, repeatable. Defaults to the
                         Mference.libraryRoot default (or
                         MFERENCE_LIBRARY_ROOT), the checkout's scratch/, and
                         ~/Library/Application Support/Mference.
  --model <dir>          Preload this install instead of loading lazily.
  --server-port <port>   MferenceServer port (default 8080).
  --webui-port <port>    Open WebUI port (default 3000).
  --max-context <tokens> Context window for every model (default 16384).
  --prompt-cache-mode <off|single-prefix>
                         Passed through to MferenceServer.
  --dry-run              Print what would run, start nothing, exit 0. Works
                         for every subcommand, before or after it.
  --help                 Show this message.

install passes any extra arguments through to MferenceRepack, so a cancelled
download continues with:  ./mference-ui.sh install gemma4 --resume
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    install)
      command_name=install
      shift
      [[ $# -gt 0 ]] || fail "install needs a model family, for example: install gemma4"
      install_family="$1"
      shift
      # Everything left belongs to MferenceRepack, except --dry-run, which
      # stays a launcher flag wherever it is written.
      while [[ $# -gt 0 ]]; do
        if [[ "$1" == "--dry-run" ]]; then dry_run=1; else repack_arguments+=("$1"); fi
        shift
      done
      ;;
    models) command_name=models; shift ;;
    --library) library_roots+=("$2"); shift 2 ;;
    --model) preload_model="$2"; shift 2 ;;
    --server-port) server_port="$2"; shift 2 ;;
    --webui-port) webui_port="$2"; shift 2 ;;
    --max-context) max_context="$2"; shift 2 ;;
    --prompt-cache-mode) prompt_cache_mode="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) note "error: unknown option $1"; usage >&2; exit 2 ;;
  esac
done

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

# The same --library selection, listing instead of serving.
list_arguments=()
if [[ ${#library_roots[@]} -eq 0 ]]; then
  list_arguments+=(--library)
else
  for root in "${library_roots[@]}"; do
    list_arguments+=(--library "$root")
  done
fi
list_arguments+=(--list-models)

# Every one of these would otherwise fire an extra generation into a queue that
# runs one generation at a time: a title, tags, follow-ups, an autocomplete, and
# two query rewrites per turn. They seed Open WebUI's persisted defaults, so
# they take effect on a fresh DATA_DIR and are overridden by anything already
# toggled in its admin settings.
# Open WebUI signs sessions with WEBUI_SECRET_KEY. Left unset, `open-webui
# serve` generates one and writes it to `.webui_secret_key` in the *current
# directory* — the checkout — where it once got committed. Keep it in the data
# directory instead, created on first launch, readable only by the user.
webui_secret_key() {
  local file="$data_directory/webui-secret-key"
  if [[ ! -s "$file" ]]; then
    mkdir -p "$data_directory"
    (umask 077; openssl rand -hex 32 > "$file") || fail "could not create $file"
  fi
  cat "$file"
}

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
  "ENABLE_EVALUATION_ARENA_MODELS=false"
  "DATA_DIR=$data_directory"
  "WEBUI_SECRET_KEY=$(webui_secret_key)"
)

# --- checks and prerequisites ------------------------------------------------

check_no_model_process() {
  # AGENTS.md: one model-owning process at a time. Never terminate one that is
  # already running; refuse to start instead.
  local running
  running="$(pgrep -fl "$model_process_pattern" || true)"
  if [[ -n "$running" ]]; then
    note "error: another Mference model process is running; refusing to start."
    printf '%s\n' "$running" >&2
    exit 1
  fi
}

ensure_server_binary() {
  if [[ -x "$server_binary" ]]; then return 0; fi
  note "building MferenceServer (release)"
  (cd "$repository_root" && swift build -c release --product MferenceServer)
  [[ -x "$server_binary" ]] || fail "$server_binary is still missing after the build."
}

open_webui_binary() {
  local found
  found="$(command -v open-webui || true)"
  if [[ -z "$found" && -x "$HOME/.local/bin/open-webui" ]]; then
    found="$HOME/.local/bin/open-webui"
  fi
  printf '%s' "$found"
}

# Installed version, or empty when it cannot be determined. `uv tool list` is
# asked first because it names the package; the CLI's own --version is the
# fallback. An unknown version is never treated as a mismatch.
open_webui_version() {
  local binary="$1" reported=""
  if command -v uv >/dev/null 2>&1; then
    reported="$(uv tool list 2>/dev/null |
      sed -n 's/^open-webui v\([0-9][0-9.]*\).*/\1/p' | head -1)"
  fi
  if [[ -z "$reported" && -x "$binary" ]]; then
    reported="$("$binary" --version 2>/dev/null |
      sed -n 's/.*[^0-9.]\([0-9][0-9.]*\).*/\1/p' | head -1)"
  fi
  printf '%s' "$reported"
}

ensure_open_webui() {
  local binary
  binary="$(open_webui_binary)"
  if [[ -z "$binary" ]]; then
    if ! command -v uv >/dev/null 2>&1; then
      note "error: neither open-webui nor uv is installed."
      note "install uv with one of:"
      note "  brew install uv"
      note "  curl -LsSf https://astral.sh/uv/install.sh | sh"
      note "then re-run this script; it will install open-webui $OPEN_WEBUI_VERSION."
      exit 1
    fi
    note "installing open-webui $OPEN_WEBUI_VERSION with uv"
    uv tool install --python 3.11 "open-webui==${OPEN_WEBUI_VERSION}"
    binary="$(open_webui_binary)"
    [[ -n "$binary" ]] || fail "open-webui is still not on PATH after the install."
  fi
  local installed
  installed="$(open_webui_version "$binary")"
  if [[ -n "$installed" && "$installed" != "$OPEN_WEBUI_VERSION" ]]; then
    note "note: open-webui $installed is installed; this launcher pins $OPEN_WEBUI_VERSION (not reinstalling)."
  fi
  printf '%s' "$binary"
}

# The families MferenceRepack accepts, read out of its own help text rather
# than restated here. The source is preferred over running the binary: it is
# the same string, it is current in any checkout, and reading it starts no
# process for another agent's model-process check to trip over.
repack_families() {
  local help_text=""
  local usage_source="$repository_root/Sources/MferenceRepack/Command/main.swift"
  if [[ -r "$usage_source" ]]; then
    help_text="$(cat "$usage_source")"
  elif [[ -x "$repository_root/.build/release/MferenceRepack" ]]; then
    help_text="$("$repository_root/.build/release/MferenceRepack" --help 2>&1 || true)"
  fi
  sed -n 's/.*--model <\([A-Za-z0-9|]*\)>.*/\1/p' <<<"$help_text" | head -1 | tr '|' ' '
}

check_family() {
  local candidate="$1" known
  known="$(repack_families)"
  [[ -n "$known" ]] || fail "could not read MferenceRepack's supported model list."
  local family
  for family in $known; do
    [[ "$family" == "$candidate" ]] && return 0
  done
  note "error: unknown model family \"$candidate\"."
  note "supported: $known"
  exit 2
}

# --- subcommands -------------------------------------------------------------

cmd_install() {
  local output="scratch/$install_family.gturbo"
  # bash 3.2 is what /usr/bin/env bash is on macOS, and there `"${empty[@]}"`
  # is an unbound-variable error under `set -u`, so every pass-through
  # expansion is guarded by its own count.
  local extra=""
  if [[ ${#repack_arguments[@]} -gt 0 ]]; then extra=" ${repack_arguments[*]}"; fi
  if [[ "$dry_run" -eq 1 ]]; then
    echo "would check: pgrep -fl '$model_process_pattern'"
    echo "would verify: \"$install_family\" is one of: $(repack_families)"
    echo "would run:   swift run -c release MferenceRepack --model $install_family --output $output$extra"
    echo "would write: $repository_root/$output"
    exit 0
  fi
  check_family "$install_family"
  check_no_model_process
  note "installing $install_family into $repository_root/$output"
  note "downloads range from ~15 GB (gemma4) to ~148 GB (inklingsmall); check disk first"
  cd "$repository_root"
  if [[ ${#repack_arguments[@]} -gt 0 ]]; then
    swift run -c release MferenceRepack \
      --model "$install_family" --output "$output" "${repack_arguments[@]}"
  else
    swift run -c release MferenceRepack --model "$install_family" --output "$output"
  fi
  note "installed; start the UI with ./mference-ui.sh"
}

cmd_models() {
  if [[ "$dry_run" -eq 1 ]]; then
    echo "would check: pgrep -fl '$model_process_pattern'"
    echo "would build: swift build -c release --product MferenceServer (if missing)"
    echo "would run:   $server_binary ${list_arguments[*]}"
    exit 0
  fi
  # Listing loads no model and binds no port, but it is still an MferenceServer
  # process, so it waits its turn like everything else here.
  check_no_model_process
  ensure_server_binary
  local listed
  listed="$("$server_binary" "${list_arguments[@]}")"
  printf '%s\n' "$listed"
  if [[ "$listed" == "no models installed" ]]; then
    note "install one with: ./mference-ui.sh install gemma4"
  fi
}

cmd_run() {
  if [[ "$dry_run" -eq 1 ]]; then
    echo "would check: pgrep -fl '$model_process_pattern'"
    echo "would ensure: open-webui $OPEN_WEBUI_VERSION (uv tool install --python 3.11 open-webui==$OPEN_WEBUI_VERSION)"
    echo "would build: swift build -c release --product MferenceServer (if missing)"
    echo "would run:   $server_binary ${server_arguments[*]}"
    echo "would wait:  http://127.0.0.1:$server_port/health"
    echo "would read:  http://127.0.0.1:$server_port/v1/models"
    echo "would run:   $(open_webui_binary) serve --host 127.0.0.1 --port $webui_port"
    echo "would run:   Scripts/openwebui-configure-models.py --webui http://127.0.0.1:$webui_port  (builtin tools off per Mference model)"
    echo "with env:"
    local entry
    for entry in "${webui_environment[@]}"; do
      case "$entry" in WEBUI_SECRET_KEY=*) echo "  WEBUI_SECRET_KEY=<from $data_directory/webui-secret-key>" ;; *) echo "  $entry" ;; esac
    done
    echo "would wait:  http://127.0.0.1:$webui_port"
    echo "would open:  http://127.0.0.1:$webui_port"
    exit 0
  fi

  check_no_model_process
  local webui_binary
  webui_binary="$(ensure_open_webui)"
  ensure_server_binary

  trap stop_children INT TERM EXIT
  mkdir -p "$data_directory"

  note "starting MferenceServer on 127.0.0.1:$server_port (library mode)"
  "$server_binary" "${server_arguments[@]}" &
  server_pid=$!
  wait_for_server

  note "starting Open WebUI on http://127.0.0.1:$webui_port"
  env "${webui_environment[@]}" \
    "$webui_binary" serve --host 127.0.0.1 --port "$webui_port" &
  webui_pid=$!
  wait_for_webui
  configure_models
  open "http://127.0.0.1:$webui_port" || note "open the UI yourself: http://127.0.0.1:$webui_port"

  note "Control-C stops both."
  # Exit when either child does, then stop the other through the trap. `wait -n`
  # would say this in one line but arrived in bash 4.3, and `/usr/bin/env bash`
  # on macOS is 3.2.
  while kill -0 "$server_pid" 2>/dev/null && kill -0 "$webui_pid" 2>/dev/null; do
    sleep 1
  done
  note "a child exited; stopping the other."
}

# --- process lifecycle -------------------------------------------------------

server_pid=""
webui_pid=""

stop_children() {
  trap - INT TERM EXIT
  # Only the two processes this script started.
  local pid
  for pid in "$webui_pid" "$server_pid"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}

configure_models() {
  # Open WebUI 0.11 attaches its builtin tool schemas to every request by
  # default (native function calling), which the server must render into the
  # prompt: a 27-token question became 5,445 tokens and a 45-second prefill.
  # There is no environment variable for it, only a per-model capability, so
  # register every Mference model with builtin tools off on every launch —
  # idempotent, and it picks up newly installed models. Non-fatal: chat works
  # without it, just slowly, and the script says what failed.
  local python="$(dirname "$webui_binary")/python"
  [[ -x "$python" ]] || python="python3"
  if ! "$python" "$repository_root/Scripts/openwebui-configure-models.py" \
      --webui "http://127.0.0.1:$webui_port"; then
    note "warning: could not switch builtin tools off for the Mference models;"
    note "         answers will be slow until you uncheck 'Builtin Tools' per model in Admin > Models."
  fi
}

wait_for_server() {
  # Library mode opens the port before loading anything, so /health answers
  # quickly. A preload with --model has to verify the install first, which is
  # why the wait is generous.
  local health_url="http://127.0.0.1:$server_port/health"
  local _
  for _ in $(seq 1 600); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      fail "MferenceServer exited before it became healthy."
    fi
    if curl --silent --show-error --fail --max-time 2 "$health_url" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  curl --silent --show-error --fail --max-time 2 "$health_url" >/dev/null 2>&1 ||
    fail "MferenceServer did not answer $health_url in time."

  local models
  models="$(curl --silent --max-time 5 "http://127.0.0.1:$server_port/v1/models" || true)"
  if [[ "$models" == *'"data":[]'* || "$models" == *'"data": []'* ]]; then
    note "no models installed — the picker will be empty."
    note "install one with: ./mference-ui.sh install gemma4"
  else
    note "serving: $models"
  fi
}

wait_for_webui() {
  local url="http://127.0.0.1:$webui_port/"
  local _
  for _ in $(seq 1 180); do
    if ! kill -0 "$webui_pid" 2>/dev/null; then
      fail "Open WebUI exited before it answered $url."
    fi
    # Any HTTP response means it is serving; --fail is deliberately absent so a
    # redirect or a 404 on / still counts.
    if curl --silent --output /dev/null --max-time 2 "$url"; then
      return 0
    fi
    sleep 1
  done
  fail "Open WebUI did not answer $url in time."
}

case "$command_name" in
  install) cmd_install ;;
  models) cmd_models ;;
  run) cmd_run ;;
esac
