#!/usr/bin/env bash
# Build and bundle Mference.app (App/Scripts/bundle.sh), then open it.
# Arguments pass through to bundle.sh (for example --no-build).
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$APP_DIR/Scripts/bundle.sh" "$@"
open "$APP_DIR/build/Mference.app"
