#!/usr/bin/env bash
# Build the parity executable from the current clean checkout, then run it.

set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository="$(cd -- "$script_directory/.." && pwd)"

if [[ -n "$(git -C "$repository" status --porcelain --untracked-files=all)" ]]; then
  echo "error: the Mference checkout must be clean before a parity build" >&2
  exit 1
fi

swift build --package-path "$repository" -c release --product MferenceMapleParity
binary_directory="$(swift build --package-path "$repository" -c release --show-bin-path)"
exec "$binary_directory/MferenceMapleParity" "$@"
