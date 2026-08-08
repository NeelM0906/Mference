#!/usr/bin/env bash
# Non-destructive safety gate for a local Maple MLX oracle run.

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <complete-local-maple-mlx-snapshot> [allowed-exporter-pid]" >&2
  exit 2
fi

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
snapshot="$1"
allowed_pid="${2:-}"
if [[ -n "$allowed_pid" && ! "$allowed_pid" =~ ^[0-9]+$ ]]; then
  echo "error: allowed exporter PID must be numeric" >&2
  exit 2
fi
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo "error: Maple model runs require Apple Silicon macOS" >&2
  exit 1
fi
macos_version="$(sw_vers -productVersion)"
macos_major="${macos_version%%.*}"
if (( macos_major < 15 )); then
  echo "error: macOS 15+ is required; found $macos_version" >&2
  exit 1
fi
swift_line="$(swift --version | head -n 1)"
if [[ ! "$swift_line" =~ Apple\ Swift\ version\ ([0-9]+)\.([0-9]+) ]]; then
  echo "error: could not determine Swift version: $swift_line" >&2
  exit 1
fi
swift_major="${BASH_REMATCH[1]}"
swift_minor="${BASH_REMATCH[2]}"
if (( swift_major < 6 || (swift_major == 6 && swift_minor < 1) )); then
  echo "error: Swift 6.1+ is required; found $swift_line" >&2
  exit 1
fi
pressure="$(memory_pressure -Q 2>&1)"
if [[ ! "$pressure" =~ memory\ free\ percentage:\ ([0-9]+)% ]] || (( BASH_REMATCH[1] < 10 )); then
  echo "error: memory pressure requires at least 10% free" >&2
  printf '%s\n' "$pressure" >&2
  exit 1
fi
free_percent="${BASH_REMATCH[1]}"
pattern='MferenceServer|MferenceMac|MferenceDecodeService|MferenceCLI|MferencePackageTests|MferenceMapleParity|swiftpm-testing-helper|mlx_lm|mlx-lm|maple_mlx_teacher_forcing|run_maple_mlx_teacher_forcing|maple_oracle_preflight'
running=""
while IFS= read -r process; do
  [[ -n "$process" ]] || continue
  pid="${process%% *}"
  if [[ "$pid" == "$$" || -n "$allowed_pid" && "$pid" == "$allowed_pid" ]]; then
    continue
  fi
  running+="$process"$'\n'
done < <(pgrep -fl "$pattern" || true)
if [[ -n "$running" ]]; then
  echo "error: another model or model-using test process is running:" >&2
  printf '%s' "$running" >&2
  exit 1
fi
if [[ ! -d "$snapshot" ]]; then
  echo "error: local snapshot is missing: $snapshot" >&2
  exit 1
fi
free_kib="$(df -Pk "$snapshot" | awk 'NR == 2 {print $4}')"
if [[ ! "$free_kib" =~ ^[0-9]+$ ]] || (( free_kib < 1048576 )); then
  echo "error: at least 1 GiB free disk is required" >&2
  exit 1
fi
python3 "$script_directory/verify_maple_snapshot.py" "$snapshot" --quiet
echo "preflight ok: macOS=$macos_version swift=$swift_major.$swift_minor memory_free=${free_percent}% free_disk_kib=$free_kib" >&2
