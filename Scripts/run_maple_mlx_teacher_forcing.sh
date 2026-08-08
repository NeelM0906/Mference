#!/usr/bin/env bash
# Select the pinned Python environment; the exporter runs its own safety gate.

set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python="${MAPLE_MLX_PYTHON:-python3}"
arguments=()
while (($#)); do
  case "$1" in
    --python) python="${2:?error: --python needs a value}"; shift 2 ;;
    --help)
      echo "usage: $0 [--python <python>] --model <local-snapshot> --output <trace.jsonl> [oracle options]"
      exec "$python" "$script_directory/maple_mlx_teacher_forcing.py" --help
      ;;
    *) arguments+=("$1"); shift ;;
  esac
done
exec "$python" "$script_directory/maple_mlx_teacher_forcing.py" "${arguments[@]}"
