# paths.sh: shared locations for the Inkling reference scripts, sourced by them.
# Found from this checkout (a git worktree resolves to its main checkout, where
# the gitignored scratch/ lives); each can be overridden from the environment.
_ink_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_ink_root=$(cd "$(git -C "$_ink_here" rev-parse --path-format=absolute --git-common-dir)/.." && pwd)
INK=${INK:-$_ink_root/scratch/inkling-program}
LOCK=${LOCK:-$_ink_root/scratch/program/lock.sh}
INKLING_GGUF=${INKLING_GGUF:-$INK/ref/unsloth-ud-q4_k_xl/Inkling-2-Mini-UD-Q4_K_XL.gguf}
