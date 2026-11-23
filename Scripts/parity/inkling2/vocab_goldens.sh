#!/bin/bash
# vocab_goldens.sh: the goldens that load no weights (header cross-check,
# tokenizer ids, chat-template renders), from the K0 reference build.
#   vocab_goldens.sh <build-dir> <goldens-dir>
set -euo pipefail
B=$1; G=$2
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/paths.sh"; GGUF=$INKLING_GGUF
mkdir -p "$G/tokenizer" "$G/chat"
# Not here: `llama-gguf <file> r n` (the header cross-check in TENSORS.md) also
# allocates and reads every tensor into heap memory (examples/gguf/gguf.cpp,
# gguf_ex_read_1, no_alloc = false): 17.6 GB on a 32 GB host. It caused the
# 2026-11-23 swap-guard trip and must not be run on this file again.
# every command that opens the model file goes through the swap guard
"$HERE/guarded_run.sh" "$G/tokenizer/log" "$GGUF" -- "$B/ink-golden" tokenize --model "$GGUF" --in "$HERE/inputs/tokenizer_texts.json" --out "$G/tokenizer/tokenizer_goldens.json"
"$HERE/guarded_run.sh" "$G/chat/log" "$GGUF" -- "$B/ink-golden" chat --model "$GGUF" --in "$HERE/inputs/chat_cases.json" --out "$G/chat/chat_goldens.json"
echo "vocab goldens done"
