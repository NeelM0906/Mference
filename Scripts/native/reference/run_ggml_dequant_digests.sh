#!/bin/bash
# Build and run ggml_dequant_digests.c against the vendored ggml-quants.c.
# Prints the digests ReferenceDequantTests pins, then the git revision of the
# ggml sources it compiled. No model, no download.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
root="$(cd "$here/../../.." && pwd -P)"
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
clang -O1 -ffp-contract=off -w \
  -I "$root/Sources/LlamaCpp/include" -I "$root/Sources/LlamaCpp/ggml/src" \
  "$here/ggml_dequant_digests.c" "$root/Sources/LlamaCpp/ggml/src/ggml-quants.c" \
  -Wl,-undefined,dynamic_lookup -o "$out/digests"
"$out/digests"
echo "ggml-quants.c $(git -C "$root" rev-parse HEAD) $(shasum -a 256 "$root/Sources/LlamaCpp/ggml/src/ggml-quants.c" | cut -d" " -f1)"
for f in "$here/ggml_dequant_digests.c" "$root/Sources/LlamaCpp/ggml/src/ggml-impl.h" "$root/Sources/LlamaCpp/ggml/src/ggml-common.h"; do
  echo "$(basename "$f") $(shasum -a 256 "$f" | cut -d" " -f1)"
done
