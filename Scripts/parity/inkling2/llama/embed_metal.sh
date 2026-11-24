#!/bin/sh
# embed_metal.sh: one embedded Metal library source, as upstream's
# ggml/src/ggml-metal/CMakeLists.txt (GGML_METAL_EMBED_LIBRARY, the default on
# Apple) generates it, for the cmake-free reference build (Makefile here).
#
#   embed_metal.sh <llama.cpp checkout> <kind> <out-dir>
#
# Writes <out-dir>/ggml-metal-embed-<kind>.metal (the kernel source with its
# headers inlined) and <out-dir>/ggml-metal-embed-<kind>.s (an .incbin of it
# between _ggml_metallib_<kind>_start and _ggml_metallib_<kind>_end).
set -eu
LL=$1; kind=$2; out=$3
M=$LL/ggml/src/ggml-metal
src=$M/kernels/$kind.metal
embed=$out/ggml-metal-embed-$kind.metal
asm=$out/ggml-metal-embed-$kind.s
sym=$(echo "$kind" | tr '-' '_')
mkdir -p "$out"

headers=$M/kernels/common.h
grep -q '#include "dequantize\.h"' "$src" && headers="$headers $M/kernels/dequantize.h"
grep -q '#include "quantize\.h"' "$src" && headers="$headers $M/kernels/quantize.h"
grep -q '#include "fa_common\.metal"' "$src" && headers="$headers $M/kernels/fa_common.metal"
grep -q '#include "fa_vec_common\.metal"' "$src" && headers="$headers $M/kernels/fa_vec_common.metal"

# shellcheck disable=SC2086
cat $headers "$src" > "$embed.tmp1"
sed -e '/#include "common.h"/d' -e '/#include "dequantize.h"/d' -e '/#include "quantize.h"/d' \
    -e '/#include "fa_common.metal"/d' -e '/#include "fa_vec_common.metal"/d' -e '/#pragma once/d' \
    < "$embed.tmp1" > "$embed.tmp2"
sed -e "/__embed_ggml-common.h__/r $LL/ggml/src/ggml-common.h" -e '/__embed_ggml-common.h__/d' \
    < "$embed.tmp2" > "$embed.tmp3"
sed -e "/#include \"ggml-metal-impl.h\"/r $M/ggml-metal-impl.h" -e '/#include "ggml-metal-impl.h"/d' \
    < "$embed.tmp3" > "$embed"
rm -f "$embed.tmp1" "$embed.tmp2" "$embed.tmp3"

{
  echo ".section __DATA,__ggml_metallib"
  echo ".globl _ggml_metallib_${sym}_start"
  echo "_ggml_metallib_${sym}_start:"
  echo ".incbin \"$embed\""
  echo ".globl _ggml_metallib_${sym}_end"
  echo "_ggml_metallib_${sym}_end:"
} > "$asm"
