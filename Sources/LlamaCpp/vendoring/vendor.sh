#!/usr/bin/env bash
# Vendors the subset of llama.cpp that the LlamaCpp target builds, from a
# checkout at the pinned commit, into Sources/LlamaCpp. Every copied file is
# byte-identical to upstream; only its location changes (public headers from
# include/ and ggml/include/ are gathered into include/). Three files are
# generated from upstream's own *.in templates with the values the reference
# Makefile build uses. UPSTREAM.sha256 records every vendored file.
#
#   Sources/LlamaCpp/vendoring/vendor.sh <llama.cpp checkout>          # write
#   Sources/LlamaCpp/vendoring/vendor.sh <llama.cpp checkout> --check  # verify
#
# --check rebuilds the tree in a temporary directory and compares it with the
# vendored files and the manifest; it changes nothing. Moving the pin needs a
# recorded reason in VENDORING.md.
set -euo pipefail

PIN=b1b770d90d94cad9e9ea5b993bdbc383a522fb28
SHORT=b1b770d9
VERSION=1.0.0

src=${1:?usage: vendor.sh <llama.cpp checkout> [--check]}
mode=${2:-write}
case "$mode" in
  write|--check) ;;
  *) echo "usage: vendor.sh <llama.cpp checkout> [--check]" >&2; exit 2 ;;
esac
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
target=$(cd -- "$here/.." && pwd)

head=$(git -C "$src" rev-parse HEAD)
if [[ "$head" != "$PIN" ]]; then
  echo "error: $src is at $head, expected $PIN" >&2
  exit 1
fi
# Files are read from the working tree, so it must be the commit itself.
if ! git -C "$src" diff --quiet HEAD --; then
  echo "error: $src has uncommitted changes" >&2
  exit 1
fi

# Our own files in the target directory: everything else must be in the
# manifest.
ours=(
  include/module.modulemap include/mference_mtp.h shim/mference_mtp.cpp
  vendoring/vendor.sh vendoring/VENDORING.md vendoring/UPSTREAM.sha256
  vendoring/LICENSE-llama.cpp vendoring/LICENSE-nlohmann-json
)

# Upstream paths, relative to the checkout. Destination is the same path,
# except public headers, which go to include/.
public_headers=(
  include/llama.h include/llama-cpp.h
  ggml/include/ggml.h ggml/include/ggml-alloc.h ggml/include/ggml-backend.h
  ggml/include/ggml-cpp.h ggml/include/ggml-cpu.h ggml/include/ggml-metal.h
  ggml/include/ggml-opt.h ggml/include/gguf.h
)

ggml_core=(
  ggml/src/ggml.c ggml/src/ggml-alloc.c ggml/src/ggml-quants.c
  ggml/src/ggml.cpp ggml/src/ggml-backend.cpp ggml/src/ggml-backend-meta.cpp
  ggml/src/ggml-opt.cpp ggml/src/ggml-threading.cpp ggml/src/gguf.cpp
  ggml/src/ggml-backend-reg.cpp ggml/src/ggml-backend-dl.cpp
  ggml/src/ggml-backend-dl.h ggml/src/ggml-backend-impl.h ggml/src/ggml-common.h
  ggml/src/ggml-feats.h ggml/src/ggml-impl.h ggml/src/ggml-quants.h
  ggml/src/ggml-threading.h
)

# The CPU backend for Apple arm64, as the reference Makefile builds it
# (llamafile sgemm, CPU repack, Accelerate). Other architectures are left out.
ggml_cpu=(
  ggml/src/ggml-cpu/ggml-cpu.c ggml/src/ggml-cpu/quants.c
  ggml/src/ggml-cpu/arch/arm/quants.c ggml/src/ggml-cpu/arch/arm/repack.cpp
  ggml/src/ggml-cpu/ggml-cpu.cpp ggml/src/ggml-cpu/repack.cpp ggml/src/ggml-cpu/hbm.cpp
  ggml/src/ggml-cpu/traits.cpp ggml/src/ggml-cpu/binary-ops.cpp ggml/src/ggml-cpu/unary-ops.cpp
  ggml/src/ggml-cpu/vec.cpp ggml/src/ggml-cpu/ops.cpp
  ggml/src/ggml-cpu/amx/amx.cpp ggml/src/ggml-cpu/amx/mmq.cpp
  ggml/src/ggml-cpu/amx/amx.h ggml/src/ggml-cpu/amx/common.h ggml/src/ggml-cpu/amx/mmq.h
  ggml/src/ggml-cpu/tiled/tiled.cpp ggml/src/ggml-cpu/tiled/tiled-kernel.cpp
  ggml/src/ggml-cpu/tiled/tiled.h ggml/src/ggml-cpu/tiled/tiled-kernel.h
  ggml/src/ggml-cpu/llamafile/sgemm.cpp ggml/src/ggml-cpu/llamafile/sgemm.h
  ggml/src/ggml-cpu/arch-fallback.h ggml/src/ggml-cpu/binary-ops.h ggml/src/ggml-cpu/common.h
  ggml/src/ggml-cpu/ggml-cpu-impl.h ggml/src/ggml-cpu/hbm.h ggml/src/ggml-cpu/ops.h
  ggml/src/ggml-cpu/quants.h ggml/src/ggml-cpu/repack.h ggml/src/ggml-cpu/simd-gemm.h
  ggml/src/ggml-cpu/simd-mappings.h ggml/src/ggml-cpu/traits.h ggml/src/ggml-cpu/unary-ops.h
  ggml/src/ggml-cpu/vec.h
)

# The Metal backend. kernels/ is not compiled into the binary: it ships as a
# resource and ggml compiles it at device init (the SWIFT_PACKAGE path).
ggml_metal=(
  ggml/src/ggml-metal/ggml-metal.cpp ggml/src/ggml-metal/ggml-metal-device.cpp
  ggml/src/ggml-metal/ggml-metal-common.cpp ggml/src/ggml-metal/ggml-metal-fusion.cpp
  ggml/src/ggml-metal/ggml-metal-ops.cpp ggml/src/ggml-metal/ggml-metal-tuning.cpp
  ggml/src/ggml-metal/ggml-metal-device.m ggml/src/ggml-metal/ggml-metal-context.m
  ggml/src/ggml-metal/ggml-metal-common.h ggml/src/ggml-metal/ggml-metal-context.h
  ggml/src/ggml-metal/ggml-metal-device.h ggml/src/ggml-metal/ggml-metal-fusion.h
  ggml/src/ggml-metal/ggml-metal-impl.h ggml/src/ggml-metal/ggml-metal-ops.h
  ggml/src/ggml-metal/ggml-metal-tuning.h
)
while IFS= read -r f; do ggml_metal+=("$f"); done < <(cd "$src" && find ggml/src/ggml-metal/kernels -type f \( -name '*.metal' -o -name '*.h' \) | LC_ALL=C sort)

# libllama: every source, including every architecture graph builder, because
# llama-model.cpp references them all.
llama=()
while IFS= read -r f; do llama+=("$f"); done < <(cd "$src" && find src -maxdepth 2 -type f \( -name '*.cpp' -o -name '*.h' \) | LC_ALL=C sort)

# The part of common/ that speculative decoding (MTP) links against.
common=(
  common/common.cpp common/common.h common/speculative.cpp common/speculative.h
  common/sampling.cpp common/sampling.h common/log.cpp common/log.h
  common/ngram-cache.cpp common/ngram-cache.h common/ngram-map.cpp common/ngram-map.h
  common/ngram-mod.cpp common/ngram-mod.h common/reasoning-budget.cpp common/reasoning-budget.h
  common/trie.cpp common/trie.h common/unicode.cpp common/unicode.h
  common/fit.cpp common/fit.h common/json.cpp common/json.h common/build-info.h
  vendor/nlohmann/json.hpp vendor/nlohmann/json_fwd.hpp
)

dest_of() {
  case "$1" in
    include/*) echo "include/${1#include/}" ;;
    ggml/include/*) echo "include/${1#ggml/include/}" ;;
    *) echo "$1" ;;
  esac
}

build_tree() {
  local out=$1 f d
  for f in "${public_headers[@]}" "${ggml_core[@]}" "${ggml_cpu[@]}" "${ggml_metal[@]}" "${llama[@]}" "${common[@]}"; do
    d=$(dest_of "$f")
    mkdir -p "$out/$(dirname "$d")"
    cp "$src/$f" "$out/$d"
    printf '%s  %s  %s\n' "$(shasum -a 256 "$src/$f" | cut -d' ' -f1)" "$d" "$f" >> "$out/vendoring/UPSTREAM.sha256.tmp"
  done
  sed -e "s/@GGML_VERSION@/$VERSION/" -e "s/@GGML_BUILD_COMMIT@/$SHORT/" \
    "$src/ggml/src/ggml-version.h.in" > "$out/ggml/src/ggml-version.h"
  sed -e "s/@LLAMA_VERSION@/$VERSION/" -e "s/@LLAMA_BUILD_COMMIT@/$SHORT/" \
    "$src/src/llama-version.h.in" > "$out/src/llama-version.h"
  sed -e 's/@LLAMA_BUILD_NUMBER@/0/' -e "s/@LLAMA_BUILD_COMMIT@/$SHORT/" \
    -e 's/@BUILD_COMPILER@/clang/' -e 's/@BUILD_TARGET@/arm64-apple-darwin-metal/' \
    "$src/common/build-info.cpp.in" > "$out/common/build-info.cpp"
  {
    echo "# llama.cpp $PIN, vendored by vendoring/vendor.sh."
    echo "# sha256  vendored-path  upstream-path"
    LC_ALL=C sort -k2,2 "$out/vendoring/UPSTREAM.sha256.tmp"
    echo "# generated from upstream *.in templates (version $VERSION, commit $SHORT, build 0)"
    for g in ggml/src/ggml-version.h src/llama-version.h common/build-info.cpp; do
      printf '%s  %s  generated\n' "$(shasum -a 256 "$out/$g" | cut -d' ' -f1)" "$g"
    done
  } > "$out/vendoring/UPSTREAM.sha256"
  rm "$out/vendoring/UPSTREAM.sha256.tmp"
}

# Every vendored path, from a manifest.
manifest_paths() { grep -v '^#' "$1" | awk '{print $2}'; }

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/vendoring"
build_tree "$tmp"

if [[ "$mode" == "--check" ]]; then
  status=0
  cmp -s "$tmp/vendoring/UPSTREAM.sha256" "$here/UPSTREAM.sha256" || { echo "manifest differs" >&2; status=1; }
  while IFS= read -r p; do
    cmp -s "$tmp/$p" "$target/$p" || { echo "differs from upstream: $p" >&2; status=1; }
  done < <(manifest_paths "$tmp/vendoring/UPSTREAM.sha256")
  cmp -s "$src/LICENSE" "$here/LICENSE-llama.cpp" || { echo "differs from upstream: vendoring/LICENSE-llama.cpp" >&2; status=1; }
  cmp -s "$src/licenses/LICENSE-jsonhpp" "$here/LICENSE-nlohmann-json" || { echo "differs from upstream: vendoring/LICENSE-nlohmann-json" >&2; status=1; }
  # Anything else under the target would be compiled by SwiftPM unnoticed.
  while IFS= read -r p; do
    echo "not in the manifest: $p" >&2; status=1
  done < <(comm -23 <(cd "$target" && find . -type f ! -name .DS_Store | sed 's|^\./||' | LC_ALL=C sort) \
                    <({ manifest_paths "$tmp/vendoring/UPSTREAM.sha256"; printf '%s\n' "${ours[@]}"; } | LC_ALL=C sort))
  [[ $status -eq 0 ]] && echo "vendored llama.cpp matches $PIN ($(manifest_paths "$tmp/vendoring/UPSTREAM.sha256" | wc -l | tr -d ' ') files)"
  exit $status
fi

# Write: replace only the files in the old manifest; the files in $ours keep
# their content (the manifest and licence copies are refreshed below).
if [[ -f "$here/UPSTREAM.sha256" ]]; then
  while IFS= read -r p; do rm -f "$target/$p"; done < <(manifest_paths "$here/UPSTREAM.sha256")
fi
while IFS= read -r p; do
  mkdir -p "$target/$(dirname "$p")"
  cp "$tmp/$p" "$target/$p"
done < <(manifest_paths "$tmp/vendoring/UPSTREAM.sha256")
cp "$tmp/vendoring/UPSTREAM.sha256" "$here/UPSTREAM.sha256"
cp "$src/LICENSE" "$here/LICENSE-llama.cpp"
cp "$src/licenses/LICENSE-jsonhpp" "$here/LICENSE-nlohmann-json"
echo "vendored $(manifest_paths "$here/UPSTREAM.sha256" | wc -l | tr -d ' ') files from llama.cpp $PIN"
