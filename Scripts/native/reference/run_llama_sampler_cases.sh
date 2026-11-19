#!/bin/bash
# Build llama_sampler_cases.cpp and link it with the LlamaCpp objects of an
# existing `swift build -c release` (llama.cpp's sampler needs ggml), then
# print the cases ReferenceSamplerTests pins. No model, no download; builds
# nothing in the package.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
root="$(cd "$here/../../.." && pwd -P)"
objects="$root/.build/out/Intermediates.noindex/Mference.build/Release/LlamaCpp-t.build"
[ -d "$objects" ] || { echo "run swift build -c release first (no $objects)" >&2; exit 2; }
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
clang++ -std=c++17 -O1 -w -I "$root/Sources/LlamaCpp/include" -c "$here/llama_sampler_cases.cpp" -o "$out/cases.o"
find "$objects" -name '*.o' > "$out/objects.txt"
clang++ "$out/cases.o" $(cat "$out/objects.txt") -framework Metal -framework Foundation -framework Accelerate \
  -framework MetalKit -o "$out/cases" 2>/dev/null
"$out/cases"
echo "    // git $(git -C "$root" rev-parse HEAD); llama-sampler.cpp $(shasum -a 256 "$root/Sources/LlamaCpp/src/llama-sampler.cpp" | cut -d' ' -f1); generator $(shasum -a 256 "$here/llama_sampler_cases.cpp" | cut -d' ' -f1)"
