# Vendored llama.cpp

`Sources/LlamaCpp` builds a subset of [llama.cpp](https://github.com/ggml-org/llama.cpp)
in-process for the GGUF models of `MferenceEngine`, with the Metal backend for
full GPU offload. Nothing in this directory (`vendoring/`) is compiled.

## Pin

| | |
|---|---|
| Upstream | `ggml-org/llama.cpp` |
| Commit | `b1b770d90d94cad9e9ea5b993bdbc383a522fb28` (2026-11-13, "examples : remove explict version from test-cmake (#32191)") |
| Why this commit | The commit Mference already pins for GGUF block layouts (see `THIRD_PARTY_NOTICES.md`); it has `LLM_ARCH_QWEN35`, the nextn (MTP) tensors and `common/speculative.cpp`'s `draft-mtp`. |

Moving the pin needs a recorded reason in the table below and a fresh G6
token-identity run for every GGUF product model.

| Date | From → to | Reason |
|---|---|---|
| 2026-11-18 | — → `b1b770d9` | initial vendoring |

## What is here

Every file listed in [`UPSTREAM.sha256`](UPSTREAM.sha256) is byte-identical to
upstream at the pin; the third column is its upstream path. Paths are kept,
except that public headers from `include/` and `ggml/include/` are gathered
into `include/`, SwiftPM's public-headers directory. Three files are generated
from upstream's own `*.in` templates (version `1.0.0`, commit `b1b770d9`,
build 0) and are marked `generated` in the manifest.

The multimodal library (`tools/mtmd`, its public `mtmd.h` gathered into
`include/`) is vendored without its helpers (`mtmd-helper*`): its callers
decode images and audio themselves (Find uses Apple's frameworks) and hand
over pixels and samples, so the helpers' decoders (stb_image, miniaudio,
ffmpeg) are not needed. Every model graph upstream's `mtmd` target builds is
included, because `mtmd.cpp` references them all. `mtmd-audio.cpp` declares a
`constexpr bool DEBUG`, which collides with the `DEBUG=1` SwiftPM defines in
debug builds, so the target's C++ flags add `-UDEBUG`.

Files that are ours, not upstream's:

- `include/module.modulemap`: exports the C headers to Swift as `LlamaCpp`.
- `include/mference_mtp.h`, `shim/mference_mtp.cpp`: a C API over llama.cpp's
  own `common_speculative` (`draft-mtp`) for MTP speculative decoding, whose
  step follows upstream's `examples/speculative-simple` at the pin.
- `vendoring/`: this note, `vendor.sh`, the manifest, and copies of the
  llama.cpp (MIT, identical to the repository's `LICENSE-GGML`) and
  nlohmann/json (MIT) licences.

The Metal kernel sources (`ggml/src/ggml-metal/kernels`) and the two headers
they include from outside that directory (`ggml-metal-impl.h`,
`ggml-common.h`) are SwiftPM resources; ggml finds them through
`SWIFTPM_MODULE_BUNDLE` and compiles them when the Metal backend first
initialises.

The design decisions (vendoring rather than a prebuilt framework, Metal kernels
as a resource, compile settings, CI cost) are recorded in the engine program's
T2 build note.

## Re-vendoring and checking

From the repository root, with a llama.cpp checkout at the pin:

```bash
Sources/LlamaCpp/vendoring/vendor.sh /path/to/llama.cpp --check
```

prints `vendored llama.cpp matches …` and exits 0 when every vendored file
equals upstream. Without `--check` it rewrites the vendored files and the
manifest (and leaves our own files alone). The script refuses a checkout at any
other commit.
