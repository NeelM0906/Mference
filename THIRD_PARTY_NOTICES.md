# Third-party software and model terms

Mference is licensed under the [MIT License](LICENSE). It does not relicense
model weights or third-party packages.

## TurboFieldfare

Mference's runtime, installer, and application code are derived from
[TurboFieldfare](https://github.com/drumih/turbo-fieldfare) by Andrey
Mikhaylov and contributors, licensed under the
[Apache License 2.0](LICENSE-APACHE). The derived portions remain governed by
that license. Significant changes from the original include: the Mference
rebrand throughout; support for Qwen 3.6, Qwen 3.8 27B, Maple, and MiniCPM5-2B
across architecture configuration,
streaming installation, Metal kernels, forward runners, tokenization, and
products; model-derived kernel specialization; and expanded validation and
performance documentation.

This file records the dependency review performed on 2026-08-09. It is an
attribution aid, not legal advice. Anyone distributing a compiled product must
also preserve the license and NOTICE material required by the exact dependency
versions included in that product.

## Apple MLX and DeepGrove MLX-LM

Maple's shape-specific Metal arithmetic, including vector SDPA, was adapted
from Apple MLX 0.32.0 and the Maple model implementation in
[`deepgrove-ai/mlx-lm-deepgrove`](https://github.com/deepgrove-ai/mlx-lm-deepgrove)
at revision `eba96c16158f032821b0bf374ea1421cfddef0a9`. Both upstream trees
carry the MIT license. The applicable Apple and DeepGrove copyright notices
are preserved in [`LICENSE-MLX`](LICENSE-MLX).

Gemma QAT's router reduction in `Sources/Mference/Metal/MoE/moe.metal` and
`Sources/Mference/Metal/Prefill/prefill.metal` adapts MLX 0.32.2's `gemv.h`
reduction order to Mference's dispatch. These portions also use the MIT
license; see [`LICENSE-MLX`](LICENSE-MLX).

The QAT affine projection helper in those same files and in
`Sources/Mference/Metal/TensorCore/tensorops.metal` follows MLX 0.32.2's
`quantized.h` FP16 input summation and SIMD accumulation order. These portions
also use the MIT license; see [`LICENSE-MLX`](LICENSE-MLX).

## ggml (llama.cpp)

The `LlamaCpp` target (`Sources/LlamaCpp`) vendors source files of llama.cpp
at commit `b1b770d90d94cad9e9ea5b993bdbc383a522fb28`, byte-identical to upstream and relocated only (public
headers gathered into `include/`): ggml's core, CPU (Apple arm64) and Metal
backends with the Metal kernel sources, libllama, and the part of `common/`
that speculative decoding links against. `Sources/LlamaCpp/vendoring/vendor.sh`
copies them and verifies them against `vendoring/UPSTREAM.sha256`; three
version and build-info files are generated from upstream's templates. llama.cpp
is under the same MIT license as ggml, with the ggml authors' notice preserved
in [`LICENSE-GGML`](LICENSE-GGML) and in
[`Sources/LlamaCpp/vendoring/LICENSE-llama.cpp`](Sources/LlamaCpp/vendoring/LICENSE-llama.cpp).
Individual vendored files carry further notices of their own, kept unchanged
in the files: for example `ggml/src/ggml-cpu/llamafile/sgemm.cpp` (Mozilla
Foundation, MIT) and the YaRN RoPE code (Jeffrey Quesnelle and Bowen Peng,
MIT).

## nlohmann/json

`Sources/LlamaCpp/vendor/nlohmann/json.hpp` and `json_fwd.hpp` are JSON for
Modern C++ by Niels Lohmann, as vendored by llama.cpp at the commit above,
under the MIT license; its copyright notice is preserved in
[`Sources/LlamaCpp/vendoring/LICENSE-nlohmann-json`](Sources/LlamaCpp/vendoring/LICENSE-nlohmann-json).

## Model weights

Model weights are not included in this repository. The installer downloads a
pinned revision of a supported checkpoint and repacks it locally. Most are
pre-quantized community conversions; MiniCPM5-2B is read from its vendor's own
BF16 repository and quantized in flight.

Qwen 3.6: revision `38740b847e4cb78f352aba30aa41c76e08e6eb46` of
[`mlx-community/Qwen3.6-35B-A3B-4bit`](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit),
a quantization of Alibaba's
[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) checkpoint, whose
[license](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/LICENSE)
governs those weights.

Maple Preview: revision `361db5da5e74ff6fcdd852d478e1f266ce11013a` of
[`deepgrove/maple-preview-2bit-mlx`](https://huggingface.co/deepgrove/maple-preview-2bit-mlx).
The installer verifies its source-index SHA-256
`56000110535c5023b43209a5c142035e12c1cde7b1118759cc9f86335d46ef95`.
That pinned repository revision does not declare a license for the checkpoint
weights in a model card, LICENSE file, or Hub license tag. Downloading, using,
or redistributing those weights therefore requires separately establishing
that the necessary rights have been obtained; Mference makes no grant of
rights to them.

MiniCPM5-2B: revision `cd199ce3ee67549c42ef7372f809f2c63599a3e9` of
[`openbmb/MiniCPM5-2B`](https://huggingface.co/openbmb/MiniCPM5-2B), the
vendor's own BF16 upload, quantized to INT4 affine group-64 at install; the
installer verifies its source-index SHA-256
`6d839cd76e8395de548a0e6cc310386f66d1ecbb2c75d198a8dfd3d70892b756`. OpenBMB
publishes the model and its weights under the
[Apache License 2.0](https://github.com/OpenBMB/MiniCPM/blob/main/LICENSE)
(`license: apache-2.0` in the model card). The W2.1b quantizer-quality control,
revision `35ac38ee7bdb0bf7fa748d0700eeb6d6675760a3` of
[`openbmb/MiniCPM5-2B-MLX`](https://huggingface.co/openbmb/MiniCPM5-2B-MLX)
(index SHA-256 `ccf202e0a06fe3c7eb8f354cfb29412a5e64956ad895413d4d9267ae4b3a6045`),
is the vendor's own conversion under the same license. No reference code was
imported for this family: its architecture was read from
`transformers` v5.6.2 (Apache-2.0) and nothing was copied from it.

GGUF models run through the `LlamaCpp` target are not installed by
`MferenceRepack`; they are single files read in place. Two are product models,
checked against the Hugging Face LFS object ids (SHA-256) on 2026-11-18:

- Quick: `Qwen3.5-4B-UD-Q4_K_XL.gguf` from revision
  `e87f176479d0855a907a41277aca2f8ee7a09523` of
  [`unsloth/Qwen3.5-4B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF),
  2,912,109,728 bytes, SHA-256
  `b252c5610a42ca82d20fe2a12813e9d069eed89292907e26c783eeb0bc961bc7`. Its
  model card declares `license: apache-2.0`, a quantization of Alibaba's
  [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B).
- Deep: `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` from revision
  `d562806dbafae37109975e970aae91b43e73b440` of
  [`ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF),
  12,120,016,960 bytes, SHA-256
  `58fd826723939933dc86f45b7fe04545cbc2de1c70f6fe2cdd3858c87a98c12f`. Its
  model card declares `license: apache-2.0`, a quantization of Alibaba's
  [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).

The Mac app's Find (content search) downloads one more GGUF when the person
turns Find on, and a second when they turn on Settings › Find › Photos, video
and audio, both checked the same way on 2026-11-19:

- Find's search model: `embeddinggemma-2-Q8_0.gguf` from revision
  `6ea3e3a309d12a4779cfda98b8c19800d2d5da71` of
  [`ggml-org/embeddinggemma-2-GGUF`](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF),
  309,855,456 bytes, SHA-256
  `2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135`. It is
  the text part of Google's
  [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2), whose
  model card and the GGUF repository both declare `license: apache-2.0`.
- Find's image and audio encoders: `mmproj-embeddinggemma-2-Q8_0.gguf` from
  the same revision of the same repository, 554,821,056 bytes, SHA-256
  `0531309472a6bc6f4b146c3736eee7795ec9db1903c25490d3c937490cef6dd8`, the
  vision and audio towers of the same model, under the same licence.

Downloaded weights remain a separate work governed by their source terms. Do
not redistribute weights as part of Mference releases.

## Qwen3.5 GGUF chat template test fixture

`Tests/MferenceEngine/LlamaChatTemplateFixtures.swift` embeds the chat
template stored in the `tokenizer.chat_template` metadata of
`Qwen3.5-4B-UD-Q4_K_XL.gguf`, revision
`e87f176479d0855a907a41277aca2f8ee7a09523` of
[`unsloth/Qwen3.5-4B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF),
whose model card declares `license: apache-2.0`, together with that template's
renders by llama.cpp at commit `b1b770d90d94cad9e9ea5b993bdbc383a522fb28`.
Model weights are not bundled with Mference.

## Swift-Qwen template test fixture

`Tests/Mference/Core/Tokenization/Fixtures/SwiftQwenTemplate/chat_template.jinja`
is from `ukisai/Swift-Qwen3.8-27b`, revision
`1b30aaaf753fe5c1cb51ada2ea0367a53445359c`. The source model card identifies
Swift Open License 1.0. See the
[pinned upstream repository](https://huggingface.co/ukisai/Swift-Qwen3.8-27b/tree/1b30aaaf753fe5c1cb51ada2ea0367a53445359c)
for attribution and terms. Model weights are not bundled with Mference.

## Swift package graph

The following table covers the complete graph reported by
`swift package show-dependencies` from the checked-in
[`Package.resolved`](Package.resolved). Remote revisions are recorded there;
the local Jinja snapshot records its revision and patch in
[`Vendor/swift-jinja/README.md`](Vendor/swift-jinja/README.md).

| Package | Version | License in locked checkout |
| --- | --- | --- |
| [swift-transformers](https://github.com/huggingface/swift-transformers) | 1.3.3 | Apache-2.0 |
| [swift-jinja](https://github.com/huggingface/swift-jinja) | Local 2.5.1 snapshot with whitespace and null-rendering fixes | [Apache-2.0](Vendor/swift-jinja/LICENSE) |
| [swift-huggingface](https://github.com/huggingface/swift-huggingface) | 0.9.0 | Apache-2.0 |
| [EventSource](https://github.com/mattt/EventSource) | 1.4.1 | MIT |
| [swift-nio](https://github.com/apple/swift-nio) | 2.99.0 | Apache-2.0; upstream NOTICE applies |
| [swift-atomics](https://github.com/apple/swift-atomics) | 1.3.0 | Apache-2.0 with Runtime Library Exception |
| [swift-collections](https://github.com/apple/swift-collections) | 1.5.1 | Apache-2.0 with Runtime Library Exception |
| [swift-system](https://github.com/apple/swift-system) | 1.6.4 | Apache-2.0 with Runtime Library Exception |
| [swift-crypto](https://github.com/apple/swift-crypto) | 4.5.0 | Apache-2.0; upstream NOTICE applies |
| [swift-asn1](https://github.com/apple/swift-asn1) | 1.7.0 | Apache-2.0; upstream NOTICE applies |
| [yyjson](https://github.com/ibireme/yyjson) | 0.12.0 | MIT |

No copyleft or custom non-commercial license was found in this resolved graph.
The dependency license files remain authoritative. For binary distribution,
collect their license and NOTICE files from the exact revisions in
`Package.resolved`; do not treat this summary as a substitute for that bundle.
