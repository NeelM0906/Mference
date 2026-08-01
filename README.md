<h1 align="center">Mference</h1>

<p align="center">
  <strong>Big MoE models in ~1.5–2 GB of RAM</strong><br>
  A Swift + Metal inference engine for any Apple Silicon Mac, even the 8 GB ones.
</p>

<p align="center">
  <img alt="Swift 6.1 or later" src="https://img.shields.io/badge/Swift-6.1%2B-F05138?logo=swift&logoColor=white">
  <img alt="Metal 3 or later" src="https://img.shields.io/badge/Metal-3%2B-5E5CE6">
  <img alt="macOS 15 or later" src="https://img.shields.io/badge/macOS-15%2B-000000?logo=apple&logoColor=white">
  <a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/License-MIT-2ea44f"></a>
</p>

<p align="center">
  <a href="#try-it">Quick start</a> ·
  <a href="docs/OPENAI_SERVER.md">Local server</a> ·
  <a href="docs/BENCHMARKS.md">Benchmarks</a> ·
  <a href="docs/COMMUNITY_BENCHMARKS.md">Contribute results</a> ·
  <a href="docs/SYSTEM_DESIGN.md">How it works</a> ·
  <a href="#acknowledgments">Acknowledgments</a>
</p>

Mixture-of-experts models activate only a few billion parameters per token.
Mference exploits that: it keeps each model's shared core and KV cache in
memory, then streams just the experts chosen for each token from SSD. The
model never has to fit in RAM — only its working set does.

Mference currently runs two pinned instruction checkpoints:

- **[Gemma 4 26B-A4B](https://ai.google.dev/gemma/docs/core/model_card_4)** —
  26B total, ~3.88B active per token, in ~2 GB of memory.
- **[Qwen 3.6 35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)** — 35B
  total, ~3B active per token, in ~1.45 GB of memory. A hybrid of 30
  gated-DeltaNet linear-attention layers and 10 full-attention layers: the
  linear layers keep a fixed-size recurrent state instead of a KV cache, so
  only a quarter of the model grows with context.

The runtime, streaming installer, CLI, native Mac app, and loopback
OpenAI-compatible server are written in Swift and Metal. Mference is
model-specific rather than a wrapper around MLX or llama.cpp: each
architecture is enumerated explicitly, with its own pinned checkpoint,
compile-time baseline, and manifest contract.

## Try it

```bash
git clone <your-clone-url> Mference
cd Mference
swift build -c release
.build/release/MferenceMac
```

When the app opens, choose **Download** and let Mference fetch and repack the
pinned model. Once it is ready, choose **Load Model**, type your prompt, and
press **Generate**. The app installs Gemma 4 by default; select Qwen 3.6 with
`defaults write Mference model qwen36` (or `MFERENCE_MODEL=qwen36` in the
environment) before launching.

From the command line:

```bash
# Install a model (streams and repacks; never materializes the full checkpoint)
swift run -c release MferenceRepack --model qwen36 --output scratch/qwen36.gturbo

# Generate
swift run -c release MferenceCLI \
  --model scratch/qwen36.gturbo \
  --prompt "The capital of France is" \
  --max-new 64
```

## At a glance

| Metric | Value |
| --- | --- |
| Models | Gemma 4 26B-A4B IT (26B total, ~3.88B active) · Qwen 3.6 35B-A3B (35B total, ~3B active) |
| Weights | MLX affine 4-bit, group 64; 8-bit routers; 4-bit shared and routed experts |
| Memory | ~2 GB (Gemma 4) · ~1.45 GB (Qwen 3.6), including a 4K KV cache |
| Storage | ~14.3 GB installed (Gemma 4) · ~19.6 GB (Qwen 3.6) |
| Hardware | Apple Silicon Mac; 8 GB of RAM |
| Platform | macOS 15+, Metal 3 (MSL 3.2), Swift 6.1+; running on macOS 26 with an Apple10 GPU adds the Metal 4 tensor-ops prefill path |
| Measured decode, Gemma 4 | 5.1–6.3 tok/s (8 GB M2 Air) · 31–35 tok/s (24 GB M5 Pro) |
| Measured decode, Qwen 3.6 | 18.8–23.1 tok/s (M5) at a 1,447–1,464 MiB peak footprint |

Qwen 3.6 numbers follow the frozen
[community benchmark protocol](docs/COMMUNITY_BENCHMARKS.md) — three fixed
prompts, fixed seeds, one discarded warmup, measured runs in fresh processes,
every run reaching a natural end of turn. Rerunning all three cases with the
host constrained to an ~8 GB working set changed neither throughput
(22.95 / 21.35 / 18.62 tok/s) nor output — every generation was
byte-identical — because the expert pool streams from SSD either way. That
constrained result is emulated memory pressure on M5 hardware, not a physical
8 GB Mac; see [Benchmarks](docs/BENCHMARKS.md) and the
[Qwen 3.6 performance notes](docs/QWEN36_PERFORMANCE.md), including the
negative results.

## Products

| Product | Purpose |
| --- | --- |
| `Mference` | Swift library containing the runtime and Metal kernels |
| `MferenceMac` | Native Mac app for installation and generation |
| `MferenceDecodeService` | One-shot local model and Metal owner used by the Mac app |
| `MferenceCLI` | Command-line instruction chat and raw completion |
| `MferenceServer` | Loopback OpenAI-compatible Chat Completions server |
| `MferenceRepack` | Streaming model installer and install verifier |

Only one model-owning product should run at a time. The server speaks each
model's native dialect — Gemma's chat format and tool-call DSL, or Qwen's
ChatML template with `<tool_call>` function calls — selected automatically
from the installed model.

### Requirements

- An Apple Silicon Mac (arm64 only)
- macOS 15 or later, with Metal 3; Xcode 16.3 and Swift 6.1 or newer
- Free storage for the model install (~14.3 GB Gemma 4, ~19.6 GB Qwen 3.6)
- An internet connection for the first install

The shader library is compiled from source at startup, and the choice of
shading-language version is made then, not at build time. A single binary
therefore covers both worlds: *running* on macOS 26 with an Apple10 GPU
compiles at MSL 4.0 and enables the Metal 4 tensor-ops prefill kernels, while
every other supported configuration compiles at MSL 3.2 and selects the
non-tensor kernels automatically. Both paths run the full Gemma 4 and Qwen 3.6
feature set; they agree to within kernel tolerance rather than bit-exactly, so
expect the same quality but not identical sampled tokens.

## How it works

The installer streams bounded byte ranges from the pinned Hugging Face
revision and repacks them directly into an on-disk layout (`.gturbo`) built
for per-expert reads: resident tensors in one mapped file, and each layer's
routed experts as fixed-stride, page-aligned blobs. At generation time the
runtime keeps the common weights mapped read-only, holds a small per-layer
expert cache (16 slots, LFU), and `pread`s only the eight experts each
layer's router selects for the current token.

Qwen 3.6's linear-attention layers replace KV storage entirely: each keeps a
2 MiB delta-rule state and a 3-row convolution tail, updated in place every
token. The gated-DeltaNet kernels are validated against a CPU reference,
including the guarantee that a chunked prefill of T rows matches T sequential
decode steps through the same kernels.

The full design, the memory budget, and the experiment record that shaped the
engine are in [System design](docs/SYSTEM_DESIGN.md) and the
[optimization journey](docs/OPTIMIZATION_JOURNEY.md).

## Roadmap

- More architectures — the model layer is built for enumeration, and the
  Qwen 3.6 port is the template for the next one
- Overlapping expert I/O with GPU work (decode is currently ~53% expert-read
  wait, serialized with compute)
- Per-model quality validation (KLD against reference implementations)
- Longer contexts, vision towers, and a hardware benchmark matrix

## Acknowledgments

Mference is heavily inspired by — and its foundation is derived from —
**[TurboFieldfare](https://github.com/drumih/turbo-fieldfare)** by Andrey
Mikhaylov, which pioneered running Gemma 4 26B in ~2 GB on Apple Silicon and
documented over a hundred experiments behind its design. Those derived
portions are licensed under the [Apache License 2.0](LICENSE-APACHE); see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details. New Mference
code is [MIT-licensed](LICENSE).

Model weights remain subject to their own terms: the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms) and the
[Qwen license](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/LICENSE).
