<p align="center">
  <img src="Sources/MferenceApp/Mac/Resources/mference-app-icon.png" alt="Mference app icon" width="160">
</p>

<h1 align="center">Mference</h1>

<p align="center">
  <strong>Big MoE models in "Small" GB of RAM</strong><br>
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

<p align="center">
  <strong>Qwen 3.6 on a 24 GB M5: 23.5–29.3 tok/s decode · 2.20× faster long-prompt prefill</strong><br>
  <strong>Inkling-Small on the same host: 3.0–3.7 tok/s decode · 16.4% faster with native top-6 Metal</strong>
</p>

Mixture-of-experts models activate only a few billion (cough) parameters per token.
Mference builds on that: it keeps each model's shared core and KV cache in
memory, then streams just the experts chosen for each token from SSD. The
model never has to fit in RAM — only its working set does.

Mference currently runs five pinned instruction checkpoints:

- **[Gemma 4 26B-A4B](https://ai.google.dev/gemma/docs/core/model_card_4)** —
  26B total, ~3.88B active per token, in ~2 GB of memory.
- **[Qwen 3.6 35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)** — 35B
  total, ~3B active per token, from ~1.45 GB of memory with the 16-slot profile.
- **[DeepSeek-V4-Flash 284B-A13B](https://huggingface.co/mlx-community/DeepSeek-V4-Flash-2bit-DQ)**
  *(experimental)* — 284B total, ~13B active per token, from
  the 2-bit dynamic-quant checkpoint (2-bit experts, 4-bit core). Budget: ~6.8 GB peak at the 8-slot floor, ~91 GB on  disk;
- **[Inkling-Small 276B-A12B](https://huggingface.co/pipenetwork/Inkling-Small-MLX-4bit)** —
  276B total, ~12B active per token, in ~9 GB of memory.
- **[Maple Preview](https://huggingface.co/deepgrove/maple-preview-2bit-mlx)** —
  a 20B total, ~1B active per token, from the ternary (1.58 bit per parameter) quantization, using ~645 MiB
  of memory. Its chat template opens a live `<think>` reasoning block, so give
  it a generous max-token allowance; an approximate FlashHead decode head is an
  opt-in via `--flash-head`.
- **[Qwen 3.8 27B](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit)** —
  the first dense family: 27B parameters, all active, text-only port of the
  multimodal checkpoint, fully resident in ~15 GB (24 GB Macs). Ships MTP
  speculative decoding with byte-identical greedy output — 15.0 tok/s decode
  on a 24 GB M5 (2.35× mlx-vlm on the same checkpoint). Its chat template
  also opens a live `<think>` block. Long contexts (past RAM, up to the
  model's 262k) run a paged KV cache with an SSD spill tier and query-aware
  sparse decode at full FP16 — see
  [docs/QWEN38_LONG_CONTEXT.md](docs/QWEN38_LONG_CONTEXT.md).
  
The runtime, streaming installer, CLI, native Mac app, and loopback
OpenAI-compatible server are written in Swift and Metal. Mference is
model-specific rather than a wrapper around MLX or llama.cpp: each
architecture is enumerated explicitly, with its own pinned checkpoint,
compile-time baseline, and manifest contract. New families merge through the
[family acceptance gate](docs/FAMILY_GATE.md).

## Try it

```bash
git clone https://github.com/NeelM0906/Mference.git
cd Mference
swift build -c release
.build/release/MferenceMac
```

When the app opens, choose **Download** and let Mference fetch and repack the
pinned model — or choose **Choose Existing Model…** to point at a `.gturbo`
directory already on disk. Once it is ready, choose **Load Model**, type your
prompt, and press **Generate**. The app installs Gemma 4 by default; select
Qwen 3.6 or Maple with `defaults write Mference model qwen36` or
`defaults write Mference model maple` (or the matching `MFERENCE_MODEL` value)
before launching.

Each chat keeps its own multi-turn history in a collapsible left sidebar
(<kbd>Command</kbd>+<kbd>N</kbd> for a new chat,
<kbd>Control</kbd>+<kbd>Command</kbd>+<kbd>S</kbd> to toggle the sidebar).
History is written locally next to the model directory and rendered through the
installed model's own chat template, so both Gemma 4 and Qwen 3.6 see their
native dialect. The composer can extract text locally from PDF, DOCX, PPTX, and
XLSX files; nothing is uploaded. Extraction is bounded, so very long documents
are trimmed and marked as truncated. The attachments on one prompt hold at most
750,000 characters of extracted text in total — all a single request can carry —
and a file that would exceed that is refused with a message rather than silently
dropped. Before a turn is committed, the app measures the rendered conversation
with the model tokenizer; when older turns no longer fit the selected context,
it compresses them into a rolling per-chat memory while the full transcript
stays visible. Turns folded into that memory then release their hidden
request-side copy — attached document text included — because the summary
stands in for them from that point on; nothing visible changes. Chats and their
transcripts are never evicted automatically, so delete chats you no longer want
kept. Appearance follows System, Light, or Dark from the **Appearance** menu.

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
| Models | Gemma 4 26B-A4B IT (26B total, ~3.88B active) · Qwen 3.6 35B-A3B (35B total, ~3B active) · DeepSeek-V4-Flash 284B-A13B (experimental) · Inkling-Small 276B-A12B (276B total, ~12B active) · Maple Preview (20B total, ~1B active) · Qwen 3.8 27B (dense, MTP speculative decode) |
| Weights | MLX affine or ternary, group 64/128; INT8 or BF16 routers; 4-bit or 2-bit routed experts |
| Memory | ~2 GB (Gemma 4) · ~1.45 GB at 16 slots (Qwen 3.6; CLI/server auto uses 96 slots on 24 GiB+ hosts, 32 on 16 GiB+) · est. ~6.8 GB (DeepSeek-V4-Flash) · ~9 GB (Inkling-Small), including a 4K KV cache · 490.64 MiB (Maple, 128-token prompt) |
| Storage | ~14.3 GB installed (Gemma 4) · ~19.6 GB (Qwen 3.6) · ~91 GB (DeepSeek-V4-Flash) · ~148 GB (Inkling-Small) · ~6.6 GB (Maple) |
| Hardware | Apple Silicon Mac; 8 GB of RAM |
| Platform | macOS 15+, Metal 3 (MSL 3.2), Swift 6.1+; running on macOS 26 with an Apple10 GPU adds the Metal 4 tensor-ops prefill path |
| Measured decode, Gemma 4 | 5.1–6.3 tok/s (8 GB M2 Air) · 31–35 tok/s (24 GB M5 Pro) |
| Measured decode, Qwen 3.6 | 23.5–29.3 tok/s (24 GB M5, 32-slot profile; auto now selects 96 slots on that host) |
| Measured decode, DeepSeek-V4-Flash | 5.3–6.1 tok/s (256 GB M3 Ultra) at a 5,671–5,679 MiB peak footprint |
| Measured decode, Inkling-Small | 3.0–3.7 tok/s (24 GB M5, optimized native top-6 path) · 5.3–6.9 tok/s (256 GB M3 Ultra) at an 8,936–8,939 MiB peak footprint |
| Measured, Maple Preview | Exact head: 18.9–24.6 tok/s decode, 25.1–44.9 tok/s prefill, and 491–1,211 MiB peak process footprint on 128-8192 context (16 GB M4) |
| Measured, Qwen 3.8 27B | 15.0 tok/s decode (MTP speculative, byte-identical; 7.9 plain) · ~60 tok/s prefill (24 GB M5); mlx-vlm on the same checkpoint: 6.41 decode / 40.5 prefill |

Qwen 3.6 numbers follow the frozen
[community benchmark protocol](docs/COMMUNITY_BENCHMARKS.md) — three fixed
prompts and seeds, one discarded warmup, measured runs in fresh processes, and
every run reaching a natural end of turn. The optimized short, medium, and long
cases decode at 29.293, 27.460, and 23.470 tok/s. Their outputs are byte-identical
to matching 16-slot controls, making the model-aware 32-slot rung worth an
18.1% geometric-mean decode gain. Long-prompt prefill fell from 58.45 to 26.54
seconds, a 2.20× speedup. With the GPU-resident slot map, auto has since moved
24 GiB-class hosts to a 96-slot rung. Hosts below 16 GiB and the Mac app retain
the 16-slot memory-first path. See [Benchmarks](docs/BENCHMARKS.md) and the
[Qwen 3.6 performance notes](docs/QWEN36_PERFORMANCE.md) for exact commands,
token counts, memory behavior, and rejected experiments.

Inkling-Small uses its own six-expert INT4 Metal pipeline rather than padding
the router result to eight experts. Resident-expert phase 1 runs while cache
misses stream from SSD; across the same frozen short, medium, and long cases,
decode improved from 2.909/2.961/2.819 to 3.434/3.670/3.038 tok/s. That is a
16.4% geometric-mean gain, with byte-identical generated output in every A/B.
See the [Inkling performance notes](docs/INKLING_SMALL.md#native-top-6-decode-2026-08-06).

# Products

| Product | Purpose |
| --- | --- |
| `Mference` | Swift library containing the runtime and Metal kernels |
| `MferenceMac` | Native Mac app for installation and generation |
| `MferenceDecodeService` | One-shot local model and Metal owner used by the Mac app |
| `MferenceCLI` | Command-line instruction chat and raw completion |
| `MferenceServer` | OpenAI-compatible Chat Completions server, on loopback by default or a Tailnet address with `--bind tailnet` |
| `MferenceRepack` | Streaming model installer and install verifier |

Only one model-owning product should run at a time. The server selects the
installed model's native dialect automatically, including Gemma's chat format,
Qwen's ChatML template with `<tool_call>` function calls, and Maple's ChatML
template with hidden reasoning.

### Requirements

- An Apple Silicon Mac (arm64 only)
- macOS 15 or later, with Metal 3; Xcode 16.3 and Swift 6.1 or newer
- Free storage for the model install (~6.6 GB Maple, ~14.3 GB Gemma 4, ~19.6 GB Qwen 3.6; the largest families require substantially more)
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
runtime keeps the common weights mapped read-only, holds a small per-layer LFU
expert cache, and `pread`s only the experts each layer's router selects for the
current token. Inkling dispatches six experts; Gemma, Qwen, and Maple dispatch
eight.
The memory-first path uses 16 slots; CLI and server auto select larger rungs
for Qwen — 96 slots on hosts with at least 24 GiB, 32 with at least 16 GiB —
because its 256 experts per layer benefit measurably from the added coverage.
Inkling remains at 16 because a 24-slot control warmup on the 24 GB M5 entered
memory pressure and regressed sharply. Each layer's slots share one contiguous
wired buffer, and on Qwen a GPU-resident expert-to-slot map lets layers whose
eight experts are all cached run their routed branch from pre-encoded,
GPU-guarded commands, with no CPU expert planning or fetching;
the routed command buffer also commits eagerly, gated on a shared event that
fires as expert fills land. An explicit `resident` mode that maps every layer
file once exists as an opt-in, but it measured slower than the slot rungs under
page-cache pressure.

Qwen 3.6's linear-attention layers replace KV storage entirely: each keeps a
2 MiB delta-rule state and a 3-row convolution tail, updated in place every
token. The gated-DeltaNet kernels are validated against a CPU reference,
including the guarantee that a chunked prefill of T rows matches T sequential
decode steps through the same kernels.

The full design, the memory budget, and the experiment record that shaped the
engine are in [System design](docs/SYSTEM_DESIGN.md) and the
[optimization journey](docs/OPTIMIZATION_JOURNEY.md).

## Roadmap

- More explicitly pinned architectures without a generic-model fallback
- Extend resident-expert compute/I/O overlap to every model family and prefill
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
[Gemma 4 license](https://ai.google.dev/gemma/apache_2) and the
[Qwen license](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/LICENSE).
Maple's pinned checkpoint declares no license; establish the necessary rights
before downloading, using, or redistributing it. Maple's MLX-derived kernel
work is covered by [LICENSE-MLX](LICENSE-MLX); see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
