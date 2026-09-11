<p align="center">
  <img src="docs/assets/mference-app-icon.png" alt="Mference app icon" width="160">
</p>

<h1 align="center">Mference</h1>

<p align="center">
  <strong>Frontier models on Apple Silicon — even when the weights exceed RAM</strong><br>
  A native Swift + Metal runtime for pinned checkpoints, with bounded-memory streaming and measured ports.
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
  <a href="docs/OPEN_WEBUI.md">The UI</a> ·
  <a href="docs/BENCHMARKS.md">Benchmarks</a> ·
  <a href="docs/COMMUNITY_BENCHMARKS.md">Contribute results</a> ·
  <a href="docs/SYSTEM_DESIGN.md">How it works</a> ·
  <a href="docs/FAMILY_GATE.md">Add a model family</a> ·
  <a href="#acknowledgments">Acknowledgments</a>
</p>

<p align="center">
  <strong>Qwen3.8-Flash-Next, a 180B-parameter MoE: 24–210 prompt tok/s and 18.2–21.0 decode tok/s resident on M3 Ultra</strong><br>
  <strong>Qwen 3.6 on a 24 GB M5: 23.5–29.3 tok/s decode · on a 256 GB M3 Ultra: 36.1–42.2 tok/s</strong><br>
  <strong>Inkling-Small 276B on a 24 GB M5: 3.0–3.7 tok/s · Qwen 3.8 27B dense on an M3 Ultra: 38.4–39.4 tok/s</strong>
</p>

Mixture-of-experts models activate only a fraction of their parameters per
token. Mference turns that sparsity into a bounded working set: it keeps the
shared core and KV cache in memory, then streams only the experts selected for
the current token from SSD. The complete model does not have to fit in RAM.
The same fixed aperture now covers three kinds of tensor: routed experts stream
into a slot cache, long-context KV pages spill to SSD, and Flash-Next's
320-million-row n-gram embedding table is read a few rows per token. Dense
families remain fully resident but share the same native runtime, verified
installer, model contract, and serving layer.

Mference currently runs eight pinned instruction checkpoints:

- **[Gemma 4 26B-A4B](https://ai.google.dev/gemma/docs/core/model_card_4)** —
  26B total, ~3.88B active per token, in ~2 GB of memory.
- **[Qwen 3.6 35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)** — 35B
  total, ~3B active per token, from ~1.45 GB of memory with the 16-slot profile.
  Installs either from the mlx-community conversion or from the vendor's BF16
  repo through the quantize-in-flight path, whose output passed the
  [W2.1b quality gate](docs/QUANTIZER_QUALITY.md) against that conversion.
- **[DeepSeek-V4-Flash 284B-A13B](https://huggingface.co/mlx-community/DeepSeek-V4-Flash-2bit-DQ)**
  *(experimental)* — 284B total, ~13B active per token, from the 2-bit
  dynamic-quant checkpoint (2-bit experts, 4-bit core). ~5.7 GB peak at the
  16-slot rung, ~91 GB on disk.
- **[Inkling-Small 276B-A12B](https://huggingface.co/pipenetwork/Inkling-Small-MLX-4bit)** —
  276B total, ~12B active per token, in ~9 GB of memory.
- **[Maple Preview](https://huggingface.co/deepgrove/maple-preview-2bit-mlx)** —
  20B total, ~1B active per token, from the ternary (1.58 bit per parameter)
  quantization, in ~645 MiB of memory. Its chat template opens a live
  `<think>` reasoning block, so give it a generous max-token allowance; an
  approximate FlashHead decode head is an opt-in via `--flash-head`.
- **[Qwen 3.8 27B](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit)** —
  the first dense family: 27B parameters, all active, text-only port of the
  multimodal checkpoint, fully resident in ~15 GB (24 GB Macs). Ships MTP
  speculative decoding with byte-identical greedy output — 15.0 tok/s decode
  on a 24 GB M5 (2.35× mlx-vlm on the same checkpoint), and an opt-in
  [DFlash2 block-diffusion drafter](docs/QWEN38_DFLASH2.md) as an alternative
  draft source. Its chat template also opens a live `<think>` block. Long
  contexts (past RAM, up to the model's 262k) run a paged KV cache with an
  SSD spill tier and query-aware sparse decode at full FP16 — see
  [docs/QWEN38_LONG_CONTEXT.md](docs/QWEN38_LONG_CONTEXT.md).
- **[Qwen3.8-Flash-Next 180B-A3.5B](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**
  *(new)* — ~180B total, ~3.5B active per token, 512 experts per layer with
  top-10 routing, gated DeltaNet plus sparse-indexed full attention,
  hyper-connections, and a hashed n-gram embedding table that is 102 GB of
  the checkpoint and is read by row lookup. Installed from the vendor's BF16
  repo (359 GB streamed, ~175 GB on disk, quantized in flight; routers at
  INT8 after a measured routing check). On a 256 GB M3 Ultra, the latest
  frozen-protocol run reaches **20.1 / 21.0 / 18.2 tok/s
  decode**, with native chunked prefill at **24 / 96 / 210 prompt tok/s**
  across its 62 / 426 / 2,940-token cases. The long prompt is 13.8× faster to
  prefill than the former scalar path. An exact passkey was retrieved from a
  3,247-token prompt, beyond the indexer's 2,048-token budget. Runs from the
  CLI, the server, and the UI. See
  the [bring-up dossier](docs/families/QWEN38_FLASH_NEXT.md).
- **[MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B)** *(new)* — the
  first plain-llama dense family: 2B total, all active, 42 layers with no
  expert routing, no q/k norms, and no output gate. Installed from the
  vendor's BF16 upload (~5 GB read, quantized in flight to INT4 group-64,
  1.43 GB on disk), and the second calibration point for the
  [W2.1b quality gate](docs/QUANTIZER_QUALITY.md) — measured against the
  vendor's own MLX conversion rather than a community one. On a 24 GB M5 the
  protocol's short-explanation case decodes at **65.56 tok/s**. Its chat
  template opens a live `<think>` block that the other two protocol cases do
  not leave within the 1,024-token cap, so those are reported as a stated
  deviation at a raised cap. See the
  [bring-up dossier](docs/families/MINICPM5.md).
- **[GLM-5.3-Flash 320B-A18B](https://huggingface.co/zai-org/GLM-5.3-Flash)**
  *(family gate green 2026-09-11; prefill perf pass continuing)* — 320B total, 18B active,
  45 layers mixing Kimi Delta Attention with NoPE latent sparse attention over
  a four-stream mHC residual, 288 experts per layer with top-8 routing. Runs
  from PipeNetwork's mixed 4/8-bit conversion (~181 GB installed) with the
  whole expert set resident on a 256 GB host. See the
  [bring-up dossier](docs/families/GLM53_FLASH.md).

The runtime, streaming installer, CLI, and loopback OpenAI-compatible server
are written in Swift and Metal; the UI is Open WebUI, driven through that
server. Mference is model-specific rather than a wrapper around MLX or
llama.cpp: each
architecture is enumerated explicitly, with its own pinned checkpoint,
compile-time baseline, and manifest contract. New families merge through the
[family acceptance gate](docs/FAMILY_GATE.md) using the
[bring-up kit](docs/FAMILY_CONTRACT.md) described below.

## Try it

```bash
git clone https://github.com/NeelM0906/Mference.git
cd Mference
swift build -c release
./mference-ui.sh install gemma4    # streams and repacks the pinned checkpoint (~14 GB)
./mference-ui.sh                   # starts the server and the UI, opens http://127.0.0.1:3000
```

The UI is [Open WebUI](https://github.com/open-webui/open-webui), installed
on first launch as a pinned Python tool (the launcher uses `uv`; it prints
the install command if `uv` is missing). Behind it, `MferenceServer` runs in
library mode: it lists every installed model it finds in the checkout's
`scratch/`, in `~/Library/Application Support/Mference`, or under the
`Mference.libraryRoot` default, and swaps the loaded model in-process when
you pick a different one. Exactly one model is ever resident. The launcher also
registers each model in Open WebUI with its builtin tool schemas switched off,
since Open WebUI otherwise attaches them to every request and a one-line
question costs thousands of prompt tokens. Chats, prompts,
documents, and settings live in Open WebUI's local data directory; both
processes bind to loopback only and Open WebUI runs with authentication off,
so do not expose either port. Details, model switching cost, and
troubleshooting are in [docs/OPEN_WEBUI.md](docs/OPEN_WEBUI.md).

Install more families the same way (`./mference-ui.sh install qwen36`,
`maple`, `qwen38`, `deepseekv4flash`, `inklingsmall`, `qwen38flashnext`,
`minicpm5`),
and they appear in the model picker. `./mference-ui.sh models` shows what the
server would expose without loading anything.

From the command line, without the UI:

```bash
# Install a model (streams and repacks; never materializes the full checkpoint)
swift run -c release MferenceRepack --model qwen36 --output scratch/qwen36.gturbo

# Generate
swift run -c release MferenceCLI \
  --model scratch/qwen36.gturbo \
  --prompt "The capital of France is" \
  --max-new 64
```

The server alone, for other OpenAI-compatible clients, is documented in
[docs/OPENAI_SERVER.md](docs/OPENAI_SERVER.md).

## At a glance

| Metric | Value |
| --- | --- |
| Models | Gemma 4 26B-A4B IT · Qwen 3.6 35B-A3B · DeepSeek-V4-Flash 284B-A13B (experimental) · Inkling-Small 276B-A12B · Maple Preview 20B-A1B · Qwen 3.8 27B (dense, MTP or DFlash2 speculative decode) · Qwen3.8-Flash-Next 180B-A3.5B · MiniCPM5-2B (dense, plain llama) · GLM-5.3-Flash 320B-A18B (new) |
| Weights | MLX affine or ternary, group 64/128; INT8 or BF16 routers; 4-bit or 2-bit routed experts; vendor BF16 quantized in flight to INT4/INT8 group 64 for Qwen 3.6, Flash-Next, and MiniCPM5 |
| Memory | ~2 GB (Gemma 4) · ~1.45 GB at 16 slots (Qwen 3.6; CLI/server auto uses 96 slots on 24 GiB+ hosts, 32 on 16 GiB+) · ~5.7 GB (DeepSeek-V4-Flash) · ~9 GB (Inkling-Small), including a 4K KV cache · 490.64 MiB (Maple, 128-token prompt) · ~15 GB (Qwen 3.8, resident) · **~2.36 GB at 16 slots (Flash-Next); high-memory auto maps its ~68 GiB routed pool** · 123 MiB (MiniCPM5-2B, short-explanation case) |
| Storage | ~14.3 GB installed (Gemma 4) · ~19.6 GB (Qwen 3.6) · ~91 GB (DeepSeek-V4-Flash) · ~148 GB (Inkling-Small) · ~6.6 GB (Maple) · ~15 GB (Qwen 3.8) · ~175 GB (Flash-Next) · 1.43 GB (MiniCPM5-2B; 1,425,981,882 bytes over 8 files) |
| Hardware | Apple Silicon Mac; RAM is family-specific — low-memory streamed families run on 8 GB, while fully resident Qwen 3.8 requires a 24 GB-class host |
| Platform | macOS 15+, Metal 3 (MSL 3.2), Swift 6.1+; running on macOS 26 with an Apple10 GPU adds the Metal 4 tensor-ops prefill path |
| Measured decode, Gemma 4 | 5.1–6.3 tok/s (8 GB M2 Air) · 31–35 tok/s (24 GB M5 Pro) · 17.1–18.7 tok/s (256 GB M3 Ultra) |
| Measured decode, Qwen 3.6 | 23.5–29.3 tok/s (24 GB M5, 32-slot profile) · 36.1–42.2 tok/s (256 GB M3 Ultra, 96-slot auto rung, 6.8 GB peak) |
| Measured decode, DeepSeek-V4-Flash | 5.6–6.7 tok/s (256 GB M3 Ultra) at a 5,695–5,736 MiB peak footprint |
| Measured decode, Inkling-Small | 3.0–3.7 tok/s (24 GB M5, native top-6 path) · 5.3–7.1 tok/s (256 GB M3 Ultra) at a ~8.95 GB peak footprint |
| Measured, Maple Preview | Exact head: 18.9–24.6 tok/s decode, 25.1–44.9 tok/s prefill, and 491–1,211 MiB peak process footprint on 128-8192 context (16 GB M4) · 38.5 tok/s decode (M3 Ultra) |
| Measured, Qwen 3.8 27B | 15.0 tok/s decode (MTP speculative, byte-identical; 7.9 plain) · ~60 tok/s prefill (24 GB M5); mlx-vlm on the same checkpoint: 6.41 decode / 40.5 prefill · 38.4–39.4 tok/s plain decode (M3 Ultra, where MTP gives no gain) · passkey exact at 10.6k tokens through the paged KV + SSD tier |
| Measured, Qwen3.8-Flash-Next | Frozen protocol (256 GB M3 Ultra, INT8-router install): current native chunked prefill **24.31 / 96.38 / 210.45 prompt tok/s** at 62 / 426 / 2,940 tokens, a **2.29× / 6.57× / 13.79×** speedup over scalar resident prefill · the same autonomous run decoded at **20.109 / 21.029 / 18.168 tok/s** · a separate byte-identical routing A/B measured a 33.9% geometric-mean decode gain · historical 16-slot memory-first decode 11.81 / 11.71 / 11.11 tok/s at ~2.36 GB peak RSS · 12.3 tok/s at 3,247 tokens of context with the needle retrieved exactly |
| Measured, MiniCPM5-2B | Protocol proper (24 GB M5): **65.56 tok/s** decode on short-explanation, 65.50–65.80 across 3 runs, 3/3 to end of turn, 123 MiB peak RSS · as the stated deviation, with only the token cap raised: 37.63 tok/s (medium-review, still inside its think block at 4,096 tokens) and 34.72 tok/s (long-synthesis) · passkey exact at 8,692 prompt tokens through the paged KV cache |

Qwen 3.6 numbers follow the frozen
[community benchmark protocol](docs/COMMUNITY_BENCHMARKS.md) — three fixed
prompts and seeds, one discarded warmup, measured runs in fresh processes, and
every run reaching a natural end of turn. On the 24 GB M5 the optimized short,
medium, and long cases decode at 29.293, 27.460, and 23.470 tok/s, byte-identical
to matching 16-slot controls, an 18.1% geometric-mean decode gain for the
model-aware 32-slot rung; long-prompt prefill fell from 58.45 to 26.54 seconds,
a 2.20× speedup. On the 256 GB M3 Ultra the 96-slot auto rung nearly doubles
the 16-slot session: 42.20 / 40.02 / 36.14 tok/s against 22.82 / 22.19 / 22.18.
Hosts below 16 GiB retain the 16-slot memory-first path. See
[Benchmarks](docs/BENCHMARKS.md), the
[M3 Ultra report](docs/BENCHMARKS_M3_ULTRA.md), and the
[Qwen 3.6 performance notes](docs/QWEN36_PERFORMANCE.md) for exact commands,
token counts, memory behavior, and rejected experiments.

Inkling-Small uses its own six-expert INT4 Metal pipeline rather than padding
the router result to eight experts. Resident-expert phase 1 runs while cache
misses stream from SSD; on the 24 GB M5 decode improved from 2.909/2.961/2.819
to 3.434/3.670/3.038 tok/s, a 16.4% geometric-mean gain with byte-identical
output in every A/B. The streaming I/O overlap work (coalesced `preadv`,
shadow prefetch, pipelined prefill) then cut its marginal prefill cost on the
M3 Ultra by 10–14×. See the [Inkling notes](docs/INKLING_SMALL.md).

## Quality gates

A model that verifies, validates, loads, and generates fluent text can still be
wrong. Mference therefore treats installs and ports as measurements with
stated gates rather than as conversions that worked because they ran:

- **Bit parity (W2.1a).** The installer's quantizer is bit-identical to the
  runtime's reference quantizer, including through the streaming path,
  enforced in CI.
- **Quantizer quality (W2.1b).** For families installed from a vendor's BF16
  repo, the quantized weights are compared against an independent conversion
  at two levels: per-tensor reconstruction error against the BF16 source, and
  end-to-end greedy rollouts plus KL divergence between the two installs. On
  Qwen 3.6 both halves passed, and the model-level half caught two defects
  nothing structural could see — a missing RMSNorm `1 + w` fold and a
  tensor-name ordering mismatch — in an install that had verified and loaded.
  [docs/QUANTIZER_QUALITY.md](docs/QUANTIZER_QUALITY.md).
- **Kernel parity.** Every new kernel is gated against a CPU reference and,
  where a family has a Python reference implementation, against committed
  goldens from a toy checkpoint: integer selections exact, floating-point
  outputs within a stated tolerance tier, greedy rollouts token-exact.
- **Matched A/Bs.** Algebraically exact optimizations must keep generated output
  byte-identical to the control across the frozen protocol. Changes that alter
  FP16 reduction order publish their drift bound and exact greedy-rollout gate;
  approximate features remain opt-in.
- **The family gate.** Every new family, internal or external, passes the same
  [acceptance gate](docs/FAMILY_GATE.md): full suite three times, release
  build, static checks, pinned install and smoke, the frozen protocol page,
  a decode-phase attribution snapshot, provenance, and a merge note.
  `./bringup-check.sh <family> <gturbo>` automates the mechanical steps, and
  the [family contract](docs/FAMILY_CONTRACT.md) enumerates every
  architecture axis a family selects over, rot-checked by a test.

The [bring-up kit](docs/superpowers/specs/2026-08-08-family-bringup-kit-design.md)
is the project's answer to its own ambition: a flagship MoE running on Macs
people own within days of release. Flash-Next was its timed rehearsal, from a
four-day-old checkpoint to coherent real-model output.

## Project direction

Mference is a **checkpoint-specialized inference engine for Apple Silicon**.
Its north star is to make important open checkpoints both usable and fast on a
Mac, including models whose total weights, context state, or sparse tables
would otherwise exceed practical memory. It earns specialization by owning the
whole execution artifact: the pinned checkpoint contract, quantization and
on-disk layout, Metal kernels, scheduler, correctness evidence, and benchmark
claim advance together.

Three commitments keep that direction concrete:

- **Specialize the complete hot path.** Static model geometry is an advantage,
  not a limitation: fuse its recurrent blocks, batch its exact routed work,
  choose a hardware-aware residency plan, and remove generic dispatch that the
  pinned checkpoint does not need.
- **Bound memory without accepting a slow-engine identity.** Stream sparse
  resources and page long-lived state when necessary, but treat that memory
  freedom as room for better scheduling. A flagship port must pursue the best
  matched throughput available on its target Mac, not merely run where another
  engine cannot.
- **Treat every port as a measured product.** Pin the source, prove its kernels
  and quantization, and publish reproducible real-model behavior. Performance
  claims require the same checkpoint, hardware class, prompt shape, generation
  mode, and memory accounting; estimates are not baselines.
- **Expose one dependable local interface.** The server and Open WebUI are the
  user path; the CLI is the diagnostic path; every surface uses the same core.

Mference is not a generic model loader, a wrapper around another inference
engine, or a separate chat application. New families are selected when they
strengthen the fixed-aperture thesis, exercise an important new architecture,
or provide a useful calibration point for the porting system — not to maximize
a model count. The intended result of a port is a native model appliance, not
another compatibility entry in a generic loader.

# Products

| Product | Purpose |
| --- | --- |
| `Mference` | Swift library containing the runtime and Metal kernels |
| `MferenceCLI` | Command-line instruction chat and raw completion |
| `MferenceServer` | OpenAI-compatible Chat Completions server and the engine behind the UI; loopback by default or a Tailnet address with `--bind tailnet`; `--library` lists every installed model and swaps in-process |
| `mference-ui.sh` | Launcher: installs Open WebUI on first run, starts the server in library mode, opens the UI; `install <family>` and `models` subcommands |
| `MferenceRepack` | Streaming model installer, quantize-in-flight repacker, and install verifier |

The Swift package builds four products; `mference-ui.sh` is the supported
launcher that composes the server with Open WebUI.

Only one model-owning product should run at a time. The server selects the
installed model's native dialect automatically, including Gemma's chat format,
Qwen's ChatML template with `<tool_call>` function calls, Maple's ChatML
template with hidden reasoning, and GLM-5.3's `[gMASK]<sop>` turns with
`<think>` reasoning and `<arg_key>` / `<arg_value>` tool calls.

### Requirements

- An Apple Silicon Mac (arm64 only)
- macOS 15 or later, with Metal 3; Xcode 16.3 and Swift 6.1 or newer (the
  Command Line Tools alone also build and run everything)
- For the UI only: [`uv`](https://github.com/astral-sh/uv), which the launcher
  uses to install Open WebUI on Python 3.11 at first run; the CLI and server
  need no Python
- Free storage for the model install (~6.6 GB Maple, ~14.3 GB Gemma 4,
  ~19.6 GB Qwen 3.6; the largest families require substantially more, up to
  ~175 GB for Flash-Next)
- An internet connection for the first install

The shader library is compiled from source at startup, and the choice of
shading-language version is made then, not at build time. A single binary
therefore covers both worlds: *running* on macOS 26 with an Apple10 GPU
compiles at MSL 4.0 and enables the Metal 4 tensor-ops prefill kernels, while
every other supported configuration compiles at MSL 3.2 and selects the
non-tensor kernels automatically. Both paths run the full feature set; they
agree to within kernel tolerance rather than bit-exactly, so expect the same
quality but not identical sampled tokens.

## How it works

The installer streams bounded byte ranges from the pinned Hugging Face
revision and repacks them directly into an on-disk layout (`.gturbo`) built
for per-expert reads: resident tensors in one mapped file, and each layer's
routed experts as fixed-stride, page-aligned blobs. For vendor BF16 repos it
quantizes in flight, so a 359 GB checkpoint becomes a 175 GB install without
ever existing whole on disk. At generation time the runtime keeps the common
weights mapped read-only, holds a small per-layer LFU expert cache, and
`pread`s only the experts each layer's router selects for the current token.
Inkling dispatches six experts; Gemma, Qwen 3.6, and Maple dispatch eight;
Flash-Next dispatches ten of 512.

The memory-first path uses 16 slots; CLI and server auto select larger measured
rungs. Qwen 3.6 uses 96 slots on hosts with at least 24 GiB and 32 with at
least 16 GiB. Flash-Next maps its routed pool on hosts with at least 192 GiB
when the pool and core leave 32 GiB of headroom. That improved the full
protocol from 11.81 / 11.71 / 11.11 to 15.12 / 15.39 / 14.91 tok/s on the
256 GiB M3 Ultra; smaller hosts retain the 16-slot bounded-memory path.
Inkling remains at 16 because a 24-slot control warmup on the 24 GB M5 entered
memory pressure and regressed sharply. Each layer's slots share one contiguous
wired buffer, and on Qwen a GPU-resident expert-to-slot map lets layers whose
experts are all cached run their routed branch from pre-encoded, GPU-guarded
commands, with no CPU expert planning or fetching; the routed command buffer
also commits eagerly, gated on a shared event that fires as expert fills land.
An explicit `resident` mode maps every routed-expert layer file once. It is the
high-memory Flash-Next default above, but remains opt-in for other families:
on a 24 GiB Qwen 3.6 host it lost to the slot rungs under page-cache pressure.

Qwen 3.6's linear-attention layers replace KV storage entirely: each keeps a
2 MiB delta-rule state and a 3-row convolution tail, updated in place every
token. The gated-DeltaNet kernels are validated against a CPU reference,
including the guarantee that a chunked prefill of T rows matches T sequential
decode steps through the same kernels. Qwen 3.8 and Flash-Next reuse that
exact geometry.

Flash-Next adds the two remaining aperture types. Its sparse attention
indexer scores every past position per query and gathers at most 2,051 KV
rows whatever the context, so decode speed does not fall with context length,
and its n-gram embedding table stays on disk as a BF16 row pool with an LFU
row cache, fetched by 64-bit hash a few KiB per token.

The full design, the memory budget, and the experiment record that shaped the
engine are in [System design](docs/SYSTEM_DESIGN.md) and the
[optimization journey](docs/OPTIMIZATION_JOURNEY.md).

## Roadmap

- **Flash-Next performance:** native chunked prefill is now shipped and reaches
  210 prompt tok/s on the frozen 2,940-token case. Long-prompt routed experts
  use grouped Metal tensor operations, contiguous QSA rows attend as a batch,
  and resident decode addresses routed experts without CPU readback. The next
  work is to move prefill route grouping fully onto the GPU, remove the
  per-layer synchronization it currently requires, and then wire the carried
  MTP sidecar as a runtime drafter. On a 256 GB M3 Ultra,
  the current measured external bar is about 21–25 tok/s ordinary decode,
  29–35 with MTP, and 262–875 tok/s prefill; the 63 tok/s catalog headline
  circulating online is explicitly an estimate, not a submitted run.
- **Hardware-aware defaults:** finish measuring mid-memory Flash-Next cache
  rungs while preserving the 16-slot low-memory path.
- **Product hardening:** keep library switching, OpenAI behavior, installation,
  and the Open WebUI path reproducible across the supported hardware matrix.
- **Longer context and multimodality:** extend the fixed-aperture design before
  adding vision towers or more families. New ports remain explicitly pinned
  and pass the family gate; there is no generic-model fallback.

## Acknowledgments

Mference is heavily inspired by — and its foundation is derived from —
**[TurboFieldfare](https://github.com/drumih/turbo-fieldfare)** by Andrey
Mikhaylov, which pioneered running Gemma 4 26B in ~2 GB on Apple Silicon and
documented over a hundred experiments behind its design. Those derived
portions are licensed under the [Apache License 2.0](LICENSE-APACHE); see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details. New Mference
code is [MIT-licensed](LICENSE).

Model weights remain subject to their own terms: the
[Gemma 4 license](https://ai.google.dev/gemma/apache_2), the
[Qwen license](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/LICENSE)
for Qwen 3.6, Qwen 3.8, and Qwen3.8-Flash-Next, and each pinned repository's
own terms for DeepSeek-V4-Flash and Inkling-Small. MiniCPM5-2B is
[Apache-2.0](https://github.com/OpenBMB/MiniCPM/blob/main/LICENSE) from
OpenBMB. Maple's pinned checkpoint
declares no license; establish the necessary rights before downloading, using,
or redistributing it. Maple's MLX-derived kernel work is covered by
[LICENSE-MLX](LICENSE-MLX); see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
