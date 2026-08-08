# Mference executive wiki

This is the repository-level map for users and agents. It describes the
shipping four-model codebase on the `main` baseline and shows where the clean
Maple port will join it. Maple's installer, schema, kernels, tokenizer,
standalone runtime, and product entry-point wiring are implemented on the clean
integration lineage. The maintained SwiftPM parity harness is implemented, but
real-model product smoke tests and final parity evidence remain open in
[PROGRESS.md](PROGRESS.md), so Maple is not yet claimed as shipping product
support. Reference-branch behavior is not present-tense product evidence. The
frozen Maple contract is [docs/MAPLE.md](docs/MAPLE.md).

## What Mference is

Mference is an architecture-specific Swift and Metal inference engine for
pinned mixture-of-experts (MoE) checkpoints on Apple Silicon. It keeps the
shared model core and inference state in unified memory, while routed experts
live in fixed-stride files and are read from SSD only when a router selects
them. The installed model can therefore be much larger than RAM; the bounded
expert working set, resident core, KV/state cache, and scratch must fit.

The project is deliberately not a generic MLX or llama.cpp wrapper. Each model
family has:

- a pinned source checkpoint and accepted index digest;
- a compile-time `ArchConfig` baseline checked against `manifest.json`;
- explicit tensor-name, quantization, attention, MoE, tokenizer, and chat
  behavior; and
- focused reference, kernel, runtime, installer, and product tests.

The platform floor is Apple Silicon, macOS 15, Swift 6.1, and Metal 3. Metal
shaders compile from source when the model process starts. On supported macOS
26/Apple10 systems the same binary may select Metal 4 tensor-operations prefill;
other hosts use the MSL 3.2 paths. See the [README](README.md) for setup and
[system design](docs/SYSTEM_DESIGN.md) for the original out-of-core design.

## Products and Swift package targets

The product graph is declared in [Package.swift](Package.swift).

| Product or target | Responsibility |
| --- | --- |
| `Mference` | Public inference library: model loading, Metal kernels, generation, KV/state caches, expert streaming, sampling, and tokenization. |
| `MferenceRepack` / `MferenceRepackCore` | Streaming installer, `.gturbo` planner/writer, resumable remote range transfer, and install verification. |
| `MferenceCLI` / `MferenceCLICore` | Raw completion (`--prompt`), templated message input (`--messages-file`), and interactive multi-turn chat (`--chat`). |
| `MferenceMac` | Native SwiftUI model library, installer, chat, composer, transcript, runtime controls, and diagnostics. |
| `MferenceDecodeService` | The Mac app's one-shot model/Metal owner. It communicates with the UI process through the framed local protocol in `MferenceDecodeProtocol`, so the UI never loads a second model. |
| `MferenceServer` / `MferenceServerCore` | Loopback-first OpenAI-compatible Chat Completions server built on SwiftNIO. |
| `MferenceAppCore` | UI-independent app state, installation, inference, context fitting, chat persistence, attachments, and diagnostics. |
| `MferenceMacPresentation` | AppKit/SwiftUI transcript, document extraction, Markdown rendering, theme, and chrome. |
| `MferenceValidationSupport` | CPU references, deterministic fixtures/PRNGs, and numerical tolerances shared by tests. |
| `MferenceMapleParity` / `MferenceMapleParityCore` | Maintained strict Maple teacher-forcing trace export, preflight, and comparison product; see [docs/MAPLE_PARITY.md](docs/MAPLE_PARITY.md). |

Source ownership follows the package graph:

- [Sources/Mference](Sources/Mference) contains the runtime and kernels;
- [Sources/MferenceRepack](Sources/MferenceRepack) contains installation;
- [Sources/MferenceCLI](Sources/MferenceCLI),
  [Sources/MferenceServer](Sources/MferenceServer), and
  [Sources/MferenceApp](Sources/MferenceApp) contain product integrations;
- [Sources/MferenceDecodeProtocol](Sources/MferenceDecodeProtocol) and
  [Sources/MferenceDecodeService](Sources/MferenceDecodeService) isolate the Mac
  app's model process; and
- [Tests](Tests) mirrors those components with focused suites.

## Models, architecture contracts, and the library

The current baseline recognizes four families. Directory names are a
convention; detection uses the directory's manifest.

| Family | High-level execution contract | Installed directory |
| --- | --- | --- |
| Gemma 4 26B-A4B | 30 layers; 25 sliding-window plus 5 full-attention layers; 128 experts, top 8; parallel shared/routed FFNs; tied embedding/head; model-specific norm sandwich and final logit softcap. | `gemma4.gturbo` |
| Qwen 3.6 35B-A3B | 40 layers; 30 Gated-DeltaNet recurrent layers plus 10 gated full-attention layers; 256 experts, top 8; sigmoid-gated shared expert; untied head. | `qwen36.gturbo` |
| DeepSeek-V4-Flash 284B-A13B | 43 MoE layers; sliding attention plus CSA/HCA compressed long-range state; four-stream mHC residual; top 6 routing, including three hash-routed layers; 2-bit routed experts. | `deepseekv4flash.gturbo` |
| Inkling-Small 276B-A12B | 42 layers; learned relative-position attention and short convolutions; 256 experts, top 6; two leading dense layers and two shared experts; padded vocabulary truncated before sampling. | `inklingsmall.gturbo` |
| Maple Preview (clean product wiring implemented; parity pending) | 24 native-BF16 layers; three 512-token partial-RoPE sliding layers followed by one NoPE full layer; 256 ternary experts, top 8; no shared expert; full exact vocabulary head. | `maple.gturbo` |

At load, `ManifestReader` decodes the model family, resolves the corresponding
known `ArchConfig`, and rejects field, file, layout, or quantization mismatches.
The resident tensor file is mapped read-only and wrapped for Metal. Routed
expert streamers open lazily per layer. This is why a renamed valid directory
still works, while an incomplete or incompatible directory does not.

The Mac app discovers models in the configured `Mference.libraryRoot`, the
checkout's `scratch/`, and `~/Library/Application Support/Mference`. It can
adopt another valid `.gturbo` directory. Outside the app, the CLI and server
take an explicit `--model` path; app-family selection can be seeded with the
`MFERENCE_MODEL` environment variable or the `Mference model` user default.

## Installation and the `.gturbo` format

`MferenceRepack` resolves a supported model selector to a repository, revision,
index SHA-256, size estimate, and disk reserve. It downloads metadata first,
classifies only the required text-model tensors, and plans deterministic byte
ranges. Payload ranges stream into their final files through bounded scratch;
the installer does not stage a full Hugging Face snapshot or dequantize and
requantize the normal affine models.

A completed install contains:

```text
<family>.gturbo/
  manifest.json
  verified-install.json
  model_weights.bin
  tokenizer/
  packed_experts/
    layout.json
    layer_00.bin
    ...
```

`model_weights.bin` stores the resident embedding/head, attention projections,
routers, shared experts, norms, and small state tensors. Every routed layer
file contains page-aligned, fixed-stride expert blobs; `layout.json` maps the
gate/up/down weight and companion subregions. The runtime binds subregions of
the cached blob instead of creating a Metal buffer per tensor.

Remote installation is transactional. An advisory lock serializes operations
for one destination. Completed ranges are made durable and recorded with
digests; cancellation preserves the partial directory and checkpoint. `--resume`
revalidates and reuses intact ranges, while `--discard-partial` is the explicit
destructive path. Final metadata and payloads are verified before the partial
directory is atomically promoted.

The strict `full-sha256` runtime policy hashes common files at load and expert
files on first touch. `trusted-receipt` still validates the manifest binding,
file set, sizes, and common hashes, but trusts receipt-bound sizes for the
large expert files; it therefore trades detection of size-preserving expert
corruption for faster first touch. The upstreaming plan also closes a verifier
provenance gap:
source origin must come from a receipt already bound to the same directory,
manifest, and complete file set—not from a manifest assertion alone.

Maple fits this format without a parallel installer. Its clean port adds
generic bounded range transforms: resident ternary codes widen from packed
INT2 to the existing INT4 layout, and row alpha values expand into group-64
affine scale/bias companions; routed experts remain INT2. Transform identity,
source/destination sizes, and output digests become part of resumability. The
approximate FlashHead tensors are excluded.

## Runtime and one-token pipeline

`Model.load` validates metadata and creates a resident mapping.
`ForwardRunnerFactory` keeps the four shipping families on
`RealForwardRunner` and selects the standalone `MapleForwardRunner` for Maple.
Each runner compiles its Metal modules, allocates reusable scratch, creates the
required attention/state managers, and opens expert streamers lazily. A raw
generation request then:

1. embeds the next token into the residual stream;
2. applies each layer's family-specific norm and attention/recurrent path;
3. runs the router and signals the CPU when selected expert IDs are available;
4. computes cached expert hits while `pread` fills misses into reusable slots;
5. reduces routed and shared/dense FFN branches and advances model state;
6. applies the final norm and language-model head; and
7. greedily selects or samples the next token, streams detokenized text, and
   stops on the model tokens, configured strings, or maximum length.

The per-layer expert cache supports 8, 16, 24, or 32 slots and LFU/LRU policy.
The production default is LFU. CLI/server auto-selection uses 32 slots for
Qwen on hosts with at least 16 GiB and 16 otherwise; the Mac app stays at an
explicit memory-first 16. Expert hits are reused, misses use positional reads,
and optional RDADVISE modes remain experimental and off by default.

The generic fast greedy head avoids materializing all logits when the request
is pure greedy. Sampling requests use the full logits path. Maple deliberately
uses only its full, non-approximate 151,936-row head: no existing FP16 fused
head is claimed as equivalent.

## KV, recurrent state, and context

State is model-specific even though the runner presents one continuation
interface:

- Gemma uses separate FP16 K and V buffers. Sliding layers use bounded rings;
  full layers grow with context. K and V stay separate even where their raw
  projection weight is shared because normalization and RoPE diverge.
- Qwen full-attention layers use FP16 KV, while Gated-DeltaNet layers use a
  fixed recurrent delta-rule matrix and a short convolution tail instead of
  per-token KV rows.
- DeepSeek owns sliding, compressed, indexer, and mHC continuation state in its
  dedicated manager rather than the generic KV arrays.
- Inkling combines FP16 local/full KV with relative-position and depthwise
  short-convolution state.
- Maple's implemented projection, normalization, Q/K, decode-attention, router,
  and ternary-expert primitives keep native BF16 boundaries. Attention consumes
  sliding layers in physical 512-row cache order and full layers as a linear
  prefix. Routed experts preserve source-group-128 arithmetic over duplicated
  group-64 companions and support disjoint cache-hit/miss phases without
  changing rank reduction. The standalone runtime owns cache writes, streamed
  expert I/O, reset, and continuation while preserving those layouts; physical
  iteration and rank-reduction order are part of exact parity.

Supported context choices are 4K, 8K, 16K, 32K, and 64K. Reset drops logical
positions and advises unused state pages back to the OS. The server may retain
one verified prefix for continuation; the CLI chat and Mac app otherwise fit
or rebuild context according to their product rules.

## Prefill

Prefill consumes known prompt tokens and produces the next-token seed. The
production configuration is chunked, with allowed sizes from 32 to 4,096.
One-shot CLI `auto` picks the smallest allowed chunk that covers the prompt;
interactive chat resolves auto to 128. Larger chunks reduce expert rereads but
increase scratch and, for ring attention, physical KV capacity.

Gemma and Qwen have family-aware batched attention/MoE paths, including Metal 4
tensor operations where supported. Inkling has its own batched relative-
attention and routed-expert kernels. DeepSeek uses its correctness-first
family path. Chunked Qwen state is tested against sequential decode semantics.

Maple's standalone runtime deliberately implements sequential prefill: it
replays the exact one-token native-BF16 path and emits the full head only for
the final uncached prompt token. The research-only
`MFERENCE_MAPLE_PREFILL=batch` path is not part of the port because its batch
primitives were never brought to the same parity standard. The family-aware
runner factory enforces this path for the CLI, Mac app/decode service, and
server and reports sequential execution with model-native BF16 KV.

## Tokenization, chat templates, and generation

`MFTokenizer` wraps Hugging Face `swift-transformers` and adds bounded
streaming detokenization, stop-token resolution, chat rendering, and
continuation framing. The loaded tokenizer resolves a supported dialect and
products pair it with the manifest's model family:

- Gemma's turn/channel format and tool-call DSL;
- Qwen/Maple ChatML with `<tool_call>` framing;
- DeepSeek's DSML sentence, thought, and tool blocks; or
- Inkling's role/content/end-message format.

The CLI and server render the installed model's own semantics rather than
assuming one universal template. `StructuredAssistantDecoder` separates
visible content, hidden reasoning, and validated tool calls across token and
string boundaries. Tool names and JSON argument objects are checked before an
event is surfaced. The server returns tool calls to the client but never
executes or authorizes them.

Maple's generation suffix opens `<think>` and leaves it live. Its product
integration suppresses that private channel until `</think>`, preserves
template-significant message whitespace, resolves EOS/end-of-turn correctly,
and selects this behavior from explicit model-family metadata—not magic token
IDs or vocabulary size.

Generation defaults are temperature 0.2, Top-K 64, and Top-P 0.95. Temperature
0 is greedy; repetition penalty, seed, stop strings, response limit, and
context limit are shared through `GenerationConfig`. The streaming completion
loop owns stop ordering, detokenizer flush, cancellation, token history, and
resume boundaries so products observe the same generation semantics.

## Native Mac app

The app is a chat shell over a separately launched decode service. Its toolbar
discovers installed families and offers download rows for missing ones. Chats
are stored locally, grouped by recency, can be created/renamed/deleted, and
retain their visible transcript. Each model renders history through its own
template.

The composer accepts locally extracted PDF, DOCX, PPTX, and XLSX text. Draft
attachments are bounded to 750,000 extracted characters in total and visibly
marked when truncated. Before generation, the app tokenizes the rendered
conversation. Older complete turns that no longer fit can be replaced in model
context by a rolling, per-chat local summary while the full transcript stays
visible; hidden attachment copies for summarized turns are then released.

The inspector exposes context length, cache slots/policy, temperature, Top-K,
Top-P, prefill, RDADVISE, and verification policy. It reports live phase,
tokens, decode rate, routed-I/O timing, and decode-service current/peak memory.
Settings that change model/state allocation require reload; request-only
generation controls do not always do so. See
[runtime controls](docs/RUNTIME_CONTROLS.md).

Maple uses the existing descriptor-backed catalog surface with its pinned
repository, revision, source digest, and `maple.gturbo` directory. The decode
service selects the Maple runner from manifest family metadata and returns the
effective sequential-prefill and BF16-KV modes over the existing diagnostics
protocol. The baseline and research branch do not contain a separate richer
multi-family picker implementation, so the clean port does not invent one.

## CLI and local server

The CLI supports a raw prompt, a JSON messages file, or an interactive chat.
It reports prefill and decode timing and can print optional phase diagnostics.
Interactive turns rerender and re-prefill the fitted conversation. Context
fitting may drop the oldest complete messages but never the system message or
current user message.

Raw CLI, chat CLI, and server sessions all construct their producer through the
same family-aware factory. Maple therefore uses sequential prefill, the exact
full head, and BF16 KV without adding family conditionals to the existing
four-model runner; existing families retain their requested chunked/off modes
and FP16 KV diagnostics.

The server provides `GET /health`, `GET /v1/models`, and
`POST /v1/chat/completions`, including blocking JSON and SSE streaming. It
accepts text messages, one choice, generation controls, function schemas, and
one retained prefix cache. Unsupported bodies/parameters fail with OpenAI-style
JSON errors before streaming; post-commit failures use an in-band SSE error and
`[DONE]`. The queue is bounded and generation is serialized.

The default bind is exactly `127.0.0.1`. `--bind tailnet` resolves exactly one
Tailscale IPv4 address and fails closed. The server has no application-level
authentication or TLS: never bind it to a wildcard or expose it via a proxy or
tunnel. Full behavior and client examples are in the
[server guide](docs/OPENAI_SERVER.md).

## Testing, validation, and safety

Run package tests through [Scripts/test.sh](Scripts/test.sh), which enforces a
serial Swift test run and supplies the active developer directory's
`Testing.framework`/interop search paths when Command Line Tools needs them.
The suites cover CPU references, Metal kernels, manifest/layout validation,
installer planning and remote retries/resume, tokenizer/template behavior,
generation/continuation/cancellation, app state and presentation, and server
HTTP/queue/prompt-cache behavior. Real-model suites are environment-gated and
skip unless the matching installed-model path is supplied.

Before any real model run, require macOS 15+, Swift 6.1+, sufficient disk,
acceptable `memory_pressure -Q`, a complete required install, and no match from:

```bash
pgrep -fl 'MferenceServer|MferenceMac|MferenceDecodeService|MferenceCLI|MferencePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'
```

If a check fails, stop and report it. Never terminate an existing model
process, start a second one, delete/reinstall a model, download a full
checkpoint, duplicate a `.gturbo` directory, or purge caches merely to test.
Keep the server session alive while requested and stop only a server started by
the current task.

Performance work must build release once and follow the
[community benchmark protocol](docs/COMMUNITY_BENCHMARKS.md) exactly. Report
commit and dirty state, hardware/RAM, macOS, Swift, exact command, exit code,
complete timing footer/error, and every deviation. Measurements are data points,
not ceilings. Maple parity has its stricter, frozen protocol in
[docs/MAPLE.md](docs/MAPLE.md).

## Documentation map

| Document | Use it for |
| --- | --- |
| [README.md](README.md) | Product overview, requirements, quick start, and published headline results. |
| [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) | Original Gemma-centered storage, memory, KV, and execution design. |
| [docs/RUNTIME_CONTROLS.md](docs/RUNTIME_CONTROLS.md) | App/CLI controls, defaults, reload rules, and metric interpretation. |
| [docs/OPENAI_SERVER.md](docs/OPENAI_SERVER.md) | Server launch, security boundary, API, errors, tools, and prefix reuse. |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | Accepted measurements and their protocols/caveats. |
| [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md) | Reproducible community benchmark procedure. |
| [docs/OPTIMIZATION_JOURNEY.md](docs/OPTIMIZATION_JOURNEY.md) | Successful and rejected performance experiments. |
| [docs/QWEN36_PERFORMANCE.md](docs/QWEN36_PERFORMANCE.md) | Qwen-specific cache, prefill, and decode evidence. |
| [docs/DEEPSEEK_V4_FLASH.md](docs/DEEPSEEK_V4_FLASH.md) | DeepSeek source, architecture, state, memory, and run notes. Read before touching that model. |
| [docs/INKLING_SMALL.md](docs/INKLING_SMALL.md) | Inkling source, architecture, validation, memory, and performance notes. Read before touching that model. |
| [docs/IMPLEMENTATION_REFERENCES.md](docs/IMPLEMENTATION_REFERENCES.md) | Pinned external model, kernel, I/O, and attention references. |
| [docs/MAPLE.md](docs/MAPLE.md) | Maple behavioral contract, reference pins, exclusions, and required validation. |
| [docs/MAPLE_PARITY.md](docs/MAPLE_PARITY.md) | Maintained local-only Maple oracle/candidate parity protocol, trace schema, and acceptance boundary. |
| [PROGRESS.md](PROGRESS.md) | Ordered branch ledger and observable acceptance criteria for the clean Maple port. |
