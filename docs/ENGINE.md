# MferenceEngine

`MferenceEngine` is the library between the runtime and its clients. It lists
the models the product offers, keeps exactly one of them loaded, and runs
chat requests on it, whatever runs the model underneath: Mference's own
runtime for `.gturbo` installs, or llama.cpp linked in-process for `.gguf`
files. `MferenceServer`'s library mode is moving onto it, and the SwiftUI app
will later use the same API.

```
MferenceServer (OpenAI API)   ·   app (later)
        │                          │
        └────────── MferenceEngine ┘
   EngineCatalog   which models exist, by stable ID
   EngineHost      one resident model, one request at a time
   EngineModel     prepare (render) → generate (stream) → completion
     ├─ NativeBackend   Model + ForwardRunnerFactory + runRawCompletion
     └─ LlamaBackend    libllama (+ llama.cpp's draft-mtp for Deep)
```

This page covers the public API, the identifiers, and the lifecycle. Server
behaviour is in [OPENAI_SERVER.md](OPENAI_SERVER.md).

## Models and identifiers

The product offers six models, and the catalog lists only these by default.

| Slot | ID | Engine | On disk |
| --- | --- | --- | --- |
| Quick | `qwen3.5-4b` | `llama.cpp` | `~/llm-models/gguf/Qwen3.5-4B-UD-Q4_K_XL.gguf` |
| Deep | `qwen3.8-27b` | `llama.cpp` | `~/llm-models/gguf/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` (carries MTP tensors) |
| Qwen 3.6 | `qwen3.6-35b-a3b` | `mference` | `qwen36.gturbo` |
| MiniCPM5 | `minicpm5-2b-int4g64` | `mference` | `minicpm5.gturbo` |
| Maple | `maple-preview-2bit-mlx` | `mference` | `maple.gturbo` |
| Inkling 2 Mini | `inkling-2-mini-ud-q4_k_xl` | `mference` | `inkling2mini.gturbo` (32 GB of memory and up) |

Deep decodes with its MTP head: `LlamaBackend` loads the GGUF's nextn
layer and drafts with llama.cpp's own `draft-mtp` (`common_speculative`, up
to three drafted tokens per step, verified by the target). Greedy output
with MTP on is equivalent to MTP off but not guaranteed byte-identical: the
target verifies the drafted tokens as one batch, which differs numerically
from single-token decode, so a near-tied token can come out differently. It
was token-identical on all 7 engine trace cases. Through the server, 8 of 10
outputs (5 requests, each streamed and not) matched; the other two, one
prompt in both modes, differ from one phrase on. The catalog requires the file's MTP tensors, and the server
serves Deep only with MTP. While MTP is on, each request starts from an empty
context (no prompt-prefix reuse).

`EngineProductSlot` holds this table in code. The IDs are stable: a client
stores them, and the native ones are the IDs `MferenceServer` already used for
those families. Flash-Next and the other installed families are not offered.

The engine name (`mference` or `llama.cpp`, `EngineKind`) is metadata. The UI
shows it only in the model picker, and the API carries it as a plain field.

An `EngineModelDescriptor` is everything known before loading: ID, display
name, engine, format (a `.gturbo` family or GGUF header facts), location, size
on disk, native context, and product slot. `EngineCatalog` is a sorted,
duplicate-free set of descriptors, plus the models it found but declined
(`EngineCatalogSkip`, with a reason). A request for a declined model fails
with that reason instead of "unknown model".

## Requests

A request is the runtime's own chat types (`MFTokenizer.Message`,
`MFTokenizer.FunctionDefinition`, `GenerationConfig`, `QwenReasoningEffort`)
wrapped in `EngineRequest`, so the server's validated request maps onto it
field by field. A loaded model handles it in two steps:

1. `prepare(_:)` renders the prompt with the model's chat template and checks
   it against the context. Anything that can reject a request happens here,
   as `EngineError.invalidRequest` with the server's error codes. Nothing has
   been streamed yet, so a client can still answer with a 400.
2. `generate(_:onEvent:)` streams `EngineEvent`s (`content`, `reasoning`,
   `toolCall`) in order and returns an `EngineCompletion`: the same text and
   calls, the finish reason (`stop`, `length`, `tool_calls`), usage, and
   timings. Cancelling the calling task stops generation at the next token
   with `CancellationError`, and the model stays usable.

`EngineModelTraits` says what the loaded model supports: chat dialect,
sampling defaults, effective context, reasoning-effort support, prefix reuse,
whether draft (MTP) decoding is active, and whether it accepts images. The dialect comes from the loaded
tokenizer, which is why a request is validated after its model is resident.

### Images

`EngineRequest.media` maps the index of a user message to its images, as
`EngineMedia` values: packed 8-bit RGB with a SHA-256 of the size and pixels.
`EngineMedia.image(encoded:)` decodes any format ImageIO reads, applies its
orientation, and caps the longest side at 4,096 px; the vendored llama.cpp
carries no image decoders.

- Only a model whose `traits.supportsVision` is true accepts images: a GGUF
  model that discovery paired with a vision projector (see
  [Discovery](#discovery)). Every other model, and an image on a message that
  is not a user message, fails in `prepare` with `unsupported_content`.
- The prompt is what llama.cpp's server builds at the pin: a message with
  images reaches the chat template as one string, a media marker per image
  and then the text, with no newline beside a marker. `mtmd_tokenize` turns
  each marker into `<|vision_start|>`, the image's tokens and
  `<|vision_end|>`. The marker is random per load, so typed text never
  matches it.
- Prefill follows llama.cpp's `mtmd-helper`: text at one position per token,
  and each image encoded by the projector and decoded as embeddings at its
  M-RoPE positions, after which the position moves on by the image's `n_pos`.
  In `EnginePreparedPrompt.promptTokens` each vision token is
  `EngineMedia.placeholderToken` (-1), so the count is the context the prompt
  fills. Images are never drafted: a request with images on Deep decodes
  plainly, without MTP, and the next request starts from an empty memory.
- Prefix reuse keys the cache by position, with an image's positions keyed
  by its hash, and the cache checks each image's full SHA-256 before it
  reuses through it. A later turn that resends the same images reuses the KV
  through them; a changed image is where reuse stops.
- The projector loads (`mtmd_init_from_file`) on the first request that
  carries an image and stays until the model unloads, so a text-only session
  never holds it. A request without images takes exactly the text-only path,
  with byte-identical output.

## The host

`EngineHost` is an actor that owns the single model slot.

- **One model at a time across all backends.** A request for another model
  first releases the resident one: the host calls its `unload()`, waits for
  it, and drops its reference. Only then does it start the next load, so peak
  memory is one model. The release finishes before the load starts, and the
  observer events show it in that order.
- **One request at a time, first come first served.** Requests, explicit
  loads and unloads, and idle unloads all pass through one FIFO gate, so a
  swap never happens under a running request. `withModel(_:onQueued:_:)` holds the gate
  for its whole body. A client validates against `traits`, prepares, commits
  its response, and generates inside one body. `generate(modelID:...)` and
  `stream(modelID:request:)` are built on it. `maxQueuedRequests` bounds the
  line, and one request more gets `EngineError.queueFull`.
- **Cancellation.** Cancelling the calling task, or `cancelActiveRequest()`,
  stops the running request at the next token. A request cancelled while its
  model loads stops before its body runs. The load itself is never
  interrupted: the old model is already released, and abandoning the new one
  halfway would leave the host empty for nothing. A cancelled waiter leaves
  the line without running. A `stream` request belongs to its iterator: it
  starts when iteration starts and is cancelled when iteration ends early,
  even if the caller keeps the sequence. A stream ends with `.completed` or
  throws; cancelling the iterating task makes it throw `CancellationError`.
- **Load failure** leaves the host empty, with the failure in
  `status.lastLoadFailure`, and rethrows the backend's own error. The previous
  model was already released before the failed load started.
- **Idle unload.** With `idleUnloadAfter` set, the resident model is released
  once nothing has used it for that long. Any request or load re-arms the
  timer, and nothing is armed while the host is empty.
- **Shutdown** fails waiting requests with `EngineError.shuttingDown`, cancels
  the running one, releases the model, and refuses new work. It completes
  even if its caller is cancelled, and a second call waits for the first.

### Status and events

`host.status` can be read synchronously from any thread, such as a NIO
channel callback or a SwiftUI view, while a load runs. It holds:

| Phase | Meaning | `health` |
| --- | --- | --- |
| `empty` | nothing resident | `ok`, no model |
| `loading` | loading, or swapping in, `targetModelID` | `loading`, target |
| `ready` | `residentModelID` is usable | `ok`, resident |
| `unloading` | releasing `targetModelID` (idle, requested, shutdown) | `ok`, no model |

It also counts active and queued requests (a request is active from the moment
it holds the model slot, including a load it triggers), keeps the last load
failure, and says whether shutdown has begun.
`health` is the server's existing `/health` shape.

The `observer` receives `EngineHostEvent`s in order. A swap from A to B is
`unloadStarted(A, .swap(to: B))`, `unloadFinished(A)`,
`loadStarted(B, replacing: A)`, then `loadFinished(B, seconds)` or
`loadFailed(B, message)`. A client logs these lines (the server keeps its
`swap started`, `swap finished` and `swap failed` lines this way), and memory
validation checks them for release before load.

## Backends

An `EngineBackend` loads models of one `EngineKind`. The host keeps one
backend per kind and calls `load` only with no other model resident.
`EngineLoadOptions` carries one set of options for every backend:

- `nativeMaxContext`: the context of `.gturbo` models. It defaults to the
  server's `--max-context` default of 16,384, and nil means each model's
  native maximum (`--max-context max`). A descriptor's `hostContextCap`
  lowers either: Inkling 2 Mini runs at most 16,384 tokens (the tested
  maximum on 32 GB) on a Mac under 48 GB, where its expert residency plan
  stages layers.
- `ggufMaxContext`: the context of GGUF models. Nil means 32,768 tokens, and
  a GGUF never runs past its native context.
- `EngineModelDescriptor.resolvedContext(options:)` gives the value before
  loading, for `/v1/models`. A backend may run with less where memory
  requires, and reports the effective value in `traits.maxContext`.
- `draftDecoding`: MTP is on for a GGUF with an MTP head (Deep) unless this
  is `.off`, which loads and decodes it plainly. `traits.usesDraftDecoding`
  reports which, and `EngineTimings` carries drafted and accepted tokens.
- The native runtime controls already documented in
  [RUNTIME_CONTROLS.md](RUNTIME_CONTROLS.md): prompt-cache mode and bytes,
  integrity policy, shadow-prefetch budget, prefill chunk, reserved KV.

A backend ignores the options that do not apply to it.

`FakeEngineBackend` is a scripted backend shipped in the library for tests of
the engine and of its clients. It loads nothing. A model is a list of events,
with latches to hold a load or a generation at a known point. Its journal
records load, unload, deinit, and generation order, and counts the most
models live and the most generations running at once.

## Embeddings

`LlamaEmbedder` turns text into vectors with a pooled GGUF embedding model
through the same vendored llama.cpp. The Mac app's Find uses it with the text
part of EmbeddingGemma 2 (`gemma-embedding2`, Q8_0, about 310 MB) and, when
photos, video and audio are on, its image and audio encoders (the `mmproj`
GGUF, Q8_0, about 555 MB), which map pictures and sound into the same space
as text.

- It is not an `EngineModel` and the host never sees it: it is a second,
  small llama.cpp model and context beside the one resident model, so a search
  never evicts the model a chat is using.
- The weights are mapped (`use_mmap`) and offloaded to Metal like a GGUF chat
  model (`MFERENCE_LLAMA_GPU_LAYERS` applies). The context has embeddings on,
  the model's own pooling, and no KV cache (non-causal attention).
- `embed(_:)` tokenizes each text with the model's special tokens, cuts it to
  `maxSequenceTokens` (2,048), packs up to `maxSequences` (16) texts into one
  batch as separate sequences, and returns one pooled vector per text, as the
  model writes it (callers truncate and normalise). Calls are synchronous and
  serialised; `unload()` frees everything.
- With `mediaModelPath`, `embed(media:)` takes an image (packed RGB, already
  at `imageSize(width:height:tokens:)`: the budget filled in 48-px tiles, as
  the reference processor sizes it) or up to 30 s of 16 kHz mono audio.
  llama.cpp's multimodal library (`mtmd`, vendored without its decoders; the
  caller decodes) encodes it, and the rows go through the text model framed
  as `<bos> <|image> rows <image|> <eos>` (or the audio markers) in one mixed
  `llama_batch_ext`. The media encoder loads on the first media call.
  `embed(media: [LlamaMediaInput])` embeds many at once: same-size images
  share one vision-encoder pass (`mtmd_batch_encode`, up to 1,024 tokens),
  and up to 16 encoded inputs share one text-model pass, with the same
  vectors as one at a time; an input that fails is nil.
- `LlamaEmbedderTests` covers batch packing and image sizing without a model;
  its real-model cases run only with `MFERENCE_EMBEDDING_GGUF` (and
  `MFERENCE_EMBEDDING_MMPROJ` for media) set, under the machine lock.

## Discovery

`EngineCatalogDiscovery.discover` builds the catalog from what is on disk.
It reads manifests, install receipts, and GGUF headers only, and never opens
a weight file or a tensor.

- **Library roots**, in order: the `Mference.libraryRoot` default (or
  `MFERENCE_LIBRARY_ROOT`), `~/llm-models`, the package checkout's
  `scratch/`, and `~/Library/Application Support/Mference`. These are the
  roots `MferenceServer --library` has always scanned. Each root
  contributes itself and its immediate subdirectories as `.gturbo`
  candidates.
- **`.gturbo` installs** pass `EngineInstallProbe`: a manifest whose
  architecture matches its family's baseline, `packed_experts/layout.json`,
  and an install receipt bound to the manifest. These are the same checks as
  the server's probe. A partial install, or a family with no runner, is
  reported as skipped with the reason.
- **GGUF files** are the `.gguf` files directly in each root and in
  `~/llm-models/gguf` (a symlinked file counts). `EngineGGUFReader` parses
  only the header, through a 1 MiB read window: the
  architecture, `general.name`, `<arch>.context_length`, the tensor count,
  and the `blk.N.nextn.*` tensors of an MTP head. Split GGUFs
  (`-00001-of-0000N`) are skipped.
- **Vision projectors** are the `.gguf` files with `mmproj` in their name.
  They are never listed as models. Each GGUF model is paired with a projector
  in its own directory whose header has a vision encoder and a
  `clip.vision.projection_dim` equal to the model's `embedding_length`; a
  product model prefers its own file names (Quick:
  `Qwen3.5-4B-mmproj-F16.gguf`, then `mmproj-F16.gguf`; Deep:
  `mmproj-Qwen3.8-27B-BF16.gguf`), then path order. The pairing is
  `EngineModelDescriptor.visionProjector`.

### Product allowlist

With the default scope (`EngineCatalogScope.product`), only the six
product models are listed:

- A native product model is the install whose family identifier is the
  slot's ID. When several are found, the one in the earliest root wins,
  then the one labelled `qwen36`, `minicpm5`, `maple`, or `inkling2mini`,
  then path order. The others are skipped, and the skip names the winner.
- A native product model with a memory floor
  (`EngineProductSlot.minimumPhysicalMemory`, 32 GB for Inkling 2 Mini) is
  skipped in either scope on a Mac with less physical memory, with the reason
  under its identifier ("Needs a Mac with 32 GB of memory or more.").
  `discover` takes the memory as a parameter, for tests.
- A product GGUF is recognized by its exact file name from D1. It needs a
  readable header with a context length. Deep also needs its MTP tensors,
  so a renamed non-MTP file is refused.
- Everything else found is skipped with "not in the product model list". A
  request for its ID gets that reason rather than "unknown model".

The developer scope `.all` (the server's `--catalog all`) also lists every
other runnable model. Native installs keep the server's identifier rules:
the family ID, or `<id>@<directory basename>` when the ID is shared or
already taken by a product model, then `#n`. A GGUF that is not a product
file is listed as its lowercased file name without `.gguf`.

## Native backend

`NativeBackend` runs `.gturbo` installs. Its model, `NativeEngineModel`, is
the server's `ServerModelSession` lifted unchanged:

- load: `Model.load`, then `ForwardRunnerFactory.make`
- render with the install's tokenizer
- single- or multi-prefix prompt reuse
- structured decoding and stop strings
- the shared `runRawCompletion` loop

It is an adapter, so its output is byte-identical to the server's. `unload()` drops the model, runner, KV, and prompt caches, so the
memory is released before the host loads the next model. The backend's
operator notices (`--verify` fallback, prompt-cache sheds) reach the
`onNotice` callback for the server's log.
