<p align="center">
  <a href="https://neelm0906.github.io/Mference/"><img src="site/assets/icon.png" alt="Mference logo" width="140"></a>
</p>

**Mference** is a private AI assistant that runs entirely on your Mac. It is a
native SwiftUI app built on its own inference engine. Models run on the Mac's
GPU, and your chats, files, mail and calendar never leave the machine.

The engine, `MferenceEngine`, serves every model through one API, with two
backends underneath:

* **Mference's native Swift and Metal runtime** for pinned `.gturbo` installs.
  Mixture-of-experts models keep their shared weights and KV cache in memory
  and read only the experts each token needs from the SSD. This is how
  Qwen 3.6 35B-A3B (a 19.6 GB install) runs in about 1.5 GB of memory.
* **llama.cpp, linked in-process,** for GGUF files.

The same engine also powers an OpenAI-compatible local server and a
command-line tool.

Website: **https://neelm0906.github.io/Mference/**

## Status

Mference is **pre-release and under active development**. There is no signed
or notarized download yet: you build the app from source, and it is signed
with your local identity or ad hoc. Work lands on the `mference-app` branch
first and reaches `main` when the maintainer merges it. The native engine
work (running the product models without llama.cpp) is on `mference-engine`.

## The Mac app

* **Chat** with the model of your choice. Answers can draw on a web search
  through a bundled, local [SearXNG](https://github.com/searxng/searxng), with
  numbered citations. An offline mode turns all network use off.
* **Models:** a model library that downloads the product models, shows how much
  memory each one needs on your Mac, and keeps one model loaded at a time.
* **Scan** receipts and documents with on-device Vision.
* **Calendar and Reminders** through EventKit: propose, create and edit events.
* **Mail:** read Apple Mail and draft replies. The app never sends anything
  without your click.

Notes, Errands and the iPhone app are planned and not built yet.

## Models

The product offers five models. Speeds are indicative readings from one
32 GB M1 Max (512-token prompt), not protocol benchmarks. Expect different
numbers on your Mac.

| Slot | Model | Engine | On disk | Prefill / decode (tok/s) | Peak memory |
| --- | --- | --- | ---: | --- | ---: |
| Quick | [Qwen3.5-4B](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF) UD-Q4_K_XL | llama.cpp | 2.9 GB | 599 / 50 | 1.3 GB |
| Deep | [Qwen3.8-27B GSQ-RCO](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) IQ3_S, MTP | llama.cpp | 12.1 GB | 85 / 17 | 3.1 GB + 12.1 GB mapped |
| | [Qwen 3.6 35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (MoE) | Mference | 19.6 GB | 258 / 29 | 8.3 GB |
| | [MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) | Mference | 1.4 GB | 422 / 72 | 0.8 GB |
| | [Maple Preview](https://huggingface.co/deepgrove/maple-preview-2bit-mlx) 20B-A1B (MoE) | Mference | 6.6 GB | 44 / 33 | 0.7 GB |

Qwen 3.6's peak reflects the automatic 96-slot expert cache on a 24 GiB+ Mac.
In its 16-slot memory-first setting it runs in about 1.45 GB. Deep decodes
with its multi-token-prediction (MTP) head. Its output matches plain decoding
in substance but is not guaranteed byte-identical. Qwen 3.8 27B (dense, 4-bit
MLX) also runs natively, but only behind the server's `--catalog all`.

Nine earlier families were retired on 2026-11-18: Gemma 4 and its QAT
install, DeepSeek-V4-Flash, Inkling-Small, GLM-5.3-Flash, Flint,
Qwen3.8-Flash-Next and its Unsloth GGUF, and Swift-Qwen3.8. Their code and
dossiers are in git history at `mference-app` commit `8ff588c`. This runtime
no longer loads their installs.

## Requirements

* An Apple Silicon Mac with macOS 26 or later.
* Xcode 26 or the matching Command Line Tools (Swift 6.2 or later).
* Enough free disk for the models you want (table above), and a fast internal
  SSD for the MoE models.

## Quick start

### The app

```sh
git clone https://github.com/NeelM0906/Mference.git
cd Mference
App/Scripts/searxng/prepare.sh --fetch   # once: downloads and verifies the pinned SearXNG runtime
App/Scripts/run.sh                       # builds, bundles and opens App/build/Mference.app
```

On first run the app walks you through picking a model. Quick and Deep
download from Hugging Face into `~/llm-models/gguf/`. For the native models,
install them with `MferenceRepack` (below); the app finds them in the library
folders. Calendar, Reminders, Mail and Full Disk Access are asked for only when
you first use the feature that needs them.

### The engine, server and CLI

```sh
swift build -c release

# Install a native model. This streams and repacks the pinned checkpoint,
# never holding the whole thing in memory or on disk.
swift run -c release MferenceRepack --model qwen36 --output scratch/qwen36.gturbo
#   other labels: minicpm5, maple, qwen38; resume an interrupted install with --resume

# Chat in the terminal
swift run -c release MferenceCLI --model scratch/qwen36.gturbo --chat

# Serve every installed product model (native and GGUF) on 127.0.0.1:8080
.build/release/MferenceServer --library --port 8080
```

The server speaks OpenAI Chat Completions, including streaming, tool calls and
prompt-prefix reuse. In library mode it swaps the resident model in place when
a request names another one. It binds to loopback by default and has no
authentication or TLS, so do not expose it. Run one model-owning process at a
time.

## How it works

* **Installer.** `MferenceRepack` reads bounded byte ranges from a pinned
  Hugging Face revision and writes a `.gturbo` install. Shared tensors go into
  one memory-mapped file, and each layer's routed experts go into fixed-stride,
  page-aligned blobs, so one expert is one read. Vendor BF16 checkpoints are
  quantized in flight.
* **Native runtime.** Shared weights stay mapped read-only. Each MoE layer
  keeps an LFU cache of expert slots, and the runtime `pread`s only the
  experts the router picks for the current token. The automatic setting sizes
  the cache to the Mac's memory.
* **Engine.** `EngineCatalog` lists the models it finds, by stable ID.
  `EngineHost` keeps exactly one loaded and releases it before loading the
  next, so peak memory is one model. The app and the server both run on it.
* **Correctness.** The installer's quantizer is bit-identical to the
  runtime's reference. Every kernel is tested against a CPU reference.
  Optimizations that should be exact must keep output byte-identical to their
  control, and approximate features stay opt-in.

## Repository layout

| Path | What it is |
| --- | --- |
| `Sources/Mference` | Native runtime, Metal kernels, tokenizers |
| `Sources/MferenceEngine` | The engine API over the native and llama.cpp backends |
| `Sources/LlamaCpp` | Vendored llama.cpp and ggml |
| `Sources/MferenceRepack` | Streaming installer |
| `Sources/MferenceServer`, `Sources/MferenceCLI` | OpenAI-compatible server and CLI |
| `App/` | The SwiftUI Mac app (its own Swift package) |
| `Tests/`, `App/Tests/` | Swift Testing suites; none of them downloads or loads a model |

## Documentation

* [Engine API](docs/ENGINE.md): models, identifiers, request lifecycle.
* [OpenAI-compatible server](docs/OPENAI_SERVER.md): launch, library mode,
  API behavior, errors.
* [Runtime controls](docs/RUNTIME_CONTROLS.md): CLI and server flags.
* [Community benchmark protocol](docs/COMMUNITY_BENCHMARKS.md): how to measure
  your Mac (`./run-benchmark.sh`).
* [Family contract](docs/FAMILY_CONTRACT.md) and
  [family gate](docs/FAMILY_GATE.md): what a new model family must provide.
* [CLAUDE.md](CLAUDE.md): architecture notes, commands and invariants for
  contributors.

Read [CONTRIBUTING.md](CONTRIBUTING.md) before sending a pull request, and
[SECURITY.md](SECURITY.md) to report a security issue.

## License

New Mference code is [MIT-licensed](LICENSE). Mference's foundation is derived
from [TurboFieldfare](https://github.com/drumih/turbo-fieldfare) by Andrey
Mikhaylov, which first ran Gemma 4 26B in ~2 GB on Apple Silicon. Those
portions are under the [Apache License 2.0](LICENSE-APACHE). Maple's
MLX-derived kernel work is covered by [LICENSE-MLX](LICENSE-MLX), and the
vendored llama.cpp and ggml sources by [LICENSE-GGML](LICENSE-GGML). The app
bundles SearXNG (AGPL-3.0, shipped with its source) and the Geist font (OFL).
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
`App/Resources/*/SOURCES.md`.

Model weights are not part of this repository and keep their own terms. Quick,
Deep and MiniCPM5-2B are published under Apache-2.0, and Qwen 3.6 and Qwen 3.8
under the
[Qwen license](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/LICENSE).
Maple's pinned checkpoint declares no license, so establish the necessary
rights before you download, use or redistribute it.
