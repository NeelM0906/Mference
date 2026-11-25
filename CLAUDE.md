# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

[AGENTS.md](AGENTS.md), imported above, is the source of truth for scope,
model-run safety, server binding, benchmark protocol, and run reporting. This
file only adds what it does not cover; do not duplicate its rules here.

## Commands

```bash
swift build -c release                           # all root products
swift build -c release --product MferenceServer  # one product
Scripts/test.sh                                  # full suite, serial
Scripts/test.sh --filter RMSNormTests            # one suite
Scripts/test.sh --filter 'RMSNormTests/rmsNorm_bf16w_d256'   # one test
LANG=en_US.UTF-8 ruby Scripts/check_markdown_links.rb        # root *.md and docs/**
python3 Scripts/check_source_archive.py          # what `git archive HEAD` would ship
./bringup-check.sh --dry-run <family> [gturbo-dir]
```

- Always go through `Scripts/test.sh`, never bare `swift test`: it forces
  `--no-parallel` (shared Metal state makes in-process parallel runs flaky) and
  adds the `Testing.framework` search paths a Command Line Tools install needs.
  Extra arguments pass straight through to `swift test`.
- Production-shape cases marked `.fullTier` (`Tests/Mference/Core/Validation/TestTier.swift`)
  run only with `MFERENCE_TEST_TIER=full`: in the nightly workflow, in release
  validation, and in the full-suite run on each merged `mference-app` tip. Pull
  request CI leaves them out; nothing they assert changes.
- Every test uses Swift Testing (`@Suite` / `@Test`), not XCTest. `--filter` is
  a regex over `Suite/test`; `bringup-check.sh` holds the per-family suite
  regexes (`Qwen38`, `Maple`, `[Mm]iniCPM5`, `Inkling2`, ...) for the six
  native families (Qwen 3.6, Qwen 3.8 27B, Maple, MiniCPM5 2B and 1B,
  Inkling 2 Mini).
- The build, the Swift suite, the link check, and `Scripts/tests/test_*.py`
  (stdlib `unittest`, no dependencies) never download or load a model. CI runs
  the suite, the Python checks and the link check on `macos-26` (the MSL 4.0
  tensor-ops path, and the declared macOS 26 floor) and must stay green; build
  release locally before merging.
- The link checker covers this file too, so keep relative links valid when
  renaming or moving docs.

The Mac app is a separate SwiftPM package in `App/` (see [The Mac app](#the-mac-app-app)):

```bash
swift build --package-path App -j 4               # debug, while iterating
swift build -c release --package-path App -j 4    # the gate build (warnings are errors)
swift test --package-path App -j 4                # model-free: fake engine, no network
App/Scripts/searxng/prepare.sh --offline          # SearXNG runtime from the verified cache
App/Scripts/searxng/prepare.sh --fetch            # by hand only: downloads the pinned runtime
App/Scripts/bundle.sh                             # build/Mference.app, ad hoc signed
```

## Architecture

This section is the architecture overview. The remaining references are
[docs/ENGINE.md](docs/ENGINE.md) (the engine API) and
[docs/FAMILY_CONTRACT.md](docs/FAMILY_CONTRACT.md) (the manifest axes). The
design specs, experiment record, release records and per-family dossiers were
removed on 2026-11-19 to keep the docs lean; read them from git history
(`mference-app` before that date) when you need the background. Retired
families are listed in the README.

### Targets

| Target | Path | Notes |
| --- | --- | --- |
| `Mference` | `Sources/Mference` | Runtime, kernels, tokenization. The `.metal` sources ship as bundle resources. |
| `MferenceRepackCore` → `MferenceRepack` | `Sources/MferenceRepack/{Core,Command}` | Installer. Deliberately does **not** depend on `Mference`. |
| `MferenceCLICore` → `MferenceCLI` | `Sources/MferenceCLI` | `Args.parse` → `run(args:)`. |
| `MferenceServerCore` → `MferenceServer` | `Sources/MferenceServer/{Core,Command}` | swift-nio HTTP, pinned to an exact version. |
| `MferenceEngine` | `Sources/MferenceEngine` | Embeddable engine: `EngineCatalog`, `EngineHost` (one resident model), `NativeBackend`, `LlamaBackend`, `FakeEngineBackend` for tests. See [docs/ENGINE.md](docs/ENGINE.md). |
| `LlamaCpp` | `Sources/LlamaCpp` | Vendored llama.cpp built in-process for the GGUF models (Quick, Deep with MTP). |
| `MferenceValidationSupport` | `Sources/MferenceValidation/Support` | Test-only: CPU references, `Tolerance`, seeded PRNG. |
| `Vendor/swift-jinja` | local package | Patched so canonical chat templates render byte-exact; its upstream tests run as `JinjaCompatibilityTests`. |

Each executable is a thin `Command/main.swift` over a `*Core` library, and the
tests import the `*Core` targets.

Because the installer is independent of the runtime, its `Int4AffineEncoder` /
`Int8AffineEncoder` are bit-exact twins of the runtime's `Quantization`
reference. `Int4AffineEncoderParityTests` and the streaming parity suites lock
the pair together; change one side and the other must change with it.

### Install format and load

`MferenceRepack` turns a pinned Hugging Face revision
(`SupportedModelSource`: repo, commit, index SHA-256) into a `.gturbo`
directory: `manifest.json`, `verified-install.json`, one `model_weights.bin` of
shared tensors, and `packed_experts/layer_NN.bin` files of fixed-stride,
page-aligned routed-expert blobs. Planners in `Core/Planning` build a canonical
byte-range plan; `RemoteStreamingRepacker` copies bounded ranges through
`BoundedScratch` with a resumable checkpoint and an install lock. A source is
either `.preQuantized` (re-layout only, quantized bytes copied unchanged) or
`.originalRepoQuantize` (vendor BF16 quantized in flight).

`Model.load` (`Runtime/Inference/Model.swift`) validates the manifest through
`ManifestReader`, maps `model_weights.bin` read-only, and wraps it in
`MTLBuffer`s without copying. Routed experts open lazily per layer behind
`ExpertStreamingMode`: `PreadExpertStreamer` (a slot slab filled by `pread`,
LFU eviction) or `ResidentExpertStreamer` (whole layer mapped).

### The family contract

`ArchConfig` in `Infrastructure/ModelIO/ModelTypes.swift` is the declarative
contract. Every family has a compiled baseline in
`ArchConfig.knownArchitectures`; `manifest.json -> arch` must match it field by
field or load throws `ModelError.archMismatch`. The `ArchConfig.init` default
values are the original install format's (first Gemma 4's, a family since
retired), and they stay as they are: manifest keys that are absent decode to
them, and kept installs must decode unchanged. A new axis must default to the
behavior existing installs assume.

Adding an axis is a fixed chain, and the suite fails until it is complete: the
stored property with a doc comment → its row in `docs/FAMILY_CONTRACT.md`
(`FamilyContractDocTests` reflects over `ArchConfig` and fails on an
undocumented field) → a baseline assertion in `ModelTypesTests` → the mirror in
the repacker's `ArchInfo`, emitted by `GTurboJSON` and checked in
`ManifestReader.validateArch`.

A baseline means the install validates, not that it can run:
`ManifestReader.familiesWithoutRunner` is the separate capability gate that
refuses a family by name until its runner lands. Family-specific tensor names
(`trunkPrefix`, `lm_head`, embedding) are resolved in `Model`'s accessors.

### Runners and generation

`ForwardRunnerFactory.make` picks the runner from `model.config.family`.
`RealForwardRunner` is the shared one, and Qwen 3.6 is its remaining family.
Maple, Qwen 3.8, MiniCPM5 (both sizes) and Inkling 2 Mini each have a
dedicated `*ForwardRunner`. All of them conform to `LogitProducer` /
`ContinuableLogitProducer` (`Runtime/Generation/LogitProducer.swift`), and
optional abilities are further protocols there (`ChunkedPrefillRunner`,
`FusedHeadLogitProducer`, ...). `runRawCompletion` is
the outer prefill → decode → sample loop shared by the CLI and the server;
tests drive it with `ScriptedLogitProducer` instead of a model.

Decode is one token per step, and each MoE layer is a GPU → CPU → GPU handoff:
`cb1` runs attention and the router, the CPU reads back the chosen expert IDs
and `pread`s only the misses (`io`) while the resident shared expert runs,
then `cb2` finishes the routed branch and the layer tail. Prefill is
layer-major over bounded chunks. KV state is in `Runtime/KVCache`: FP16
sliding rings plus linear full-attention storage, per-family recurrent state
managers, and the paged-to-SSD tier for long contexts.

`RuntimeConfiguration` is the typed set of public runtime controls; CLI `Args`
and `ServerArguments` map flags onto it. Keep public controls limited to those
in [docs/RUNTIME_CONTROLS.md](docs/RUNTIME_CONTROLS.md). The `MFERENCE_*`
environment variables read in `Sources/` are diagnostic and experiment toggles,
not supported controls.

### Kernels and Metal

`Kernels/<Area>/Foo.swift` pairs with `Metal/<Area>/foo.metal`: the Swift class
builds pipeline states with `MetalContext.pipeline(name, constants:)` and
exposes `encode...(commandBuffer:)` methods, and its test under
`Tests/Mference/Core/Kernels/<Area>/` compares the kernel with a CPU reference
from `MferenceValidationSupport` within a `Tolerance` bound.

`MetalContext` is the single owner of the device, queue, and library. It
concatenates the modules listed in `shaderModules` into **one** `MTLLibrary`
compiled at startup, so shader function names share one namespace and a new
`.metal` file must be registered in both `shaderModules` and
`shaderSubdirectories`. Modules outside that list (`tensorops`, `maple_*`)
compile on demand through `MetalContext.moduleLibrary`.

### Server

`HTTPServer.swift` (NIO) hands requests to the `ServerCoordinator` actor, which
runs one generation at a time behind a bounded queue and renders prompts
outside that gate. `ServerModelSession` owns the loaded model and its reusable
prompt-prefix KV. In `--library` mode `ServerLibraryDiscovery` indexes installs
by manifest and the `ServerModelLibrary` actor swaps the resident model,
releasing the old one before loading the next so peak memory stays at one
model. The chat dialect comes from the loaded tokenizer's special tokens
(`ChatDialect`), not from the family. Server tests use `NIOEmbedded` and
injected loaders.

## The Mac app (`App/`)

`App/Package.swift` (tools 6.2, macOS 26) links the engine in-process through
`.package(path: "..")`; the app never edits the engine package (root
`Sources/`, `Tests/`, `Package.swift`). [App/CONTRACTS.md](App/CONTRACTS.md)
lists the shared contracts (design tokens, `EngineServicing`, the SwiftData
schema, `AppSection`, `PermissionBroker`); changing one is its own PR.

| Target | Path | Notes |
| --- | --- | --- |
| `MferenceDesign` | `App/Sources/MferenceDesign` | Tokens (`MFColor`, `MFFont`, `MFSpacing`, `MFRadius`, `MFLineSpacing`) and components. Views use tokens only. |
| `MferenceAppCore` | `App/Sources/MferenceAppCore` | No SwiftUI views. `Engine/` (`EngineServicing`, `EngineService` over `EngineHost`), `Services/AppServices` (the one composition root), `Models/` (library, downloads, memory fit), `Chat/`, `Web/` (bundled SearXNG, tools, citations), `Apple/` (Vision, EventKit, Mail, FoundationModels), `Find/` (content search: FTS5 + vector index, extractors, photo/video/audio sampling, crawler, searcher), `Persistence/` (SwiftData), `Permissions/`, `Memory/`, `Settings/`. Only `Engine/` imports `MferenceEngine` or `Mference`. |
| `MferenceAppUI` | `App/Sources/MferenceAppUI` | SwiftUI screens, one folder per mockup board; `Shell/` routes `AppSection`s; `Adapters/` (`ScreenModels.live`) maps core services to view-facing protocols. |
| `MferenceMac` | `App/Sources/MferenceMac` | `@main`: sets `AGX_RELAX_CDM_CTXSTORE_TIMEOUT` before any Metal device exists; `AppRuntime` must build exactly one `AppServices.live` and shut it down in order on quit (web first, then the engine). |
| `MferenceSmoke` | `App/Sources/MferenceSmoke` | Real-model smoke through the app's engine path; refuses to run without `MFERENCE_APP_SMOKE=1`. |

- One `AppServices` per process, so one `EngineHost` and one resident model;
  never construct a second `EngineService.live()`. Find's small search model
  (`LlamaEmbedder`, EmbeddingGemma 2 text) is the one exception: a separate
  llama.cpp context, freed when Find is idle (App/CONTRACTS.md).
- `App/Scripts/bundle.sh` assembles `build/Mference.app` (bundle ID
  `ai.mference.app`, no App Sandbox, ad hoc signed), copies the
  `Mference_Mference` and `Mference_LlamaCpp` resource bundles and the SearXNG
  runtime, and verifies the bundle. `prepare.sh` never downloads unless run
  with `--fetch`; every download is pinned with its SHA-256 in
  `App/Resources/SearXNG/SOURCES.md` or `App/Resources/Fonts/SOURCES.md`.
- Tests are Swift Testing, model-free: `EngineService` over an `EngineHost`
  with `FakeEngineBackend`, `ScriptedEngineService`, `AppleServices.preview`,
  synthetic Mail/Calendar fixtures, loopback servers. Live suites
  (`MFERENCE_EVENTKIT_LIVE`, `MFERENCE_MAIL_LIVE`, `MFERENCE_SEARXNG_DIR`,
  `MFERENCE_SCAN_ACCURACY_*`) are env-gated and skipped by default.

### App work never loads a model

Engine work needs the models and the machine for long, frequent model runs,
and app work must never gate it. So for all frontend, UI and application work
(development, local testing, iteration, PR gates):

- Use the fake engine only (`FakeEngineBackend`, `ScriptedEngineService`,
  `AppleServices.preview`, fixtures). Never load a real model: no
  `EngineService.live()` generation, no `MferenceSmoke`, no `MferenceCLI` or
  `MferenceServer` runs, no Apple on-device model runs, no chats in the bundled
  app.
- Never take the machine lock for app work; app builds and model-free tests
  run unlocked in their own worktree with `-j 4`.
- Real-model app checks happen only at program checkpoints, scheduled around
  engine work, and then follow the model-run rules in
  [AGENTS.md](AGENTS.md) (one model process, AC power, swap guard).

## Invariants

- **App work never loads a model.** See
  [App work never loads a model](#app-work-never-loads-a-model).
- **Bounded memory.** Never put a whole checkpoint, shard, expert pool, or
  large tensor in Swift heap memory. Weights are mapped and wrapped with
  `makeBuffer(bytesNoCopy:)`; installer copies go through bounded scratch.
- **macOS 26 / Swift 6.2 floor** (`Package.swift`: tools 6.2, `.macOS(.v26)`,
  `.iOS(.v26)`). Shaders always compile at MSL 4.0
  (`MetalContext.shaderLanguageVersion`). Metal 4 tensor-ops code stays behind
  `__HAVE_TENSOR__` in shaders with a working fallback, since not every GPU
  builds the tensor pipelines; Swift asks `supportsApple10TensorOps` at run
  time. Do not reintroduce older-OS availability branches or SDK shims.
- **Exact vs approximate.** An exact optimization must keep output
  byte-identical to its control. Approximate features (Maple FlashHead, the
  DFlash2 drafter) are opt-in flags and never change a default path.
- **Slot ownership.** A queued GPU command owns its expert slot and scratch
  until it completes; the CPU must not reuse either sooner.
- **Shared test fixtures.** A test that touches `FakeHFURLProtocol` statics
  must be an `extension RemotePayloadCopyTests`, never a new `@Suite`, or it
  races the serialized remote-install suite.
- **Tests with behavior changes.** Add or update a focused test; a new kernel
  gets a CPU-reference test and, where available, committed goldens
  (`Tests/Mference/Fixtures`, generators in `Scripts/parity/`).

## Repository notes

- `scratch/` is gitignored and is where local installs
  (`scratch/<family>.gturbo`), parity venvs, and toy checkpoints live.
- `App/.build`, `App/build/` and the generated SearXNG runtime
  (`App/Resources/SearXNG/runtime`) are ignored. The root `.gitignore` drops
  any `models/` directory, so `App/.gitignore` re-includes the app's `Models`
  source folders.
- `.gitignore` also drops `*_SPEC.md`, `*_NOTES.md`, `*_PLAN.md`, `PHASE*.md`,
  `notes/`, `*.bin`, and `*.pdf`; a new doc with one of those names will not be
  tracked.
- `site/` is the static GitHub Pages site, published by
  `.github/workflows/pages.yml` when `site/**` changes on `main`.
- A family PR must pass [docs/FAMILY_GATE.md](docs/FAMILY_GATE.md) (three
  consecutive green suite runs, release build, static checks, pinned install,
  protocol benchmark page), and pull requests follow
  `.github/PULL_REQUEST_TEMPLATE.md`, including its memory and performance
  section.
