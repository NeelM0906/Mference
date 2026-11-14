# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

[AGENTS.md](AGENTS.md), imported above, is the source of truth for scope,
model-run safety, server binding, benchmark protocol, and run reporting. This
file only adds what it does not cover; do not duplicate its rules here.

## Commands

```bash
swift build -c release                           # all four products
swift build -c release --product MferenceServer  # one product
Scripts/test.sh                                  # full suite, serial
Scripts/test.sh --filter RMSNormTests            # one suite
Scripts/test.sh --filter 'RMSNormTests/rmsNorm_bf16w_d256'   # one test
LANG=en_US.UTF-8 ruby Scripts/check_markdown_links.rb        # root *.md and docs/**
python3 Scripts/check_source_archive.py          # what `git archive HEAD` would ship
bash -n mference-ui.sh && python3 Scripts/tests/test_mference_ui.py
./mference-ui.sh doctor                          # prerequisites only; downloads nothing
./bringup-check.sh --dry-run <family> [gturbo-dir]
```

- Always go through `Scripts/test.sh`, never bare `swift test`: it forces
  `--no-parallel` (shared Metal state makes in-process parallel runs flaky) and
  adds the `Testing.framework` search paths a Command Line Tools install needs.
  Extra arguments pass straight through to `swift test`.
- Every test uses Swift Testing (`@Suite` / `@Test`), not XCTest. `--filter` is
  a regex over `Suite/test`; `bringup-check.sh` holds the per-family suite
  regexes (`Qwen38`, `FlashNext`, `Glm53`, `Inkling`, ...).
- The build, the Swift suite, the link check, and `Scripts/tests/test_*.py`
  (stdlib `unittest`, no dependencies) never download or load a model. CI runs
  all of them on `macos-15` (the Swift 6.1 / MSL 3.2 floor) and `macos-26` (the
  MSL 4.0 tensor-ops path); both legs must stay green.
- The link checker covers this file too, so keep relative links valid when
  renaming or moving docs.

## Architecture

Read [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) first (Gemma 4 is its
running example), then [docs/FAMILY_CONTRACT.md](docs/FAMILY_CONTRACT.md). Each
family has a dossier under `docs/families/` or `docs/`, and the design specs
behind the larger ports are in `docs/superpowers/specs/`.

### Targets

| Target | Path | Notes |
| --- | --- | --- |
| `Mference` | `Sources/Mference` | Runtime, kernels, tokenization. The `.metal` sources ship as bundle resources. |
| `MferenceRepackCore` → `MferenceRepack` | `Sources/MferenceRepack/{Core,Command}` | Installer. Deliberately does **not** depend on `Mference`. |
| `MferenceCLICore` → `MferenceCLI` | `Sources/MferenceCLI` | `Args.parse` → `run(args:)`. |
| `MferenceServerCore` → `MferenceServer` | `Sources/MferenceServer/{Core,Command}` | swift-nio HTTP, pinned to an exact version. |
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
field or load throws `ModelError.archMismatch`. Gemma 4's values are the
`ArchConfig.init` defaults, so a new axis must default to Gemma 4 behavior.

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
`RealForwardRunner` is the shared one (Gemma 4 and its QAT install, Qwen 3.6,
DeepSeek-V4-Flash, Inkling-Small, with family logic in extensions such as
`RealForwardRunner+DSV4Prefill` and `Model+Inkling`). Maple, Qwen 3.8, Flint,
MiniCPM5, GLM-5.3-Flash, and Flash-Next each have a dedicated
`*ForwardRunner`. All of them conform to `LogitProducer` /
`ContinuableLogitProducer` (`Runtime/Generation/LogitProducer.swift`), and
optional abilities are further protocols there (`ChunkedPrefillRunner`,
`FusedHeadLogitProducer`, `GemmaPrefixRecovering`, ...). `runRawCompletion` is
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
`shaderSubdirectories`. Modules outside that list (`tensorops`, `maple_*`,
`gemma_qat_*`) compile on demand through `MetalContext.moduleLibrary`.

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

## Invariants

- **Bounded memory.** Never put a whole checkpoint, shard, expert pool, or
  large tensor in Swift heap memory. Weights are mapped and wrapped with
  `makeBuffer(bytesNoCopy:)`; installer copies go through bounded scratch.
- **macOS 15 / Swift 6.1 floor.** Metal 4 tensor-ops code stays behind
  `__HAVE_TENSOR__` in shaders and the `MetalSDKCompatibility.swift` shims in
  Swift, with a working fallback. MSL 4.0 is selected only under
  `#available(macOS 26)`: handing it to the macOS 15 compiler fails the whole
  shader library.
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
