# Maple upstreaming progress

This file is the durable handoff ledger for the clean Maple reimplementation.
It answers three questions for every branch: what scope belongs there, what an
observer can verify, and whether that verification is complete. Update the
checkbox immediately when a branch meets its acceptance criterion; the
corresponding change and progress update belong in the same logically scoped
commit.

The integration target is `feature/maple-integration`, created from original
`main` at `42f75fa973d5fbdbc308637d5d3d5e5eaf151a12`. Feature branches start from
the integration branch and merge back into it after validation. Nothing in
this project merges into the repository's original `main`. The research branch
`codex/maple-integration` is read-only behavioral/reference material, never a
cherry-pick source.

The required behavior is frozen in [docs/MAPLE.md](docs/MAPLE.md); the existing
system map is [WIKI.md](WIKI.md).

## Completed setup

- [x] **`bugfix/swift-command-line-tools-tests`** — Acceptance: the repository's
  complete [Scripts/test.sh](Scripts/test.sh) run succeeds with the active
  Command Line Tools developer directory, while remaining compatible with a
  full Xcode developer directory. **Status:** complete in `57100b1`; merged to
  `feature/maple-integration` as `9c891d3`. The script supplies
  `Testing.framework` and `_TestingInterop` runtime search paths only when the
  active toolchain provides them. A full serial package test run passed under
  Command Line Tools.

## Ordered implementation checklist

- [x] **`methodology/maple-upstreaming`** — Acceptance: merged agent/git
  instructions preserve repository safety rules; this progress ledger, the
  executive wiki, and the frozen Maple parity contract exist with resolving
  links and explicit current-versus-planned status. **Status:** complete on
  this branch; repository rules are merged, all local Markdown links resolve,
  and the whitespace check passes.

- [x] **`refactor/model-tensor-paths`** — Acceptance: family-specific resident
  tensor paths/accessors are isolated behind one small model abstraction;
  existing four-family behavior and tests remain unchanged; Maple names can be
  added without widening unrelated runtime switches. **Status:** complete;
  the generic attention accessors use the existing family trunk prefix, the
  full 907-test suite passed before merge, and focused post-merge model-loader
  coverage resolves all six accessors from the Gemma fixture.

- [x] **`feature/range-copy-transforms`** — Acceptance: the installer supports
  bounded, deterministic identity/INT2-to-INT4/BF16-repeat transforms; source
  and destination byte counts, transform identity, plan fingerprints, digests,
  cancellation, and resume are covered by synthetic tests; identity-only
  checkpoints retain compatible fingerprints. **Status:** complete; planning
  and streamed execution use checked, bounded transforms, identity-only v1
  hashes are frozen, all 908 pre-merge package tests passed, and the post-merge
  Repack suite passed 80/80 with exact byte, scratch, digest, cancellation, and
  resume coverage.

- [x] **`bugfix/verified-install-provenance`** — Acceptance: `--verify-install`
  never invents repository/revision provenance from `manifest.json`; it
  preserves origin only from an existing receipt bound to the exact directory,
  manifest, and complete file set; forged/stale/missing receipts have tests.
  **Status:** complete; verification retains provenance only from an exactly
  bound existing receipt, never from the manifest snapshot hash. The full
  918-test package suite passed before merge, and post-merge synthetic tests
  cover repeated verification plus missing, stale, moved, partial, malformed,
  oversized, unknown-tool, and file-mismatched receipts.

- [x] **`feature/maple-model-schema`** — Acceptance: explicit `.maple` family,
  pinned architecture, manifest fields, quantization contract, source selector,
  and compile-time cross-check load a valid synthetic install and reject every
  material mismatch. **Status:** complete; both Swift targets built and all 919
  pre-merge package tests passed. Post-merge schema coverage validates the
  complete pinned config mutation boundary, exact five-slot quant contract,
  required manifest fields, I32 FlashHead exclusion, source fingerprint, and
  Maple tensor paths. Current writers publish `numDenseLayers: 0`; the reader
  also preserves verified research-era manifests where that redundant zero was
  omitted, without relaxing nonzero-value or routed-layout checks. The final
  932-test package suite passed in 148 suites.

- [x] **`feature/maple-ternary-repack`** — Acceptance: a synthetic and then real
  pinned Maple snapshot streams into `.gturbo`; resident ternary matrices are
  losslessly represented in the existing INT4/group-64 layout, routed experts
  remain INT2/group-64, row-alpha companions are exact, FlashHead tensors are
  excluded, resume avoids redownloading verified transformed ranges, and
  strict install verification passes. **Status:** complete within repository
  safety constraints. A full 24-layer fake-Hub install matches independent
  resident and routed byte oracles, excludes FlashHead, emits the exact
  manifest, passes strict receipt verification, and resumes without
  redownloading a committed transformed range. The exact pinned local source
  index and every real safetensor header plan successfully. A second 6.1 GB
  `.gturbo` was intentionally not created because `AGENTS.md` forbids
  duplicating an installed model just for tests. The final 938-test package
  suite passed in 148 suites after malformed resident dtype hardening.

- [x] **`feature/maple-tokenization`** — Acceptance: model-family metadata—not
  vocabulary size or hard-coded identifying token IDs—selects Maple ChatML;
  golden tests cover verbatim message whitespace, the live `<think>` generation
  suffix, EOS/end-of-turn/stop tokens, continuation framing, hidden-thought
  suppression, and fail-closed tool behavior. **Status:** complete. The pinned
  compact sidecar drives exact prompt and token-ID fixtures; CLI, Mac app, and
  server hide prompt-opened reasoning, while the server alone accepts bounded,
  duplicate-key-safe Maple JSON tool calls. Existing Qwen ChatML behavior is
  unchanged. The tokenizer wrapper also retains its pre-Maple stored layout and
  callable overloads, preventing stale Swift incremental objects from crashing
  without deleting build caches. The final package run passed 952 tests in 150
  suites (202.276 seconds).

- [x] **`feature/maple-bf16-primitives`** — Acceptance: focused Metal/CPU tests
  cover native-BF16 embedding, ternary projection, residual-add/RMSNorm, final
  norm, and full logits, including values representable in BF16 but not FP16;
  no unused FP16-to-BF16 bridge is added. **Status:** complete. Safe-math Metal
  modules implement native-BF16 embedding, source-group-128 ternary reduction,
  exact INT4/group-64 full-head arithmetic, and residual-carry/final RMSNorm.
  Tests cover odd byte-aligned packed weights, group/code order, nine-row tails,
  offsets and sentinels, and values that underflow FP16. The full package run
  passed 957 tests in 152 suites (200.560 seconds); runtime wiring remains in
  the later ordered branches.

- [x] **`feature/maple-attention`** — Acceptance: per-head Q/K norm, sliding
  partial NeoX RoPE, full-layer NoPE, grouped-query attention, BF16 KV, the
  512-row rotating-cache wrap, and full-prefix attention match the pinned MLX
  reference in focused tests and representative layer traces. **Status:**
  complete at the isolated decode-primitive boundary. A dedicated safe-math
  module implements fixed-shape native-BF16 Q/K RMSNorm, 64-dimension NeoX
  RoPE on sliding layers, NoPE on full layers, and MLX-shaped one-/two-pass
  vector SDPA with explicit physical-ring and linear-prefix APIs. Tests cover
  production head shapes, GQA mapping, offsets and sentinels, BF16-only values,
  a mid-cycle 512-row ring, prefix growth, nonuniform CPU goldens, and the real
  GPU-family dispatch threshold. The full package run passed 962 tests in 154
  suites (186.353 seconds). Runtime/KV ownership and end-to-end layer traces
  remain in the later runtime and parity branches.

- [x] **`feature/maple-routed-moe`** — Acceptance: raw BF16 router/FP32
  accumulation, stable top-8 selection, post-top-k normalization, INT2
  gate/up/down, clipped SwiGLU, FP32 weighted reduction, cache-hit/miss split,
  and SSD-streamed expert order match CPU/reference fixtures. **Status:**
  complete at the isolated router/expert-kernel boundary. A dedicated
  safe-math module implements the fixed 256-way BF16 router, all-expert FP32
  softmax, stable top-8 and renormalization, source-group-128 ternary expert
  arithmetic over the expanded group-64 companions, exact clamp/BF16
  boundaries, and FP32 rank-order reduction. Tests cover production geometry,
  cross-SIMD ties, BF16-only values, all four packed codes, odd byte offsets,
  full versus disjoint hit/miss phases, rank-pair permutation, and reusable
  command-buffer lifetimes; they caught and fixed a shared subset-rank buffer
  race. The full package run passed 964 tests in 155 suites (114.443 seconds).
  Runtime ownership, cache policy, and SSD orchestration remain in
  `feature/maple-runtime`.

- [x] **`feature/maple-runtime`** — Acceptance: the 24-layer forward pass uses
  native BF16 activations/KV, exact sequential prefill, streamed top-8 experts,
  and the complete non-approximate vocabulary head; reset/continuation/context
  behavior is tested; a representative real trace reaches both a 512-token
  sliding-cache wrap and a longer full-attention prefix. **Status:** the clean
  runtime implementation is complete in `61d6a1d`, `a87ed5c`, and `e89a7fb`.
  A standalone `MapleForwardRunner` validates the exact resident/expert layout,
  owns native-BF16 KV and scratch, overlaps cached expert work with positional
  reads, and replays prefill token by token. Follow-up commits `cd354ba` and
  `169cef4` keep the complete layer/KV/expert path on every prompt token while
  skipping final norm and vocabulary-head work until the final uncached prompt
  token; decode steps still use the complete head.
  Synthetic 24-layer coverage exercises all-miss then all-hit execution,
  reset, continuation, forced full logits, malformed metadata, and non-Maple
  factory behavior; focused KV coverage crosses the 512-row sliding wrap while
  a full layer remains linear. Headless reset/resume tests prove that
  intermediate prompt tokens leave logits untouched, the final uncached token
  seeds decode, and non-Maple producers retain scalar replay. The full package
  run at `169cef4` passed 977 tests in 157 suites (117.982 seconds). Product
  entry-point wiring is implemented by the following product branch. The
  clean full-checkpoint trace at `84d7b62` covers 1,639 positions, crosses the
  512-token wrap, grows the full-attention prefix, and matches the pinned MLX
  oracle exactly at every ordered top-10 logit.

- [ ] **`feature/maple-product-integration`** — Acceptance: `maple` installs,
  auto-detects, and runs through Repack, CLI, Mac app/decode service, and server;
  model IDs, directory convention, library rows, native-KV diagnostics, hidden
  thought, stop reasons, streaming, and tool-call surfaces behave observably as
  specified without regressing existing families. **Status:** clean product
  integration is complete in `6e55a75`, `7c06dd0`, and `24d9f30`. The app
  catalog publishes the exact pinned Maple descriptor and `maple.gturbo`
  convention; Repack accepts the selector; and raw/chat CLI, app/decode service,
  and server construct their producer through the family-aware factory. Maple
  reports effective sequential prefill and model-native BF16 KV through the
  decode protocol, while all existing families retain the prior runner,
  prefill, and FP16 behavior. Focused descriptor/location, selector, factory,
  diagnostics, and protocol tests pass; the full package run passed 973 tests
  in 157 suites (113.923 seconds). The baseline and research branch expose
  descriptor-backed app selection, not a separate richer download-row picker,
  so none was invented. This checkbox remains open until the env-gated real
  Maple CLI, app/decode-service, and server smoke workflows run with the parity
  evidence.

- [x] **`feature/maple-parity-harness`** — Acceptance: a maintained harness
  pins and verifies checkpoint/runtime/corpus inputs, exports the complete
  1,639-position full-vocabulary teacher-forcing traces from clean builds,
  compares all ordered top-10 IDs and retained logits at zero tolerance, fails
  on metadata/environment/dirty-tree mismatch, and records reproducible
  metadata without copyrighted or bulky artifacts. It must use a supported
  Swift target/script interface, not brittle manual object-list linking.
  **Status:** complete at clean commit `84d7b62`. The SwiftPM product, 12
  model-free tests, and static script checks pass. A 32-position diagnostic
  produced 32/32 ordered matches and was correctly rejected as ineligible. The
  full-SHA acceptance run produced 1,639/1,639 top-1 and ordered-top-10 matches
  with zero retained-logit difference and comparator exit 0. Full traces stay
  ignored; compact environment, command, hash, timing, and result metadata is
  recorded in [the parity evidence](docs/evidence/maple-parity-2026-08-09.json).

- [ ] **`feature/maple-documentation`** — Acceptance: user-facing README,
  implementation references, licenses/notices, install/run instructions,
  product support, limitations, and accepted validation results describe only
  the clean implementation; [WIKI.md](WIKI.md), this ledger, and
  [docs/MAPLE.md](docs/MAPLE.md) reflect final code and all links resolve.
  **Status:** planned.

## Intentional exclusions and deviations from the research branch

The following are not deferred parity items; they are deliberately absent from
the clean port unless separately designed and validated later:

- `MFERENCE_MAPLE_PREFILL=batch`, `PrefillMaple*` kernels, and their tests;
  Maple prefill is the exact sequential one-token path.
- The unused FP16-to-native-BF16 conversion bridge.
- Approximate FlashHead tensors, code, and `use_flash_head=true`; Maple always
  uses the complete exact vocabulary head.
- The generic fused greedy-row head for Maple. No performance or parity claim
  may imply that path is active.
- Token-ID/vocabulary-size inference of model family or KV diagnostics; family
  and storage mode must be explicit data.
- The research harness's manually assembled Swift object-list executable.
- Copyrighted Raven text, full traces, raw generated text, downloaded oracle
  snapshots, scratch virtual environments, and other bulky research artifacts.
- The dated dirty-worktree benchmark and stale acceptance JSON as evidence for
  this reimplementation. Historical records may guide protocol design only.

A separate benchmark branch is conditional, not currently planned. Add it only
if clean, reproducible performance claims are retained; otherwise keep the
reproduction/measurement plumbing with `feature/maple-parity-harness` and make
no headline claim.

## Completion gate

Before the integration branch is upstream-ready, every checkbox above must be
complete, the clean validation matrix in [docs/MAPLE.md](docs/MAPLE.md) must be
green, all relevant unit/integration/real-model tests and formatting/linting
must pass, and `main..feature/maple-integration` must contain no dead code,
stubs, experimental paths, debug artifacts, research-only files, or accidental
compatibility layers. Review both the full diff and ordered Git history; each
commit must be understandable and limited to one meaningful item.
