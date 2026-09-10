# Family Bring-Up Kit — Phase A design

**Date:** 2026-08-08
**Status:** Phase A in progress (2026-08-31): W1.1 (FAMILY_CONTRACT.md +
rot-check test), W3.3 (`bringup-check.sh`, PASS on qwen36 end-to-end), and
W3.4 (`docs/families/TEMPLATE.md`) landed. W4's rehearsal target changed
from LFM2.5-8B-A1B to **Qwen3.8-Flash-Next** by owner decision — a real
flagship drop four days old, which also forces W2 (no faithful community
conversion exists). Day-0 dossier: [docs/families/QWEN38_FLASH_NEXT.md](../../families/QWEN38_FLASH_NEXT.md).
W2 deliverables 1-4 landed for `qwen38flashnext` (quantize-in-flight core,
original-repo source kind, sharded BF16 reading, fused-expert split, PLE
row-lookup pool, sidecar policy) with W2.1a bit parity enforced. **W2.1b split
in two on 2026-09-02, and BOTH HALVES NOW PASS**
([docs/QUANTIZER_QUALITY.md](../../QUANTIZER_QUALITY.md)). The **weight-level**
half: against mlx-community's independent conversion of Qwen 3.6, our quantizer
is at least as faithful to the BF16 source (0.09612 mean relative error vs
0.09648, better on 118 of 124 sampled INT4 tensors; and better on 10 of 10
INT8 router tensors at 0.00763 vs 0.01068). The **model-level KLD** half was
unblocked the same day by building the per-tensor bit policy the scope cut had
deferred, then run: against a noise floor of exactly zero, the two installs
agree on 86.3 % of top-1 tokens over 882 teacher-forced positions at a median
KL of 0.036 nats, with every greedy divergence landing at a control-side top-2
margin below the median inter-install logit difference. Clearing the blocker
also surfaced **two defects nothing else in the kit could see** — resident
tensor naming, and a missing RMSNorm `1 + w` fold that made every norm wrong by
one in an install that still verified, validated and loaded. See also
[docs/families/QWEN38_FLASH_NEXT.md](../../families/QWEN38_FLASH_NEXT.md).
Still open: W1.2 mapping-as-data (first data file written, repacker does not
read it yet), W1.3 capability gate as a general mechanism (a named per-family
refusal exists in `ManifestReader.familiesWithoutRunner`), W3.1 toy
generator, W3.2 parity harness. **W2.1b is closed** (both halves, 2026-09-02).
**Strategic context:** Mference's trajectory is "the standard way to run
flagship MoE models on the Macs people own." That requires supporting each
major MoE release within about a week of its drop. Today a family port is
a multi-week hand enumeration; this phase makes bring-up cheap so the
release-calendar strategy (Phase B) becomes executable.

## Goal and the one metric that matters

**A new MoE family reaches protocol-benchmarked, ladder-enabled support in
≤5 working days**, proven by a timed rehearsal bring-up (Workstream 4).
"Support" means: repacked install from the model's original weights, byte-
stable decode with parity against a reference implementation, the full
residency ladder and instrumentation working untouched, and a community-
protocol benchmark page.

## What already exists (the kit is 60% latent in the codebase)

- **`ArchConfig` is a 44-axis declarative contract** (`ModelTypes.swift:174`):
  layer-type mask, linear/compressed attention configs, hash routing,
  router scoring functions, sandwich norms, shared-expert gating, sconv
  kernel size, hyper-connections, swiglu limits, and more. Families are
  already mostly *selections over axes*; the axes were just grown ad hoc.
- **Per-family toy synthetics** (`gemma4Toy`, `qwen36Toy`, …) and parity
  test suites exist, but each was hand-built.
- **The runtime spine is shared**: slot cache, residency ladder, slot map,
  eager fills, phases instrumentation, benchmark protocol — all
  family-agnostic already.

The actual per-family cost concentrates in four places, one per workstream.

## Workstream 1 — Formalize the family contract

**Problem:** axes live as Swift fields with per-family constructor blocks
scattered in `knownArchitectures`; tensor-name mapping is implicit in the
repacker and model loader; nothing states which axis *combinations* are
supported.

**Deliverables:**

1. `docs/FAMILY_CONTRACT.md` — the canonical enumeration of every axis,
   its allowed values, the kernels it selects, and its conformance tests.
   Generated-checked: a test asserts the doc lists every `ArchConfig`
   field so it cannot rot.
2. A `FamilyManifest` JSON schema (checked into the model's `.gturbo` as
   today's `manifest.json -> arch`, which already carries most axes) plus
   a **tensor-name mapping table** per family — source checkpoint names →
   gturbo roles — moved out of repacker code into data
   (`Sources/MferenceRepack/Core/Families/<family>.json`).
3. An explicit **capability gate**: loading a manifest whose axis values
   fall outside the supported set fails with a named, actionable error
   ("needs axis: attention=sliding-sink") instead of a silent wrong path.

**Non-goal:** rewriting `RealForwardRunner` into a graph executor. The
runner's flag-driven branches stay; Phase A only guarantees that every
branch condition is an axis with a name, a doc entry, and a conformance
test. A graph-executor rewrite is a possible Phase C-era refactor, decided
only if axis growth makes the runner unmaintainable.

## Workstream 2 — Repacker independence (the hard dependency to break)

**Problem:** the repacker only re-lays-out *pre-quantized MLX community
conversions* (u32-packed int4 + bf16 norms, pinned repo IDs in
`SupportedModelSource.swift`). Consequences: we can only support models
someone else converted, conversions strip sidecars we need (measured: the
19-tensor Qwen MTP module), and day-≤7 support is hostage to a third
party's schedule.

**Deliverables:**

1. **bf16/fp16 → int4 group-64 affine quantizer** in `MferenceRepack`,
   bit-compatible with the runtime's `Quantization` dequant. Gates:
   (a) unit parity — quantize→dequant matches the existing
   `Quantization.quantizeInt4Affine` reference on random tensors;
   (b) model-level quality — greedy rollouts and KLD against the
   mlx-community conversion of the same checkpoint on a family we already
   support (Qwen 3.6), so the first use has a trusted control.
2. **Original-repo safetensors reading**: bf16 shards from the model
   vendor's HF repo, sharded-index aware, streamed (no full-checkpoint
   materialization — same discipline as today's streaming install).
3. **Configurable sources**: `SupportedModelSource` gains "original repo +
   quantize" entries alongside "pre-quantized repo" entries; pinning and
   SHA discipline unchanged.
4. **Sidecar policy**: optional tensor groups (MTP heads, vision towers)
   are carried or skipped by manifest flag rather than silently dropped —
   this un-blocks the deferred MTP work at zero extra cost.

**Scope cut:** int4 group-64 only. 6/8-bit become axes later if a model
demands them (the sibling NVMAI fork proves demand exists, but YAGNI now).

**2026-09-02 — the scope cut came due, and was paid.** It was not only a future
model that demanded mixed widths: it was the *control* this workstream's own
quality gate is measured against. mlx-community's Qwen 3.6 conversion carries
**80** per-tensor overrides (40 layers x `mlp.gate` + `mlp.shared_expert_gate`)
keeping every router and shared-expert gate at INT8, and the Qwen 3.6 router
kernel decodes one `uint8` per weight, so a uniformly-INT4 install of that
checkpoint was refused at load.

*(An earlier revision of this line said 83. The figure is 80, confirmed from the
control's own `config.json` and its installed resident index.)*

The repacker now has INT8 affine group-64 (`Int8AffineEncoder`,
`StreamingInt8Quantizer`, `RangeCopyTransform.quantizeInt8G64`) behind its own
W2.1a-style bit-parity gate, plus a general per-tensor `QuantBitPolicy` —
one base width and a table of name-suffix overrides resolved per family, so
adding a family adds rows rather than branches. `qwen36original` now reproduces
the control's mixture exactly (INT4 312 / INT8 80 / unquantized 221, override
set identical) and loads. **The scope cut is lifted for 8-bit; 2/3/6-bit remain
YAGNI.** See §6 of
[docs/QUANTIZER_QUALITY.md](../../QUANTIZER_QUALITY.md).

## Workstream 3 — The conformance harness (one command to trust a port)

**Deliverables:**

1. **Manifest-driven toy generator**: one `ToySynthetic.write(config:)`
   that emits a miniature install for *any* axis selection, replacing the
   four hand-built toy writers (which become thin presets of it). New
   family ⇒ toy fixtures for free.
2. **Reference-parity harness**: a scripted comparison against Hugging
   Face `transformers` (python sidecar, like `scratch/bench/mlx_bench.py`)
   producing per-layer logit deltas and golden greedy rollouts. Tolerance
   policy per the existing discipline: exact where transforms are exact,
   KLD/reference gates where float order differs.
3. **`./bringup-check.sh <family>`**: runs, in order — toy suite, install
   verify, reference parity, ladder smoke (16/32/96 slots + slot-map
   byte-gate), phases attribution snapshot, and scaffolds the community-
   protocol page. Green output is the definition of "supported."
4. **Docs template** (`docs/families/TEMPLATE.md`): the per-family page
   (sources, axes used, benchmark table, known limits) so documentation
   is a fill-in, not an essay.

## Workstream 4 — Rehearsal: LFM2.5-8B-A1B, timed

The kit is proven by using it once, on a clock. **LFM2.5-8B-A1B** is the
right rehearsal target: a real MoE the owner already wants, small enough
to iterate fast on every Mac tier, and it exercises exactly **one new
axis** (double-gated short-conv layers — `sconvKernelSize` exists, the
kernel does not), which measures the kit's marginal cost for the honest
common case where a new family brings one novelty.

Exit report: elapsed days, hours spent inside vs outside the kit, and the
axis-addition cost in isolation. If the rehearsal exceeds 5 days, the
overrun analysis becomes the punch list before Phase A closes.

## Sequencing and effort

W2 (quantizer + sources) and W3 (harness) are independent and can proceed
in parallel; W1 is a thin layer that lands alongside both; W4 is last and
gates the phase. Rough shape: W2 ≈ 2 weeks, W3 ≈ 1.5 weeks, W1 ≈ 3 days
woven in, W4 ≈ 1 week. Existing families must stay green throughout —
every workstream lands behind the standing gates (full suite, byte-gates,
community protocol on Qwen + DSV4).

## Risks, stated plainly

- **Axis explosion:** genuinely novel architectures (new attention math)
  still require kernel work no kit can amortize. The kit bounds the
  *glue* — mapping, fixtures, verification, docs — which is where the
  current weeks actually go. Kernel invention stays priced separately.
- **Quantizer quality:** wrong rounding ships silent model damage. Hence
  the double gate (bit parity vs reference quantizer, KLD vs a known-good
  conversion) before any new-model use.
- **Runner entanglement:** some family behavior hides in the runner's
  branches rather than `ArchConfig`. W1's capability-gate work will
  surface these; each becomes a named axis or a documented family quirk,
  and the count of "quirks" is itself a health metric for the eventual
  graph-executor decision.

## Non-goals

Dense model families; multi-bit quantization; runner rewrite; new UI;
serving/batching; any change to the `.gturbo` on-disk contract beyond
additive sidecar groups.
