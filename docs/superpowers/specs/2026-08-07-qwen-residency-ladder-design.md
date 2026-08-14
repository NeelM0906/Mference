# Qwen 3.6 residency ladder — Phase 1 design

**Date:** 2026-08-07
**Status:** Approved by project owner
**Scope:** Qwen 3.6 35B-A3B decode/prefill performance on Apple silicon

## Mission context

Mference's long-term goal is to be the fastest MoE serving engine on Apple
silicon at every memory budget, for MoE models exclusively. The strategy is a
"residency ladder": the existing expert cache scales from 8 slots on an 8 GB
Mac up to a new full-residency mode on large-memory Macs, so a single
architecture covers both the low-memory niche (where Mference is the only
engine that runs at all) and the head-to-head race against mlx-lm (where all
weights fit in RAM). Phase 1 proves the ladder on Qwen 3.6 35B-A3B on the
project's 24 GB M5 host (`Mac17,2`).

Later phases, explicitly out of scope here: porting wins to Gemma 4,
Inkling-Small, and DeepSeek-V4-Flash (Phase 2, including the DSV4
chunked-prefill and context-cap items from the existing perf plan);
continuous-batching multi-client serving (Phase 3); new MoE model bring-up
such as LFM2.5-8B-A1B and gpt-oss-20b/120b (Phase 4). Dense models are
permanently out of scope.

## Success criteria

1. **Top rung:** Mference Qwen 3.6 decode tok/s meets or beats mlx-lm running
   the same checkpoint quantization on the same 24 GB M5 host, measured with
   the community benchmark protocol's three frozen prompts, full production
   sampling path.
2. **Lower rungs:** no decode or prefill regression at the 16- and 32-slot
   profiles. Exact-transformation changes must produce byte-identical output
   to their controls; float-reordering kernels must pass the existing
   reference-output quality gates.
3. **Discipline:** every accepted change shows a repeatable end-to-end gain
   under alternating control/candidate runs, per the method documented in
   `docs/OPTIMIZATION_JOURNEY.md`. Inconclusive results do not ship.

## Workstream 1 — Ground truth (baseline & attribution)

The MLX comparison on record (76–82 tok/s) is Gemma 4, not Qwen. Before any
optimization:

- Reinstall `qwen36.gturbo` via the streaming repacker (the previous install
  was deleted in a disk cleanup; only lock files remain).
- Install `mlx-lm` and the matching Qwen 3.6 35B-A3B 4-bit MLX checkpoint;
  measure its decode and prefill on the three frozen community prompts on the
  same host. Record RSS/GPU allocation alongside tok/s.
- Profile Mference's current 32-slot decode step into three buckets: expert
  I/O + cache bookkeeping, GPU compute, CPU orchestration. This attribution
  decides how much of the gap residency removes for free versus what the
  kernel program must earn.

Deliverable: a benchmark note with the measured gap and its composition.

## Workstream 2 — The resident rung

A new expert-cache mode, `resident`:

- Map each `packed_experts/layer_XX.bin` once as a read-only file-backed
  Metal buffer and dispatch expert kernels directly into it by subregion
  offset. Expert strides are already page-aligned and kernels already bind
  subregions of existing buffers, so no repack format change is required.
- On the decode and prefill hot paths, resident mode bypasses slot copies,
  LRU bookkeeping, and I/O workers entirely.
- The CLI/server auto profile gains a top rung: choose `resident` when the
  expert pool plus common weights, KV cache, and a safety headroom fits
  physical memory (Qwen's ~18.1 GB pool qualifies on a 24 GB host). Explicit
  `--expert-cache-slots` values keep their current meaning; `resident`
  becomes a new accepted value.
- Gate: resident mode must produce byte-identical output to slot mode for
  the same seed and prompts.

Risk and fallback: 18.1 GB of experts plus core, KV, and app overhead is
tight under macOS memory pressure on a 24 GB host. If pinning the full pool
thrashes, the fallback rung is "near-resident" — the same direct-mapped path
with a bounded hot set (for example 96 or 128 slots per layer) — and the
first-use page-in cost is treated as warmup, not steady-state decode.

## Workstream 3 — Kernel supremacy program

With expert I/O off the hot path, decode is GPU-bound. Iterate the repo's
proven loop — profile the whole token step, attack the largest measured
share, verify with a clean end-to-end A/B — over this candidate queue,
re-ranked after each profile:

1. **Batched multi-expert GEMV:** compute all top-k routed experts in one
   dispatch, gathering weights from the resident pool, instead of per-expert
   launches.
2. **GPU router top-k** (carried over from the ds4/colibri port plan): keep
   routing on-GPU and cut the per-layer CPU round-trip.
3. **Command-buffer consolidation:** one command buffer per token step where
   hazards allow, reducing per-layer encode and scheduling overhead.
4. **M5 Neural Accelerator paths:** MPP `tensor_ops` decode variants where
   group-64 affine INT4 shapes allow. This is believed to be the source of
   MLX's M5-generation advantage. Risk: MPP may not consume the affine
   format directly; if dequant staging eats the gain, the candidate is
   dropped by the standard end-to-end gate.
5. **Prefill compounding:** resident mode removes per-chunk expert re-reads;
   re-tune prefill chunk sizing on top of the existing TensorOps INT4 path.

Failed candidates are recorded in the optimization journey as usual.

## Workstream 4 — Verification and the public claim

- Community benchmark protocol (three frozen `real-generation-v1` prompts,
  fixed seeds, app sampling defaults, one discarded warmup, fresh-process
  measured runs, medians where multiple runs are taken).
- Alternating control/candidate runs whenever page cache, warmup, or
  thermals could bias a result.
- The headline claim under construction: "Qwen 3.6 35B-A3B: X tok/s on
  Mference vs Y tok/s on mlx-lm on the same Mac — and Z tok/s in ~1.45 GB,
  where MLX cannot run at all." Both halves must be reproducible from
  documented commands before being published in `docs/BENCHMARKS.md`.

## Non-goals

- No changes to the `.gturbo` format or the streaming installer.
- No new model families in Phase 1.
- No server/batching work in Phase 1.
- No dense-model support, ever.
