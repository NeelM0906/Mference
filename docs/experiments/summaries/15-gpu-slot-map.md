# GPU-resident slot map — accepted Qwen default, M5/24 GB, 2026-08-08

The decode attribution showed Qwen orchestration-bound: the GPU idle 57% of
decode, ~13 ms/token spent in 40 per-layer CPU round-trips (router wake →
plan → miss memcpy → encode → commit). The slot map removes the CPU from
every layer whose routed experts are all cached.

## Design (stages S1–S3)

- **S1:** each layer's 64 slots became one contiguous wired slab; slot n is
  a byte offset, and every argument-buffer consumer carries offsets.
- **S2:** a 256-entry expert→slot table per layer (Int16, shared storage)
  mirrors slot state under the cache lock; measurement showed **29.9% of
  layer-steps are all-hit** at 64 slots.
- **S3:** `router_slot_lookup_k8` resolves the router's top-k to slab
  offsets and writes an all-hit flag before the router signal; guarded
  phase-1/phase-2/residual-add kernels — sharing the production math
  bodies — run in a dedicated command buffer committed behind cb1 and
  no-op unless every expert is cached. On all-hit layers the CPU's work
  shrinks to reading the flag and bumping LFU counters; the fallback path
  is byte-for-byte the previous code.

## Gates

- Kernel parity: `MoEFusedFFNTests.slotMapPipelineMatchesArgumentBufferPath`
  (permuted slots, GPU-resolved offsets) — bit-exact.
- Toy rollout parity (`QwenSlotMapParityTests`) and real-model greedy +
  seeded 128-token byte gates — identical with the skip path active.
- Community protocol, four alternating blocks (6 baseline / 6 slot-map
  runs): medians short 33.15→34.31 (+3.5%), medium 32.13→33.21 (+3.3%),
  long 27.58→28.03 (+1.6%); slot map won 3 of 4 blocks pairwise and every
  case median. Accepted as the Qwen default; `MFERENCE_SLOT_MAP=0`
  disables.

## Journey note

A mid-development line-range deletion silently removed the guarded-FFN
encode, producing a convincing "kernels exact but chain no-ops" mystery;
the debug comparator (`MFERENCE_SLOT_MAP_DEBUG=1`) exposed it. The gain
lands below the +10–15% projection because the router-wake event wait
itself remains on every layer; removing that wait on all-hit layers needs
S4 encode-ahead, which stays deferred with its known ordering hazard.
