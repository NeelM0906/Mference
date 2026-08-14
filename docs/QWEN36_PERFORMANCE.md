# Qwen 3.6 decode performance notes

Measured 2026-07-31 on an Apple M5 with 24 GB of memory, macOS 26.5, against
the installed `scratch/qwen36.gturbo` (19.55 GB) with the production runtime
profile: 16 expert slots, 4K context, RDADVISE off. The phase breakdown below
uses a greedy 128-token decode from a 10-token prompt; the headline throughput
and memory figures come from the
[community benchmark protocol](COMMUNITY_BENCHMARKS.md) and are reported in
[Benchmarks](BENCHMARKS.md#qwen-36-35b-a3b-measured-decode).

These are measurements from one host, not performance ceilings. Decode rate is
sensitive to OS page-cache state; see [Warming the page cache backfires]
(#warming-the-page-cache-backfires).

## Production acceleration (2026-08-06)

The production CLI and server resolve their expert-cache default by model
family and physical memory; at this record's date Qwen used 32 slots per layer
on hosts with at least 16 GiB and 16 slots on smaller hosts (the ≥24 GiB rung
has since moved to 96 slots — see round 3 below). An explicit
`--expert-cache-slots 16` keeps the lower-memory path. The Mac app keeps 16 as
its explicit default and offers every allowed slot count in the inspector.

On the M5/24 GB host, the exact community protocol at commit `3d6996b` produced:

| Case | Prompt / generated | Prefill | Decode |
| --- | ---: | ---: | ---: |
| short-explanation | 62 / 538 | 7.57 s | 29.293 tok/s |
| medium-review | 426 / 704 | 8.48 s | 27.460 tok/s |
| long-synthesis | 2,940 / 695 | 26.54 s | 23.470 tok/s |

Every case ended with `stop=endOfTurn`. The generated files are byte-identical
to the accepted 16-slot controls at the same source state. Those controls ran
at 21.487, 24.828, and 21.490 tok/s respectively, so the production decode
gain is 36.3%, 10.6%, and 9.2% (18.1% geometric mean) without changing the
routing workload. The larger cache is responsible for the last step; it stacks
with the dedicated decode kernels described below.

Against the original community run's prefill times, the same final protocol is
1.00x, 1.42x, and 2.20x faster for short, medium, and long prompts. Generated
token counts changed after the batched TensorOps prefill arithmetic landed, so
the original and final *decode* rows are not token-for-token comparable. Their
raw decode rates are 15.5–18.3% higher, but the byte-identical 16-vs-32 rows
above are the defensible decode A/B.

The accepted compute work comprises a prompt-block Qwen scalar-gate kernel, a
TensorOps INT4 shared-expert prefill path, fused Qwen shared-expert gate/up/
activation decode, fixed-shape GDN input/delta/gated-normalization kernels, and
Qwen full-attention specialization. Candidate pilot routing, staged projection/
convolution, conv/QK normalization, and packed-Q epilogue fusions were removed
after production runs failed to beat the accepted path.

## Memory: the 8 GB envelope

The whole point of the runtime is a 26B-class MoE on an 8 GB Mac in about
2 GB of process memory. Qwen 3.6 meets that budget with more headroom than
Gemma 4, because its experts are half the size and only 10 of its 40 layers
keep a KV cache:

| Component | Qwen 3.6 35B-A3B | Gemma 4 26B-A4B |
| --- | ---: | ---: |
| Common weights (mapped, file-backed) | 1.39 GB | 1.35 GB |
| Routed-expert slots, 16 per layer | 1.13 GB | 1.61 GB |
| KV cache at 4K | 84 MB | 320 MB |
| Gated-DeltaNet recurrent state | 64 MB | — |
| Routed-expert files on disk | 18.1 GB | 12.9 GB |

Only the on-disk footprint is larger; every resident component is equal or
smaller.

### Measured under an emulated 8 GB machine

**Correction (2026-08-08):** the original version of this section reported
decode unchanged within noise under an emulated 8 GB working set
(22.95 / 21.35 / 18.62 tok/s). That experiment's memory pin did not hold —
the ballast was reclaimable, so the expert pool stayed in the page cache
and the run measured a 24 GB machine twice. The claim also fails
arithmetic: 16-slot decode above 20 tok/s would need 5–9 GB/s of
sustained random SSD reads. The numbers below use a verified ballast
(15.2 GiB `mlock`ed of 16 GiB held, free memory observed near zero), and
the pre/post-optimization spread was cross-checked by rerunning with the
2026-08-08 features disabled (12.3 / 12.4 tok/s short/long — same regime,
so the correction is about the pin, not the code).

Community protocol, 16 slots, one measured rep under pressure:

| Case | Decode, 24 GB unconstrained | Decode, ~8 GB verified ballast |
| --- | ---: | ---: |
| short-explanation | 26.45 tok/s | 14.35 tok/s |
| medium-review | 24.99 tok/s | 12.75 tok/s |
| long-synthesis | 21.96 tok/s | 10.81 tok/s |

All runs reached `stop=endOfTurn`; process footprint stayed ~1.1 GB. With
the pool unable to cache, decode is SSD-latency-bound: the honest 8 GB
story is 11–14 tok/s on M5-class hardware, and less on older 8 GB
machines with slower SSDs and GPUs — compare the 8 GB M2 and 24 GB M5
Gemma 4 rows in [Benchmarks](BENCHMARKS.md).

The practical 8 GB requirement is therefore disk, not memory: the install
needs about 19.6 GB free, against Gemma's 14.3 GB.

## Where decode time goes

`MFERENCE_PHASES=1` on the CLI reports the runner's phase counters:

| Phase | Time | Share |
| --- | ---: | ---: |
| Routed-expert I/O await | 3477 ms | 53% |
| GPU execution (waits) | 2976 ms | 45% |
| `cb1` encode + commit | 83 ms | 1% |
| `cb2` encode + commit | 52 ms | 1% |
| **Total decode** | **6588 ms** | 19.4 tok/s |

I/O and GPU are essentially serial: their sum is within 2% of the wall time.
That is inherent to the streaming design — a layer's routed experts cannot be
read until that layer's router has run, and the next layer cannot start until
this layer's MoE has finished.

The table is the July 16-slot record; the 2026-08-08 defaults relaxed the
serialization — on all-hit layers the GPU slot map skips the CPU plan
entirely, and on miss layers the eager routed commit runs the preads in the
background (see round 3 below). The phase report now also splits the I/O
await into GPU-overlapped and exposed time, and prints the all-hit
layer-step rate, GPU busy/span/gap, and the speculative-prefetch counters.

Per token the runtime touches about 1.6 GB of weights, of which roughly
540 MiB is routed-expert data (40 layers x 8 experts x 1.69 MiB).

## What is not the bottleneck

Each of these was measured before being ruled out.

- **Kernel math.** The INT4 GEMV kernels reach 140-150 GB/s at Qwen's shapes,
  at or near this machine's memory bandwidth. Specializing the pipelines on
  the model's shapes (now derived from `ArchConfig`) lifted the narrow
  4096x2048 projection from 101.6 to 141.0 GB/s, but moved end-to-end decode
  only about 3% because GEMV work is a small share of the token.
- **Metal encoder overhead.** Creating one encoder per kernel costs ~1.3 us
  versus ~1.0 us for a shared encoder. Across ~900 dispatches that is a
  0.25 ms/token difference — not worth restructuring for.
- **Expert cache policy.** The July workload moved from 19.0 to 19.8 tok/s when
  raising slots from 16 to 32, but the frozen August same-output cases gained
  9.2–36.3%. LFU remains the replacement policy. Slot counts above 32 were not
  offered at the time of this record; the allowed set has since grown to
  8/16/24/32/64/96/128 plus an explicit `resident` mode, and auto now gives
  Qwen 96 slots on hosts with at least 24 GiB and 32 at 16 GiB; the 8 GB path
  and every other family retain 16.
- **RDADVISE read-ahead.** `--rdadvise default` cut the I/O await from 3474 to
  2931 ms, but the synchronous advice calls cost what the reads saved; total
  decode was unchanged (19.3 vs 19.3 tok/s). It stays off by default.

## Why Qwen 3.6 decodes slower than Gemma 4 here

The repository's published M5 Pro rows put Gemma 4 at 31-35 tok/s; the current
Qwen 3.6 production cases measure 23.5-29.3 tok/s on this M5. The workloads and
M5 variants differ, so this is a project-level range comparison rather than a
token-for-token model A/B. Two facts about the remaining gap are measured here,
and the explanation for it is not.

Measured:

- Decode is 53% expert-I/O wait, and I/O is serial with GPU work.
- Qwen's throughput does not depend on the OS page cache. Repeated runs of the
  same case do not speed up (21.4, 22.1, 20.8 tok/s back to back), and
  constraining the host to an ~8 GB working set does not slow it down. Both
  point to its expert reads already being served from SSD rather than cache.
- Qwen reads *fewer* routed-expert bytes per token than Gemma: at most 540 MiB
  (40 layers x 8 experts x 1.69 MiB) against at most 769 MiB (30 x 8 x
  3.36 MiB). So the gap is not read volume.
- With the same 16 slots, Qwen caches at most 6.2% of a layer's 256 experts
  against Gemma's 12.5% of 128, so it misses more often. Raising Qwen to 32
  slots recovers Gemma-equivalent coverage; the frozen byte-identical A/B gains
  9.2-36.3% by case, or 18.1% by geometric mean.

Not measured, and stated here as a hypothesis rather than a finding: that
Gemma is faster mainly because its 12.9 GB expert pool largely stays in the OS
page cache on a 24 GB host while Qwen's 18.1 GB pool does not, so Gemma's reads
are served from memory and Qwen's from SSD. It is consistent with everything
above, and with scattered reads across Qwen's expert files measuring 5-10 GB/s
against 30+ GB/s for reads that hit cache. But confirming it requires running
Gemma 4 under the same protocol on this host, and only Qwen is installed here.
The actual per-run cache hit rate was also not instrumented, so no
bytes-per-second figure is quoted for either model.

The dedicated Qwen compute paths and larger-host cache default narrow this gap.
Whatever remains follows from the checkpoint's shape against the host's memory;
the 16-slot path remains available for the original memory budget.

## Round-trip latency

Decode performs one `commit` + `waitUntilCompleted` per layer to read the
router's top-8 expert IDs back to the CPU. A measured round trip on this host
is ~196 us, so Qwen's 40 layers spend ~7.8 ms/token (15%) in submission and
completion latency alone — against Gemma 4's 30 layers at ~5.9 ms. This is
the structural cost of having 10 more layers, and it cannot be removed
without breaking the router-then-read dependency.

## Warming the page cache backfires

Reading all expert files into the OS page cache before a run
(`cat packed_experts/*.bin > /dev/null`) **halved** throughput, from 19.4 to
11.1 tok/s, and raised the I/O await from 3477 to 4500 ms. Indiscriminate
warming evicts the pages the runtime actually reuses — the mapped common
weights and the LFU-hot experts — and drives up memory pressure. Throughput
recovers over the next few runs as the cache re-warms naturally.

This reproduces, on much newer hardware, the original finding that the
virtual-memory system cannot be trusted to keep the expert working set warm.
See [`mmap` versus `pread`](experiments/summaries/01-model-install-and-expert-io.md#io-01).

Because decode rate depends on cache state, benchmark runs should be repeated
until the rate stabilizes, and the steady-state value reported.

## Measuring

```bash
MFERENCE_PHASES=1 .build/release/MferenceCLI \
  --model scratch/qwen36.gturbo \
  --prompt "Write a detailed essay about the history of computing." \
  --max-new 128 --temperature 0
```

`--expert-cache-slots` (8, 16, 24, 32, 64, 96, 128, `resident`, or `auto`)
and `--rdadvise` (off, default, bounded, adaptive) vary the two policies
discussed above.

## Production acceleration round 2 (2026-08-08)

Two accepted defaults on the 24 GB M5, both byte-identical to their
controls under the community protocol:

- **64 expert-cache slots** on hosts with ≥24 GiB (16 GiB hosts keep 32):
  +4.6–5.9% decode across all three cases. The auto rung has since moved
  to 96 — see round 3 below.
- **GPU-resident slot map** (`MFERENCE_SLOT_MAP=0` to disable): the router
  top-k is resolved to slot-slab offsets on-GPU, and the ~30% of
  layer-steps whose experts are all cached complete their routed FFN
  inside cb1 with no CPU plan, fetch, or routed command buffer. +1.6–3.5%
  decode on top of the 64-slot rung. Details:
  [experiments/summaries/15](experiments/summaries/15-gpu-slot-map.md).

Community-protocol medians moved from 29.29 / 27.46 / 23.47 tok/s
(2026-08-06) to **34.31 / 33.21 / 28.03 tok/s** — a 17–21% cumulative
decode gain with unchanged outputs. mlx-lm cannot load this model on the
same host (see
[experiments/summaries/11](experiments/summaries/11-mlx-qwen-baseline.md)).

## Production acceleration round 3 (2026-08-08)

Two further accepted defaults, byte-identical to their controls:

- **96 expert-cache slots** as the auto rung on hosts with ≥24 GiB (16 GiB
  hosts keep 32, smaller hosts 16). The slot map changed the slot-count
  economics: at 96 slots the all-hit layer rate doubles to 50.7% (from
  29.9% at 64), and the pre-slot-map 96-slot regression no longer
  reproduces.
- **Eager routed commit** (`MFERENCE_EAGER_ROUTED=0` to disable): the
  routed command buffer is encoded against the plan's slot-slab views,
  committed gated on a shared fill event, and the preads run in the
  background — the fetch leaves the CPU critical path, and the phase
  report's `expert io await` reads near zero. +1.5–3.3% median by case on
  top of the round-2 state. A failed eager expert read aborts the decode
  step with `ModelError.eagerExpertFillFailed` instead of letting a
  partially filled slot corrupt output.

Community-protocol medians moved to **34.85 / 33.97 / 28.27 tok/s**. An
explicit `--expert-cache-slots resident` mode (map every layer file once,
no slot cache) also exists but lost the community A/B on this 24 GB host
on every case (short −2%, long −56% from page-cache thrash), so auto
always uses the slot cache and resident stays opt-in. Details:
[experiments/summaries/15](experiments/summaries/15-gpu-slot-map.md) and
[10](experiments/summaries/10-qwen-resident-rung.md).
