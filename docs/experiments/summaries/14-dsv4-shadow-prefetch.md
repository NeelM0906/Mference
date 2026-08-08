# DSV4 shadow speculative prefetch — accepted, M5/24 GB, 2026-08-07

Follow-up to [13-dsv4-streaming-iterations.md](13-dsv4-streaming-iterations.md):
pilot's predictor kept, its transport replaced. Three measured changes, each
byte-identical to the no-speculation control:

1. **Join-if-useful.** A real plan waits for an in-flight speculative read
   only when a predicted expert for that layer is actually routed — the wait
   then always replaces a larger demand re-read. Mispredictions never block;
   late records are reaped lazily. (Pilot joined always; a never-join
   variant re-read in-flight experts on the demand path and won nothing.)
2. **`userInitiated` read QoS.** The speculative reads were running at
   `.utility`, which Darwin throttles to low I/O priority — a hidden tax on
   exactly the reads the critical path may join.
3. **Issue budget 2 per layer**, taken from the front of the weight-ranked
   prediction list. Swept 1/2/3/4/6 → 5.50/5.64/5.51/5.38/5.31 tok/s on the
   64-token diagnostic; the peak is sharp, matching the eviction/SSD-
   contention model of pilot's 48 GB flood.

## Community-protocol A/B (alternating blocks, fresh processes)

`--rdadvise adaptive --verify trusted-receipt`, cases restricted to the two
that reach a natural end of turn. medium-review is excluded for *any*
config: DSV4 runs it to the 1,024-token cap by model behavior and the
protocol rejects capped runs.

| Case | baseline decode (3 runs) | shadow decode (3 runs) | gain |
| --- | --- | --- | ---: |
| short-explanation | 4.455 / 4.413 / 4.423 | 5.098 / 5.285 / 5.186 | +17.8% |
| long-synthesis | 3.746 / 3.753 / 3.820 | 4.305 / 4.322 / 4.272 | +13.7% |

Long-prompt prefill (through the decode path) improved as a side effect:
388–401 s vs 416–436 s (~7%).

## Decision

`MFERENCE_SPEC_PREFETCH=shadow` becomes the DSV4 production default
(env overrides in either direction). Other families keep `off`: Qwen's
decode loop has no pilot GEMV wiring yet — porting it is the next
streaming-lane item, targeting Qwen's measured 23% exposed I/O.

Remaining headroom on DSV4: exposed I/O fell from 5.26 s to 2.71 s per 64
diagnostic tokens — roughly half the original stall survives (join waits and
sub-window read latency). Deeper fixes if wanted later: issue at L−2 via a
two-layer lookahead router, or per-expert join granularity instead of
per-record.

## Qwen port attempt (2026-08-07, rejected)

The same lookahead was wired into the Qwen/Gemma decode loop — layer L+1's
router encoded against layer L's pre-FFN normed state in the same command
buffer, read at the existing router wake (no new synchronization). Greedy
128-token diagnostic at 32 slots, byte-identical output:

- Recall 81.9%; exposed I/O 1,215 → 521 ms (−57%).
- Decode 24.57 → 21.60 tok/s (−12%). **Rejected.**

The economics invert on a host whose page cache holds the whole Qwen pool:
a demand miss is a ~page-cache memcpy, so join waits plus 14 GB of extra
speculative copies cost more than the misses they avoid. Shadow stays
DSV4-only (family-gated default); the Qwen pilot wiring remains in the
code, inert unless `MFERENCE_SPEC_PREFETCH` requests it, as the substrate
for future copy-free-hit experiments. Qwen's 23% exposed I/O wants a
different fix: serving page-cache-resident hits without the slot copy.

## NVMAI phase-1 kernel diff (2026-08-08, no action)

The sibling fork's "routed MoE phase-1 rewrite (−36% routedCB)" was diffed
against our tree: identical shared body, identical function-constant
specialization — our production kernel *is* their rewrite (r8 variant);
their −36% was measured against their own older baseline. Their only novel
knob, 16 simdgroups per threadgroup (`_r16`), was ported behind an env flag
and measured byte-identical but 2.5–3% slower on the M5 across two
alternating pairs (24.76/25.16 vs 25.54/25.81 tok/s). Rejected and removed;
r8 with 256-thread threadgroups stays the default.

## MTP speculative-decoding scoping (2026-08-08, deprioritized on evidence)

The original `Qwen/Qwen3.6-35B-A3B` ships a complete 19-tensor MTP draft
module (fc + one full attention+MoE block + norms); the mlx-community 4-bit
conversion strips it, and our repacker has no bf16→int4 quantizer, so
carrying it requires a new quantization encoder, a second-repo fetch, format
plumbing, a draft block runtime with its own KV stream, and a verify loop —
a multi-session build.

Before building, the ceiling was measured: batched verification economics.
Qwen prefill at chunk 32 runs 33.2 tok/token-s vs ~29 tok/s decode — only
~1.15x cheaper per token, versus the ~2x dense engines see. Cause: routed
expert reads scale with the *union* of the batch's experts (only the shared
core amortizes), so k-token verification on a streaming MoE saves far less
than on dense weights. Projected MTP gain at k=2 drafting with ~80%
acceptance: roughly +15–25%, against a large engineering cost. The NVMAI
fork's MTP benchmarks (2-token capability completions) contain no valid
throughput evidence either way.

Verdict: deprioritized behind copy-free cache hits (attacks Qwen's measured
23% exposed-memcpy share directly) and the GPU-compute kernel program
(~52% share). Revisit MTP if those lanes exhaust.

## Copy-free miss serving (2026-08-08, rejected — mechanism identified)

Qwen's ~23% exposed I/O is page-cache memcpys into slots, so misses were
served as zero-copy mapped views (per-expert MTLBuffers over the layer
mapping) with background slot promotion keeping the LFU learning. Byte
gate passed; unit tests passed; decode fell 26.2 → 17.0 tok/s. The phase
breakdown isolates why: expert io await collapsed 2,370 → 69 ms, but GPU
waits exploded 2,741 → 6,199 ms — the driver wires the file-backed pages
on the command-buffer timeline, and wiring ~140 mapped buffers per token
costs more than copying the same bytes into already-wired slots.

Combined with the resident-rung long-context collapse and the DSV4
32-slot regression, the platform rule is now established three ways:
**on macOS/Metal, hot-loop GPU access wants wired scratch and explicit
copies, not mapped file-backed pages.** The copying slot cache is
load-bearing, not legacy. Qwen's remaining exposed-copy share is
addressable only by fewer misses (the shipped 64-slot default) or by
hiding copies behind more GPU overlap — a command-buffer pipelining
question, not a memory-mapping one.

## GPU busy/gap attribution (2026-08-08): decode is orchestration-bound

`MFERENCE_PHASES=1` now reports GPU busy vs span from command-buffer
timestamps. Qwen 3.6, 64 slots, 128 greedy tokens, 29.8 tok/s:

- gpu busy 1,864 ms / span 4,295 ms → **gap 2,430 ms (57% idle)**
- per token: 33.6 ms wall = 14.6 ms GPU compute + ~5.9 ms exposed I/O
  + ~13 ms CPU↔GPU round-trips (40 router wakes: event wait, plan,
  miss memcpy, routedCB encode+commit per layer)

Consequences: the compute roofline is not the binding constraint — a
gap-free schedule alone reaches ~68 tok/s on today's kernels. The ranked
attack is now (1) a GPU-resident slot map + GPU router top-k so all-hit
layers run with zero CPU involvement (the CPU intervenes only on misses),
(2) deeper cross-layer encode-ahead. MPP tensor-ops drops in priority:
faster kernels widen an already-dominant gap.
