# S4 encode-ahead: the ordering hazard, made precise

Status: analysis only — no implementation. Written so the next attempt does
not rediscover the constraint the hard way.

## What S4 wants

`15-gpu-slot-map.md` measured that the slot-map gain lands below projection
because "the router-wake event wait itself remains on every layer." At 96
slots 50.7% of layer-steps are all-hit, yet the CPU still serializes every
layer: encode L → commit → wait wake L → encode L+1. S4 = commit layer L+1's
cb1 *before* layer L's wake, so consecutive all-hit layers flow on the GPU
with no CPU-paced gap.

## The hazard, precisely

The natural design — pre-encode L+1's cb1 and gate its start on a shared
event that is signaled once layer L's routed contribution is in `hidden` —
deadlocks on the miss path:

1. Metal executes command buffers on one queue in commit order, and an
   `encodeWaitForEvent` stalls the *queue*, not just the buffer.
2. On a miss layer, the routed command buffer is encoded by the CPU after
   the wake and committed *after* the pre-committed L+1 cb1.
3. L+1 cb1 stalls on the event; the routed CB behind it can never start;
   the event it waits for can never be signaled. Deadlock.

Conditional GPU-side signaling cannot rescue this: `encodeSignalEvent` is
unconditional in the command stream, so the guarded FFN buffer cannot
signal "only if all-hit."

## Viable shapes (all unbuilt)

- **Second queue for the speculative chain.** The routed fallback stays on
  the primary queue; pre-encoded L+1 work waits on its own queue without
  blocking the fallback. Caution: the plain second-queue experiment
  (ORCH-10) *lost* throughput; this variant is different (event-gated, not
  free-running) but must re-run that A/B.
- **Guarded mega-buffer (Task 11 convergence).** Encode the whole token
  step in one CB with every layer's routed FFN guarded by its slot-lookup
  flag, and a *re-encode* fallback pass for the miss layers after the CB
  completes. Miss layers then cost a second partial pass — only pays while
  the all-hit rate stays high; needs the arg-buffer-reuse caution from the
  ORCH table (prior reuse experiment lost 9% on prefill).
- **CPU-signaled event, wake off the GPU path.** Keep the CPU wake, but
  pre-encode+commit L+1 gated on a CPU-signaled event value; at an all-hit
  wake the CPU's only work is `event.signaledValue = v` (microseconds)
  instead of encode+commit (~hundreds of µs × 40 layers). Same queue-stall
  constraint applies to the miss path, so this still needs one of the two
  shapes above for the fallback, or a bounded speculation depth of exactly
  one layer with the routed fallback encoded into the *same* pre-committed
  buffer as predicated no-op work.

## Measurement gate

Any attempt must show, on the community protocol with the real Qwen 3.6
install: byte-identical outputs, and a decode gain consistent with the
attribution ceiling (~68 tok/s gap-free bound from
`13-dsv4-streaming-iterations.md` methodology applied to Qwen). Toy-model
parity (QwenSlotMapParityTests pattern) gates correctness before any real
bench.
