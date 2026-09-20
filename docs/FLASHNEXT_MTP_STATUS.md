# Flash-Next native MTP: implementation boundary

Native Flash-Next speculative decoding is **not enabled**. Carrying the MTP
weights in the install does not qualify a decoder or establish any speedup.
The dense Qwen MTP implementation cannot simply be selected for this model.

## Architecture contract

The pinned [SGLang MTP implementation](https://github.com/sgl-project/sglang/blob/745de73ba3c136b6f99b7a3e2177ed1a8eef4a56/python/sglang/srt/models/qwen4_exp_mtp.py)
uses a single full-attention layer with no PLE. Its input is the target's
four-stream hidden bundle, not the final mixed head input. It normalizes the
embedding and the complete hidden bundle separately, applies distinct linear
projections, then broadcasts the embedding projection into each hidden stream.
The hidden projection is shared across streams. Pre-FC norms use the
zero-centered convention. The draft returns both its mixed output for the
head and its full hidden bundle for subsequent drafting.

## Implemented foundations

- `FlashNextMTPInputFusion` composes the existing RMSNorm, projection and add
  encoders. It expects already-folded norm weights. It has no caller in ordinary
  generation and allocates no scratch unless explicitly constructed.
- `FlashNextForwardRunner` can checkpoint recurrent GDN state, convolution
  tails, PLE state/history and the sequence cursor. KV and indexer rows are
  append-only; rollback hides discarded rows by rewinding the cursor.
- Checkpoints belong to one runner and one branch of its history. Reset or
  restoration invalidates older checkpoints, preventing resurrection of KV
  rows overwritten by a different continuation.
- Interrupted decode now marks sequence state dirty, just as interrupted
  chunked prefill does. Ordinary continuation must reset; an internal verifier
  may restore a previously committed checkpoint.

These are implementation boundaries, not a passed native-MTP qualification.
The [dated validation record](RELEASE_VALIDATION_2026-09-20.md) is authoritative
for which synthetic and installed checks have actually executed.

## Remaining integration and gates

1. Strictly load the sidecar's fusion/norm tensors, one-layer HC/QSA/MoE block
   and its own expert pool, preserving tensor dtype and install integrity.
   The auxiliary pool does not have the trunk's `layout.json`; its manifest
   metadata must be checked, not blindly treated as the trunk layout.
2. Preserve the target's final full hidden bundle at both cold-prefill and
   warm/decode boundaries; align the draft token and position convention.
3. Implement target verification and accepted-prefix replay/rollback. Ordinary
   chunked prefill is not automatically an exact speculative verifier: its
   numerical and greedy equivalence must be demonstrated for this use.
4. Qualify correct, partially accepted and fully rejected drafts; EOS, stop
   strings, context exhaustion, cancellation, prefix reuse and model switching.
   Compare against MTP-off target results, including all recurrent state.
5. Compare the native drafter with an independent reference of the same pinned
   weights. Measure acceptance and completed-answer time on separate resident
   and bounded-memory profiles. Keep MTP off unless its benefit exceeds run
   variability without changing the target result or memory guarantees.

The input-fusion and rollback components alone satisfy none of the final
speed/default-promotion gates. No additional model download is needed for the
current host's installed sidecar.
