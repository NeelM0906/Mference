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

- `FlashNextMTPWeights` strictly resolves all 29 resident tensors and the
  separate layer-0 auxiliary expert pool. It checks shapes, BF16/INT4/INT8
  payload and companion sizes before matrix dispatch, preserves each matrix's
  stored dtype, and applies the two pre-FC `1 + w` norm folds exactly once.
  Pool geometry, canonical paths, file size and the selected install-integrity
  policy are checked without borrowing the trunk's layout. It returns a
  verified stream layout; it does not allocate a draft cache or run a draft.
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

1. Connect the validated sidecar to a one-layer HC/QSA/MoE draft runner and
   open its expert pool with the selected bounded/resident memory policy.
   The loader is tested against the installed sidecar; executing that layer
   and qualifying its cache/state lifecycle remain separate work.
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

The loader, input-fusion and rollback components alone satisfy none of the final
speed/default-promotion gates. No additional model download is needed for the
current host's installed sidecar.

## Installed metadata inspection

Read-only inspection of `scratch/qwen38flashnext-r8.gturbo` on September 20
confirms 29 resident `mtp.*` entries. The manifest's `tensorCount=31` counts
source tensors, including the two fused routed-expert tensors stored in the
auxiliary pool; it is **not** the resident-entry count. The auxiliary layer
contains 512 experts with a 2,768,896-byte stride. No weights were changed or
loaded into a second inference process for this inspection.

The pre-FC norm shapes are `[2560]` and `[10240]`; both projection matrices are
`[2560,2560]`, matching the input-fusion component's whole-bundle/shared-matrix
contract. The loader at `49deb89` adds the two exact pre-FC names to the
zero-centered policy without changing the trunk suffix set. The installed
loader gate passes with INT8 router/shared-gate matrices, INT4 projections,
and full-SHA verification of the auxiliary pool. Metadata/loading consistency
is not numerical parity or evidence that a native draft decode has run.
