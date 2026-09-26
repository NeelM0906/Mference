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
  stored matrix dtype, and applies native `1 + w` norm folds exactly once in
  FP32 (or widens already-folded norms without another addition).
  Pool geometry, canonical paths, file size and the selected install-integrity
  policy are checked without borrowing the trunk's layout. It returns a
  verified stream layout; it does not allocate a draft cache or run a draft.
- `FlashNextMTPInputFusion` keeps normalization, projections and the broadcast
  sum in FP32; its standalone default writes FP16, while the native runner
  retains an FP32 bundle. It expects already-folded
  norm weights. It has no caller in ordinary generation and allocates no
  scratch unless explicitly constructed. BF16/INT4/INT8 and nonzero offsets
  are gated at hidden dimensions 64 and 2,560.
- `FlashNextForwardRunner` can checkpoint recurrent GDN state, convolution
  tails, PLE state/history and the sequence cursor. KV and indexer rows are
  append-only; rollback hides discarded rows by rewinding the cursor.
- Target-state capture returns an owned FP16 copy of the full, unmixed HC
  bundle and the processed-token count after prefill, warm append or decode.
  Checkpoints retain it across rollback; reset/dirty state rejects capture.
  Ordinary generation performs no extra GPU copy. This is not a draft-token
  position-alignment or speculative-verification implementation.
- An internal opt-in consumer can receive **every** unmixed target row from
  each normal prefill chunk and decode step, with its original tokens and
  absolute positions. Copies are owned; no target replay is introduced.
  Consumption happens after GPU completion but before target commit, so an
  exception leaves the target dirty. The orchestrator must restore/reset both
  target and consumer state. Ordinary generation has no consumer or GPU copy.
  The installed resident/16-slot boundary test checks every emitted row for
  finite/nonzero values, exact tokens/positions/counts and exact last-row
  equality with the separate snapshot accessor, including 1,024-token chunks.
- Checkpoints belong to one runner and one branch of its history. Reset or
  restoration invalidates older checkpoints, preventing resurrection of KV
  rows overwritten by a different continuation.
- Interrupted decode now marks sequence state dirty, just as interrupted
  chunked prefill does. Ordinary continuation must reset; an internal verifier
  may restore a previously committed checkpoint.
- The internal `FlashNextMTPDraftRunner` now connects fusion, one HC/QSA/MoE
  layer, the native mixer and the shared output head. It opens only the separate
  auxiliary expert pool, using an explicit bounded or resident policy. Its
  contiguous draft-row cursor, owned output bundles and runner-local checkpoint
  epochs prevent skipped cache priming or resurrection of discarded branches.
  Its mechanical resident/bounded, rollback, reset and dirty-state tests pass.
  Installed INT4 projections / INT8 gates and finite native-layer output probes
  also pass, including exact resident/16-slot replay. These probe inputs are
  synthetic, not aligned target states. It is not called by CLI/server generation.
- The native-only attention, HC mixer and expert encoders now retain FP32
  intermediates and attention cache entries. Installed weight storage stays
  unchanged; returned bundles and sampler logits remain FP16. The target
  decoder does not use these correctness-first encoders.
- `FlashNextMTPPrimer` now consumes captured target rows and pairs row `i`
  with the embedding of token `i + 1` at draft position `i`. One owned tail
  row waits for the next chunk's first token or an explicit target-selected
  token; a later append must match that token. It primes every preceding draft
  KV row without replaying the target. Runner-local checkpoints preserve the
  pending tail, token expectation and draft cursor. Cancellation requires
  restoring both target and primer. Synthetic cold/warm/decode partition,
  ownership, invalid-input and paired-recovery checks pass byte-for-byte.
  The installed 16-slot target-to-draft test also passes: 43 actual target HC
  rows across cold/warm/decode boundaries match an explicitly shifted manual
  native feed exactly, without target replay.
  This internal alignment component has no CLI/server caller; it does not
  verify proposals, sample tokens or claim a generation speedup.

The tested independent numerical and internal correctness gates pass; these
implementation boundaries do not qualify accelerated production generation.
The [dated validation record](RELEASE_VALIDATION_2026-09-22.md) is authoritative
for which synthetic and installed checks have actually executed.

## Remaining integration and gates

The post-launch work adds an internal `FlashNextMTPGreedyVerifier` reference:
it checks proposals using ordinary sequential target decode, commits only the
matching prefix plus a target correction/bonus, and restores target, primer and
logits together after an interrupted round. Its output contract consumes all
returned tokens. It can now request native proposals from the aligned primer:
the target-selected seed is followed by drafts using each preceding draft's
owned full HC output. The speculative draft branch is restored before target
verification, so only verified target rows prime committed draft state. The
target-selected seed is excluded from draft-acceptance counts. This internal
path is **not** wired to CLI/server generation and makes no speed claim. It
deliberately does not substitute numerically different batched prefill for
exact target verification.
Its synthetic and installed 16-slot validation passes in the
[post-launch qualification record](POSTLAUNCH_QUALIFICATION.md);
this component does not clear any upstream-native or default-promotion gate.

1. **Resolved September 22:** the unrounded-FP32 synthetic numerical gate
   passes after widening fusion intermediates. Row zero's hidden error falls
   from 5.263% to 2.212% initially, then to 0.036193% with the final native
   FP32 intermediates; the 5% threshold is unchanged and its known-issue
   exemption is removed. All 40 hidden/logit rows and greedy choices pass.
   The independent upstream-component test additionally runs pinned SGLang
   fusion and actual Transformers decoder/QSA/HC/mixer code with the same
   dequantized weights. Its explicit lowest-index QSA tie contract passes all
   40 rows (worst hidden 2.2122%, logits 0.4004%). Unmodified CPU top-k differs
   on a zero-score tied boundary at row 39, giving 21.85% logit error; that
   negative result is retained, not counted as a pass. See the reproducible
   [reference harness and exact boundary](../Scripts/parity/README.md#native-mtp-component-reference-september-22).
   Stage readback remains opt-in. This is component-composition parity, not
   a CUDA/SGLang serving run or an accelerated generation path.
2. **Resolved for the tested aligned fixture September 22:** the
   [independent installed comparison](RELEASE_VALIDATION_2026-09-22.md#final-native-precision-comparison-passed)
   executes actual pinned upstream components on the installed sidecar and
   all 43 aligned target rows. After widening the native intermediates,
   worst hidden/logit error is 0.041973%/0.037470%, all greedy choices agree,
   and there are no QSA boundary ties or adaptations. The 5% bounds remain
   unchanged. Earlier failing comparisons are preserved in the record.
   The target-row-to-draft connection, complete prefix priming and shifted
   token/position checks are implemented internally. In the pinned
   [upstream prefill worker](https://github.com/sgl-project/sglang/blob/745de73ba3c136b6f99b7a3e2177ed1a8eef4a56/python/sglang/srt/speculative/eagle_worker_v2.py#L928-L1045),
   embeddings shift one token left, using the next prompt chunk's first token
   or the target's next generated token at the tail. A chunk's last pair must
   not be invented from its last token. The primer enforces this convention;
   it is not an enabled speculative generation path or an upstream numerical
   golden. See the [September 21 record](RELEASE_VALIDATION_2026-09-21.md).
3. Accelerate target verification against the sequential reference above and
   qualify accepted-prefix replay/rollback in production generation. Ordinary
   chunked prefill is not automatically an exact speculative verifier: its
   numerical and greedy equivalence must be demonstrated for this use.
4. Extend the reference's passing accepted/rejected proposal, stop-token,
   context and paired-recovery checks to actual native drafts and generation:
   EOS, stop strings, cancellation, prefix reuse and model switching.
   Compare against MTP-off target results, including all recurrent state.
5. Extend the independent comparison beyond the tested aligned fixture,
   including the installed sidecar's sparse-selection boundary. Measure
   acceptance and completed-answer time on separate resident
   and bounded-memory profiles. Keep MTP off unless its benefit exceeds run
   variability without changing the target result or memory guarantees.

The loader, input-fusion and rollback components alone satisfy none of the final
speed/default-promotion gates. No additional model download is needed for the
current host's installed sidecar.

The corrected finite synthetic prefill gate has bounded numerical drift, not
byte-exact sequential equivalence. It must not be used to justify an exact
speculative verifier; see the alignment correction in the dated validation log.

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
