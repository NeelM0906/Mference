# Runtime diagnostics

Phase 1 adds executed-prefill reporting and a post-generation memory snapshot.
It does not change checkpoint selection, kernels, prefill policy, or fallbacks.

## Enable

Set `MFERENCE_DIAGNOSTICS=1` on an otherwise unchanged CLI or server launch.
Follow the model-process checks in AGENTS.md first; do not start another model
process alongside a running server.

For example, prefix your existing command with `MFERENCE_DIAGNOSTICS=1`.
The CLI (raw completion and interactive chat) writes a separate
`[runtime-diagnostics] {JSON}` line to stderr. It also does so with
`--quiet` when explicitly enabled. The server writes
`request <id> runtime-diagnostics {JSON}` to its operator log for both
streaming and non-streaming completed requests.

The existing timing footer, generated stdout, and OpenAI response/SSE schemas
are unchanged. Diagnostics contain no prompt text, generated text, token IDs,
or file paths. Disabled by default. Failed/cancelled requests do not emit a
successful completion snapshot. Counts are available on library results
independently of whether logging is enabled.

## Executed prefill

`RawDecodeResult.prefillExecution` carries the report returned by the runner.
`cachedPromptTokens` and `computedPrefillTokens` remain separate; cached tokens
and subsequent decode tokens never contribute to the prefill report.

| Field under `prefill` | Meaning |
| --- | --- |
| `executedMode` | `chunked`, `sequential`, `mixed`, or `off` for no computed work |
| `batchedTokens` | Tokens executed by the batched prefill path |
| `replayedTokens` | Tokens executed through the per-token path |
| `batchedChunkSizes` | Actual batch sizes in execution order, including ragged/cutover chunks |
| `replayReasons` | Reason codes mapped to the number of replayed tokens |

Counters are local to each completed prefill call, not cumulative runner
statistics. Token-ordered recurrent operations *within* a layer-major batch do
not count as full-model token-by-token replay. Chunk sizes describe execution,
not an inference from the requested chunk size; GLM's resident engine currently
uses its own capacity.

For example, the DeepSeek synthetic 60-token cutover regression now reports:

```json
{"executedMode":"chunked","batchedTokens":60,"replayedTokens":0,"batchedChunkSizes":[32,28],"replayReasons":{}}
```

This is a test fixture, not a real-checkpoint benchmark. Before GPU sparse
selection, it reported 51 batched / 9 replayed tokens. Mixed reports remain
supported for paths that actually mix execution modes.

Known reasons:

- `prefill_disabled`: explicitly requested sequential prefill.
- `inkling_reference_override`: Inkling's existing sequential reference override.
- `glm_streamed_experts`: GLM's current streamed-expert fallback.
- `glm_batched_prefill_disabled`, `glm_capture_reference`,
  `glm_dense_selection_reference`, `glm_batched_engine_unavailable`:
  other GLM eligibility/reference conditions.
- `deepseek_sparse_selection_cutover`: historical lightning-indexer fallback;
  production DeepSeek now batches above the cutover and no longer emits it.
- `deepseek_batched_prefill_disabled`, `deepseek_batched_engine_unavailable`,
  `deepseek_expert_slots_unavailable`, `deepseek_unsupported_geometry`:
  other DeepSeek eligibility conditions.

A legacy/uninstrumented producer yields explicit JSON `"prefill":null`.
Never interpret this as evidence of batching. The old factory property
`ForwardRuntime.executedPrefillMode` is deprecated and returns `unreported`:
constructing a runner cannot establish what a later request actually executes.

## Memory

`memory.bytes` contains byte counts; unavailable values are explicit `null`.
The snapshot is taken after generation, not at the prefill peak.

| Metric | Scope |
| --- | --- |
| `mappedCoreWeightBuffers` | Unique mapped core Metal buffer capacities |
| `mappedExpertRegions` | Already-opened expert mapping lengths, including alignment |
| `copiedExpertBuffers` | Owned resident expert buffer capacities, deduplicated across views |
| `expertSlotBuffers` | Already-created streamed expert-cache slab capacities |
| `expertMetadataBuffers` | Slot/identity-table buffer capacities |
| `convertedWeightBuffers` | Already-created converted weight buffers |
| `targetKVStateBuffers` | Target model KV, recurrent, convolution and paged-state buffers, including reserved capacity |
| `completionScratchBuffers` | Completion logits, probabilities and output-token buffers |
| `runnerScratchBuffers` | Currently `null`: complete family scratch ownership is not yet inventoried |
| `metalAllocated` | Metal's current process allocation count on the selected device |
| `processPhysicalFootprint` | Current Mach process physical-footprint counter |
| `processRSS` | Current Mach process resident-size counter |
| `systemPhysicalMemory` | Installed system RAM, for context only |
| `mappedWeightsPhysicalResidency` | `null`: mapping length does not measure physical residency |
| `filesystemCache` | `null`: no reliable model-specific filesystem-cache measurement |

These categories overlap. **Do not sum them.** In particular, Metal capacities
are not additional memory on top of process counters, and per-expert views
must not be added again to their underlying mapping or copied slab.

The target KV count excludes speculative draft state. Process/Metal counters
can include drafts, other live allocations and retained caches. There is no
claim of a complete application working-set inventory or minimum-RAM estimate.
Reading the snapshot does not open lazy expert streamers or read weight pages
to manufacture residency statistics.

The JSON envelope has `schemaVersion:1`; consumers should tolerate additive
fields and reason codes. Memory `scope` explicitly says the categories overlap
and the snapshot is not a peak.
