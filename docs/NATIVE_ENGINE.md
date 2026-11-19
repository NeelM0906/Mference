# Native engine

The native engine runs GGUF-derived models on Mference's own Metal kernels,
replacing llama.cpp one model at a time behind `EngineModel`. Its first target
is **Quick**, Qwen3.5-4B UD-Q4_K_XL (llama.cpp architecture `qwen35`). Deep,
Qwen3.8-27B IQ3_S-mtp (also `qwen35`), reuses every layer below and is next.
llama.cpp at the pinned commit (`Sources/LlamaCpp`, b1b770d9) stays in the
tree. It is the golden reference, and it remains the fallback until the native
path passes the parity bar and beats it.

This page is the architecture. The frozen interfaces are the Swift protocols
and value types in
[`Sources/Mference/Native/Contracts/`](../Sources/Mference/Native/Contracts/),
all nested in the `Native` namespace (`Native.WeightStore`,
`Native.QuantGEMM`, and so on).

Related pages, each landing with the component that owns it:
- `docs/NATIVE_PARITY.md`: the parity bar and how to run it.
- `docs/NATIVE_BENCHMARKS.md`: benchmark protocol and baselines.
- `docs/MFERENCE_PACK.md`: the `.mfpack` format.

## Decisions this design rests on

- **Parity bar (D1).** The engine is measured against llama.cpp at the pin.
  - Greedy token agreement must be at least 99% on the fixed prompt set.
  - Per-token KL divergence of the logits must be at most ε, with ε set in
    `NATIVE_PARITY.md`.
  - Agentic-benchmark results must match.
  - Inside Mference the rule is still byte identity. A change must match its
    own control byte for byte, and any speculative path must match the same
    engine's plain greedy decode.
- **Weights (D3).** Weights come from a one-time exact repack of the GGUF
  blocks into kernel tile order, with no requantization and hash checks
  against the source. The result is an `.mfpack` install directory. Running a
  GGUF in place is not the product path.
- **KV precision and attention (D4).** FP16 KV and dense attention are the
  defaults. INT8 KV and sparse attention are opt-in.
- **Hardware (D5).** Every M-series chip is first-class. Every fast path must
  run correctly and be measured on Apple7 (M1). No fast path may require
  Apple9 or later.

## The six layers

| Layer | Contract | Phase 1 owner | Code |
|---|---|---|---|
| Weight store | `WeightStore`, `QuantTensor`, `PackManifest`, `TileLayout`, `TokenizerSource`, `TextTokenizer`, `ChatTemplateSource`, `Qwen35Config` | W1 `mfe-weights` | `Native/Weights/`, `MferenceRepack/Core/Pack/` |
| Kernel library | `QuantGEMM`, `KernelCalibration`, `AttentionOp`, `GDNOp`, `GlueOps` | W2 `mfe-gemm` (GEMM), W3 `mfe-attn` (attention, GDN) | `Native/Kernels/`, `Metal/Native/` |
| Family runner | `ForwardModel`, `TapPoint`, `TensorTap` | Phase 2 | `Native/Runner/` |
| State manager | `KVStore`, `MemoryPlan`, `MemoryPlanner`, `KVPageGeometry`, `RecurrentLayout` | W4 `mfe-memory` | `Native/Memory/` |
| Decoding strategies | `DecodingStrategy` (plain only in this program), `Sampler` | W5 `mfe-sched` | `Native/Sampling/` |
| Scheduler | `StepScheduler`, `GenerationJob`, `StepTimings` | W5 `mfe-sched` | `Native/Scheduler/` |

Every contract has a slow, frozen reference implementation in
`Native/Reference/`, written in Phase 0. The references are naive Metal
kernels: one direct port of ggml's dequantization per type, F32
accumulation, and no tiling. Each has a pure-Swift CPU twin for unit tests.
Together they form an all-reference `qwen35` path that runs Quick end to end.
A Phase 1 worker tests its fast component against the references for every
other layer, so no worker waits on another.

## The `qwen35` model (Quick)

Quick has 32 blocks with a hidden size of 2560. Block `i` is full attention
when `(i + 1) % 4 == 0`, which gives 8 full-attention blocks (3, 7, …, 31).
The other 24 blocks are Gated DeltaNet (GDN). The vocabulary has 248,320
tokens. The LM head is tied to `token_embd.weight` (Q6_K), and Quick has no
`nextn` blocks.

| Tensor (per block) | Type in Quick | Shape `ne` | Used by |
|---|---|---|---|
| `attn_norm`, `post_attention_norm` | F32 | 2560 | `GlueOps.encodeRMSNorm` |
| `attn_q` (full) | Q5_K | 2560 → 8192 (q and gate) | `QuantGEMM` |
| `attn_k`, `attn_v` (full) | Q5_K, Q6_K | 2560 → 1024 | `QuantGEMM` |
| `attn_q_norm`, `attn_k_norm` (full) | F32 | 256 | `AttentionOp` |
| `attn_output` (full) | Q4_K | 4096 → 2560 | `QuantGEMM` (accumulate into the residual) |
| `attn_qkv` (GDN) | Q5_K and others | 2560 → 8192 | `QuantGEMM` |
| `attn_gate` (GDN, z) | Q5_K and others | 2560 → 4096 | `QuantGEMM` |
| `ssm_alpha`, `ssm_beta` (GDN) | F16 | 2560 → 32 | `QuantGEMM` |
| `ssm_conv1d`, `ssm_dt.bias`, `ssm_a`, `ssm_norm` (GDN) | F32 | — | `GDNOp` |
| `ssm_out` (GDN) | Q8_0 | 4096 → 2560 | `QuantGEMM` (accumulate) |
| `ffn_gate`, `ffn_up` | Q4_K / Q5_K | 2560 → 9216 | `QuantGEMM` |
| `ffn_down` | Q6_K | 9216 → 2560 | `QuantGEMM` (accumulate) |
| `token_embd` | Q6_K | 2560 x 248320 | `QuantGEMM.encodeGetRows` and the LM head |

Per-tensor types vary by layer; the manifest is authoritative. Across the
pack, Quick contains Q4_K (74 tensors), Q5_K (69), Q6_K (24), Q8_0 (24),
IQ4_XS (10), F16 (48) and F32 (177). The fast `QuantGEMM` therefore covers
Q4_K, Q5_K, Q6_K, Q8_0 and IQ4_XS, plus F16 for the two small GDN gate
projections.

### Precision

- Activations and the residual stream are F32, as in ggml's Metal graph.
- GEMM inputs and outputs are F32. Kernels may stage inputs in half precision
  internally, as ggml-metal's `mul_mm` does.
- KV is stored as F16, or as INT8 plus an F16 scale when that is opted in.
- GDN conv state and delta-rule state are F32.
- Logits are F32.
- `docs/NATIVE_PARITY.md` sets per-op tolerances for each op type.

## Data flow of one step

A step processes `rows` tokens at positions `p ..< p + rows`. A decode step
has `rows == 1`, a verify step has 2 to 16 rows, and a prefill chunk has up to
`maxPrefillChunk` rows. The same encode path serves all three. Only the
`QuantGEMM`, `AttentionOp` and `GDNOp` entry points differ
(`encodeGEMV`/`encodeDecode`, `encodeVerify`, `encodePrefill`).

1. The scheduler calls `kv.markStepStart` and `kv.reserve(rows:)`. It opens
   one command buffer with one compute encoder and calls
   `ForwardModel.encodeStep`.
2. **Embedding.** `QuantGEMM.encodeGetRows(token_embd, tokens)` writes the
   residual stream `x` [rows x 2560]. The tokens come from the host, or, in a
   pipelined decode, from the `TokenSlot` the previous step's
   `Sampler.encodeChoice` wrote.
3. **Each block `i`:**
   1. `h = RMSNorm(x, attn_norm)`
   2. A full-attention block runs:
      1. `qGate = attn_q · h`, `k = attn_k · h`, `v = attn_v · h`
      2. `AttentionOp`: per-head q/k RMSNorm, RoPE on the first 64 dims, K/V
         appended to the pages, causal GQA attention, then
         `× sigmoid(gate)`. This gives `a` [rows x 4096].
      3. `x += attn_output · a`
   3. A GDN block runs:
      1. `qkv = attn_qkv · h`, `z = attn_gate · h`, `α = ssm_alpha · h`,
         `β = ssm_beta · h`
      2. `GDNOp`: conv1d with the conv state and SiLU, then the q/k/v split,
         q/k L2 norm, the decay and beta gates, the delta-rule recurrence, and
         finally RMSNorm(`ssm_norm`) `× silu(z)`. This gives `g` [rows x 4096].
      3. `x += ssm_out · g`
   4. `h = RMSNorm(x, post_attention_norm)`
   5. `u = silu(ffn_gate · h) × (ffn_up · h)` (two GEMMs and `GlueOps.encodeSwiGLU`)
   6. `x += ffn_down · u`
4. **Head.** `RMSNorm(x, output_norm)`, then the LM head `token_embd · x`. The
   head runs on the last row (`LogitRows.last`) or on all rows (`.all`, used
   for verify). The result is F32 [n x 248320].
5. **Choice.** On the GPU path the scheduler encodes
   `Sampler.encodeChoice(logits) → TokenSlot` into the same encoder. Otherwise
   it reads the logits back and calls `Sampler.sample`.
6. Commit. On a verify step, the scheduler reads the accepted count and calls
   `kv.rollback(to:)` for the rejected rows. Rollback restores recurrent
   state from verify-arena slot `r`.

A **prefill chunk** is one step with `rows = chunk` and `LogitRows.none`. The
prompt's last chunk uses `.last`. Long prompts are processed as several
chunks in sequence. GDN prefill is a chunked scan that ends with the live
state advanced past the chunk. Attention prefill reads the pages that
earlier chunks and this chunk wrote.

**Taps.** When `ForwardModel.tap` is set, the model copies each `TapPoint`
into the destination the tap returns:
- `embedding`
- `mixerInput`, `mixerOutput`, `ffnOutput` and `layerOutput` for every block
- `finalNorm`
- `logits`

This is how per-op parity against the llama.cpp tensor dumps runs.
`TensorTap` is a debug hook and is never set in production.

## Buffer ownership and lifetimes

| Memory | Owner | Lifetime | Notes |
|---|---|---|---|
| Weights | `WeightStore` | Model load to unload | `weights.bin` is mapped read-only and wrapped with `makeBuffer(bytesNoCopy:)`. Every `QuantTensor` is an offset into that buffer. Never copied. |
| KV arena, page table | `KVStore` | Sized by the `MemoryPlan` at load | One arena buffer holds every physical page, laid out per `KVPageGeometry` (64 tokens per page, `[attnLayer][K/V][kvHead][token][dim]`). Quick FP16 is 2 MiB per page, 32 KiB per token. |
| Recurrent state | `KVStore` | Same | Per GDN layer, laid out per `RecurrentLayout`: F32 conv tail `[3][8192]`, then F32 S `[32][128][128]`. Quick uses 24 x ~2.1 MiB. |
| Verify arena | `KVStore` | Same | `maxVerifyRows` slots per GDN layer. In this program `maxVerifyRows` is 1 (plain decoding), so the arena exists but is unused. |
| Prefix cache | `KVStore` | Across requests, bounded by the plan | A radix tree over token ids. Each node holds shared, ref-counted KV pages and a recurrent snapshot, in RAM or on the SSD tier. |
| Activations and scratch | `ForwardModel` | Model lifetime, reused every step | Sized for `maxPrefillChunk` rows. Logits returned by `encodeStep` stay valid until the next `encodeStep`. |
| Token slots | Scheduler | Model lifetime | Shared-storage `u32` ring used by pipelined decode. |
| Pipelines | Each op | Built in `prepare`, never while encoding | |

A queued command buffer owns every buffer region it touches until it
completes. The CPU must not rewrite a staging buffer, a token slot or a page
a pending command buffer reads. This is the same slot-ownership invariant as
the rest of Mference. With pipelining, the scheduler double-buffers the
regions the CPU writes.

`MemoryPlanner.resolve` runs before any allocation. It accounts for the
weights mapped, the KV arena at the resolved context, recurrent state, the
verify arena, activations, the RAM prefix cache and everything else. It
refuses with `memoryBudgetExceeded` instead of allocating past the budget.
One model is resident at a time.

## Threading and command buffers

- **One generation thread per model.** The scheduler owns a dedicated
  high-QoS thread (`userInteractive`), not the Swift cooperative pool. A
  measured gap in the earlier engine program came from running llama.cpp
  calls on the pool. Every encode, commit, readback and `KVStore` mutation
  happens on that thread. Contract objects other than the scheduler are not
  `Sendable` and are confined to it.
- **One command buffer per step** on `MetalContext`'s queue, holding one
  compute encoder for the model's work. `KVStore` methods that take a command
  buffer (rollback, snapshot, prefix attach and insert) encode their own blit
  or compute work, either before the step's encoder opens or after it ends.
- **Pipelining.** When `Sampler.canChooseOnGPU` holds, the scheduler encodes
  and commits step t+1 while step t runs: step t+1's embedding reads step t's
  `TokenSlot`. The CPU reads the token afterwards for streaming and stop
  checks. If step t ends the generation, step t+1 has already run, so the
  scheduler rolls it back (`rollback(to:)`). Output must stay byte-identical
  to the sequential loop.
- **Encode rules for ops.** An op encodes compute work only, into the encoder
  it is given. It never ends the encoder, commits, waits, or allocates while
  encoding.
- **Cancellation** is checked between steps. A cancelled generation stops at a
  step boundary with the store consistent: the partial step is rolled back.
- **Shaders.** Native kernels compile on demand through
  `MetalContext.moduleLibrary(module:)`. The registered modules are:
  - `native_gemm`, `native_gemm_iq`, `native_attention`, `native_gdn`,
    `native_glue`, `native_reference` and `native_sampling`
  - Each is one file, `Metal/Native/<Area>/<module>.metal`.
  - They are never part of the shared runtime library, so their function
    names cannot collide with it.
  - A new module is a coordinator change to `shaderSubdirectories`.

## Error model

- Anything that crosses a contract boundary throws `Native.NativeError`:
  - `missing`, `invalidShape`, `unsupported` and `integrity` are raised at
    load or `prepare` time.
  - `stateOutOfRange` covers positions, pages and snapshots.
  - `memoryBudgetExceeded` comes from the plan.
  - `commandBufferFailed` is raised when the scheduler checks a completed
    command buffer's status.
  - `cancelled` signals a cancelled generation.
- Encode methods throw only for contract violations (shapes, row counts,
  layouts), never for GPU conditions.
- Components may keep local error enums internally.
- A failed command buffer ends the generation with an error. The scheduler
  then resets the sequence, because the KV and recurrent state can no longer
  be trusted. The prefix cache is unaffected, because it holds only
  snapshots that completed.

## Plugging into `EngineModel`

The native model joins `MferenceEngine` as one more `EngineModel`. The
adapter is Phase 2 work and lives in `MferenceEngine`, beside the native
runner:
- **load:** open the `.mfpack` (`WeightStore`), resolve the `MemoryPlan`,
  build the `KVStore`, the ops, the `ForwardModel`, the `Sampler`, the plain
  `DecodingStrategy` and the `StepScheduler`.
- **prepare:** render the chat template from `ChatTemplateSource`, using the
  same Jinja path as the llama.cpp backend. Then tokenize with `TextTokenizer`.
- **generate:** map `GenerationConfig` to `SamplingParams` and run a
  `GenerationJob`. Stream `StepEvent.token` as engine events, detokenizing
  with `TextTokenizer.piece`. Stop strings are matched here, and the adapter
  calls `cancel()` when one matches.
- **unload:** release the scheduler, the store and the mapping.

The catalog keeps the llama.cpp Quick entry as the fallback until the Phase 2
exit gate. The product slot chooses the backend; the API does not change.

## Glue

`GlueOps` covers RMSNorm, add, SwiGLU and copy. It sits between the
contracted ops and is not a Phase 1 component. The reference implements it.
The fast version is a Phase 2 integration item: an adapter over the existing
`rmsnorm` and `utility` kernels, or work fused into the GEMMs. The Phase 2
coordinator assigns it to a Phase 1 worker thread.

## Tests and gates

- Swift file names must be unique within a target (SwiftPM rejects a
  second `Sampler.swift` in `Mference`), so native files carry a prefix:
  `Native*.swift` for contracts, and the owning area's prefix elsewhere (for
  example `ReferenceQuantGEMM.swift`, `PackWeightStore.swift`).
- New tests live under `Tests/Mference/Core/Native/<Area>/`, in the
  `MferenceTestsCore` target. They must be small and fast, and they load no
  model unless env-gated.
- Real-model checks (packs, goldens, the all-reference Quick path) are gated
  on environment variables documented in `NATIVE_PARITY.md`.
- The PR gates are L1 to L7 of the program spec: release build, acceptance
  tests plus the needed test tier, static checks, attribution, benchmark gate,
  memory, and docs.
- Kernel PRs also pass their benchmark gate against ggml-metal at the pin on
  the M1 Max.
- The scheduler's gate is measured with a synthetic `ForwardModel` whose step
  is a fixed GPU kernel of about 10 ms. It must meet all of the following:
  - median GPU idle gap between consecutive decode command buffers of at most
    50 µs
  - p99 gap of at most 250 µs
  - median scheduler CPU time per step, excluding the model's encode, of at
    most 50 µs

## Contract change rule

The contracts were frozen by the commit
`contracts: freeze native engine interfaces v1` (tag
`native-contracts-v1`). After the freeze, a contract changes only through a
coordinator PR that names every affected worker. Workers that need a change
send the exact request to the coordinator and keep working against the frozen
contract in the meantime.
