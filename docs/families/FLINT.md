# Flint-40B-A2.5B on Mference

Checkpoint selection, architecture contract, memory budget, and measured
results for running [`OsaurusAI/Flint-40B-A2.5B-JANG_4M`](https://huggingface.co/OsaurusAI/Flint-40B-A2.5B-JANG_4M)
with SSD-streamed experts. This document is the architecture contract for the
`flint` family.

| | |
|---|---|
| Family identifier | `flint` (`ModelFamily.flint`) |
| Source repository | `OsaurusAI/Flint-40B-A2.5B-JANG_4M` |
| Pinned revision | `2cc33f8d9d211217febf00e1cbf503c3eac7b16e` |
| Parameters | 40B total, 2.5B active |
| Install size | 23,600,000,000 bytes reserved |
| Status | in port |

## Checkpoint selection

Flint was published by RecursiveSI under Apache-2.0. The JANG_4M conversion is
a third-party MLX affine repack calibrated against Flint-generated traces. It
was selected for the first port because it matches the runtime's mixed
INT8/INT4 contract and fits a 24 GiB Apple-Silicon host while keeping the
routed-expert pool streamable.

### Rejected candidates

| Candidate | Routed experts | Attention / embeddings | Disk | Verdict |
|---|---|---|---|---|
| `OsaurusAI/Flint-40B-A2.5B-JANG_4M` | affine INT4 g64 | resident projections INT8 g64, embedding INT8 | 23.4 GB | **Selected** for the first port. |
| `RecursiveSI/flint-int4` | affine INT4 g64 | resident projections INT4 g64, embedding INT8 | 22.6 GB | Deferred. The architecture is identical, but its resident projections need a second runner quant path. |
| BF16 `RecursiveSI/flint` | BF16 | BF16 | ~79.6 GB | Rejected: too large for the target host and outside the streaming-expert budget. |

The installed JANG source has 1751 tensors across six weight shards. Resident
core, attention, shared expert and head weights are INT8 group-64; the 47
routed layers are INT4 group-64; routers, norms and attention sinks are BF16.

**License note.** Apache-2.0. The pinned conversion is redistributed by its
uploader. No upstream implementation source is copied into Mference.

## Architecture contract

| Axis | Value | Covered by existing kernel? | Gap |
|---|---|---|---|
| `family` | `flint` | yes | — |
| `numLayers` | 49 | yes | 2 leading dense FFNs, 47 MoE layers. |
| `hiddenSize` / `intermediateSize` | 2048 / 1024 | yes | Shared-expert width. |
| `moeIntermediateSize` | 512 | yes | Routed FFN width. |
| `numHeads` / `numKVHeads` | 32 / 4 | yes | GQA 8:1. |
| `headDim` | 128 | yes | Full 128-wide RoPE on sliding layers. |
| `slidingWindow` | 513 | yes | SWA ring includes the current token. |
| `fullAttentionLayerMask` | 14 full / 35 sliding | yes | Full layers are `0,1,5,...,45,48`. |
| `noRopeLayerMask` | equals full mask | yes | Full attention is NoPE. |
| `attentionSinks` | true | yes | `softmax_offset`, denominator only. |
| `numExperts` / `topKExperts` | 256 / 6 | yes | Sigmoid scores, top-6 renormalized. |
| `numSharedExperts` | 1 | yes | Ungated shared expert. |
| `numDenseLayers` | 2 | yes | Layers 0 and 1. |
| `routerScoringFunc` | `sigmoid` | yes | BF16 router gate. |
| `routerNormAfterTopK` | true | yes | Renormalize selected weights. |

| Quirk | Where it lives | Why it is not an axis |
|---|---|---|
| Chat excludes BOS | `Tokenizer.swift` | Serving contract of this checkpoint, not an architecture axis. |
| `developer` is a native role | `Tokenizer.swift` | Flint tokenizer contract. |
| GLM XML tool calls | `FlintToolCallParser.swift` | Dialect vocabulary, not model geometry. |
| Full layers are explicitly NoPE | `noRopeLayerMask` | This is an axis; it is never inferred from `layer_types`. |

## Memory budget

| Group | Bytes |
|---|---:|
| Resident core / attention / shared expert / head | ~1.9 GB |
| Routed experts | ~20 GB |
| **Installed total** | **~23.4 GB** |

### KV cache

Full-attention KV is FP16. At 128K context the 14 full layers hold roughly
3.5 GiB of K+V before allocator and page overhead; sliding layers use a
513-token ring.

### Expert-cache ladder

| Slots | Resident expert bytes | Notes |
|---|---:|---|
| 16 | ~1.3 GB | Floor; initial smoke rung. |
| 32 | ~2.7 GB | More all-hit layers at additional RAM. |
| 96 | ~8.0 GB | Near-resident rung; measured on a 24 GiB host only. |

## Port status

- [x] **Repack** — pinned source, mixed quant contract, expert layout and
  manifest validation.
- [x] **Toy parity** — attention sink/SWA, router, tokenizer parser, model
  schema and repack planner suites.
- [x] **Runner** — decode, sequential prefill, batched prefill, paged KV and
  continuation paths implemented.
- [x] **Ladder** — 16 / 32 / auto expert-cache slots produced byte-identical
  greedy output; bring-up stages 0–4 passed on the pinned install.
- [ ] **Gate** — full suite, community protocol and phases snapshot remain to
  be completed on the clean branch.

## Measured results

The bring-up ladder on a 24 GiB M5 measured 9.59 / 12.88 / 11.59 tok/s at
16 / 32 / auto expert-cache slots with the 5-token `The capital of France is`
prompt and a 24-token cap. The three outputs were byte-identical.

With the chat template and a 15-token prompt, both the linear and paged-KV
paths produced `The capital of France is Paris.`; the measured decode rates
were 8.23 and 9.06 tok/s respectively. These are first-light observations, not
community-protocol benchmark rows.

## Known limits

- The first source is JANG_4M. Official `RecursiveSI/flint-int4` has the same
  architecture but uses INT4 resident projections, which the first runner path
  does not yet accept.
- The 262,144-token maximum is declared by the checkpoint; long-context
  passkey and full-boundary execution remain to be measured.
- The source's `kv_key_bits` / `kv_value_bits` fields exist in its config, but
  `flint-int4.json` declares BF16 KV and this port keeps FP16 runtime KV.
- The community protocol benchmark and phases attribution are pending.

## Reproduction

```bash
swift build -c release

swift run -c release MferenceRepack \
  --model flint \
  --output scratch/flint40b.gturbo

./bringup-check.sh flint scratch/flint40b.gturbo

.build/release/MferenceCLI \
  --model scratch/flint40b.gturbo \
  --prompt "The capital of France is" \
  --max-new 64 --temperature 1 --top-k 20 --top-p 0.95
```
