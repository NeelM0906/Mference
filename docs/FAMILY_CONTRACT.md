# The family contract

`ArchConfig` in
[`Sources/Mference/Infrastructure/ModelIO/ModelTypes.swift`](../Sources/Mference/Infrastructure/ModelIO/ModelTypes.swift)
is the declarative contract a model family satisfies. A family is a
selection over its axes: the runner's branches, the kernels it dispatches,
and the tensor shapes it expects are all conditioned on these fields, and
`manifest.json -> arch` must match the compiled baseline field-by-field or
`Model.load` throws `ModelError.archMismatch`.

This document is the canonical enumeration of those axes. It is
rot-checked: `FamilyContractDocTests` in
[`Tests/Mference/Core/Infrastructure/ModelIO/FamilyContractDocTests.swift`](../Tests/Mference/Core/Infrastructure/ModelIO/FamilyContractDocTests.swift)
reflects over every registered family config, collects the stored-property
names of `ArchConfig` and of the nested config structs it holds, and fails
if any of them is missing from this file. Adding an axis without a row here
breaks the suite; the failure names the undocumented fields.

The enumeration is of axes, not of behavior. What a kernel does with an
axis lives in the family's own page —
[Gemma 4](../README.md), [DeepSeek-V4-Flash](DEEPSEEK_V4_FLASH.md),
[Inkling-Small](INKLING_SMALL.md), [Qwen 3.8](QWEN38_LONG_CONTEXT.md) —
and in [System design](SYSTEM_DESIGN.md).

## Reading the tables

Gemma 4 is the baseline. Its values are the `ArchConfig.init` defaults for
every axis that has one, so a legacy manifest that omits the family
extensions still validates as Gemma 4. Each table gives the Gemma 4 value
and then only the families that differ from it. Values are the compiled
baselines, not a supported range: an axis with one observed value is
documented as what has been exercised, not as what is permitted.

| Tag | Family | `arch.family` | Baseline constant |
| --- | --- | --- | --- |
| G4 | Gemma 4 26B-A4B | `gemma4` (absent in legacy manifests) | `ArchConfig.gemma4_26B_A4B` |
| Q36 | Qwen 3.6 35B-A3B | `qwen36` | `ArchConfig.qwen36_35B_A3B` |
| Q38 | Qwen 3.8 27B | `qwen38` | `ArchConfig.qwen38_27B` |
| DSV4 | DeepSeek-V4-Flash 284B-A13B | `deepseekV4Flash` | `ArchConfig.deepseekV4Flash_284B_A13B` |
| INK | Inkling-Small 276B-A12B | `inklingSmall` | `ArchConfig.inklingSmall_276B_A12B` |
| MPL | Maple Preview | `maple` | `ArchConfig.maplePreview` |

`ArchConfig.knownArchitectures` maps `arch.family` to the baseline used for
auto-detection at load.

## Family identity

| Field | Type | Selects | G4 | Divergence |
| --- | --- | --- | --- | --- |
| `family` | `ModelFamily` | The tensor-name contract, the layer graph shape (norm sandwich vs plain pre-norm), and family-specific kernel behavior. Stored in `manifest.json -> arch.family`; absent means Gemma 4, the format's original architecture. | `.gemma4` | One value per family; every family diverges by definition. |

## Dimensions

Sliding-window and full-attention layers are sized independently, so the
model carries both a sliding pair (`numKVHeads`, `headDim`) and a full pair
(`numFullKVHeads`, `fullHeadDim`). A family with no sliding-window layers
mirrors the full values into the sliding slots, where they are never used
to size storage.

| Field | Type | Selects | G4 | Q36 | Q38 | DSV4 | INK | MPL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `hiddenSize` | `Int` | Residual width. | 2816 | 2048 | 5120 | 4096 | 4096 | 2048 |
| `intermediateSize` | `Int` | Shared-expert FFN width. Mirrored as `ffnIntermediate` in the manifest. | 2112 | 512 | 17408 | 2048 | 2048 | 512 |
| `moeIntermediateSize` | `Int` | Per-expert FFN width. | 704 | 512 | 0 | 2048 | 2048 | 512 |
| `numLayers` | `Int` | Layer count; also the length of `fullAttentionLayerMask`. | 30 | 40 | 64 | 43 | 42 | 24 |
| `numHeads` | `Int` | Query heads. | 16 | 16 | 24 | 64 | 32 | 16 |
| `numKVHeads` | `Int` | KV heads on sliding-window layers. | 8 | 2 | 4 | 1 | 8 | 4 |
| `numFullKVHeads` | `Int` | KV heads on full-attention layers. | 2 | 2 | 4 | 1 | 8 | 4 |
| `headDim` | `Int` | Head width on sliding-window layers. | 256 | 256 | 256 | 512 | 128 | 128 |
| `fullHeadDim` | `Int` | Head width on full-attention layers. | 512 | 256 | 256 | 512 | 128 | 128 |

DSV4's `numKVHeads` and `numFullKVHeads` are 1 because its attention is
shared-KV MQA: one 512-dim KV head read as both K and V.

## Attention

### Layer kinds and windowing

| Field | Type | Selects | G4 | Divergence |
| --- | --- | --- | --- | --- |
| `fullAttentionLayerMask` | `[UInt8]` | Per-layer attention kind. 0 = sliding-window, 1 = full, 2 = gated-DeltaNet linear, 3 = compressed sparse attention (CSA), 4 = heavily compressed attention (HCA). Values 3 and 4 additionally include the sliding-window branch, because DeepSeek V4 concatenates compressed entries onto the window KV. | `1` on layers 5, 11, …, 29; `0` elsewhere (5 of 30 full). | Q36/Q38: `2` everywhere, `1` on every 4th layer. DSV4: `0` on layers 0–1, then `3` on even and `4` on odd layers. INK: `1` on layers 5, 11, …, 41; `0` elsewhere. MPL: `1` on every 4th layer, `0` elsewhere. |
| `slidingWindow` | `Int` | Sliding-window width. | 1024 | Q36/Q38 `0` (no sliding layers). DSV4 `128`. INK `512`. MPL `512`. |
| `attentionKEqV` | `Bool` | Full-attention K and V share the `k_proj` weight, so one dequant and GEMV produces the raw projection for both. The K and V cache slots stay separate regardless — they diverge at the norms and RoPE. | `true` | Q36, Q38, INK, MPL `false`. DSV4 `true` in the strongest sense: K and V are the same cache entry. |
| `attnOutputGate` | `Bool` | Full-attention `q_proj` emits `2 * numHeads * fullHeadDim` rows as per-head [query ; gate] halves, and the attention output is multiplied by sigmoid(gate) before `o_proj`. | `false` | Q36, Q38 `true`. |
| `attentionScale` | `Double` | Softmax scale for full attention. | 1.0 | Q36/Q38 0.0625 (256^-0.5). DSV4 0.044194173824159216 (512^-0.5). MPL 1/sqrt(128). INK 1/128 — it RMS-normalizes q and k per head, so the scale is 1/d, not 1/sqrt(d). |

### Rotary position

| Field | Type | Selects | G4 | Divergence |
| --- | --- | --- | --- | --- |
| `ropeTheta` | `Double` | RoPE base on sliding-window layers. | 10000.0 | Q36/Q38 1.0e7. DSV4 10000.0 (the `main` rope; CSA/HCA layers use `compressRopeTheta`). INK 0.0 — no RoPE at all. MPL 10000.0. |
| `fullRopeTheta` | `Double` | RoPE base on full-attention layers. | 1000000.0 | Q36/Q38 1.0e7. DSV4 10000.0. INK 0.0. MPL 0.0 — its global layer is NoPE. |
| `partialRotaryFactor` | `Double` | Fraction of each head's channels that rotate; the rotary width is `headDim * partialRotaryFactor`. | 0.25 | Q36/Q38 0.25. DSV4 0.125 (64 of 512). INK 0.0. MPL 0.5. |
| `ropeNeoxSubdim` | `Bool` | Partial-RoPE convention. `false` (Gemma): pairs (i, `headDim`/2 + i) for i below the rotated-pair count, frequency divisor `headDim`. `true` (Qwen / NeoX sub-dim): rotation confined to the first `rotaryDim` elements, pairing (i, `rotaryDim`/2 + i), frequency divisor `rotaryDim`. | `false` | Q36, Q38, MPL `true`. DSV4 `false`, but with its own interleaved-trailing convention — neither Gemma's proportional nor Qwen's sub-dim layout; the family's kernels implement it. |

### Short convolution

| Field | Type | Selects | G4 | Divergence |
| --- | --- | --- | --- | --- |
| `sconvKernelSize` | `Int` | Depthwise short-convolution width applied to the block inputs and to the K/V streams. 0 disables the short-conv path entirely. | 0 | INK 4. |

### `linearAttention` — `LinearAttentionConfig`

Gated-DeltaNet dimensions for layers with mask value 2. `.none` (all
fields zero) for architectures without linear-attention layers, which is
every family except Qwen 3.6 and Qwen 3.8.

| Field | Type | Selects | Q36 | Q38 |
| --- | --- | --- | --- | --- |
| `numKHeads` | `Int` | Key heads. | 16 | 16 |
| `numVHeads` | `Int` | Value heads. | 32 | 48 |
| `keyHeadDim` | `Int` | Key head width. | 128 | 128 |
| `valueHeadDim` | `Int` | Value head width. | 128 | 128 |
| `convKernelSize` | `Int` | Depthwise conv width on the fused qkv stream. | 4 | 4 |

Two derived widths follow from these and are not separate axes: `qkvDim`
(`2 * K-dim + V-dim`, the fused qkv projection rows and the depthwise conv
channel count) and `valueDim` (the z-gate projection rows and `out_proj`
columns).

### `compressedAttention` — `CompressedAttentionConfig`

DeepSeek-V4 compressed-attention dimensions, for layers with mask values 3
and 4 and for the family's shared-KV MQA sliding layers. `.none` for every
family except DSV4.

V4 attention is shared-KV MQA with a low-rank query path, a grouped
low-rank output projection, per-head learnable attention sinks, and
interleaved partial RoPE on the trailing `ropeHeadDim` channels of each
head. CSA layers pool every `csaCompressRate` source tokens into one
compressed KV entry — two overlapping series — and gather the top
`indexTopK` entries per query with a lightning indexer. HCA layers pool
every `hcaCompressRate` tokens non-overlapping and attend densely over the
result. Sliding layers rope at the `ArchConfig` `ropeTheta`; CSA and HCA
layers and their compressors rope at `compressRopeTheta`.

| Field | Type | Selects | DSV4 |
| --- | --- | --- | --- |
| `qLoraRank` | `Int` | Rank of the low-rank query path. | 1024 |
| `oLoraRank` | `Int` | Rank of the output projection. | 1024 |
| `oGroups` | `Int` | Groups the output projection is split into. | 8 |
| `ropeHeadDim` | `Int` | Trailing channels per head that carry interleaved partial RoPE. | 64 |
| `indexNHeads` | `Int` | Lightning-indexer heads. | 64 |
| `indexHeadDim` | `Int` | Lightning-indexer head width. | 128 |
| `indexTopK` | `Int` | Compressed entries gathered per query on CSA layers. | 512 |
| `csaCompressRate` | `Int` | Source tokens pooled into one compressed KV entry on CSA layers, in two overlapping series. | 4 |
| `hcaCompressRate` | `Int` | Source tokens pooled non-overlapping on HCA layers. | 128 |
| `compressRopeTheta` | `Double` | RoPE base for CSA/HCA layers and their compressors. | 160000.0 |
| `ropeScalingFactor` | `Double` | YaRN scaling on the compress rope only; the sliding-window `main` rope is left unscaled and attention_factor is forced to 1.0. `0` disables scaling. | 16.0 |
| `ropeScalingOriginalMax` | `Int` | Original maximum position the YaRN correction is computed against. | 65536 |
| `ropeScalingBetaFast` | `Double` | YaRN fast-corrections boundary. | 32.0 |
| `ropeScalingBetaSlow` | `Double` | YaRN slow-corrections boundary. | 1.0 |

### `relativePosition` — `RelativePositionConfig`

Learned relative-attention position encoding, used by architectures that
carry no RoPE at all. `.none` for every family except Inkling-Small.

Inkling projects the residual to `projDim` (`attn.wr_du`, width
`numHeads * dRel`) and reshapes it to a per-head `dRel` relative-state
vector, which mixes a bank of bias-vs-distance profiles
(`attn.rel_logits_proj.proj`, shape `[dRel, extent]`) into one bias per
backward distance. The bias is zero outside `0 ..< extent`.

| Field | Type | Selects | INK |
| --- | --- | --- | --- |
| `dRel` | `Int` | Per-head relative-state width. | 16 |
| `extent` | `Int` | Bias width on full-attention layers. Sliding layers use `slidingWindow` instead, so the two layer kinds ship differently shaped `proj` tensors — `[16, 512]` local, `[16, 1024]` global. | 1024 |
| `projDim` | `Int` | Output width of `attn.wr_du`, i.e. `numHeads * dRel`. | 512 |
| `logScalingFloor` | `Int` | Position floor below which the log-scaling correction is inactive. Applies to full-attention layers only. | 128000 |
| `logScalingAlpha` | `Double` | Strength of that correction. | 0.1 |

## MoE and routing

### Expert inventory

| Field | Type | Selects | G4 | Q36 | Q38 | DSV4 | INK | MPL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `numExperts` | `Int` | Routed experts per MoE layer. | 128 | 256 | 0 | 256 | 256 | 256 |
| `topKExperts` | `Int` | Routed experts selected per token. | 8 | 8 | 0 | 6 | 6 | 8 |
| `numSharedExperts` | `Int` | Shared experts active on every token. | 1 | 1 | 0 | 1 | 2 | 0 |
| `numDenseLayers` | `Int` | Leading layers that use a plain dense FFN instead of the MoE block. | 0 | 0 | 64 | 0 | 2 | 0 |
| `denseIntermediateSize` | `Int` | FFN width of those dense layers; 0 when `numDenseLayers` is 0. | 0 | 0 | 17408 | 0 | 16384 | 0 |

Qwen 3.8 has no routed experts at all: `numExperts` 0 with
`numDenseLayers == numLayers` makes every layer one resident SwiGLU MLP,
and the expert streamer never opens a pool.

### Router behavior

| Field | Type | Selects | G4 | Divergence |
| --- | --- | --- | --- | --- |
| `routerScaled` | `Bool` | Router carries `router.scale` (an input multiplier) and `per_expert_scale` tensors. `false` means a plain quantized linear router with renormalized top-k softmax weights and no auxiliary scale tensors. | `true` | Q36, Q38, DSV4, INK, MPL `false`. |
| `routerScoringFunc` | `String` | Score activation applied to the router logits before top-k selection. | `"softmax"` | DSV4 `"sqrtsoftplus"`. INK `"sigmoid"`. |
| `routedScalingFactor` | `Double` | Multiplier applied to the renormalized top-k routing weights. | 1.0 | DSV4 1.5. INK 8.0 (`route_scale`). |
| `routerNormAfterTopK` | `Bool` | Renormalize the top-k router weights after selection rather than before. | `false` | INK, MPL `true`. |
| `routerGateBias` | `Bool` | Learned additive bias on the router logits, used for selection only. | `false` | INK `true`. |
| `routerGlobalScale` | `Bool` | Per-layer learned scalar multiplying the router weights. | `false` | INK `true`. |
| `numHashRoutedLayers` | `Int` | Leading MoE layers whose expert selection is a frozen token-id lookup (`tid2eid`) instead of a learned argmax. | 0 | DSV4 3. |
| `sharedExpertGated` | `Bool` | Shared-expert output is gated by sigmoid(`shared_expert_gate(x)`). | `false` | Q36 `true`. |
| `sharedExpertSink` | `Bool` | Shared experts occupy their own router logits as sinks, so the gate emits `numExperts + numSharedExperts` scores. | `false` | INK `true`. |

### Expert FFN

| Field | Type | Selects | G4 | Divergence |
| --- | --- | --- | --- | --- |
| `hiddenActivation` | `String` | FFN activation. | `"gelu_pytorch_tanh"` | Q36, Q38, DSV4, INK, MPL `"silu"`. |
| `swigluLimit` | `Double` | Clamp for the expert gate (max) and up (±) pre-activations. 0 disables the clamp. | 0.0 | DSV4 10.0. MPL 7.0. |

## Normalization, residual, and scaling

| Field | Type | Selects | G4 | Divergence |
| --- | --- | --- | --- | --- |
| `ffnSandwichNorms` | `Bool` | Gemma's dual-branch FFN sandwich: pre- and post-feedforward norms plus a per-layer residual scalar. `false` is a plain pre-norm residual block. | `true` | Q36, Q38, DSV4, INK, MPL `false`. |
| `embeddingScaledBySqrtHidden` | `Bool` | Embedding lookup is multiplied by sqrt(`hiddenSize`). | `true` | Q36, Q38, DSV4, INK, MPL `false`. |
| `embedNormEnabled` | `Bool` | RMS norm applied to the token embeddings before the first layer. | `false` | INK `true`. |
| `logitsWidthMultiplier` | `Double` | muP output scaling divided into the logits. 1.0 disables. | 1.0 | INK 16.0. |
| `finalLogitSoftcap` | `Double` | Soft cap applied to the final logits. 0 disables. | 30.0 | Q36, Q38, DSV4, INK, MPL 0.0. |

### `hyperConnections` — `HyperConnectionConfig`

Manifold-Constrained Hyper-Connection (mHC) residual dimensions. `.none`
means a plain single-stream residual, which is every family except DSV4.

The residual is `mult` parallel streams. Each sublayer site owns a learned
mix `fn: [(2 + mult) * mult, mult * hiddenSize]` — plus per-output `base`
biases and 3 scales — producing sigmoid `pre` collapse weights, sigmoid
`post` placement weights in range [0, 2], and a `mult × mult` combine
matrix projected onto the doubly-stochastic manifold.

| Field | Type | Selects | DSV4 |
| --- | --- | --- | --- |
| `mult` | `Int` | Parallel residual streams. | 4 |
| `sinkhornIters` | `Int` | Alternating row/column normalizations that project the combine matrix onto the doubly-stochastic manifold. | 20 |
| `eps` | `Double` | Floor used by those normalizations. | 1.0e-6 |

## Vocabulary and head

| Field | Type | Selects | G4 | Q36 | Q38 | DSV4 | INK | MPL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `vocabSize` | `Int` | Embedding and `lm_head` rows. | 262144 | 248320 | 248320 | 129280 | 201024 | 151936 |
| `unpaddedVocabSize` | `Int` | Real vocabulary size when the embedding matrix is padded for alignment. Logits beyond this are padding and must be dropped before sampling, or the model can emit ids the tokenizer cannot decode. 0 means no padding — `vocabSize` is the real vocabulary. | 0 | 0 | 0 | 0 | 200058 | 0 |
| `tieWordEmbeddings` | `Bool` | `lm_head` reuses the embedding matrix. | `true` | `false` | `false` | `false` | `false` | `false` |

## How to add an axis

An axis is added in four steps, and the suite fails until all four are
done. **One:** add the stored property to `ArchConfig` — or to the
relevant nested config struct — with a default that reproduces today's
Gemma 4 behavior, so existing manifests keep validating unchanged.
**Two:** give it a doc comment that says what it selects, in the same
terms as its neighbors; that comment is the source this document is
written from, so an axis with no doc comment cannot be documented
honestly. **Three:** add its row to the matching section here, with the
value observed in every family that sets it. **Four:** add a conformance
test — a baseline assertion in `ModelTypesTests` pinning the value against
the source checkpoint's `config.json`, and a behavioral test for whichever
kernel or runner branch the axis selects.

An axis also has to survive the round trip through the install format to
be enforceable. Mirror it in the repacker's `ArchInfo`, emit it from
`GTurboJSON`, and check it in `ManifestReader.validateArch`; an axis the
manifest does not carry is a compile-time constant that a mismatched
checkpoint will silently violate.
