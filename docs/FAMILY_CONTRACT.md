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
axis lives in the family's runner and kernels.

**Retired families.** Gemma 4 (with its QAT install), DeepSeek-V4-Flash,
Inkling-Small, Flint, Qwen3.8-Flash-Next (base and Unsloth GGUF) and GLM-5.3-Flash
were retired on 2026-11-18: their
baselines, `ModelFamily` cases and runners are gone. Their axes stay in the
manifest schema, so the tables below keep their columns (G4, DSV4, INK, FLT,
FNX, FNU, G53) as the values those baselines had at `mference-app` 8ff588c.
The G4 column still matters: it is the `ArchConfig.init` default an absent
manifest key validates against.

## Reading the tables

The `ArchConfig.init` defaults are the original install format's values,
Gemma 4's (G4), for every axis that has one, so a manifest that omits a
family-extension key validates against that value. Each table gives the G4
value and then only the families that differ from it. Values are the compiled
baselines, not a supported range: an axis with one observed value is
documented as what has been exercised, not as what is permitted.

| Tag | Family | `arch.family` | Baseline constant |
| --- | --- | --- | --- |
| Q36 | Qwen 3.6 35B-A3B | `qwen36` | `ArchConfig.qwen36_35B_A3B` |
| Q38 | Qwen 3.8 27B | `qwen38` | `ArchConfig.qwen38_27B` |
| MPL | Maple Preview | `maple` | `ArchConfig.maplePreview` |
| MC5 | MiniCPM5-2B | `minicpm5` | `ArchConfig.miniCPM5_2B` |
| IK2 | Inkling 2 Mini (Unsloth UD-Q4_K_XL GGUF) | `inkling2Loop` | `ArchConfig.inkling2Mini` |

Retired (values only, no baseline): G4 Gemma 4 26B-A4B (`gemma4`, also any
manifest without `arch.family`), DSV4 DeepSeek-V4-Flash (`deepseekV4Flash`),
INK Inkling-Small (`inklingSmall`), FLT Flint-40B-A2.5B (`flint`), FNX
Qwen3.8-Flash-Next (`qwen38flashnext`), FNU its Unsloth UD-IQ3_XXS GGUF
(`qwen38flashnextudiq3xxs`), G53 GLM-5.3-Flash (`glm53Flash`). Loading one
fails with `ModelError.retiredFamily` (`ManifestReader.retiredFamilies`).

`ArchConfig.knownArchitectures` maps `arch.family` to the baseline used for
auto-detection at load.

A baseline in that registry means the install validates, **not** that the
runtime can run it. `ManifestReader.familiesWithoutRunner` is the separate,
authoritative capability gate, and `peekFamily` consults it before any of this
machinery is reached.

That table holds one family today: `inkling2Loop` (IK2), refused by the axis
names in `ManifestReader.inkling2RequiredAxes` until `Inkling2ForwardRunner`
lands. IK2 is not the retired INK (`inklingSmall`, still refused): it is a
different checkpoint and layer graph that reuses some of INK's axes.

FNU has no column of its own in the field tables below: every value equals
FNX's except `family` and the two storage-shaped PLE fields,
`pleNgramShardCount` (1: the GGUF stores the n-gram table as one merged
tensor) and `pleNgramVocabSizeBase` (0: the GGUF carries the explicit per-head
vocab sizes instead of the base). MC5 never entered the gate: its baseline and `MiniCPM5ForwardRunner`
arrived together. Keep the two facts
separate — a new
port earns its `ArchConfig` baseline, `ModelFamily` case, tensor accessors and
manifest validation well before its kernels do, and it belongs in the gate for
that whole stretch so every load path refuses it by axis name rather than
guessing.

## Family identity

| Field | Type | Selects | G4 | Divergence |
| --- | --- | --- | --- | --- |
| `family` | `ModelFamily` | The tensor-name contract, the layer graph shape (norm sandwich vs plain pre-norm), and family-specific kernel behavior. Stored in `manifest.json -> arch.family`; an absent value was Gemma 4, the format's original (now retired) architecture, and is refused. The `ArchConfig.init` default is `.qwen36`. | `.gemma4` | One value per family; every family diverges by definition. |

## Dimensions

Sliding-window and full-attention layers are sized independently, so the
model carries both a sliding pair (`numKVHeads`, `headDim`) and a full pair
(`numFullKVHeads`, `fullHeadDim`). A family with no sliding-window layers
mirrors the full values into the sliding slots, where they are never used
to size storage.

| Field | Type | Selects | G4 | Q36 | Q38 | DSV4 | INK | MPL | FNX | MC5 | G53 | IK2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `hiddenSize` | `Int` | Residual width. | 2816 | 2048 | 5120 | 4096 | 4096 | 2048 | 2560 | 2048 | 4096 | 2304 |
| `intermediateSize` | `Int` | Shared-expert FFN width. Mirrored as `ffnIntermediate` in the manifest. | 2112 | 512 | 17408 | 2048 | 2048 | 512 | 640 | 6144 | 2048 | 1152 |
| `moeIntermediateSize` | `Int` | Per-expert FFN width. | 704 | 512 | 0 | 2048 | 2048 | 512 | 640 | 0 | 2048 | 1152 |
| `numLayers` | `Int` | Layer count; also the length of `fullAttentionLayerMask`. | 30 | 40 | 64 | 43 | 42 | 24 | 48 | 42 | 45 | 24 |
| `numHeads` | `Int` | Query heads. | 16 | 16 | 24 | 64 | 32 | 16 | 24 | 16 | 64 | 24 |
| `numKVHeads` | `Int` | KV heads on sliding-window layers. | 8 | 2 | 4 | 1 | 8 | 4 | 2 | 2 | 1 | 8 |
| `numFullKVHeads` | `Int` | KV heads on full-attention layers. | 2 | 2 | 4 | 1 | 8 | 4 | 2 | 2 | 1 | 8 |
| `headDim` | `Int` | Head width on sliding-window layers. | 256 | 256 | 256 | 512 | 128 | 128 | 256 | 128 | 256 | 128 |
| `fullHeadDim` | `Int` | Head width on full-attention layers. | 512 | 256 | 256 | 512 | 128 | 128 | 256 | 128 | 256 | 128 |

DSV4's `numKVHeads` and `numFullKVHeads` are 1 because its attention is
shared-KV MQA: one 512-dim KV head read as both K and V. G53's are 1 for the
same reason on its sparse layers: one `glm53.kvLoraRank`-wide latent is both
K and V, and its `headDim` / `fullHeadDim` are the 256-wide query heads
(`glm53.qkNopeHeadDim`); its KDA layers take their geometry from
`linearAttention`.

`hiddenSize` is the width the *blocks* run at, which is the residual width
only when the family has a single residual stream. FNX's residual is
`flashNext.hcCount` parallel streams, so the value carried between blocks is
`hcCount * hiddenSize` = 10 240 wide; the blocks themselves still run at 2560.
DSV4 and G53 carry `hyperConnections.mult` streams the same way.

## Attention

### Layer kinds and windowing

| Field | Type | Selects | G4 | Divergence |
| --- | --- | --- | --- | --- |
| `fullAttentionLayerMask` | `[UInt8]` | Per-layer attention kind. 0 = sliding-window, 1 = full, 2 = gated-DeltaNet linear, 3 = compressed sparse attention (CSA), 4 = heavily compressed attention (HCA), 7 = Kimi Delta Attention (G53's linear attention), 8 = NoPE latent sparse attention (G53). Values 3 and 4 additionally include the sliding-window branch, because DeepSeek V4 concatenates compressed entries onto the window KV. | `1` on layers 5, 11, …, 29; `0` elsewhere (5 of 30 full). | Q36/Q38: `2` everywhere, `1` on every 4th layer. DSV4: `0` on layers 0–1, then `3` on even and `4` on odd layers. INK: `1` on layers 5, 11, …, 41; `0` elsewhere. MPL: `1` on every 4th layer, `0` elsewhere. FNX: same 3:1 shape as Q36/Q38 over 48 layers (36 linear, 12 full). MC5: `1` on all 42 layers. G53: `7` everywhere, `8` on layers 3, 7, …, 43 (34 KDA, 11 sparse). IK2: `1` on layers 5, 11, 17, 23; `0` (a 1,024-token window) on the other 20. The mask is per stored layer; each of the 48 visits uses its stored layer's kind. |
| `slidingWindow` | `Int` | Sliding-window width. | 1024 | Q36/Q38/FNX/MC5/G53 `0` (no sliding layers). DSV4 `128`. INK `512`. MPL `512`. IK2 `1024`, as G4. |
| `attentionKEqV` | `Bool` | Full-attention K and V share the `k_proj` weight, so one dequant and GEMV produces the raw projection for both. The K and V cache slots stay separate regardless — they diverge at the norms and RoPE. | `true` | Q36, Q38, INK, MPL, FNX, MC5, IK2 `false`. DSV4 and G53 `true` in the strongest sense: K and V are the same cache entry (G53: the `kv_a` latent). |
| `attnOutputGate` | `Bool` | Full-attention `q_proj` emits `2 * numHeads * fullHeadDim` rows as per-head [query ; gate] halves, and the attention output is multiplied by sigmoid(gate) before `o_proj`. | `false` | Q36, Q38, FNX `true` (FNX's `output_gate_type` is `sigmoid`). G53 `false`: its KDA output gate is the low-rank `g_b(g_a(x))` inside `o_norm`, part of the `kimiDeltaAttention` axis, not this one. IK2 `false`: its output gate is `sigmoid(attn_gate(x))` from a separate projection of the layer input, part of the family's tensor contract, not a doubled `q_proj`. |
| `attentionScale` | `Double` | Softmax scale for full attention. | 1.0 | Q36/Q38/FNX 0.0625 (256^-0.5). DSV4 0.044194173824159216 (512^-0.5). MPL and MC5 1/sqrt(128). INK 1/128 — it RMS-normalizes q and k per head, so the scale is 1/d, not 1/sqrt(d). G53 0.0625 (256^-0.5, the `qkNopeHeadDim` query width — the latent dot product is taken after `embed_q` folds q into the 512-wide latent). IK2 0.08838834764831845 (128^-0.5) on both layer kinds; global layers also scale q and the relative states by `1 + logScalingAlpha · log(max(1, (pos + 1) / logScalingFloor))`. |
| `attentionSinks` | `Bool` | Add one learned logit per query head to the attention softmax denominator only. The sink has no value contribution. | `false` | FLT `true`, backed by `self_attn.softmax_offset`. IK2 `true` on every layer kind, with the logit computed per token (`inkling2.attentionSinkProjected`). |
| `noRopeLayerMask` | `[UInt8]` | Per-layer NoPE switch. A nonzero entry disables rotary position encoding on that layer; empty means the family's normal RoPE policy applies everywhere. | `[]` | FLT mirrors `fullAttentionLayerMask`, so all 14 full-attention layers are NoPE while the 35 sliding layers rotate. |

### Per-head query/key norms

| Field | Type | Selects | G4 | Divergence |
| --- | --- | --- | --- | --- |
| `qkNorm` | `Bool` | Full-attention layers carry per-head RMSNorm gains on the query and key projections (`self_attn.q_norm.weight` / `k_norm.weight`), applied before RoPE. `false` is plain-llama attention: the projections go straight to RoPE, and the fused QKV epilogues — which require the gain tensors — are bypassed for the standalone RoPE kernels. | `true` | MC5 `false`. IK2 `false`: it scales each query head by `attn_q_clip` and L2-clips queries and keys (`inkling2.qkClipThreshold`) instead. G53 `false`: its sparse layers norm the low-rank query latent (`q_a_layernorm`) and the KV latent (`kv_a_layernorm`), not the per-head projections, and its KDA layers have no q/k norm (they l2-normalize q and k instead). Every other family `true`; their manifests predate the axis and omit it, which validates as the default. |

### Rotary position

| Field | Type | Selects | G4 | Divergence |
| --- | --- | --- | --- | --- |
| `ropeTheta` | `Double` | RoPE base on sliding-window layers. | 10000.0 | Q36/Q38/FNX 1.0e7. DSV4 10000.0 (the `main` rope; CSA/HCA layers use `compressRopeTheta`). INK 0.0 — no RoPE at all. MPL 10000.0. MC5 5.0e6 (mirrored: it has no sliding layers). G53 0.0 — no RoPE anywhere in the text stack (`qk_rope_head_dim` 0, `mla_use_nope`). IK2 0.0 — the relative bias carries positions. |
| `fullRopeTheta` | `Double` | RoPE base on full-attention layers. | 1000000.0 | Q36/Q38/FNX 1.0e7. DSV4 10000.0. INK 0.0. MPL 0.0 — its global layer is NoPE. MC5 5.0e6. G53 0.0. IK2 0.0. |
| `partialRotaryFactor` | `Double` | Fraction of each head's channels that rotate; the rotary width is `headDim * partialRotaryFactor`. | 0.25 | Q36/Q38/FNX 0.25. DSV4 0.125 (64 of 512). INK 0.0. MPL 0.5. MC5 1.0 — the whole 128-wide head rotates (`rotate_half` over `head_dim`). G53 0.0. IK2 0.0. |
| `ropeNeoxSubdim` | `Bool` | Partial-RoPE convention. `false` (Gemma): pairs (i, `headDim`/2 + i) for i below the rotated-pair count, frequency divisor `headDim`. `true` (Qwen / NeoX sub-dim): rotation confined to the first `rotaryDim` elements, pairing (i, `rotaryDim`/2 + i), frequency divisor `rotaryDim`. | `false` | Q36, Q38, MPL, FNX, MC5 `true`. DSV4 `false`, but with its own interleaved-trailing convention — neither Gemma's proportional nor Qwen's sub-dim layout; the family's kernels implement it. G53 `false` and moot: nothing rotates. IK2 `false`, moot for the same reason. |

### Short convolution

| Field | Type | Selects | G4 | Divergence |
| --- | --- | --- | --- | --- |
| `sconvKernelSize` | `Int` | Depthwise short-convolution width applied to the block inputs and to the K/V streams. 0 disables the short-conv path entirely. | 0 | INK 4. IK2 4: causal convolutions on K and V (dynamic, with per-token taps, on global layers), after attention and after the MoE, each `(conv(x) + x) / sqrt(k + 1)` (`sqrt(2k + 1)` when dynamic). Every layer visit keeps its own four convolution states. |

### `linearAttention` — `LinearAttentionConfig`

Gated-DeltaNet dimensions for layers with mask value 2, and the head
geometry of G53's Kimi Delta Attention layers (mask value 7 — a different
recurrence over the same shapes; see `glm53`). `.none` (all fields zero)
for architectures without linear-attention layers, which is every family
except Qwen 3.6, Qwen 3.8, Qwen3.8-Flash-Next and GLM-5.3-Flash (MiniCPM5 is
all full attention).

| Field | Type | Selects | Q36 | Q38 | FNX | G53 |
| --- | --- | --- | --- | --- | --- | --- |
| `numKHeads` | `Int` | Key heads. | 16 | 16 | 16 | 64 |
| `numVHeads` | `Int` | Value heads. | 32 | 48 | 48 | 64 |
| `keyHeadDim` | `Int` | Key head width. | 128 | 128 | 128 | 128 |
| `valueHeadDim` | `Int` | Value head width. | 128 | 128 | 128 | 128 |
| `convKernelSize` | `Int` | Depthwise conv width on the fused qkv stream. | 4 | 4 | 4 | 4 |

Two derived widths follow from these and are not separate axes: `qkvDim`
(`2 * K-dim + V-dim`, the fused qkv projection rows and the depthwise conv
channel count) and `valueDim` (the z-gate projection rows and `out_proj`
columns).

### `compressedAttention` — `CompressedAttentionConfig`

DeepSeek-V4 compressed-attention dimensions, for layers with mask values 3
and 4 and for the family's shared-KV MQA sliding layers. `.none` for every
family except DSV4 and G53. G53 rides only the low-rank query rank and the
lightning-indexer head shape on this struct (its indexer pools keys, see
`glm53`); it has no output LoRA, no rope and no compress rates, so those
fields are zero.

V4 attention is shared-KV MQA with a low-rank query path, a grouped
low-rank output projection, per-head learnable attention sinks, and
interleaved partial RoPE on the trailing `ropeHeadDim` channels of each
head. CSA layers pool every `csaCompressRate` source tokens into one
compressed KV entry — two overlapping series — and gather the top
`indexTopK` entries per query with a lightning indexer. HCA layers pool
every `hcaCompressRate` tokens non-overlapping and attend densely over the
result. Sliding layers rope at the `ArchConfig` `ropeTheta`; CSA and HCA
layers and their compressors rope at `compressRopeTheta`.

| Field | Type | Selects | DSV4 | G53 |
| --- | --- | --- | --- | --- |
| `qLoraRank` | `Int` | Rank of the low-rank query path. | 1024 | 1536 |
| `oLoraRank` | `Int` | Rank of the output projection. | 1024 | 0 |
| `oGroups` | `Int` | Groups the output projection is split into. | 8 | 0 |
| `ropeHeadDim` | `Int` | Trailing channels per head that carry interleaved partial RoPE. | 64 | 0 |
| `indexNHeads` | `Int` | Lightning-indexer heads. | 64 | 32 |
| `indexHeadDim` | `Int` | Lightning-indexer head width. | 128 | 128 |
| `indexTopK` | `Int` | Compressed entries gathered per query on CSA layers; for G53 the token budget the pooled selection expands to. | 512 | 2048 |
| `csaCompressRate` | `Int` | Source tokens pooled into one compressed KV entry on CSA layers, in two overlapping series. | 4 | 0 |
| `hcaCompressRate` | `Int` | Source tokens pooled non-overlapping on HCA layers. | 128 | 0 |
| `compressRopeTheta` | `Double` | RoPE base for CSA/HCA layers and their compressors. | 160000.0 | 0.0 |
| `ropeScalingFactor` | `Double` | YaRN scaling on the compress rope only; the sliding-window `main` rope is left unscaled and attention_factor is forced to 1.0. `0` disables scaling. | 16.0 | 0.0 |
| `ropeScalingOriginalMax` | `Int` | Original maximum position the YaRN correction is computed against. | 65536 | 0 |
| `ropeScalingBetaFast` | `Double` | YaRN fast-corrections boundary. | 32.0 | 0.0 |
| `ropeScalingBetaSlow` | `Double` | YaRN slow-corrections boundary. | 1.0 | 0.0 |

### `relativePosition` — `RelativePositionConfig`

Learned relative-attention position encoding, used by architectures that
carry no RoPE at all. `.none` for every family except Inkling-Small and
Inkling 2. IK2 clips each per-head relative state to an L2 norm of
`inkling2.relClipThreshold` after its per-head scale, and its `extent` and
`slidingWindow` are equal, so both layer kinds ship the same table shape:
`attn_rel_bias` is GGUF `ne` `[16, 1024]`, row-major `[1024, 16]` (one
16-wide profile row per backward distance).

Inkling projects the residual to `projDim` (`attn.wr_du`, width
`numHeads * dRel`) and reshapes it to a per-head `dRel` relative-state
vector, which mixes a bank of bias-vs-distance profiles
(`attn.rel_logits_proj.proj`, shape `[dRel, extent]`) into one bias per
backward distance. The bias is zero outside `0 ..< extent`.

| Field | Type | Selects | INK | IK2 |
| --- | --- | --- | --- | --- |
| `dRel` | `Int` | Per-head relative-state width. | 16 | 16 |
| `extent` | `Int` | Bias width on full-attention layers. Sliding layers use `slidingWindow` instead, so the two layer kinds ship differently shaped `proj` tensors — `[16, 512]` local, `[16, 1024]` global. | 1024 | 1024 |
| `projDim` | `Int` | Output width of `attn.wr_du`, i.e. `numHeads * dRel`. | 512 | 384 |
| `logScalingFloor` | `Int` | Position floor below which the log-scaling correction is inactive. Applies to full-attention layers only. | 128000 | 8192 |
| `logScalingAlpha` | `Double` | Strength of that correction. | 0.1 | 0.10000000149011612 (f32 0.1, widened) |

## MoE and routing

### Expert inventory

| Field | Type | Selects | G4 | Q36 | Q38 | DSV4 | INK | MPL | FNX | MC5 | G53 | IK2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `numExperts` | `Int` | Routed experts per MoE layer. | 128 | 256 | 0 | 256 | 256 | 256 | 512 | 0 | 288 | 128 |
| `topKExperts` | `Int` | Routed experts selected per token. | 8 | 8 | 0 | 6 | 6 | 8 | 10 | 0 | 8 | 8 |
| `numSharedExperts` | `Int` | Shared experts active on every token. | 1 | 1 | 0 | 1 | 2 | 0 | 1 | 0 | 1 | 0 |
| `numDenseLayers` | `Int` | Leading layers that use a plain dense FFN instead of the MoE block. | 0 | 0 | 64 | 0 | 2 | 0 | 0 | 42 | 3 | 0 |
| `denseIntermediateSize` | `Int` | FFN width of those dense layers; 0 when `numDenseLayers` is 0. | 0 | 0 | 17408 | 0 | 16384 | 0 | 0 | 6144 | 12288 | 0 |

Qwen 3.8 and MiniCPM5 have no routed experts at all: `numExperts` 0 with
`numDenseLayers == numLayers` makes every layer one resident SwiGLU MLP,
and the expert streamer never opens a pool.

### Router behavior

| Field | Type | Selects | G4 | Divergence |
| --- | --- | --- | --- | --- |
| `routerScaled` | `Bool` | Router carries `router.scale` (an input multiplier) and `per_expert_scale` tensors. `false` means a plain quantized linear router with renormalized top-k softmax weights and no auxiliary scale tensors. | `true` | Q36, Q38, DSV4, INK, MPL, FNX, MC5, G53, IK2 `false`. |
| `routerScoringFunc` | `String` | Score activation applied to the router logits before top-k selection. | `"softmax"` | DSV4 `"sqrtsoftplus"`. INK and G53 `"sigmoid"` (G53's router logits are computed in fp32 from a BF16 gate, `moe_router_dtype`). IK2 `"sigmoid"`, logits in fp32 from an F32 router. |
| `routedScalingFactor` | `Double` | Multiplier applied to the renormalized top-k routing weights. | 1.0 | DSV4 1.5. INK 8.0 (`route_scale`). G53 2.5. IK2 8.0 (`expert_weights_scale`). |
| `routerNormAfterTopK` | `Bool` | Renormalize the top-k router weights after selection rather than before. | `false` | INK, MPL, G53, IK2 `true` (G53: `norm_topk_prob`; IK2: the eight chosen unbiased scores sum to 1, then times 8). |
| `routerGateBias` | `Bool` | Learned additive bias on the router logits, used for selection only. | `false` | INK, G53, IK2 `true` (G53: `e_score_correction_bias`, added to the sigmoid scores for selection; the weights are the unbiased scores. IK2 the same, as `blk.N.exp_probs_b.bias`). |
| `routerGlobalScale` | `Bool` | Per-layer learned scalar multiplying the router weights. | `false` | INK `true`. |
| `numHashRoutedLayers` | `Int` | Leading MoE layers whose expert selection is a frozen token-id lookup (`tid2eid`) instead of a learned argmax. | 0 | DSV4 3. |
| `sharedExpertGated` | `Bool` | Shared-expert output is gated by sigmoid(`shared_expert_gate(x)`). | `false` | Q36, FNX `true`. |
| `sharedExpertSink` | `Bool` | Shared experts occupy their own router logits as sinks, so the gate emits `numExperts + numSharedExperts` scores. | `false` | INK `true`. |

### Expert FFN

| Field | Type | Selects | G4 | Divergence |
| --- | --- | --- | --- | --- |
| `hiddenActivation` | `String` | FFN activation. | `"gelu_pytorch_tanh"` | Q36, Q38, DSV4, INK, MPL, FNX, MC5, G53 `"silu"`. IK2 `"situ"`: `c·tanh(g/c) · sigmoid(g) · c'·tanh(u/c')` with `c` and `c'` the `inkling2` expert soft caps (4 and 25). |
| `swigluLimit` | `Double` | Clamp for the expert gate (max) and up (±) pre-activations. 0 disables the clamp. | 0.0 | DSV4 10.0. MPL 7.0. G53 10.0, on the dense layers, the shared expert and the routed experts alike. |

## Normalization, residual, and scaling

| Field | Type | Selects | G4 | Divergence |
| --- | --- | --- | --- | --- |
| `ffnSandwichNorms` | `Bool` | Gemma's dual-branch FFN sandwich: pre- and post-feedforward norms plus a per-layer residual scalar. `false` is a plain pre-norm residual block. | `true` | Q36, Q38, DSV4, INK, MPL, FNX, MC5, G53, IK2 `false`. FNX has no per-sublayer pre-norm at all: its hyper-connection sites carry the norm (see `flashNext`). IK2 is pre-norm but has no single residual: see `inkling2.attnResBlockSize`. |
| `embeddingScaledBySqrtHidden` | `Bool` | Embedding lookup is multiplied by sqrt(`hiddenSize`). | `true` | Q36, Q38, DSV4, INK, MPL, FNX, MC5, G53, IK2 `false`. |
| `embedNormEnabled` | `Bool` | RMS norm applied to the token embeddings before the first layer. | `false` | INK, IK2 `true` (IK2: `token_embd_norm.weight`). |
| `logitsWidthMultiplier` | `Double` | muP output scaling divided into the logits. 1.0 disables. | 1.0 | INK 16.0. |
| `finalLogitSoftcap` | `Double` | Soft cap applied to the final logits. 0 disables. | 30.0 | Q36, Q38, DSV4, INK, MPL, FNX, MC5, G53 0.0. IK2 30.0, as G4. |

### `hyperConnections` — `HyperConnectionConfig`

Manifold-Constrained Hyper-Connection (mHC) residual dimensions. `.none`
means a plain single-stream residual, which is every family except DSV4 and
G53. G53 applies the same formulation with the same site's `pre` (as DSV4
does) and collapses the streams after the last layer by a plain **mean**
rather than a learned `hc_head`; its `fn` arrays are stored BF16 in the pinned
conversion, `base` and `scale` FP32.
Qwen3.8-Flash-Next's residual streams are a different mechanism and live in
`flashNext`, not here: they are mixed through a low-rank factorization with no
combine matrix and no Sinkhorn projection.

The residual is `mult` parallel streams. Each sublayer site owns a learned
mix `fn: [(2 + mult) * mult, mult * hiddenSize]` — plus per-output `base`
biases and 3 scales — producing sigmoid `pre` collapse weights, sigmoid
`post` placement weights in range [0, 2], and a `mult × mult` combine
matrix projected onto the doubly-stochastic manifold.

| Field | Type | Selects | DSV4 | G53 |
| --- | --- | --- | --- | --- |
| `mult` | `Int` | Parallel residual streams. | 4 | 4 |
| `sinkhornIters` | `Int` | Alternating row/column normalizations that project the combine matrix onto the doubly-stochastic manifold. | 20 | 20 |
| `eps` | `Double` | Floor used by those normalizations. | 1.0e-6 | 1.0e-6 |

### `flashNext` — `FlashNextConfig`

Qwen3.8-Flash-Next's three new axes. `.none` (all fields zero, `pleLayerIDs`
empty) for every other family.

**Low-rank hyper-connections.** The residual is `hcCount` parallel copies of
`hiddenSize`, produced by repeating the embedding and staying that wide through
all 48 layers. Each sub-block site (`attn_hyper_connection`,
`mlp_hyper_connection`) owns a gated residual: `input_mix_weight_down`
`[hcLowRank, hcCount * hidden]` and `input_mix_weight_up`
`[hcCount * hidden, hcLowRank]` factorize the mix down to the 2560-wide block
input, `block_inject_weight` `[hcCount, hcCount * hidden]` places the block
output back into all streams, and `hc_norm` is a group RMSNorm with group size
`hiddenSize`. One global `hyper_connection_mixer` collapses the bundle after the
last layer — **this family has no final norm**; the mixer's output goes straight
into `lm_head`. Distinct from `hyperConnections` (DSV4's mHC), which has a
Sinkhorn-projected combine matrix and no low-rank factorization.

Two facts about this family are established against the reference
implementation rather than inferred, and constrain the kernels: every
`Qwen4ExpTextRMSNorm` upcasts internally (`_norm(x.float()) * (1 + w.float())`,
cast back), and the gated `q_proj` packs query and gate **per head** —
`[heads, 2 * headDim]` split on the last dimension — which is already what
`attnOutputGate` means in the shipped Qwen path, so that axis is reusable
unchanged. The one RMSNorm in the stack that is *not* zero-centered is the GDN
gated norm (`linear_attn.norm`), which is ones-initialized; the loader's
`(1 + w)` bake excludes it by name.

**QSA indexer.** Every full-attention layer carries one. It groups the visible
prefix into blocks of `indexerCompressRatio` tokens, scores each block against
the query's `indexerNumHeads` indexer heads, and keeps the top
`indexerBudget / indexerCompressRatio` blocks plus the always-selected
incomplete tail. The result is an attention mask: KV entries are never dropped.
Below a context of about `indexerBudget` the selection is exhaustive and the
layer is byte-identical to dense attention. The indexer does **not** guarantee
that a query's own block is selected: there is no "always keep self" rule, only
the top-k plus the tail.

**PLE n-gram embedding.** At the layers named by `pleLayerIDs` a hashed n-gram
lookup is added to the residual before attention. Its 320-million-row table does
not sit resident: it streams from the row pool published as `manifest.plePool`
(see `PleRowPool`), addressed by hashes computed from the installed
`layer_multipliers` / `ngram_heads_offsets` / `ngram_heads_vocab_sizes` I64
tables — which are loaded, never re-derived. The number of n-gram heads is
derived from the pool's row width rather than stored as an axis
(`hiddenSize / rowDim` = 2560/160 = 16), and is cross-checked against those
tables' length when the pool is opened.

| Field | Type | Selects | FNX |
| --- | --- | --- | --- |
| `hcCount` | `Int` | Parallel residual streams; the residual is `hcCount * hiddenSize` wide. | 4 |
| `hcLowRank` | `Int` | Rank of the hyper-connection mix factorization. | 320 |
| `indexerNumHeads` | `Int` | Indexer query heads scoring the prefix. | 4 |
| `indexerHeadDim` | `Int` | Indexer head width, shared by query and key heads. | 128 |
| `indexerNumKVHeads` | `Int` | Indexer key heads (one pooled key per block). | 1 |
| `indexerBudget` | `Int` | Tokens kept visible per query before the always-selected tail. | 2048 |
| `indexerCompressRatio` | `Int` | Consecutive tokens pooled into one block key. | 4 |
| `pleLayerIDs` | `[Int]` | **One-indexed** layer ids carrying a PLE block: id `n` is `layers[n-1]`. | `[2]` (i.e. `layers[1]`) |
| `pleNgramShardCount` | `Int` | Shards the source n-gram table arrives in, and page-aligned regions in the installed pool. | 128 |
| `pleNgramVocabSizeBase` | `Int` | `ngram_vocab_size_base` verbatim: the PER-HEAD base vocab, not the row count. Row counts come from `manifest.plePool.layers[].rows`; nothing validates one against the other. | 20000000 |
| `pleConvKernelSize` | `Int` | Depthwise causal conv width in the PLE mixer. | 4 |
| `pleEosTokenID` | `Int` | Token id delimiting PLE n-gram segments; a shifted token stream must not read across it. **Not yet published by the installer**, so `validateArch` checks it only when the manifest carries it and otherwise trusts this constant — a mismatched checkpoint would violate it silently until the repacker emits `arch.pleEosTokenID`. | 248044 |

### `glm53` — `Glm53Config`

GLM-5.3-Flash's new axes. `.none` (all fields zero) for every other family.
The family is a hybrid of two attention kinds over the DSV4-style mHC residual
(`hyperConnections`), with its dense-prefix / MoE stack described by the
inventory and router axes above. The three mechanisms are the three names in
`ManifestReader.familiesWithoutRunner` while the runner is being built:

- **`kimiDeltaAttention`** — mask-7 layers run a gated delta-rule recurrence
  over the `linearAttention` head geometry, with a **per-channel** decay
  vector: `g = kdaGateLowerBound * sigmoid(exp(A_log[h]) * (f_b(f_a(x)) +
  dt_bias))` per (head, key channel), so state row `S[h, dk, :]` decays by
  `exp(g[h, dk])`; write strength `beta = sigmoid(b_proj(x))` per head; q and
  k l2-normalized (q also scaled by `keyHeadDim^-0.5`) after a depthwise
  causal conv over `[q ; k ; v]` and SiLU; the output through a
  sigmoid-gated per-head RMSNorm whose gate is the low-rank `g_b(g_a(x))`.
  Qwen's GDN (mask 2) has a per-head scalar decay, a SiLU-or-sigmoid gate on
  a full-width `z` projection and no low-rank gates, so the two are separate
  kernels.
- **`nopeLatentSparseAttention`** — mask-8 layers are MLA in absorbed form
  with no rotary channels: `q = q_b(rmsnorm(q_a(x)))` per head
  (`qkNopeHeadDim`), folded into the latent by the per-head `embed_q`
  `[heads, kvLoraRank, qkNopeHeadDim]`, attending over one shared cache of
  `rmsnorm(kv_a(x))` latents (`kvLoraRank` wide, K = V), unfolded per head by
  `unembed_out` `[heads, vHeadDim, kvLoraRank]` before `o_proj`. The KV cache
  is `kvLoraRank` FP16 per token per sparse layer (~11 KB per token in
  production).
- **`pooledLightningIndexer`** — each sparse layer scores the visible prefix
  with `compressedAttention.indexNHeads` query heads (`wq_b(rmsnorm(q_a(x)))`)
  against pooled keys: `layernorm(wk(x))` keys grouped `indexKPool`
  consecutive tokens at a time and combined by a per-channel softmax over
  `x @ gate^T + ape[j]`; only complete groups are candidates. Scores are
  `relu(q · k) * indexHeadDim^-0.5`, weighted by `weights_proj(x) *
  indexNHeads^-0.5` per head and summed; the top `indexTopK / indexKPool`
  visible groups expand to token indices and the up-to-`indexKPool - 1`
  tokens of the incomplete tail are always selected. While the cache holds
  at most `indexTopK` tokens the selection is exhaustive and the layer is
  dense attention.

Facts read from PipeNetwork's parity-fixed runtime (1e-6 against
`transformers` 5.16) that constrain the kernels: the swiglu clamp applies to
every text FFN (dense, shared and routed); the mHC `base` / `scale` and the
KDA `A_log` / `dt_bias` are fp32 and must stay so (a bf16 `base` puts the
combine matrix off by ~0.5); the low-rank norms use `rmsNormEps` (1e-5) while
the indexer's key LayerNorm uses `indexerKNormEps` (1e-6); router logits are
fp32; the final collapse is the stream mean.

| Field | Type | Selects | G53 |
| --- | --- | --- | --- |
| `kvLoraRank` | `Int` | Width of the shared attention latent that is both K and V on sparse layers. | 512 |
| `qkNopeHeadDim` | `Int` | Query head width on sparse layers (no rotary part). | 256 |
| `vHeadDim` | `Int` | Per-head output width unfolded from the latent. | 256 |
| `indexKPool` | `Int` | Consecutive tokens pooled into one indexer key. | 4 |
| `indexKPoolAlwaysSelectTail` | `Bool` | Whether the incomplete tail group's tokens are always attended. | `true` |
| `indexerKNormEps` | `Double` | LayerNorm epsilon of the indexer key path. | 1.0e-6 |
| `kdaGateLowerBound` | `Double` | Log-space floor of the KDA decay: `lowerBound * sigmoid(...)`, so every channel decays by at least `exp(lowerBound)` per step. | -5.0 |
| `rmsNormEps` | `Double` | RMSNorm epsilon threaded to every norm the runner encodes (the runtime's kernels default to 1e-6). | 1.0e-5 |

### `inkling2` — `Inkling2Config`

Inkling 2's own axes. `.none` (every field zero, `false` or empty) for every
other family. Values are IK2's, read from Unsloth's UD-Q4_K_XL GGUF of
`thinkingmachines/Inkling-2-Mini` @ `b85c3cf` (llama.cpp `inkling2_loop`
keys in parentheses). The `Double` fields are the f32 values the GGUF
carries, widened exactly, because that is all an install from the GGUF can
record; the runner computes in f32, where the two agree.

**Looped core.** Stored layers `0 ..< loopCoreStart` (the prelude) run once,
the core `loopCoreStart ..< loopCoreEnd` runs `loopNumPasses` times with
shared weights, and the rest (the coda) runs once: IK2's 24 stored layers make
48 layer visits per token. Every pass restarts from the prelude's output.
Passes after the first also read a message from the previous pass: per core
layer, four learned queries (`loop_msg_query`, rows query, key, value, ffn)
attend over the previous pass's two core block sums, and four rank-
`loopGateRank` gates mix each message into its input as
`(1 - tanh(forget(z))) · x + tanh(gain(z)) · m`, `z = down([norm(x); norm(m)])`.
Each visit keeps its own KV and convolution state.
`Inkling2Config.layerVisits(numLayers:)` gives the visits in execution order
with their stored layer, pass and cache slot (the reference numbering: pass 0
uses the stored index, later passes `numLayers + (pass - 1) · coreLayers +
(layer - loopCoreStart)`). A pass's messages need the previous pass's whole
output, so passes cannot interleave layer by layer.

**Block attention residuals.** There is no single running residual. The state
is the completed sums of blocks of `attnResBlockSize` layers (the token
embedding first) plus the current block's partial sum; each attention input
(one query on sliding layers, three — q, k, v — on global layers), each MoE
input and the final output read that stack through a learned query
(`attn_res_query`, `ffn_res_query`, `output_res_query`) with a softmax at
scale `hiddenSize^-0.5` over RMS-normalized entries.

**Attention.** Queries are scaled per head (`attn_q_clip`) and keys pass the
short convolution; both are L2-clipped to `qkClipThreshold`. Values pass the
convolution and a weightless RMSNorm. The relative bias is
`relativePosition`; the sink is `attentionSinks` with a per-token logit
(`attentionSinkProjected`); the output is gated by `sigmoid(attn_gate(x))`.

**Experts.** `[gate; up] = W13 · x` (the GGUF's fused `ffn_gate_up_exps`,
gate rows first), activation `hiddenActivation` `"situ"` with the two soft
caps, then `W2` (`ffn_down_exps`). The block format is per layer
(`routedExpertFormats`); an NVFP4 projection's output is also multiplied by
its expert's F32 factor (`ffn_gate_up_exps.scale`, `ffn_down_exps.scale`).

| Field | Type | Selects | IK2 |
| --- | --- | --- | --- |
| `loopCoreStart` | `Int` | First stored layer of the looped core, inclusive (`loop.start`). | 6 |
| `loopCoreEnd` | `Int` | One past the last core layer (`loop.end`). | 18 |
| `loopNumPasses` | `Int` | Passes over the core per token (`loop.count`). 0 means no loop. | 3 |
| `loopGateRank` | `Int` | Rank of each message gate (`loop.gate_rank`): `down` `[rank, 2 · hidden]`, `gain` and `forget` `[hidden, rank]`. | 64 |
| `attnResBlockSize` | `Int` | Layers per attention-residual block (`attn_res.block_size`); the loop bounds are multiples of it. | 6 |
| `qkClipThreshold` | `Double` | L2 clip on each query and key head vector (`attention.qk_clip`). | 22.627416610717773 (f32 sqrt(512)) |
| `relClipThreshold` | `Double` | L2 clip on each per-head relative-state vector (`attention.relative_clip`). | 8.0 |
| `attentionSinkProjected` | `Bool` | The sink logit is `attn_sink_proj(x)`, one per head per token, on every layer kind, rather than a learned constant. | `true` |
| `expertGateSoftcap` | `Double` | Soft cap `c` on the expert gate pre-activation (`activation.situ_beta`). | 4.0 |
| `expertUpSoftcap` | `Double` | Soft cap `c'` on the expert up pre-activation (`activation.situ_linear_beta`). | 25.0 |
| `routedExpertFormats` | `[String]` | GGML block format of each stored layer's routed experts, fused gate-up and down alike. Empty means one format for all layers, set by `manifest.quant.routedExpert`. | `"NVFP4"` on layers 0–19, `"Q8_0"` on 20–23 |

The rest of the file's formats are per tensor, not axes: the runtime
dispatches on each resident tensor's index dtype (`GGMLBlockFormat`: Q8_0 for
attention, embedding and head; F32 for the norms, router, convolutions,
clips, residual queries and loop gates; BF16 for `attn_sink_proj` and the
dynamic-convolution projections in the shipped file).

## Vocabulary and head

| Field | Type | Selects | G4 | Q36 | Q38 | DSV4 | INK | MPL | FNX | MC5 | G53 | IK2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `vocabSize` | `Int` | Embedding and `lm_head` rows. | 262144 | 248320 | 248320 | 129280 | 201024 | 151936 | 248320 | 130560 | 154880 | 201024 |
| `unpaddedVocabSize` | `Int` | Real vocabulary size when the embedding matrix is padded for alignment. Logits beyond this are padding and must be dropped before sampling, or the model can emit ids the tokenizer cannot decode. 0 means no padding — `vocabSize` is the real vocabulary. | 0 | 0 | 0 | 0 | 200058 | 0 | 0 | 0 | 0 | 200058 |
| `tieWordEmbeddings` | `Bool` | `lm_head` reuses the embedding matrix. | `true` | `false` | `false` | `false` | `false` | `false` | `false` | `false` | `false` | `false` |

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
