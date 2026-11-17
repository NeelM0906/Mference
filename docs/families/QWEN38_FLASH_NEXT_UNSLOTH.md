# Qwen3.8-Flash-Next, Unsloth UD-IQ3_XXS GGUF, on Mference

Checkpoint selection, architecture contract, memory budget, fidelity gate and
readings for running the `UD-IQ3_XXS` folder of
[`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)
with SSD-streamed experts. This document is the architecture contract for the
`qwen38flashnextudiq3xxs` family.

This install is a **third-party quantization**. Any task-benchmark figures
published for it are Unsloth's own; this project has not measured task
quality for it. What this page measures is fidelity to the GGUF file and
speed. The architecture is the vendor model's: see
[Qwen3.8-Flash-Next on Mference](QWEN38_FLASH_NEXT.md) for the shared facts
(hybrid GDN / full attention, 512 experts top-10, low-rank hyper-connections,
the attention indexer, the PLE n-gram table).

| | |
|---|---|
| Family identifier | `qwen38flashnextudiq3xxs` (`ModelFamily.qwen38flashnextudiq3xxs`) |
| Source repository | `unsloth/Qwen3.8-Flash-Next-GGUF`, folder `UD-IQ3_XXS` |
| Pinned revision | `3f3ba9cb826b49c665510ce1702fd513a393ce39` |
| Parameters | ~180B total, ~3.5B active (vendor model); the GGUF holds 176.94B (no MTP layer, no vision tower) |
| Install size | 84,705,550,593 bytes |
| Status | **supported** — family gate green 2026-11-17 (#68 merged: `FAMILY_GATE.md` steps 1–4 and 6–8 in the family PR, protocol run 15 of 15 at `54d658c`) |

## Checkpoint selection

Qwen3.8-Flash-Next was published by Qwen; this quantization was published by
Unsloth on 2026-11-04 (commit `3f3ba9cb…`, "Better UD 3.1 quants"), built with
an Unsloth importance matrix. The model card does not state which vendor
revision it was quantized from. Weights are fetched from the pinned revision
on the user's machine, not redistributed by this project.

The pin is four GGUF v3 shards, 83,626,438,426 bytes in all, each checked by
SHA-256 and size before its first byte is read (`SupportedModelSource
.qwen38FlashNextUnslothUDIQ3XXS`, kind `.gguf`). A GGUF repo has no
safetensors index, so `sourceIndexSHA256`
(`5cb4ea60ec38794a952d2310f81f94cd39c699ae1d21bd2dbeaa26ec13e73049`) is the
fingerprint of the canonical shard list, not an index hash. The label is
`unlisted`: it installs by name but is not offered by `mference-ui.sh install`.

| Shard | Bytes | SHA-256 | Contents |
|---|---:|---|---|
| `…-00001-of-00004.gguf` | 10,946,618 | `9d5644b3…f9915` | metadata only (tokenizer, chat template, hyperparameters) |
| `…-00002-of-00004.gguf` | 49,946,027,488 | `aacaff3b…7ddf4` | 1,210 tensors |
| `…-00003-of-00004.gguf` | 4,869,325,888 | `ef8f14e7…07b07` | 13 tensors |
| `…-00004-of-00004.gguf` | 28,800,138,432 | `5de8b5db…9ec36` | the PLE n-gram table |

The runtime enforces the install's quantization in
`ManifestReader.validateQuant`: for this family every `manifest.quant` slot
must be advisory, `scheme: "gguf"`, and name only formats the runtime has
kernels for (`INT8-2L(Q6_K)`, `INT8-2L(Q8_0)`, `Q6_K`, `F32`, `IQ2_S`,
`IQ3_XXS`, `IQ4_NL`, `IQ4_XS`, `Q2_0`). Dispatch is per tensor on the
resident index dtype and per layer on the expert layout, never on the slots.

**License note.** The model card declares `license: other`,
`license_name: qwen-community-1.0`, `license_link: LICENSE`, but the Unsloth
repo has no `LICENSE` file at the pin, so the link is dangling. The text was
read from the base model `Qwen/Qwen3.8-Flash-Next` instead ("Qwen Community
License 1.0", 3,235 bytes, SHA-256 `a0dc4225…6c08`, byte-identical at
`de4b8e4d` and `812d96e1`). It permits use, copying, modification,
redistribution and derivative works, including quantized ones. Conditions:
keep the copyright and permission notice; a product above 100M monthly active
users or US$20M monthly revenue must display the model name; a "Model as a
Service" or "AI Work Assistant" business needs a separate license for
commercial use. The GGUF i-quant tables and block decoders are re-implemented
from ggml (MIT) at llama.cpp `b1b770d9`; see the ggml section of
[`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md) and
[`LICENSE-GGML`](../../LICENSE-GGML).

### Rejected candidates

| Candidate | Routed experts | Attention / embeddings | Disk | Verdict |
|---|---|---|---|---|
| `unsloth/Qwen3.8-Flash-Next-GGUF` `UD-IQ3_XXS` @ `3f3ba9cb` | IQ2_S / IQ3_XXS / IQ4_XS gate-up, IQ4_NL / Q2_0 down, by layer | Q6_K / Q8_0, BF16 indexer, F32 router | 83.6 GB | **Selected** |
| `Qwen/Qwen3.8-Flash-Next` @ `de4b8e4d` (family `qwen38flashnext`) | INT4 affine g64, quantized at install | INT4 g64, INT8 routers | 360.0 GB read, ~175 GB installed | Not a candidate here: the existing vendor family, kept unchanged beside this one |
| Other quantizations in `unsloth/Qwen3.8-Flash-Next-GGUF` | | | | Not evaluated: the maintainer chose `UD-IQ3_XXS`, and no other folder was downloaded |

Verified against revision `3f3ba9cb` by reading the GGUF headers directly
(all 1,224 tensors, published as
[`qwen38flashnext-unsloth-ud-iq3_xxs.tensors.json`](qwen38flashnext-unsloth-ud-iq3_xxs.tensors.json)).
Byte accounting closes exactly: each tensor starts at the previous tensor's
end rounded up to 32 bytes, and each data shard's last tensor ends at the file
size. Nine ggml types occur:

| Type | Tensors | Bytes |
|---|---:|---:|
| IQ4_NL | 46 | 50,033,802,240 (PLE table 28.80 GB, 45 `ffn_down_exps` 21.23 GB) |
| IQ2_S | 52 | 13,972,275,200 |
| IQ3_XXS | 42 | 13,487,308,800 |
| Q6_K | 250 | 3,328,281,600 |
| IQ4_XS | 2 | 891,289,600 |
| Q8_0 | 248 | 841,850,880 |
| Q2_0 | 3 | 707,788,800 |
| F32 | 557 | 313,495,040 |
| BF16 | 24 | 39,321,600 |

Deviations from the vendor checkpoint that matter to the runner: there is no
MTP layer (no `blk.48`, no `nextn` tensor); the converter added `+1` to the
zero-centred norms (`attn_{q,k}_norm`, `hc_*_norm`, `indexer.{q,k}_norm`,
`ple_norm_*`, `output_hc_norm`), stores `ssm_a` as `-exp(A_log)`, reorders the
GDN V heads, splits the indexer's `index_qk_proj` into `q_proj` / `k_proj`,
and concatenates the 128 PLE shards into one table. Every 2-D F32 matrix
(routers, `hc_*_inject`, `ssm_alpha` / `ssm_beta`, conv taps; 77,594,880
values) is BF16-exact; the non-exact F32 values are the `+1` norm vectors and
`ssm_a`.

## Architecture contract

Every `ArchConfig` value equals the vendor baseline
`ArchConfig.qwen38FlashNext_180B_A3_5B` except `family` and two storage-shaped
PLE fields (see the FNU row in [`FAMILY_CONTRACT.md`](../FAMILY_CONTRACT.md)).
The GGUF's `qwen4exp.*` keys were cross-checked against the vendor baseline
field by field. Axes are listed against the Gemma 4 defaults, as in the
template; all of them run on the existing `FlashNextForwardRunner`.

| Axis | Value | Covered by existing kernel? | Gap |
|---|---|---|---|
| `layerTypes` (`fullAttentionLayerMask`) | 3 GDN : 1 full attention, 48 layers | yes | — |
| `linearAttention` | 16 K heads, 48 V heads, dk = dv = 128, conv 4 | partial | GGUF V-head order and stored `ssm_a`: two function constants on `gdn.metal`, unset for every other family |
| `attention` | 24 heads, 2 KV heads, head dim 256, partial RoPE 0.25, θ 1e7, sigmoid output gate | yes | — |
| `numExperts` / `topKExperts` / `moeIntermediateSize` | 512 / 10 / 640 | partial | routed experts run as a new LUT4 slot format, not INT4 g64 (quirk below) |
| `sharedExpertGated` | true, 1 shared expert of 640 | yes | runs on INT8-2L matrices (quirk below) |
| `tieWordEmbeddings` | false | yes | `token_embd` stays native Q6_K, decoded per gathered row |
| `flashNext.hcCount` / `hcLowRank` | 4 / 320 | yes | — |
| `flashNext.indexer*` | 4 heads × 128, 1 KV head, budget 2048, compress 4 | yes | `q_proj` and `k_proj` arrive as two BF16 tensors, run as two GEMVs into one fused row |
| `flashNext.pleLayerIDs` | [2] (GGUF `ple.layers` [1], 0-based) | yes | — |
| `flashNext.pleNgramShardCount` | **1** (vendor: 128) | yes | the GGUF stores one merged table |
| `flashNext.pleNgramVocabSizeBase` | **0** (vendor: 20,000,000) | yes | the GGUF carries explicit per-head vocab sizes and offsets, read from `manifest.plePool.layers[].hash` |

The install's `arch.requiredAxes` carries the shared Flash-Next axis names
(`FlashNextAxes.requiredAxisNames`), as the vendor install's does.
`ManifestReader.qwen38FlashNextGGUFRequiredAxes` lists the three GGUF kernel
families this install needs: `ggufLUT4RoutedExperts`,
`ggufInt8TwoLevelResident` and `ggufIQ4NLPleRows`. It is documentation only,
since no install publishes it. All three are implemented, and
`familiesWithoutRunner` does not list the family.

| Quirk | Where it lives | Why it is not an axis |
|---|---|---|
| GGUF tensor names (`blk.N.*`, `token_embd`, `output`) | `Model.swift:175`, `:187` | naming, resolved in `Model`'s accessors |
| Resident dtype whitelist: BF16, INT8-2L (`0x80 \| Q6_K/Q8_0`), native Q6_K (`0x40 \| 14`, embedding only); anything else throws | `FlashNextWeightMatrix.fromGGUF` (`FlashNextMatVec.swift`), called from the runner's init | storage format, not model shape |
| F32 2-D matrices and BF16-exact vectors narrowed to BF16 at load, re-proving exactness and throwing otherwise | `FlashNextGGUFResident.swift` (`ggufBF16`, `.exact`) | storage format |
| `+1` norms read as F32 by grouped RMSNorm, the per-head q/k norm and the indexer norm (function constants 120, 121, 122) | `RMSNorm(groupedWeightF32:)` (120), `PrefillPerHeadNorm` through `PrefillQKVEpilogue(normWeightsF32:)` (121), `FlashNextIndexer(normWeightsF32:)` (122), each set from the runner's `isGGUF`; `FlashNextResident.swift` accessors | precision of stored weights, not shape |
| Norms already in `1 + w` form (`zeroCenteredNormsBakedAtInstall: true`), so the load-time bake is skipped | manifest; `FlashNextResident.swift:36` (`zeroCenteredNormFamilies` excludes the family) | converter convention |
| Routed experts stored as native GGUF blocks (`ggufNativeBlocksV1`, per-layer stride), expanded on the GPU into LUT4 slots per miss | the runner's `GGUFRouted` setup and `encodeGGUFExpansion`; `PreadExpertStreamer.executeExpertCachePlanRaw` | storage format |
| PLE hash constants from the manifest, never also from resident tensors | `FlashNextGGUFResident.ggufPleHash(layer:)` | storage location |
| Resident expert mode refused by name (`residentExpertMode`) | `Model.ggufExpertModeRefusal` | capability not wired |
| Native MTP refused (no sidecar in the GGUF) | `FlashNextMTPWeights.supportedFamilies` | absent from the checkpoint |

## Memory budget

Params and bytes are summed from the GGUF tensor table; the "as installed"
figures are from the install on disk.

| Group | Params | Bytes in the GGUF |
|---|---:|---:|
| `token_embd` + `output` (Q6_K) | 1.271B | 1.043 GB |
| full attention + indexer (12 layers) | 0.617B | 0.530 GB |
| GDN (36 layers) | 2.087B | 1.755 GB |
| hyper-connections | 0.640B | 0.691 GB |
| PLE key / value projections and conv | 0.033B | 0.035 GB |
| shared experts and their gates | 0.236B | 0.214 GB |
| routers (F32) | 0.063B | 0.252 GB |
| norms | 0.004B | 0.004 GB |
| **Resident total** | **4.948B** | **4.523 GB** |
| routed experts (48 × 512) | 120.796B | 50.292 GB |
| PLE n-gram table (IQ4_NL, 320,001,536 rows × 160) | 51.200B | 28.800 GB |
| **GGUF total** | **176.944B** | **83.615 GB** |

As installed (84,705,550,593 bytes):

| File | Bytes | Notes |
|---|---:|---|
| `model_weights.bin` | 5,378,477,056 | resident core: Q6_K / Q8_0 matrices expanded to INT8-2L at install (+855,408,640 B over the GGUF; `token_embd` stays Q6_K), the rest copied verbatim; the remaining 119,296 B is the tensor index and alignment |
| `packed_experts/layer_NN.bin` (48) | 50,482,642,944 | native GGUF bytes, one page-aligned blob per expert at a per-layer stride |
| `packed_experts/layout.json` | 27,125,556 | |
| `ple/layer_01_ngram_rows.bin` | 28,807,184,384 | IQ4_NL rows, 90 B each, 182 rows per 16 KiB block |
| tokenizer, manifest, receipt | 10,120,653 | tokenizer synthesized from the GGUF metadata |

Per-layer expert strides (stored on disk, and LUT4 slot in memory):

| gate/up + down types | Layers | Stored stride | Slot stride |
|---|---:|---:|---:|
| IQ2_S + IQ4_NL | 23 | 1,982,464 | 2,801,664 |
| IQ3_XXS + IQ4_NL | 21 | 2,179,072 | 2,703,360 |
| IQ2_S + Q2_0 | 3 (8, 10, 13) | 1,523,712 | 2,752,512 |
| IQ4_XS + IQ4_NL | 1 (47) | 2,670,592 | 2,703,360 |

The mean stored expert is 2,054,144 B, 25.8% below the vendor install's INT4
g64 expert (2,768,896 B), so each miss reads less from SSD. The LUT4 slot is
about the same size as an INT4 g64 slot, so slot RAM per expert does not
shrink. The core is mapped read-only and is not part of the process
footprint; the PLE table is read with `pread` into a 20 MB row cache and is
never mapped.

### KV cache

Full attention is 12 layers × 2 KV heads × 256 × (K + V) × FP16 =
24,576 B per token. GDN recurrent state is about 0.11 GB at any context.
Only the 4K figure comes from a run-time itemisation; the others are
estimates.

| Context | KV bytes |
|---|---|
| 4 K | 0.10 GB (plus 0.01 GB indexer K) |
| 32 K | ~0.8 GB (estimate; not measured) |
| 128 K | ~3.2 GB (estimate; not measured) |

### Expert-cache ladder

The slot slab is anonymous memory: slots × the sum of the 48 per-layer slot
strides (132,169,728 B, 126 MiB, per slot). `defaultExpertCacheSlots` gives
16 for this family; the resident rung is refused.

| Slots | Resident expert bytes | Notes |
|---|---:|---|
| 16 | 2.11 GB | floor; the `auto` default for this family |
| 32 | 4.23 GB | this 32 GB host's ceiling: 64 slots tripped the swap guard |
| 96 | 12.69 GB (arithmetic) | not run on the 32 GB host |

## Port status

- [x] **Repack** — `SupportedModelSource.qwen38FlashNextUnslothUDIQ3XXS`
      pinned to `3f3ba9cb826b49c665510ce1702fd513a393ce39`, with per-shard
      SHA-256 and size; `--verify-install` passes on the produced install
      (receipt 2026-11-15T17:58:40Z, each layer checked at its own stored
      stride).
- [x] **Toy parity** — `bringup-check.sh` stage 1 runs the GGUF suites:
      quant tables and goldens, LUT4 expansion and MoE kernels, INT8-2L,
      `GDNGGUFLayoutTests`, `GGUFF32NormKernelTests`,
      `FlashNextGGUFFamilyTests`, `PleRowPoolIQ4NLTests`,
      `PreadExpertStreamerRawTests`, `FlashNextForcedRoutesTests`, and the
      installer's GGUF suites and `gguf*` install tests. There is no whole-model toy for this
      family; its real-install suites are env-gated
      (`MFERENCE_UNSLOTH_GTURBO`). The
      vendor `FlashNext` toy suites were run with a regenerated fixture: one
      near-tie test (`discreteDecisionsAgreeUntilTheFirstNearTieFlip`) fails
      identically on `main` and on the family branch, so it is pre-existing.
- [x] **Runner** — the forward pass meets the re-based U5 gate against
      llama.cpp; see [Fidelity gate (U5)](#fidelity-gate-u5).
- [x] **Ladder** — the `bringup-check.sh` stage 3 commands (greedy, 24
      tokens, 16 / 32 / `auto` slots) give byte-identical output at `b0b1a71`.
- [x] **Gate** — [`FAMILY_GATE.md`](../FAMILY_GATE.md) steps 1–4 and 6–8 are
      recorded in the family PR. Two pre-existing failures are quoted against
      `main` under the maintainer's ruling: `MapleQKNormRoPETests:74`, and the
      Flash-Next near-tie test's long case.
- [x] **Protocol bench** — the community protocol at `54d658c`, 15 of 15
      counted runs `stop=endOfTurn` (one voided run replaced); see
      [Measured results](#measured-results).

## Fidelity gate (U5)

The port must reproduce the GGUF's values. The gate has three layers, all
required.

1. **Weights.** Every decoded tensor class is bit-identical to llama.cpp
   `b1b770d9`'s own `to_float` on real blocks of the pinned GGUF
   (`GGUFQuantGoldensTests.realBlocksReproduceLlamaCpp`, env-gated, local
   fixture), plus committed synthetic goldens covering every grid and sign
   index and the signed-zero cases. **Passes.**
2. **Kernels.** LUT4 expand is bit-exact against its CPU decomposition; the
   LUT4 decode and prefill kernels, the INT8-2L GEMV and the GDN GGUF layout
   are within their `Tolerance` of a CPU reference; the Q6_K embedding gather
   and the IQ4_NL PLE rows are bit-exact. **Passes.**
3. **Model.** Teacher-forced logits on a fixed 8-prompt corpus (prose, code,
   shell plus JSON, a tool-call transcript in the install's chat template, and
   a worked word problem; 1,882 positions), compared against llama.cpp.

**Reference.** llama.cpp `b1b770d90d94cad9e9ea5b993bdbc383a522fb28` with a
probe-only patch to `ggml-cpu.c` ("f32act") that dequantizes weights with
ggml's own `to_float` and takes the dot products against F32 activations. It
ran CPU-only on the raw reference GGUF (all four shard SHA-256s match the pin)
with an F32 KV cache, no flash attention, mmap, 8 threads and
`GGML_CPU_TILED_MM=0`, one prompt per run under a swap guard. Mference ran
the production decode path token by token with 16 slots and default flags
(`FlashNextGGUFTeacherForcedLogitsTests`), fed llama.cpp's token ids;
Mference's own tokenizer reproduced those ids on all 8 prompts. Mference's
head writes FP16 logits; rounding the reference's logits to FP16 and comparing
them with themselves gives pooled KL 2.2e-6, so the FP16 head does not account
for the results below. The corpus, the probe patch and the comparison tool
are outside the repository.

**Original limit.** Pooled over all positions, mean KL(ref ‖ Mference)
≤ 0.005 nats **and** top-1 agreement ≥ 99%, against the f32act reference.
Mference measured KL 0.00665 and top-1 98.51%, so it **missed** this limit.

**Why it was re-based.** Two diagnostics followed the miss:

- *Route forcing* (the U5 plan's diagnostic (a)). Prompt p6 (tool call) carried
  77% of the pooled KL. A per-layer trace found no error above the FP16
  activation floor before the first route-set difference; every drift step was
  a swapped expert at a near-tie of the 10th and 11th router logits. Replaying
  llama.cpp's top-10 sets inside Mference's decode (`MFERENCE_U5_FORCE_ROUTES`,
  a diagnostic, off by default) took p6 from KL 0.03188 to 0.0000476, top-1
  303/303 and max |Δlogit| 0.34: it **removed 99.85% of p6's KL**. Only 2.9%
  of the 14,544 forced layer decisions differed from Mference's own choice.
- *Default llama.cpp* (diagnostic (b)). Unpatched llama.cpp, which quantizes
  activations to Q8_K / Q8_0, measured against the same f32act reference
  scores pooled KL 0.01540 and top-1 97.02%: further from the original limit
  than Mference. No non-F32 activation path met it on this corpus.

With that evidence the maintainer re-based the limit (2026-11-16).

**Re-based gate.** Against the f32act reference:

- each prompt's mean KL is no higher than default llama.cpp's on that prompt
  (met on 8 of 8);
- pooled KL and pooled top-1 are no worse than default llama.cpp's
  (KL 0.00665 vs 0.01540; top-1 98.51% vs 97.02%: met);
- per-prompt top-1 is reported, not gated. On p5, Mference agrees on 229
  positions and default llama.cpp on 230.

Later changes to this family may be no worse than the family PR's figures. An
F32 router-input path was not approved.

**Result: passes the re-based gate.** The figures were measured at commit
`afd78c2`. The PR head `b0b1a71`, rebased on `main`, produces logits
byte-identical to `afd78c2`'s on all 8 prompts, so the figures apply to it
unchanged.

Mference against f32act:

| Prompt | Positions | Mean KL | Max KL | Top-1 | Max \|Δlogit\| |
|---|---:|---:|---:|---:|---:|
| p0 prose | 125 | 0.00236 | 0.029 | 125 (100.0%) | 1.96 |
| p1 code | 209 | 0.00095 | 0.037 | 208 (99.5%) | 5.95 |
| p2 prose, technical | 199 | 0.00293 | 0.055 | 195 (98.0%) | 2.69 |
| p3 prose, narrative | 199 | 0.00443 | 0.028 | 192 (96.5%) | 3.10 |
| p4 Python | 308 | 0.00046 | 0.011 | 308 (100.0%) | 7.75 |
| p5 shell + JSON | 231 | 0.00185 | 0.028 | 229 (99.1%) | 3.11 |
| p6 tool call | 303 | 0.03188 | 2.263 | 292 (96.4%) | 5.25 |
| p7 math | 308 | 0.00104 | 0.023 | 305 (99.0%) | 3.46 |
| **Pooled** | **1,882** | **0.00665** | | **1,854 (98.51%)** | |

Default llama.cpp (`--no-repack`, F32 KV, no flash attention) against f32act:

| Prompt | Positions | Mean KL | Max KL | Top-1 |
|---|---:|---:|---:|---:|
| p0 prose | 125 | 0.00997 | 0.090 | 119 (95.2%) |
| p1 code | 209 | 0.00273 | 0.054 | 206 (98.6%) |
| p2 prose, technical | 199 | 0.01293 | 0.078 | 184 (92.5%) |
| p3 prose, narrative | 199 | 0.01319 | 0.124 | 191 (96.0%) |
| p4 Python | 308 | 0.00262 | 0.060 | 306 (99.4%) |
| p5 shell + JSON | 231 | 0.00696 | 0.080 | 230 (99.6%) |
| p6 tool call | 303 | 0.05971 | 2.479 | 290 (95.7%) |
| p7 math | 308 | 0.00476 | 0.156 | 300 (97.4%) |
| **Pooled** | **1,882** | **0.01540** | | **1,826 (97.02%)** |

Mference's KL is lower than default llama.cpp's on every prompt, by a factor
of 2 to 6, and its top-1 is higher on 7 of 8. Without p6, as a diagnostic
only, Mference is KL 0.00181 and top-1 98.92%; default llama.cpp is 0.00690
and 97.28%.

## Measured results

Protocol: [`COMMUNITY_BENCHMARKS.md`](../COMMUNITY_BENCHMARKS.md).

**Setup.**
- Commit `54d658c`, release build.
- MacBook Pro (MacBookPro18,4), M1 Max (8P + 2E), 32 GB. macOS 27.2.1
  (26B131), Apple Swift 6.4.2, on AC power.
- The three frozen `real-generation-v1` cases, with `--max-new 1024
  --max-context 4096 --temperature 0.2 --top-k 64 --top-p 0.95`, the case
  seeds, `--verify full-sha256`, and the default slot count and prefill chunk.

**Runs.**
- Each case had one discarded warmup, then five measured runs, each in a fresh
  process.
- One medium-review run (run 3) was voided because the host ran on battery
  for 45 s of it. A sixth run on mains power, with its own discarded warmup,
  replaced it.
- All 15 counted runs exited 0 with `stop=endOfTurn`, and swap-outs did not
  grow.
- Output was identical across runs within each case.

| Case | Prompt / generated | Prefill (median) | Decode (median) | Range | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: |
| short-explanation | 62 / 547 | 31.96 s | 4.12 tok/s | 4.10–4.14 | 830 MiB |
| medium-review | 426 / 706 | 50.86 s | 3.83 tok/s | 3.44–4.08 | 858 MiB |
| long-synthesis | 2,940 / 569 | 178.59 s | 3.40 tok/s | 3.33–3.48 | 1,056 MiB |

Peak memory footprint (`/usr/bin/time`) was 0.66–0.68, 1.23–1.24 and 5.35 GB
for the three cases.

**Prefill includes first-touch hashing.** Under `--verify full-sha256`, as the
protocol requires, prefill time includes SHA-256 hashing of each expert-layer
file on first touch. With the default `--verify auto`, which checks against
the install receipt, the indicative readings below show short-explanation
prefill at about 9.4 s and long-synthesis at about 151 s.

**Deviations, each labelled:**
1. The runs used a per-case driver that passes `run-benchmark.sh` `run_case`'s
   arguments, not the script itself.
2. long-synthesis was split across two machine holds, each with its own
   discarded warmup.

**Server and 32K, same build.** These complete the first baseline for this
family at `54d658c` (ledger M32). Ledger entries name the measurement log
described in
[Benchmarks](../BENCHMARKS.md#m1-max-32-gb-measured-2026-11-performance-program).

- **S 3K TTFT:** a fresh `MferenceServer` with `--max-context 36864
  --prompt-cache-mode off` and the default `--verify auto`, the
  long-synthesis prompt (2,940 tokens), 32 generated tokens, one warmup and
  three measured requests, all on AC power. TTFT was 319.27, 320.85 and
  322.43 s (median 320.85 s), decode 2.81-2.93 tok/s, server peak footprint
  1.86 GB, 0 cached tokens on every request. This TTFT is about 1.8× the
  CLI's prefill of the same prompt (178.59 s); the gap was not investigated.
- **L 32K CLI: no valid run.** One attempt (`--max-context 32768 --max-new
  32`, the default `--verify auto`) completed with `stop=maxTokens`, but it
  is void because swap-outs grew by about 177 MB. The maintainer chose to
  record its figures as **estimates, not measurements**: about 29 min of
  prefill (1,736.71 s for 31,480 tokens) and 2.43 tok/s over 32 tokens, at a
  peak footprint of 7.63 GB. No retry was run.
- The server cold-then-warm 32K pair was not run.

**Library mode.** A `MferenceServer --library` from `034757d` (#68 merged),
default roots, on `127.0.0.1`, found both installs on this host (Qwen 3.6
35B-A3B and this one) with no skips, and served one 25-token streamed
request to each with 32 tokens allowed. The Unsloth request paid a model
swap from Qwen 3.6 of 4.7 s; its TTFT including that load was 10.63 s. The
process peak footprint was 7.67 GB, one model's worth, and swap-outs did not
grow (ledger M33; `034757d`; default `--verify auto`). Footprint samples
showed Qwen 3.6's memory leaving about 2-6 s after the swap finished rather
than before the load began.

#78 fixed that: the expert slot slab is now mapped, so a released model's pages
leave at once. In #78's own gated hold (`ws3-slab-mmap-switch`, `8f76f61`) the
footprint went from 6,986 MiB before the swap to 271 MiB when it finished;
the process peak was 7,047 MiB (7.39 GB). On the final `main`, `48461a1`
(ledger M41), the same smoke answered both requests ("The capital of France is
Paris.", 8 tokens each; first token 5.76 s for Qwen 3.6 and 10.62 s for this
install after a 4.8 s swap). The footprint went 7,064 MB → 313-321 MB during
the load → 2,837 MB steady, with a 7.41 GB process peak and swap at 0. The
8-token decode rates (14.6 and 2.65 tok/s) are indicative only.

Decode rate excludes model installation, model loading, and prompt prefill.
Compare rows against [`BENCHMARKS.md`](../BENCHMARKS.md) only when the case,
prompt tokens, generated tokens, settings, and stop reason all match.

### Changes since the family PR

Both changes are class E: every run's output was byte-identical to `main`'s
(short `8e18117d…`, medium `e4ab0458…`, long `138d79ce…`), every run ended at
`stop=endOfTurn`, and swap-outs did not grow. All runs used the protocol's
CLI arguments with `--verify full-sha256`, one discarded warmup per segment
and fresh processes. An A/B/B/A runs control, candidate, candidate, control
segments in one session; an effect counts only when both comparisons agree
in direction and the ranges do not overlap.

**#69, decode MoE join after the router readback** (ledger M34; `c16a9cf`
against `main` `034757d`; A/B/B/A, short-explanation only, 3 measured runs
per segment):

| Segment | Build | Decode tok/s (median) | Prefill s (median) |
| --- | --- | ---: | ---: |
| 1 | `034757d` | 4.143 | 31.34 |
| 2 | `c16a9cf` | 4.481 | 32.30 |
| 3 | `c16a9cf` | 4.388 | 32.03 |
| 4 | `034757d` | 4.055 | 31.97 |

Short decode rose 8.2% in both comparisons with non-overlapping ranges;
pooled medians 4.10 → 4.39 tok/s (+7.1%). Short prefill was +3.1% in one
comparison and +0.2% in the other, so it is not slower in both. Medium and
long were not run: the change touches decode only.

**#70, batched per-token GEMV loops in prefill** (ledger M35 and M36; base
`main` `ad99455`, measured with a build of `c16a9cf`, whose sources are
identical). The first measurement was a single A-then-B pair on `a68b588`,
one run per case:

| Case | Prefill s | Decode tok/s | Peak footprint |
| --- | ---: | ---: | ---: |
| short-explanation | 32.08 → 31.15 (-2.9%) | 4.273 → 4.173 (-2.3%) | 0.68 → 0.62 GB |
| medium-review | 49.21 → 42.21 (-14.2%) | 4.024 → 3.920 (-2.6%) | 1.23 → 0.95 GB |
| long-synthesis | 179.11 → 120.81 (-32.6%) | 3.361 → 3.289 (-2.1%) | 5.35 → 3.39 GB |

The claimed metric, long prefill, cleared the 10% bar, and no metric was more
than 7% worse. Decode was 2.1-2.6% lower on every case, so the final head
`1be8d3d`, which keeps the single-token GEMV for one-token calls, was
measured as a short-explanation A/B/B/A (3 measured runs per segment):

| Segment | Build | Decode tok/s (median) | Prefill s (median) |
| --- | --- | ---: | ---: |
| 1 | `ad99455` | 4.258 | 31.96 |
| 2 | `1be8d3d` | 4.254 | 31.05 |
| 3 | `1be8d3d` | 4.257 | 31.09 |
| 4 | `ad99455` | 4.219 | 31.93 |

Short decode is -0.1% and +0.9% (pooled 4.238 → 4.256 tok/s, ranges
overlap): the decode drop seen on `a68b588` is gone. Short prefill is -2.8%
and -2.6% with non-overlapping ranges (pooled 31.94 → 31.07 s). One
long-synthesis run on `1be8d3d` prefilled in 120.78 s at a 3.38 GB peak
footprint. The merged head differs from `1be8d3d` only in one test file.

These are A/B arms, not a new five-run protocol table; the gate-5 table above
stays the family's protocol baseline at `54d658c`.

The changes below were measured by the workstream leads in their own gated
holds (machine lock, quiet machine, AC power, swap guard). Each names its hold
record and the commit measured. None was re-run on the final `main`,
`48461a1`.

**#71, the server prefill chunk follows host RAM** (class T, gate exact; hold
`ws1-server-chunk`). On a 32 GiB host the server now prefills in 4,096-token
chunks. A fresh server (`--max-context 36864`, prompt cache off, default
`--verify auto`, the 2,940-token long-synthesis prompt, 32 tokens, one warmup
and two measured requests) took 87.32 / 87.71 s to first token with a
4,096-token default (`0b86be7`) and 108.30 / 110.12 s with a 1,024-token
default (`882fe49`). Peak footprint was 4,388 MB and 2,318 MB. The merged head
was not re-measured.

**#77, simdgroup-matrix GEMM for the INT8-2L prefill projections** (class T,
pooled top-1 0.9875; hold `ws1-b-mma`, `bb5c2a9`, whose Sources match the
merged head). With the same server setup and the 4,096-token default chunk,
first token took **72.11 / 72.42 s** (server-reported prefill 72.05 s; warmup
75.38 s), at a peak footprint of 4,449 MB. Against #71's 87.32 / 87.71 s that
is about 17.5% faster, but the two were measured in different sessions, not
interleaved.

**#75, this family in the multi-prefix prompt cache** (class E at chunks 4,096
and 128; hold `ws3-k4d-classE-4k-defaultchunk`, `6ce8deb` against `80106b8`).
The automatic cache budget is capped at 1.5 GiB for both Flash-Next families.
Setup: a server at `--max-context 4096`, greedy, 32 tokens, diagnostics on, one
sample per arm. After an unrelated 426-token turn, the 2,996-token follow-up to
the long-synthesis turn restored 2,971 cached tokens and finished in 14.72 s
(5.12 s prefill); `main`'s live continuation of the same turn took 13.77 s
(4.72 s). The cold 2,940-token first turn took 100.55 s on `main` and 122.56 s
with the change. Later cold repeats (holds `ws3-k4-abba-x1-defaultchunk` and
`ws3-k4-ba-x1-defaultchunk`) put the change 2.9% slower on average; this is
not attributed. Peak footprint was 3.37 GB in both arms.

**#78, the slot slab is mapped** (class E on all six cases of both installs;
holds `ws3-slab-mmap-classE`, `ws3-slab-mmap-abba-long`,
`ws3-slab-mmap-switch`; `8f76f61`, whose Sources match the merged head). Cold
long-synthesis prefill in an A/B/B/A read 93.63 / 91.13 s on `782715d` and
91.88 / 94.54 s with the change, so there was no effect. The model-switch result is
under [Measured results](#measured-results).

### Indicative readings (not protocol)

**Unmerged branch build, not a protocol measurement.** Commit `135799c`
(release CLI), MacBook Pro M1 Max, 32 GB, macOS 27.2.1, Swift 6.4.2. Frozen
`real-generation-v1` prompts and sampling settings (`--max-new 1024
--max-context 4096 --temperature 0.2 --top-k 64 --top-p 0.95`, case seeds),
one warmup and three measured fresh processes per case. Deviations from the
protocol: `--verify auto` instead of `full-sha256`, and medium-review was not
run. All runs exited 0 with `stop=endOfTurn`; swap stayed flat.

| Case | Prompt / generated | Prefill ≈ TTFT (warmup / 1 / 2 / 3) | Decode tok/s (warmup / 1 / 2 / 3) | Peak footprint |
|---|---|---|---|---:|
| short-explanation | 62 / 547 | 9.45 / 9.40 / 9.34 / 9.57 s | 4.191 / 4.165 / 4.089 / 4.111 | 0.65 GB |
| long-synthesis | 2,940 / 569 | 150.89 / 151.16 / 150.24 / 150.90 s | 3.245 / 3.171 / 3.223 / 3.244 | 5.34 GB |

With `--max-new 1` and `MFERENCE_PHASES=1`, prefill took 2.45 s for a 5-token
prompt (1,554 experts, 3.05 GB of expert reads, 7.9 GB process disk read) and
9.74 s for short-explanation's 62 tokens (7,951 experts, 15.6 GB; 16.2 GB).
At 4K context the anonymous worst case is about 5.4 GB beside the 5.4 GB clean
mapped core; about 3.0 GB of the 3K footprint is prefill and Metal scratch not
yet itemised.

### Phases attribution

One `MFERENCE_PHASES=1` decode run is the baseline that later optimization
A/Bs are judged against ([`FAMILY_GATE.md`](../FAMILY_GATE.md) step 6).

Run conditions: commit `b0b1a71` (release CLI), the same host as above,
short-explanation (62 prompt tokens), frozen sampling (seed 20260721),
`--verify auto`, 16 slots. It ended at `stop=endOfTurn` after 547 tokens.
Prefill took 9.39 s, decode 130.93 s (4.178 tok/s), and swap stayed flat.

| Phase | ms/token | Share of decode |
|---|---:|---:|
| Expert I/O (`pread` of slot-cache misses, all exposed) | 118.5 | 49.5% |
| PLE n-gram row fetch | 1.7 | 0.7% |
| Unaccounted: CPU encode and GPU waits | 119.1 | 49.8% |
| **Decode total** | **239.4** | **100%** |

GPU busy time was 41.7 s, or 76.3 ms/token. It falls inside the "GPU waits"
row, since the GPU span is the whole decode window.

Over the whole run, prefill included, the slot cache read 138,748 experts in
26,728 coalesced runs (271.7 GB of expert reads). The process read 118.9 GB
from disk; the remainder came from the page cache.

## Known limits

- **3K-prompt prefill is far from the 11–16 s target** on an M1 Max with
  32 GB. Under the protocol's `--verify full-sha256`, long-synthesis prefill
  was 178.59 s at `54d658c` (ledger M32) and 120.8 s after #70 (ledger M35
  and M36, `a68b588` and `1be8d3d`). The server's time to first token on the
  same prompt was 320.85 s at `54d658c` (ledger M32), 87.3-87.7 s after #71
  and **72.11 / 72.42 s after #77** (hold `ws1-b-mma`, `bb5c2a9`, default
  `--verify auto`). That is still about 4.5 times the target's upper end. It was
  not re-measured on `48461a1`.
  Chunked prefill uses the row-form LUT4 routed kernel and does not yet read
  each expert once per prompt.
- **Prefill uses the row-form LUT4 routed kernel only.** A grouped form is
  separate follow-up work, not part of this family's baseline.
- **Resident expert mode is refused by name** (`residentExpertMode`): the
  family runs only on the slot path, and `auto` never selects resident mapping
  for it.
- **Native MTP is refused**: the GGUF has no MTP layer, and
  `FlashNextMTPWeights.supportedFamilies` excludes the family.
- **The `+1` norms** (`attn_{q,k}_norm`, `hc_*_norm`, `indexer.{q,k}_norm`,
  `ple_norm_*`, `output_hc_norm`) are not BF16-exact. They are kept F32 and
  read by F32-weight variants of grouped RMSNorm, the q/k per-head norm and
  the indexer norm (function constants 120, 121, 122, set only for this
  family). An earlier pass rounded them to BF16; that is superseded.
- **Routing near-ties.** Top-10 routing is discontinuous, and some 10th/11th
  router-logit margins sit below the FP16-activation floor, so a different
  expert can be chosen than llama.cpp chooses. This is the measured source of
  the remaining fidelity gap (see U5) and of the chunked-prefill versus decode
  difference (`FlashNextGGUFRealPrefillTests`: relative 1.01% / 1.22% on the
  vendor protocol prompt, same argmax, 16 greedy tokens exact).
- **Tokenizer and chat template** come from the GGUF metadata, not vendor
  files. There is no normalizer, as in llama.cpp. The chat template is
  Unsloth's modified one. The six FIM / repo tokens are special here, unlike
  Qwen 3.6's tokenizer; this affects only special-token skipping on decode.
- Optional or approximate features are opt-in flags with no default-path
  claims: none. `MFERENCE_U5_FORCE_ROUTES` is a diagnostic, not a control.
- Untested: there is no valid 32K run (the one CLI attempt was void under the
  swap-out rule; its figures are estimates only), and neither the server's
  cold-then-warm 32K pair nor a warm 32K follow-up was run. Every reading on
  this page is from one 32 GB M1 Max.
- **GGUF prefill, row path: a failed command-buffer creation leaves slots
  planned.** In `FlashNextForwardRunner`'s row path the routed-expert fetch
  commits the slot plan before Metal creates the command buffer. If that
  creation fails, the throw leaves those slots planned. Decode is not affected,
  because it creates its command buffer before the fetch. The fix is the same
  exact reorder (create the command buffer first); it is not applied.
- **The batched BF16 prefill kernels share decode's fast-math library.** Their
  bit-identity with decode is checked by #70's CI test on both compilers. That
  is a weaker guarantee than the separate safe-math module the INT8-2L kernels
  use.

### Targets and measured shortfall

The program's targets for this install on a 32 GB M1 Max, against the latest
measurement of each. None was re-run on the final `main`, `48461a1`.

| Target | Latest measured | Source | Status |
| --- | --- | --- | --- |
| Decode, ~3K prompt: 7–10 tok/s | 3.40 tok/s, median of 5 (3.33–3.48), at `54d658c`; single runs of 3.35–3.44 tok/s after #69 and #70 (`1be8d3d`, `67951bd`) | ledger M32, M36, M39 | Short by 2–3 times; not re-measured after #77 or #78 |
| Decode at 32K: 6.5–9.5 tok/s | No valid run. The one attempt was void (swap-outs grew); its 2.43 tok/s is an **ESTIMATE** | ledger M32 | Not measured |
| First token, ~3K prompt: 11–16 s | 72.11 / 72.42 s (server, `bb5c2a9`) | hold `ws1-b-mma` | Short by about 4.5 times |
| First token, 32K cold: 2.2–4.4 min | No valid run. The void attempt's 28.9 min (1,736.71 s) is an **ESTIMATE**, on code before #70 | ledger M32 | Not measured |
| Warm 32K follow-up: 12–20 s | Not measured | — | Not measured |

**Decode read-ahead was tried and dropped.** A prototype (`ef7cac1`: four
reader queues, a budget of four reads) kept output byte-identical, but its
measured block is void under the program's host-noise rule (other processes
used more than 0.6 cores). In those void rows, read-ahead decoded more slowly
than without it on every run: short-explanation 3.94–3.98 tok/s against
4.27–4.40, and long-synthesis 2.95 against 3.15–3.16. It was closed without a
rerun and is not on `main` (hold `ws2-readahead-block`).

## Reproduction

```bash
# Build once.
swift build -c release

# Install from the pinned revision (~83.6 GB read, ~84.7 GB written).
# The label is unlisted, so mference-ui.sh does not offer it.
swift run -c release MferenceRepack --model qwen38flashnextudiq3xxs \
  --output scratch/qwen38flashnextudiq3xxs.gturbo

# Conformance stages (toy suites, install verify, 16 / 32 / auto ladder).
./bringup-check.sh qwen38flashnextudiq3xxs scratch/qwen38flashnextudiq3xxs.gturbo

# Community protocol benchmark (long; run on a quiet machine).
./run-benchmark.sh qwen38flashnextudiq3xxs scratch/qwen38flashnextudiq3xxs.gturbo 3

# Phases attribution baseline.
MFERENCE_PHASES=1 .build/release/MferenceCLI \
  --model scratch/qwen38flashnextudiq3xxs.gturbo \
  --messages-file docs/benchmark-prompts/real-generation-v1/short-explanation.json \
  --max-new 1024 --max-context 4096 --temperature 0.2 --top-k 64 --top-p 0.95 \
  --seed 20260721

# Real-model suites (env-gated; one at a time).
MFERENCE_UNSLOTH_GTURBO=scratch/qwen38flashnextudiq3xxs.gturbo \
  Scripts/test.sh --filter FlashNextGGUFRealPrefillTests
```

Run one model process at a time, and re-read the preconditions in
[`AGENTS.md`](../../AGENTS.md) before any model run.
