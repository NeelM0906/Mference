# Qwen3.8-Flash-Next on Mference — bring-up dossier

Checkpoint selection, architecture contract, memory budget, and the priced
port plan for running
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(~180B total, ~3.5B active per token, natively multimodal) with SSD-streamed
experts. This document is the Day-0 output of the bring-up kit's timed
rehearsal ([Phase A spec, Workstream 4](../superpowers/specs/2026-08-08-family-bringup-kit-design.md));
the rehearsal clock started 2026-08-31T19:42:22Z. It is the architecture
contract for a proposed `qwen38flashnext` family. Nothing below is
implemented yet unless marked so.

## Why this model, strategically

Released 2026-08-27 — four days before this dossier — and already the
highest-traction open drop of the week (4.5k likes, 159k downloads on the
original repo alone). It is the exact scenario the bring-up kit exists for:
a flagship MoE, days old, that people want on their Macs *now*. It is also
an unusually strong architectural fit:

- **~180B total / ~3.5B active** — a purer expression of the working-set
  thesis than anything currently shipped: the ratio between what exists and
  what runs per token is ~51×.
- **512 experts of 2.76 MiB each (int4)** — finer-grained than any current
  family (Qwen 3.6: 256/layer; DSV4: larger, fewer). Small experts favor the
  slot cache: more of the routing distribution fits in the same slot budget.
- **A 320M-row hashed n-gram embedding table (51.2B params, 102 GB of the
  360 GB checkpoint) read by sparse lookup** — the expert-streaming pattern
  applied to embeddings. Verified geometry: 128 shards of BF16
  [2,500,012 × 160], i.e. 16 n-gram heads × 160 dims. Rows are fetched by
  n-gram hash, a few KiB per token. This tensor *cannot* sit in RAM on any
  consumer Mac and never needs to. It is the third instantiation of the
  fixed-aperture idea (weights → expert slots, KV → pages, and now
  embeddings → row cache).
- **The KV cache is nearly free**: 12 full-attention layers × 2 KV heads ×
  256 dims = 24 KiB/token — 38× lighter than Qwen 3.8-27B's 64 KiB/token.
  262k context costs 6.3 GiB of KV, before any paging.

## Checkpoint selection

No faithful pre-quantized MLX conversion exists (checked 2026-08-31), so
this family is the first that **requires Workstream 2** (original-repo
reading + repacker-side quantization) rather than merely benefiting from it:

| Candidate | Verdict |
|---|---|
| `Qwen/Qwen3.8-Flash-Next` (original, BF16, 131 shards, 360.0 GB) | **Selected** — via W2 streamed read + int4 g64 quantize |
| `sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit` | Rejected: REAP-pruned to 288 experts — a different model |
| `orcarouter/…-Uncensored-{MLX,NVFP4,GGUF}` | Rejected: modified weights, gated repos |
| `Jundot/Qwen3.8-Flash-Next-oQ4e-mtp` | Rejected: nonstandard quant layout, unreadable by the repacker |
| `Qwen/…-FP8`, `unsloth/…-{GGUF,FP8}`, `RadixArk/…-NVFP4`, `VnimanieAI/…-W4A16` | Rejected: formats without kernels or readers |

Pin: revision `de4b8e4d43b917e7706784d8bb445c9af86a3540` (2026-08-27),
`model.safetensors.index.json` SHA-256
`99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de`
(170,726 bytes). Both are recorded in `SupportedModelSource.qwen38FlashNext`.
License: the repo declares `license: other` with a `LICENSE` file (Qwen
license family) — establish redistribution terms before shipping an
installer entry, as was done for Qwen 3.6.

The W2.1 quantizer nucleus already exists in-tree: `Int4AffineEncoder`
(used by `--attach-mtp`, bit-exact with the runtime's reference dequant).
W2's remaining work is the streamed BF16 shard reader, the
quantize-in-flight repack path, configurable sources, and the double
quality gate (bit parity + KLD vs a known-good conversion on Qwen 3.6).

## Architecture contract (from `config.json` @ `de4b8e4d` and the tensor index)

`model_type: qwen4_exp`, text stack `qwen4_exp_text`, 48 layers, hidden
2560, `full_attention_interval: 4` — the same 3 : 1 gated-DeltaNet-to-full-
attention hybrid as Qwen 3.6/3.8 (36 GDN layers, 12 full-attention layers).

### Axes already covered by existing kernels

| Axis | Flash-Next value | Covered by |
|---|---|---|
| Hybrid GDN layer mask | 3:1, 48 layers | `fullAttentionLayerMask` (qwen36/qwen38) |
| Gated DeltaNet dims | dk 128, Hk 16, **Hv 48**, dv 128, conv 4 | qwen38's exact geometry, incl. the fused Hv=48 decode kernel; tensor names identical (`in_proj_qkv/z/a/b`, `A_log`, `dt_bias`, `conv1d`, `norm`, `out_proj`) |
| Full attention | 24 heads, 2 KV heads, head_dim 256, GQA | existing attention path |
| Gated attention output | `output_gate_type: sigmoid` | `attnOutputGate` (Qwen) |
| Partial RoPE | factor 0.25, θ 1e7, NeoX sub-dim | `partialRotaryFactor`, `ropeNeoxSubdim` |
| MoE | 512 experts, top-10, expert FFN 640, SiLU | `numExperts`/`topKExperts`/`moeIntermediateSize` (values new, axes existing) |
| Shared expert, gated | 1 × 640 with `shared_expert_gate` | `sharedExpertGated` (Qwen 3.6) |
| Untied embeddings | `tie_word_embeddings: false` | `tieWordEmbeddings` |
| Vocab | 248,320 (padding TBD from tokenizer) | `vocabSize`/`unpaddedVocabSize` |
| MTP sidecar | 1 hybrid draft layer (`fc_embedding`/`fc_hidden` both [2560, 2560]) | qwen38 MTP machinery + sidecar policy (extended: see gaps) |
| Vision tower | 27-block SigLIP-style, excluded at repack | qwen38 text-only precedent |
| mRoPE sections | `mrope_interleaved`, [11,11,10] | text-only reduction — verify against the qwen38 port's handling |

### New axes (the priced kernel work)

1. **Low-rank hyper-connections** (`hc_count: 4`, `hc_lowrank: 320`).
   Per sub-block `attn_hyper_connection`/`mlp_hyper_connection` with
   `input_mix_weight_down/up` (2560↔320 factorization), `block_inject_weight`,
   `hc_norm`, plus one global `hyper_connection_mixer`. DSV4's
   `HyperConnectionConfig` covers the residual-streams concept; the
   factorized mix and inject/mixer tensors are a new variant. **Small
   kernel work** (a handful of GEMV/mix dispatches per layer).
2. **Attention indexer** (`indexer_*`: 4 heads × 128, 1 KV head,
   budget 2048, compress ratio 4; tensors `self_attn.indexer.{index_qk_proj,
   q_layernorm, k_layernorm}`). DSA-style learned sparse attention: the
   indexer scores history and full attention reads only the top-`budget`
   entries. **Exactness note:** for contexts ≤ the effective budget the
   selection is exhaustive, so dense attention is exact — a v0 port can ship
   dense-exact short-context decode and gate longer contexts behind the
   indexer implementation, the same "exact under a covering budget"
   discipline as the paged-KV MTP gate. **Medium kernel work** (score +
   top-k mask into the existing attention kernel).
3. **Per-layer n-gram embedding (PLE / engram)** at one layer
   (`ple_layer_ids: [2]`; tensors sit at `layers.1.ple.*` — resolve the
   off-by-one against the reference implementation). **VERIFIED against the
   shard headers @ `de4b8e4d`:** the table is **128 shards of BF16
   [2,500,012 × 160] = 320,001,536 rows × 160 dims** — 16 n-gram heads × 160 =
   `ple_embed_dim` 2560, *not* one 20M × 2560 table. `ngram_heads_offsets` and
   `ngram_heads_vocab_sizes` are **I64 [16]**; `layer_multipliers` is
   **I64 [3]**. Plus key/value projections, a width-4 conv
   (`ple_conv_kernel_size`) and three norms. Compute is trivial (hash, gather,
   small GEMVs); the substance is **storage**: 160 is not a multiple of 64, so
   int4 group-64 cannot quantize these rows and the pool is **BF16, ~102.8 GB**
   — the Day-0 ~29 GB int4 estimate is superseded. It is a row-lookup
   pool with a per-row read path and an LFU row cache — structurally the expert
   pool with token-derived rather than router-derived indices.
   **Medium engineering, no novel math.**
4. **Fused expert tensor layout**: `mlp.experts.{gate_up_proj, down_proj}`
   store all 512 experts in single tensors with gate/up fused. **VERIFIED:**
   `gate_up_proj` is BF16 `[512, 1280, 2560]` = `[E, 2I, H]` and `down_proj` is
   `[512, 2560, 640]` = `[E, H, I]`. Repacker mapping work only — split to
   Mference's per-expert page-aligned blobs at install; no runtime change.
   (Which half of the 1280 is gate and which is up still needs the reference
   implementation, but that is a runner question: the layout is settled.)

### Sizing (shapes verified against the shard headers @ `de4b8e4d`)

The "as installed" column is what the W2 repack path actually writes. The PLE
row pool is the correction to the Day-0 estimate: its rows are 160 wide, which
group-64 cannot quantize, so it stays BF16 and the install is ~175 GB rather
than the ~101 GB originally projected.

| Component | Params | BF16 | As installed |
|---|---:|---:|---:|
| Routed experts (48 × 512 × 4.92M) | 120.8 B | 241.7 GB | ~68 GB (INT4 g64) |
| PLE n-gram table (320,001,536 × 160) | 51.2 B | 102.4 GB | **~102.8 GB (BF16 + 0.4% block slack)** |
| MTP sidecar (1 hybrid layer, own experts) | 2.6 B | 5.3 GB | ~1.4 GB (INT4 g64) |
| Resident core (GDN, attention, shared experts, routers, HC, embeddings) | 4.4 B | 8.9 GB | ~2.5 GB (INT4 g64; norms stay BF16) |
| Vision tower (excluded at repack) | 0.4 B | 0.8 GB | — |
| **Total (sum ≈ index's 360.0 GB)** | **179.5 B** | **359 GB** | **~175 GB installed** |

Per-token budget: top-10 × 48 layers = 480 expert reads × 2.76 MiB ≈
1.33 GB/token fully uncached — DSV4-class, but from experts 3× smaller, so
the LFU slot cache and page cache bite harder. PLE adds a few KiB/token of
row reads (one 16 KiB page per row touched, 51 rows per page). KV:
24 KiB/token; GDN state: 113 MB FP32, constant. Estimated footprint class:
**~4–6 GB process memory** for a ~175 GB install — still the strongest
capability-per-byte ratio of any family, though the disk cost is 74% higher
than Day 0 assumed.

## Port plan (kit-priced)

| Step | Work | Kit coverage |
|---|---|---|
| 1. Repack | W2 streamed BF16 read + quantize-in-flight; fused-expert split; PLE shard → row-pool layout; MTP + vision sidecar policy | W2 deliverables (quantizer nucleus exists); layout mapping is data (`docs/families/qwen38flashnext.tensors.json`) |
| 2. ArchConfig | all covered axes are value changes; add low-rank HC variant, indexer config, PLE config as named axes with capability gate | W1 contract + gate |
| 3. Toys + parity | manifest-driven toy (W3.1) for the axis selection; transformers reference parity (W3.2) | kit |
| 4. Runner | GDN reuse; attention + indexer (v0: dense-exact ≤ budget); HC mix; PLE gather; 512-expert top-10 streaming on existing slot machinery | kernels priced above |
| 5. Gate + bench | `./bringup-check.sh qwen38flashnext`, FAMILY_GATE ×3, protocol page | kit |

Honest estimate at Day 0: steps 1–3 are kit-shaped (days), step 4 carries
the three new axes (low-rank HC: small; indexer: medium; PLE: medium).
Against the ≤5-working-day target, this rehearsal is harder than the spec's
LFM2.5 candidate (one new axis); it is also the honest flagship case the
release-calendar strategy has to survive.

## Repack layout as built (Workstream 2)

The installer entry is `--model qwen38flashnext`
(`SupportedModelSource.qwen38FlashNext`, kind `originalRepoQuantize`): it reads
the vendor's BF16 shards through the existing range-streaming installer and
quantizes to INT4 affine group-64 in flight
(`StreamingInt4Quantizer`, planned by `FlashNextPlanner`). Nothing is
materialized: the quantizer rewrites one write tile of bounded scratch at a
time.

**Resident.** A tensor becomes INT4 g64 only when it is a BF16 `.weight` of
rank 2 whose last dimension the group size divides and whose name is not a norm
or a conv kernel. Everything else — norms, `A_log`, `dt_bias`,
`layer_multipliers`, `conv1d`, and the I32 `ngram_heads_*` tables — rides
through at its source dtype, matching what the shipped families' conversions do.

**Routed experts.** `mlp.experts.gate_up_proj` and `mlp.experts.down_proj` split
into the existing per-expert page-aligned blobs, sub-tensors in the same order
Qwen 3.6 and Inkling emit (`gate | up | down` × `weights | scales | biases`).
The fused axis order is **VERIFIED against the shard headers @ `de4b8e4d`**:
`gate_up_proj` is `[512, 1280, 2560]` = `[experts, 2 * moeIntermediate, hidden]`
and `down_proj` is `[512, 2560, 640]` = `[experts, hidden, moeIntermediate]`.
The planner fails loudly on any other shape. Which half of the 1280 rows is gate
and which is up is a runner question, not a layout one, and still needs the
reference implementation.

**PLE n-gram row pool (new pool kind, strictly additive).** The 128 BF16 shards
of `[2,500,012 × 160]` become one page-aligned row-lookup pool per PLE layer at
`ple/layer_{LL}_ngram_rows.bin`, described by an additive `manifest.plePool`
block (`kind: "rowLookupPoolV1"`).

Storage is chosen from the row width, not hard-coded: group-64 quantization
needs `rowDim % 64 == 0`, and **160 does not qualify**, so the pool stores rows
as **BF16** (`plePool.layers[].storage == "bf16"`, `weightBits` 16,
`groupSize` 0, companion sizes 0). The int4 branch remains for a table whose
width the group size divides, and is covered by `plePoolStorageFollowsTheRowWidth`.

```
row record   = BF16:  rowDim * 2                        (160 -> 320 bytes)
               INT4:  [ rowDim/2 packed | scales | biases ]
rowStride    = that record, dense
rowsPerBlock = floor(16384 / rowStride)                 (320 -> 51)
blockStride  = 16384                                    (one page)
row i of shard s -> shards[s].offset
                    + (i / rowsPerBlock) * blockStride
                    + (i % rowsPerBlock) * rowStride
```

Rows are **not** individually page-aligned — at 320 bytes a row that would
inflate the ~102 GB table to ~5 TB. Blocks are, and a record never straddles a
page, so one row costs one page fault and a cached page serves 51 neighbours:
the LFU row cache's natural line. Per-shard regions (published as
`plePool.layers[].shards`) keep row addressing a division rather than a search
and keep the install plan at two byte-range copies per shard — 256 rather than
6.3 million — instead of one per block. Slack is 64 bytes per block (0.4%).

**I64 tables ride through raw.** `layer_multipliers` (I64 [3]) and
`ngram_heads_offsets` / `ngram_heads_vocab_sizes` (I64 [16]) are carried
byte-for-byte at dtype code 4; the install test byte-compares all three against
their source. They are what the runner's hash lookup indexes with, so a silent
coercion would be undetectable at load time.

**Sidecars (W2.4).** `model.visual.*` is always skipped; `mtp.*` is carried by
default and dropped with `--skip-mtp`. Either way the decision is recorded in
`manifest.sidecars`. The draft layer's own routed experts go to an additive pool
at `packed_experts_mtp/` (`manifest.auxiliaryExpertPools`), outside
`packed_experts/layout.json` so the shipped layout validator and the runtime's
routed-expert reader are untouched.

**Manifest axes.** `arch` publishes `hcCount`, `hcLowRank`, `indexer*`, `ple*`
and a `requiredAxes` list. `pleNgramVocabSizeBase` is `ngram_vocab_size_base`
**verbatim** — the per-head base vocab (20,000,000), *not* the row count. True
row counts are derived from the shard headers and published in
`plePool.layers[].rows` and `...shards[].rows`; nothing validates one against
the other. The runtime gate is
`ManifestReader.familiesWithoutRunner`, which is the authority (a manifest does
not get to tell the runtime what it can run): loading this family fails with
`family qwen38flashnext is installed but its runner is not implemented; missing
axes: hyperConnectionsLowRank, attentionIndexer, pleNgramEmbedding`.

## Port status

- [x] Checkpoint selected and pinned (`de4b8e4d`)
- [x] Architecture contract and axis gap list (this document)
- [x] Tensor-name mapping table (`qwen38flashnext.tensors.json`)
- [x] Sizing model closed against the shard index (±0.3%)
- [x] W2 repack path: streamed BF16 read, quantize-in-flight, fused-expert
      split, PLE row pool, sidecar policy, synthetic end-to-end install test
- [x] **Real install completed and verified** (2026-08-31): the full 359 GB
      streamed from the vendor repo and quantized in flight in one attempt,
      2 h 35 m, no CDN throttles; `--verify-install` green over 57 files /
      175,173,302,167 bytes; the dry-run plan's weight accounting
      (175,106,382,264 bytes) matched to 0.04% (the difference is manifest,
      receipt and tokenizer files outside the plan). Layout on disk:
      `model_weights.bin`, `packed_experts/`, `packed_experts_mtp/`, `ple/`.
- [x] Repacker `ArchInfo` entry + runtime capability gate (named refusal) —
      exercised against the real install: every entry point reports
      `family qwen38flashnext is installed but its runner is not implemented;
      missing axes: hyperConnectionsLowRank, attentionIndexer,
      pleNgramEmbedding`.
- [x] **W2.1b weight level PASSED** (2026-09-02,
      [docs/QUANTIZER_QUALITY.md](../QUANTIZER_QUALITY.md)) — this family
      inherits it, because `Int4AffineEncoder.encodeGroup` is the same nucleus.
      Measured against mlx-community's independent conversion of
      `Qwen/Qwen3.6-35B-A3B`: our relative Frobenius error against the BF16
      source is 0.09612 mean vs the control's 0.09648, better on 118 of 124
      sampled tensors. Known deficit: no zero-point snapping, which costs up to
      1.54× on tensors with a large mass of near-exact zeros inside live groups.
- [ ] **W2.1b model level BLOCKED** — greedy rollouts and KLD have still not
      been run, and the blocker is structural rather than a matter of finding
      the time. The control conversion keeps INT8 routers; quantize-in-flight is
      INT4-only by the W2 scope cut; the Qwen 3.6 router GEMV decodes one
      `uint8` per weight. So `qwen36original` installs and verifies but is
      refused at load, and the control experiment cannot execute. Harness
      (`QuantizerQualityMeasurement`) and comparator
      (`Scripts/quantizer-quality-compare.py`) are built and ready. Until it
      runs, weights this path produces are unvalidated at the model level.
- [x] Source index SHA-256 pinned (`99e81524…c590de`, 170,726 bytes)
- [x] Fused-expert axis order, gated `q_proj`, indexer, GDN, HC and MTP shapes
      verified against ranged shard-header reads @ `de4b8e4d` — and the fused
      axis order is now **independently confirmed** on a second vendor
      checkpoint: `Qwen/Qwen3.6-35B-A3B` declares `gate_up_proj` as
      `[256, 1024, 2048]` = `[E, 2·I, H]` and `down_proj` as `[256, 2048, 512]`
      = `[E, H, I]`, and splitting at the halfway row reproduces
      mlx-community's separately converted `switch_mlp.gate_proj`/`up_proj`, so
      gate-first ordering is confirmed too.
- [x] PLE geometry corrected: 128 × [2,500,012 × 160] BF16, pool stays BF16,
      install ~175 GB (not the Day-0 ~101 GB)
- [ ] **Gate/up half order within the fused 1280 rows** still needs the
      reference implementation — a runner question, not a layout one.
- [x] Runtime `ArchConfig` baseline (`qwen38FlashNext_180B_A3_5B`) and
      `ModelFamily.qwen38flashnext`, with the `flashNext` axis group
      (`hcCount`/`hcLowRank`, `indexer*`, `ple*`) documented in
      [the family contract](../FAMILY_CONTRACT.md). Only five exhaustive
      switches existed; the pre-norm accessors this family does not have
      (`inputNorm`, `postAttnNorm`) route to the same named
      `familyRunnerNotImplemented` refusal. The baseline validates
      field-by-field against the real install's manifest, and the capability
      gate is unchanged — a baseline means the install validates, not that the
      runtime can run it.
- [x] Runtime loading: `ManifestReader` parses the new arch axes, the
      `plePool` / `auxiliaryExpertPools` / `sidecars` blocks and the uniform
      INT4 quant slots; resident accessors cover the HC, indexer and PLE
      tensor groups and the three I64 hash tables (typed `[Int64]`, loaded
      never re-derived); `PleRowPool` reads the row pool with an LFU row cache.
- [x] Zero-centered `(1 + w)` norm bake wired as a family-gated **load-time**
      transform, since the W2 install still copies norms verbatim. A future
      install that bakes at repack sets `manifest.zeroCenteredNormsBakedAtInstall`
      and the loader stands down. The GDN gated norm (`linear_attn.norm`) is
      excluded: it is ones-initialized, not zero-centered.
- [ ] **Repacker follow-ups**: bake `+1` into the zero-centered norm set at
      install (and set `zeroCenteredNormsBakedAtInstall`), and publish
      `arch.pleEosTokenID` — until it is published, `validateArch` can only
      check it when present and otherwise trusts the compiled 248044.
- [x] **Router widened to 512 experts / top-10.** `MoE.maxRouterExperts` is 512
      and the router-logits buffer is sized to it; `PrefillRouter.maxExperts`
      and `kPrefillRouterMaxExperts` likewise (the prefill bound only sizes a
      threadgroup staging array, so the shipped families read the same values in
      the same order). Selection is now a template on `K`:
      `router_topk_select_k8{,_par}` and `router_topk_select_k10{,_par}` share
      one body, with `kRouterWideMaxPerLane = 16` giving the k10 form the
      16-candidates-per-lane array 512 experts need across 32 lanes. The shipped
      k6/k8 kernels keep their eight-element lane array and their arithmetic
      unchanged — `RouterTopKParityTests` is their byte gate and passes
      untouched, and `./bringup-check.sh qwen36` returns PASS with a
      byte-identical 16/32/auto ladder.
      Semantics note: the reference's softmax-over-all-experts → top-k of the
      probs → renormalize (`norm_topk_prob`) is algebraically identical to the
      shipped kernels' softmax over the k selected logits, because softmax is
      strictly monotone in the logit. The port relies on that identity rather
      than materializing a 512-wide prob vector.
      New gate: `RouterWideTopK10Tests` — `_par` bit-identical to serial at
      10…512 experts including tie-saturated logits, both k10 kernels exact
      against `FlashNextRouterReference` (the CPU reference forward's own
      router, tied back to it by `FlashNextRouterReferenceTieBackTests`), and
      the whole decode and prefill routers agreeing at the production
      `512 x 2560` INT8 shape.
- [x] **Expert compute widened to top-10.** `RoutedBlobs` now carries
      `kRoutedBlobSlots = 10` pointers (`MoE.routedBlobSlots`) and
      `moe_phase2_down_reduce_k10` is the k8 body with two more simdgroups and
      two more terms in the same residual-first, rank-ordered FP32 accumulation.
      `kMaxStreamedExperts` stays 8: it bounds the DeepSeek-V4 prefill tile's
      `local_slot`, not the argument-buffer array, and every kernel indexes only
      `slot < top_k`, so the widening is layout-only for the shipped families.
      `MoEDeepseekV4` now points *every* unused slot at slot 0 rather than the
      two it knew about.
      A dense **BF16** expert path lands beside it in
      `Sources/Mference/Metal/FlashNext/flashnext_moe.metal` +
      `FlashNextMoE`: the parity install's `moe_intermediate_size` is 32, which
      group-64 cannot quantize at all, so the same runner drives both dtypes —
      the split `FlashNextWeightMatrix` already makes for resident projections.
      Its reduce is general in `top_k` (<= 10) rather than one kernel per width.
      New gate: `FlashNextExpertComputeTop10Tests` — at 512 experts / top-10,
      with the selection coming from the real wide router, INT4 relative error
      `2.8e-4` and BF16 `5.0e-4` against `FlashNextExpertReference` (the CPU
      reference forward's own expert block, tied back to the runner bit-for-bit
      by `FlashNextExpertReferenceTieBackTests`); the BF16 hit/miss split is
      bit-identical to a full pass; and `moe_phase2_down_reduce_k10` with two
      zero-weight ranks is **bit-identical** to `moe_phase2_down_reduce_k8` on
      the same bytes, which is what makes the widening safe for Gemma 4 and
      Qwen 3.6. `./bringup-check.sh qwen36` returns PASS with a byte-identical
      16/32/auto ladder.
- [x] **Toy fixtures + reference parity — CPU float32 reference GREEN**
      (`Tests/Mference/Core/Runtime/FlashNext/`). `FlashNextReferenceRunner` is a
      straight-line float32 CPU forward for the whole stack — hyper-connections,
      PLE (int64 hashing, EOS-segmented shifts, dilated depthwise conv), the QSA
      indexer, GDN, gated full attention and 8-expert top-2 MoE — reading every
      weight through the real `Model` loader, `PleRowPool` and
      `packed_experts/layout.json`. All six parity gates pass for both prompts,
      prefill and cached decode: PLE row ids exact, indexer selected/visible sets
      exact, router top-k exact with weights in tolerance, per-layer tensors in
      tolerance (worst `2.9e-5` max-abs, LONG `layer05.mlp_hc_mixed`, against a
      `1e-4` gate), both 16-token greedy rollouts exact, and cached decode equal
      to the runner's own re-prefill.
- [ ] **Goldens: the fp32 set is not reachable from the checkpoint.** The
      committed `Tests/Mference/Fixtures/qwen4exp/` goldens were captured from
      the float32 weights `Qwen4ExpForCausalLM(cfg)` initialized; the emitted
      checkpoint is a lossy bfloat16 copy. Loading it moves the logits by
      `1.16e-3` (SHORT) / `3.09e-2` (LONG) max-abs, flips router top-k and
      indexer selections, and diverges the LONG greedy rollout at token 7. The
      harness gained `--weight-dtype bf16` and a parallel
      `Tests/Mference/Fixtures/qwen4exp-bf16/` set captured from the weights the
      checkpoint carries; that is what the Swift gates run against. The fp32 set
      is untouched and still reproduces byte-for-byte. **Decide** whether the
      fp32 set stays as the record of the reference's arithmetic or is retired.
- [ ] **The toy checkpoint cannot go through the production installer.** Three
      refusals, pinned by `FlashNextToyRepackBlockerTests`: (1) the harness saves
      the text-only config (`model_type: qwen4_exp_text`, no `text_config`),
      (2) its `layer_types` carry the post-`__post_init__`
      `"qwen_sparse_attention"` rather than the `"full_attention"` the vendor
      config ships — both cosmetic, and repaired by
      `FlashNextParity.productionShapedConfig()`; and (3) **structural**:
      `moe_intermediate_size` is 32, and `FlashNextPlanner.planLayerFile`
      requires the expert intermediate dim to be a multiple of the INT4 group
      size (64) because every routed expert is quantized in flight. Removing (3)
      needs either a wider toy (regenerating every golden) or a no-quantize mode
      in the planner. Until then the parity install is written by
      `FlashNextParity.installToyCheckpoint()` — the same `.gturbo` byte
      contract, real resident index / layout / row pool / manifest, with every
      tensor left at its source BF16 so the goldens are meetable. Measured cost
      of the alternative: an INT4 g64 round-trip of exactly the tensors the
      planner would quantize puts SHORT logits `5.13e-2` max-abs off, with
      1402/1408 elements outside the gate.
- [x] **Metal kernels for the new axes — all 5 landed.**
      - [x] **Group RMSNorm** (`rmsnorm_bf16w_grouped`): `group_size = hidden`,
            one threadgroup per (row, stream), weight indexed by
            (stream, channel) over the whole 10240 bundle rather than shared per
            head. Observed vs the CPU reference at `2560 x 4`: `9.8e-4` max-abs,
            which is the FP16 store floor at these magnitudes.
      - [x] **Low-rank hyper-connections** (`flashnext_hc_*` plus a BF16 mat-vec
            the BF16-passthrough parity install needs): mix `4.5e-4` (BF16
            weights) / `4.4e-4` (INT4 g64), inject `1.2e-4` / `1.4e-4`, inject
            accumulate `2.0e-3`, embedding tile bit-exact. The pre-sigmoid mix
            gate, the low-rank vector before its SiLU and the four injection
            scalars stay FP32 through the chain; rounding a pre-activation to
            FP16 costs more than the buffer saves.
      - [x] **PLE** (`flashnext_ple_stream_gate` / `_apply_gate` / `_conv`, plus
            the shipping `FlashNextPleHash`): output `2.3e-3`, stream `2.1e-3`.
            Chunked and stepped decode are **bit-identical** at both `512 x 4`
            and production `2560 x 4` — the property the nine-row conv state
            exists for. The hash is gated directly against the reference
            runner's captured `ple_ngram_row_ids`, in prefill and through eight
            cached decode steps.
      - [x] **QSA indexer** (`flashnext_indexer.metal` + `FlashNextIndexer`):
            append-only raw-key cache, pooled block keys written once when a
            block's fourth token lands, per-(query, block) scores, and top-k on
            the **CPU** through `FlashNextDescendingTopK` — the exact
            `torch.topk` ordering, libc++ `nth_element` short-circuits included.
            This path is **FP32 end to end**, breaking the runtime's FP16
            convention on purpose: its output is a selection, not a tensor, and
            relu-zero ties at the boundary are common enough that there is
            nothing for a tolerance to absorb. Cost: 6 KiB/token raw keys +
            1.5 KiB/token amortized block keys across the 12 layers, twice the
            design doc's FP16 estimate, against a 24 KiB/token KV.
            Gate (`FlashNextIndexerKernelTests`): selections **EXACT** against
            both `FlashNextIndexerReference` and the reference runner's own
            captures — every prefill position and every stepped decode step, on
            both prompts, including four rows whose top-k boundary is a
            bit-exact tie. No lag-one shortcut: prefill scores each row against
            its own visible block count in one dispatch.
      - [x] **Gated full attention over the selected KV subset**
            (`FlashNextAttention`). Sparsity is applied by **gathering** the
            selected rows into contiguous scratch, never by masking and never by
            dropping KV: the selection is at most `budget + ratio` (2051)
            positions whatever the context, so the shipped dense `attention_full`
            decode kernel runs over the gathered run unchanged, and every
            gathered position is at or before the query so no causal mask is
            needed. Projections, `split_q_gate_fp16`, the fused per-head q/k norm
            + partial NeoX RoPE epilogue and `sigmoid_gate_mul_fp16` are all the
            shipped Qwen 3.8 kernels. Measured worst relative error against
            `FlashNextAttentionReference`: `8.4e-4` (short) / `5.8e-4` (long).
      - [x] **GDN**, with the gated norm's activation **parameterized**. The
            shipped kernel hard-coded silu; `FC_GDN_GATE_SIGMOID`
            (`GDN.OutputGate`) selects sigmoid, and an UNSET constant means silu
            so every shipped pipeline — all built with no constants — compiles to
            exactly the code it did before. `FlashNextGDNGateTests` measures the
            sigmoid form against an FP32 reference at the production Hv=48
            geometry, measures the silu default in the same run, asserts the two
            differ (otherwise the constant never reached the compiler), and A/Bs
            the fused `gdn_delta_gated_decode_qwen38` under sigmoid against the
            unfused pair including the recurrent state.
            `flashnext_gdn.metal` adds a **dimension-generic** fallback, because
            `GDN.init` requires `key_head_dim % 32 == 0` and the parity toy's Dk
            is 8. Real installs never take it; the runner branches on geometry.
- [x] **`FlashNextForwardRunner`** — the production layer loop, both install
      dtypes, expert streaming at top-10 through the real LFU slot cache
      (`pread`, 16 slots — the default rung, and the lowest that can hold a
      top-10 layer's working set), PLE row pool through `PleRowPool`, and the
      global mixer standing in for the absent final norm. Prefill is
      **sequential** (`PrefillRuntimeConfig.off`, `HeadlessSequentialPrefillRunner`),
      which is the first thing a perf pass should take.
- [ ] **The family gate stays DOWN.** `ManifestReader.familiesWithoutRunner`
      still lists `qwen38flashnext`, and `FlashNextCapabilityGateTests` is
      unchanged. The rule is token-exact-or-report, and the toy's long prompt is
      7/8. What was measured (`FlashNextForwardRunnerParityTests`, both prompts,
      prefill plus 8 cached decode steps):
      - **PLE n-gram row ids exact at every position** — a pure 64-bit integer
        hash, no float in the path.
      - **Router and indexer exact until a near-tie flips.** SHORT: exact
        through position 16, then one indexer selection flip at L3 (0 router
        flips in 19x6). LONG: exact through position 10, then a router *rank*
        swap at L5 that does not change the expert set; over 56x6 positions,
        12 router rank flips of which 8 changed the set, and 2 indexer flips.
      - **Greedy rollouts**: SHORT **8/8 token-exact**; LONG 7/8, diverging at
        generated token 7 where the reference's own top-2 margin is `1.31e-4`
        against a measured max-abs logit drift of `6.84e-3` — the argmax was
        never determined at this precision, a 52x margin-to-noise deficit.
      - **Attribution** (`FlashNextForwardRunnerAttributionTests`): fed the
        runner's *own* input and routing, the GPU MoE block reproduces the CPU
        oracle to `7.8e-7` over 354 (position, layer) pairs. The kernels are not
        the source. At layer 0, where the stream reaching it is bit-exact, the
        block inputs carry `7.9e-4` max-abs on a magnitude of `1.33` — 1.2 FP16
        ULP, the storage floor and nothing more.
      - **The amplifier is the toy.** Its block outputs are ~1e-3 against block
        inputs of ~1.1 — three orders of magnitude of attenuation with an O(1)
        Jacobian — so the FP16 input floor emerges as an output perturbation
        comparable to the output itself, lands in a residual stream of magnitude
        ~4e-2 as ~1% per layer, and reaches ~2.4% by the last layer. A trained
        model does not attenuate its blocks by 1000x. Settling this needs the
        real 175 GB install, which the gate currently blocks — an owner
        decision.
- [x] **Real-model first light — the toy hypothesis confirmed.** Measured
      2026-09-01 on the 256 GB M3 Ultra via a measurement-only door
      (`FlashNextRealGenerationMeasurement`, env-gated on
      `MFERENCE_FLASHNEXT_GTURBO`; the production gate is **not** lifted — it
      still refuses the family for CLI/server/app, asserted by
      `productionDoorStillRefusesRealInstall`). The runner is
      `FlashNextForwardRunner` on the real INT4 install, unmodified.
      - *Greedy, "The capital of France is", 24 tokens:* **" Paris. The
        capital of Germany is Berlin. The capital of Italy is Rome. The
        capital of Spain is Madrid. The"** — coherent and correct. 11.6 tok/s
        decode, **2.39 GB peak RSS**.
      - *Chat, storm-surge explanation, 128 tokens:* a fluent, technically
        accurate answer (frictional drag, vegetation roughness, energy
        dissipation, vertical accretion). 10.7 tok/s decode, 5.84 s prefill
        for 62 tokens, RSS 2.39 GB.
      - *Long context, needle-in-haystack:* a passkey planted mid-body in a
        **3,247-token prompt — beyond the 2,048 indexer budget** — is
        retrieved **exactly** (`739215`), the model finishing at end-of-turn.
        Decode holds at **12.3 tok/s** at that context (no long-context decode
        penalty — the sparse indexer's whole purpose), prefill 310 s
        (sequential, unoptimized). This exercises the QSA sparse-attention path
        on real weights past its budget and confirms it attends correctly. (A
        first probe capped generation at 16 tokens, too few for the model's
        `<think>` block to finish; at a 320-token budget it answers cleanly.)
      - The result settles the toy near-tie question: a **trained** model does
        not attenuate its blocks 1000×, so the FP16 floor that flipped one toy
        token does not surface — real output is coherent. **Caveat that
        stands:** greedy token-exactness vs a reference cannot be checked at
        180B scale (no reference rollout exists), so this is a *read* of the
        kernels at scale, not a proof. **W2.1b's model-level half (KLD vs a
        known-good conversion) remains the missing quantitative quality gate**
        — its weight-level half passed on 2026-09-02
        ([docs/QUANTIZER_QUALITY.md](../QUANTIZER_QUALITY.md)), which says each
        tensor is individually faithful but not that 40 layers of accumulated
        error leave the output distribution intact. The production gate stays
        down until the model-level half and a maintainer decision clear it —
        and note that gate refuses this family for **missing axes**, not for
        quantizer quality, so closing W2.1b would not by itself lift it.
      - Footprint headline: **~2.39 GB of process memory for a 180B-parameter
        model** (~75× the resident set), the most extreme expression of the
        working-set thesis in the project.
- [ ] `bringup-check.sh` green ×3 and the community protocol page — downstream
      of the gate lift, which awaits W2.1b and a maintainer decision.

## Reproduction of this dossier's facts

```bash
# config and shard index (API routes; the CDN challenges bulk paths)
curl -s https://huggingface.co/api/models/Qwen/Qwen3.8-Flash-Next   # revision sha
# tensor patterns and total_size: model.safetensors.index.json @ de4b8e4d
# sizing model: arithmetic from config.json dims (see table above)
```
