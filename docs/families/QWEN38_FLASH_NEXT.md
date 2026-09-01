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
- [ ] **W2.1b quality gate OPEN** — greedy rollouts and KLD against the
      mlx-community Qwen 3.6 conversion have not been run (needs a ~70 GB
      download and a model run). Bit parity against the runtime reference
      (W2.1a) *is* enforced, including through the streaming path. Until W2.1b
      runs, weights this path produces are unvalidated at the model level.
- [x] Source index SHA-256 pinned (`99e81524…c590de`, 170,726 bytes)
- [x] Fused-expert axis order, gated `q_proj`, indexer, GDN, HC and MTP shapes
      verified against ranged shard-header reads @ `de4b8e4d`
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
      Still open for top-10: the *expert compute* path. `MoE.maxStreamedExperts`
      is 8, `RoutedBlobs` carries 8 blobs, and `moe_phase2_down_reduce_k{6,8}`
      are the only reduce widths — a top-10 forward needs either a k10 reduce or
      two passes. Router only, so far.
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
- [ ] **Metal kernels for the new axes — 2 of 5 landed.**
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
      - [ ] QSA indexer (raw-key cache, incremental pooled block keys, scoring,
            top-512 selection reproducing the reference's tie-breaking)
      - [ ] Gated full attention over the indexer-selected KV subset
      - [ ] GDN reuse from qwen38, with the gated norm's **sigmoid** activation
            confirmed against what that port bakes
- [ ] `FlashNextForwardRunner`: the production layer loop, expert streaming at
      top-10, and removing `qwen38flashnext` from
      `ManifestReader.familiesWithoutRunner`
- [ ] `bringup-check.sh` green ×3 · community protocol page

## Reproduction of this dossier's facts

```bash
# config and shard index (API routes; the CDN challenges bulk paths)
curl -s https://huggingface.co/api/models/Qwen/Qwen3.8-Flash-Next   # revision sha
# tensor patterns and total_size: model.safetensors.index.json @ de4b8e4d
# sizing model: arithmetic from config.json dims (see table above)
```
