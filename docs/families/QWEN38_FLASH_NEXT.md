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
- [x] Repacker `ArchInfo` entry + runtime capability gate (named refusal)
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
- [ ] Runtime `ArchConfig` baseline / `ModelFamily` case (deliberately absent:
      adding one would force a dozen exhaustive switches to grow branches the
      runner cannot honour)
- [ ] Toy fixtures + reference parity
- [ ] Runner: covered axes wired, new axes implemented
- [ ] `bringup-check.sh` green ×3 · community protocol page

## Reproduction of this dossier's facts

```bash
# config and shard index (API routes; the CDN challenges bulk paths)
curl -s https://huggingface.co/api/models/Qwen/Qwen3.8-Flash-Next   # revision sha
# tensor patterns and total_size: model.safetensors.index.json @ de4b8e4d
# sizing model: arithmetic from config.json dims (see table above)
```
