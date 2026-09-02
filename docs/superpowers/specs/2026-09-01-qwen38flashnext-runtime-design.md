# Qwen3.8-Flash-Next runtime — reference semantics and port design

**Date:** 2026-09-01
**Status:** Implementation contract for the `qwen38flashnext` runner. Every
statement below is read from the reference implementation
(`transformers` main, `models/qwen4_exp/modular_qwen4_exp.py`, v5.8.0.dev0 —
the version the pinned checkpoint declares) and its parents
(`qwen3_5`, `qwen3_next`), not inferred from tensor shapes. Where the port
must diverge (Metal, streaming, caches), the divergence is stated and gated.

Companion documents: [QWEN38_FLASH_NEXT.md](../../families/QWEN38_FLASH_NEXT.md)
(family dossier, verified shapes, install layout) and
[qwen38flashnext.tensors.json](../../families/qwen38flashnext.tensors.json).

## Scope (owner-approved)

v1 ships: text-only plain greedy + sampled decode, **full QSA indexer**
(long context is in scope), PLE n-gram embeddings, low-rank
hyper-connections, 512-expert top-10 streaming on the existing slot cache.
Deferred: MTP draft decode (sidecar already installed), vision.
Parity policy: any module that cannot be brought to reference parity at toy
scale stops the port and is reported — nothing ships with unverified math.

## The residual stream is 4×2560, not 2560

`hidden_states = embed(input_ids).repeat(1, 1, hc_count)` — after embedding
(no sqrt scaling), the stream is `hc_count(4) × hidden(2560) = 10240` wide
and stays that wide through all 48 layers. Blocks (GDN / attention / MoE)
run at 2560; hyper-connections mix the 4 streams down to a block input and
inject the block output back into all 4. **There is no final norm**: after
the last layer, the global `hyper_connection_mixer` (a GatedResidual with
`use_combine=False`) produces the 2560-wide `last_hidden_state`, and
`lm_head` applies directly to it.

## RMSNorm convention: zero-centered (1 + weight)

All Qwen4-Exp RMSNorms initialize weight at 0 and apply `(1 + weight)` —
the same convention the qwen38 MTP attach already handles by baking `w + 1`
into stored weights at repack. **The port bakes `(1 + w)` into every norm
weight at install** so the runtime's standard RMSNorm applies unchanged.
(Requires a repacker follow-up: the current W2 install copies norms
verbatim; add the `+1` bake for the same norm set the reference zero-centers
— all `Qwen4ExpTextRMSNorm` instances: attention q/k norms, indexer q/k
layernorms, GDN gated norm's RMS part per qwen3_5 convention (check against
the qwen38 port's existing handling), hc_norms, PLE norms.)
Grouped norms: `hc_norm`, `norm_key`, `norm_query`, `norm_conv` are
**group RMSNorms with group_size = hidden (2560)** — each of the 4 streams
normalized independently over its 2560 channels, one shared 10240-wide
weight vector.

## Hyper-connections (`Qwen4ExpTextGatedResidual`)

Per sub-block (attn and mlp each have one), input `hyper ∈ R^10240`:

```
h_n   = group_rmsnorm_2560(hyper) * (1 + hc_norm_w)          # 10240
m     = silu( W_down · h_n / hc_count )                      # 320   (W_down: [320, 10240])
g     = sigmoid( W_up · m )                                  # 10240 (W_up: [10240, 320])
mixed = mean over the 4 streams of (g ⊙ h_n) viewed [4, 2560]   # 2560 → block input
inj   = 2 · sigmoid( W_inject · h_n / hc_count )             # 4     (W_inject: [4, 10240])
after block: hyper ← hyper + flatten( block_out[None,2560] ⊙ inj[4,None] )
```

The global mixer is identical minus the inject path (returns `mixed` only).
Note both `/hc_count` scalings and that the mix/inject read the **normed**
stream while the residual adds to the **raw** stream.

## Decoder layer order

```
if PLE layer:  hyper += PLE(hyper, input_ids)                # pre-attention
mixed, inj = attn_hyper_connection(hyper)
block_out  = GDN(mixed)            # linear_attention layers (L%4 != 3)
           | Attention(mixed)      # qwen_sparse_attention layers (L%4 == 3)
hyper     += inject(block_out, inj)
mixed, inj = mlp_hyper_connection(hyper)
block_out  = SparseMoE(mixed)
hyper     += inject(block_out, inj)
```

## GDN and full attention: qwen3_5 classes, i.e. the existing qwen38 port

`Qwen4ExpTextGatedDeltaNet` **is** `Qwen3_5GatedDeltaNet` (dims dk128/Hk16/
Hv48/dv128/conv4 — the fused Hv=48 kernel applies) with the gated norm's
activation from `output_gate_type` = **sigmoid** (qwen38 uses the same
family; verify which activation the existing port bakes — this model gates
with sigmoid, not silu). `Qwen4ExpTextAttention` **is** `Qwen3_5Attention`
plus per-head q/k RMSNorm ([256], zero-centered) plus the indexer mask:
gated q_proj (2×24×256 rows; gate half order per the existing qwen38
implementation), GQA 2 KV heads × 256, partial RoPE rotary_dim = 64
(0.25 × 256), θ 1e7, NeoX sub-dim pairing, softmax scale 1/√256.
mRoPE: text-only positions collapse the sections; reuse the qwen38 port's
handling. KV cache: standard full KV for these 12 layers (24 KiB/token) —
selection sparsity is applied through the mask, never by dropping KV.

## MoE (`Qwen3NextSparseMoeBlock`)

- Router: `logits = W_router · x` (BF16 in, logits), `probs = softmax(logits)
  in float32`, top-10 **of the probs**, renormalize the 10 to sum 1
  (`norm_topk_prob = true`), weights applied to each expert's output.
- Experts: `gate, up = chunk2(W_gate_up[e] · x)` — **gate is rows [0, 640),
  up is rows [640, 1280)** of the fused tensor; `out = W_down[e] ·
  (silu(gate) ⊙ up)`, summed over the 10 experts with weights.
- Shared expert: standard SwiGLU MLP (640) gated by
  `sigmoid(shared_expert_gate · x)` (a [1, 2560] row), added to the routed
  sum. Identical to the qwen36 path.
- Mference mapping: existing routed-expert streaming; ArchConfig axes
  `numExperts 512 / topKExperts 10 / routerNormAfterTopK` semantics — match
  against qwen36's exact router path (softmax → topk → renorm is the qwen36
  behavior; verify flag-for-flag rather than assuming).

## QSA indexer (`Qwen4ExpTextQSAIndexer`) — the long-context axis

Per full-attention layer. Projections: `index_qk_proj: [640, 2560]` =
4 query heads × 128 + 1 key × 128.

Per position: `q = rope(rmsnorm(q_heads))` (RoPE on the first 64 dims at
the query's position); the **raw key** (un-normed, un-roped, 128) is
appended to an indexer key cache (grows 128/token/layer ≈ 3 KiB/token
across the 12 layers).

Selection for a query at position t (over the visible prefix 0..t):
1. Group the prefix into consecutive **blocks of compress_ratio = 4**;
   only complete blocks participate; the tail (1–3 tokens, and the query's
   own token) is **always selected**.
2. Per complete block: `k_blk = rope_at(block_start,
   rmsnorm_k(mean_fp32(raw keys of the 4 tokens)))` — pooling in float32,
   norm after pooling, RoPE at the **block's first position**. These are
   immutable once the block completes → cache the finished `k_blk` rows
   (one 128-vector per 4 tokens per layer) instead of recomputing.
3. Score per block: `relu(q_h · k_blk)` summed over the 4 query heads,
   `/ √128`, in float32.
4. Take the top-`indexer_budget / 4 = 512` blocks; their 4 tokens each plus
   the tail are the visible set; everything else is masked out of that
   layer's attention for this query.

Exactness properties the tests must pin:
- When complete blocks ≤ 512 (context ≲ 2048 + tail), every block is
  selected → **byte-identical to dense attention**; gate this with a dense
  A/B at short context.
- Beyond the budget, the sparse mask IS the reference semantics — parity is
  against the reference implementation at toy scale (long toy contexts with
  a small budget), not against dense.
- Prefill: the reference computes selection per query position over full
  position embeddings; a chunked prefill must reproduce per-position
  selection exactly (lag-free, unlike the paged-KV Quest lag-one policy —
  do not reuse that policy here).

## PLE (`Qwen4ExpTextPLELayer` + `Qwen4ExpTextNGramEmbedding`)

Applies at **one-indexed** layer id 2 → `layers[1]`, a GDN layer. Input is
the raw token ids (`ple_input_ids = input_ids`), the 10240 hyper stream,
and two small cache states; output adds to the hyper stream pre-attention.

N-gram hashing (all in 64-bit integer arithmetic):
- Per PLE layer, 3 multipliers derived by splitmix64:
  `base = seed(1234) + 10007 · ple_layer_index`;
  `m_i = 2 · (splitmix64((base + Γ·(i+1)) mod 2^64) mod half_bound) + 1`
  where `Γ = 0x9E3779B97F4A7C15`, `half_bound = max(1, (2^63−1) /
  unigram_vocab / 2)`. **The installed `layer_multipliers` (I64 [3]) carry
  exactly these values — load them from the checkpoint, never re-derive.**
- Token history: previous 2 ids cached (`context_len = ngram_size − 1`);
  shifted token streams `t_0` (current), `t_1`, `t_2` where shifts do not
  cross an EOS boundary (positions whose segment starts after the shift
  read as `eos_token_id`; segments are delimited by EOS **inclusive** —
  see `_shift_right_ignore_eos`).
- For n-gram order n ∈ {2, 3}: `mix_n = XOR over i<n of (t_i · m_i)` (int64
  wrap semantics; values stay < 2^63 by construction), then for each of the
  8 heads of that order: `id = mix_n mod head_vocab[h] + head_offset[h]`.
  `head_vocab` / `head_offset` are the installed I64 [16] buffers — load,
  never re-derive (they are consecutive primes ≥ 20,000,000 whose layout
  produced the 320,001,536-row padded table).
- Lookup 16 rows × 160, concatenate → `e ∈ R^2560`. **This is the streamed
  row pool**: 16 row reads/token from `ple/layer_01_ngram_rows.bin` through
  an LFU row cache (page-aligned 51-row blocks; BF16 rows).

PLE mixing (per token):
```
k = group_rmsnorm(W_key · e)   viewed [4, 2560]     (W_key: [10240, 2560])
v = W_value · e                       # 2560        (W_value: [2560, 2560])
qn = group_rmsnorm(hyper)      viewed [4, 2560]
gate_s = (k_s · qn_s) / √2560         # per stream s, float
gate_s = sign(gate_s) · sqrt(max(|gate_s|, 1e-6)) → sigmoid
gv = flatten(gate_s ⊙ v)              # 10240
out = gv + silu(depthwise_conv1d(group_rmsnorm(gv)))   # kernel 4, dilation 3,
                                                        # causal, per-channel
hyper += out
```
Conv cache: `(kernel−1)·dilation = 9` positions of the normed gated value
per channel (10240 × 9 BF16 ≈ 180 KiB). The conv weight `[10240, 1, 4]`
is depthwise with dilation `ngram_size = 3`.

## Caches (decode state per sequence)

| State | Size | Notes |
|---|---|---|
| GDN recurrent + conv tails (36 layers) | ~113 MB FP32 + small | existing |
| Full KV (12 layers) | 24 KiB/token | existing paths |
| Indexer raw keys (12 layers) | 3 KiB/token | new, append-only |
| Indexer pooled block keys (12 layers) | 0.75 KiB/token amortized | new, derived, immutable per block |
| PLE: last 2 token ids + conv state | ~180 KiB constant | new |
| PLE LFU row cache | configurable pool | new, the third aperture |

## Port plan and gates

1. **Golden harness first** (W3.2): python venv with torch (CPU) +
   transformers@main; a toy `qwen4_exp` checkpoint (2–8 layers incl. one
   PLE layer and ≥2 attention layers, tiny dims that satisfy the config
   validators, small indexer budget so toy contexts exceed it); emit golden
   per-module tensors (HC mix/inject, PLE embedding rows and layer output,
   indexer selected sets and masked attention outputs, router
   probs/indices, per-layer hidden states) and greedy rollouts, both short
   (dense-equivalent) and long (sparse-active). Goldens are committed;
   the venv is not.
2. **Swift modules against goldens**, in order: group RMSNorm → HC →
   router/experts wiring at 512/top-10 → PLE (hash exactness in integers
   first, then the mix) → indexer (selection sets must match exactly as
   integer sets, then masked attention) → full toy forward parity → greedy
   rollout parity (exact where BF16 order allows; tolerance policy per the
   repo's two-tier discipline, stated per gate).
3. **Real model**: repacker norm-bake follow-up (+1), first light, dense
   A/B at ≤2048 context (byte gate), long-context needle with the indexer
   active, ladder smoke, `bringup-check.sh qwen38flashnext`, family-doc
   measurements, docs.
4. Full suite green throughout; existing families untouched except shared
   plumbing that their byte-gates must confirm.

Open questions to resolve during implementation (tracked, not guessed):
- Whether `Qwen3_5RMSNorm` upcasts to FP32 internally (parity will show it).
- The exact float/bf16 casts around indexer pooling and scoring (reference
  uses fp32 for mean and scores; match it).
- `eos_token_id` for PLE segmentation comes from `config.json`
  `text_config.eos_token_id` (248044) — carry it in the manifest.
