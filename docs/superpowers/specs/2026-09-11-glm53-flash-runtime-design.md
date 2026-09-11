# GLM-5.3-Flash runtime design

Date: 2026-09-11. Family: `glm53Flash`. Dossier:
[`docs/families/GLM53_FLASH.md`](../../families/GLM53_FLASH.md).

## Goal

Run `pipenetwork/GLM-5.3-Flash-MLX-mixed-4_8bit` (320B-A18B, 181.9 GB) through
Mference's production funnel — installer, CLI, server, UI — with the same
discipline as the shipped families: a Day-0 contract, toy parity against the
vendor-faithful reference, chunked prefill equal to sequential decode, the
frozen protocol page, and a phases snapshot. Target host for first light and
the protocol: Mac Studio M3 Ultra, 256 GB, where the whole install fits in
memory. The targets we are chasing there, from the byte counts and from two
sibling engines: decode 25–35 tok/s after a perf pass (roofline ≈ 50: a step
reads ≈ 14 GB of weights at ≈ 700 GB/s), 6–10 tok/s at first light; batched
prefill 150–400 tok/s (roofline ≈ 780 at 36 GFLOP per token).

## The model, as the reference computes it

Source of truth: PipeNetwork's `glm53-flash-mlx` at `a61a7c7d` (MIT), whose
`language.py` reproduces `transformers` 5.16 at 1e-6 end to end and fixes four
upstream mlx-vlm bugs (missing swiglu clamp, bf16 mHC `base`/`scale`, two
epsilons). Everything below is read from it.

Residual: the embedding row is tiled into `hc_mult` = 4 streams. Each layer:

```
xc, post, comb = attn_hc(streams)          # mixes from rmsnorm(flat streams) @ fn.T
r  = attention(input_layernorm(xc))
streams = post * r + comb^T streams
xc, post, comb = ffn_hc(streams)
m  = mlp(post_attention_layernorm(xc))
streams = post * m + comb^T streams
```

then `norm(mean over streams)` and the INT8 `lm_head`. The mHC arithmetic
(pre = sigmoid + eps, post = 2 sigmoid, comb = softmax + eps then Sinkhorn:
column, then 19 × (row, column), each with `+ eps`) is DeepSeek V4's, which
`dsv4_hc_weights` / `dsv4_hc_collapse` / `dsv4_hc_place_mix` implement; the
runner threads `rms_norm_eps` = 1e-5 and reads `fn` as BF16.

**KDA layer** (34 of 45, mask 7). With `H` = 64 heads of `D` = 128:

```
q, k, v      = q_proj(x), k_proj(x), v_proj(x)               # INT8, each H*D
mixed        = concat(q, k, v)                                # 3*H*D channels
conv_in      = concat(conv_state[3 rows], mixed)              # carried tail
conv_out     = silu(depthwise_conv1d(conv_in, kernel 4))      # BF16 taps
q, k, v      = split(conv_out)
q            = l2norm(q) * D^-0.5 ; k = l2norm(k)             # eps 1e-6 inside the sum
a            = f_b(f_a(x))                                    # low-rank, INT8, [H*D]
g            = -5.0 * sigmoid(exp(A_log[h]) * (a + dt_bias))  # per (h, d): log decay
beta         = sigmoid(b_proj(x))                             # per head
S[h]        *= exp(g[h, :])[:, None]                          # row-wise decay over dk
kv_mem       = S[h]^T k[h]  ;  delta = (v[h] - kv_mem) * beta[h]
S[h]        += k[h] ⊗ delta ;  y[h] = S[h]^T q[h]
gate         = g_b(g_a(x))                                    # low-rank, [H*D]
out          = o_proj( rmsnorm_D(y) * o_norm.weight * sigmoid(gate) )
```

State per layer: `S` `[H, Dk, Dv]` fp32 (4 MB) and the 3-row conv tail.
Mference's GDN kernels keep the same state layout but decay per head and gate
`z` with SiLU/sigmoid on a full-width projection; the per-channel decay, the
low-rank gates and the `-5 · sigmoid` form make this a separate kernel. The
decode and prefill kernels are ported from `metal/glm53_kda.metal` in
`IngeniousIdiocy/ds4` (`glm53-m3ultra` at `90d71e0d`, MIT with the ggml
copyright notice retained), whose inputs are fp32 q/k/v/gate/beta and which
carries per-row state snapshots for a future speculative verifier.

**Sparse layer** (11 of 45, mask 8). NoPE MLA in absorbed form:

```
qr        = rmsnorm(q_a(x))                          # 1536, eps 1e-5
q         = q_b(qr) → [64 heads, 256]
latent_t  = rmsnorm(kv_a(x))                         # 512, eps 1e-5; appended to the cache
sel       = indexer(x, qr) or all                    # token indices, ≤ 2051
q_lat[h]  = embed_q[h] q[h]                          # [512] per head (INT8 [64, 512, 256])
p         = softmax(q_lat · latent[sel] * 256^-0.5)  # one shared K = V per position
o_lat[h]  = Σ p latent[sel]                          # [512]
o[h]      = unembed_out[h] o_lat[h]                  # [256] (INT8 [64, 256, 512])
out       = o_proj(concat o)                         # INT8 [4096, 16384]
```

`dsv4_attention_decode` already attends one 512-wide shared K = V over an
explicit selection list; the GLM variant drops the window ring and the sinks
and adds the per-head fold / unfold GEMVs.

**Pooled indexer** (on every sparse layer):

```
k_t   = layernorm(wk(x_t); eps 1e-6, with bias)      # 128
g_t   = x_t @ pool_gate^T                            # 128
pools j: tokens 4j..4j+3 (complete only); w = softmax_c(g[4j+c] + ape[c]) per channel
pool_key[j] = Σ_c w[c] * k[4j+c]
q_idx  = wq_b(qr) → [32 heads, 128]
score[j] = Σ_h (weights_proj(x)[h] * 32^-0.5) * relu(q_idx[h] · pool_key[j] * 128^-0.5)
sel    = expand(top 512 visible pools) ∪ tail (the ≤ 3 tokens past the last complete pool)
bypass: while the cache holds ≤ 2048 tokens the layer attends everything
```

Flash-Next's indexer pools keys too (mean of a block) and keeps a tail, so its
state manager and score kernel are the starting point; the gated per-channel
softmax pooling and the `[k | gate | valid]` packed cache are new.

**FFN.** Layers 0–2: dense `down(silu(min(gate, 10)) * clip(up, ±10))` at
width 12 288, INT8. Layers 3–44: router logits `x @ gate.T` in fp32 from the
BF16 gate; `scores = sigmoid(logits)`; selection on `scores + bias` (top-8,
no groups); weights `scores[sel] / Σ scores[sel] * 2.5`; `y = Σ w_e
expert_e(x) + shared(x)`, every expert and the shared expert clamped the same
way. Experts are INT4 g64 `[288, 2048, 4096]` / `[288, 4096, 2048]` stacked.

**Precision facts** that gate the kernels: mHC `base`/`scale`, `A_log`,
`dt_bias`, `e_score_correction_bias` fp32; router logits fp32; low-rank norms
eps 1e-5, indexer LayerNorm eps 1e-6; the l2norm adds its eps inside the sum
of squares.

## Layout

Generic pre-quantized path. Resident file: every `language_model.` tensor
under its conversion name, INT8 with companions, BF16 / FP32 passthroughs.
Expert blobs: the stacked `mlp.switch_mlp.{gate,up,down}_proj` triplets into
the standard per-layer blobs (14,155,776 B per expert, page-aligned as
shipped), none for layers 0–2. `vision_model.*` excluded and recorded as the
`vision` sidecar. Tokenizer sidecars: `tokenizer.json`, `tokenizer_config.json`,
`chat_template.jinja`, `config.json`.

## Parity plan

Two tiers, as for V4.1. (1) Goldens from PipeNetwork's runtime at the toy
geometry (`ArchConfig.glm53Toy()`, every quantized inner dimension a multiple
of 64 so the toy is quantized exactly as production is), capturing per layer:
the conv output, `g`, `beta`, the recurrent state and `y` on KDA layers; the
pooled keys, scores, selections and attention output on the sparse layer;
router indices and weights; every mHC `pre`/`post`/`comb`; logits. An fp32 CPU
oracle in Swift must reproduce them at 1e-4 with every integer decision exact.
(2) The Metal runner against the oracle at the FP16 tier, margin-aware for the
discrete decisions, and chunked prefill bit-identical to sequential decode.
The toy prompt exceeds `index_topk` = 4 so the pooled selection is exercised,
and `swiglu_limit` = 0.5 so the clamp is load-bearing.

## Runner shape

Per-token first (correctness), then layer-major prefill (each expert read once
per chunk) and the ds4-derived decode work: fold residual add and mHC expand
into matvec epilogues, fold router top-8 onto the router tail and the router
into the shared-expert grid, expert-parallel down projection, a radix fast
path for the indexer's top-512 pools, fewer command buffers per layer. The
expert cache: `resident` when the host can hold the set, slot rungs otherwise.

## Open questions

- Whether the INT8 `embed_q` / `unembed_out` per-head folds run as 64 small
  GEMVs or one batched kernel (the toy will not decide this; the phases
  snapshot will).
- The tail rule's interaction with the dense bypass at exactly 2,048 tokens:
  the reference bypasses while `T <= index_topk`, so position 2,048 is the
  first selected step; the toy prompt crosses `index_topk` = 4 to pin it.
