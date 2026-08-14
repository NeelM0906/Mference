# DeepSeek-V4-Flash on Mference

Port design, memory budget, and expected throughput for running
[mlx-community/DeepSeek-V4-Flash-2bit-DQ](https://huggingface.co/mlx-community/DeepSeek-V4-Flash-2bit-DQ)
(284B total, ~13B active) with SSD-streamed experts. This document is the
architecture contract for the `deepseekV4Flash` family; the numbers below are
budget *estimates* pending measurement on hardware.

## Source checkpoint

- Repo: `mlx-community/DeepSeek-V4-Flash-2bit-DQ` — an MLX affine dynamic
  quant: **2-bit routed experts, 4-bit everything else** (BF16 scales +
  biases), ~96.5 GB on disk. Verified against revision `722bf559`: routed
  `gate_proj` is **group 32** on layers 0–41 and group 64 on layer 42;
  `up_proj`/`down_proj` and the 4-bit core are group 64 throughout; the
  router gate (`ffn.gate.weight`) is **unquantized BF16** `[256, 4096]`.
- Upstream architecture: `deepseek_v4` (`DeepseekV4ForCausalLM`), 43 layers,
  hidden 4096, vocab 129 280, untied lm_head.

## Architecture summary

Every one of the 43 layers is an MoE layer (there are no dense-FFN layers):
1 shared expert plus 256 routed experts of intermediate width 2048, top-6
routing. The first 3 layers route by a frozen `tid2eid[token]` lookup (hash
routing); the rest use a learned router with `sqrtsoftplus` scoring,
`e_score_correction_bias` for selection only, top-6 weights renormalized and
scaled by 1.5. Gate/up activations are clamped to ±10 (`swiglu_limit`)
before SiLU.

Attention is shared-KV MQA: 64 query heads over a **single** 512-dim KV head
where K and V are literally the same tensor. Per layer:

- `q_a_proj` 4096→1024, RMS norm, `q_b_proj` 1024→64×512, per-head
  unweighted RMS norm.
- `kv_proj` 4096→512, RMS norm; interleaved partial RoPE on the trailing 64
  dims (rope pairs are adjacent channels `(2i, 2i+1)`, θ = 10 000 for
  sliding layers, θ = 160 000 for CSA/HCA layers with **YaRN** frequency
  correction — factor 16, original max positions 65 536, beta 32/1,
  attention_factor forced to 1.0, applied at every position).
- Per-head learnable attention sinks (gpt-oss style) folded into the softmax.
- After attention, the conjugate rotation (−sin at the query position) is
  applied to the output's trailing 64 dims (K=V means V carried RoPE).
- Grouped output projection: 8 groups of 4096 → 1024 each (`o_a_proj`),
  then 8192 → 4096 (`o_b_proj`).

Layer kinds (`compress_ratios` = `[0, 0, 4, 128, 4, 128, …]`):

- Layers 0–1: sliding-window only (window 128).
- Layers 2, 4, …, 42 (21 layers): **CSA** — sliding window ∪ compressed
  entries (4 source tokens → one 512-dim entry via softmax-gated pooling
  with learned position bias, two overlapping series Ca/Cb of stride 4 and
  width 8), gathered per query by a lightning indexer (64 heads × 128 dims,
  ReLU scoring, top-512 entries).
- Layers 3, 5, …, 41 (20 layers): **HCA** — sliding window ∪ non-overlapping
  128→1 compressed entries, no indexer (dense over compressed entries).

The residual is 4 parallel streams (`hc_mult = 4`) mixed by two
Manifold-Constrained Hyper-Connection sites per layer (Sinkhorn-projected
4×4 doubly-stochastic combine, 20 iterations, plus sigmoid `pre`/`post`
weights computed from the flattened streams). The embedding is broadcast
into 4 streams; a HyperHead collapses them before the final norm.

### KV state per layer at 4K context

| Kind | Sliding K=V | Compressed | Indexer keys | Pending buffers |
| --- | --- | --- | --- | --- |
| SWA (2) | 127 × 512 fp16 ≈ 127 KiB | — | — | — |
| CSA (21) | 127 KiB | ≤1024 × 512 fp16 = 1 MiB | ≤1024 × 128 fp16 = 256 KiB | ≤4 rows kv+gate (compressor 2×1024, indexer 2×256) + Ca overlap |
| HCA (20) | 127 KiB | ≤32 × 512 fp16 = 32 KiB | — | ≤128 rows kv+gate (2×512) ≈ 256 KiB |

Total attention state at 4K ≈ **38 MiB** — the streamed-expert design's KV
cost is negligible for this model.

## Memory budget (decode, 4K context)

Weights use MLX affine quantization: bits + 4 bytes of BF16 scale+bias per
group of 64, i.e. 4-bit ⇒ 0.5625 B/weight, 2-bit ⇒ 0.3125 B/weight.

Resident (mmap'd, hot at decode):

| Component | Params | Bytes |
| --- | ---: | ---: |
| Attention stack (q_a/q_b/kv/o_a/o_b), 43 layers | 4.60 B | 2.59 GB |
| CSA/HCA compressors + indexers | 0.49 B | 0.27 GB |
| Routers + norms + sinks + `tid2eid` | ~0.06 B | ~0.03 GB |
| Shared experts (43 × 25.2M, 4-bit) | 1.08 B | 0.61 GB |
| Hyper-connection mixes (BF16, unquantized) | 0.034 B | 0.07 GB |
| `lm_head` (untied, 4-bit) | 0.53 B | 0.30 GB |
| `embed_tokens` (4-bit, cold — one row/token) | 0.53 B | ~0 resident |
| **Resident hot total** | | **≈ 3.9 GB** |

Attention/compressed state at 4K (≈38 MiB) plus decode scratch (<100 MiB)
add ~0.1 GB.

Streamed experts: one expert blob = 25.17M params at 2-bit ≈ **7.87 MB**
(16 KiB-padded). Per token, 43 layers × 6 experts = 258 blobs ⇒ **~2.03 GB
of expert reads per token** at 0% cache hit.

Expert-slot memory and the throughput ladder (per-layer slot pools, like
Gemma/Qwen; the runtime's allowed slot counts are now 8/16/24/32/64/96/128
plus an explicit `resident` mode, and top-6 routing needs at least 6
in-flight blobs per layer):

| Slots/layer | Slot RAM | Peak footprint | Expected decode |
| ---: | ---: | ---: | --- |
| 8 (runtime minimum) | 2.71 GB | **≈ 6.8 GB** | I/O-bound: ~2–3 tok/s on ~6–7 GB/s SSDs (M4/M5-class), ~1.2 tok/s on M2-Air-class |
| 16 | 5.41 GB | ≈ 9.4 GB | ~4–7 tok/s if routing skew matches Qwen 3.6's measured hit rates |
| 32 | 10.8 GB | ≈ 14.9 GB | ~6–10 tok/s; the only configuration with headroom to hold ≥7 tok/s |

The floor is therefore **≈ 6.8 GB peak footprint** with the current
per-layer 8-slot minimum — dominated by resident 4-bit attention weights
plus in-flight expert slots. Two future-work levers lower it: a 6-slot
(= top-k) per-layer mode saves ~0.7 GB, and a shared cross-layer slot pool
(double-buffered 6+6 slots, ~100 MB total) would cut the floor to
**≈ 4.1 GB** at the cost of all cross-token caching.

Unlike Gemma 4 (~2 GB) and Qwen 3.6 (~1.45 GB), this model cannot approach
1.4 GB: those budgets work because their *active* cores are ~3–4B params in
a 2–2.8K hidden size. V4-Flash's shared core alone (attention + shared
experts + head at the checkpoint's own 4-bit) is ~3.9 GB, and re-quantizing
is out of scope — the installer is a byte mover by design.

### Why >7 tok/s is SSD-limited

7 tok/s × 2.03 GB/token = **14.2 GB/s** of sustained random 8 MB reads —
beyond any current Apple SSD (~5–8 GB/s). Reaching ≥7 tok/s therefore
requires ≥~60% of expert reads served from RAM. Qwen 3.6 measurements show
LFU hit rates of that order *are* achievable with 16–32 slots when routing
is skewed; whether V4-Flash's routing (especially with hash-routed layers
0–2, which are perfectly prefetchable by token id) matches that skew is a
measurement question. Honest expectation: **1.5–3 tok/s at the ~6.8 GB
floor; 4–7 tok/s at ~9.4 GB (16 slots) on M4-Pro/M5-class hardware;
>7 tok/s only at 24–32 slots (≥12 GB) or after the roadmap's I/O–compute
overlap work.** GPU compute alone (13B active, ~4.3× Qwen 3.6's) bounds
even a fully-cached decode to roughly 8–12 tok/s on an M5-class part.

Storage: ~97.5 GB installed (`layer_%02d.bin` = 256 × 8 MiB = 2 GiB × 43 —
every layer padded to the widest expert stride so the format keeps one
global `expertStride` — plus ~4 GB resident file).

## Port mapping (family `deepseekV4Flash`)

- Layer mask values: `0` = sliding-window, `3` = CSA, `4` = HCA (mask
  `[0,0,3,4,3,4,…]`). Values 1/2 stay Gemma/Qwen-only.
- New `ArchConfig` extension blocks (all defaulted so Gemma/Qwen manifests
  are untouched): `compressedAttention` (qLoraRank 1024, oLoraRank 1024,
  oGroups 8, ropeHeadDim 64, indexer dims, compress rates 4/128,
  compressRopeTheta 160 000, sinks), `hyperConnections` (mult 4, 20
  Sinkhorn iterations, eps 1e-6), `numHashRoutedLayers` 3,
  `routerScoringFunc "sqrtsoftplus"`, `routedScalingFactor 1.5`,
  `swigluLimit 10`.
- Quant slots: routedExpert allows {2, 4}; the 2-bit MoE kernels unpack 16
  weights per u32 (`row_bytes = N/4`).
- Repack: `model_type == "deepseek_v4"`; routed container
  `.ffn.switch_mlp.` under the `model.` prefix, attention at `attn.wq_a` /
  `attn.wq_b` / `attn.wkv` / `attn.wo_a` / `attn.wo_b` / `attn.attn_sink`,
  compressor at `attn.compressor.{wkv,wgate,norm,ape}`, indexer at
  `attn.indexer.{compressor.*,wq_b,weights_proj}`, norms `attn_norm` /
  `ffn_norm`, mHC at `attn_hc` / `ffn_hc` / `hc_head.{fn,base,scale}`
  (verified checkpoint naming); hash tables (I64, read dtype-aware on the
  CPU), sinks, position biases (BF16 `ape`, widened to FP32 at load), and
  HC mixes are resident pass-through tensors.
- Chat dialect: `deepseek` — `<｜begin▁of▁sentence｜>`,
  `<｜User｜>…<｜Assistant｜></think>…<｜end▁of▁sentence｜>` (non-thinking
  default mirrors the ChatML port's `enable_thinking: false` posture), DSML
  tool-call blocks (`<｜DSML｜tool_calls>` / `invoke` / `parameter`).
- Decode-time CPU sync points per layer reuse the existing router-readback
  structure; CSA layers add an indexer-score readback only once the
  compressed count exceeds `index_topk` (context > 2048), i.e. never at the
  default 4K context's first half.
- Prefill v1 ran the decode path token-by-token (the Qwen port's
  chunked-prefill ≡ sequential-decode guarantee is the correctness
  contract). The chunked implementation has since landed
  (`DSV4ChunkedPrefill`): a span's eligible prefix is batched, and only
  the remainder past the lightning-selection cutover replays
  token-by-token.

## First-install verification record

The port was originally written against the HF reference implementation
without the checkpoint on hand; the items below were verified against the
real snapshot (revision `722bf559`) on an M5 24 GB Mac:

1. **Pins.** Recorded: commit `722bf559b7de93575b2320973cf2002e05bfe6c9`,
   index SHA-256 `d1c2d929ab0a35be32cf18026bb31d6f99dad58d6c93a5a2abbe43791f9d6c30`
   in `SupportedModelSource.deepseekV4Flash` and the app descriptor.
2. **Tensor names.** The conversion uses MLX-native names (`attn.wq_a`,
   `.ffn.switch_mlp.`, `attn_norm`, `hc_head.fn`, …); the planner, runtime
   lookups, and fixtures were realigned to them (see Port mapping above).
3. **Unquantized dtypes.** Norms BF16; `attn_sink` / hc mixes /
   `e_score_correction_bias` FP32 as expected. Divergences handled:
   compressor `ape` ships BF16 (widened to FP32 once at load) and
   `tid2eid` ships I64 `[129280, 6]` (admitted into the format, read
   dtype-aware on the CPU decode path).
4. **Quant recipe.** Mixed: 2-bit routed experts with gate_proj at group
   32 (layers 0–41) / 64 (layer 42), 4-bit core at group 64, router BF16.
   The INT2 kernels take a per-layer gate group size derived from the
   manifest; layer files are padded to one global expert stride.
5. **Numerics.** YaRN frequency correction (compress rope only) is
   implemented and unit-tested against the transformers reference values.
   Remaining open item: a KLD/logit comparison against the HF
   implementation on a fixed prompt — greedy decode produces coherent,
   factually correct output and clean end-of-turn stops, but no
   token-level parity check has been run yet.

## Measured performance (M5, 24 GB, macOS 26)

Greedy decode, 128 new tokens, 4 K context option, short prompt; one-time
per-process load (mmap + full SHA-256 of 90 GiB) is ~39–43 s and excluded:

| Slots/layer | rdadvise | Decode tok/s | Peak footprint |
| ---: | --- | ---: | ---: |
| 8 | off | 3.62 | 3.0 GB |
| 8 | adaptive | 4.17 | 3.0 GB |
| 16 | off | 4.02 | 5.9 GB |
| 16 | adaptive | **4.80** | 5.9 GB |
| 32 | off | 4.05 | 11.7 GB |
| 32 | adaptive | 4.10 | 11.7 GB |

Sustained 512-token generation at 16 slots + adaptive: 3.77 tok/s.
16 slots + `--rdadvise adaptive` is the sweet spot; 32 slots buys nothing
over 16 (decode is SSD-read-bound and LFU hit-rate gains saturate), so the
budget table's ≥7 tok/s tier remains out of reach without the roadmap's
I/O–compute overlap work — consistent with the analysis above.

### Shadow speculative prefetch (accepted default, 2026-08-07)

The first slice of that overlap work shipped: the pilot lookahead router
(layer L+1's routing computed from layer L's state) now feeds an
asynchronous prefetcher that joins the critical path only when a predicted
expert is actually routed, reads at user-initiated I/O priority, and issues
at most two weight-ranked predictions per layer (`MFERENCE_SHADOW_BUDGET`
overrides the budget). Output is byte-identical;
the mode is the DSV4 production default (`MFERENCE_SPEC_PREFETCH` still
overrides). Community-protocol A/B on the two naturally-terminating cases,
16 slots + adaptive:

| Case | Decode before | Decode after | Gain |
| --- | ---: | ---: | ---: |
| short-explanation | 4.41–4.46 tok/s | 5.10–5.29 tok/s | +17.8% |
| long-synthesis | 3.75–3.82 tok/s | 4.26–4.32 tok/s | +13.7% |

Long-prompt prefill improved ~7% as a side effect. Details and the
rejected variants (blocking pilot, unthrottled issue, utility-QoS reads,
32-slot scaling) are recorded in
[experiments/summaries/13](experiments/summaries/13-dsv4-streaming-iterations.md)
and [14](experiments/summaries/14-dsv4-shadow-prefetch.md).
