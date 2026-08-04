# Inkling-Small on Mference

Checkpoint selection, memory budget, and the architecture gap list for running
[pipenetwork/Inkling-Small-MLX-4bit](https://huggingface.co/pipenetwork/Inkling-Small-MLX-4bit)
(276B total, ~12B active, natively multimodal) with SSD-streamed experts. This
document is the architecture contract for the `inklingSmall` family.

Conversion has been run and verified against the real checkpoint (see
**Status**). Throughput numbers remain *estimates* calibrated against measured
DSV4-Flash performance on the same machine — no Inkling token has been
generated yet, because the forward pass is not implemented.

## Checkpoint selection

Thinking Machines published Inkling-Small under Apache 2.0 in July 2026. Roughly
a dozen MLX conversions exist. Only one of them matches the quantization
contract Mference already enforces in `ManifestReader.validateQuant`
(`Sources/Mference/Infrastructure/ModelIO/ManifestReader.swift:206`).

| Candidate | Routed experts | Attention / embeddings | Disk | Verdict |
|---|---|---|---|---|
| `pipenetwork/Inkling-Small-MLX-4bit` | affine 4-bit g64 | affine 4-bit g64 | 148 GB | **Selected** |
| `mlx-community/Inkling-Small-mlx-3bit` | affine 3-bit g64 | BF16 | 121 GB | No INT3 kernel; no BF16 resident path |
| `Sawfwair/Inkling-Small-MLX-Mixed-2bit` | affine 2-bit **g128** | BF16 | 84.5 GB | g128 unsupported; 12 GB resident |
| `mlx-community/Inkling-Small-mxfp4` | MXFP4 | mixed | 140 GB | Non-affine number format |
| `pipenetwork/…-MLX-6bit` / `-8bit` | 6/8-bit | 6/8-bit | 200 GB+ | Exceeds free disk |
| `unsloth/…-GGUF`, `…-EXL3-*`, `…-NVFP4`, `…-AWQ-INT4` | — | — | — | Format not readable by the repacker |

Verified against revision `9d6e4720` by reading the safetensors headers
directly: `embed`, `unembed`, `attn.{wq_du,wk_dv,wv_dv,wo_ud,wr_du}`,
`mlp.experts.*`, `mlp.shared_experts.*`, the two dense MLP layers, and the
vision/audio projections are all **affine 4-bit, group 64, BF16 scales and
biases** — e.g. routed `gate_proj` is `U32 [256, 2048, 512]` with
`scales/biases BF16 [256, 2048, 64]` (512 × 32 ÷ 4096 = 4 bits, 4096 ÷ 64 = 64
groups). The router gate (`mlp.gate.weight`, `BF16 [258, 4096]`) is
**unquantized**, as it is in the DSV4 checkpoint.

That maps onto Mference's slot table with exactly one deviation:

| Slot | Mference allows | Inkling-Small 4-bit |
|---|---|---|
| embedding | 4 | 4 ✓ |
| attention | 4 | 4 ✓ |
| sharedExpert | 4, 8 | 4 ✓ |
| routedExpert | 2, 4 | 4 ✓ |
| router | 8 | BF16 ✗ |

The router deviation is cosmetic — 42 MB total across 40 layers. DSV4 ships a
BF16 router too, so whatever the repacker does there already applies.

### Why not the smaller quants

The 3-bit and mixed-2-bit builds are genuinely attractive on decode bandwidth
(see below), and the 3-bit build in particular protects the every-token path.
Both are rejected for the same structural reason rather than a quality one:
they leave attention and embeddings in **BF16**, and Mference has no resident
BF16 matmul path — the attention slot is required to be INT4 and every decode
GEMV is an INT4 or INT8 kernel. Adopting either means adding a BF16 weight path
*and* (for 3-bit) an INT3 unpack kernel whose weights straddle `u32` word
boundaries (3 does not divide 32). That is a second project layered on top of
an already large one.

Revisit once the text path is working: a 2-bit-routed / 4-bit-core build in the
shape of the DSV4 dynamic quant would reuse the existing INT2 MoE kernels and
roughly double decode throughput (see the budget table). No such checkpoint
exists publicly today; producing one means quantizing from the 527 GB BF16
release ourselves.

## Architecture summary

42 decoder layers, hidden 4096, vocab 201 024 (200 058 unpadded), untied
`unembed`, `model_max_length` 1 048 576.

- **Layers 0–1 are dense** (`dense_mlp_idx: 2`, intermediate 16 384). The
  remaining 40 are MoE: 256 routed experts of intermediate width 2048, top-6,
  plus **2 shared experts** with `shared_expert_sink: true` — the router emits
  258 logits, so the shared experts compete in the same softmax as sinks.
- **Router**: `gate_activation: sigmoid` with a learned bias (`use_gate_bias`),
  `norm_after_topk: true`, a per-layer learned `global_scale`, and
  `route_scale: 8.0`.
- **Attention is hybrid local/global**: 35 of 42 layers are sliding-window 512
  (`local_layer_ids`); the other 7 (layers 5, 11, 17, 23, 29, 35, 41) are
  global. 32 query heads over 8 KV heads, head dim 128, no q/o bias, per-head
  `q_norm`/`k_norm`.
- **No RoPE.** Position is encoded by a relative-attention bias. Each layer
  projects the residual through `wr_du` (4096 → `num_heads * d_rel` = 512) and
  reshapes it to a per-head 16-dim relative state, which mixes a bank of
  bias-vs-distance profiles (`rel_logits_proj.proj`, shape `[d_rel, extent]`)
  into one bias per backward distance; the bias is zero outside
  `0 ..< extent`.

  **The extent is per layer kind.** The reference takes
  `sliding_window_size if is_sliding else rel_extent`, so local layers ship
  `proj [16, 512]` and the 7 global layers ship `[16, 1024]` — confirmed
  against the checkpoint's safetensors headers (layers 5, 11 and 41 are 1024;
  layers 2, 4, 6, 7, 9, 10, 24 and 39 are 512). The `log_scaling` correction
  is likewise applied **only on global layers**.
- **Short convolutions** (`use_sconv`, `sconv_kernel_size: 4`) at four sites
  per layer: `attn_sconv` and `mlp_sconv` on the block inputs, and
  `attn.k_sconv` / `attn.v_sconv` on the K and V streams.
- **Log-scaled attention** beyond 128 K positions (`log_scaling_n_floor:
  128000`, `log_scaling_alpha: 0.1`).
- `use_embed_norm: true`; logits scaled by `logits_mup_width_multiplier: 16.0`;
  `rms_norm_eps: 1e-6`.
- 8 MTP (multi-token-prediction) layers and the vision (hMLP, 40 px patches)
  and audio (dMel) towers are present in the checkpoint and are **out of scope
  for v1**.

## Memory budget

Parameter counts derived from the config, byte counts at 4.5 bits effective
(4-bit payload + BF16 scale and bias per 64 weights = 0.5625 B/param).

| Group | Params | Bytes |
|---|---|---|
| `embed` + `unembed` | 1.647 B | 0.93 GB |
| attention (42 layers) | 1.850 B | 1.04 GB |
| shared experts (40 × 2) | 2.013 B | 1.13 GB |
| dense MLP (layers 0–1) | 0.403 B | 0.23 GB |
| router (BF16) + norms + sconv | 0.052 B | 0.10 GB |
| **Resident total** | **5.96 B** | **≈ 3.4 GB** |
| routed experts (40 × 256 × 3) | 257.7 B | 145.0 GB |
| **On disk** | | **≈ 148 GB** |

The 3.4 GB resident set is what makes this viable on a 24 GB machine, and it is
the strongest practical argument for the 4-bit build: the 3-bit build's BF16
attention and embeddings push resident to ≈ 8.3 GB, and the mixed-2-bit build
to ≈ 12 GB, on a box that also has to hold the KV cache, activations, and the
expert staging ring.

### KV cache

8 KV heads × 128 dims = 1024 per K and V, so 4 KB/token/layer. The 35 local
layers are pinned at window 512 (71.7 MB total, fixed). Only the 7 global
layers grow: **28.7 KB/token**.

| Context | Global-layer KV |
|---|---|
| 32 K | 0.94 GB |
| 128 K | 3.76 GB |
| 1 M | 28.7 GB — not feasible |

The advertised 1 M context cannot be served on 24 GB. Cap the context option at
128 K.

### Decode throughput

Per token the router touches 6 of 256 experts across 40 MoE layers:
40 × 6 × 3 × (2048 × 4096) = **6.040 B params**, i.e. **3.40 GB/token** at 4.5
bits — this is the decode bottleneck, exactly as it is for DSV4.

Calibrating against the measured DSV4 number rather than raw SSD bandwidth:
DSV4 reads 43 × 6 × 3 × 8.389 M = 6.493 B params/token at ~2.667 bits average
(g32 `gate_proj`) = 2.164 GB/token, and sustains 4.8 tok/s with 16 slots +
adaptive expert caching — an effective ~10.4 GB/s.

| Variant | Bytes/token | Projected |
|---|---|---|
| 4-bit (selected) | 3.40 GB | **≈ 3 tok/s** |
| hypothetical 2-bit routed / 4-bit core | 1.89 GB | ≈ 5.5 tok/s |

So expect roughly 60 % of DSV4's decode rate. That is the price of 4-bit
experts, and it is the main reason to revisit a 2-bit-routed build later.

## Forward-pass contract

Transcribed from the reference `inkling_mlx` package shipped in the checkpoint
repo (`attention.py`, `layers.py`, `moe.py`, `text.py`, `common.py`). This is
the spec the kernels must satisfy.

**Decoder layer** (`layers.py`) — note both short convs sit on the *sublayer
output*, inside the residual:

```
r = x;  h = attn_norm(x);  h = attn(h);  h = attn_sconv(h);  x = r + h
r = x;  h = mlp_norm(x);   h = mlp(h);   h = mlp_sconv(h);   x = r + h
```

**Short convolution** (`common.py`) — depthwise causal, `groups == channels`,
kernel 4, **no bias, no activation, and a residual add**, held in fp32
regardless of model dtype. Weight layout `[C, K, 1]`. Four independent conv
states per layer (`k_conv`, `v_conv`, `attn_conv`, `mlp_conv`), each carrying
the last `K-1 = 3` inputs; zero-filled at sequence start.

```
out = depthwise_causal_conv1d(x, w) + x        # fp32
```

**Attention** (`attention.py`):

- `q = wq_du(h)`; `k = k_sconv(wk_dv(h))`; `v = v_sconv(wv_dv(h))`;
  `rel = wr_du(h)`. The short convs apply to K and V **only** — not Q.
- `q_norm` / `k_norm` are per-head RMS norms over `head_dim`.
- **`scale = 1 / head_dim`, not `1 / sqrt(head_dim)`** — the reference is
  explicit that per-head RMS-normalized q/k call for `1/d`. Getting this wrong
  scales every logit by ~11.3×.
- Relative bias: `rel_logits = rel_states @ proj` → `[B, H, Lq, extent]`;
  `distance = q_pos - kv_pos`; gather at `clip(distance, 0, extent-1)`; the
  bias is **zero** (not masked) outside `0 <= distance < extent`.
- Log scaling, **global layers only**:
  `tau = 1 + alpha * log(max((pos+1)/n_floor, 1))`, applied to **both** `q`
  and the position bias.
- Additive mask = position bias + causal (`distance >= 0`, and
  `distance < sliding_window` on local layers), then standard SDPA.

**Router** (`moe.py`) — all of it in fp32; the reference warns that bf16
rounding flips near-tied top-k choices and compounds across layers:

```
router_logits = x @ weight.T                    # [T, 258] = 256 routed + 2 shared
scores        = sigmoid(router_logits)[:, :256]
selection     = scores + bias                   # bias affects SELECTION ONLY
topk_idx      = top-6(selection)
topk_logits   = concat(routed_logits[topk_idx], shared_logits)   # [T, 8] raw logits
weights       = softmax(logsigmoid(topk_logits)) * route_scale * global_scale
```

The final softmax spans the 6 selected **and** both shared experts — that is
the `shared_expert_sink`. `topk_weights = weights[:, :6]`,
`shared_gammas = weights[:, 6:]`, and

```
out = Σ_k routed_k · topk_weights_k  +  Σ_s shared_s · shared_gammas_s
```

**Dense MLP** (layers 0–1): `down(silu(gate(x)) * up(x)) * global_scale`.

**Head** (`text.py`): `embed_tokens = embed_norm(embed(ids))`;
`logits = unembed(h_final / logits_mup_width_multiplier)` truncated to
`unpadded_vocab_size = 200058`. Skipping the truncation lets the model emit
966 padding ids the tokenizer cannot decode.

## Gap list

Reusable as-is: sliding-window and full attention masks, GQA, per-head
`q_norm`/`k_norm`, INT4 GEMV and embedding lookup, the INT4 tiled prefill
matmul, the packed-expert streaming layout, and the expert cache.

New work, roughly in dependency order:

1. **Relative position bias** replacing RoPE outright (`wr_du` +
   `rel_logits_proj`, `d_rel` 16, `rel_extent` 1024). Mference's attention path
   assumes RoPE everywhere; this is a new kernel and a new attention variant.
2. **Short convolutions** (kernel 4) at four sites per layer, including
   per-sequence conv state that must live in the cache manager alongside KV and
   be rolled back on speculative rejection.
3. **Router variant**: sigmoid + bias + `norm_after_topk` + per-layer
   `global_scale` + `route_scale` 8.0, over 258 logits with 2 shared-expert
   sinks. Existing scoring functions are softmax / `sqrtsoftplus` / hash.
4. **Mixed dense/MoE layer graph** — 2 dense layers at a different intermediate
   width (16 384) ahead of 40 MoE layers.
5. **Log-scaled attention** past 128 K.
6. `embed_norm`, separate `unembed`, `logits_mup_width_multiplier` 16.0.
7. **Tokenizer**: tiktoken-based, 201 024 vocab, new chat template and
   `ChatDialect`, plus a tool-call parser.
8. ~~Registration surface~~ — **done**. `ModelFamily.inklingSmall`,
   `RelativePositionConfig`, the `inklingSmall_276B_A12B` baseline and
   `knownArchitectures`; the tensor-name contract in `Model.swift`; manifest
   arch round-trip (writer, decoder, `validateArch`);
   `model_type: "inkling_mm_model"` dispatch plus production cross-check in
   `ArchInfo.swift`; planner prefix/routed-container/slot ordering and
   vision-audio exclusion; `SupportedModelSource` and the app descriptor
   pinned to revision `9d6e4720`.

Items 1–3 are the substantial ones and have no analogue in the existing three
families. The overall shape is comparable to the DSV4 port (38 files).

## Status

**Conversion is done and verified against the real checkpoint.** The 148.4 GB
source was converted to `inklingsmall.gturbo`, all 48 output files
(148,421,176,017 bytes) pass `--verify-install` hashing, and the runtime
loader's `ManifestReader.load(directoryURL:expecting:)` accepts it against the
compiled baseline.

Measured on the produced install:

| | |
|---|---|
| `model_weights.bin` (resident) | 3,415,372,964 B — 3.42 GB, matching the 3.4 GB budget |
| expert blobs | 40 × 3,623,878,656 B (layers 2–41) |
| `expertStride` | 14,155,776 B = 3 × 4096 × 2048 at 4.5 bits |
| resident tensors | 840 |
| excluded vision/audio tensors | 18 |

Five real defects were caught — three by running the conversion against the
actual weights, two by transcribing the reference implementation:

0. **`attentionScale` was wrong by 11.3×.** It was derived as
   `1 / sqrt(head_dim)`, the near-universal convention. Inkling RMS-normalizes
   q and k per head and therefore scales by `1 / head_dim`
   (0.0078125, not 0.0883883). Nothing in the config file signals this; only
   the reference does.
1. **`unpaddedVocabSize` was missing.** Logits must be truncated to 200 058 of
   201 024 before sampling, or the model can emit padding ids that do not
   decode.

2. `attentionScale` was additionally hardcoded as the `128^-0.5` literal
   (0.088388347648318**45**) while the converter computes
   `1.0 / sqrt(128)` (…**43**) — one ULP apart, and `validateArch` compares
   exactly. The baseline now uses the same expression rather than a literal.
   DSV4's hardcoded 512 value happens to agree bit-for-bit, which is why this
   never bit before.
3. `ManifestReader` required a `packed_experts/layer_NN.bin` for *every* layer;
   the two leading dense-FFN layers legitimately have none. It now skips
   `numDenseLayers`.
4. `encodeLayout` published `expertsPerLayer` from `layers.first` — the dense
   layer 0 — writing 0 into `layout.json` while the manifest said 256, so
   `--verify-install` rejected the install. Both now select the first layer
   that actually has experts, and the verifier skips empty layout entries.

### Forward pass (2026-08-03, second session)

The decode path is implemented end to end: `produceTokenInkling` in
`RealForwardRunner`, four new Metal kernels in `Metal/Inkling/inkling.metal`
(depthwise short-conv step with fp32 streaming state, per-head Q/K RMS norm
with no V norm, single-token GQA attention with the relative-position bias
computed inline plus ring/window/log-`tau` handling, and the sigmoid router
select with shared-expert sinks), each parity-tested against a CPU reference
on random data (`InklingKernelTests`). Routed experts reuse the existing INT4
MoE phase kernels by padding top-6 to the contract's 8 slots with weight-0
duplicates (cache hits, no extra I/O). Prefill v1 replays the decode path
token by token — the DSV4-v1 pattern; conv state makes batched prefill a
follow-up, not a correctness need. The `.inkling` chat dialect implements the
shipped Jinja's framing (role tokens + `<|content_text|>` + `<|end_message|>`,
effort line, `<|content_model_end_sampling|>` stop).

**First light achieved 2026-08-04**: greedy completion of "The capital of
France is" → " Paris. The capital of Germany is Berlin. …"; chat-framed
"capital of France?" → "Paris" with a clean end-of-turn stop. The engine
matches pipenetwork's reference implementation layer-by-layer (cos ≥ 0.999,
first 11 layers verified against the real weights). Decode ≈ 2.4-6.5 tok/s
unoptimized; prefill v1 ≈ 12 s/token (sequential replay — batched prefill is
the top perf follow-up).

Post-first-light bug ledger (all found by CPU/oracle parity, in order):
FP16 residual overflow at L23 (stream is now FP32); FP16 sconv-delta
overflow (deltas now FP32); a single shared-expert channel clipping FP16 at
L41 (fixed by an exact ÷32/×32 prescale through the FFN output path); and
the root cause of the garbage output — `mlp.global_scale` on the two dense
layers is **BF16** while `mlp.gate.global_scale` is FP32, and the scalar
reader assumed FP32, poisoning layers 0-1 with a garbage gain. The engine's
own CPU parity harness could not catch that one (it shared the accessor);
only the independent reference oracle could — which is why
`Tests/.../InklingLayerParityTests.swift` keeps both modes.

An earlier note in this section blamed a first garbage run on corrupt
resumed shards — that was real, but The first real run
produced garbage, and per-stage numeric probes (`MFERENCE_INKLING_DEBUG=1`)
traced every NaN source to tensors whose bytes came from source shards 1–3 —
exactly the three shards the original download had left truncated and the
install had "repaired" with a byte-offset resume. The truncated files were
not clean prefixes (the downloader wrote sparsely), so 128 tensors were
corrupt on disk while all 28 shard headers still parsed and every
size/hash check passed — install verification hashes what the converter
*wrote*, not what it *read*. Lesson recorded: a resumed foreign download is
untrusted input; re-fetch the whole shard or verify against upstream LFS
digests before converting. A clean streaming reinstall from Hugging Face is
the fix.

Bring-up config note: the fused QKV GEMV and the constant-folded INT4 GEMV
variants are bypassed for this family (plain generic GEMVs) — they were
briefly suspected during the corruption hunt and are unproven at this
family's shapes (`m == n == 4096`; fused (4096, 1024, 4096)). Re-enable one
at a time against known-good output.

Scope note: the text path is the target. The vision and audio towers ship in
the checkpoint and are excluded by the planner. The MTP config block is
present but the checkpoint carries **no** MTP tensors, so there is nothing to
exclude there.
