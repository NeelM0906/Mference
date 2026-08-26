# Inkling-Small on Mference

Checkpoint selection, memory budget, and the architecture gap list for running
[pipenetwork/Inkling-Small-MLX-4bit](https://huggingface.co/pipenetwork/Inkling-Small-MLX-4bit)
(276B total, ~12B active, natively multimodal) with SSD-streamed experts. This
document is the architecture contract for the `inklingSmall` family.

Conversion has been run and verified against the real checkpoint (see
**Status**), and the forward pass has since shipped. The current native top-6
path measures 3.0–3.7 tok/s on a 24 GB M5, a 16.4% geometric-mean gain over
the prior engine. A 256 GB M3 Ultra measures 5.3–6.9 tok/s at an ~8.9 GiB peak
footprint ([BENCHMARKS_M3_ULTRA.md](BENCHMARKS_M3_ULTRA.md)). Throughput figures
elsewhere in this document that are labeled *estimates* predate those runs
and are kept for the planning context they carried; the benchmark document
is the measured record.

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
they leave attention and embeddings in **BF16**, and Mference's shared
streamed-expert engine has no resident BF16 matmul path — the attention slot
is required to be INT4 and every decode GEMV is an INT4 or INT8 kernel. (The
later Maple family's self-contained runner carries its own ternary and BF16
paths, but they do not extend to this engine.) Adopting either means adding
a BF16 weight path
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
on random data (`InklingKernelTests`). The first implementation reused the
existing INT4 MoE phase kernels by padding top-6 to eight slots with
weight-zero duplicates; the native top-6 path described below removed that
compute waste. Prefill v1 replays the decode path
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
L41 (fixed by an exact ÷32/×32 prescale through the FFN output path — only
partly, see **FFN output range** below); and
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

### FFN output range (2026-08-04)

The ÷32/×32 prescale above was an incomplete fix, and the gap produced the
first *plausible-looking* corruption this family has shipped: occasional long
or rare words truncated mid-word and replaced by a burst of exclamation marks
("salt marshes, mang**!!!!**" for "mangroves"; "cancellation / ded**!!!!**"
for "deduplication"). Deterministic, reproducible under greedy decode, and
absent from all three other families under the same harness.

The prescale divides the router weights and the shared gammas, so it protects
`h1`/`h2` — the *post*-gamma FFN output. It does nothing for the **raw
shared-expert down-projection rows**, which are written before any gamma is
applied. On the released checkpoint, layer 41 channel 3895 of that raw row
runs at **1.5e4 – 6e4 on every token** — permanently within a factor of two
of FP16's 65 504 ceiling. The token that pushed it to **69 307** clipped to
`-inf`.

The failure then laundered itself through four stages, which is why nothing
caught it earlier:

1. `-inf` reached the FP32 residual via the ×32 residual add. Layer 41 is the
   last layer, so no KV entry was poisoned — only the layer's `mlp_sconv`
   state, which holds the last `K-1 = 3` inputs. That is exactly the observed
   **four-token** burst, self-clearing on the fifth.
2. The head's RMS norm squared the `inf` into an `inf` sum-of-squares, so
   `1/sqrt(ss)` was `0` and `-inf * 0` produced **NaN**, which the INT4 GEMV
   then smeared across all 200 058 logits.
3. The CPU greedy argmax seeds its running best at `(index 0, -infinity)`.
   Every `v > best` comparison against NaN is false, so it returned **token
   id 0** — and id 0 in this vocabulary is `!`. Both output-head paths emit the
   same `!`, because the NaN row is produced upstream of the split: the greedy
   fallback above is traced, and the sampled path's own degeneration on an
   all-NaN distribution was not traced further once the shared cause was
   established.
4. Nothing in the manifest, install verification, or per-layer parity harness
   inspects a mid-generation logit row, and a stray `!` reads as a model tic
   rather than an engine fault.

Fix: the shared-expert (and, in prefill, every streamed expert's) down
projection writes **FP32** rows —
`dequant_int4_gemv_simd_f32out`, consumed by `inkling_gamma_combine_f32in` in
decode and `inkling_scale_accum_f32_from_f32` in prefill. Arithmetic is
unchanged; only the store widens. Post-fix the same token's channel reads
-69 307.5 and `h1` lands at -7 064, a 9× margin inside FP16, and the
completion is "salt marshes, mangroves, and seagrass beds".

Second fix, independent of the first: `inkling_head_epilogue` now counts
non-finite real logits and the runner throws `InklingHeadError` on a non-zero
count, with the argmax seed moved to index `-1`. Any future numeric blowup
fails loudly at the position where it happens instead of rendering as `!`.

Lesson recorded: a *scale applied after a narrowing store* protects nothing.
When an outlier channel forces a prescale, the prescale has to sit upstream of
every FP16 store on that path, or the store has to widen.

Audit of the remaining FP16 stores on the FFN output path, for whoever hits
the next one:

| Store | Prescaled before it? | Measured headroom |
|---|---|---|
| routed down projection (decode, `moe_phase2_down_reduce_k6`) | yes — `routing_w` carries ÷32 and the reduce is FP32 | `h2` peaked at 4 764 of 65 504 |
| shared down projection (decode + prefill) | **no** — gamma applies after | **overflowed**; now FP32 |
| `h1` / `h2` after the gamma combine | yes | 7 064 of 65 504 at the worst observed token |
| dense MLP down projection, layers 0-1 | **no** — `mlp.global_scale` applies after, same pattern | not overflowing: the L0/L1 residual peaks near 1e2, so ~500× margin. Left FP16 deliberately; widen it if those layers ever grow. |

### Historical prefill cost profile (2026-08-04)

Measured on an M3 Ultra / 256 GB with `MFERENCE_PREFILL_BREAKDOWN=1`, greedy,
`--verify trusted-receipt --prefill-chunk auto`. Three changes landed together:
receipt-based verification instead of re-hashing every expert file on first
touch, a prefill chunk capacity that follows `--prefill-chunk` instead of a
hardcoded 128, and per-expert batched GLUs replacing per-(token, expert)
dispatches.

| Prompt | Before | After | Gain |
|---|---:|---:|---:|
| short-explanation, 59 tok | 65.8 s | 9.6 s | 6.8x |
| medium-review, 421 tok | 107.0 s | 28.2 s | 3.8x |
| long-synthesis, 2 785 tok | 483.2 s | 205.7 s | 2.3x |

Where the remaining 205.7 s of the long case goes:

| Component | Time | Share |
|---|---:|---:|
| Routed-expert streaming (serialized, 1.89 ms x 8 651) | 16.4 s | 8.0 % |
| Routed-expert compute (batched GLUs) | 13.5 s | 6.6 % |
| Per-token attention / norm / short-conv / tail | 175.8 s | **85.5 %** |

The last row is the one that matters, and it was not obvious: the whole expert
path — streaming *and* compute — is ~15 % of prefill. It also grows
superlinearly, 13.3x for 6.6x the tokens, because 7 of 42 layers are global
while the other 35 are capped at window 512.

Those measurements identified two follow-ups:

1. **Batched prefill attention was the main compute opportunity.** The
   chunk-wide portable and Apple10 TensorOps paths described below now cover
   Inkling's relative-position attention, including a wrapped sliding KV ring.
2. **Expert streaming is serialized.** `prefillInklingChunk` awaited a fetch,
   encoded, then drained, so the fetch never overlapped GPU work.
   *Addressed 2026-08-20:* the loop now runs a depth-1 pipeline on the
   `PrefillRoutedTileScheduler` contract — expert e+1 is planned and preaded
   while expert e's GLU runs, misses placed only in slots the in-flight
   buffer does not touch, output byte-identical.
   `MFERENCE_INKLING_PREFILL_PIPELINE=0` restores the serialized loop.
   Decode-side, the DSV4 pilot/shadow speculative prefetch is also ported
   (`SpeculativeRouterInkling`, opt-in via `MFERENCE_SPEC_PREFETCH=shadow`
   until an install A/B accepts a default).

Method note, recorded because it cost real time: the first three attempts at
this ranked the levers from dispatch counts and a per-pair cost fitted across
prompt sizes. That fit was total prefill time divided by pair count, so it
silently contained attention, the tail, and streaming — it looked constant
because it measured everything, and every extrapolation from it was wrong by
5-20x. Profile with `MFERENCE_PREFILL_BREAKDOWN=1` before optimizing.

### Batched prefill kernels (2026-08-05)

Inkling prefill now operates on a whole chunk through dedicated FP32 RMSNorm,
INT4 projection, fixed-K short-convolution, Q/K normalization, relative GQA
attention, BF16 routing, and fused short-convolution/residual kernels. Below
512 tokens, the portable relative-attention kernel avoids cooperative-tile
setup cost. On Apple10, chunks starting at position 512 use an 8-query x 64-key
TensorOps online-softmax kernel. A wrapped sliding-window range is split into
its two physical KV spans, so QK/PV tiles never require a staging copy or cross
the ring boundary.

Targeted prefill A/B on an Apple M5 MacBook Pro (10 cores, 24 GB), macOS 26.5,
Swift 6.3.3, with a clean release build of base commit
`aae03d4c59dc8167f8c1031c864e05afe9cc8b90` and the optimized source tree
recorded as commit `7282abebd00c6ef312d2d40a3c4c9324644725c0`, using the same warm local
model/cache state:

| Prompt | Base | Batched kernels | Gain |
|---|---:|---:|---:|
| medium-review, 421 tok | 64.05 s | 56.52 s | 1.13x |
| long-synthesis, 2 785 tok | 580.81 s | 412.09 s | **1.41x** |

The long case saves 168.72 s (29.0%). Every command below exited 0. Complete
timing footers, in base/optimized order, were:

```text
[stop=maxTokens prefill=421tok/64.05s new=1tok decode=0.00s tok/s=371.342]
[stop=maxTokens prefill=421tok/56.52s new=1tok decode=0.00s tok/s=669.375]
[stop=maxTokens prefill=2785tok/580.81s new=1tok decode=0.01s tok/s=100.969]
[stop=maxTokens prefill=2785tok/412.09s new=1tok decode=0.00s tok/s=303.210]
```

The exact medium commands were:

```bash
# aae03d4c59dc8167f8c1031c864e05afe9cc8b90; exit 0
benchmark-results/inkling-prefill-baseline/bin-base/MferenceCLI \
  --model scratch/inklingsmall.gturbo \
  --messages-file docs/benchmark-prompts/real-generation-v1/medium-review.json \
  --max-new 1 --max-context 4096 --temperature 0 \
  --top-k 64 --top-p 0.95 --seed 20260722 \
  --verify trusted-receipt

# 7282abebd00c6ef312d2d40a3c4c9324644725c0; exit 0
.build/release/MferenceCLI \
  --model scratch/inklingsmall.gturbo \
  --messages-file docs/benchmark-prompts/real-generation-v1/medium-review.json \
  --max-new 1 --max-context 4096 --temperature 0 \
  --top-k 64 --top-p 0.95 --seed 20260722 \
  --verify trusted-receipt
```

The exact long commands used the same two binaries and options, replacing the
messages file and seed as follows:

```bash
--messages-file docs/benchmark-prompts/real-generation-v1/long-synthesis.json
--seed 20260723
```

This is an isolated-prefill measurement, not the public community generation
protocol: it uses one new token, trusted-receipt verification, one measured run
per binary, and the production 128-token chunk default. Treat it as a local A/B
measurement rather than a performance ceiling.

### Native top-6 decode (2026-08-06)

Inkling now has a dedicated top-6 INT4 MoE decode path. The old implementation
padded six selected experts to the shared top-8 contract: duplicate buffers
cost no extra SSD reads, but phase 1 still evaluated eight gate/up rows and
phase 2 still evaluated eight down projections. `moe_phase2_down_reduce_k6`
launches six SIMD groups per output row and reduces the six routed values in
router order. Phase 1 specializes its existing function-constant top-k to six.

The runtime also plans the six cache slots before reading. On a mixed hit/miss
layer, it submits resident phase 1 immediately, reads only the misses, and then
submits the missing subset before the native top-6 reduction. The shared sinks
and FP32 residual path retain their original ordering and arithmetic.

Release A/B at base `6bf428f` and optimized commit `485df08` followed the full
[community protocol](COMMUNITY_BENCHMARKS.md) on a 24 GB M5 (`Mac17,2`),
macOS 26.5, and Swift 6.3.3. Runs used strict verification, production defaults,
one discarded warmup, and one measured fresh process per case. Every command
exited 0 and ended naturally.

| Case | Prompt / generated | Base | Native top-6 | Gain |
|---|---|---:|---:|---:|
| short-explanation | 59 / 469 | 2.909 tok/s | 3.434 tok/s | 18.0% |
| medium-review | 421 / 576 | 2.961 tok/s | 3.670 tok/s | 23.9% |
| long-synthesis | 2,785 / 372 | 2.819 tok/s | 3.038 tok/s | 7.8% |

The geometric-mean gain is **16.4%**, and every optimized stdout file is
byte-identical to its base counterpart. Full timing footers and the generic
reproduction command are recorded in
[BENCHMARKS.md](BENCHMARKS.md#inkling-small-276b-a12b-measured-decode).

A 24-slot short warmup was rejected: it fell to 0.710 tok/s and pushed
free-memory pressure to 24%, while the optimized 16-slot warmup reached 3.374
tok/s with identical output. Since 24 slots already crossed the useful memory
envelope on this host, 32 was not run; Inkling remains on the 16-slot automatic
default.

Bring-up config note: the fused QKV GEMV and the constant-folded INT4 GEMV
variants are bypassed for this family (plain generic GEMVs) — they were
briefly suspected during the corruption hunt and are unproven at this
family's shapes (`m == n == 4096`; fused (4096, 1024, 4096)). Re-enable one
at a time against known-good output.

Scope note: the text path is the target. The vision and audio towers ship in
the checkpoint and are excluded by the planner. The MTP config block is
present but the checkpoint carries **no** MTP tensors, so there is nothing to
exclude there.
