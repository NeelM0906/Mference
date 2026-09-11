# GLM-5.3-Flash on Mference

Checkpoint selection, architecture contract, memory budget, and measured
results for running
[`pipenetwork/GLM-5.3-Flash-MLX-mixed-4_8bit`](https://huggingface.co/pipenetwork/GLM-5.3-Flash-MLX-mixed-4_8bit)
with SSD-streamed (or, on a 256 GB host, fully page-cached) experts. This
document is the architecture contract for the `glm53Flash` family.

| | |
|---|---|
| Family identifier | `glm53Flash` (`ModelFamily.glm53Flash`), install label `glm53flash` |
| Source repository | `pipenetwork/GLM-5.3-Flash-MLX-mixed-4_8bit` |
| Pinned revision | `d43ea8b407ce4e9c25e6ac9baec3feab70d9f5f3` (index SHA-256 `5e0a3768…314383`) |
| Parameters | 320B total, 18B active (vendor figures) |
| Download / install size | 181,944,533,258 bytes downloaded; ~181 GB on disk (computed from the shard headers, not yet measured on a produced install) |
| Status | **first light green 2026-09-11** — runner, tokenizer and resident-expert path landed; the capability gate is lifted; perf pass and the FAMILY_GATE protocol run in progress |

## Checkpoint selection

GLM-5.3-Flash was published by Z.ai (`zai-org`) under the MIT license on
2026-08-25 as the first natively multimodal GLM-5 model: 45 text layers, a
hybrid of 34 Kimi-Delta-Attention linear layers and 11 DeepSeek-sparse-attention
layers, manifold-constrained hyper-connections, 288-expert top-8 MoE with one
shared expert after 3 dense layers, and a 24-layer vision tower. The vendor
ships FP8 (block-scaled) and BF16 uploads; some forty community conversions
existed within two weeks. Weights are redistributed by the conversion author,
not by this project.

**License note.** MIT on the vendor checkpoint and on PipeNetwork's conversion.
No attribution or use restriction affects redistribution or benchmarking.
PipeNetwork's parity-fixed MLX runtime
([`PipeNetwork/glm53-flash-mlx`](https://github.com/PipeNetwork/glm53-flash-mlx),
MIT) is the golden reference the toy parity gates are cut from; no code from
it is imported. See [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md).

### Rejected candidates

| Candidate | Routed experts | Attention / core | Disk | Verdict |
|---|---|---|---|---|
| `pipenetwork/GLM-5.3-Flash-MLX-mixed-4_8bit` | affine INT4 g64 | affine INT8 g64; router BF16; mHC / KDA decay fp32 | 181.9 GB | **Selected**: the precision policy the runtime already reads (INT4 experts, INT8 core, MLX affine g64), fp32 kept where the reference keeps it, perplexity +3.2 % vs the author's 8-bit anchor, and the same author's parity-fixed runtime to cut goldens from. No MTP layer. |
| `pipenetwork/GLM-5.3-Flash-MLX-4bit` | affine INT4 g64 | affine INT4 g64 | 177.6 GB | rejected: saves 4 GB for +8.5 % perplexity. |
| `pipenetwork/GLM-5.3-Flash-MLX-6bit` | affine 6-bit g64 | affine 6-bit g64 | 255.9 GB | rejected: no 6-bit kernels, and it does not fit beside the process on a 256 GB host. |
| `pipenetwork/GLM-5.3-Flash-MLX-8bit` | affine INT8 g64 | affine INT8 g64 | 334.1 GB | rejected: exceeds 256 GB. |
| `orcarouter/GLM-5.3-Flash-MLX` (OrcaSAQ 2/3/4/6-bit) | mixed 2–8 bit, g32/64/128 | BF16 for every non-FP8 tensor (34 KDA layers, indexer, mHC) | 102–296 GB | rejected: 5- and 6-bit widths and group 32 are unsupported; BF16 KDA layers inflate the resident core; the 2-bit builds lose 57–141 % perplexity. |
| `Vontra/GLM-5.3-Flash-MLX-4bit-MTP` | affine INT4 g64 | affine INT4 g64; BF16 embed / head; MTP layer included | 181.7 GB | deferred: no quality numbers against an 8-bit anchor; the included MTP layer is the natural source for a later speculative-decoding step. |
| `zai-org/GLM-5.3-Flash-BF16` (quantize in flight) | INT4 g64 | INT8 g64 | 643 GB download | rejected for now: the existing quantize-in-flight path would reproduce the selected policy at three times the download. |
| `zai-org/GLM-5.3-Flash` (FP8 e4m3, 128×128 block scales) | — | — | 328 GB | rejected: the installer has no FP8 dequantization path. |

Verified against revision `d43ea8b4` by reading all 18 safetensors headers
directly (2,998 tensors): every `language_model.` projection is U32-packed
affine with BF16 `scales` / `biases` companions at group 64 — INT4 (`x8`
packing) for the 42 × 3 stacked `mlp.switch_mlp.*` expert triplets, INT8
(`x4` packing) for everything else the `quantization` map names (embedding,
head, KDA and sparse projections, indexer projections, shared and dense
FFNs); `mlp.gate.weight` is BF16 `[288, 4096]` with an FP32
`e_score_correction_bias`; `attn_hc.fn` / `ffn_hc.fn` are BF16 `[24, 16384]`
with FP32 `base` `[24]` and `scale` `[3]`; `forget_gate.A_log` `[64]` and
`dt_bias` `[8192]` are FP32; `conv1d.weight` is BF16 `[24576, 4, 1]`; the
indexer's `k_norm` carries a bias. The `vision_model.*` tensors (24 blocks,
BF16) are excluded at install. `text_config` carries the vendor block
verbatim.

## Architecture contract

| Axis | Value | Covered by existing kernel? | Gap |
|---|---|---|---|
| `fullAttentionLayerMask` | `7` (KDA) ×34, `8` (NoPE latent sparse) on layers 3, 7, …, 43 | no | two new layer kinds; see the three axes below |
| `linearAttention` | 64 heads × 128, conv 4 | partial | GDN kernels exist for the scalar-decay recurrence; KDA needs a **per-channel** decay, sigmoid-gated low-rank output gate and l2-normalized q/k — new decode and prefill kernels (`kimiDeltaAttention`) |
| `compressedAttention.qLoraRank` / `indexNHeads` / `indexHeadDim` / `indexTopK` | 1536 / 32 / 128 / 2048 | partial | low-rank query and indexer scoring exist (DSV4, Flash-Next); the pooled keys and tail rule are new (`pooledLightningIndexer`) |
| `glm53.kvLoraRank` / `qkNopeHeadDim` / `vHeadDim` | 512 / 256 / 256 | partial | DSV4's shared-KV decode kernel attends one latent as K = V; needs the per-head `embed_q` fold / `unembed_out` unfold and no window / sinks (`nopeLatentSparseAttention`) |
| `glm53.indexKPool` / `indexKPoolAlwaysSelectTail` | 4 / true | no | pooled-key construction (per-channel softmax over gate + ape) and the tail rule |
| `hyperConnections` | mult 4, Sinkhorn 20, eps 1e-6 | yes | DSV4's `dsv4_hc_*` kernels with the same site's `pre`; `fn` is BF16 here (kernel reads fp32) and the final collapse is a stream mean |
| `numDenseLayers` / `denseIntermediateSize` | 3 / 12 288 | yes | INT8 GEMVs |
| `routerScoringFunc` / `routedScalingFactor` / `routerGateBias` / `routerNormAfterTopK` | sigmoid / 2.5 / true / true | partial | the selection math is Inkling's shape; an fp32 BF16-gate GEMV and a CPU or GPU top-8 |
| `swigluLimit` | 10.0 | partial | `dsv4_swiglu_clamp_mul` covers the shared and dense FFNs; the INT4 routed-expert kernel needs the clamp inside its phase-1 body |
| `ropeTheta` / `fullRopeTheta` / `partialRotaryFactor` | 0 / 0 / 0 | yes | nothing rotates |
| `qkNorm` | false | yes | low-rank norms are plain RMSNorms |
| `glm53.rmsNormEps` / `indexerKNormEps` | 1e-5 / 1e-6 | yes | RMSNorm takes eps; the indexer key path needs a LayerNorm with bias |

Axes left at their defaults are not listed.

| Quirk | Where it lives | Why it is not an axis |
|---|---|---|
| Final collapse is the stream **mean**, not a learned `hc_head` | runner, after the last layer | one value per family; DSV4 is the only other mHC family and has the head |
| `attn_hc.fn` / `ffn_hc.fn` stored BF16 | `Model+Glm53.swift`, hc weights kernel | storage width of one tensor group in one conversion |
| Vision tower dropped, `<|image|>` spans unsupported | planner exclusion, tokenizer | text-only port; recorded as the `vision` sidecar decision |
| `q_a_layernorm` / `kv_a_layernorm` eps 1e-5, indexer `k_norm` eps 1e-6 | `Glm53Config` | carried as two eps axes |

## Memory budget

Computed from the pinned conversion's safetensors headers (2026-09-11); the
produced install will replace these with measured bytes.

| Group | Params | Bytes |
|---|---:|---:|
| `embed_tokens` + `lm_head` (INT8) | 1.27B | 1.35 GB |
| KDA attention, 34 layers (INT8 + BF16 conv + fp32 gates) | 4.68B | 4.98 GB |
| sparse attention + indexer, 11 layers (INT8 + BF16 pooling gate) | 1.37B | 1.47 GB |
| dense FFN, layers 0–2 (INT8) | 0.45B | 0.48 GB |
| shared experts, 42 layers (INT8) | 1.06B | 1.12 GB |
| routers (BF16) + correction biases + mHC (BF16 `fn`, fp32 `base`/`scale`) + norms | 0.09B | 0.18 GB |
| **Resident total** | **8.9B** | **≈ 9.6 GB** |
| routed experts, 42 × 288 (INT4 g64, 14,155,776 B each — already page-aligned) | 304B | 171.2 GB |
| **On disk** | | **≈ 181 GB** |

Per decoded token the routed path reads 8 × 14.16 MB × 42 ≈ **4.76 GB** of
expert bytes; the resident core adds ≈ 9.6 GB of reads. On a 256 GB host the
whole expert set fits in the page cache beside the process, so the decode
roofline is memory bandwidth, not the SSD.

### KV cache and recurrent state

| Context | Sparse-layer latent KV (11 × 512 FP16) | Indexer key cache (11 × 256 FP16) | KDA state (fixed) |
|---|---:|---:|---:|
| 32 K | 369 MB | 185 MB | 141 MB |
| 128 K | 1.48 GB | 0.74 GB | 141 MB |

The KDA state is 34 × (64 × 128 × 128 fp32 + 3 × 24,576 fp16 conv tail).

### Expert-cache ladder

| Slots | Resident expert bytes | Notes |
|---|---:|---|
| 16 | 9.5 GB | floor; `auto` default for non-Qwen families |
| 32 | 19.0 GB | |
| 96 | 57.1 GB | |
| 128 | 76.1 GB | |
| `resident` | 171.2 GB | the whole expert set; fits a 256 GB host with ~60 GB to spare |

## Port status

- [x] **Repack** — `SupportedModelSource.glm53Flash` pinned to `d43ea8b4`;
      the generic pre-quantized planner lays the family out
      (`Glm53RepackPlannerTests`); installed 2026-09-11 in 1 h 15 min
      (180.8 GB over 2,744 ranges), `--verify-install` verified 49 files
      (180,843,651,755 bytes) in 68 s.
- [x] **Toy parity (fp32 tier)** — goldens from PipeNetwork's parity-fixed
      MLX runtime (`Scripts/parity/README.md`, "glm53flash"); the fp32 oracle
      `Glm53ReferenceRunner` reproduces every layer at `1e-4` on the
      reference's inputs with every integer decision and every argmax exact,
      and free-running keeps every decision and 16/16 rollouts on both prompts
      (`Glm53ReferenceParityTests`); suites wired into `bringup-check.sh`'s
      stage 1 filter table (`Glm53`). The Metal tier follows with the runner.
- [x] **Runner** — `Glm53ForwardRunner` (per-token; KDA decode, pooled
      indexer with CPU selection, latent attention, INT4 experts with the
      swiglu clamp, mHC), dispatched by `ForwardRunnerFactory`; the Metal tier
      of toy parity is green (see "Toy parity"). First light on the real
      install is green (see "Measured results") and the gate is lifted:
      `familiesWithoutRunner` no longer names the family;
      `shippedFamiliesAreNotGated` covers it.
- [x] **Tokenizer** — `ChatDialect.glm5` (`Glm5ChatTemplate.swift`,
      `Glm5ToolCallParser`), detected by the `[gMASK]` special token; the
      three `generation_config` EOS ids (`<|endoftext|>`, `<|user|>`,
      `<|observation|>`) in `stopTokenIDs`, `<|user|>` as the end-of-turn
      bridge; 22 HF renders byte-identical (`Glm5TemplateTests`, fixtures from
      `Scripts/parity/glm5_make_template_fixtures.py`); the decoder starts in
      thought because the generation prompt opens `<think>`. See "Tokenizer
      dialect" below for the two recorded deviations.
- [x] **Ladder** — 16 / 32 / auto expert-cache slots produce byte-identical
      greedy output (`bringup-check.sh glm53flash` stage 3, 2026-09-11:
      7.0 / 7.5 / 28.3 tok/s; `auto` resolves to resident on this host).
- [x] **Gate** — [`FAMILY_GATE.md`](../FAMILY_GATE.md) steps 1–9: full suite
      three times consecutively (see "Gate record"), release build clean,
      `git diff --check` and the Markdown link check clean, pinned install
      with strict verification plus raw and chat CLI footers, protocol page,
      phases snapshot, no approximate default paths (the batched prefill is
      an FP16-tier execution of the same model, the per-token path stays one
      switch away), provenance recorded, no credentials or weights in fixtures.
- [x] **Protocol bench** — the three frozen `real-generation-v1` cases with one
      discarded warmup, 9/9 measured footers `stop=endOfTurn` at
      `MFERENCE_GLM5_REASONING_EFFORT=low` (stated deviation; see "Measured
      results"), run twice: per-token prefill and batched prefill.

### Gate record

Full suite, three consecutive passes on 2026-09-11 at the final commit
(`Scripts/test.sh`, each a fresh process): 1,184 tests in 206 suites passed
in 257 s, 248 s and 244 s, each with the one pre-existing known issue that
`main` already carried. Release build `swift build -c release` clean;
`git diff --check` and `Scripts/check_markdown_links.rb` clean.

## Port plan

1. **Day-0 contract** (this commit): `ModelFamily.glm53Flash`, `Glm53Config`,
   `ArchConfig.glm53Flash_320B_A18B`, manifest fields and validation, the
   capability gate, tensor accessors, the repacker's `glm5_next` loader and
   planner classification, the pinned source, tests and this page.
2. **Reference-parity goldens** (done 2026-09-11) from PipeNetwork's runtime
   at the toy geometry (`Scripts/parity/glm53_make_goldens.py`): per-token
   captures of the KDA recurrence, the pooled indexer's selections, the latent
   attention, the routed selections and the mHC coefficients, plus an fp32
   Swift oracle that reproduces them (see "Toy parity" below).
3. **Kernels**: KDA decode / prefill (ported from the MIT-licensed
   `metal/glm53_kda.metal` in `IngeniousIdiocy/ds4` at `90d71e0d`, with
   attribution), the pooled indexer, the latent attention with per-head fold /
   unfold, the INT4 expert swiglu clamp, a BF16-`fn` mHC weights variant, an
   fp32 BF16-gate router GEMV.
4. **Runner** `Glm53ForwardRunner`, dispatched by `ForwardRunnerFactory`,
   chunked prefill ≡ sequential decode, toy parity against the goldens.
5. **Tokenizer dialect** `glm5`: `[gMASK]<sop>`, `<|system|>` /
   `<|user|>` / `<|assistant|>` / `<|observation|>`, `Reasoning Effort:
   Low|High|Max`, `<think>` opening the assistant turn, `<tool_call>` with
   `<arg_key>` / `<arg_value>`, `clear_thinking`.
6. **Real model and gate.** First light, dense A/B at ≤ 2,048 context (below
   `index_topk` the selection is exhaustive), a needle beyond 2,048 with the
   pooled selection active, ladder smoke, `bringup-check.sh glm53flash`, the
   frozen protocol, one phases snapshot; then the gate lift.

## Tokenizer dialect

`glm5` is a hand-port of the checkpoint's `chat_template.jinja` (SHA-256 in
`Tests/Mference/Core/Tokenization/Fixtures/Glm5Tokenizer/renders.json`),
byte-matched to `transformers` 5.17 `apply_chat_template` on 22 fixtures that
cover the effort line, system / user / assistant / tool turns, historical
thinking with and without `clear_thinking`, tool schemas, tool calls with
typed arguments and result ordering by `tool_call_id`. The reasoning effort renders as
`Max` (the template's own default when the caller passes nothing) unless
`MFERENCE_GLM5_REASONING_EFFORT=low|high` is set for the process — the
template's own `reasoning_effort` kwarg, exposed the way the runtime exposes
its other knobs. Measured on the frozen short-explanation case: at `Max` the
model was still inside its think block after 2,400 tokens (it word-counts
its own draft), at `low` it closed the think block after 288 characters and
reached `stop=endOfTurn` at 668 tokens with a complete answer.
`generation_config.json`'s three EOS ids all stop generation.

Two deviations by construction, both shared with the MiniCPM5 and DeepSeek
dialects: `JSONValue` objects are unordered, so tool-schema keys and call
arguments render in sorted key order where Jinja keeps insertion order; and
`developer` guidance renders as a `<|system|>` turn (the server folds it there
for every non-Gemma dialect). Raw prompts carry no `[gMASK]<sop>` prefix — the
chat render supplies it, as ChatML supplies its own framing.

## Toy parity

Two tiers, both from the committed goldens (seed 12, toy geometry in
`Scripts/parity/README.md`):

| Tier | What is compared | Result |
|---|---|---|
| fp32 oracle, anchored | `Glm53ReferenceRunner` given the reference's own layer inputs and cache appends, every capture of both prompts (12 + 48 tokens) and 16 decode steps each | every value within `atol = rtol = 1e-4`; every indexer selection (dense bypass, pooled top-2, tail), router top-8 and argmax exact |
| fp32 oracle, free-running | the oracle carrying its own fp32 drift | decisions and 16/16 rollouts exact on both prompts; logits drift 1.8e-4 (short) / 2.5e-4 (long) against the reference's own batched-vs-per-token gap of 4.7e-5 / 3.55e-4 |
| Metal runner vs oracle (FP16 tier) | `Glm53ForwardRunner` on a planner-written install of the toy checkpoint, the oracle anchored to the runner's layer inputs and cache appends, both prompts and 16 decode steps each (`Glm53ForwardRunnerTests`) | no decision flips at any margin, greedy 17/17 on both prompts; worst abs 2.07e-2 (`post_attention_layernorm_out`, long decode), streams 1.58e-2, mHC coefficients 8.6e-3, logits 3.6e-3 — gate `atol = rtol = 5e-2`; chunked prefill == sequential decode bit for bit at three chunkings; dense A/B arm bit-equal below `index_topk` and refused above |

## First light (real install, 2026-09-11)

Host: Mac Studio, Apple M3 Ultra, 256 GB, macOS 26.3.1 (Darwin 25.3.0),
Swift 6.3.3; `scratch/glm53flash.gturbo` verified with strict SHA-256. The
env-gated `Glm53RealGenerationMeasurement` harness loads through the
explicit-baseline door with `MFERENCE_GLM53_RESIDENT=1` (the debug test
bundle; the release CLI figures follow).

| Probe | Result |
|---|---|
| `The capital of France is` (greedy, 24 new, twice) | ` Paris. It is the largest city in France and serves as the country's political, economic, and cultural center.` both passes, identical token ids |
| frozen short-explanation (greedy, 160 new, raw deltas) | coherent think-block reasoning about wetlands, mechanisms, limits |
| dense A/B (60-token prompt + 32 new, below `index_topk`) | indexer arm and dense arm identical, 32/32 tokens |
| load | 22 s for 181 GB (copied resident experts at 6.4 GB/s), device working set 239 GB recommended / 181 GB allocated, peak RSS 164 GB |

Two findings on the way there, both fixed before the numbers above:

1. **The mapped resident strategy does not work at this size.** With the
   expert set `mmap`'d and wrapped `bytesNoCopy`, the GPU faulted the 171 GB
   in page by page at ~0.7 GB/s (one token a minute) and ~140 GB of
   incompressible pages landed in the compressor. `ResidentExpertStreamer`
   gained a `.copied` strategy (one anonymous shared buffer per layer filled
   by `F_NOCACHE` direct reads) and the runner pins every layer buffer in an
   `MTLResidencySet`; without the residency set the same buffers still cost
   2.7 s of GPU time per token.
2. **`dsv4_hc_weights` was 52 ms of an 86 ms token.** The production-shape
   kernel benchmark (`Glm53KernelBenchmarks`, `MFERENCE_GLM53_KBENCH=1`) put
   every GEMV and the resident expert FFN at 400-750 GB/s and the mHC weights
   kernel at 2.9 GB/s (0.58 ms per site, one threadgroup walking 24 x 16,384
   fp32). `glm53_hc_dots` / `glm53_hc_finalize` bring a site to 0.10 ms
   including collapse, norm and place-mix; the router select moved to one
   simdgroup (0.068 -> 0.021 ms). Decode went from 11 to 27 tok/s.

Release CLI after both fixes (commit `7e0821d`, `MFERENCE_PHASES=1`):

| Run | Footer |
|---|---|
| raw `The capital of France is`, greedy, 48 new | `prefill=5tok/0.89s new=48tok decode=1.74s tok/s=27.622` |
| chat short-explanation, greedy, 1,024 new (Max effort, still thinking at the cap) | `prefill=60tok/2.73s new=1024tok decode=40.90s tok/s=25.039` |
| chat short-explanation, greedy, low effort (harness) | `stop=endOfTurn prefill=60tok/2.71s new=668tok decode=26.29s tok/s=25.412` |

Prefill at that commit was the per-token path (decode speed, ~22 tok/s);
the batched prefill below replaced it.

## Batched prefill

`Glm53PrefillEngine` walks a chunk of up to 256 prompt tokens layer by layer
with every weight read once per chunk: INT8 projections on the simdgroup
matrix units (`glm53p_int8_gemm_mma`, fp32 operands so the dequantized
weight is the GEMV's, one 64-column group per barrier), BF16 GEMMs for the
router and pooling gate, the mHC maps per token, GDN's batched conv with the
tail update, a KDA chunk kernel that carries the head's state in registers
across the tokens (bit-identical to the same number of decode steps), the
indexer's keys / gates / pooled keys for the chunk, and the routed experts
grouped by expert over the chunk (one router readback per MoE layer; the
resident slab addressed by expert index; measured 511 GB/s on the real
~2-routes-per-expert distribution). Queries at or below `index_topk` attend
densely and causally; later queries score every visible pool on the GPU and
the CPU picks each query's set with the decode path's rule (one readback per
sparse layer). `MFERENCE_GLM53_BATCHED_PREFILL=0` keeps the per-token path,
`MFERENCE_GLM53_PREFILL_CHUNK=n` caps the chunk, `MFERENCE_GLM53_GEMM=scalar`
swaps the scalar GEMM in, `MFERENCE_GLM53_PREFILL_TRACE=1` prints a line per
chunk.

Numerics: on the toy, the whole 48-token prompt (44 queries past
`index_topk` 4) lands within 2e-3 of the per-token path with 6/6 greedy
steps agreeing; on the real install every A/B so far agreed on all 32 greedy
tokens and the next-token argmax (short-explanation, medium-review,
long-synthesis), with prompt-logit deltas up to ~1.0 on top logits of ~19.
A one-token-chunk run reproduces the 128-token chunk's logits exactly, so
that gap is accumulation order (the GEMM sums in a different order than the
GEMV and the batched RMSNorm kernel differs from the decode one), not chunk
size. An fp16-operand GEMM variant was tried and dropped: its weight rounding
moved logits by 1.4 and flipped a 0.6-margin next token.

| Prompt | Per-token prefill | Batched prefill |
|---|---:|---:|
| short-explanation, 60 tokens | 2.03 s (29.6 tok/s) | 0.67 s (89 tok/s, one 60-token chunk incl. first-use setup) |
| medium-review, 420 tokens | 14.5 s (28.9 tok/s) | 3.48 s (121 tok/s) |
| long-synthesis, 2,792 tokens | 116.6 s (23.9 tok/s, release CLI) | 33.6 s (83 tok/s, release CLI) |

Per-chunk trace on long-synthesis (release CLI, `MFERENCE_GLM53_PREFILL_TRACE=1`):
chunks of 256 run 2.7 s at position 0 rising to 3.7 s at position 2,048 and
beyond — the latent attention's cost grows with the visible rows (dense) or
the 2,048 + tail selected rows, and the selection chunks cost the same as the
last dense ones. The remaining prefill levers, in order: a multi-query latent
attention kernel (queries of a chunk share the key rows they load; today each
query re-reads them), the INT8 GEMM (about 4x above its instruction floor)
and the grouped expert kernels (weight-bound at ~500 GB/s, amortized by chunk
size).

## Measured results

Host: Mac Studio, Apple M3 Ultra, 256 GB, macOS 26.3.1 (Darwin 25.3.0),
Swift 6.3.3. Protocol: [`COMMUNITY_BENCHMARKS.md`](../COMMUNITY_BENCHMARKS.md),
3 measured repetitions per case after one discarded warmup, each run a fresh
process. **Stated deviation:** `MFERENCE_GLM5_REASONING_EFFORT=low`. At the
template's default (`Max`) the model does not leave its think block inside
the 1,024-token cap on any case (2,400 tokens of thinking measured on
short-explanation), so no footer could read `stop=endOfTurn`; `low` is the
vendor template's own setting, not a sampling change.

Run 2026-09-11, release `MferenceCLI` at commit `166f6a2` (batched prefill),
`auto` expert mode (resident),
`./run-benchmark.sh glm53flash-batched scratch/glm53flash.gturbo 3` with
`MFERENCE_GLM5_REASONING_EFFORT=low`; `stop=endOfTurn` on 9/9 measured runs.

| Case | Prompt / generated | Prefill | Decode | Range | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: |
| short-explanation | 60 / 653 | 1.25 s (48 tok/s) | 25.79 tok/s | 25.73 – 25.79 | 160.0 GiB |
| medium-review | 420 / 841 | 4.16 s (101 tok/s) | 24.30 tok/s | 24.30 – 24.30 | 160.0 GiB |
| long-synthesis | 2,792 / 709 | 33.28 s (84 tok/s) | 19.16 tok/s | 19.13 – 19.17 | 160.0 GiB |

The earlier run of the same protocol at commit `011a016` (per-token prefill)
measured prefill 2.77 / 15.29 / 116.64 s and decode 25.61 / 24.09 / 19.09
tok/s (686 / 801 / 712 generated); the generated lengths differ because the
two prefills are FP16-tier different executions and sampling at temperature
0.2 takes a different path. The 60-token prefill includes the batched
engine's first-use setup (scratch and pipelines) in a fresh process.

The long-synthesis prompt crosses `index_topk`, so its decode runs the
pooled indexer's CPU selection on the 11 sparse layers every token (one
readback each) and the latent attention over ~3k rows; that is the 19 vs
25 tok/s gap.

Against the targets set for the port (decode 25–35 tok/s after the perf
pass, prefill 150–400 tok/s): decode is inside the band; prefill sits at
roughly half of the lower bound on prompts past a few hundred tokens, with
the three levers listed under "Batched prefill".

### Phases attribution

Release CLI, chat short-explanation, greedy, 1,024 generated tokens, resident
experts (`scratch/glm53-cli-smoke-2.log`, commit `7e0821d`):

| Phase | ms/token | Share |
|---|---:|---:|
| GPU busy (one command buffer per token) | 37.6 | 94% |
| GPU span minus busy (CPU encode, sampler, detokenizer between tokens) | 2.3 | 6% |
| expert I/O | 0 | resident: none |
| router readback wait | 0 | resident: the router never leaves the GPU |
| indexer CPU top-k | 0 | prompt + generation stayed below `index_topk` |

Per-kernel attribution at production shape (`Glm53KernelBenchmarks`): KDA
layer projections 0.23 ms (609 GB/s), routed experts 0.22 ms (508 GB/s),
mHC site 0.10 ms, lm_head 0.86 ms (740 GB/s), sparse `o_proj` 0.33 ms
(413 GB/s), latent attention over 2,048 rows 0.95 ms (141 GB/s, the next
kernel to widen). About 700 dispatches per token; ~11.5 GB of weights per
token puts the memory-bandwidth floor near 15 ms.

| Phase | ms/token | Share |
|---|---:|---:|
| | | |

## Known limits

- **Prefill numerics are FP16-tier, not bit-identical to decode.** The
  batched prefill and the per-token path are two valid FP16 executions of the
  model that differ in accumulation order (see "Batched prefill"); greedy
  continuations can differ at near-ties. The per-token path remains
  available (`MFERENCE_GLM53_BATCHED_PREFILL=0`).
- **256 GB-class hosts.** The family's runner is built for the whole expert
  set resident in memory (`auto` picks `.resident` when pool + core + 20 GiB
  fit physical memory); the slot-cache mode works on smaller hosts but reads
  ~4.75 GB of experts per token through `pread` and is not what the measured
  figures describe.
- **Text-only.** The vision tower, image spans and video tokens are excluded;
  the chat template's image / video / audio markers are not rendered.
- **No MTP.** The pinned conversion omits the multi-token-prediction layer;
  speculative decoding from it is deferred.
- **Reasoning effort** defaults to `max` as the vendor template does; `low`
  and `high` are opt-in.
- Untested on the real model: everything under "Measured results".

## Reproduction

```bash
swift build -c release
swift run -c release MferenceRepack --dry-run --model glm53flash --output scratch/glm53flash.gturbo
swift run -c release MferenceRepack --model glm53flash --output scratch/glm53flash.gturbo
./bringup-check.sh glm53flash scratch/glm53flash.gturbo
./run-benchmark.sh glm53flash scratch/glm53flash.gturbo 3
```

Run one model process at a time, and re-read the preconditions in
[`AGENTS.md`](../../AGENTS.md) before any model run.
