# MiniCPM5-2B on Mference — bring-up dossier

Checkpoint selection, architecture contract, memory budget, and measured
results for running
[`openbmb/MiniCPM5-2B`](https://huggingface.co/openbmb/MiniCPM5-2B) as the
`minicpm5` family. This is the project's first plain-llama dense family. It
does not exercise the SSD expert-streaming thesis; its purpose is to prove the
bring-up kit on a llama-architecture checkpoint installed from the vendor's
**BF16** upload through the repacker's quantize-in-flight path, so that the
install inherits the W2.1b quantizer quality gate
([QUANTIZER_QUALITY.md](../QUANTIZER_QUALITY.md)). Nothing below is
implemented unless the "Port status" list says so.

| | |
|---|---|
| Family identifier | `minicpm5` (`ModelFamily.minicpm5`) |
| Source repository | `openbmb/MiniCPM5-2B` (BF16, quantized at install) |
| Pinned revision | `cd199ce3ee67549c42ef7372f809f2c63599a3e9` |
| Pinned index SHA-256 | `6d839cd76e8395de548a0e6cc310386f66d1ecbb2c75d198a8dfd3d70892b756` (31,378 bytes) |
| Control repository | `openbmb/MiniCPM5-2B-MLX` (vendor INT4 g64 affine, pre-quantized path) |
| Control revision | `35ac38ee7bdb0bf7fa748d0700eeb6d6675760a3` |
| Control index SHA-256 | `ccf202e0a06fe3c7eb8f354cfb29412a5e64956ad895413d4d9267ae4b3a6045` (68,721 bytes) |
| Parameters | 2.517B total, 2.517B active (dense); 1.982B non-embedding |
| Install size | 1,425,981,879 bytes verified (8 files); `model_weights.bin` 1,415,974,912 |
| Status | **supported** — family gate green 2026-09-10 (`bringup-check.sh` PASS, full suite ×3, protocol run with the stated deviation) |

## Checkpoint selection

MiniCPM5-2B was published by OpenBMB in September 2026 under Apache-2.0
(`license: apache-2.0` in the model card front matter; "This repository and
MiniCPM model weights are released under the Apache-2.0 License" in the README).
The vendor ships two conversions in its own namespace: the BF16 original and an
MLX INT4 conversion. Weights are redistributed by the vendor, not by this
project.

**License note.** Apache-2.0 — permits use, modification, and redistribution
with attribution and a copy of the license; no use restriction affects
benchmarking or redistribution of a repacked install. No reference code is
imported by this port (the architecture is read from `transformers` v5.6.2's
`modeling_llama.py`, which is itself Apache-2.0, and nothing is copied from
it), so [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md) gains only the
weights entry.

### Rejected candidates

| Candidate | Linear weights | Embeddings / head | Disk | Verdict |
|---|---|---|---|---|
| `openbmb/MiniCPM5-2B` (BF16) | BF16, quantized in flight to INT4 g64 | same | 5.03 GB source, ~1.43 GB installed | **Selected** — the vendor's own upload; the install inherits W2.1b |
| `openbmb/MiniCPM5-2B-MLX` | INT4 g64 affine (`quantization {group_size 64, bits 4, mode affine}`) | INT4 g64 affine | 1.42 GB | **Selected as the W2.1b control only** (`minicpm5mlx`), not as the shipped source: the point of this bring-up is the quantize-in-flight path |
| `openbmb/MiniCPM5-2B-DSpark` | — | — | — | Not a base checkpoint: the vendor's block-diffusion drafter (5-layer qwen3-arch, block_size 7). Follow-on project, see "Known limits" |

Verified against revision `cd199ce3` by reading the single shard's safetensors
header directly (44,128-byte header, 381 tensors, data 5,033,512,960 bytes,
file 5,033,557,096 bytes — the index's `total_size` and the CDN's
`Content-Length` agree): every rank-2 tensor is BF16, every norm is a BF16
`[2048]` vector, and there are no bias tensors. The control's header
(973 tensors, `__metadata__.format = mlx`) carries U32 `[rows, cols/8]` weights
with BF16 `[rows, cols/64]` scales and biases for every projection **including
`embed_tokens` and `lm_head`**, and BF16 norms; its `chat_template.jinja` and
`tokenizer.json` are byte-identical to the BF16 repo's (`cmp`, 2026-09-10).
Full per-tensor map: [`minicpm5.tensors.json`](minicpm5.tensors.json).

## Architecture contract

Read from `config.json` @ `cd199ce3` (`LlamaForCausalLM`, `model_type
"llama"`, `transformers_version 5.6.2`) and verified against
`transformers` **v5.6.2** `src/transformers/models/llama/modeling_llama.py`
and `configuration_llama.py` (fetched from the `v5.6.2` tag, 2026-09-10):

- `LlamaRMSNorm.forward`: `weight * (x.float() * rsqrt(mean(x²) + eps)).to(input_dtype)`.
  **Plain `w · x̂`, no `(1 + w)` fold.** This is the convention the W2.1b
  history warns about (§7b of QUANTIZER_QUALITY.md); for this family the fold
  must be *off*, and `FlashNextPlanner.foldsNormBias` returns `false`.
- `LlamaRotaryEmbedding.compute_default_rope_parameters`: `dim = head_dim`,
  `inv_freq = 1 / theta^(arange(0, dim, 2) / dim)`; `apply_rotary_pos_emb`
  uses `rotate_half` (NeoX half rotation over the whole head). `rope_scaling`
  is `null`, so `rope_type` is `default` and `attention_scaling` is 1.0.
- `LlamaAttention`: `scaling = head_dim ** -0.5`; `attention_bias` and
  `mlp_bias` default `false` (no bias tensors in the header); **no
  `q_norm` / `k_norm`**; no output gate.
- `LlamaMLP`: `down(silu(gate(x)) * up(x))`.
- `LlamaDecoderLayer`: plain pre-norm residual (`input_layernorm` → attn →
  add; `post_attention_layernorm` → MLP → add). `LlamaModel` applies no
  embedding scale; `config.json` carries none of the older MiniCPM
  `scale_emb` / `scale_depth` / `dim_model_base` fields.

Every value below is set away from the Gemma 4 default of the corresponding
`ArchConfig` axis ([FAMILY_CONTRACT.md](../FAMILY_CONTRACT.md)).

| Axis | Value | Covered by existing kernel? | Gap |
|---|---|---|---|
| `family` | `minicpm5` | — | new case at every exhaustive `switch` |
| `hiddenSize` | 2048 | yes | — |
| `intermediateSize` / `denseIntermediateSize` | 6144 | yes (`SharedExpertRuntime`, `PrefillSharedExpert`, INT4 QMM) | — |
| `moeIntermediateSize` | 0 | yes | — |
| `numLayers` / `numDenseLayers` | 42 | yes | — |
| `numHeads` | 16 | yes | — |
| `numKVHeads` / `numFullKVHeads` | 2 (GQA 8:1) | yes (`Attention.encodeFull`, `PrefillAttention`, paged kernels take arbitrary head counts) | — |
| `headDim` / `fullHeadDim` | 128 | yes (Maple and Inkling run 128) | — |
| `vocabSize` | 130,560 | yes | — |
| `slidingWindow` | 0 | yes | — |
| `finalLogitSoftcap` | 0.0 | yes | — |
| `ropeTheta` / `fullRopeTheta` | 5,000,000.0 | yes | — |
| `partialRotaryFactor` | 1.0 (`rotaryDim == headDim == 128`) | yes (`rope_neox_subdim`, `prefill_rope_neox_subdim_block`) | — |
| `ropeNeoxSubdim` | `true` | yes | — |
| `numExperts` / `topKExperts` / `numSharedExperts` | 0 / 0 / 0 | yes (Qwen 3.8 already runs zero experts) | — |
| `tieWordEmbeddings` | `false` | yes | — |
| `attentionKEqV` | `false` | yes | — |
| `fullAttentionLayerMask` | `1` on all 42 layers | yes | — |
| `hiddenActivation` | `"silu"` | yes | — |
| `attnOutputGate` | `false` | yes | — |
| `attentionScale` | 0.08838834764831845 (128^-0.5) | yes | — |
| `embeddingScaledBySqrtHidden` / `routerScaled` / `ffnSandwichNorms` / `sharedExpertGated` | `false` | yes | — |
| `linearAttention` | `.none` | yes | — |
| **`qkNorm`** (new axis) | **`false`** | **partial** | `Model.qNorm`/`kNorm` are unconditional and both fused QKV epilogues (`FusedQKVEpilogue`, `PrefillQKVEpilogue`) require the weights. Gap closed by configuration, not kernels: a `qkNorm: Bool` axis (default `true`, so every shipped manifest keeps validating) and a runner branch that skips the epilogue and calls the standalone `RoPE` / `PrefillRoPE` kernels. |

Axes left at their defaults are not listed. Anything the runner branches on
that is *not* an axis is a family quirk:

| Quirk | Where it lives | Why it is not an axis |
|---|---|---|
| `rms_norm_eps` = 1e-6 | `config.json`; runners hard-code `epsilon = 1e-6` | The value coincides with the runtime's compile-time epsilon. Rather than add an axis nobody else's runner reads, `ArchInfo.loadMiniCPM5` refuses a config whose `rms_norm_eps` is not 1e-6, so the coincidence is enforced at install rather than assumed. |
| Two EOS ids (`</s>` = 1, `<|im_end|>` = 130073) | tokenizer (`stopTokenIDs`) | A tokenizer fact: `config.json` and `generation_config.json` both list `eos_token_id [1, 130073]`. Both terminate generation. |
| BOS from the template, not the tokenizer | `chat_template.jinja` opens with `{{- bos_token }}`; `tokenizer_config.json` has `add_bos_token false` | The rendered chat prompt starts with the literal `<s>` and `encode` must not prepend a second BOS. Raw prompts get no BOS unless the caller asks (`bosPrefixID` is `<s>` = 0). |
| `<think>` / `</think>` are added tokens with `special: false` | tokenizer.json | They still tokenize as single ids (8, 9) but survive `skip_special_tokens`; the decoder keys on the ids, not on the flag. |
| `<tool_sep>` is plain text | `chat_template.jinja` | The template splits assistant content on the literal string; there is no token for it. |

## Tokenizer and chat template

`tokenizer.json` is HF-tokenizers byte-level BPE (130,072 BPE ids, 510 added
tokens up to id 130,559, no normalizer, digit-split + GPT-4-style regex
pre-tokenizer, `TemplateProcessing` post-processor that prepends `<s>` only when
`add_special_tokens` is on). Ids the runtime keys on:

| Token | id | special |
|---|---:|---|
| `<s>` (bos) | 0 | yes |
| `</s>` (eos, pad) | 1 | yes |
| `<|im_start|>` | 130,072 | yes |
| `<|im_end|>` (eos #2, end of turn) | 130,073 | yes |
| `<think>` / `</think>` | 8 / 9 | **no** |
| `<function` / `</function>` / `<param` | 18 / 19 / 20 | yes |
| `<tool_response>` / `</tool_response>` | 10 / 11 | yes |
| `<tools>` / `</tools>` | 12 / 13 | yes |
| `<tool_call>` / `</tool_call>` | 2 / 3 | yes (present, **unused by the template**) |

`chat_template.jinja` (9,060 bytes, SHA-256 `cc945752…`): ChatML framing,
`bos_token` first. Tool definitions are `tojson`-rendered inside
`<tools>…</tools>` in the system turn (or replace a `<tool_def_sep>` marker
inside a user-supplied system message). Assistant turns without a think block
are re-rendered with an empty `<think>\n\n</think>\n\n`; a `reasoning_content`
field or an inline `<think>…</think>` becomes `<think>\n…\n</think>\n\n`.
Historical tool calls render as `<function name="…"><param name="…">value</param></function>`,
CDATA-wrapped when a string value contains `<`, `&` or a newline, interleaved
with the text at `<tool_sep>` boundaries; tool results are wrapped in
`<tool_response>…</tool_response>` inside a `user` turn (consecutive results
share one turn). The generation prompt is `<|im_start|>assistant\n` followed by
`<think>\n` when `enable_thinking` is true, `<think>\n\n</think>\n\n` when it
is false, and nothing when it is undefined.

**Decision for Mference:** `enable_thinking = true` is the family default,
because it is the vendor's canonical usage (the README's only
`apply_chat_template` example passes `enable_thinking=True`) and matches the
convention the Qwen 3.8 and Maple families already use. The dialect exposes
the other two branches for the render fixtures. Two consequences are recorded
rather than hidden: the decoder starts in thought (as for `qwen38`), and the
community-protocol runs may not reach `stop=endOfTurn` inside 1,024 tokens
(see "Measured results").

One place the hand-ported render will differ from HF by construction:
`HistoricalToolCall.arguments` is an unordered `JSONValue.object`, so parameter
tags render in **sorted key order**, while Jinja preserves the request's
insertion order. The fixture set uses sorted keys on both sides; a request with
unsorted keys gets the same parameters in a different order, which the model
reads identically.

## Memory budget

Numbers from the shard header (params) and the INT4 g64 layout
(0.5 B/weight nibbles + one BF16 scale and one BF16 bias per 64-group =
0.5625 B/weight; norms BF16). They will be replaced by the produced install's
bytes once it exists.

| Group | Params | Bytes (INT4 g64 / BF16) |
|---|---:|---:|
| `embed_tokens` | 267,386,880 | 150,405,120 |
| `lm_head` (untied) | 267,386,880 | 150,405,120 |
| attention (42 layers) | 396,361,728 | 222,953,472 |
| dense MLP (42 layers) | 1,585,446,912 | 891,813,888 |
| norms (85 BF16 vectors) | 174,080 | 348,160 |
| **Resident total** | **2,516,756,480** | **1,415,925,760** |
| routed experts | 0 | 0 |
| **On disk** (measured, 2026-09-10) | | **1,425,981,879 bytes** |

The resident total equals the control's `model.safetensors.index.json`
`total_size` (1,415,925,760) to the byte: same tensor set, same format.

**Plan vs. actual (2026-09-10).** `MferenceRepack --dry-run --model minicpm5`
against the pinned repo: source bytes to read 5,033,512,960 (= the index's
`total_size`), output bytes 1,415,974,912 (the resident total plus a 49,152-byte
page-rounded index for 381 entries), 296 INT4 tensors + 85 unquantized norms, 0
bit-width overrides. The real install wrote `model_weights.bin` at exactly
1,415,974,912 bytes — a 0-byte delta from the plan — in 3 min 59 s on this host,
and `--verify-install` reported 8 files / 1,425,981,879 bytes (the difference is
the tokenizer sidecars, `layout.json`, manifest and receipt). The control
(`minicpm5mlx`, pre-quantized path) installed in 1 min 11 s and verified at
7 files / 1,425,887,156 bytes with an identical 1,415,974,912-byte resident
file.

### KV cache

FP16, 2 KV heads × 128 × 2 B × (K + V) = 1,024 B per token per layer,
43,008 B per token over 42 layers.

| Context | KV bytes |
|---|---:|
| 4 K | 176 MB |
| 32 K | 1.41 GB |
| 128 K (131,072 = `max_position_embeddings`) | 5.64 GB |

131k context is sane only through the paged path (`--kv-paged`), which this
family keeps from Qwen 3.8 (chunked prefill, Quest page selection, SSD spill).

### Expert-cache ladder

Dense: the expert streamer never opens a pool, so 16 / 32 / auto slots are
byte-identical by construction (the ladder smoke still runs for the record).

## Port plan and gap list

Ordered by risk, each with its own gate. The rule for this port: prefer
configuration over new kernels and reuse of the `qwen38` dense paths over
anything new. No Metal is expected to change.

1. **`qkNorm` axis** — `ArchConfig.qkNorm: Bool = true`, mirrored in
   `ArchInfo`, emitted by `GTurboJSON` for this family only, checked in
   `ManifestReader.validateArch` (absent → `true`), one row in
   FAMILY_CONTRACT.md. The runner (below) branches on it. Gate: the shipped
   families' toy suites and manifests are unchanged (their manifests do not
   carry the key and validate against the default).
2. **Tokenizer dialect `.minicpm`** — resolved when the tokenizer carries
   `<function` (18) *and* `<|im_end|>`, or when the family is passed; both EOS
   ids in `stopTokenIDs`; text render and tool render hand-ported from the
   Jinja; `MiniCPMToolCallParser` for the XML body with CDATA; decoder branch
   keyed on the `<function` / `</function>` special tokens; think spans keyed
   on ids 8 / 9. Gate: committed render fixtures byte-identical to
   `transformers` 5.6.2 `apply_chat_template` for system / user / multi-turn
   assistant with and without think / tools / tool call / tool response /
   `enable_thinking` true, false, undefined.
3. **Repacker** — `ArchInfo.loadMiniCPM5` (flat config, `model_type
   "llama"`, production cross-check against the pinned shape including
   `rms_norm_eps`), `SupportedModelSource.minicpm5` (BF16,
   `.originalRepoQuantize`) and `.minicpm5mlx` (control, pre-quantized),
   `IndexLoader` accepting a missing `quantization` block **only when the
   resolved source entry is an `originalRepoQuantize` entry** (no `"llama"` in
   the `model_type` allowlist), `RepackModelFamily.minicpm5` at every planner
   switch, `QuantBitPolicy.uniformInt4` (the control has no overrides),
   `foldsNormBias` false, identity resident naming. Gates: synthetic
   end-to-end install through the fake remote; dry-run byte total closes
   against the real repo; real install `--verify-install` green.
4. **W2.1b for this family** — weight level with a generalized
   `Scripts/quantizer-weight-gate.py --family minicpm5`; model level with
   `QuantizerQualityMeasurement` against `scratch/minicpm5-mlx.gturbo`.
   Thresholds as documented: noise floor exactly zero, top-1 ≥ 0.50, median
   KL ≤ 0.50 nats. Stamp `manifest.quantizedAtInstall.qualityGate`.
5. **Golden harness, then runner** — `Scripts/parity/minicpm5_make_goldens.py`
   (toy `LlamaForCausalLM`, pinned `transformers` 5.6.2, torch CPU fp32),
   goldens for two prompts, then `MiniCPM5ForwardRunner` derived from
   `Qwen38ForwardRunner` (no GDN, no gate, no q/k norm, standalone RoPE,
   chunked prefill and paged KV kept). Gates in order: per-layer parity, full
   forward parity, greedy rollouts token-exact, cached decode == recompute.
6. **Exhaustive switches** — `Model.swift` (7 sites), `ForwardRunnerFactory`,
   `ServerInference.defaultModelID`, `AppModelInstallDescriptor` (2),
   `RepackPlanner` (4), `FlashNextPlanner` (2), `QuantBitPolicy` (1). One new
   case each; nothing reordered.

While the runner does not exist, `minicpm5` sits in
`ManifestReader.familiesWithoutRunner` with the axis it is missing
(`qkNormFreeAttention`), exactly as Flash-Next did, so an install can exist and
validate without being loadable.

### Deviations from the task brief, recorded where they happened

- **Host.** The brief describes a Command-Line-Tools-only Mac with installs
  under `/Users/studio2/Documents/ChatGPT/Mference/scratch/`. This host is
  `/Users/zidane/Mference` with Xcode installed
  (`xcode-select -p` → `/Applications/Xcode.app/Contents/Developer`), macOS
  26.5, Swift 6.3.3, 24 GB. Installs go to `/Users/zidane/Mference/scratch/`
  (the main checkout's gitignored `scratch/`), and the workaround flags are
  still passed to `Scripts/test.sh` so the commands match the brief.
- **Jinja.** The brief says the repo has no Jinja engine. It has one for the
  *tool* path only: `encodeToolChat` renders the shipped `chat_template.jinja`
  through swift-transformers/swift-jinja for the Gemma and ChatML dialects.
  Text-only renders are hand-coded Swift for every dialect. This family
  hand-ports both paths, as the brief asks, because the template's Python-side
  semantics (`{{- param_value }}` on non-strings, an undefined `has_tool_sep`,
  `messages[::-1]`) are exactly what a second engine would get subtly wrong.

## Port status

- [x] Checkpoint selected and pinned (`cd199ce3`); control pinned (`35ac38ee`)
- [x] Architecture contract and axis gap list (this document), norm convention
      and RoPE verified against `transformers` v5.6.2 source
- [x] Tensor-name mapping table (`minicpm5.tensors.json`), 381 tensors verified
      against the shard header
- [x] Sizing model closed against the shard header (params exact) and the
      control's index `total_size` (bytes exact)
- [x] **Contract** — `ModelFamily.minicpm5`, `ArchConfig.miniCPM5_2B`, `qkNorm`
      axis, registry, 13 switch sites, FAMILY_CONTRACT rows; capability gate up
      (`familiesWithoutRunner["minicpm5"] = ["qkNormFreeAttention"]`)
- [x] **Repack** — both `SupportedModelSource` entries pinned; synthetic
      end-to-end installs through both paths; dry run closed against the real
      repo to the byte; real install + control install `--verify-install` green
- [x] **W2.1b weight level PASSED** (2026-09-10) —
      `Scripts/quantizer-weight-gate.py --family minicpm5` (generalized to take
      `--family` / `--orig` / `--control`): 39 INT4 tensors sampled by HTTP
      range request (embedding rows 0 and 65,000; `lm_head` rows 0 and 130,000;
      q/k/v/o and gate/up/down on layers 0, 10, 20, 30, 41), ~1 min 40 s of
      transfer. Relative Frobenius error against the BF16 source: **ours mean
      0.095035 / median 0.091632 vs the control's 0.096476 / 0.092975**; ours
      strictly better on 38 of 39 (worst ratio 1.0014, `l0_q_proj`); max-abs
      error better on 39 of 39 (mean ratio 0.7486). Bit-identical to the
      control on 0 of 39, as expected: the MLX affine grid anchors on the
      larger-magnitude endpoint and snaps a bin to zero, and that convention
      reproduces 98.0% of the control's packed nibbles. Same signature as the
      Qwen 3.6 result in [QUANTIZER_QUALITY.md](../QUANTIZER_QUALITY.md) §4.
- [x] **W2.1b model level PASSED** (2026-09-10) — `QuantizerQualityMeasurement`
      run three times, one process each (control `minicpm5-mlx.gturbo` →
      `minicpm5.gturbo` teacher-forced on the control's own sequences → control
      again), then `Scripts/quantizer-quality-compare.py`; 869 teacher-forced
      positions over the six corpus prompts, 64-token continuations.
      **Noise floor exactly zero** (the repeat dumps are byte-identical, so
      decode is deterministic and every difference is the quantizer).
      **Signal, D_KL(control ‖ ours): top-1 agreement 0.718, top-5 overlap
      0.717, KL median 0.178 / mean 0.448 nats (p99 3.96, max 13.2),
      max |Δlogit| mean 6.2.** Both gate criteria hold (top-1 ≥ 0.50, median
      KL ≤ 0.50). Greedy rollouts diverge early on every prompt (first
      divergence at token 0–7; the control's top-2 margin at the flip ranges
      from 0 to 2.8), so the METH-01 rollout comparison is uninformative here,
      as the method section predicts for two INT4 grids. This is a noisier
      pair than Qwen 3.6's healthy point (0.863 / 0.036): every tensor of this
      family is INT4, including the 130,560-row embedding and head, and a 2B
      dense model spends its whole logit budget on those grids. It is recorded
      as the second healthy calibration point the quality document asked for,
      not tuned toward. `Scripts/gturbo-tensor-diff.py` over the two installs:
      381 tensors, median relative difference 0.126, worst 0.207
      (`layers.0.self_attn.k_proj`), none above 0.5, norms identical — two
      independent INT4 grids of the same weights and nothing else.
      `manifest.quantizedAtInstall.qualityGate` is stamped
      `W2.1b-weight+kld-2026-09-10-vs-openbmb-MiniCPM5-2B-MLX`.
- [x] **Goldens committed** — `Scripts/parity/minicpm5_make_goldens.py`
      (torch 2.14.0 CPU, `transformers` 5.6.2 pinned): toy `LlamaForCausalLM`
      (hidden 64, 4 layers, 4 heads / 2 KV heads of 16, intermediate 128,
      vocab 128, theta 5e6, untied head), two prompts (12 and 48 tokens), 16
      greedy steps, per-layer attention / MLP / hidden captures, cached ==
      uncached rollout asserted. Two sets: `Fixtures/minicpm5-bf16/` (the
      bf16-rounded weights the checkpoint carries) and `Fixtures/minicpm5/`
      (the **gate set**: every projection replaced by its INT4 g64
      reconstruction under the `Int4AffineEncoder` transcription, so a Metal
      forward measures the port, not the quantizer). The 333 KB toy checkpoint
      is committed alongside so the Swift gates never skip. Byte-reproducible
      (20 files, re-run and diffed). Minimum top-1/top-2 rollout margin on the
      gate set: 0.022 (short), 0.053 (long); asserted ≥ 5e-3.
- [x] **Toy parity PASSED** (2026-09-10, `MiniCPM5ReferenceParityTests`, the
      Metal runner loaded from a real INT4 install of the toy checkpoint):
      - *Gate 1–2, per-module and full forward at every position of both
        prompts* — `numpy.allclose` at **atol = rtol = 1e-2** (the
        FP16-activation tier). Observed worst max-abs: `attn_out` 1.0e-3,
        `mlp_out` 1.5e-3, `hidden_out` 3.4e-3, logits 3.2e-3; argmax exact at
        every position.
      - *Gate 3, greedy rollouts* — **token-exact** for 16 steps on both prompts
        through chunked prefill + the fused greedy head, and again through the
        exact logits head (step logits inside the same tier). The goldens'
        smallest top-1/top-2 margin is 0.022, so a flip would be a defect.
      - *Gate 4, cached decode == recompute* — the runner's cached rollout
        equals its own uncached re-prefill rollout, and both equal the
        reference's (which asserts cached == uncached in-script).
      - Chunked prefill vs sequential decode: **bit-exact** on the per-row path
        (chunks under 32 tokens); on the batched INT4 QMM path the logits differ
        by at most 3.2e-3 with identical argmax — the batched MLP's summation
        order differs from the decode GEMV. Recorded as a known limit (the
        INT8 Qwen 3.8 toy never exercised this path).
      - Paged KV: full-selection paged decode is byte-identical to dense across
        page boundaries and through prefill + decode; blocked streamed prefill
        under a 5-page pool agrees with dense on the greedy head.
- [x] **Runner** — `MiniCPM5ForwardRunner` (`ForwardRunnerFactory` dispatches
      on the family); `familiesWithoutRunner` no longer lists `minicpm5` and
      `shippedFamiliesAreNotGated` covers it
- [x] **Tokenizer** — `ChatDialect.minicpm`, `MiniCPMToolCallParser`, both EOS
      ids in `stopTokenIDs`; 16 HF renders byte-identical; the installed real
      tokenizer's encodings match `transformers` 5.6.2 on every probe and every
      render (`MFERENCE_MINICPM5_TOKENIZER_DIR` run)
- [x] **Ladder** — 16 / 32 / auto byte-identical greedy output
      (`bringup-check.sh` stage 3)
- [x] **Gate** — [`FAMILY_GATE.md`](../FAMILY_GATE.md) steps 1–9 (table under
      "Measured results")
- [x] **Protocol bench** — short-explanation 3/3 `stop=endOfTurn`; the other
      two cases recorded as the stated think-block deviation

## First light (real install, 2026-09-10)

Host: Mac with Apple silicon, 24 GB, macOS 26.5, Swift 6.3.3, release
`MferenceCLI` at commit `5758e6a`; `scratch/minicpm5.gturbo` (the
quantize-in-flight install). One process at a time; `--temperature 0`.

| Run | Command (abridged) | Footer | Output |
|---|---|---|---|
| raw | `--prompt "The capital of France is" --max-new 48` | `stop=maxTokens prefill=6tok/0.37s new=48tok decode=0.61s tok/s=78.452` | coherent English, evades the answer (" a well-known fact, but the question of what constitutes a capital can be more nuanced…") |
| chat | `--messages-file chat.json --max-new 512` ("Explain in two sentences why the sky is blue.") | `stop=maxTokens prefill=21tok/0.33s new=512tok decode=7.17s tok/s=71.382` | **empty**: all 512 tokens stayed inside the think block; at 2048 tokens the same (`decode=37.35s tok/s=54.826`) |
| chat, think exposed | the rendered chat prompt via `--prompt` (no decoder) `--max-new 400` | `stop=maxTokens … tok/s=72.204` | coherent but looping: "We need to answer: … The user wants it in two sentences. So we need to produce a response that is exactly two sentences." repeated |
| needle, paged | `--messages-file needle.json --max-new 96 --max-context 16384 --kv-paged on --kv-pool-pages 96` (8,692 prompt tokens, passkey at 45 %) | `stop=endOfTurn prefill=8692tok/139.58s new=94tok decode=5.33s tok/s=17.646` | **`4917`** — the passkey, exactly, after a think block that closed on its own |

Reading: the runner is coherent at real shape, and the needle run is the strong
evidence — 8,692 tokens against a 96-page (6,144-token) pool means the sealed
past spilled to SSD and streamed back through the blocked prefill, Quest
selection ran over 136 pages per layer during decode, and the model still
produced the exact passkey and closed its own think block (`stop=endOfTurn`).
The greedy think-loop on the short chat prompt is a temperature-0 behaviour of
a 2B thinking model (the vendor recommends `temperature 1.0`), not a runner
defect the toy gates or the needle would have missed; the community protocol
below samples at 0.2 / top-k 64 / top-p 0.95 and is where end-of-turn is
measured. An independent HF-reference greedy decode of the real checkpoint was
**not** run: the BF16 shard is streamed at install and never staged, and
re-downloading 5 GB for a diagnostic is exactly what AGENTS.md forbids.

## Server round trip (2026-09-10)

`MferenceServer --model scratch/minicpm5.gturbo --port 8095 --max-context 8192`
(release build at `806b046`), one server process, requests with
`temperature 0.2`, `seed 20260721`, `max_completion_tokens 1024`:

1. `GET /v1/models` → `minicpm5-2b-int4g64`.
2. A `tools` request ("What is the weather in Paris right now? Use the tool.",
   `get_weather` schema) → `finish_reason: "tool_calls"`, one call parsed from
   the model's XML: `get_weather` with arguments `{"city":"Paris"}`;
   232 completion tokens (the think block included), content `"\n\n"`.
3. The same history plus the assistant call and a `role: tool` result
   (`{"temp_c": 21, "sky": "clear"}`) → `finish_reason: "stop"`, content
   "The current weather in Paris is 21°C with a clear sky."
4. A streamed request with `stream_options.include_usage` → 6 SSE events, the
   last content chunk followed by a `finish_reason: "stop"` chunk, the usage
   chunk, and `[DONE]`; the generation ended on `<|im_end|>` (an EOS id).

### Install determinism and the quality stamp

The install was produced three times on this host (first install, then twice
with `--overwrite` while adding the family-specific `qualityGate` stamp).
`model_weights.bin` hashed to the same SHA-256 each time
(`ed1e9f57…eb41aa`, 1,415,974,912 bytes): quantize-in-flight is
byte-deterministic. The shipped manifest now carries
`quantizedAtInstall.qualityGate =
W2.1b-weight+kld-2026-09-10-vs-openbmb-MiniCPM5-2B-MLX` and verifies at
8 files / 1,425,981,882 bytes. A byte comparison of every BF16 passthrough
tensor against the control install (85 norm vectors) found all 85 identical,
and the two installs' resident indexes carry the same 381 names.

## Measured results

Host: MacBook Pro (`Mac17,2`), Apple M5 (10 cores), 24 GB, macOS 26.5,
Swift 6.3.3, Mference commit `b70ac36` (release build). Protocol:
[`COMMUNITY_BENCHMARKS.md`](../COMMUNITY_BENCHMARKS.md), 3 measured
repetitions per case after one discarded warmup, each run a fresh process,
`--max-new 1024 --max-context 4096 --temperature 0.2 --top-k 64 --top-p 0.95`
with the frozen seeds. Medians across the measured repetitions; peak RSS is
the maximum.

| Case | Prompt / generated | Prefill | Decode | Range | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: |
| short-explanation | 57 / 761, `stop=endOfTurn` 3/3 | 0.20 s | **65.56 tok/s** | 65.50–65.80 | 123 MiB (summarizer's figure) |
| medium-review | 431 / 1024, **`stop=maxTokens`** | 0.74 s | (rejected by the protocol) | — | — |
| long-synthesis | 2929 / —, not reached | — | (aborted after medium-review) | — | — |

**Deviation, stated as Qwen 3.8 did in
[`BENCHMARKS_M3_ULTRA.md`](../BENCHMARKS_M3_ULTRA.md).** The chat template
opens a `<think>` block (the family default, see "Tokenizer and chat
template"), and for medium-review the model does not leave it within the
protocol's 1,024 tokens, so `run-benchmark.sh` correctly rejects that warmup
and aborts (`ABORT: warmup medium-review did not reach a natural end of turn`,
footer `stop=maxTokens prefill=431tok/0.74s new=1024tok decode=18.28s
tok/s=56.011`). The short-explanation row above is the protocol proper, run
alone (`BENCH_CASES=short-explanation ./run-benchmark.sh minicpm5 … 3`). The
other two cases were then measured **outside the protocol** with only the
token cap changed — identical prompts, sampling, seeds, fresh processes, one
discarded warmup, `--max-new 4096 --max-context 8192`:

| Case (deviation) | Prompt / generated | Prefill | Decode | Range | Stop |
| --- | --- | ---: | ---: | ---: | --- |
| medium-review | 431 / 4096 | 0.74 s (583 tok/s) | 37.63 tok/s | 37.63–37.81 | **`stop=maxTokens` in all 3 + warmup** — still inside the think block at 4,096 tokens |
| long-synthesis | 2929 / 573 | 11.96 s (245 tok/s) | 34.72 tok/s | 34.72–34.82 | `stop=endOfTurn` 3/3 + warmup |

The medium-review think block at temperature 0.2 does not terminate within
4,096 tokens on this host; the output is not reported as a speed result. It is
not a rendering defect: HF `apply_chat_template` and the Swift dialect produce
the same 431 token ids for that prompt (57 and 2,929 for the other two), and
the three protocol renders are now part of the byte-exact fixture set. It is,
however, a one-newline-sensitive behaviour: the same prompt fed as a raw
completion with the render's trailing `\n` after `<think>` stripped (a
`$(cat …)` artefact, 430 tokens) reasoned for 820 tokens, closed the think
block, and wrote a coherent review at the same seed
(`stop=endOfTurn prefill=430tok/0.74s new=820tok decode=13.99s tok/s=58.632`).
The exact template is what ships; the sensitivity is recorded, not worked
around. Rows
here are not comparable with [`BENCHMARKS.md`](../BENCHMARKS.md) unless the
case, prompt and generated token counts, settings and stop reason all match.

### Phases attribution

One `MFERENCE_PHASES=1` run of the short-explanation case at the protocol
settings (release CLI, commit `806b046`+docs; `stop=endOfTurn`, 761 tokens,
`decode=11.49s tok/s=66.239`), the baseline later optimization A/Bs are judged
against. A dense family has no expert I/O; one command buffer per token.

| Phase | ms/token | Share |
|---|---:|---:|
| GPU execution (command-buffer `gpuStartTime`→`gpuEndTime`) | 13.89 | 91.9 % |
| CPU encode + commit | 0.15 | 1.0 % |
| wait / readback (fused greedy token) | 0.19 | 1.2 % |
| unaccounted (loop, sampling, detokenize) | 0.89 | 5.9 % |
| **decode step** | **15.12** | 100 % |

Decode is GPU-bound at 92 %; the 1.42 GB of INT4 weights read per token put the
achieved bandwidth at ~102 GB/s on this M5.

### Family gate (`FAMILY_GATE.md`)

| Step | Result |
|---|---|
| 1 full suite ×3 | see "Suite runs" below |
| 2 release build | clean (all products) |
| 3 static checks | `git diff --check` clean; markdown link checker clean under `LC_ALL=en_US.UTF-8` |
| 4 pinned install + smoke | `minicpm5` from `cd199ce3` with strict verification; raw and chat CLI smoke with normal footers (first light above) |
| 5 protocol page | short-explanation 3/3 `stop=endOfTurn`; medium-review and long-synthesis recorded as the stated deviation |
| 6 phases snapshot | above |
| 7 optional features | none; nothing approximate on the default path |
| 8 provenance | `THIRD_PARTY_NOTICES.md` updated; no reference code imported; no credentials or private paths in fixtures |
| 9 merge-compatibility | note below |
| `bringup-check.sh minicpm5 scratch/minicpm5.gturbo` | **PASS, 0 stages skipped** (preflight; toy suite `[Mm]iniCPM5`; install verify 8 files / 1,425,981,882 bytes; ladder smoke 16 / 32 / auto byte-identical at 85.9 / 85.4 / 83.3 tok/s; protocol scaffold) — `benchmark-results/minicpm5-bringup/bringup-report.txt` |

## Known limits

- The vendor's DSpark drafter (`openbmb/MiniCPM5-2B-DSpark`: 5-layer
  qwen3-architecture block-diffusion draft model, block_size 7, target layers
  `[1, 10, 20, 30, 39]`, confidence head) is **not** part of this bring-up.
  Scoping note, written after the gate went green:
  [2026-09-10-minicpm5-dspark-drafter-scoping.md](../superpowers/specs/2026-09-10-minicpm5-dspark-drafter-scoping.md).
- Greedy decoding (`--temperature 0`) can loop inside the think block on
  short chat prompts, and at the protocol's 0.2 the medium-review case does not
  leave its think block within 4,096 tokens; the vendor recommends
  `temperature 1.0`. Recorded under "First light" and "Measured results".
- Chunked prefill is FP16-tier equal to sequential decode on the batched INT4
  QMM path (max-abs 3.2e-3, argmax identical), bit-exact only on the per-row
  path (chunks under 32 tokens).
- Optional or approximate features: none.
- Untested: everything below the "Port status" checkboxes that is not ticked.

## Merge-compatibility note

**Landed after #25 (2026-09-10).** This branch was written against
`neel/qwen-flash-benchmarks-frontend-3d297b` (main plus the three W2.1b
commits) while two other branches were in flight, and both of those merged
first as PR #25: the Flash-Next capability-gate lift and Open WebUI plus
`MferenceServer --library` mode, which also deleted the native Mac app. The
suggested land order in the original note was followed. What the integration
merge actually had to resolve:

| File | Resolution |
|---|---|
| `ManifestReader.familiesWithoutRunner` | no conflict — this branch is net-unchanged there, and the table now ships empty |
| `Model.swift` | both sides' cases kept: `.qwen38flashnext` throws the post-#25 `accessorNotAvailable(_:)` (the old `runnerNotImplemented()` helper is gone), `.minicpm5` returns its `input_layernorm` / `post_attention_layernorm` |
| `FlashNextCapabilityGateTests.swift` | `shippedFamiliesAreNotGated` lists both `.qwen38flashnext` and `.minicpm5`; the rest of the suite is #25's lifted-gate form |
| `QuantBitPolicyTests.swift` | #25's `.moeRouterInt8` tests kept; `minicpm5`'s uniform-INT4 answer asserted on its own line rather than in the no-entry loop |
| `Scripts/quantizer-weight-gate.py` | one script: #25's `--family/--orig/--control/--tensors/--rows`, g32 awareness and 429 backoff (the importable surface `Scripts/flashnext-router-int4-check.py` depends on) plus this branch's `minicpm5` pins and sample plan. `control_name` now derives the head's control name from the control prefix instead of hard-coding Qwen's `language_model.lm_head` |
| `bringup-check.sh` | both `case` lines and both usage labels; #25's `pgrep` pattern |
| `THIRD_PARTY_NOTICES.md` | both notices; "one of six checkpoints" became seven, with Flash-Next and MiniCPM5 named as the two vendor-BF16 installs |
| `docs/FAMILY_CONTRACT.md` | both the MC5 column and #25's empty-gate prose, which now records that MC5 never entered the gate |
| `Sources/MferenceApp/Core/Installation/AppModelInstallDescriptor.swift` | deleted. The whole app went with #25; the UI discovers installs through `MferenceServer --library` (`Sources/MferenceServer/Core/ServerModelDiscovery.swift`), which needs no per-family descriptor |
| `Sources/MferenceServer/Core/ServerModelDiscovery.swift` | not a text conflict but a build break: `ServerFamilyModelID.modelID(for:)` is a deliberately exhaustive `switch` over `ModelFamily`, so `.minicpm5` was added there with the same `minicpm5-2b-int4g64` id `ServerModelSession.defaultModelID` uses |

## Reproduction

```bash
# Build once.
swift build -c release

# Install from the pinned BF16 revision (quantize-in-flight, ~5 GB read).
swift run -c release MferenceRepack --model minicpm5 --output scratch/minicpm5.gturbo
swift run -c release MferenceRepack --verify-install --input-gturbo scratch/minicpm5.gturbo

# The W2.1b control (vendor MLX conversion, pre-quantized path, ~1.4 GB).
swift run -c release MferenceRepack --model minicpm5mlx --output scratch/minicpm5-mlx.gturbo

# Conformance stages.
./bringup-check.sh minicpm5 scratch/minicpm5.gturbo

# Community protocol benchmark.
./run-benchmark.sh minicpm5 scratch/minicpm5.gturbo 3

# Phases attribution baseline.
MFERENCE_PHASES=1 .build/release/MferenceCLI \
  --model scratch/minicpm5.gturbo \
  --messages-file docs/benchmark-prompts/real-generation-v1/short-explanation.json \
  --max-new 1024 --max-context 4096 --temperature 0.2 --top-k 64 --top-p 0.95 \
  --seed 20260721
```

### Reproduction of this dossier's facts

```bash
curl -s https://huggingface.co/api/models/openbmb/MiniCPM5-2B          # sha cd199ce3…
curl -s https://huggingface.co/api/models/openbmb/MiniCPM5-2B-MLX      # sha 35ac38ee…
# shard header: bytes 0-7 give the header length (44,128); bytes 8..8+44127 are the JSON
# transformers v5.6.2: raw.githubusercontent.com/huggingface/transformers/v5.6.2/src/transformers/models/llama/{modeling,configuration}_llama.py
```

Run one model process at a time, and re-read the preconditions in
[`AGENTS.md`](../../AGENTS.md) before any model run.
