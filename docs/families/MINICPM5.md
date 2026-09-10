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
| Install size | ~1.43 GB estimated (see Memory budget); measured value recorded below once installed |
| Status | **in port** (Day-0 dossier) |

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
| **On disk** (+ index ≤ 64 KiB, tokenizer 10.0 MB, manifest, receipt) | | **~1,426 MB** |

The resident total equals the control's `model.safetensors.index.json`
`total_size` (1,415,925,760) to the byte: same tensor set, same format.

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
- [ ] **Contract** — `ModelFamily.minicpm5`, `ArchConfig.miniCPM5_2B`, `qkNorm`
      axis, registry, switches, FAMILY_CONTRACT rows; capability gate up
- [ ] **Repack** — `SupportedModelSource` entries pinned; synthetic install;
      dry run vs real bytes; `--verify-install` green
- [ ] **W2.1b** — weight level and model level vs the vendor MLX control
- [ ] **Toy parity** — goldens committed; per-module, forward, rollout, cache gates
- [ ] **Runner** — `MiniCPM5ForwardRunner`; capability gate lifted; first light
- [ ] **Tokenizer** — dialect, XML parser, two EOS ids, fixtures byte-identical
- [ ] **Ladder** — 16 / 32 / auto byte-identical greedy output
- [ ] **Gate** — [`FAMILY_GATE.md`](../FAMILY_GATE.md) steps 1–9
- [ ] **Protocol bench** — three frozen `real-generation-v1` cases

## Measured results

Host: recorded with the first measurement. Protocol:
[`COMMUNITY_BENCHMARKS.md`](../COMMUNITY_BENCHMARKS.md), 3 measured
repetitions per case after one discarded warmup, each run a fresh process.

| Case | Prompt / generated | Prefill | Decode | Range | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: |
| short-explanation | | | | | |
| medium-review | | | | | |
| long-synthesis | | | | | |

Decode rate excludes model installation, model loading, and prompt prefill.

### Phases attribution

| Phase | ms/token | Share |
|---|---:|---:|
| (not yet measured) | | |

## Known limits

- Not yet loadable: the family is in `ManifestReader.familiesWithoutRunner`
  until the runner lands and its parity gates pass.
- The vendor's DSpark drafter (`openbmb/MiniCPM5-2B-DSpark`: 5-layer
  qwen3-architecture block-diffusion draft model, block_size 7, target layers
  `[1, 10, 20, 30, 39]`, confidence head) is **not** part of this bring-up. A
  one-page scoping note on plugging it into a `RoundDrafter`-style protocol is
  written only after the family gate is green.
- Optional or approximate features: none.
- Untested: everything below the "Port status" checkboxes that is not ticked.

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
