# Quantizer quality — the W2.1b gate

**Status (2026-09-02): weight level PASSED. Model level BLOCKED — not failed,
blocked, by a production gap named in [§6](#6-the-blocker-why-the-model-level-half-could-not-be-run).
`manifest.quantizedAtInstall.qualityGate` therefore still reads
`W2.1b-kld-open`.**

This is the reusable quality gate for `MferenceRepack`'s quantize-in-flight
path. Every family installed from a vendor's original BF16 repo goes through the
same `Int4AffineEncoder.encodeGroup` nucleus, so a family that uses that path
inherits this validation rather than needing its own. Qwen 3.6 is the subject
because it is the one checkpoint that exists on **both** sides: a vendor BF16
upload we can quantize, and an independent mlx-community conversion we can treat
as a trusted control.

| Gate | What it proves | State |
|---|---|---|
| W2.1a | The repacker's quantizer is bit-identical to the runtime's own reference quantizer, including through the streaming path | Enforced in CI (`Int4AffineEncoderParityTests`, `Int4AffineStreamingParityTests`) |
| W2.1b weight level | Our quantized weights are as faithful to the BF16 source as an independent, trusted quantizer's | **Passed** — [§4](#4-result) |
| W2.1b model level | Greedy rollouts and KLD against the control, end to end | **Blocked** — [§6](#6-the-blocker-why-the-model-level-half-could-not-be-run) |

---

## 1. The control pair

| | Trusted control | Ours |
|---|---|---|
| Installer entry | `qwen36` | `qwen36original` |
| Source repo | `mlx-community/Qwen3.6-35B-A3B-4bit` | `Qwen/Qwen3.6-35B-A3B` |
| Revision | `38740b847e4cb78f352aba30aa41c76e08e6eb46` | `995ad96eacd98c81ed38be0c5b274b04031597b0` |
| Index SHA-256 | `0b28df60…3842ea` | `41b93561…0be83` |
| Source form | INT4 affine g64, already packed | BF16, 26 shards, 71.9 GB |
| Who quantized it | MLX `affine_quantize` | `Int4AffineEncoder` at install time |

Both are the same trained checkpoint. Any difference between them is
quantization, not training.

## 2. The cheap decisive experiment

The obvious way to compare two quantizers is to install both and diff. That
costs 72 GB and ~90 minutes before it says anything.

There is a much cheaper exact route. Group-64 quantization **never straddles a
row** — every quantizable last dimension is a multiple of 64 — so a *row slice*
is self-contained: those BF16 rows fully determine the packed nibbles, scales
and biases for exactly those rows, on both sides. So matched row slices can be
pulled by HTTP range request from both repos and compared bit for bit.

`Scripts/quantizer-weight-gate.py` does this. It samples 124 INT4 tensors —
embeddings, the LM head, linear-attention and full-attention projections, shared
experts, and routed experts `{0, 5, 137, 255}` across layers `{0, 7, 19, 27,
39}` — for about **11 MB of transfer**. Run it before committing to any install:

```
Scripts/quantizer-weight-gate.py --cache /tmp/mference-quant-gate
```

### 2a. A side finding: the fused-expert axis order is now confirmed

`FlashNextPlanner` split `mlp.experts.gate_up_proj` on an explicitly
**unverified** assumption about its axis order, because checking it needed
Flash-Next's 360 GB download. Qwen 3.6 — same vendor, same fused naming — settles
it from a 2 KB shard header:

* `gate_up_proj` is `[256, 1024, 2048]` = `[E, 2·I, H]`,
* `down_proj` is `[256, 2048, 512]` = `[E, H, I]`,

with `moe_intermediate_size` 512 and `hidden_size` 2048. Splitting `gate_up_proj`
at the halfway row also reproduces mlx-community's independently converted
`switch_mlp.gate_proj` / `switch_mlp.up_proj`, so the **gate-first** ordering is
confirmed too, not merely the shape.

## 3. What the bytes said, and why

**The packed bytes are not bit-identical, and never can be: 0 of 124 tensors
match.** That is not a bug. The two quantizers implement genuinely different
affine grids.

| | Mference `Int4AffineEncoder` | MLX `affine_quantize` |
|---|---|---|
| Scale | `(max − min) / 15`, always positive | `±(max − min) / 15`, sign chosen by which endpoint has larger magnitude |
| Bias | `min` | whichever endpoint has larger magnitude |
| Zero point | wherever it falls | **snapped**: the scale is rescaled so an integer bin lands exactly on `0.0` |
| Rounding | indices computed against the **BF16-rounded** scale/bias | indices computed in FP32, scale/bias only *stored* as BF16 |

The zero-point snap is the dominant term. A worked example from
`layers.0.mlp.shared_expert.gate_proj`, group 0: the group's min is
`−0.02380371`, so a plain grid gives `scale = 0.00280762`. MLX instead stores
`0.00297546`, which is exactly `0.02380371 / 8` — bin 8 lands on `0.0` on the
nose. Where MLX anchors on the maximum instead, its stored scale is negative and
the bin order reverses, which is why the per-nibble difference reaches the full
15 steps on some groups.

Modelling that convention (`encode_mlx_model` in the script) reproduces
**98.35 % of the control's packed bytes** in non-degenerate groups — that is the
`attrib` column in the script's output. The two halves of the convention
contribute unevenly: endpoint anchoring alone accounts for 59.5 % of bytes,
and adding the zero-point snap takes it to 98.2 % (measured over the first 40
sampled tensors), so the snap is the larger term but not the only one. The
residual ~1.7 % is tie-breaking at half-integer boundaries, both in the snap
decision (`round(q0)`) and in the index rounding; it is second-order and does
not affect any conclusion below. The model exists only to *attribute* the
difference — no gate depends on it.

**Degenerate groups.** Layer 0's routed experts carry a large mass of values at
`±2^-123` (~7.8e-38) — an effectively dead region of the checkpoint. Those
groups are excluded from the statistics (1,920 of 56,832); both quantizers
reconstruct them as ~0 and the nibble differences there are meaningless.

## 4. Result

Reconstruction error against the BF16 source — the only thing that actually
matters, since neither grid is "the" right answer:

```
124 INT4 tensors compared (degenerate groups excluded)
  bit-identical to the control: 0/124
  relative Frobenius error  ours mean 0.09612 median 0.09291
  relative Frobenius error   mlx mean 0.09648 median 0.09396
  ours strictly better on 118/124; worst ratio 1.5435 on l0_e255_down
```

**Our quantizer is at least as faithful as the trusted control, and slightly
better on average.** It is better on 118 of 124 tensors by relative Frobenius
error, and by *maximum* absolute error it is better on **123 of 124** — mean
ratio 0.72, worst case 1.03. That asymmetry is the expected signature of the
convention difference: a plain min/max grid minimises the worst-case half-step,
which is precisely what the zero-point snap trades away in exchange for exact
zeros.

### The six tensors where we are worse, and why

All six are **layer-0 routed-expert `down_proj`** (ratios 1.21–1.54). The cause
is attributable, not mysterious: 55–69 % of the values in layer 0's expert
matrices are that `±2^-123` dead mass. `down_proj`'s groups run along the
`moe_intermediate` axis (512), so each group *mixes* dead columns with live
ones — the group is not degenerate overall, but most of its entries want to be
zero. MLX's snap makes `0.0` exactly representable and reconstructs them
perfectly; our grid puts them on the nearest bin, up to half a step away.

**This is the one real, localised quality deficit of `Int4AffineEncoder`, and it
generalises**: any checkpoint with a large mass of exact or near-exact zeros
inside otherwise-live groups will show it. Adopting MLX's zero-point snap would
close it. That is an owner decision, not a mechanical fix — it changes every
byte the repacker has ever written, invalidates the W2.1a fixtures, and would
require re-running this gate. `Int4AffineEncoderConventionTests` now locks the
current convention so that the choice cannot drift silently.

## 5. The model-level harness

Built, compiling, and ready to run the moment [§6](#6-the-blocker-why-the-model-level-half-could-not-be-run)
is cleared.

* `Tests/Mference/Core/Runtime/Quantizer/QuantizerQualityMeasurement.swift` —
  env-gated, skips without an install. Each model runs **alone** and dumps to
  disk; nothing ever loads two models, per AGENTS.md.
* `Scripts/quantizer-quality-compare.py` — offline comparator.

Three design points that make the measurement valid:

1. **Teacher forcing, not free-running rollouts.** Two rollouts diverge at the
   first token they disagree on, and every position after that is conditioned on
   a different history; a "KLD" over them measures divergence of *contexts*, not
   quantization damage. Both models are therefore driven over one fixed token
   sequence — prompt plus the trusted model's own greedy continuation.
2. **Full vocabulary, no top-K.** The dump holds all 248,320 logits at every
   position, so the reported KL is exact rather than a bound, with no log-sum-exp
   correction needed.
3. **FP16 dumps, not FP32.** FP16 is the runner's own output precision, so the
   dump is bit-exact, half the size, and makes the noise floor checkable by
   simple byte equality: two deterministic runs must produce identical files.

One trap worth naming, because it fails silently: the harness must build its
runner with `RuntimeConfiguration(forceLogitsHead: true)`. The default fused
head **skips the logits write entirely** and leaves only a greedy argmax in
`lastGreedyToken` (`RealForwardRunner.swift:352`). A logit-dumping harness that
misses this reads never-written memory and produces confident, meaningless
numbers.

Corpus: the three frozen prompts in `docs/benchmark-prompts/real-generation-v1/`
(content clipped to 900 characters before templating, to keep the dump around
850 MB per run) plus three short raw prompts (`2 + 2 =`,
`The capital of France is`, `Three primary colors are red, blue, and`).

Run order — three sequential processes, never two at once:

```
# 1. trusted: rollout + teacher-force its own sequences, publishing tokens.json
MFERENCE_QUANT_QUALITY_GTURBO=…/qwen36.gturbo \
MFERENCE_QUANT_QUALITY_DUMP=…/dump MFERENCE_QUANT_QUALITY_LABEL=trusted \
  ./Scripts/test.sh --filter QuantizerQualityMeasurement

# 2. ours: teacher-force the SAME sequences
MFERENCE_QUANT_QUALITY_GTURBO=…/qwen36-ourquant.gturbo \
MFERENCE_QUANT_QUALITY_DUMP=…/dump MFERENCE_QUANT_QUALITY_LABEL=ours \
MFERENCE_QUANT_QUALITY_TEACHER=…/dump/trusted/tokens.json \
  ./Scripts/test.sh --filter QuantizerQualityMeasurement

# 3. the noise floor: the trusted install again, in a fresh process
MFERENCE_QUANT_QUALITY_GTURBO=…/qwen36.gturbo \
MFERENCE_QUANT_QUALITY_DUMP=…/dump MFERENCE_QUANT_QUALITY_LABEL=trusted-repeat \
MFERENCE_QUANT_QUALITY_TEACHER=…/dump/trusted/tokens.json \
  ./Scripts/test.sh --filter QuantizerQualityMeasurement

Scripts/quantizer-quality-compare.py …/dump \
  --reference trusted --candidate ours --noise trusted-repeat
```

The noise floor is not ceremony. Decode is expected to be deterministic at
temperature 0, so the floor should be exactly zero and the dumps byte-identical;
if it is not, that non-determinism is itself a finding and every KL number has to
be read against it. The comparator reports it first for that reason, and
`METH-01` applies to the rollout comparison: a top-1 flip at a top-2 logit margin
below the measured noise is not evidence of damage, which is why the harness
records that margin at every position.

## 6. The blocker: why the model-level half could not be run

**The control install is not uniformly INT4. Ours is. The Qwen 3.6 runner only
accepts the control's mixture.**

mlx-community's conversion carries 83 per-tensor overrides in its `config.json`:
every layer's `mlp.gate` (the router) and `mlp.shared_expert_gate` is **INT8**
group-64, while everything else is INT4. Its manifest records
`quant.router.weightBits = 8`.

Our quantize-in-flight path has exactly one target. That was a deliberate W2
scope cut — *"int4 group-64 only"* — and it is harmless for Flash-Next, whose
runner drives the router through the generic INT4 matvec. On Qwen 3.6 it is
fatal, in two independent places:

1. **`ManifestReader.validateQuant`** (`Sources/Mference/Infrastructure/ModelIO/ManifestReader.swift:462`)
   admits `("router", quant.router, [8])` for every non-Flash-Next family. Our
   install reports 4 and is refused at load. This is observed, not predicted —
   `qwen36original` was installed and verified (47 files, 19,533,682,725 bytes,
   against the control's 19,551,394,758) and loading it through the harness
   fails immediately with:

   ```
   resident index is corrupt: unsupported quantization for router
   ```
2. Even bypassed, the kernel could not decode it. `router_gemv_gemma4_body`
   (`Sources/Mference/Metal/MoE/moe.metal:109`) reads `W_row[idx]` as **one
   `uint8` per weight**. There is no nibble unpacking on that path at all; a
   4-bit router would be read as garbage routing over 256 experts, which would
   not be a quality measurement of anything.

The refusal is the right behaviour — it fails loudly rather than silently
mis-routing — but it means `qwen36original` installs and verifies and cannot be
*run*, so the KLD half of the gate cannot be measured on this control.

### The router really does need those extra bits

It would be easy to read the above as bureaucratic and reach for the smallest
change that gets a number out. It is not bureaucratic. Quantizing the router to
INT4 and scoring the resulting top-8-of-256 selection against the BF16 router
(512 random unit-norm probe vectors per layer):

| Layer | rel. weight error INT4 | INT8 | top-8 set agreement INT4 | INT8 | top-1 agreement INT4 | INT8 |
|---|---|---|---|---|---|---|
| 0 | 0.105 | 0.008 | 0.868 | 0.989 | 0.773 | 0.975 |
| 19 | 0.137 | 0.011 | 0.823 | 0.982 | 0.705 | 0.977 |
| 39 | 0.105 | 0.008 | 0.848 | 0.984 | 0.768 | 0.986 |

An INT4 router changes roughly **13–18 % of the selected expert set** and about
**a quarter of the top-1 expert**, against ~1.5 % and ~2 % for INT8. That is why
mlx-community kept 83 tensors at INT8, and it carries a sharper consequence for
this gate: **even if the runner accepted an INT4 router, the measurement would
be invalid.** The KL divergence would be dominated by routing divergence — a
different subset of experts running — rather than by the weight quantizer the
gate is supposed to be testing.

*Caveat:* the probes are isotropic Gaussians, not real hidden states, which are
not isotropic. Treat these as the right order of magnitude and the right
ordering, not as exact rates.

This tension was invisible until now because Flash-Next was the only
original-repo family, and its runner happens to accept a uniform-INT4 router.

### Options, sized

* **(A) Give the repacker an INT8 affine group-64 mode and a per-family
  bit policy** so `qwen36original` reproduces the control's mixture exactly.
  Correct, and it makes the two installs byte-comparable on *every* tensor.
  Cost: a new `Int8AffineEncoder`, a new `StreamingInt4Quantizer` component and
  `RangeCopyTransform` case, a per-tensor policy in the original-repo planner,
  and manifest plumbing — roughly 200–300 lines across five files. It also has a
  bootstrap problem: new production quantization math needs its own W2.1a-style
  bit-parity gate before it can be trusted, and that gate is what this document
  is.
* **(B) Leave the router BF16 for the original-repo path and dispatch the
  existing `router_gemv_bf16_r4`** (already used by DeepSeek V4). Much smaller —
  the kernel exists — but it still touches the shipped Qwen 3.6 runner's
  dispatch, and it makes our router *better* than the control's INT8 rather than
  equal, so the router term of the comparison stops being apples-to-apples.
* **(C) Teach the Qwen 3.6 runner an INT4 router.** New fused-kernel work on a
  shipped family. Largest, buys the least, and — per the table above — would
  produce a measurement of the wrong thing even once it worked.

**(A) is the recommendation.** It is the only option that leaves the control
experiment honest, the router table shows the extra bits are load-bearing rather
than ceremonial, and a per-tensor bit policy is something the bring-up kit will
need anyway the first time a vendor ships a checkpoint whose community
conversion mixes widths — which is now known to be the common case, not the
exotic one.

## 7. Verdict

* **Weight level: PASS.** Our INT4 affine group-64 quantizer reconstructs the
  BF16 source at least as faithfully as an independent trusted implementation —
  mean relative error 0.09612 against the control's 0.09648, better on 118 of
  124 sampled tensors. The one deficit (layer-0 `down_proj`, up to 1.54×) is
  fully attributed to the absence of zero-point snapping and is confined to
  tensors with a large mass of near-exact zeros.
* **Model level: NOT MEASURED.** Blocked by §6. No threshold is proposed here,
  because proposing one without the measurement behind it would be inventing a
  passing number.
* **The stamp stays.** `manifest.quantizedAtInstall.qualityGate` remains
  `W2.1b-kld-open`. It is not honest to promote it on the weight-level half
  alone: the weight gate says each tensor is individually faithful, which is not
  the same claim as 40 layers of accumulated error leaving the output
  distribution intact.
* **Flash-Next.** Closing W2.1b would *not* by itself lift
  `ManifestReader.familiesWithoutRunner`. That gate refuses Flash-Next for
  missing **axes** (`hyperConnectionsLowRank`, `attentionIndexer`,
  `pleNgramEmbedding`), not for quantizer quality. W2.1b is one of the blockers
  behind the eventual lift decision, not the lift itself.
