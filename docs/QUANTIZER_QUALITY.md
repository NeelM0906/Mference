# Quantizer quality — the W2.1b gate

**Status (2026-09-02): BOTH HALVES PASSED.** The weight level passed first
([§4](#4-result)). The model level was then unblocked — by giving the repacker
a per-tensor bit policy, [§6](#6-the-blocker-and-how-it-was-cleared) — run, and
passed ([§8](#8-the-model-level-result)). `manifest.quantizedAtInstall.qualityGate`
now reads `W2.1b-weight+kld-2026-09-02-vs-mlx-community-qwen36`.

**Clearing the blocker turned up two further defects that nothing else in the
kit could see, and the second is the reason this half of the gate exists.**
Fixing the manifest gate only revealed the next one: the resident index named
every tensor the vendor's way, so no runner accessor could find any of them.
And once the model finally loaded, the gate's first run reported **zero** top-1
agreement — which turned out to be a missing RMSNorm `1 + w` fold that made
every norm in the model wrong by one. That install verified, validated and
loaded, and generated fluent nonsense at full speed. See
[§7](#7-two-defects-the-model-level-half-caught).

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
| W2.1b model level | Greedy rollouts and KLD against the control, end to end | **Passed** — [§8](#8-the-model-level-result) |

The two halves answer different questions, and the second is not implied by the
first. The weight level says each tensor is individually faithful. The model
level asks whether 40 layers of accumulated error leave the output distribution
intact — and, as it turned out, whether the install is *correct at all* in ways
that have nothing to do with quantization ([§7](#7-two-defects-the-model-level-half-caught)).

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

Run on 2026-09-02; results in [§8](#8-the-model-level-result).

* `Tests/Mference/Core/Runtime/Quantizer/QuantizerQualityMeasurement.swift` —
  env-gated, skips without an install. Each model runs **alone** and dumps to
  disk; nothing ever loads two models, per AGENTS.md.
* `Scripts/quantizer-quality-compare.py` — offline comparator. It now checks
  **dump authenticity** before reporting any number: finite, non-degenerate, a
  sane log-sum-exp, and an argmax matching the known greedy token. That is the
  `forceLogitsHead` trap below, turned from a warning into an assertion.
* `Scripts/gturbo-tensor-diff.py` — not part of the measurement, but the tool
  that diagnosed it when the first run came back wrong. Dequantizes every
  resident tensor of two installs on disk and reports the relative difference;
  no GPU, no model load. See [§7b](#7b-the-rmsnorm-1-w-convention-found-by-the-first-run).

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
records that margin at every position. Both mattered in practice: the floor came
back at exactly zero, and every observed rollout divergence sat at a control-side
margin below the median inter-install logit difference ([§8](#8-the-model-level-result)).

## 6. The blocker, and how it was cleared

**The control install was not uniformly INT4. Ours was. The Qwen 3.6 runner
only accepts the control's mixture.**

mlx-community's conversion carries **80** per-tensor overrides in its
`config.json`: every layer's `mlp.gate` (the router) and
`mlp.shared_expert_gate` is **INT8** group-64, while everything else is INT4.
Its manifest records `quant.router.weightBits = 8`, and its own
`bitWidthOverridesHonored` is 80.

> *Correction, 2026-09-02:* earlier revisions of this document, and the
> bring-up spec, said **83**. The figure is 80 — 40 layers x 2 gating tensors —
> confirmed independently from the control's source `config.json` (80 override
> keys, all `bits: 8`) and from its installed resident index (80 tensors whose
> stored width is 8). Nothing depended on the wrong number; the policy derives
> the set from a rule rather than from a count.

Our quantize-in-flight path had exactly one target. That was a deliberate W2
scope cut — *"int4 group-64 only"* — and it is harmless for Flash-Next, whose
runner drives the router through the generic INT4 matvec. On Qwen 3.6 it was
fatal in two independent places:

1. **`ManifestReader.validateQuant`** admits `("router", quant.router, [8])` for
   every non-Flash-Next family. Our install reported 4 and was refused at load
   with `resident index is corrupt: unsupported quantization for router`.
2. Even bypassed, the kernel could not decode it. `router_gemv_gemma4_body`
   (`Sources/Mference/Metal/MoE/moe.metal:109`) reads `W_row[idx]` as one
   `uint8` per weight. There is no nibble unpacking on that path at all.

### What was built

**Option (A) of the three sized below.** The repacker gained an INT8 affine
group-64 mode and a general per-tensor bit policy:

* `Int8AffineEncoder` — the INT8 twin of `Int4AffineEncoder`, deliberately the
  *same* affine convention (plain min/max grid, `scale = (max - min) / 255`,
  bias `= min`, both rounded through BF16 before the indices are computed, no
  zero-point snap). `Int8AffineEncoderConventionTests` locks that, because two
  widths disagreeing about what an affine grid means would make the comparison
  below confound a width effect with a convention effect.
* `StreamingInt8Quantizer` — the bounded-scratch component transform, plus a
  `RangeCopyTransform.quantizeInt8G64` case and its tile capacity.
* `QuantBitPolicy` — one base width plus a table of name-suffix overrides,
  matched longest-suffix-first, resolved per family. The mechanism is general
  (adding a family adds rows, not branches); the table's *contents* are
  necessarily family-specific because they mirror a particular community
  conversion's overrides.

**Its own bootstrap gate came first.** New production quantization math is not
trusted on assertion, so `Int8AffineStreamingParityTests` — the W2.1a-style
bit-parity gate — proves `Int8AffineEncoder` and the streaming path reproduce
the runtime's `Quantization.quantizeInt8Affine` reference **bit for bit**:
packed bytes, BF16 scale bits, BF16 bias bits, across single-row, wide-row and
coprime-row-count shapes and tile sizes down to one group per tile. It passes.

### Weight-level fidelity of the new INT8 encoder

The control install contains mlx-community's *actual INT8 router tensors*, so
the ideal fixture already existed. `Scripts/quantizer-weight-gate.py` — which
previously skipped these tensors as "not 4-bit on the control side" — now
quantizes the same BF16 source rows with `Int8AffineEncoder` and scores both
against the BF16 source, by the identical methodology §3-§5 used for INT4:

```
10 INT8 tensors compared (routers and shared-expert gates, layers 0/7/19/27/39)
  bit-identical to the control: 0/10
  relative Frobenius error  ours mean 0.007629 median 0.007840
  relative Frobenius error   mlx mean 0.010683 median 0.010779
  ours strictly better on 10/10; worst ratio 0.7633 on l27_segate
  max-abs error  ours better on 10/10; mean ratio 0.5769 worst 0.7840
```

Not bit-identical, exactly as at INT4 and for the same documented reason (§4).
What matters is fidelity, and **ours is better on 10 of 10**, by a wider margin
than at INT4 (ratio ~0.70-0.76 rather than ~0.99). That is the expected
direction: MLX's zero-point snap buys exact-zero representability at the cost
of step size, and that trade pays off least on router tensors, which carry no
dead-zero mass for it to redeem.

The INT4 numbers re-ran unchanged (0.096123 vs 0.096479, better on 118/124),
confirming the extension did not perturb the existing gate.

### The mixture reproduces the control exactly

`qwen36original` now installs 613 resident tensors as INT4 312 / INT8 80 /
unquantized 221 — the control's mixture, tensor for tensor.
`Scripts/quantizer-mixture-compare.py` derives each side's width from its own
resident index (`8 * sizeBytes / prod(shape)`, trusting neither installer's
manifest) and reports:

```
shared tensors: 613
width mismatches: 0
INT8 override set: control 80, ours 80
  identical
MIXTURES MATCH
```

The set is produced by a rule, not a hardcoded list: the policy overrides
`.mlp.gate.weight` and `.mlp.shared_expert_gate.weight`, so a 40-layer
checkpoint yields 80 and a checkpoint of another depth yields the right number
without the policy being edited.

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

**(A) was chosen and built** (see [What was built](#what-was-built)). It is the
only option that leaves the control experiment honest, the router table shows
the extra bits are load-bearing rather than ceremonial, and a per-tensor bit
policy is something the bring-up kit needed anyway the first time a vendor
shipped a checkpoint whose community conversion mixes widths — which is now
known to be the common case, not the exotic one.

## 7. Two defects the model-level half caught

Neither of these is a quantizer defect. Both would have shipped. Both are the
argument for running this half of the gate rather than reasoning about it.

### 7a. Resident tensor naming, found before the first run

`Model.resident(name:)` is an exact dictionary lookup — a miss is
`tensorNotFound`, with no aliasing anywhere — and each family's runner asks for
one fixed spelling. Qwen 3.6's is `Model.trunkPrefix` = `language_model.model.`
with the head at `language_model.lm_head.weight`. The pre-quantized path
inherits those names for free, because that is what mlx-lm's conversion writes.
The vendor's own repo does not agree: it ships the trunk as
`model.language_model.` and a bare top-level `lm_head.weight`.

So the first mixed-width install — which cleared `validateQuant` exactly as
intended — failed one step later with

```
no IndexEntry named language_model.model.layers.0.mlp.shared_expert.gate_proj.weight
```

**0 of 12** sampled runner-expected names were present. `FlashNextPlanner`
now renames resident tensors to the family's own spelling at install time
(`residentName(for:family:)`), which is what makes an original-repo install a
drop-in replacement for a conversion of the same checkpoint. Flash-Next is
deliberately identity: its `trunkPrefix` *is* `model.language_model.`, it has no
runner to satisfy, and renaming would change every byte of an install that
already ships.

This was invisible for as long as Flash-Next was the only original-repo family,
because Flash-Next had no runner to disagree with.

### 7b. The RMSNorm 1 + w convention, found by the first run

With the naming fixed, `qwen36original` loaded, ran at full speed, and produced
fluent, confident text. The gate's first run then reported:

```
top-1 agreement 0.000000   top-5 overlap 0.000000
KL (nats)  mean=13.2386 median=13.5546 p99=17.0182 max=17.9177
```

Zero top-1 agreement at *every one* of 882 positions, and a mean KL larger than
`ln(248320) = 12.4` — the entropy of a *uniform* distribution over the whole
vocabulary. A quantizer measured better than the control on 128 of 134 sampled
tensors cannot do that. It had to be structural.

`Scripts/gturbo-tensor-diff.py` (written for this, and kept) dequantizes every
resident tensor from both installs on disk and reports the relative difference.
It localized the defect in one pass: **100 tensors at relative difference ~1.0,
every one of them an RMSNorm gain** — unquantized BF16 passthrough, where
quantization cannot account for any difference at all. Everything else sat at
the ~0.13 that two legitimate INT4 grids produce.

The cause: Qwen 3.6 applies most of its RMSNorms as `(1 + w) * x̂`. mlx-lm's
conversion folds the `+1` into the stored weight so the kernel can just
multiply, and the Mference runtime — built against that conversion — multiplies
by the stored weight directly. The vendor's repo stores the bare `w`. Every norm
in the model was therefore off by one.

The fold is exact and its scope was measured, not guessed:

| | count | treatment |
|---|---|---|
| `input_layernorm`, `post_attention_layernorm` | 40 + 40 | fold `1 + w` |
| `self_attn.q_norm`, `self_attn.k_norm` (full-attention layers) | 10 + 10 | fold `1 + w` |
| final `norm` | 1 | fold `1 + w` |
| `linear_attn.norm` (gated DeltaNet's own) | 30 | **bare** |

Computing `bf16(w + 1)` reproduces the control's stored bytes **bit-exactly on
all 101** folded tensors, and all 30 `linear_attn.norm` tensors are already
byte-identical unfolded. `RangeCopyTransform.addOneBF16` applies it in flight;
`FlashNextPlanner.foldsNormBias` decides which tensors get it;
`NormBiasFoldTests` locks the set. After the fix the tensor diff reports **0**
tensors above 0.5 relative difference, median 0.127, max 0.177 — all pure
quantization.

**Why nothing else could have caught this.** Norms are not quantized, so the
weight-level half never samples them. `--verify-install` checks SHA-256 against
the manifest the same installer wrote. `ManifestReader` has no opinion about the
numeric convention of a passthrough tensor. The install verified, validated,
loaded, and generated fluent nonsense at full speed. Only a measurement against
an independent conversion of the same checkpoint could see it.

## 8. The model-level result

Three sequential model processes (never two at once, per AGENTS.md), each
loading one install alone and dumping full-vocab FP16 logits to disk;
comparison offline. Corpus: the three frozen prompts in
`docs/benchmark-prompts/real-generation-v1/` (`long-synthesis`,
`medium-review`, `short-explanation`, content clipped to 900 characters before
templating) plus three short raw prompts (`2 + 2 =`, `The capital of France is`,
`Three primary colors are red, blue, and`) — 882 teacher-forced positions in
total, 64 greedy tokens per prompt.

### The noise floor, first

```
=== NOISE FLOOR: D_KL(trusted || trusted-repeat) ===
  KL (nats)     n=882 mean=0 median=0 p99=0 max=0
  max |dlogit|  n=882 mean=0 median=0 p99=0 max=0
  top-1 agreement 1.000000   top-5 overlap 1.000000
  dumps byte-identical: True
```

**Exactly zero, and byte-identical.** Decode at temperature 0 is deterministic,
so every part of the signal below is attributable to the quantizer rather than
to run-to-run variation.

### Dump authenticity

The documented trap is that the default fused head skips the logits write
entirely, leaving only a greedy argmax in `lastGreedyToken`, so a harness
missing `RuntimeConfiguration(forceLogitsHead: true)` dumps never-written memory
and reports confident nonsense. The harness sets it; the comparator now
*verifies* it. Every dump is finite, non-degenerate, has a log-sum-exp within a
few nats of its max, and — for the trusted runs — an argmax matching the known
greedy token at **1.0000** of continuation positions.

### Signal

```
=== SIGNAL: D_KL(trusted || ours) ===
  prompt              pos    KL mean    KL p99   KL max   cont-only   top1    top5   max|dlogit|
  long-synthesis      296    0.399139   5.53767  13.0931  0.106566    0.8514  0.7831  17.77
  medium-review       249    0.161318   1.89319   2.93985 0.072429    0.8474  0.8217  10.09
  short-explanation   126    0.188333   2.73216   8.92035 0.098430    0.8968  0.8127  16.18
  raw-arith            69    0.084103   0.691951  1.62742 0.058405    0.8116  0.8232   7.797
  raw-capital          69    0.065858   0.722751  1.71196 0.037273    0.8986  0.8609   9.67
  raw-list             73    0.042676   0.432077  0.620617 0.028832   0.9178  0.8466   8.758

  KL (nats)       n=882  mean=0.221662  median=0.0362621  p99=3.2663   max=13.0931
  max |dlogit|    n=882  mean=3.6687    median=3.16296    p99=13.3839  max=17.7676
  top-1 agreement 0.862812   top-5 overlap 0.812698
```

### Greedy rollouts (METH-01)

| prompt | first divergence | control's top-2 margin there |
|---|---|---|
| `raw-arith` | token 1 | **0.015625** |
| `raw-capital` | token 1 | 0.2969 |
| `raw-list` | token 3 | 1.578 |
| `short-explanation` | token 6 | 1.484 |
| `medium-review` | token 19 | 0.4844 |
| `long-synthesis` | token 25 | 0.7969 |

Every divergence lands where the control is itself close to indifferent, and
all six margins are at or below the **median inter-install logit difference of
3.16** — that is, smaller than the ordinary difference between the two
quantizations. The extreme case is worth spelling out: at `2 + 2 =` the
control's top two logits are `4` at 18.0625 and `5` at 18.0469. The margin,
0.015625, is *exactly one FP16 ulp* at that magnitude — the control has no
preference at all, and which one it picks is decided by the last representable
bit. Ours picks `5` (by 1.47, itself below the median inter-install difference)
and then reasons coherently about the statement being false.

Both installs produce fluent, on-topic, factually correct text: both answer
`Paris`, both complete the primary colours with `yellow`, and both give a
substantively equivalent account of coastal wetlands. Their entropy profiles
match too — log-sum-exp-minus-max of 2.65-3.84 for ours against 1.69-3.65 for
the control — so there is no blurring, which is what a damaged model shows (the
broken install of §7b sat at a near-constant 8.1 regardless of prompt).

## 9. Verdict

* **Weight level: PASS.** Our INT4 affine group-64 quantizer reconstructs the
  BF16 source at least as faithfully as an independent trusted implementation —
  mean relative error 0.09612 against the control's 0.09648, better on 118 of
  124 sampled tensors. The one deficit (layer-0 `down_proj`, up to 1.54×) is
  fully attributed to the absence of zero-point snapping and is confined to
  tensors with a large mass of near-exact zeros.
* **INT8, weight level: PASS.** Better on 10 of 10 sampled router and
  shared-expert-gate tensors, by both relative Frobenius and max-abs error.
* **Model level: PASS.** Against a zero noise floor, the two installs agree on
  86.3 % of top-1 tokens and 81.3 % of top-5 sets across 882 teacher-forced
  positions, at a median per-position KL of 0.036 nats and a mean of 0.222 —
  a small fraction of the distribution's own entropy. Every greedy divergence
  occurs at a control-side top-2 margin below the median inter-install logit
  difference. Behaviour, factual accuracy and output entropy are preserved.

### What this does and does not claim

`D_KL(control ‖ ours)` measures how far apart two quantizations are. It does
**not** say which is closer to the BF16 original — for that you would need a
BF16 forward pass, which requires the 72 GB checkpoint resident and was not run.
The *direction* comes from the weight level, which is measured against the BF16
source directly and puts us at least on par (better on 128 of the 134 tensors
sampled across both widths).

So the model-level claim is precisely this: **40 layers of accumulated,
comparably-sized quantization error do not compound into behavioural damage.**
The two installs are different — they must be, since 0 of 124 tensors are
bit-identical and cannot be (§3) — and they are different by an amount
consistent with two independent INT4 grids of the same checkpoint driving a
top-8-of-256 router.

### The threshold, and why it is this one

The gate now has a measured failure and a measured pass on the same harness and
the same corpus:

| | broken (§7b, missing norm fold) | healthy (§8) |
|---|---|---|
| top-1 agreement | 0.000 | 0.863 |
| KL mean / median (nats) | 13.24 / 13.55 | 0.222 / 0.036 |
| top-5 overlap | 0.000 | 0.813 |
| max \|Δlogit\| mean | 23.0 | 3.67 |

**Proposed gate: noise floor exactly zero (byte-identical repeat dumps), top-1
agreement ≥ 0.50, and median per-position KL ≤ 0.50 nats.**

These sit between two *measured* populations rather than being fitted to the
observed pass: they are ~1.7× away from the healthy point and ~14× away from the
broken one on the KL axis, and equidistant in the wrong direction on top-1. A
threshold drawn to just-clear 0.863 would be tuning; one drawn at 0.50 would
still have caught the only real defect this harness has ever seen, with room to
spare, and would not fail a healthy install that happened to be somewhat noisier
than this one.

The zero-noise-floor requirement is not a quality threshold at all — it is a
correctness precondition. If temperature-0 decode is not deterministic, every
number above is uninterpretable and the run must be investigated rather than
scored.

**Calibration is thin and should be widened.** This is one healthy sample and
one broken sample, on one checkpoint. The second original-repo family to reach
a runner should re-run this gate and the thresholds revisited with two healthy
points rather than one.

### Flash-Next

Closing W2.1b does **not** by itself lift `ManifestReader.familiesWithoutRunner`.
That gate refuses Flash-Next for missing **axes**
(`hyperConnectionsLowRank`, `attentionIndexer`, `pleNgramEmbedding`), not for
quantizer quality. What W2.1b's closure does do is remove the *quality*
objection: the quantizer nucleus Flash-Next shares is now validated at both the
weight and model level. The remaining blockers to an eventual lift are the axes
and a maintainer decision, not this gate.

Two caveats are worth carrying into that decision. Flash-Next is uniform INT4
by policy and folds no norm bias — both correct for it today, and both now
*explicit* per-family choices rather than defaults nobody had examined. Neither
has been checked against an independent conversion of Flash-Next, because none
exists. §7 is the record of what that kind of unchecked assumption costs.
