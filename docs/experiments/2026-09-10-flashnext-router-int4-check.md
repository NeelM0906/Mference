# Flash-Next: does the INT4 router change expert selection?

**Date:** 2026-09-10. **Verdict: yes, materially — recommendation (B), reinstall
with the two gating tensors at INT8** (see [§6](#6-verdict)). Measured on the
installed bytes, not on a model of them; no model process was started.

## 1. Question

The install at `scratch/qwen38flashnext.gturbo` (163 GB, `modelID
qwen3.8-flash-next-int4g64`, source `Qwen/Qwen3.8-Flash-Next` rev
`de4b8e4d43b9…`, `bitWidthOverridesHonored: 0`) quantizes **every** eligible
tensor at INT4 affine group-64 under `QuantBitPolicy.uniformInt4`, including the
MoE router `model.language_model.layers.N.mlp.gate.weight` (`[512, 2560]`,
top-10 of 512) and the shared-expert gate `mlp.shared_expert_gate.weight`
(`[1, 2560]`). The independent conversion `mlx-community/Qwen3.8-Flash-Next-4bit`
(landed 2026-09-02) quantizes everything else at INT4 **group-32** but keeps
exactly those two suffixes at **INT8 group-64**: its `config.json`
`quantization` block carries 224 per-tensor entries, of which **96** are
`{bits: 8, group_size: 64}` — 48 layers × {`mlp.gate`, `mlp.shared_expert_gate`}
(the other 128 are the PLE n-gram shards, restated at the 4-bit g32 base). The
task brief said 192; the count from the control's own config is 96.

For Qwen 3.6, §6 of [docs/QUANTIZER_QUALITY.md](../QUANTIZER_QUALITY.md)
found an INT4 router swapping 13–18 % of the top-8 set and ~25 % of the top-1
expert, which is why `QuantBitPolicy.moeRouterInt8` exists for that family.
Flash-Next was left at uniform INT4 because "its runner drives the router
through the generic INT4 matvec" and no control existed. Now one does.

**Decision rule, fixed before any number was read:** recommend (B) — reinstall
with `mlp.gate` / `mlp.shared_expert_gate` at INT8 — if ours-vs-BF16 exact
top-10 agreement is below the control's by more than 5 points absolute, **or**
if ours-vs-BF16 top-1 agreement is below 0.97, on Gaussian probes. Otherwise
(A), INT4 routers are fine.

## 2. Method

Scripts, both Python/numpy, both committed with this note:

* [`Scripts/quantizer-weight-gate.py`](../../Scripts/quantizer-weight-gate.py)
  was generalized: `--family {qwen36,qwen38flashnext}`, `--orig REPO[@REV]`,
  `--control REPO[@REV]`, `--tensors REGEX` (sample any vendor tensors the
  control also quantizes, full rows or `--rows N`), control group size derived
  from the `.scales` shape (so a g32 control is compared correctly and its byte
  columns flagged as non-comparable), and 429/5xx backoff on the CDN. **The
  default is unchanged and re-run:** the Qwen 3.6 plan reproduces §4/§5 of
  QUANTIZER_QUALITY.md exactly — INT4 0.096123 vs 0.096479 (better on 118/124),
  INT8 0.007629 vs 0.010683 (10/10). The g32 path was exercised once on
  `layers.23.self_attn.q_proj` (64 rows): ours g64 0.1055 vs the control's g32
  0.0899, the expected finer-grid advantage, byte columns correctly flagged.
* [`Scripts/flashnext-router-int4-check.py`](../../Scripts/flashnext-router-int4-check.py)
  (new) imports the gate's `Repo`/decoders and does the measurement below.

**Sources.** BF16 reference rows by HTTP range request from
`https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/de4b8e4d/<shard>`
(safetensors header parse → byte range; 2.6 MB per router). Control tensors the
same way from `mlx-community/Qwen3.8-Flash-Next-4bit@main`: MLX affine, `.weight`
U32 `[512, 640]` (four INT8 codes per word, element *i* at byte *i*), `.scales`
and `.biases` BF16 `[512, 40]`, decoded as `s·q + b`. Total download ≈ 48 MB;
one transient HTTP 000 on a control shard, retried once — no throttling, so the
full 12-layer sample was kept.

**Our side.** Read directly from the install's `model_weights.bin`: the
`ResidentIndexReader` layout (24-byte header, 72-byte entries, string table),
dtype 0 = INT4 affine g64, packed nibbles at `fileOffset` (low nibble = even
index, row-major), BF16 scales/biases at `scaleOffset`/`biasOffset`
(`Int4AffineEncoder.encodeTensor` companion layout). **Decode proof:** for each
sampled tensor the BF16 source was re-encoded with the transcribed
`Int4AffineEncoder.encodeGroup` and compared byte-for-byte with the install.
21 of 24 tensors are bit-identical. The other three (routers of layers 0, 4, 9)
differ in **2 nibbles each out of 1,310,720**, every one at an exact rounding
tie (`(w − b)/s = 0.5` in float32) where the install holds the lower index and
the numpy transcription the upper — 1.5 × 10⁻⁶ of the tensor, one quantization
step, no effect on any number below. It is noted here because
`Int4AffineEncoderConventionTests` claims round-half-away-from-zero; the Swift
side may want to look at how those ties are evaluated in the streaming path.

**Reconstruction.** Relative Frobenius error `‖Ŵ − W‖_F / ‖W‖_F` and max-abs
error against the BF16 source, ours (INT4 g64) and control (INT8 g64).

**Routing agreement.** N = 4096 probes `x ~ N(0, I₂₅₆₀)` (seed 20260910),
float32. Router math per the family spec
(`docs/superpowers/specs/2026-09-01-qwen38flashnext-runtime-design.md`, "MoE")
and `FlashNextMoE.encodeRouterSelect`: `probs = softmax(W·x)`, top-10 of the
probs, renormalize (`norm_topk_prob`); no `routed_scaling_factor`, no
sigmoid/group scoring. Softmax is monotone, so the selected set is the top-10 of
the logits. For each probe the set is computed from BF16, ours and control
logits; reported are exact top-10 set agreement, mean Jaccard, top-1 agreement,
mean number of swapped experts (10 − |∩|), and `D_KL(BF16 ‖ X)` between the
renormalized weights **over the BF16-selected set** (what weights router X
would hand the experts the BF16 router chose — well defined even when the sets
differ). The shared-expert gate is scored as mean and max `|σ(ŵ·x) − σ(w·x)|`.

**Probe set (b), real hidden states: skipped — none exist.** `scratch/w21b-dump/`
holds Qwen 3.6 full-vocabulary *logit* dumps (`qwen36.gturbo` /
`qwen36-ourquant.gturbo`), not Flash-Next router inputs, and no other
activation dump was found under `scratch/`. The script takes `--activations
FILE` (raw float32 `[N, 2560]`) so the run can be repeated the day one exists.

**Layers.** 12 of the 48 MoE layers (every layer is MoE in this family):
0, 4, 9, 13, 17, 22, 26, 30, 35, 39, 43, 47.

Exact commands (from the worktree root; `$C` is any scratch cache dir):

```bash
python3 Scripts/quantizer-weight-gate.py --cache $C/gate-qwen36          # §4/§5 reproduction
python3 Scripts/flashnext-router-int4-check.py --cache $C/fn-router \
    --install /Users/studio2/Documents/ChatGPT/Mference/scratch/qwen38flashnext.gturbo \
    --json $C/fn-router/results.json
python3 Scripts/quantizer-weight-gate.py --cache $C/gate-fn --family qwen38flashnext \
    --tensors 'layers\.(0|22|47)\.mlp\.(gate|shared_expert_gate)\.weight'   # our INT8 encoder vs control
```

## 3. Per-layer results — router `mlp.gate.weight`

Reconstruction against BF16 (ours = install bytes, INT4 g64; control = INT8
g64), then routing agreement of each against the BF16 router on the same 4096
Gaussian probes. "ref margin@10" is the median gap between the BF16 router's
10th and 11th logits — how close the boundary is under these probes.

| Layer | rel. Frob. ours | control | max-abs ours | control | top-10 exact ours | control | Jaccard ours | control | top-1 ours | control | swapped ours | control | KL ours | control | ref margin@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.1417 | 0.0124 | 0.1024 | 0.0103 | 0.116 | 0.837 | 0.750 | 0.970 | 0.783 | 0.981 | 1.50 | 0.17 | 4.6e-02 | 3.8e-04 | 0.047 |
| 4 | 0.1141 | 0.0100 | 0.0649 | 0.0120 | 0.128 | 0.853 | 0.764 | 0.973 | 0.780 | 0.975 | 1.40 | 0.15 | 9.3e-03 | 1.6e-04 | 0.023 |
| 9 | 0.1109 | 0.0095 | 0.0723 | 0.0071 | 0.128 | 0.849 | 0.766 | 0.972 | 0.787 | 0.981 | 1.39 | 0.15 | 9.6e-03 | 8.7e-05 | 0.025 |
| 13 | 0.1090 | 0.0090 | 0.0687 | 0.0108 | 0.143 | 0.866 | 0.772 | 0.975 | 0.770 | 0.976 | 1.35 | 0.14 | 1.2e-02 | 1.3e-04 | 0.027 |
| 17 | 0.1051 | 0.0085 | 0.0278 | 0.0035 | 0.128 | 0.860 | 0.765 | 0.974 | 0.787 | 0.982 | 1.39 | 0.14 | 5.5e-03 | 3.7e-05 | 0.024 |
| 22 | 0.1185 | 0.0096 | 0.0582 | 0.0063 | 0.103 | 0.855 | 0.743 | 0.973 | 0.742 | 0.978 | 1.55 | 0.15 | 9.0e-03 | 5.9e-05 | 0.022 |
| 26 | 0.1141 | 0.0093 | 0.0852 | 0.0112 | 0.152 | 0.863 | 0.771 | 0.975 | 0.769 | 0.974 | 1.36 | 0.14 | 2.4e-02 | 1.8e-04 | 0.035 |
| 30 | 0.1195 | 0.0104 | 0.0780 | 0.0122 | 0.148 | 0.843 | 0.765 | 0.971 | 0.723 | 0.968 | 1.41 | 0.16 | 1.8e-02 | 2.2e-04 | 0.031 |
| 35 | 0.1046 | 0.0085 | 0.0170 | 0.0015 | 0.122 | 0.853 | 0.760 | 0.973 | 0.783 | 0.980 | 1.43 | 0.15 | 4.1e-03 | 2.7e-05 | 0.019 |
| 39 | 0.1019 | 0.0083 | 0.0472 | 0.0055 | 0.135 | 0.854 | 0.769 | 0.973 | 0.782 | 0.977 | 1.37 | 0.15 | 3.3e-03 | 3.2e-05 | 0.015 |
| 43 | 0.1040 | 0.0084 | 0.0123 | 0.0014 | 0.117 | 0.853 | 0.761 | 0.973 | 0.798 | 0.981 | 1.42 | 0.15 | 2.3e-03 | 1.5e-05 | 0.015 |
| 47 | 0.1097 | 0.0090 | 0.0375 | 0.0057 | 0.161 | 0.876 | 0.786 | 0.977 | 0.791 | 0.977 | 1.26 | 0.12 | 3.9e-03 | 3.2e-05 | 0.018 |

## 4. Per-layer results — shared-expert gate `mlp.shared_expert_gate.weight`

| Layer | rel. Frob. ours | control | max-abs ours | control | mean \|Δσ\| ours | control | max \|Δσ\| ours | control |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.1329 | 0.0105 | 0.00284 | 0.00031 | 0.0078 | 0.00063 | 0.0364 | 0.00342 |
| 4 | 0.1131 | 0.0086 | 0.00189 | 0.00019 | 0.0079 | 0.00059 | 0.0387 | 0.00314 |
| 9 | 0.1083 | 0.0081 | 0.00281 | 0.00024 | 0.0085 | 0.00063 | 0.0393 | 0.00317 |
| 13 | 0.0993 | 0.0079 | 0.00166 | 0.00019 | 0.0067 | 0.00055 | 0.0289 | 0.00278 |
| 17 | 0.1069 | 0.0086 | 0.00208 | 0.00024 | 0.0076 | 0.00062 | 0.0349 | 0.00286 |
| 22 | 0.1022 | 0.0078 | 0.00160 | 0.00021 | 0.0063 | 0.00049 | 0.0339 | 0.00221 |
| 26 | 0.0964 | 0.0074 | 0.00171 | 0.00016 | 0.0063 | 0.00051 | 0.0295 | 0.00274 |
| 30 | 0.1078 | 0.0088 | 0.00166 | 0.00019 | 0.0063 | 0.00052 | 0.0290 | 0.00242 |
| 35 | 0.1067 | 0.0077 | 0.00174 | 0.00018 | 0.0062 | 0.00044 | 0.0286 | 0.00194 |
| 39 | 0.1199 | 0.0093 | 0.00206 | 0.00021 | 0.0066 | 0.00052 | 0.0344 | 0.00257 |
| 43 | 0.1147 | 0.0094 | 0.00276 | 0.00028 | 0.0066 | 0.00054 | 0.0359 | 0.00242 |
| 47 | 0.1094 | 0.0087 | 0.00102 | 0.00012 | 0.0040 | 0.00032 | 0.0213 | 0.00160 |

## 5. Aggregates (12 layers, 4096 Gaussian probes each)

| Metric | ours (INT4 g64, install bytes) | control (INT8 g64) | ratio |
|---|---|---|---|
| router rel. Frobenius error, mean | 0.1128 | 0.0094 | 12.0× |
| router max-abs error, mean | 0.0560 | 0.0073 | 7.7× |
| shared-expert-gate rel. Frobenius error, mean | 0.1098 | 0.0086 | 12.8× |
| **top-10 exact set agreement vs BF16** | **0.132** [0.103, 0.161] | **0.855** [0.837, 0.876] | −72 points |
| mean Jaccard vs BF16 | 0.764 | 0.974 | |
| **top-1 agreement vs BF16** | **0.774** [0.723, 0.798] | **0.977** [0.968, 0.982] | |
| swapped experts per token (of 10) | 1.40 (14.0 %) | 0.146 (1.5 %) | 9.6× |
| KL(BF16 ‖ X) over BF16's set, mean / p99 (nats) | 0.0123 / 0.070 | 0.0001 / 0.0009 | ~120× |
| shared-expert gate, mean \|Δσ\| | 0.0067 | 0.00053 | 12.7× |

Two side results from the generalized gate, run on the same tensors with
`--tensors`: **our INT8 encoder** re-encoding these routers and gates from BF16
lands at 0.0070 mean relative error against the control's stored 0.0097 (better
on 6/6, max-abs ratio 0.52) — so an INT8 reinstall would be at least as
faithful as the control on exactly these tensors, matching the Qwen 3.6 §5
result. And the vendor's `mtp.layers.0.mlp.gate` / `shared_expert_gate` are
carried **unquantized** by the control (no `.scales`), where our install
quantizes the MTP sidecar's copies at INT4 too (`sidecars.mtp.carried: true`).

### Norm-fold byte check (task item 4)

18 zero-centered RMSNorm gains from `FlashNextResident.zeroCenteredNormSuffixes`
(`hyper_connection_mixer.hc_norm`; PLE `norm_key`/`norm_query`/`norm_conv` on
layer 1; `q_norm`, `k_norm`, `indexer.q_layernorm`, `indexer.k_layernorm`,
`attn_hyper_connection.hc_norm`, `mlp_hyper_connection.hc_norm` on layers 3
and 19; the two `hc_norm`s on layer 40), BF16 on all three sides:

| Comparison | count |
|---|---|
| install bytes == vendor bare `w` | **18/18** |
| control bytes == vendor bare `w` | **18/18** |
| control bytes == `bf16(1 + w)` | 0/18 |

So the control stores the **bare** zero-centered `w` too — it did *not* fold
`1 + w` the way mlx-community's Qwen 3.6 conversion did (§7b of
QUANTIZER_QUALITY.md) — and our install copies the vendor bytes verbatim,
leaving `Model.normWeight` to apply `(1 + w)` at load (`zeroCenteredNormPolicy
== .bakeAtLoad`, manifest has no `zeroCenteredNormsBakedAtInstall`). This
agrees with the family doc's expectation. The "148/148 bit-identical" figure
quoted in the brief could not be located in the repo's docs (the Qwen 3.6 fold
work reports 101 folded + 30 bare = 131); the number measured here is 18/18 on
the sampled set, and it is the control that must apply `(1 + w)` itself. Note
`hyper_connection_mixer.hc_norm` has `|w|max = 12.4`, so a missed fold there
would be an order-one error, not a subtle one.

## 6. Verdict

**(B): reinstall with the two gating suffixes at INT8.** Both halves of the
pre-stated rule fire, and not marginally: ours-vs-BF16 exact top-10 agreement is
0.132 against the control's 0.855 (72 points below, threshold 5), and top-1
agreement is 0.774 (threshold 0.97). On isotropic probes the INT4 router swaps
1.4 of the 10 selected experts per token and changes the top-1 expert 22.6 % of
the time, where INT8 swaps 0.15 and changes top-1 2.3 % — the same 14 % vs 1.5 %
ratio the Qwen 3.6 §6 measurement found for top-8-of-256, now on the installed
bytes of this family. The weight-space cause is unambiguous (12× the relative
error of INT8 at 12 of 12 layers), and the shared-expert gate drifts its sigmoid
by 0.0067 mean / 0.039 max against INT8's 0.0005 / 0.003.

*Isotropic-probe caveat, stated honestly:* real router inputs are not
`N(0, I)`. Under these probes the BF16 top-10 boundary is very tight (median
margin 0.025 logits, table above), which inflates *both* sides' disagreement —
even the INT8 control gets the exact set right only 85.5 % of the time. On real
hidden states the routers are peakier and the absolute agreement rates will be
higher for both. What the probes do establish robustly is the *ratio*: INT4
routing noise is ~10× INT8 routing noise at every layer, and the ~13–18 % of
swapped experts on Qwen 3.6 (same probe design) was corroborated there by the
model-level KL run. A real-activation re-run (`--activations`) should follow
once a Flash-Next hidden-state dump exists; it can only tighten the numbers,
not change the ordering.

**What (B) takes, from the code as it stands** (no CLI flag exists; the policy
is per family):

1. `Sources/MferenceRepack/Core/Quantization/QuantBitPolicy.swift`,
   `originalRepo(family:)`: `case .qwen38flashnext: return .uniformInt4` →
   `.moeRouterInt8` (rules `.mlp.gate.weight` and
   `.mlp.shared_expert_gate.weight` at 8 bits; `bitWidthOverridesHonored` would
   then read 96, equal to the control's count — plus the two MTP-sidecar copies
   if they are kept quantized). The repacker has streamed INT8 affine g64 since
   commit `1b04014` (`Int8AffineEncoder`, `StreamingInt8Quantizer`), and
   `MferenceRepack --model qwen38flashnext --output … --overwrite` reports the
   width census and `Bit-width overrides: N`.
2. `Sources/Mference/Infrastructure/ModelIO/ManifestReader.swift:433–455`
   (`validateQuant`, the `.qwen38flashnext` branch) requires `weightBits == 4`
   on every slot including `router`; it must accept 8 there, as the generic
   branch already does (`("router", quant.router, [8])`).
3. `Sources/Mference/Kernels/FlashNext/FlashNextMatVec.swift:19–31`
   (`FlashNextWeightMatrix.from`) accepts only dtype 0 (INT4 affine) and 1
   (BF16). The router and shared gate go through `matVec.encode` in
   `FlashNextForwardRunner.swift:770` and `:999`, so an INT8 case (or a
   dtype-1 BF16 passthrough for these 96 tensors — 48 × 2.6 MB = 126 MB, which
   the existing BF16 matvec already handles and which needs no new kernel) is
   required before an INT8 install can load. The Swift side should pick between
   INT8 (mirrors the control byte-for-byte on the mixture) and BF16 passthrough
   (zero router error, no kernel work); this note only establishes that INT4
   is not acceptable.

Until then, the current install's routing differs from the BF16 model's on
roughly one expert in seven per token; any quality comparison against the
mlx-community conversion made with it will be measuring routing divergence
rather than the weight quantizer, exactly the failure mode QUANTIZER_QUALITY.md
§6 describes.

## Result on the INT8-router install (2026-09-10, `qwen38flashnext-r8.gturbo`)

The reinstall recommended above was run the same day: `MferenceRepack --model
qwen38flashnext --output scratch/qwen38flashnext-r8.gturbo` (2 h 44 m, 360 GB
streamed, no throttling), landing 175,205,477,629 verified bytes in 57 files —
102 bytes off the prediction, all manifest. The manifest reports
`quant.router.weightBits 8`, `quantizedAtInstall.overriddenTensorCount 98`
(48 text routers + 48 shared-expert gates + the MTP layer's pair). The script
was taught to decode either width from the resident index (dtype 0 at both
widths; width derived from the byte size, as the runtime does) and re-run with
the same 12 layers, 4,096 Gaussian probes, seed and cache:

| Metric (12 layers, Gaussian probes) | INT4 install (before) | **INT8 install (now)** | control (mlx-community INT8 g64) |
|---|---:|---:|---:|
| install bytes == re-encoded BF16 | 21/24 (3 rounding ties) | **24/24** | — |
| router rel. Frobenius error vs BF16 | 0.1128 | **0.00661** | 0.00940 |
| router max-abs error | — | **0.00384** | 0.00729 |
| exact top-10 set agreement vs BF16 | 0.132 | **0.896** [0.887, 0.912] | 0.855 [0.837, 0.877] |
| mean Jaccard | 0.764 | **0.981** | 0.974 |
| top-1 agreement | 0.774 | **0.986** | 0.977 |
| swapped experts per token (of 10) | 1.40 | **0.104** | 0.146 |
| KL(BF16‖X) over the selected set, mean / p99 | 0.0123 / 0.070 | **0.0000 / 0.0002** | 0.0001 / 0.0009 |
| shared-expert gate mean \|Δsigmoid\| | 0.0067 | **0.00040** | 0.00053 |

Verdict: the routing deficit is closed. Our INT8 group-64 routers reconstruct
the BF16 source more faithfully than the control's (0.70× the relative error,
better on every sampled layer, consistent with the Qwen 3.6 INT8 result in
`docs/QUANTIZER_QUALITY.md`) and select the same experts as BF16 more often
than the control does. The norm-fold byte check is unchanged: 18/18 install
bytes are the vendor's bare `w`, as are the control's; the `(1 + w)` bake at
load remains the correct treatment. The uniform-INT4 install is kept alongside
as the "before" side of this table; nothing else about the two installs
differs (same 57 files, expert stride and PLE pool; +32,175,360 weight bytes).

```bash
python3 Scripts/flashnext-router-int4-check.py --cache $C/fn-router \
    --install scratch/qwen38flashnext-r8.gturbo --json r8-router-check.json
```
