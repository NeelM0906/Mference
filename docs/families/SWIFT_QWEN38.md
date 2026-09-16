# Swift-Qwen3.8 27B: qualification candidate

Swift-Qwen is an optional, separately identified checkpoint using the existing
dense `qwen38` runner. It does **not** replace the base Qwen installation or
default recommendation until the model-level gates below pass.

## Pinned source and conversion

- Source: [ukisai/Swift-Qwen3.8-27b](https://huggingface.co/ukisai/Swift-Qwen3.8-27b/tree/1b30aaaf753fe5c1cb51ada2ea0367a53445359c).
- Revision: `1b30aaaf753fe5c1cb51ada2ea0367a53445359c`.
- Source index SHA-256: `77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df`.
- Source template SHA-256: `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`.
- Installer selector: `swiftqwen38`; manifest/API ID: `swift-qwen3.8-27b-int4g64`.
- Source model card identifies Swift Open License 1.0; consult its terms before
  use or redistribution. This project does not redistribute weights.

The 64-layer architecture has hidden size 5120, dense intermediate size 17408,
48 Gated-DeltaNet blocks, 16 full-attention blocks, and a 248320-row vocabulary.
No routed expert pool is needed. The importer streams BF16 ranges into affine
INT4 group-64 projections, embeddings and output head; no full source snapshot
is staged. Norms, convolution and auxiliary vectors retain source precision.
All 15 included MTP tensors come from this same source, not the base model.

Conversion follows the Qwen3.5 RMSNorm convention: add one to 161 trunk norms
and seven MTP norms, **not** the Gated-DeltaNet gated norm. Original convolution
metadata changes from `[channels, 1, 4]` to `[channels, 4, 1]`; contiguous bytes
are unchanged. The independent reference implementations are
[Transformers Qwen3.5](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
and [MLX-LM Qwen3.5](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py).
These establish the conversion contract, not full-model numerical parity.

Metadata-only planning on 2026-09-16 reported 54,641,395,712 source payload bytes,
15,371,847,680 planned output bytes, 866 resident entries, and 333 excluded vision
tensors. There are 506 INT4 entries and 360 unquantized entries. Disk estimates
in the installer include headroom. Dry-run's 64 expert-layer records are empty
dense layout records, not 64 routed expert pools.

## Install and request policy

Apply the repository's disk/memory/process checks before installation or runs.
Keep the existing `qwen38.gturbo` as the control. Inspect the plan first:

```bash
swift run -c release MferenceRepack --model swiftqwen38 --output scratch/swiftqwen38.gturbo --dry-run
./mference-ui.sh install swiftqwen38
```

The CLI accepts `--reasoning-effort xhigh|medium|low|none` with `--chat` or
`--messages-file`; raw `--prompt` is intentionally not template-rendered and
rejects that option. The server accepts the same values in `reasoning_effort`.
Omitting the field uses the pinned source's `xhigh` default. `none` closes the
thinking block in the prompt. `medium` adds no effort instruction. Tools use
the same policy and do not silently disable thinking.

The installed Jinja template renders both ordinary and tool conversations.
Leading `system` guidance is supported; `developer` is explicitly rejected,
matching the source's role contract. Assistant history may include
`reasoning_content`. Responses return generated reasoning in that separate
field (or SSE delta), never mixed into `content`. CLI interactive history retains
it without printing it as the answer. Preserve the field in client tool loops.
Generation stops at `<|im_end|>` or `<|endoftext|>`.

Open WebUI sees the distinct API ID through library discovery. Where the client
can forward a custom request parameter, use `reasoning_effort` with the exact
values above; an omitted parameter uses `xhigh`. End-to-end UI forwarding and
reasoning-history persistence still require qualification; no UI behavior is
assumed from server unit tests. Base and Swift release their previous resident
session when switching. Swift only reuses an exactly matching rendered prefix;
it does not use the legacy hand-written continuation bridge.

MTP is disabled by default for this candidate. `MFERENCE_MTP=1` explicitly
enables it for qualification; the base checkpoint's existing default is unchanged.

## Qualification gates

Implemented checks: pinned source metadata/dry run; tiny synthetic remote
range install including own MTP; norm payload folds and convolution metadata;
independent Python Jinja2 oracle versus Swift rendering across four modes and
multi-turn/tool-result history; request/response reasoning separation; distinct
discovery IDs, model lifecycle and cache-domain isolation.

Initial real-model evidence on 2026-09-16 (M3 Ultra, 256 GiB, macOS 26.3,
Swift 6.3.3; Phase 2 changes based on `3247c3d`):

- Streamed install completed; strict verification passed for seven files,
  15,384,698,420 bytes. Base Qwen was preserved and both IDs were discovered.
- All 37 sampled source conversions matched installed bytes. Of 29 sampled
  projections, 27 had lower reconstruction error than MLX INT4; embedding
  and MTP FC had approximately 19.8% and 30.8% higher relative error. These
  eight-row samples do not establish whole-model quality.
- Independent MLX execution on the installed weights agreed at 713/714
  full-vocabulary top-1 positions (99.86%), with mean KL 0.0002701 nats and
  maximum KL 0.0730. All six 16-token greedy continuations matched. This is
  short-corpus runtime evidence, not bit-exact parity or a BF16 quality gate.
- Plain chat returned the expected answer; a 51-token MTP-on/off smoke test
  produced identical output. MTP remains disabled by default.
- All four reasoning modes returned valid live HTTP responses with reasoning
  separate from the answer; `none` emitted no reasoning field in that test.
- Live XML tool-call/result round trip passed with reasoning history retained
  and 358 exact-prefix tokens reused. Live SSE kept reasoning separate from
  visible content and completed normally. Thinking-open prompts explicitly
  select Qwen XML tool parsing for this checkpoint, not Maple's JSON parser.

Still required before promotion:

1. Broader same-Swift logits/state and long-context family acceptance checks.
2. End-to-end Open WebUI parameter forwarding and reasoning-history checks.
3. MTP state parity, acceptance and latency per supported hardware profile.
4. Base-versus-Swift task quality, token usage and latency at matched policies,
   including quantization-quality impact. Base Qwen is a task-quality control,
   not a numerical reference for different fine-tuned weights.

Until these pass, do not claim token savings, quality preservation or a faster
runtime from the source model's name/card, and do not promote it to the default.

### Independent runtime comparison

`Scripts/swift_qwen_mlx_reference.py` uses pinned MLX-LM 0.31.3 / MLX 0.32.2
directly on the installed resident tensor payload. It does not download or
write another checkpoint, does not call Mference kernels, and deliberately
does not sanitize already-converted norms a second time. It excludes MTP.

After safety checks and strict install verification, run the existing
`QuantizerQualityMeasurement` suite with `MFERENCE_QUANT_QUALITY_GTURBO` pointing
to Swift, a fresh `MFERENCE_QUANT_QUALITY_DUMP` directory, and label `native`.
Then, only after that process exits:

```bash
uv run Scripts/swift_qwen_mlx_reference.py scratch/swiftqwen38.gturbo /path/to/dump/native
```

The checker teacher-forces the identical native token sequences and reports
full-vocabulary KL divergence, top-1 agreement and maximum absolute logit error.
This isolates runtime implementation differences on the same installed weights.
It also checks that native teacher-forced logits reproduce the recorded greedy
continuation, then reports the reference's matching greedy prefix on that
sequence. It is **not** an independent BF16 conversion-quality measurement,
a base-versus-Swift task evaluation, or a performance benchmark.

For the separate source-row gate:

```bash
uv run Scripts/swift_qwen_source_gate.py scratch/swiftqwen38.gturbo --cache /path/to/source-samples
```

This samples eight rows per projection across embeddings, head, MTP FC, early,
middle and final layers, plus norm/convolution/FP32 vectors. It checks installed
bytes against the pinned BF16 source's expected conversion and compares row
reconstruction error with MLX's independent INT4 group-64 quantizer. Exact HTTP
206 ranges and byte counts are required; a full-shard response is rejected.
This is sampled weight evidence, not a replacement for model/task evaluation.
