# Maple behavioral parity contract

This document is the frozen contract for reimplementing Maple on the clean
Mference integration branch. It records what the reference inputs and research
branch say should happen; it does **not** certify that the clean port currently
does it. Only a new clean run of the validation matrix below can become
acceptance evidence. Implementation status lives in [PROGRESS.md](../PROGRESS.md).

## Reference pins

These pins are inputs to parity, not moving recommendations.

| Reference | Frozen value |
| --- | --- |
| Checkpoint | [`deepgrove/maple-preview-2bit-mlx`](https://huggingface.co/deepgrove/maple-preview-2bit-mlx) |
| Checkpoint revision | `361db5da5e74ff6fcdd852d478e1f266ce11013a` |
| Source index SHA-256 | `56000110535c5023b43209a5c142035e12c1cde7b1118759cc9f86335d46ef95` |
| Snapshot-manifest SHA-256 | `ac8b6d4b118d982b215c98697cca50ebe770ad3d8f68b7ef10a582fd52fb9a5c` |
| `config.json` SHA-256 | `57eb521da63629196ebda2c103be929c81c1027ddf2766e7b19e2d2427f77443` |
| Executable reference `maple.py` SHA-256 | `4685745e9cba8802e578b833504213efce21bec9b45d0550d6040340f367f007` |
| `tokenizer.json` SHA-256 | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| `tokenizer_config.json` SHA-256 | `bcab9fe0f4f56a0cffba7c916f943c376810d32c08f2cebf01f8b15849039d35` |
| `chat_template.jinja` SHA-256 | `83e4c58ca602ade89b126cc75a036eb8bd06d373d4fd94b09d2277d609131089` |
| MLX runtime fork | [`deepgrove-ai/mlx-lm-deepgrove`](https://github.com/deepgrove-ai/mlx-lm-deepgrove/tree/eba96c16158f032821b0bf374ea1421cfddef0a9) at `eba96c16158f032821b0bf374ea1421cfddef0a9` |
| Oracle environment | CPython 3.12.13; MLX 0.32.0; mlx-lm 0.31.3; Transformers 5.14.1; Tokenizers 0.22.2; NumPy 2.5.1 |
| Research implementation | `codex/maple-integration` at `997b5ade1d09ead4f9647628cebb34b402f5b015` |

The full checkpoint contains three large safetensors shards. A clean parity run
must validate their pinned Hugging Face LFS size/object metadata and perform one
explicit full-SHA audit before using a trusted local oracle snapshot. Per-run
preflight may validate immutable metadata without rereading 5+ GB immediately
before only one engine, because that would asymmetrically warm its cache.

## Architecture and model schema

The accepted `maple` family is exactly:

| Field | Contract |
| --- | --- |
| Layers / hidden size | 24 / 2,048 |
| Attention | 16 query heads, 4 KV heads, head dimension 128, scale `1/sqrt(128)` |
| Layer schedule | Repeating three sliding layers then one full layer; full layers are 3, 7, 11, 15, 19, 23 |
| Position | Sliding window 512; partial NeoX RoPE over dimensions `0..<64`, theta 10,000; full layers are NoPE |
| MoE | 256 routed experts, top 8, expert width 512, no dense-leading layers, no shared expert |
| Router | Raw BF16 weights, FP32 dot/selection, softmax scoring, normalize selected weights after top-k |
| Expert activation | SiLU/SwiGLU with gate capped at `7` and up value clamped to `[-7, 7]` |
| Vocabulary / head | 151,936 rows, untied embedding and head, no embedding `sqrt(hidden)` scale, no logit softcap |
| Norm/residual | Plain pre-norm block, epsilon `1e-6`, native-BF16 residual boundaries |

`config.json` must be parsed as a flat `model_type: "maple"` configuration and
cross-checked field by field. At minimum it must require Q/K norm, FP32 router
semantics, normalized top-k probability, no leading dense layer, and the exact
production shape above. The manifest must explicitly publish the family,
architecture, absent shared expert, router representation, and all quantization
slots; defaults must not silently turn a malformed Maple install into Gemma.

## Repack and install contract

The clean installer uses the existing `.gturbo` topology and bounded remote
range workflow:

- selector `--model maple`, model ID `maple-preview-2bit-mlx`, conventional
  directory `maple.gturbo`;
- pinned repository, revision, and source-index digest above;
- embedding, attention, and exact head represented as affine INT4/group-64;
- routed gate/up/down weights kept packed INT2/group-64;
- router kept raw BF16, with `sharedExpert` explicitly absent;
- approximate `lm_head_flash.*` tensors classified and excluded;
- no full-snapshot staging and no unbounded transform buffer.

For a source ternary row, packed two-bit codes are widened losslessly where the
resident INT4 kernels require it. Each BF16 `row_alpha` expands across the
logical group-64 companions: scale is `alpha`, bias is `-alpha`. Routed expert
codes stay two-bit and receive the same group-64 companion semantics. Shapes,
packing-unit alignment, source/destination sizes, transform identity, and
destination intervals must be validated before transfer.

Transformed copy plans and completed-range digests must bind the transform and
destination size. Cancellation and resume must revalidate transformed bytes
and avoid redownloading intact ranges. Identity-only plan/digest serialization
must remain compatible with existing four-family partial installs. Final
verification hashes the promoted payload. A verification pass may preserve
source provenance only from an existing receipt already bound to the exact
directory, manifest, and complete file set.

## Numerical forward-pass contract

The observable arbiter is the pinned MLX execution, not the research branch's
class/file boundaries. A clean implementation may organize kernels differently
if it reproduces these boundaries:

1. Embedding lookup emits native BF16 without hidden-size scaling.
2. Residual additions and RMSNorm outputs round at the same native-BF16
   boundaries as the reference.
3. Ternary projections preserve reference grouping and reduction order closely
   enough to pass the zero-tolerance full-trace acceptance below.
4. Q and K receive per-head RMSNorm. Sliding layers rotate their first 64
   dimensions with NeoX RoPE; V is unrotated. Full layers apply no positional
   encoding.
5. Grouped-query attention reads native-BF16 K/V. Sliding attention sees the
   latest 512 tokens through a physical 512-row rotating cache; full attention
   sees the complete prefix. Cache traversal/reduction must survive the wrap
   boundary without a parity change.
6. The router computes all 256 scores with BF16 weights and FP32 accumulation,
   chooses a deterministic top 8, and renormalizes selected softmax weights.
7. Each selected INT2 expert performs gate/up, clipped SwiGLU, and down;
   selected outputs reduce with FP32 route weights and re-enter the BF16
   residual stream at the reference boundary. Cache-hit and miss scheduling may
   overlap but cannot change router order or arithmetic.
8. Final RMSNorm feeds the complete 151,936-row exact head. The runtime may
   expose its existing FP16 logit buffer only if every retained BF16-rounded
   value is represented exactly. Maple never uses the generic fused-row head.

All activation and KV buffers on this path contain native BF16 bit patterns.
Their two-byte physical size is shared with the existing allocation model, but
they must never be interpreted as FP16. No conversion bridge is needed because
the path begins and remains native BF16.

## Prefill, tokenization, and products

Maple prefill is sequential replay through the exact one-token forward path.
Only the final prompt token needs to emit the vocabulary head. There is no
accepted batch-prefill environment switch or `PrefillMaple*` path.

Maple uses its pinned ChatML template. Model-family metadata explicitly selects
these differences from Qwen:

- message content is inserted verbatim, including meaningful outer whitespace;
- the generation suffix is `<|im_start|>assistant\n<think>\n` and leaves the
  reasoning block open;
- `<|im_end|>` is end-of-turn/EOS and both it and `<|endoftext|>` stop decode;
- text continuation reproduces the same framing; and
- generated text stays hidden until `</think>` closes the private channel.

CLI and the Mac app have no tool-execution surface and must fail closed if the
structured stream produces a tool call. The server may return a validated
OpenAI-style function call but never executes it. The model must install,
auto-detect, load, reset, continue, and report BF16 KV through Repack, CLI, Mac
app/decode service, and server without changing existing-family behavior.

The clean product wiring selects the standalone Maple runner from manifest
family metadata at every production construction site. Its effective
sequential-prefill and BF16-KV modes cross the Mac decode protocol and enter
the server prompt-cache identity; the existing four families retain their
prior runner, prefill, and FP16-KV behavior. This model-free wiring evidence is
not a substitute for the installed-model product smoke and parity gates below.

## Frozen parity corpus and trace policy

The parity corpus is Edgar Allan Poe's *The Raven*, fetched from
`shahules786/PoetryFoundationData`, `default/train`, row 8507. Copyrighted text
is never committed. Normalize with Unicode NFC and LF line endings; trim
trailing space, drop empty source lines, group each six verse lines into a
stanza separated by one empty line, and end with one LF.

| Corpus property | Pin |
| --- | --- |
| Normalized SHA-256 | `6a8ffabe83cf524659bb2c83137e5d41545155f4c650a83c31642c287cc95555` |
| Shape | 6,877 UTF-8 bytes; 108 verse lines; 18 stanzas |
| Token policy | `raw-utf8-lf-nfc;add_special_tokens=false;bos=false;eos=false` |
| Token sequence | 1,639 IDs; SHA-256 `979f944999a0a5039bfe3a9074ca0886ea54a228663fc8ad828a8759871f261f` |

Teacher forcing applies no chat template or special token. At position `p`, an
engine consumes corpus token `p` with state from `0..<p`, then exports the top K
IDs and raw logits from the complete vocabulary. `target_id` is token `p+1`, or
null at the last position. This is vocabulary top K, not router top K.

The oracle runs with `use_flash_head=false`; Mference runs its full logits head.
Both use stable descending logit order with ascending token ID as the tie break.
Acceptance requires all 1,639 positions, ordered top-10 token-ID equality, and
zero absolute difference for retained shared-token logits (`logit_atol=0`). A
prefix run, top-1-only comparison, unordered set comparison, approximate head,
dirty source tree, or mismatched metadata is diagnostic only and cannot pass.

The harness must record and verify source commit/dirty state, binary hash,
checkpoint/runtime pins, tokenizer and corpus hashes, cache slots, integrity
policy, commands, and exit codes. Full traces remain ignored local artifacts;
only compact non-copyrighted metadata may be committed.

## Required clean validation matrix

“Reference inventory” below means that the research branch contains a relevant
test or record. It is a pointer for rebuilding coverage, not evidence that a
new implementation passed.

| Gate | Observable clean acceptance | Reference inventory | Clean status |
| --- | --- | --- | --- |
| Toolchain/package baseline | [Scripts/test.sh](../Scripts/test.sh) exits 0 serially under the active supported toolchain. | Command Line Tools blocker was fixed before Maple work. | Required rerun after every branch. |
| Existing-family regression | Full package suite and relevant product builds pass with no Gemma/Qwen/DeepSeek/Inkling behavior changes. | Research port was a broad two-commit diff. | Required. |
| Schema/manifest | Synthetic valid Maple install loads; one-field architecture and quantization mutations fail. | Planner/manifest tests exist on the reference branch. | Required clean tests. |
| Range transforms | Byte-exact widening/repeat, bounded scratch, interval validation, fingerprint/digest, cancel, and resume tests pass. | Synthetic transformed-install tests exist. | Required clean tests. |
| Provenance | Forged, stale, partial, moved, or absent receipts cannot mint source origin. | Reference diff exposed the issue and a candidate fix. | Required clean tests. |
| Real install | Pinned source streams without snapshot staging; `--verify-install` and strict runtime verification pass; manifest/layout/file hashes are recorded. | Reference branch reports an install. | Required new install verification; do not reinstall only to satisfy a unit gate. |
| BF16 primitives | Tiny BF16-only values, offsets, residual/norm boundaries, ternary projections, and full head match CPU fixtures. | Focused reference tests exist. | Required clean tests. |
| Attention/KV | Sliding partial-RoPE, full NoPE, GQA, 512-row wrap, full-prefix growth, reset, and continuation match reference fixtures. | Focused reference tests exist. | Required clean tests plus real trace. |
| Router/MoE | Stable top-8 IDs/weights, clipped activation, INT2 experts, FP32 reduce, cache hits/misses, and router-order streaming match fixtures. | Focused reference tests exist. | Required clean tests plus real trace. |
| Tokenization | Golden prompt/continuation bytes, stop IDs, open-thought suppression, and tool behavior pass using explicit family plumbing. | Reference tests used vocabulary/token-ID inference. | Required with that accidental mechanism removed. |
| Full teacher forcing | Clean pinned engines produce 1,639/1,639 ordered top-10 matches with retained-logit absolute difference 0. | Reference metadata records such a result for commit `997b5ad`. | Required complete rerun; historical result is not transferred. |
| Product smoke | Repack, CLI raw/chat, Mac/decode service, and loopback server install/detect/load/generate/stop/reset correctly, one model process at a time. | Reference branch wired each product. | Required clean workflows. |
| Release build | `MferenceRepack`, `MferenceCLI`, `MferenceServer`, `MferenceDecodeService`, and `MferenceMac` build in release. | Historical binaries were recorded. | Required clean builds; old hashes are irrelevant. |
| Benchmark (optional) | Fresh-process protocol records complete environment, commands, timing footers, outputs, and deviations without overstating warm-cache results. | A dirty-worktree, one-run-per-engine Raven benchmark exists. | Excluded unless rerun cleanly. |

## Historical evidence boundary

The reference branch's final metadata says its clean research implementation
matched the pinned oracle for all 1,639 ordered top-10 positions at zero retained
logit tolerance. That single fact justifies the strict acceptance target; it
does not mark any checkbox in [PROGRESS.md](../PROGRESS.md).

Do not import its dated evidence JSON, full traces, binary hashes, raw outputs,
or benchmark claims as results for the clean port. In particular, its Raven
throughput/memory comparison came from an earlier dirty integration worktree,
one measured process per engine, and an uncontrolled warm OS cache. It is
research context only. If the final documentation contains performance claims,
they require a clean rerun and must follow the safety/reporting rules in the
[community benchmark guide](COMMUNITY_BENCHMARKS.md), with every Maple-specific
protocol deviation named.

Before any oracle or Mference model process, apply the repository model-process
preflight: supported OS/Swift/hardware, sufficient disk, acceptable memory
pressure, complete required install, and no other Mference or MLX model process.
Never terminate an existing process, run two model owners, purge caches, or
download/duplicate a checkpoint merely to make a test convenient.
