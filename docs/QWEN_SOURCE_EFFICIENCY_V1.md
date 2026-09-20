# Qwen source-efficiency screen v1

Predeclared September 20, 2026, before inference. This functional/token-count
experiment uses the unchanged 60 cases and strict rubrics from
[release-screen-v1](benchmark-prompts/release-screen-v1/README.md), but is a
**separate protocol**, not its 512-token greedy/three-repeat benchmark.

Both installed dense checkpoints receive explicit `xhigh`, temperature 1,
Top-K 20, Top-P 0.95, repetition penalty 1, 4,096 maximum completion tokens,
8,192 server context, no MTP and no prefix cache. Five seeds are fixed:
20260920–20260924. There are 600 requests (60 cases × five seeds × two
checkpoints), no warmups; latency is diagnostic only and no speed ranking is
made. Run base then Swift in one serial library server; no model copies or
concurrent model owners. The source revision uncertainty for the legacy base
receipt is retained in the profiles, not retrospectively repaired.

Primary: successes out of all 300 attempts per model and cases passing all
five seeds. Secondary: all-request completion/reasoning/visible token counts.
Failures, empty responses, malformed calls and truncations remain in the
denominator; missing usage is null. Preserve all responses and scores. No
post-hoc rubric, seed, effort or budget changes. Three consecutive transport
errors stop the run as partial. This is not 300 independent questions, an
upstream benchmark reproduction, or broad coding/agent quality certification.
The larger cap is still finite; any truncation is evidence, not a fast win.

Use `python3 Scripts/qwen_efficiency_screen.py --port PORT --engine-commit SHA
--machine-record PATH --output NEW_JSONL` against an existing loopback server
with the policy above. Output is create-only. Record hardware/RAM, OS, Swift,
model manifest hashes and exact server command in the machine record.

Separately, `defaults.json` beside the frozen release corpus omits effort for
both models. Run it with the original release-screen-v1 tool/settings/context.
That comparison evaluates **default template/reasoning policies at frozen
greedy screen settings**, not product-default sampling or effort matching:
legacy base tool requests disable thinking, while Swift follows its source
template. Do not attribute this policy difference to the fine-tune alone.

Neither experiment changes the product default recommendation. Swift promotion
requires successful quality and completed-answer performance evidence, not
token savings alone. The prior matched-low comparison remains separately
[recorded](families/QWEN_MATCHED_QUALIFICATION_2026-09-18.md).

### Upstream reports versus local diagnosis

Rechecked September 20: in [discussion #7](https://huggingface.co/ukisai/Swift-Qwen3.8-27b/discussions/7),
a user reports higher reasoning-token use on AIME 2026 under a different
sampling/precision setup. The publisher requests its documented five-seed
protocol and acknowledges a math-token training-penalty bug. This establishes
that savings are not guaranteed for every workload; it does not identify the
cause of Mference's local failures or make this small corpus an AIME replica.

For the observed XML `False` issue, the pinned
[SGLang Qwen parser](https://github.com/sgl-project/sglang/blob/745de73ba3c136b6f99b7a3e2177ed1a8eef4a56/python/sglang/srt/function_call/qwen3_coder_detector.py)
already converts case-insensitive boolean spellings using the parameter schema.
Mference adopts that narrow compatibility behavior only for explicitly declared
booleans and recognizable true/false values. It does **not** adopt coercion of
other strings to false or general Python-literal evaluation. The JSON wire
argument remains properly typed; a declared string stays a string.

## Completed five-seed run — September 20

Engine `8c46ad3ec6e594ccbd786d8d7ceeb618020f558f` (runtime `c8f3298`),
Mac Studio M3 Ultra / 256 GiB, macOS 26.3 (25D125), Swift 6.3.3. Preflight
reported 98% memory free and 619 GiB available disk; both completed receipts
were checked. Full SHA-256 verification, one model owner, no model downloads.
Exact commands:

```sh
env MFERENCE_MTP=0 /tmp/mference-phase1-build.sXnNTs/release/MferenceServer \
  --library scratch/qwen38.gturbo --library scratch/swiftqwen38.gturbo \
  --port 18489 --max-context 8192 --prompt-cache-mode off \
  > scratch/qwen-seed-evidence.CzLD25/server.log 2>&1
python3 Scripts/qwen_efficiency_screen.py --port 18489 \
  --engine-commit 8c46ad3ec6e594ccbd786d8d7ceeb618020f558f \
  --machine-record scratch/qwen-seed-evidence.CzLD25/machine.txt \
  --output scratch/qwen-seed-evidence.CzLD25/results.jsonl \
  > scratch/qwen-seed-evidence.CzLD25/progress.log 2>&1
```

Screen exit 0; complete summary: 600 received / 600 expected. Server exited 0
after termination following the last response. There is no CLI timing footer
for this HTTP experiment; per-request timings remain in the raw JSONL.
Source edits and CPU-only debug builds occurred during this fixed-binary run.
No concurrent model tests or GPU inference occurred; **latency is not a
benchmark result**. The declared settings, seeds, corpus and scoring did not
change. All 600 requests completed: 501 `stop`, 99 `tool_calls`, no truncations
or transport errors.

| Metric | Base xhigh | Swift xhigh |
| --- | ---: | ---: |
| Passing attempts | 297/300 | 293/300 |
| Cases passing every seed | 57/60 | 57/60 |
| Completion tokens, all attempts | 26,474 | 22,505 |
| Reasoning tokens, all attempts | 22,662 | 18,721 |
| Visible tokens, all attempts | 1,626 | 1,569 |

Swift used **14.992% fewer completion tokens** and **17.390% fewer reasoning
tokens** in this corpus. These totals include failures. Completion usage also
includes tool/structural tokens, so reasoning plus visible counts need not
equal completion counts. This supports token savings under this policy, not
equal quality or lower completed-answer latency in general.

The ten failures are preserved without rescoring:

- Base `tool-order`, seed 20260920: claims the supplied lookup tool is absent.
- Base `tool-boolean`, seed 20260924, and Swift `tool-boolean`, seeds 20260920,
  20260921, 20260923, 20260924: XML `False` became a JSON string. This exposed
  a schema-aware parser compatibility gap, not five independent reasoning
  errors. The subsequent fix and targeted rerun are separate evidence.
- Base `tool-string`, seed 20260923, and Swift `tool-string`, seeds 20260920,
  20260924: an extra final period. The prompt's punctuation is ambiguous, but
  its frozen exact-match rubric is retained; the parser must not remove text.
- Swift `code-slice`, seed 20260920: valid JSON but incorrect Python slice
  result `[2, 4, 5]` instead of `[2, 4]`.

Raw evidence is retained locally at
`scratch/qwen-seed-evidence.CzLD25/results.jsonl`; SHA-256
`2c1d06e5640a1da4b78c3808b096c2641d1cc206c5bf2b39b2c4587bcabf5848`.
It is not bundled in the source release. This hash identifies the unchanged
original run, not a new aggregate result after the parser fix. The separate
default-policy screen is recorded below.

### Separate live boolean-parser recheck

Release `13a0367a78916f6c11fdfeb119baa03dabb04677`, same host/toolchain,
98% memory free and 619 GiB disk before launch. Both seven-file install
receipts matched; strict verification. The server used the exact command
above, with output instead at `boolean-recheck-server.log` in the same
evidence directory. No other model owner, builds, downloads or profiling.

The unchanged `tool-boolean` case was repeated for both profiles and all five
seeds, with the same xhigh/sampling/cap/context/MTP/cache policy: **10/10 pass**,
all finishing with a tool call and JSON boolean `false`. Runner exit 0 and
final footer `{"passed": 10, "expected": 10}`; server exited 0 after all replies.
The temporary diagnostic runner command was:

```sh
python3 scratch/qwen-seed-evidence.CzLD25/boolean_recheck.py \
  --port 18489 --engine-commit 13a0367a78916f6c11fdfeb119baa03dabb04677 \
  --output scratch/qwen-seed-evidence.CzLD25/boolean-recheck.jsonl \
  > scratch/qwen-seed-evidence.CzLD25/boolean-recheck-progress.log 2>&1
```

Raw recheck SHA-256:
`bfd10443cfedb1f0a9d4c93a3d494be7671b7b7125dd7c142676ce482012e16e`.
This is targeted regression evidence, not a new aggregate 60-case score.
The original base 297/300 and Swift 293/300 results remain unchanged.

## Completed default-policy screen — September 20

The separately frozen `defaults.json` profiles now complete all **480/480**
requests: one discarded warmup and three measured repeats per case/profile.
This is the unchanged `release-screen-v1` protocol, not a rerun of the xhigh
experiment. Binary `13a0367`, same M3 Ultra / 256 GiB, macOS 26.3 and Swift
6.3.3; 98% memory free, 619 GiB disk, both seven-file receipts checked and
strict verification. Machine-record documentation head is `0832751`; later
test/document/comment-only edits did not rebuild or change the running binary.
No other model owner, downloads, builds, profiling or demanding workloads.
Light source/document editing, GitHub and upstream-reference checks continued.

```sh
env MFERENCE_MTP=0 /tmp/mference-phase1-build.sXnNTs/release/MferenceServer \
  --library scratch/qwen38.gturbo --library scratch/swiftqwen38.gturbo \
  --port 18489 --max-context 4096 --prompt-cache-mode off \
  > scratch/qwen-default-evidence.UPJUs6/server.log 2>&1
python3 Scripts/release_task_screen.py --port 18489 \
  --profiles docs/benchmark-prompts/release-screen-v1/defaults.json \
  --engine-commit 13a0367a78916f6c11fdfeb119baa03dabb04677 \
  --machine-record scratch/qwen-default-evidence.UPJUs6/machine.txt \
  --output scratch/qwen-default-evidence.UPJUs6/results.jsonl \
  > scratch/qwen-default-evidence.UPJUs6/progress.log 2>&1
python3 Scripts/release_screen_summary.py \
  scratch/qwen-default-evidence.UPJUs6/results.jsonl \
  > scratch/qwen-default-evidence.UPJUs6/summary.json
```

Runner and summary exit 0; server exits 0 after the final reply. Completion
summary: `complete=true`, `received_records=480`, `expected_records=480`.
No CLI timing footer applies; every request's timings and usage are retained.
Both profiles have 150 measured `stop` and 30 measured `tool_calls` finishes;
no transport errors, empty-answer failures or truncations occurred. Warmups
are excluded from every count in this table:

| Metric | Base default | Swift default |
| --- | ---: | ---: |
| Passing measured attempts | 177/180 | 180/180 |
| Cases passing all three repeats | 59/60 | 60/60 |
| Completion tokens, all measured attempts | 16,368 | 13,080 |
| Reasoning tokens, all measured attempts | 14,286 | 10,812 |
| Visible tokens, all measured attempts | 783 | 939 |

Swift uses **20.088% fewer completion tokens** and **24.318% fewer reasoning
tokens** here. For the 177 paired successful attempts, completion totals are
16,281 versus 12,768 (21.577% fewer); this conditional statistic does not
replace the all-attempt result. Base's only failing case, `tool-string`, adds
the period discussed above in all three measured repetitions. The rubric is
unchanged; this ambiguous item does not establish a substantive quality gap.
Swift passes all ten tool cases, including booleans, under the corrected parser.

Omitted effort deliberately means different template/reasoning policies:
base tools disable thinking, Swift follows source-default xhigh. Greedy
sampling is the frozen screen's setting, not the app's default sampler.
Do not attribute this comparison's savings solely to fine-tuning. Base's
legacy hidden reasoning is not streamed, so first model-delta time is also
not directly comparable with Swift's reasoning deltas. No latency ranking is
claimed. Three identical greedy repeats remain 60 tasks, not 180 independent
questions. The prior matched-low and xhigh failures still stand.

Raw results SHA-256:
`17db9fde3b6062cc0594d45db1aa98b411156f9ab14a0dd52219f11eec827da2`.
This closes the declared bounded default-policy screen, not broad quality,
long-form completed-answer performance, quantization impact, or new physical
smaller-memory qualification. Swift remains optional.
