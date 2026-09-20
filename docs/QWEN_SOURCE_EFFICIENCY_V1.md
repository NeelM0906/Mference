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
default-policy screen remains unexecuted.

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
