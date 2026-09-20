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
