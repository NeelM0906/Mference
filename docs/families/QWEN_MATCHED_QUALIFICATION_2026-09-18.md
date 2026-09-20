# Matched base/Swift Qwen qualification — September 18, 2026

Implementation: `0c7532c0b1106f3d06839e6ddc8e7c8694fee4f9`, PR #37. This
continues the [cutoff/parser investigation](SWIFT_QWEN38_INVESTIGATION.md), not a
claim that Swift is qualified as the replacement or that the roadmap is done.

## Policy contract

Explicit `reasoning_effort` (`xhigh`, `medium`, `low`, `none`) selects the installed
source template for both pinned dense Qwen checkpoints. Tools no longer put
the base comparison into a different thinking policy. Explicit source-template
base requests expose/preserve `reasoning_content` like Swift. Developer turns
are rejected for this contract; use leading system guidance.

Omitted effort is unchanged: Swift uses source-default xhigh; base retains its
legacy ordinary/tool rendering. Sampling and MTP defaults are unchanged. Source
policy requests only reuse an exact rendered prefix. Both directions of a
source/legacy policy switch disable the legacy cache bridge, even if the
messages themselves look unchanged. Exact token-prefix matches remain safe.
Custom model aliases obtain capabilities from the loaded backend, not names.

## Verification environment

Mac Studio Mac15,14, M3 Ultra, 32 CPU cores, 256 GiB; macOS 26.3 (25D125);
Swift 6.3.3 (`swiftlang-6.3.3.1.3`). Xcode developer directory
`/Applications/Xcode.app/Contents/Developer`; build scratch
`/tmp/mference-phase1-build.sXnNTs`. Preflight: 97–98% memory free, 619 GiB disk
free, no model/test/MLX owner. No model copies, cache purges, profiling, or
experimental controls. GLM's previously approved range install finished before
this work's model runs; its real-model qualification is still separate.

The first focused build failed because one new test assertion omitted `try`.
The assertion was corrected; no runtime workaround or tolerance change.
Focused rerun (exit 0):

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  MFERENCE_MATCHED_QWEN_BASE=/Users/studio2/Documents/ChatGPT/Mference/scratch/qwen38.gturbo \
  MFERENCE_MATCHED_QWEN_SWIFT=/Users/studio2/Documents/ChatGPT/Mference/scratch/swiftqwen38.gturbo \
  Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs \
  --filter 'SwiftQwenChatTests|SwiftQwenServerTests|ServerPromptCacheTests|HTTPServerTests|QwenMatchedPromptTests'
```

Complete build/test footer:

```text
Build complete! (9.66s)
[qwen-matched-prompts] 60 cases x 4 efforts: identical installed prompt IDs
Test run with 46 tests in 6 suites passed after 10.689 seconds.
```

Log: `/tmp/mference-qwen-matched-focused-v2.log`. Installed-tokenizer coverage
is 240 prompt-ID comparisons, not a weight/inference correctness test. Fixture
coverage additionally checks ordinary/history/tool inputs, all efforts,
unchanged legacy defaults, developer-role handling, aliases, reasoning
channels, and cache transitions. Four summary tests and seven frozen-screen
tests also passed with `python3 Scripts/tests/test_release_screen_summary.py`
and `python3 Scripts/tests/test_release_task_screen.py` (both exit 0).

Full package suite, same implementation commit, exit 0:

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs
```

Full footer (`/tmp/mference-qwen-matched-full.log`):

```text
Build complete! (1.41s)
Test run with 1252 tests in 223 suites passed after 268.475 seconds with 1 known issue.
```

The known issue is the existing absent optional Flash-Next toy checkpoint.
Ordinary package runs do not execute environment-gated installed-model tests;
the tokenizer gate above was separately enabled in the focused run.

The complete release build on the same runtime source exited 0:

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  swift build -c release --scratch-path /tmp/mference-phase1-build.sXnNTs
```

Full footer: `Build complete! (59.31s)`.
Log: `/tmp/mference-qwen-matched-release-build.log`.

## First functional comparison — selected before inference

Profiles: [base-low and Swift-low](../benchmark-prompts/release-screen-v1/matched-low.json).
Follow the unchanged [release-screen-v1 protocol](../benchmark-prompts/release-screen-v1/README.md):
60 cases, one discarded warmup and three measured repeats for each profile/case,
512 output tokens, 4,096 context, greedy sampling, seed 20260918, MTP off,
prefix cache off, serial requests. Expected evidence is 480 request records
(120 warmups plus 360 measured). Three repeats do not create 180 independent
quality questions per model. Failures/truncations retain their denominator.

The base receipt predates revision recording. Its manifest index fingerprint
matches configured pin `3e6447f082e89cc7f0bc6e5441afd38dfce760ff`; this is not
retrospective proof of a recorded source revision. The profile keeps that
uncertainty explicit. Swift's recorded source is
`1b30aaaf753fe5c1cb51ada2ea0367a53445359c`.

This compares installed deployments, not BF16 versus adapter with all other
variables isolated. Their quantization histories and prefill implementations
differ. It is not a replication of upstream benchmarks, a general quality
ranking, or an accepted community speed result. No post-hoc effort/cap changes
will be made to improve the score. Default-vs-default and sufficiently budgeted
multi-seed efficiency evaluations remain separate requirements.

## Completed matched-low screen

Run commit: `ed69598f1a3ea34b921e44d8fc454b6860cb534e` (same runtime as
`0c7532c` above). Tracked tree clean; the unrelated untracked execution-plan
document was preserved. Same hardware/toolchain as above; 98% memory free,
619 GiB disk available, AC power and low-power mode off. No other model owner,
download, build, tests or demanding workload ran during the screen.

Exact commands, from the checkout root:

```sh
env MFERENCE_MTP=0 /tmp/mference-phase1-build.sXnNTs/release/MferenceServer \
  --library scratch/qwen38.gturbo --library scratch/swiftqwen38.gturbo \
  --port 18489 --max-context 4096 --prompt-cache-mode off \
  > /tmp/mference-qwen-matched.6l3Qeo/server.log 2>&1
python3 Scripts/release_task_screen.py --port 18489 \
  --profiles docs/benchmark-prompts/release-screen-v1/matched-low.json \
  --engine-commit ed69598f1a3ea34b921e44d8fc454b6860cb534e \
  --machine-record /tmp/mference-qwen-matched.6l3Qeo/machine.txt \
  --output /tmp/mference-qwen-matched.6l3Qeo/results.jsonl \
  > /tmp/mference-qwen-matched.6l3Qeo/progress.log 2>&1
python3 Scripts/release_screen_summary.py \
  /tmp/mference-qwen-matched.6l3Qeo/results.jsonl \
  > /tmp/mference-qwen-matched.6l3Qeo/summary.json
```

Harness and summarizer exited 0; all **480/480** expected records are present
(120 discarded warmups, 360 measured requests). This harness has no aggregate
timing footer; complete per-request timing, usage and responses are in the
JSONL, not a selected fast subset. The server exited 0 after SIGINT following
the completed run. No deviations from the frozen functional protocol; this
was **not** the community performance protocol and no speed claim follows.
Profiles ran serially, all base cases before all Swift cases, not randomized.
The five run artifacts are also preserved locally under
`scratch/qwen-matched-evidence.8fm9dG/` (ignored, no model weights), so the raw
evidence does not depend only on temporary-directory retention.

Corpus SHA-256:
`b3ab27c38a8d0f4b5fa062801573bfa46e06e9dcb62cb16f0dfaae05514d6680`.
Manifest SHA-256 values:

- Base: `ff778735cda6cc6c077240faf095539c4a0c033c6ae33558959aded87d95ad45`.
- Swift: `553bad0bf8ba819405bf1c6ca857cf86f734a620ade08708c248c3a66f406254`.

| Measured outcome | Base-low | Swift-low |
| --- | ---: | ---: |
| Passing requests / 180 | 177 | 162 |
| Cases passing all three repeats / 60 | 59 | 54 |
| Everyday / 30 | 30 | 30 |
| Reasoning / 30 | 30 | 30 |
| Code reading / 30 | 30 | 30 |
| Instructions / 30 | 30 | 30 |
| Tool calls / 30 | 30 | 27 |
| Synthesis / 30 | 27 | 15 |
| Completion tokens, all measured requests | 14,526 | 13,542 |
| Reasoning tokens, all measured requests | 12,366 | 11,301 |
| Visible-channel tokens, all measured requests | 831 | 912 |

Both profiles have 150 `stop` and 30 `tool_calls` measured finishes, with no
truncations or request errors. Completion counts include tool payloads and
delimiters; reasoning plus visible is not their total. Swift used **6.77% fewer
completion tokens** and **8.61% fewer reasoning tokens**, with worse strict
format compliance. For the 162 paired requests where both answers passed,
completion totals were 12,936 versus 12,003 (**7.21% fewer**); this conditional
comparison must not replace the all-request failures above.

Per-case exceptions below failed on all three measured repeats; every other
case in the frozen corpus passed all three repeats:

- Base: `synthesis-overwrite` returned the correct object inside Markdown
  fences, violating the JSON-only rubric.
- Swift: `synthesis-stock`, `synthesis-owner`, `synthesis-dedup`,
  `synthesis-cost` and `synthesis-overwrite` returned the correct values inside
  Markdown fences. The scorer did not strip them or relax the rubric.
- Swift: `tool-string` returned a valid `echo` call with a string argument,
  `She said "hello".`, where the expected value omits the final period. The
  prompt's sentence-final punctuation leaves an interpretive ambiguity; retain
  the frozen failure but do not characterize it as malformed arguments or a
  parser failure. Base emitted the expected value.

Decision: keep Swift **optional**, not the recommended base-Qwen replacement.
This short low-effort corpus shows modest token savings and a formatting
tradeoff, not universal efficiency or broad quality parity. Source-default
xhigh, sufficiently budgeted multi-seed tasks, representative end-to-end MTP
performance, multi-turn tool quality and smaller-memory hardware remain open.
