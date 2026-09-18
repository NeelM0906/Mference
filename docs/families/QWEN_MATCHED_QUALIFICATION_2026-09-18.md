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
