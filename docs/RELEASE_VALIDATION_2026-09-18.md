# Source release validation — September 18, 2026

Work in PR #37, based on merged `main` revision `c67e857`. This is a running
evidence record, **not a published-release or all-roadmap-complete claim**.
The user's requested deliverable is a tested source release, not a signed
prebuilt app. No model weights are redistributed.

## Environment and test boundaries

Mac Studio Mac15,14, Apple M3 Ultra (32 CPU cores), 256 GiB RAM; macOS 26.3
(25D125); Apple Swift 6.3.3 (`swiftlang-6.3.3.1.3 clang-2100.1.1.101`).
`DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`;
build scratch `/tmp/mference-phase1-build.sXnNTs`.
The non-default scratch path avoids incompatible existing toolchain artifacts;
none were purged. Before model owners: no matching model/server/test/MLX
process, acceptable memory pressure (94–97% free), more than 750 GiB free disk,
and completed installs with matching receipt sizes. Loaders use default strict
SHA verification. Model owners run serially; only task-owned services stop.

GLM was not found in the local library, scratch, available volumes or searched
user download/project locations. The user explicitly authorized installation
if missing. Its pinned range-streaming install uses about 180.8 GB without
staging a full source checkpoint. It ran during correctness tests, **not a
performance benchmark**. It was interrupted at 473 verified ranges / 31.001 GB
to test the real UI launcher, then resumed with those ranges preserved.

## Correctness and builds

Swift-Qwen's formerly failed mean-logit gate passes on `d2b84f1` with unchanged
tolerances. See [the complete numerical record](families/SWIFT_QWEN38_QUALIFICATION.md).

On `01bdb3d`:

```bash
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs
```

Exit 0. Full footer:

```text
Build complete! (0.19s)
Test run with 1231 tests in 218 suites passed after 275.233 seconds with 1 known issue.
```

The known issue is the existing missing optional Flash-Next toy checkpoint.
Environment-gated installed tests do not execute in this ordinary suite.
Log: `/tmp/mference-release-full-suite.log`.
Focused new resident-grouping/streamed-recovery/TensorOps tests also passed:

```bash
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs --filter 'PrefillDeviceMoEGroupingTests|FlashNextChunkedPrefillTests|MPPGroupedRoutedMoETests'
```

Exit 0; `Test run with 8 tests in 3 suites passed after 2.214 seconds.`
Log: `/tmp/mference-release-flash-overlap.log`. This includes exact GPU/CPU
route groups, indirect TensorOps dispatch with empty groups, observed resident
path selection, warm appends and cancellation/reset for both memory modes.

Release products, same runtime source at `01bdb3d`:

```bash
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift build -c release --scratch-path /tmp/mference-phase1-build.sXnNTs --product MferenceCLI
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift build -c release --scratch-path /tmp/mference-phase1-build.sXnNTs --product MferenceServer
```

Exit 0 each. Full completion lines:

```text
Build of product 'MferenceCLI' complete! (60.34s)
Build of product 'MferenceServer' complete! (6.48s)
```

Existing Sendable-capture warnings remain in resident streaming, DFlash2 and
KV spill code. The installer binary was not replaced while it was running.
These local product builds do not substitute for both full CI builds.

## Real Flash-Next prefill

Revision `31f0067`, existing INT8-router install `qwen38flashnext-r8.gturbo`,
source `Qwen/Qwen3.8-Flash-Next` at `de4b8e4d43b917e7706784d8bb445c9af86a3540`.
One model loaded at a time, first 16-slot streamed, then resident:

```bash
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer MFERENCE_FLASHNEXT_GTURBO=/Users/studio2/Documents/ChatGPT/Mference/scratch/qwen38flashnext-r8.gturbo Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs --filter chunkedPrefillMatchesSequentialOnRealInstall
```

Exit 0, build `10.23s`. Full result/footer:

```text
[flashnext-prefill-ab] resident=false prompt=62 prompt_max_abs=0.5338135 relative=0.021746699 prompt_argmax=7108/7108 next_max_abs=0.4140625 relative=0.017074741 next_argmax=544/544 greedy16=exact
[flashnext-prefill-ab] resident=true prompt=62 prompt_max_abs=0.5338135 relative=0.021746699 prompt_argmax=7108/7108 next_max_abs=0.4140625 relative=0.017074741 next_argmax=544/544 greedy16=exact
Test chunkedPrefillMatchesSequentialOnRealInstall(resident:) with 2 test cases passed after 92.890 seconds.
Suite FlashNextRealGenerationMeasurement passed after 92.890 seconds.
Test run with 1 test in 1 suite passed after 92.890 seconds.
```

The unchanged gate requires relative error below 0.05, matching top-1 and exact
16-token greedy continuation. It asserts zero replay and actual GPU grouping
in resident mode. This short-context test does not qualify long-context real
prefill, TensorOps-sized chunks or speedups. GLM installation ran concurrently;
debug timings are not throughput evidence. Log:
`/tmp/mference-release-flash-installed.log`.

## Browser → API → model → history → answer

Real launcher, isolated data, installed Open WebUI 0.11.3, browser automation
0.27.0 (`mference-release` session). Original UI database untouched.

```bash
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer ./mference-ui.sh doctor --server-port 18489 --webui-port 18490
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer ./mference-ui.sh --build-path /tmp/mference-phase1-build.sXnNTs --data-dir /tmp/mference-release-ui.4sDPjh/data --library scratch/swiftqwen38.gturbo --library scratch/qwen38.gturbo --server-port 18489 --webui-port 18490 --max-context 4096
```

Doctor exited 0 without opening the UI database. The launcher checked package
version, incrementally built the server, started both on loopback and disabled
builtin tools per model. The first browser navigation preceded first-use DB
migration completion and received connection-refused; it succeeded once the
health endpoint was ready. No browser errors were reported after readiness.

On runtime `01bdb3d`, Swift answered 42, then 50 on a follow-up; the second
request reused 110 prompt tokens. Page reload preserved history. An explicitly
enabled, locally authored pure-addition tool was installed **only in the test
database**. The native UI called `release_add(a=17,b=25)`, displayed execution,
fed back the result, and rendered 42. Its result turn reused 539 prompt tokens.
No generated code, shell, network or file-access tool was executed.
Cancelling a number-list generation recovered on the next request (answer 7).

Switching to base Qwen after that tool history exposed an empty-answer bug.
The tool-history template had already closed thinking, but the response decoder
assumed the ordinary-chat thinking-open default. `9fa34ee` now derives the
initial channel from the actual generation suffix, shared by CLI and server.
Focused regression command:

```bash
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs --filter 'SwiftQwenChatTests|SwiftQwenServerTests|ChatMLDecoderTests|ChatMLTemplateTests|MapleTokenizationTests|Glm5DecoderTests|MiniCPM5DecoderTests|ServerPromptCacheTests'
```

Exit 0; `Test run with 62 tests in 8 suites passed after 4.220 seconds.`
Server rebuild exited 0:
`Build of product 'MferenceServer' complete! (56.61s)`.
The same launcher command restarted on the preserved test database. The exact
failed base request now displayed 21, then switching back to Swift returned 20.
History/reasoning/tool records remained available after restart. Both launcher
runs were stopped with TERM, exit 143 as designed, and no model owner remained.
The browser session was closed; test data and evidence are retained.

Complete relevant request footers, including the original failure:

```text
[2026-09-18T15:04:39Z] request chatcmpl-95170386ea9c4a69aa75f2fbde0d222c completed in 10.3s prompt=68 cached=0 completion=43 finish=stop
[2026-09-18T15:05:12Z] request chatcmpl-6fbb48041ed14183a4af9a87a8a840c7 completed in 1.6s prompt=136 cached=110 completion=35 finish=stop
[2026-09-18T15:06:15Z] request chatcmpl-6952c5358b93455d94ec75426ddd0b80 completed in 8.3s prompt=469 cached=0 completion=71 finish=tool_calls
[2026-09-18T15:06:15Z] request chatcmpl-80b262f58a244e5f8e16414e94acdc9e completed in 0.8s prompt=557 cached=539 completion=15 finish=stop
[2026-09-18T15:06:51Z] request chatcmpl-f284c71826874c6f89f6b1996becb68f failed status=500 streaming=true error=CancellationError()
[2026-09-18T15:07:12Z] request chatcmpl-b8853756c21a42e8a3272604b5e20fe0 completed in 12.7s prompt=886 cached=0 completion=23 finish=stop
[2026-09-18T15:07:46Z] request chatcmpl-8c0bab07f8b2495cb579d5b6536025cc completed in 10.0s prompt=269 cached=0 completion=3 finish=stop
[2026-09-18T15:12:59Z] request chatcmpl-9a48d70e671f4f7f953cfb0e20ed94ab completed in 9.5s prompt=269 cached=0 completion=3 finish=stop
[2026-09-18T15:13:54Z] request chatcmpl-447d7f7ec9bc4b18af4f4f918ef12aff completed in 17.8s prompt=712 cached=0 completion=13 finish=stop
```

Intentional disconnected-client cancellation still logs as a 500-class operator
error; the subsequent request succeeds. These timings include model loads and
interactive activity, not a comparative performance protocol. Logs/screenshots:
`/tmp/mference-release-ui.4sDPjh/{launcher.log,restart.log,tool.png,switch-recovered.png}`.

## Remaining release gates

### Subsequent regression results

On `fa07319`, the same full-suite command above exited 0:

```text
Build complete! (8.29s)
Test run with 1236 tests in 221 suites passed after 280.945 seconds with 1 known issue.
```

Log: `/tmp/mference-release-full-suite-v2.log`. The known issue and installed
test-gate limitations are unchanged. This includes the GLM deterministic GPU
selection and paired-query attention checks, plus installer-progress tests.
The targeted GLM checks on the preceding change also passed:

```text
Test run with 12 tests in 3 suites passed after 20.808 seconds.
```

These synthetic attention checks include bit-exact paired/unpaired FP16 output,
odd chunks, nonzero append positions, production head geometry and output
guards. Selector checks cover stable ties, complete-pool expansion and tails.
Log: `/tmp/mference-release-glm-paired-attention.log`. Real GLM remains pending.

Reproduce the targeted GLM coverage with:

```bash
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs --filter 'Glm53DeviceSelectionTests|Glm53ForwardRunnerTests|Glm53PairedAttentionTests'
```

Exit 0; build `11.59s`.

Payload token accounting was then added for ChatML/GLM/MiniCPM, separately from
total completion usage. An initial new test used malformed tool syntax and
failed; after correcting its fixture to the dialect's newline-delimited form,
the focused decoder/HTTP suites passed (55 tests / 6 suites, 3.052s, exit 0;
`/tmp/mference-release-token-accounting-v2.log`). Seven Python screen tests
also passed. Visible counts are read explicitly, never inferred by subtracting
reasoning from completion (which would incorrectly include EOS/tool markers).

Exact focused accounting command (build `6.04s`, exit 0):

```bash
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs --filter 'ChatMLDecoderTests|Glm5DecoderTests|MiniCPM5DecoderTests|HTTPServerTests|SwiftQwenServerTests'
```

Release build commands recorded above were repeated at `048d04f`; CLI `57.25s`,
server `6.53s`, then the full `swift build -c release --scratch-path ...`
command `16.34s`, all exit 0. Source archive inspection passed 823 entries;
all then-current 67 Markdown files passed link checking. Launcher, adapter,
12-task harness and 60-task harness tests passed (8, 5, 3, 7 tests respectively).
A broad system-Python discovery command failed importing the optional MLX
reference-reader suite because MLX was absent from system Python. The required
declared runner, `uv run Scripts/tests/test_swift_qwen_mlx_reference.py`, then
passed its 4 tests in `0.028s`, exit 0, without model inference/downloads.

GitHub CI run `35362494265` on `fa07319` passed both `macos-15` (Swift 6.1 floor)
and `macos-26`, plus docs and security checks. Subsequent changes require their
own final CI result; this does not certify a later commit automatically.

### Tiled Swift projection and additional recovery gates

The first Swift correctness fix regressed medium/long prefill time. Complete
failed community attempts, settings and footers are preserved in
[the performance experiment record](RELEASE_PERFORMANCE_2026-09-18.md).
`ee0bfb9` uses up to four prompt rows per GPU tile to share weight reads while
retaining per-token decode arithmetic; no host per-token/full-model replay.

The same installed Swift command used for the earlier numerical gate was run
on `ee0bfb9` with `MFERENCE_MTP=1`, unchanged tolerances and strict verification:

```text
Build complete! (1.48s)
Test prefillAppendAndSpeculativeContinuation() passed after 77.314 seconds.
Suite SwiftQwenInstalledQualificationTests passed after 77.314 seconds.
Test run with 1 test in 1 suite passed after 77.314 seconds.
```

Exit 0; all three head-error values are unchanged from `d2b84f1` (mean
0.0037590205 / 0.0037336384 / 0.0074094827). Top-1, continuation and all 20 MTP
state probes pass; the MTP probes have zero bit mismatches. Log:
`/tmp/mference-release-swift-tiled-qualified.log`. The resumed GLM download ran
concurrently: these debug durations are correctness evidence, not performance.

Before that commit, its exact source changes passed:

```bash
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs --filter 'PrefillAffineTests|MultiXKernelParityTests|ProductionPrefillContractTests|MapleForwardRunnerTests|MiniCPM5ForwardRunnerTests'
```

Exit 0, build `6.05s`;
`Test run with 26 tests in 5 suites passed after 27.507 seconds.`
Log: `/tmp/mference-release-tiled-and-recovery-v2.log`. Projection tests compare
every output bit with repeated decode for production shapes, ragged tails,
nonzero offsets and output guards. New recovery tests interrupt warm appends
after GPU state writes, reject dirty continuation/decode, then reset and
reproduce clean results for Gemma/Qwen (eight-slot/resident), MiniCPM and Maple.
Gemma/Qwen and MiniCPM exercise actual task cancellation, not just an injected
error. Production adds cancellation checks without extra GPU synchronization;
failure-injection hooks are nil outside tests. Initial test compilation exposed
an SDK Metal-buffer Sendable annotation gap and a Float/Float16 test comparison;
the tests were corrected without weakening runtime checks or numerical limits.

The complete serial suite on `ee0bfb9` then passed (same command/environment
as above, exit 0):

```text
Build complete! (1.51s)
Test run with 1241 tests in 221 suites passed after 279.658 seconds with 1 known issue.
```

Log: `/tmp/mference-release-full-suite-v3.log`; the same absent optional
Flash-Next toy fixture is the known issue. GLM installation ran concurrently;
no speed result is inferred from test duration. Its resumed release installer
now visibly reports saved bytes separately from bytes downloaded this run.

CI run `35366071136` on `ee0bfb9` also completed successfully: release builds,
serial tests and launcher/adapter checks on both macOS 15 / Swift 6.1 and
macOS 26, plus the docs/source-archive job. This is the result after correcting
the test-only Sendable compile failure in run `35364757262`, not a suppression
of that failure. Later revisions require their own final checks.

An additional CLI notice explains an empty token-limit truncation and the
available budget/Swift effort controls, without modifying generated content,
sampling, exit status or the existing timing footer. Its focused verification:

```bash
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs --filter 'CLIArgumentsTests|ChatHistoryTests'
```

Exit 0; build `6.99s`; `Test run with 34 tests in 2 suites passed after 0.004 seconds.`
Log: `/tmp/mference-release-cli-notice.log`.

### Final-build diagnostics and browser-origin checks

Release CLI and server at `c9c3378` were built with the product commands above,
exit 0 each:

```text
Build of product 'MferenceCLI' complete! (58.44s)
Build of product 'MferenceServer' complete! (5.80s)
```

CI run `35368664968` on `c9c3378` completed successfully on both macOS 15 /
Swift 6.1 and macOS 26, including release builds, serial tests, launcher/adapter
checks and docs/source-archive checks. Later recovery changes are not covered
by that earlier commit's CI result.

A single loopback server was then started for **correctness diagnostics**, not
performance (the approved GLM installer was running concurrently):

```bash
env MFERENCE_MTP=0 /tmp/mference-phase1-build.sXnNTs/release/MferenceServer --model scratch/swiftqwen38.gturbo --port 18489 --max-context 4096 --prompt-cache-mode off
```

The frozen short-explanation messages were submitted to
`http://127.0.0.1:18489/v1/chat/completions`, non-streaming, with model
`swift-qwen3.8-27b-int4g64`, `max_completion_tokens=1024`, temperature 0.2,
top-k 64, top-p 0.95 and seed 20260721. Default effort, explicit `medium`, then
explicit `low` were diagnostic follow-ups, **not** a replacement community
benchmark or a new task-screen protocol. Each HTTP command exited 0:

| Effort | Prompt tokens | Completion tokens | Reasoning payload | Visible payload | Finish |
| --- | ---: | ---: | ---: | ---: | --- |
| Default (`xhigh`) | 102 | 1024 | 1024 | 0 | length |
| medium | 60 | 1024 | 1024 | 0 | length |
| low | 90 | 847 | 248 | 597 | stop |

The default response contained coherent internal drafting and word counting,
not a hidden completed answer lost by the decoder. Low produced a visible
answer, but this one response is not a quality evaluation or reason to promote
a new default. Marker/EOS tokens explain why payload counts need not sum to
completion tokens. Complete server request footers:

```text
[2026-09-18T16:15:24Z] request chatcmpl-d923c46299f54bd58c52917de6863717 completed in 31.4s prompt=102 cached=0 completion=1024 finish=length
[2026-09-18T16:17:04Z] request chatcmpl-52d7d8f0e2554ddfacec2c61050c88b9 completed in 29.2s prompt=60 cached=0 completion=1024 finish=length
[2026-09-18T16:17:29Z] request chatcmpl-85857e831e3845e89b4de8c116f089a5 completed in 24.8s prompt=90 cached=0 completion=847 finish=stop
```

Responses: `/tmp/mference-release-swift-reasoning-diagnostic.json`,
`/tmp/mference-release-swift-reasoning-medium.json`, and
`/tmp/mference-release-swift-reasoning-low.json`. Startup/request log:
`/tmp/mference-release-reasoning-diagnostic-server.log`.

Pinned Open WebUI 0.11.3 was started manually using the launcher's loopback
environment, existing isolated test data and secret, and this same model server.
This is a live CORS component check, not another full launcher/browser run.
Health returned `{"status":true}`. The following OPTIONS request was repeated
with origins `http://127.0.0.1:18490`, `http://localhost:18490`, and
`https://untrusted.example`:

```bash
curl -sS --max-time 10 -X OPTIONS -H 'Origin: http://127.0.0.1:18490' -H 'Access-Control-Request-Method: POST' -H 'Access-Control-Request-Headers: content-type' -D - -o /dev/null http://127.0.0.1:18490/api/config
```

All curl commands exited 0. Both allowed origins returned HTTP 200 and their
exact `Access-Control-Allow-Origin`; the foreign origin returned HTTP 400 with
no allow-origin header. UI log:
`/tmp/mference-release-ui.4sDPjh/cors-verification.log`. Only these task-owned UI
and server processes were stopped; their absence was confirmed before the next
model run. Existing chats and the test database were retained.

### Recovery across dense, paged, spilled and Inkling state

Source committed as `7505e82`; same hardware/toolchain/build path as above.
Before model owners: matching-process check empty, memory-free check 97%,
disk 717–721 GiB, completed existing Inkling install. GLM range installation
ran concurrently; these are correctness checks, **not throughput timings**.

```bash
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs --filter 'Qwen38BlockedPrefillTests|Qwen38ForwardRunnerTests|MiniCPM5ForwardRunnerTests|InklingPrefillRecoveryTests'
```

Exit 0, build `17.01s`;
`Test run with 22 tests in 4 suites passed after 3.606 seconds.`
Log: `/tmp/mference-release-recovery-expanded.log`. Inkling's env gate was
unset in this first command: its near-zero duration is a skip, not real-model
evidence. Qwen and MiniCPM each exercise dense, paged and five-page spilled KV:
a 400-token warm prefix, cancellation during a 33-token append after GPU state
writes, rejection of dirty continuation/decode, reset and exact reproduction
of the full prefix and next logits. This is a fixture state/recovery gate, not
smaller-Mac qualification or every sparse-page-budget combination.

Then the installed Inkling gate was explicitly enabled, with no other model
owner:

```bash
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer MFERENCE_INKLING_GTURBO=/Users/studio2/Documents/ChatGPT/Mference/scratch/inklingsmall.gturbo Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs --filter InklingPrefillRecoveryTests
```

Source `pipenetwork/Inkling-Small-MLX-4bit`, receipt revision
`9d6e4720ab7002af25d6129c88ccea6cd9f19372`; manifest SHA-256
`61fdbed85a221652b229a476561cd60dd65a6c0b3b437eec0acc6e123865a58b`.
Strict verification, 16 expert slots, context 128, chunk 32; no weight copies
or downloads for this test. Exit 0; complete result:

```text
Build complete! (1.50s)
[inkling recovery] strict verification; slots=16; warm=33; cancelled append=32; reset and next full-logit rows exact
Test cancelledWarmAppendResetsKVAndConvolutions() passed after 80.243 seconds.
Suite InklingPrefillRecoveryTests passed after 80.243 seconds.
Test run with 1 test in 1 suite passed after 80.243 seconds.
```

Log: `/tmp/mference-release-inkling-recovery.log`. The test cancels the actual
task before layer 3, after both dense layers and the first routed layer have
completed their KV/convolution/expert work. Production now checks cancellation
between chunks, layers and expert groups; existing in-flight expert cleanup
drains before reset. It does not add production GPU waits or alter arithmetic.
Qwen's nil test hook likewise preserves production command-buffer batching.

The complete serial suite on `7505e82` then passed:

```bash
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs
```

Exit 0; complete completion lines:

```text
Build complete! (1.51s)
Test run with 1244 tests in 222 suites passed after 281.137 seconds with 1 known issue.
```

Log: `/tmp/mference-release-full-suite-v4.log`. The known issue is still the
absent optional Flash-Next toy checkpoint. Installed gates were unset in this
ordinary suite; their separate executions above remain the model evidence.

Release CLI and server were rebuilt on the same `7505e82` source using the
product-specific commands above, exit 0 each. The running installer binary was
not relinked or replaced:

```text
Build of product 'MferenceCLI' complete! (57.86s)
Build of product 'MferenceServer' complete! (6.64s)
```

Finally, the installed Swift numerical/state/MTP command recorded above was
repeated on `7505e82`, with `MFERENCE_MTP=1` and unchanged tolerances. Exit 0:

```text
Build complete! (1.47s)
Test prefillAppendAndSpeculativeContinuation() passed after 77.826 seconds.
Suite SwiftQwenInstalledQualificationTests passed after 77.826 seconds.
Test run with 1 test in 1 suite passed after 77.826 seconds.
```

All three head errors remain exactly the values recorded for `ee0bfb9`;
top-1/continuation checks and all 20 zero-mismatch MTP state probes pass.
Log: `/tmp/mference-release-swift-recovery-qualified.log`. Preflight had no
other model owner, 97% memory free and 709 GiB disk. GLM download was active,
so this is correctness evidence only. No server or UI remained running after
these tests; the approved resumable GLM installer continued separately.

Launcher shell syntax and Python checks passed again (8 launcher, 5 adapter,
3 legacy screen and 7 release-screen tests), exit 0. The adapter suite's printed
version-mismatch usage error is the expected negative fixture. Markdown links
passed for 69 files; tracked source archive inspection passed 826 entries, with
no weights, build output or the user-owned untracked execution plan included.

The four-row Swift release comparison at `c9c3378` completed separately before
these tests: prefill median time reductions of 10.2%, 17.5% and 18.5% against
the first exact-arithmetic fix. All 24 attempts still truncated without visible
output, so none is an accepted completed-answer benchmark. Full commands,
provenance, all footers and limitations are in
[the performance record](RELEASE_PERFORMANCE_2026-09-18.md).

### Open gates

- Finish GLM install and execute its installed cutover/continuation/recovery gate.
- Qualify and measure optimizations on matched release builds before speed claims.
- Execute the separately frozen [60-case screen](benchmark-prompts/release-screen-v1/README.md).
- Complete final serial regression and both CI platform legs after all changes.
- Record the final support limits and tagged source-release version.

Passing on this 256 GiB Mac does not establish a 24 GiB hardware recommendation.
Swift default promotion and MTP speed claims remain gated on their own evidence.
