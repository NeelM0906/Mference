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

- Finish GLM install and execute its installed cutover/continuation/recovery gate.
- Qualify and measure optimizations on matched release builds before speed claims.
- Execute the separately frozen [60-case screen](benchmark-prompts/release-screen-v1/README.md).
- Complete final serial regression and both CI platform legs after all changes.
- Record the final support limits and tagged source-release version.

Passing on this 256 GiB Mac does not establish a 24 GiB hardware recommendation.
Swift default promotion and MTP speed claims remain gated on their own evidence.
