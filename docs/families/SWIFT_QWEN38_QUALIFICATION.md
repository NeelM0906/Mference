# Swift-Qwen qualification record — updated 2026-09-18

Status: candidate, not promoted. This records successes **and failed gates**;
passing package tests does not imply full model qualification.

Post-launch update: the separately labeled low-effort / 4,096-output / 8,192-context
profile now completes all nine measured frozen-prompt runs on the 256-GiB M3
Ultra. The low/1,024 medium warmup failed. This is not base-model superiority or
default promotion; see [complete evidence and quality caveats](../POSTLAUNCH_QUALIFICATION.md).

## September 18 numerical correction and retest

The previously failed installed prefill gate below now passes on `d2b84f1`.
No thresholds changed. Swift-only batched INT4 projections now preserve the
decode kernel's affine factoring and reduction order. They still dispatch the
whole prompt block, without host-side full-model replay. The base checkpoint's
projection selection is unchanged. This addresses arithmetic differences that
compound through the fine-tuned network; it does not establish bit-exact
whole-model prefill or a performance improvement.

Host: Mac Studio Mac15,14, Apple M3 Ultra, 32 CPU cores, 256 GiB RAM;
macOS 26.3 (25D125); Swift 6.3.3 (`swiftlang-6.3.3.1.3 clang-2100.1.1.101`).
Safety checks found no existing model owner, 97% memory free and more than
760 GiB disk free. The existing completed Swift install was reused.

```bash
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer MFERENCE_SWIFT_QWEN_GTURBO=/Users/studio2/Documents/ChatGPT/Mference/scratch/swiftqwen38.gturbo MFERENCE_MTP=1 Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs --filter SwiftQwenInstalledQualificationTests
```

Exit **0**, build `13.40s`. Full footer:

```text
Test prefillAppendAndSpeculativeContinuation() passed after 81.928 seconds.
Suite SwiftQwenInstalledQualificationTests passed after 81.929 seconds.
Test run with 1 test in 1 suite passed after 81.929 seconds.
```

| Prompt | Max absolute | Mean absolute | Top-1 | Result |
| --- | ---: | ---: | --- | --- |
| 65 | 0.037109375 | 0.0037590205 | Same | Pass |
| 257 | 0.025390625 | 0.0037336384 | Same | Pass |
| 1025 | 0.046875 | 0.0074094827 | Same | Pass |

All 15 subsequent full-logit state probes and all 20 MTP reconciliation probes
passed, including stop lengths 2/7/32/64. The log is
`/tmp/mference-release-swift-qualified-final.log`.
The separate affine/golden/paged/blocked regression command was:

```bash
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs --filter 'PrefillAffineTests|Qwen38ForwardRunnerTests|Qwen38BlockedPrefillTests|Qwen38PagedKVParityTests'
```

Exit 0; `Test run with 20 tests in 4 suites passed after 19.264 seconds.`
New kernel checks require bit-identical batched-versus-decode projections,
including production-size matrices. No diagnostic tracing remains enabled.

Protocol limits: these are debug-build correctness tests, not performance
measurements. The separately approved GLM range-streaming installation ran
concurrently and may affect wall time. The non-default build scratch directory
avoids incompatible toolchain artifacts; no cache was purged. MTP was explicitly
enabled for correctness only and remains off by default. Broader task quality,
matched latency, long contexts and other hardware profiles remain unqualified.
The September 16 failures below remain as historical evidence.

## Environment

Engine baseline `049d987` plus the follow-up changes in PR #33 (explicit
family-based tool grammar, UI compatibility adapter and qualification tests).
Mac Studio Mac15,14, Apple M3 Ultra, 32 CPU cores, 256 GiB RAM; macOS 26.3
(25D125); Swift 6.3.3 (`swiftlang-6.3.3.1.3 clang-2100.1.1.101`).
Full Xcode: `/Applications/Xcode.app/Contents/Developer`. Before model owners:
no matching server/CLI/test/MLX processes, 97% system-wide memory free and
788–789 GiB available disk. Existing strict-verified base/Swift installs were
reused; no model copy, download or cache purge for these follow-up tests.

The existing `/tmp/mference-phase1-build.sXnNTs` build directory avoids the
checkout's other-toolchain artifacts. This build-path deviation is explicit.
All model owners ran serially. Only task-owned servers were stopped.

## Browser → API → history → answer

Open WebUI 0.11.3, isolated database at
`/tmp/mference-phase2-ui.78zbs3/data`, browser session `mference-phase2`.
The user's database/settings were untouched. API library:

```bash
/tmp/mference-phase1-build.sXnNTs/release/MferenceServer --library scratch/swiftqwen38.gturbo --library scratch/qwen38.gturbo --port 18489 --max-context 4096
```

UI command (isolated test secret, not a production credential):

```bash
env OPENAI_API_BASE_URL=http://127.0.0.1:18489/v1 OPENAI_API_KEY=local ENABLE_OLLAMA_API=false WEBUI_AUTH=false ENABLE_TITLE_GENERATION=false ENABLE_TAGS_GENERATION=false ENABLE_FOLLOW_UP_GENERATION=false ENABLE_AUTOCOMPLETE_GENERATION=false ENABLE_RETRIEVAL_QUERY_GENERATION=false ENABLE_SEARCH_QUERY_GENERATION=false ENABLE_EVALUATION_ARENA_MODELS=false OFFLINE_MODE=true DATA_DIR=/tmp/mference-phase2-ui.78zbs3/data WEBUI_SECRET_KEY=mference-isolated-qualification-only-not-for-reuse /Users/studio2/.local/share/uv/tools/open-webui/bin/python Scripts/openwebui-mference.py serve --host 127.0.0.1 --port 18490
python3 Scripts/openwebui-configure-models.py --webui http://127.0.0.1:18490
```

The configuration helper exited 0. Both services stopped with exit 0. Browser
automation used agent-browser 0.27.0 with the named isolated session; it was
closed after testing. Screenshots and logs are in the temporary test directory.
No console errors were reported during the functional checks.

- Picker displayed distinct base/Swift IDs. Swift loaded on selection/request.
- Controls → Advanced Params → Reasoning Effort forwarded `low` (confirmed
  in the browser request); thinking and visible answer rendered separately.
- Four arithmetic turns answered 4 → 7 → 8 → 6 correctly. After UI process
  restart and browser reload, reasoning history remained available.
- The fourth turn reused 187 exact-prefix tokens. The initial adapter attempt
  missed Open WebUI's nested `openai.owned_by` identity and reused zero tokens;
  the corrected adapter and regression test cover the actual merged shape.
- Switching effort to `none` answered 5 with no new reasoning block.
- Stopping a long number-list generation cancelled the server request; a
  subsequent question answered 7 correctly with a fresh cache.

Complete relevant server timing/error lines:

```text
[2026-09-16T23:28:20Z] request chatcmpl-aa6764be95a7491fa51a6cdb4f40e6cd completed in 8.9s prompt=54 cached=0 completion=26 finish=stop
[2026-09-16T23:28:31Z] request chatcmpl-d6e984608b2946f9b28fc5fbde723be6 completed in 1.7s prompt=83 cached=0 completion=39 finish=stop
[2026-09-16T23:29:42Z] request chatcmpl-ce5376c7002d4f3b994ea5405f40d511 completed in 1.7s prompt=169 cached=0 completion=19 finish=stop
[2026-09-16T23:30:07Z] request chatcmpl-ddb7412218434a30906420b5d3f77923 completed in 1.1s prompt=210 cached=187 completion=20 finish=stop
[2026-09-16T23:31:15Z] request chatcmpl-d7bd2052014d482b990063a28ea48e2c completed in 1.3s prompt=224 cached=0 completion=2 finish=stop
[2026-09-16T23:31:48Z] request chatcmpl-dc5ff5c036314d5393386348c397e7b1 failed status=500 streaming=true error=CancellationError()
[2026-09-16T23:32:11Z] request chatcmpl-da50d1cbfdc646ddaff496fecd2b4610 completed in 3.3s prompt=614 cached=0 completion=2 finish=stop
```

Cancellation currently appears as a 500-class operator-log error after the
client has disconnected; this was an intentional cancellation, not a model
failure. No time above is a throughput or comparative latency benchmark.
Native Open WebUI tool execution and other package versions are not qualified
by these browser checks; the direct server tool-result loop passed separately.

## Expanded installed-model gate

```bash
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer MFERENCE_SWIFT_QWEN_GTURBO=/Users/studio2/Documents/ChatGPT/Mference/scratch/swiftqwen38.gturbo MFERENCE_MTP=1 Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs --filter SwiftQwenInstalledQualificationTests
```

Build complete! (9.71s). Exit **1**. Complete test footer:

```text
✘ Test prefillAppendAndSpeculativeContinuation() failed after 72.727 seconds with 2 issues.
✘ Suite SwiftQwenInstalledQualificationTests failed after 72.727 seconds with 2 issues.
✘ Test run with 1 test in 1 suite failed after 72.727 seconds with 2 issues.
```

Log: `/tmp/mference-phase2-installed-qualification.log`. Production plain
decode versus 64-row chunked prefill, prompts of 65/257/1025 tokens, a warm
append at position 33, and five subsequent full-vocabulary state probes each.
Predeclared limits: max absolute logit error ≤0.25, mean absolute ≤0.01,
finite logits and matching top-1. Both failing rows were final prefill heads:

| Prompt | Max absolute | Mean absolute | Top-1 | Result |
| --- | ---: | ---: | --- | --- |
| 65 | 0.20703125 | 0.013872731 | Same | Fail mean limit |
| 257 | 0.029296875 | 0.004467301 | Same | Pass |
| 1025 | 0.18652344 | 0.014277136 | Same | Fail mean limit |

All 15 state-probe rows passed (worst max 0.15039062, worst mean 0.007820662).
No threshold was loosened after execution. This historical failure was corrected
and retested on September 18 above; the suite remains opt-in.

In the same run, explicit native MTP stop/rewind checks at 2/7/32/64 generated
tokens matched plain greedy output; all 20 full-logit state probes after cursor
reconciliation were **bit-identical**. Drafted/accepted counts: 3/1, 9/5,
36/30, 87/79; rounds 1/3/12/29 and rollbacks 1/2/3/4. One counting prompt is
not a representative acceptance-rate or hardware-performance benchmark.
MTP remains disabled by default. No experimental prefill controls or profiling
were enabled; explicit MTP was used only for correctness qualification.

## Follow-up engineering checks

```bash
python3 Scripts/tests/test_openwebui_mference.py
python3 Scripts/tests/test_swift_qwen_task_screen.py
bash -n mference-ui.sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift build -c release --scratch-path /tmp/mference-phase1-build.sXnNTs
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs --filter 'SwiftQwenChatTests|QwenToolCall|MapleTokenization|MapleToolCall|ChatMLTemplate'
```

All exited 0. Python suites: 3 tests / OK each. Release: Build complete!
(61.93s). Swift: 53 tests in 6 suites passed after 0.170 seconds. This extends,
not replaces, the earlier full 1,215-test run. The explicitly enabled real-model
gate above failed; it is not silently included in the passing-suite claim.

## Frozen 12-task screen

Protocol and fixed rubrics: [swift-screen-v1](../benchmark-prompts/swift-screen-v1/README.md).
Corpus SHA-256: `2c7df5d0e146f25cbdc88df357207b716d5e4c155ad2b63d8043e086096cfded`.

```bash
env MFERENCE_MTP=0 /tmp/mference-phase1-build.sXnNTs/release/MferenceServer --library scratch/swiftqwen38.gturbo --library scratch/qwen38.gturbo --port 18489 --max-context 4096
python3 Scripts/swift_qwen_task_screen.py --output /tmp/mference-phase2-task-screen-v1-final.jsonl
```

Screen exit 0 (all records written, not all passed); server stopped with exit
0. Complete requests/responses, usage, wall times and failure answers are in
that JSONL. Server log: `/tmp/mference-phase2-task-screen-server-final.log`.

| Profile | Exact successes | Total completion tokens, including reasoning |
| --- | ---: | ---: |
| Base Qwen | 12/12 | 1049 |
| Swift medium | 12/12 | 998 |
| Swift xhigh (source default) | 12/12 | 865 |
| Swift low | 12/12 | 892 |
| Swift none | 10/12 | 146 |

Thinking-off returned 68 instead of 66 for discount-then-tax, and 8 instead
of 20 for a Python comprehension. Do not recommend it as a quality-preserving
default from its shorter output. The xhigh total was 17.5% below base on these
12 cases only; this is not a general savings claim, and tool cases have
different reasoning policies in the legacy base template. The more closely
matched medium total was 4.9% below base. No formal latency claim: this screen
does not use community benchmark warmups/repeats and includes model swaps.

Deviations / earlier attempts: an initial batch used server-invalid top-k 0
and recorded HTTP 400s only; corrected to top-k 1 before generated answers.
The next batch exposed base Qwen's XML-versus-JSON parser-selection bug (10/12,
two server errors), which was fixed by explicit family grammar selection.
The table above is the full repeated batch after that fix, not a mixture of
favorable cases from different revisions. Original evidence is preserved in
`/tmp/mference-phase2-task-screen-v1.jsonl` and
`/tmp/mference-phase2-task-screen-v1-corrected.jsonl`.

September 16 conclusion: integration, bounded functional comparison and local MTP
stop/rewind evidence were available. At that time the numerical screen, wider task
coverage, native UI tool execution, long contexts and other hardware profiles
remain open. Base installation/default recommendation and Swift's MTP-off
default are unchanged.
