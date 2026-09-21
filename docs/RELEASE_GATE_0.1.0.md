# Source release 0.1.0: final gate

Release target: `4ff63ff555c137d924f024cd3fad8e0b486959f2` (the merge of
[PR #38](https://github.com/NeelM0906/Mference/pull/38)). This record is separate
from the historical candidate records. It does not qualify the entire
optimization roadmap or new hardware profiles.

Publication requires success of the exact-ref
[source release workflow](https://github.com/NeelM0906/Mference/actions/runs/35639217168)
and verification of its downloadable assets against the source below.
[GitHub Releases](https://github.com/NeelM0906/Mference/releases) is the
authoritative publication state; a packaging manifest or this record alone is
not a publication announcement. The separate
[merged-main CI](https://github.com/NeelM0906/Mference/actions/runs/35625622315)
and [PR-head CI](https://github.com/NeelM0906/Mference/actions/runs/35611271559)
both completed successfully on their exact revisions.

## Release scope

Swift/Metal source, installer, CLI, loopback server and the launcher for pinned
Open WebUI 0.11.3. Apple Silicon, macOS 15+, Swift 6.1+; `uv` is required only
for the UI. No signed app, prebuilt executable or model weights are included.
Gemma 4 remains the first-use recommendation; Qwen 3.6 is the established MoE
alternative. See [checkpoint choices](RELEASE_SUPPORT.md).

The following are **not** gates passed by this release:

- Native Flash-Next MTP: internal foundations only, disabled in generation;
  independent full-native reference, verification/acceptance and benefit remain
  open. Its narrow row-zero numerical known issue is still disclosed.
- Swift-Qwen replacement promotion: optional separate checkpoint, not a
  replacement recommendation. Broader task/latency and hardware evidence remain.
- GLM completed-answer performance: prior default-profile truncations are
  failures, not fast answers; no new speed claim is made here.
- Physical smaller-Mac, all-context and old-OS installed-model qualification:
  a 256 GiB host and hosted CI cannot substitute for those measurements.

## Exact source artifact

- Archive: `mference-0.1.0.tar.gz`, 15,898,066 bytes.
- SHA-256: `1299845e54c6c070c174036b253f2032537eaf25b19a0d22ac6c7346d7928498`.
- Git tree: `0fffd9699966aea8c92484921655e9f4b8487375`.
- Entries: 861 source entries, 862 including the archive root directory.
- Companion files: `mference-0.1.0.json` and `mference-0.1.0-SHA256SUMS`.

Local packaging and checksum commands (exit 0):

```sh
python3 Scripts/package_source_release.py --version 0.1.0 \
  --output scratch/release-candidates/0.1.0
cd scratch/release-candidates/0.1.0
shasum -a 256 -c mference-0.1.0-SHA256SUMS
tar -xzf mference-0.1.0.tar.gz -C /tmp/mference-release-0.1.0.pQ7cNL
```

Both checksum entries report `OK`. No untracked execution plan, installed
model, UI data or local build artifacts enter the archive.

## Local environment and extracted-source checks

Mac Studio Mac15,14, M3 Ultra (32 CPU cores), 256 GiB; macOS 26.3 (25D125).
`DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer` selects Swift 6.3.3
(`swiftlang-6.3.3.1.3`). Preflight: 615 GiB disk free, 97–98% memory-pressure
free, no model/test owner. Gemma's 37 and Qwen 3.6's 47 receipt-file sizes
match before loading; inference uses default full-SHA verification.
No weights downloaded or duplicated, no caches purged, no other apps stopped,
no profiling or experimental controls enabled.

All following commands run in the extracted
`/tmp/mference-release-0.1.0.pQ7cNL/mference-0.1.0`, not the development tree.

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer ./mference-ui.sh doctor
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  swift build -c release --force-resolved-versions
```

Both exit 0. Build log `/tmp/mference-v010-build.log`:

```text
Build complete! (98.57s)
```

The three executable help commands exit 0. Script commands, all exit 0:

```sh
python3 Scripts/tests/test_mference_ui.py
python3 Scripts/tests/test_openwebui_mference.py
python3 Scripts/tests/test_release_task_screen.py
python3 Scripts/tests/test_release_screen_summary.py
python3 Scripts/tests/test_qwen_efficiency_screen.py
python3 Scripts/tests/test_swift_qwen_task_screen.py
python3 Scripts/tests/test_source_release.py
python3 -O Scripts/tests/test_source_release.py
LANG=en_US.UTF-8 ruby Scripts/check_markdown_links.rb
```

Test counts: 8 / 5 / 7 / 4 / 2 / 3 / 6 / 6. All 74 distributed Markdown files
pass. The deliberately wrong-version UI fixture prints its expected refusal;
the five adapter tests pass.

### CLI smoke: raw mode is not a chat-quality test

```sh
.build/release/MferenceCLI \
  --model /Users/studio2/Documents/ChatGPT/Mference/scratch/gemma4.gturbo \
  --prompt 'The capital of France is' --max-new 32 --temperature 0 --max-context 4096
```

Exit 0, but this untemplated raw completion produces irrelevant text and is
**not** counted as a useful-answer pass. Complete footer:

```text
[stop=maxTokens prefill=6tok/4.95s new=32tok decode=0.96s tok/s=33.334]
```

These smoke-test timings are not community-protocol performance results.
They use neither the frozen prompts nor its discarded warmup/three-run
method, and establish no throughput recommendation.

### CLI chat-template check

`/tmp/mference-v010-chat.json` contains:

```json
[{"role":"user","content":"What is 17 plus 25? Reply with only the number."}]
```

```sh
.build/release/MferenceCLI \
  --model /Users/studio2/Documents/ChatGPT/Mference/scratch/gemma4.gturbo \
  --messages-file /tmp/mference-v010-chat.json \
  --max-new 32 --temperature 0 --max-context 4096
```

Exit 0; answer `42`, log `/tmp/mference-v010-cli-chat.log`, complete footer:

```text
[stop=endOfTurn prefill=29tok/5.04s new=3tok decode=0.07s tok/s=40.429]
```

### Extracted-source browser and API gate

Story: an extracted release starts the launcher, the browser sends a chat to
the native server, the selected installed checkpoint generates, and the UI
renders and persists the response. Fresh isolated data keeps existing chats
untouched.

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer ./mference-ui.sh \
  --data-dir /tmp/mference-v010-ui.JFuCax/data \
  --library /Users/studio2/Documents/ChatGPT/Mference/scratch/gemma4.gturbo \
  --library /Users/studio2/Documents/ChatGPT/Mference/scratch/qwen36.gturbo \
  --server-port 18491 --webui-port 18492 --max-context 4096
```

Browser automation used installed `agent-browser` 0.27.0, session
`mference-v010`; no browser dependency or checkpoint was downloaded.
First navigation was attempted during the fresh database migration and returned
`ERR_CONNECTION_REFUSED`. After readiness, the one retry loaded correctly.
The upstream first-run release-notes dialog was closed normally. Both listeners
were verified as `127.0.0.1` only; `/health` returned `status=ok`. Browser error
collection was empty, the page had content and no framework error overlay.

| Boundary | Result |
| --- | --- |
| Browser → Gemma → rendered answer | `17 + 25` → `42` |
| Full history → prefix reuse | Add 8 → `50`; server reports 31 cached tokens |
| Persisted data → browser reload | Both prompts and answers remain visible |
| Browser model switch → Qwen 3.6 | `6 × 9` → `54` |
| Qwen tool envelope → client-supplied result | `get_weather` returns `{"city":"Paris, France"}` with a call ID; appending a simulated 21°C result produces a completed 21°C answer and 297 cached tokens |
| Switch back → streamed response → Stop | Gemma visibly streams the number list; Stop cancels the request |
| Cancellation → next request | `3 + 4` → `7`; no stale/dirty-state error |

The weather result is a test fixture, not a real weather lookup; no model tool
was executed. Its client asserts `finish_reason=tool_calls`, one known function,
JSON-object arguments and the matching call-ID history, then `finish_reason=stop`
and the fixture value in the completed answer. Command
`python3 /tmp/mference-v010-api-smoke.py` exits 0; log
`/tmp/mference-v010-api-smoke.log`.

The interrupted number-list output contains a model error (`99` in place of
`49`); it is used only to test streaming/cancellation, not scored as a correct
counting task or broader model-quality evidence.

Complete server timing/cancellation footers:

```text
[2026-09-21T18:39:21Z] request chatcmpl-edfb05806b51498cb60a3cbd90f2586d completed in 7.0s prompt=29 cached=0 completion=3 finish=stop
[2026-09-21T18:39:35Z] request chatcmpl-14716eda02494800bb9a855e9433eefd completed in 0.4s prompt=59 cached=31 completion=3 finish=stop
[2026-09-21T18:40:25Z] request chatcmpl-0a99882c91304d0b81f77e71b6db7732 completed in 9.2s prompt=80 cached=0 completion=3 finish=stop
[2026-09-21T18:40:54Z] request chatcmpl-835d62d550fa41bebb5287d3c72d8402 completed in 3.0s prompt=270 cached=0 completion=28 finish=tool_calls
[2026-09-21T18:40:54Z] request chatcmpl-55d34cb734944c2386a55953cf077d28 completed in 0.8s prompt=331 cached=297 completion=19 finish=stop
[2026-09-21T18:41:36Z] request chatcmpl-2b0cb9b9113746f0a140c2d8d14eca99 cancelled streaming=true
[2026-09-21T18:41:51Z] request chatcmpl-365cadad0fba407784820ece47c2b491 completed in 2.0s prompt=432 cached=0 completion=2 finish=stop
```

Evidence directory `/tmp/mference-v010-ui.JFuCax` contains `launcher.log`,
`first-load.png` and `recovered.png`; both screenshots were inspected.
Exactly one model owner, server PID 53239, served the entire run. Browser
closed; the verified task-owned launcher PID 53012 received TERM (exit 143)
and its trap stopped server 53239 and UI 53247. No model owner remained before
the subsequent serial package test. The isolated database is retained.
Upstream Python emitted a one-semaphore `resource_tracker` warning on shutdown;
it did not prevent service cleanup or chat persistence.

### Full serial extracted-source suite

After stopping the task-owned services, fresh preflight again reported 98%
memory free, 615 GiB disk and no model/test owner. Command in the same extracted
tree:

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  Scripts/test.sh > /tmp/mference-v010-tests.log 2>&1
```

Exit 0. Complete timing footer:

```text
Build complete! (44.07s)
Test run with 1297 tests in 233 suites passed after 280.004 seconds with 2 known issues.
```

The two known issues are the existing optional absent Flash-Next toy checkpoint
and the disabled native draft's narrow row-zero FP32 comparison. They are not
new failures or passed native-MTP gates. Installed-model environment gates were
unset; actual Gemma/Qwen CLI/server/browser checks are reported separately above.
No model/test process remained after completion.

## Publication procedure and remaining roadmap

1. Require the exact-ref source workflow to finish successfully, including both
   macOS legs and the extracted-archive job. Do not substitute a previous run.
2. Download its three artifacts, verify both checksums, and check manifest
   commit/tree/version and the archive against the locally tested source.
3. Publish the immutable `v0.1.0` tag at `4ff63ff` with the three verified files
   and notes that retain the exclusions above. Do not retarget an existing tag.
4. Download the published assets again and recheck their bytes and manifest.
   Confirm the public release and tag resolve to the expected commit.

The release page and workflow are the durable external results of these steps.
Documentation-only follow-up commits do not change the source selected for this
release. The [prefill/roadmap matrix](PREFILL_QUALIFICATION.md#roadmap-status)
continues to track unqualified work rather than marking it complete because
an archive can be downloaded. In particular, there is no claim of native MTP
generation, a universal prefill/hardware contract, or new completed-answer
performance for GLM.
