# Release qualification — September 21, 2026

Continuation of [September 20](RELEASE_VALIDATION_2026-09-20.md), in
[PR #38](https://github.com/NeelM0906/Mference/pull/38), targeting `main`.
The starting head `8d65245be0c895ff3c3881718f8f1762350d5f05` passes macOS 15 /
Swift 6.1, macOS 26, documentation and security
([CI run](https://github.com/NeelM0906/Mference/actions/runs/35546983391)).
That result does not certify newer changes. No tag or release was published.

## Environment and safety

Mac Studio Mac15,14, M3 Ultra (32 CPU cores), 256 GiB; macOS 26.3 (25D125).
Tests select `/Applications/Xcode.app/Contents/Developer`, Apple Swift 6.3.3
(`swiftlang-6.3.3.1.3`), rather than the shell's default Swift 6.2.4. Initial
memory check recovered to 97%; fresh test checks report 98% free, 619 GiB disk,
and no model/test/installer owner. The prior GLM benchmark safety stop remains
recorded, not erased or reinterpreted as a pass. No apps were terminated,
weights downloaded/copied, caches purged or experimental/profiling controls
enabled. These are correctness checks, not performance measurements.

## Native draft alignment

Implementation commit: `5af3c97` (tests developed on `8d65245` plus the changes
recorded in that commit).

The internal primer preserves a pending target HC row until its successor
token is known. It uses shared BF16/INT4 embeddings and fills all preceding
native draft KV rows at the original target positions. Finalization requires
an explicit next token, and the next target append must start with that token.
Foreign/stale checkpoints, gaps, duplicate finalization, invalid tokens,
insufficient buffers and context overflow reject. Partial priming failure
marks the primer dirty; paired target/primer restoration reproduces the clean
target and draft logits exactly. Ordinary CLI/server paths do not construct
this component or perform these additional copies/computations.

Initial development failed compilation because a nested test-fixture mapping
exceeded Swift's type-checking budget; it was replaced by explicit loops.
Command below, log `/tmp/mference-mtp-priming-20260921.log`, exit 1, final error
`the compiler is unable to type-check this expression in reasonable time` /
`error: fatalError`. No model test ran in that attempt.

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs \
  --filter 'FlashNextMTPPrimingTests|FlashNextMTPDraftRunnerTests'
```

After the fixture fix, the first pass exited 0: build 6.32s, 11 tests / 2 suites, 6.675s,
one known issue (`/tmp/mference-mtp-priming-v2-20260921.log`). Adding the paired
target/primer recovery test and opt-in installed probe then exited 0:

```text
Build complete! (6.01s)
Test run with 13 tests in 2 suites passed after 6.839 seconds with 1 known issue.
```

Log `/tmp/mference-mtp-priming-v3-20260921.log`. Installed environment gate was
unset, so its early return is **not** an installed-model pass. Both resident
and bounded toy drafts match independently single-row-fed shifted pairs
exactly across partitions `[40]`, `[32,8]`, `[1,31,1,7]` and forty single rows.
Actual toy target prefill (35 tokens in chunks 32/3), warm append (7), and
decode (1) prime all 43 native rows without target replay. Replaying those same
captured HC rows manually reproduces the final hidden bundle and full logits
exactly; it does not claim equivalence to a different target arithmetic path.

The existing unrounded-FP32 row-zero precision expectation remains the one
known issue, with the unchanged 5% gate and hard failure above 6%. No tolerance
was relaxed. Native MTP remains disabled and unqualified for proposal
verification, end-to-end generation, quality or speed.

### Installed target-to-native-draft check

At runtime/test commit `5af3c97`, the same environment and fresh 98%-free /
619-GiB single-owner preflight, all 57 receipt-file sizes checked before loading:

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  MFERENCE_FLASHNEXT_GTURBO=/Users/studio2/Documents/ChatGPT/Mference/scratch/qwen38flashnext-r8.gturbo \
  Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs \
  --filter installedTargetRowsPrimeNativeDraft \
  > /tmp/mference-mtp-priming-installed-20260921.log 2>&1
```

Exit 0; complete timing footer:

```text
Build complete! (1.51s)
Test installedTargetRowsPrimeNativeDraft() passed after 32.071 seconds.
Suite FlashNextMTPPrimingTests passed after 32.071 seconds.
Test run with 1 test in 1 suite passed after 32.071 seconds.
```

The existing completed INT8-router install loads with full SHA verification;
target and native auxiliary pool both use 16 slots in the same process. All
43 captured actual target HC rows are finite/nonzero. Cold chunks 32/3, warm
append 7 and one decode row produce the same final native hidden bundle and
full-vocabulary logits as manually feeding those rows with explicitly shifted
tokens, byte-for-byte. Target prefill reports zero replay. This closes the
installed alignment component gate, **not** upstream full-native numerical
parity, proposal verification, accepted-token output equivalence or speed.

## Source-release delivery tooling

`Scripts/package_source_release.py` produces a versioned source tarball,
commit/tree manifest and SHA-256 checksum file from committed source only.
`.github/workflows/source-release.yml` runs the reusable two-platform CI before
uploading these as a downloadable candidate artifact. It has read-only
repository permissions and no publish/tag/merge step. Documentation in
[Source release](SOURCE_RELEASE.md#build-a-downloadable-source-candidate)
separates packaging validation from model/hardware qualification.

`python3 Scripts/tests/test_source_release.py` and the same command with `-O`
both exit 0, six tests each (1.109s and 1.148s in the first final check).
Coverage includes repeated byte-identical artifacts, manifest/checksum
verification, executable modes, exclusion of untracked models/secrets,
refusal of tracked edits/overwrites/bad versions, missing licenses, unsafe
paths, symlinks, special entries and duplicate tar members. Archive safety
checks use explicit exceptions and therefore still execute with Python `-O`.
These packaging-only checks ran while the installed correctness gate ran;
there is no performance claim from either duration.

## Extracted source, not a development checkout

The first local candidate, `0.1.0-rc.1`, packages commit
`866cf49bd04066336e0e4ceea33c5f69c9d6eacf` (runtime `5af3c97`). Its source
archive SHA-256 is
`f25e0dadb156388ca207368e03411a1a213fa1f263917c233c96d9a7caccd819`.
`0.1.0-rc.2` packages `6ba01c721a1ad22547f7be210e71498df6d9071e` with the
subsequent extracted-archive CI step and stricter prerelease-label validation;
its archive SHA-256 is
`476512f4e75f4496b60fcf442724c816132b6caec022424c1f4bec069dd44c05`.
Both checksum files verify. They are local candidate labels, not published tags.
The latter's complete runtime source/resources are byte-identical to the former
(an untracked `.DS_Store` in the development checkout is correctly absent).

Packaging command (repeat with `rc.2` for the second candidate):

```sh
python3 Scripts/package_source_release.py --version 0.1.0-rc.1 \
  --output scratch/release-candidates/0.1.0-rc.1
```

The validated tarball was extracted into
`/tmp/mference-source-smoke.IqCbHj/mference-0.1.0-rc.1`, with no `.git`,
development build directory, local model installs or untracked documents.
From that directory:

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer ./mference-ui.sh doctor
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  swift build -c release --force-resolved-versions \
  > /tmp/mference-source-build-20260921.log 2>&1
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  Scripts/test.sh > /tmp/mference-source-tests-20260921.log 2>&1
```

All exit 0. Doctor checks pinned Open WebUI 0.11.3 without opening its database;
release build uses the committed dependency lock. Fresh serial-test preflight:
98% memory free, 618 GiB disk, no model/test/installer owner; no installed-model
environment gates. Complete build/test footers:

```text
Build complete! (98.97s)
Build complete! (44.28s)
Test run with 1295 tests in 232 suites passed after 280.105 seconds with 2 known issues.
```

Known issues are the same optional missing toy checkpoint and disabled native
draft's row-zero precision gate, not newly hidden failures. The extracted
CLI/server/repacker `--help` commands each exit 0. Launcher syntax and Python
checks pass: launcher 8, adapter 5, task-screen 7, summary 4, Qwen efficiency 2,
Swift task-screen 3 and source packaging 6. All 74 distributed root/docs
Markdown files resolve their local links/anchors. These non-model checks ran
during part of the full correctness run; no performance inference is made.

## Browser → API → installed model → saved answer

The actual launcher and server from the extracted `rc.1` archive were used,
not the development binaries. Same hardware/toolchain, 98% memory free,
617 GiB disk and no model owner before launch; all 37 Gemma / 47 Qwen 3.6
receipt-file sizes verified. Default full SHA model verification remains on.
No second model process or full checkpoint download. Browser automation is
the already-installed `agent-browser` 0.27.0, session `mference-source-release`.

From the extracted source directory:

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer ./mference-ui.sh \
  --data-dir /tmp/mference-source-ui.8ycUoa/data \
  --library /Users/studio2/Documents/ChatGPT/Mference/scratch/gemma4.gturbo \
  --library /Users/studio2/Documents/ChatGPT/Mference/scratch/qwen36.gturbo \
  --server-port 18491 --webui-port 18492 --max-context 4096
```

Incremental server build exits 0: `Build of product 'MferenceServer' complete! (3.80s)`.
`lsof` confirms both listeners bind **127.0.0.1**, ports 18491 and 18492. An
early readiness probe received connection-refused before UI startup; the UI
health endpoint subsequently returned `{"status":true}`. After client bootstrap
and dismissing the upstream first-run release-notes dialog, the page contains
the chat composer and both installed models, with no browser errors or error
overlay. The browser skill was used to check the visible flow, not just `/health`.

| Boundary | Result / evidence |
| --- | --- |
| UI → local API → Gemma | `17 + 25` renders `42`; normal end-of-turn |
| Follow-up / prefix reuse | `Add 8` renders `50`; server reports 31 cached prompt tokens |
| History persistence | Reload preserves both messages and answers |
| Library switch | Gemma → Qwen 3.6 answers `42` for `6 × 7`; one model owner |
| Return switch | Qwen → Gemma answers `21` for `9 + 12` |
| Streaming cancellation | Number list is visibly streaming before Stop; partial output ends at 67 |
| Recovery | Next request answers `7` for `3 + 4`, normal end-of-turn |

Complete generation footers (all from the same log):

```text
[2026-09-21T14:01:53Z] request chatcmpl-ef54e4cf8ce94990976c761adf6d2776 completed in 8.1s prompt=29 cached=0 completion=3 finish=stop
[2026-09-21T14:02:13Z] request chatcmpl-63e1f8be7d2b421da1753fc1936dfd51 completed in 0.4s prompt=59 cached=31 completion=3 finish=stop
[2026-09-21T14:02:53Z] request chatcmpl-42b6a88a753c4d6db7c6af310c22421c completed in 10.1s prompt=81 cached=0 completion=3 finish=stop
[2026-09-21T14:04:08Z] request chatcmpl-b734f07e39cd44bc91aebd5a2db0a12c completed in 6.0s prompt=109 cached=0 completion=3 finish=stop
[2026-09-21T14:04:43Z] request chatcmpl-b27d5c2660cb4e24aa8343e1fd788c66 failed status=500 streaming=true error=CancellationError()
[2026-09-21T14:04:58Z] request chatcmpl-33893a25b0d44c33b75713cbdd6acd30 completed in 2.0s prompt=423 cached=0 completion=2 finish=stop
```

A browser-tool `wait --state hidden` invocation was interpreted as the CLI's
global state-file option and reset the automation session. This was a tooling
failure, not a server failure: the completed Qwen answer persisted, and opening
the observed chat URL again recovered it. Subsequent waiting used a DOM
predicate. No user browser/profile or existing UI database was reused.

Screenshots (`first-load.png`, `answer.png`, `recovered.png`) and `launcher.log`
are retained under `/tmp/mference-source-ui.8ycUoa`. Final browser error list
is empty. The task-owned browser closed and launcher PID 31357 received TERM;
its cleanup stopped only its server/UI children, exit 143. A subsequent owner
check returned no model process. Open WebUI emitted a Python resource-tracker
warning about one semaphore at shutdown; it did not prevent process cleanup
or corrupt the test history. These interactive timings are not benchmarks.

### Cancellation-log correction after the browser check

The old footer above incorrectly categorized expected cancellation as a
500-class operator failure. `ServerLog` now labels `CancellationError` as
`request <id> cancelled streaming=<bool>`, without pretending to know whether
the cause was disconnect or shutdown. Genuine failures retain their original
status and details. This changes only operator log classification, not
HTTP/SSE status, error envelopes, cancellation handling or model state.

After a fresh 98%-free / 617-GiB single-owner preflight:

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs \
  --filter 'ServerLogTests|HTTPServerTests' \
  > /tmp/mference-cancellation-log-20260921.log 2>&1
```

Exit 0: 27 tests / 3 suites in 3.008s, including HTTP disconnect cancellation,
stream/non-stream formatting, real-error preservation and library HTTP behavior.
The preceding browser run used the old logger; it is not relabelled as a live
test of the new message.
