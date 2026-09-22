# September 22 native MTP and GLM qualification

Work starts at `364a5d021ad35c117e8cc9bd47a4d1b136bb91e8` on
`codex/postlaunch-qualification`, updating [PR #41](https://github.com/NeelM0906/Mference/pull/41)
against `main`. The published 0.1.0 tag/assets are not changed.

## Current outcome

- Native MTP: the independent installed 43-row comparison and complete
  installed internal correctness suite pass after the FP32 fixes below.
  Earlier failures in this chronological record are superseded by the
  explicitly identified final run, not erased or counted as successes.
- GLM: resident low-effort matched performance, resident Max/8,192 and
  bounded 16-slot low-effort completed-answer qualification pass. Both new
  profiles complete 9/9 measured cases; Max/1,024 truncations remain failures.
- Release engineering: exact-code macOS 15/26 CI and the clean extracted
  source-candidate workflow pass at `ef9484d`.
- Physical 16/24-GiB testing is waived as a release blocker, not passed.
- Native accelerated CLI/server generation remains disabled and unqualified.
  Numerical verification is not completed-answer speed/default promotion;
  this record does not declare the whole optimization roadmap complete.

## Scope and host

Per the explicit user decision, physical 16/24-GiB evaluation is **not a
release blocker**. Those profiles remain unqualified; a slot-limited run on
this machine is not a substitute for them.

Mac Studio Mac15,14, M3 Ultra, 32 CPU cores, 256 GiB; macOS 26.3 (25D125),
Apple Swift 6.3.3 (`swiftlang-6.3.3.1.3`, clang 2100.1.1.101), selected with
`DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`. Preflight: 612–613
GiB disk free, 98% memory-pressure free and no model/test owner. All 49 GLM
and 57 Flash-Next receipt-file sizes match. Swift installed checks use full
SHA verification. No checkpoint download, model duplication, cache purge,
other-app termination or published-release replacement.

## Native fusion fix and committed upstream regression

Normalization and both projection intermediates now stay FP32; only the
final broadcast sum becomes FP16. The unchanged synthetic 5% assertion is
now unconditional, with no known-issue exemption. Row-zero hidden error:
`0.23217869 / 10.497804 = 2.2117%`, previously 5.263%. All 40 hidden/logit
rows and greedy choices pass. BF16, INT4 and INT8 kernel cases at D=64/2560,
including nonzero weight/companion/norm offsets, pass (six cases, maxAbs=0).

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  MFERENCE_MTP_REFERENCE_EXPORT=/tmp/mference-mtp-reference-20260922.json \
  Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs \
  --filter 'FlashNextMTPInputFusionTests|FlashNextMTPDraftRunnerTests'
```

Exit 0; build 14.11s; seven tests/two suites pass in 6.111s.
Log `/tmp/mference-mtp-fp32-20260922.log`. After adding the committed
upstream golden, `--filter 'FlashNextMTP'` exits 0: build 7.36s,
25 tests/five suites in 10.599s (`/tmp/mference-mtp-all-v2-20260922.log`).

The independent Python reference uses CPU torch 2.14.0, Python 3.12.12,
Transformers `4da05482135896a529d5536c3c003102d36528a2`, and the actual
SGLang fusion method from `745de73ba3c136b6f99b7a3e2177ed1a8eef4a56`.
Source SHA-256 is checked before AST extraction. Decoder, QSA/cache, HC,
norms and mixer execute upstream code, not the Swift scalar transcription.
See [reproduction instructions](../Scripts/parity/README.md#native-mtp-component-reference-september-22).

Unmodified CPU top-k fails synthetic row 39 (21.8485% logits); seven block
scores are exactly zero, producing a different tied support set. The explicitly
selected lowest-index tie diagnostic passes all 40 rows: worst hidden
2.2122%, logits 0.4004%, all greedy choices equal. It asserts that selected
score values are unchanged. The raw failure remains recorded; the adapted
result is not presented as an unmodified SGLang CUDA run. The adapted upstream
outputs are committed for ordinary Swift CI, with the tie policy explicit.

## Installed state and same-weight reference

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  MFERENCE_FLASHNEXT_GTURBO=/Users/studio2/Documents/ChatGPT/Mference/scratch/qwen38flashnext-r8.gturbo \
  Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs \
  --filter 'FlashNextMTP'
```

Serial rerun exits 0: build 1.40s; 25 tests/five suites in 132.120s.
Log `/tmp/mference-mtp-installed-serial-20260922.log`. Installed bounded and
resident replay/reset are exact. Actual proposals retain exact target tokens
and full heads through acceptance/rejection, stop, budgets, interrupted-round
recovery and priming. Natural chat stops at 51 tokens, 20/48 draft guesses
accepted (target-selected seeds excluded), with the same two-sentence answer.
These are correctness timings, not community performance measurements.

Protocol deviation: brief CPU-reference diagnostics overlapped the first
installed correctness run (16 tests, 130.923s). That is not the required serial
protocol; the complete installed native suite was rerun alone as above.

The reference reader reads only the existing MTP sidecar and shared head,
dequantizes weights in memory and checks receipt sizes. It neither copies nor
downloads the model. Two deterministic installed input rows pass unadapted
upstream comparison: worst hidden 0.3070%, logits 0.3650%, greedy choices equal.
Capture command adds
`MFERENCE_MTP_INSTALLED_REFERENCE_EXPORT=/tmp/mference-mtp-installed-capture-20260922.json`
and filters `FlashNextMTPDraftRunnerTests/installedDraftExecutesAndRestoresItsOwnState`;
exit 0, build 9.74s, one test in 3.370s.

The stronger aligned capture uses:

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  MFERENCE_FLASHNEXT_GTURBO=/Users/studio2/Documents/ChatGPT/Mference/scratch/qwen38flashnext-r8.gturbo \
  MFERENCE_MTP_ALIGNED_REFERENCE_EXPORT=/tmp/mference-mtp-aligned-capture-20260922.json \
  Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs \
  --filter 'FlashNextMTPPrimingTests/installedTargetRowsPrimeNativeDraft'
scratch/qwen4exp-parity-venv/bin/python Scripts/parity/verify_flashnext_mtp.py \
  /tmp/mference-mtp-aligned-capture-20260922.json \
  --sglang-source /tmp/mference-upstream-qwen4_exp_mtp.py \
  --install scratch/qwen38flashnext-r8.gturbo
```

Swift capture exits 0: build 12.48s; one test in 35.052s. It covers 43 actual
target HC rows across cold prefill, warm append and decode, using deterministic
test tokens, not a natural-language quality benchmark. Capture SHA-256:
`987f69dd12f99dad45c67c3d16123c08f0b0e35e7f0ff72077d64eb42c60cb04`.

**Independent aligned qualification fails (Python exit 1).** There are no QSA
boundary ties. Row 1: hidden 9.1242%, logits 9.2322%; row 33: hidden 5.6032%,
logits 8.1395%. Greedy choices differ on rows 1, 28 and 29. At row 1 the tenth
expert is 230 upstream versus 119 in Metal; at row 33 it is 279 versus 147.
Small pre-router FP16 drift changes the discrete route, amplifying the output
error. Logs `/tmp/mference-mtp-aligned-upstream-20260922.log` and
`/tmp/mference-mtp-aligned-upstream-routes-20260922.log` preserve both runs.
No threshold was relaxed and no failing row was removed.

Therefore native MTP remains disabled and **must not be described as fully
qualified or accelerated**. The exact sequential verifier protects emitted
target results, but does not resolve this independent draft numerical gate
or establish an accelerated completed-answer benefit.

### Effective-norm precision follow-up

The independent reference exposed another rounding site: load-time `1 + w`
was rounded back to BF16. Native MTP now preserves effective norms in FP32
and selects FP32-weight specializations for fusion, grouped HC norms, Q/K
norms and indexer norms. All ordinary target callers retain BF16 defaults.
Already-folded installs are widened, not folded twice. Loader tests assert
exact FP32 values, separate storage and the pre-folded contract.

The native-only suite passes again: build 16.26s, 25 tests/five suites in
11.209s (`/tmp/mference-mtp-f32norm-20260922.log`). The same 43 aligned rows
were recaptured, with the same command above but export/log names containing
`aligned-f32norm`; build 5.18s, one test in 35.066s, exit 0.
Capture SHA-256:
`aaf6c601b8666b36eac8cec3194c106276fd1ceaa5f7eb912e4556e7e2109f24`.

The unchanged independent gate still **exits 1**: row 33 now passes, but row 1
remains at 9.0569% hidden / 9.1534% logits, with the same tenth-expert change.
Greedy choices differ on rows 1, 23 and 41. No QSA boundary ties occur.
Log `/tmp/mference-mtp-aligned-f32norm-upstream-20260922.log`. This precision
fix is not sufficient to qualify native MTP; it does not turn the original
failed aligned comparison into a pass.

### Final native precision comparison: passed

The attention-only follow-up reduced worst error to 0.0984% hidden / 0.0796%
logits, but still failed one greedy choice at row 29. A wide final mixer alone
did not resolve it. Upstream logits for IDs 18/16 were 11.582445/11.581841;
the draft rounded both to 11.5859375. These failures are retained in
`/tmp/mference-mtp-aligned-attention-upstream-20260922.log` and
`/tmp/mference-mtp-aligned-mixer-upstream-20260922.log` (both exit 1).

The final native-only path keeps fusion, HC mixing/injection, attention/cache,
routed/shared experts and the final head intermediates in FP32. Installed
weights retain their BF16/INT4/INT8 storage. Inputs, returned hidden bundles
and sampler-facing logits retain their FP16 ABI. The indexer retains its
existing activation representation and selection contract; the target
decoder and ordinary RMSNorm implementation are unchanged. This is a
correctness-first draft implementation, not a speed claim.

Re-running the aligned capture with export path
`/tmp/mference-mtp-aligned-wide-capture-20260922.json` and the same installed
test filter above exits 0: build 19.70s, one test in 36.152s. Then:

```sh
scratch/qwen4exp-parity-venv/bin/python Scripts/parity/verify_flashnext_mtp.py \
  /tmp/mference-mtp-aligned-wide-capture-20260922.json \
  --sglang-source /tmp/mference-upstream-qwen4_exp_mtp.py \
  --install scratch/qwen38flashnext-r8.gturbo
```

**Exit 0: all 43 aligned rows pass**, with no QSA ties or tie adaptation,
all raw greedy choices equal, worst hidden error 0.041973% and logits
0.037470%. The original 5% bounds are unchanged. Capture SHA-256:
`1b085af3e2ce2bbf364956fe25ecbc7ecb21180a9060a974daf1117c23ef11f4`.
Log `/tmp/mference-mtp-aligned-wide-upstream-20260922.log`.
The synthetic suite also passes: 26 tests/six suites in 11.824s,
`/tmp/mference-mtp-wide-20260922.log`. Synthetic row-zero hidden error is
now 0.036193%, without a known-issue exemption.

The final complete installed native suite (same installed command, no export
environment variables) exits 0: build 1.41s, 26 tests/six suites in 137.316s.
Log `/tmp/mference-mtp-wide-final-installed-20260922.log`. The natural chat
again produces the exact target answer and full heads, with natural stopping.

This closes the tested **same-weight aligned numerical verification** gate,
not accelerated generation, large-context sparse selection on the installed
sidecar, or native default promotion. CLI/server native MTP remains off.

Two intermediate harness/regression failures were resolved, not suppressed:
the initial norm specialization changed exact ordinary BF16 per-head results;
that modification was removed entirely. A full installed capture attempt
also refused an existing file after the toy test exported to the installed
path; exports are now explicitly enabled only by the installed priming test.
The subsequent installed captures use fresh paths. No output was overwritten.

### Full package regression

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs
```

Exit 0: **1,302 tests / 235 suites passed in 284.068s**, with the one existing
known issue for the absent regenerable toy source checkpoint (not an MTP
numerical exemption). Log `/tmp/mference-full-wide-20260922.log`. Installed
model gates are recorded separately above; ungated suite success does not
stand in for running them.

Release build:

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  swift build -c release --scratch-path /tmp/mference-phase1-build.sXnNTs
```

Exit 0, 59.43s; log `/tmp/mference-release-wide-20260922.log`. All eight
launcher/UI/screen/source/benchmark script suites pass (43 tests), log
`/tmp/mference-scripts-wide-20260922.log`. Both new Python parity modules
compile, and `git diff --check` passes. No new checkpoint was downloaded.

Code and the initial record are committed at
`ef9484d5c0977d13bcfb8f05adf3d9369c576236`. Its
[PR CI](https://github.com/NeelM0906/Mference/actions/runs/35767341334)
passes all three jobs: macOS 15 / Swift-floor build-and-test, macOS 26
build-and-test, and documentation/source-archive checks.

The separate [source-candidate workflow](https://github.com/NeelM0906/Mference/actions/runs/35767355921)
also passes for that exact commit: both toolchains, archive checksums, fresh
extraction, release build, serial tests, all three executable help commands,
launcher/source/benchmark script tests and Markdown validation. Its artifact
is `mference-source-ef9484d5c0977d13bcfb8f05adf3d9369c576236`
(artifact ID `10715315045`), containing the `0.1.1-rc.1` source candidate,
provenance and checksums. This workflow does not create a tag or publish a
GitHub release; the existing 0.1.0 release is untouched.

## GLM Max profile

The Max/1,024-output failures remain historical failures. A distinct profile
uses unchanged Max effort with 8,192 output tokens and 16,384 context:

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  BENCH_CLI=/tmp/mference-glm-measured.Dr7tT9/MferenceCLI \
  BENCH_SETTLE_SECONDS=10 MIN_FREE_PCT=75 MFERENCE_GLM5_REASONING_EFFORT=max \
  ./run-benchmark.sh glm-max-resident-8192-20260922 scratch/glm53flash.gturbo 3 \
  --expert-cache-slots resident --max-new 8192 --max-context 16384
```

This uses the previously preserved, hashed GLM candidate executable; native
MTP changes are not in its binary and have no GLM caller.
The executable SHA-256 is
`29d9eefeb8686b43d7a26f9af0ec748cf9d66a4aa154cfa85b799ed690646f4e`,
built from `fba8196b09424100071ab08a9c25419f31f2eb79`. All `Sources/`
differences from that revision to `ef9484d` are Flash-Next/native-MTP files;
the GLM runtime and CLI source are unchanged. Whole-process times below are
measurements of this named frozen executable, not a rebuilt binary.
Frozen prompts,
seeds and sampling are unchanged. Output/context allowances are explicit
protocol deviations. The batch launcher was paused **between** its second and
third discarded warmups and again after the first measured medium case for
native validation. Each active CLI was allowed to finish naturally before
any build/test began; no timed process was interrupted. These inter-run
pauses are additional protocol deviations, not measured runtime.
No measured repetition overlapped builds/tests/downloads. A read-only cache
filename scan during the second discarded warmup was stopped; that warmup
is not used in reported performance.

All three discarded warmups produce complete visible answers:

```text
[stop=endOfTurn prefill=60tok/1.23s new=2127tok decode=92.31s tok/s=23.042]
[stop=endOfTurn prefill=420tok/3.96s new=3708tok decode=181.22s tok/s=20.461]
[stop=endOfTurn prefill=2792tok/29.89s new=2795tok decode=145.74s tok/s=19.177]
```

The answers were read for completion, not independently factual-scored.
The wetlands answer includes an overbroad claim about rainfall flooding;
completion is not a broad accuracy endorsement.

**Completed, exit 0:** all 12 CLI processes and the harness succeeded; 9/9
measured answers reached `endOfTurn`. Each measured stdout matches its
case's manually read warmup byte-for-byte. All process exit files contain 0.
Evidence: `benchmark-results/glm-max-resident-8192-20260922/`.

| Case (prompt/new tokens) | Median prefill, s | Median decode, tok/s | Median prefill + decode, s (range) | Median whole-process wall, s |
| --- | ---: | ---: | ---: | ---: |
| Short (60/2127) | 1.23 | 23.041 | 93.54 (93.49–93.92) | 176.72 |
| Medium (420/3708) | 3.96 | 20.451 | 185.27 (184.27–186.78) | 268.71 |
| Long (2792/2795) | 29.89 | 19.150 | 175.85 (175.57–175.89) | 259.09 |

Maximum reported process footprint: 172,639,103,856 bytes. This is not a
minimum-RAM certificate. Cold-process wall time includes strict verification
and loading; prefill-plus-decode does not. These Max outputs differ from the
low-effort workload, so comparing their token rates does not establish an
optimization gain or equal answer quality. Max **with the larger explicit
budget** now passes completion; Max/1,024 is still not qualified.

Complete measured footers (warmup footers above):

```text
short 1 [stop=endOfTurn prefill=60tok/1.22s new=2127tok decode=92.32s tok/s=23.041]
short 2 [stop=endOfTurn prefill=60tok/1.25s new=2127tok decode=92.67s tok/s=22.954]
short 3 [stop=endOfTurn prefill=60tok/1.23s new=2127tok decode=92.26s tok/s=23.054]
medium 1 [stop=endOfTurn prefill=420tok/3.95s new=3708tok decode=180.32s tok/s=20.563]
medium 2 [stop=endOfTurn prefill=420tok/3.96s new=3708tok decode=182.82s tok/s=20.282]
medium 3 [stop=endOfTurn prefill=420tok/3.96s new=3708tok decode=181.31s tok/s=20.451]
long 1 [stop=endOfTurn prefill=2792tok/29.89s new=2795tok decode=145.68s tok/s=19.186]
long 2 [stop=endOfTurn prefill=2792tok/29.90s new=2795tok decode=145.95s tok/s=19.150]
long 3 [stop=endOfTurn prefill=2792tok/29.89s new=2795tok decode=146.00s tok/s=19.144]
```

## GLM bounded profile

The separate 16-slot profile retains the same frozen executable and original
1,024-output / 4,096-context allowances. It explicitly selects vendor-supported
low effort. No result from it is a physical smaller-Mac qualification.

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  BENCH_CLI=/tmp/mference-glm-measured.Dr7tT9/MferenceCLI \
  BENCH_SETTLE_SECONDS=10 MIN_FREE_PCT=75 MFERENCE_GLM5_REASONING_EFFORT=low \
  ./run-benchmark.sh glm-low-16slot-20260922 scratch/glm53flash.gturbo 3 \
  --expert-cache-slots 16
```

Initial preflight: 98% free memory, 612 GiB disk, no model/test owner, all 49
receipt-file sizes match. The run uses full-SHA install verification. Explicit
protocol deviations: low effort, 16 slots, three measured repetitions and
fixed ten-second inter-process settling. No builds, tests or downloads run
alongside timed processes.

**Completed, exit 0:** all three warmups, all nine measured CLI processes and
the harness succeed. Every answer reaches `endOfTurn`; all 12 stdout files
match the manually read low-effort resident answers byte-for-byte. This checks
completion and equivalence on the frozen cases, not broad factual accuracy.
Every preflight reports 98% free memory and 612 GiB free disk.
Evidence: `benchmark-results/glm-low-16slot-20260922/`.

| Case (prompt/new tokens) | Median prefill, s | Median decode, tok/s | Median prefill + decode, s (range) | Median whole-process wall, s |
| --- | ---: | ---: | ---: | ---: |
| Short (60/653) | 15.19 | 6.768 | 111.71 (111.51–111.73) | 179.34 |
| Medium (420/841) | 46.02 | 6.428 | 176.81 (176.81–176.95) | 244.53 |
| Long (2792/709) | 302.22 | 6.032 | 419.67 (419.44–419.83) | 487.27 |

The maximum reported process footprint is 10,528,826,512 bytes. This metric
does not fully account for GPU/mapped/page-cache residency and does **not**
certify operation on a 16/24-GiB Mac. The 256-GiB machine can cache file data
outside the bounded expert slots. Bounded mode completes the same answers
but is substantially slower than the separate resident low-effort run; no
bounded-versus-prior-revision optimization gain is claimed.

Complete footers:

```text
warm short [stop=endOfTurn prefill=60tok/15.25s new=653tok decode=96.54s tok/s=6.764]
warm medium [stop=endOfTurn prefill=420tok/46.08s new=841tok decode=130.81s tok/s=6.429]
warm long [stop=endOfTurn prefill=2792tok/302.22s new=709tok decode=117.56s tok/s=6.031]
short 1 [stop=endOfTurn prefill=60tok/15.19s new=653tok decode=96.54s tok/s=6.764]
short 2 [stop=endOfTurn prefill=60tok/15.18s new=653tok decode=96.33s tok/s=6.779]
short 3 [stop=endOfTurn prefill=60tok/15.23s new=653tok decode=96.48s tok/s=6.768]
medium 1 [stop=endOfTurn prefill=420tok/46.04s new=841tok decode=130.91s tok/s=6.424]
medium 2 [stop=endOfTurn prefill=420tok/46.02s new=841tok decode=130.79s tok/s=6.430]
medium 3 [stop=endOfTurn prefill=420tok/45.98s new=841tok decode=130.83s tok/s=6.428]
long 1 [stop=endOfTurn prefill=2792tok/302.22s new=709tok decode=117.61s tok/s=6.028]
long 2 [stop=endOfTurn prefill=2792tok/301.89s new=709tok decode=117.55s tok/s=6.032]
long 3 [stop=endOfTurn prefill=2792tok/302.23s new=709tok decode=117.44s tok/s=6.037]
```
