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
