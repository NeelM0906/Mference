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

After the fixture fix, the first pass exited 0: 11 tests / 2 suites, 6.675s,
one known issue (`/tmp/mference-mtp-priming-v2-20260921.log`). Adding the paired
target/primer recovery test and opt-in installed probe then exited 0:

```text
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
