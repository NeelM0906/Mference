# Source candidate: review and publication boundary

This is the source candidate delivered by
[PR #38](https://github.com/NeelM0906/Mference/pull/38), not a published tag or
signed application. The reproducible archive workflow and local packaging
commands are in [Source release](SOURCE_RELEASE.md).

## Included

- Swift/Metal library, streaming installer, CLI, loopback OpenAI-compatible
  server and the launcher for pinned Open WebUI 0.11.3.
- Source-faithful Swift-Qwen identity/template/reasoning handling, boolean tool
  argument correction, explicit truncation reporting and UI history adapter.
- Production batched prefill and truthful execution/memory diagnostics,
  with installed sparse/window-boundary and recovery evidence.
- Flash-Next routing/target-state/rollback improvements and GLM INT8 ragged-tile
  input bounds correction. Native draft priming is internal only.
- Source tarball, exact commit/tree manifest and checksums, produced after
  two-platform CI by the manually triggered source-candidate workflow.

For a new user, keep the established first-use path: build the source, install
Gemma 4, then start the UI. Builds and checkpoint downloads are required;
“source release” does not mean instant first-run inference without them.
The source archive contains no downloaded checkpoint or existing UI history.

## Deliberately not promoted

| Area | Candidate behavior | Remaining evidence |
| --- | --- | --- |
| Swift-Qwen | Optional distinct checkpoint; base Qwen remains available | Broader task quality, completed-answer latency and fresh physical smaller-Mac qualification before recommending replacement |
| Native Flash-Next MTP | Disabled; no CLI/server generation caller | Independent full-native reference, exact target verification/acceptance, stop/cancellation/client integration and measured benefit |
| GLM performance | No new completed-answer speed claim | Default Max warmups exhausted 1,024 tokens; a separate bounded-reasoning comparison must be labelled, not substituted for those failures |
| Hardware/context | Only the profiles in the support table are evidenced | Reduced expert slots on a 256 GiB Mac do not qualify a physical 16/24 GiB Mac or every selectable context |

The native draft's unrounded-FP32 row-zero test remains a narrow known issue;
the optional absent Flash-Next toy reference is separately disclosed. Neither
is described as a passed native-MTP qualification. Do not enable or advertise
experimental features merely because the source build and packaging pass.

## Maintainer handoff

1. Review the exact PR head and its macOS 15 / Swift 6.1, macOS 26, docs and
   security results. Do not reuse green CI from an older head.
2. Review the [September 21 validation](RELEASE_VALIDATION_2026-09-21.md),
   [checkpoint support](RELEASE_SUPPORT.md), [prefill matrix](PREFILL_QUALIFICATION.md)
   and [performance evidence](RELEASE_PERFORMANCE_2026-09-20.md).
3. Select the release commit/version and run **Source release candidate** for
   that ref. Download its three files together and verify checksums. Confirm
   that the manifest commit is the reviewed commit.
4. Test the extracted source with `doctor`, release build and an existing
   verified model on an otherwise idle Apple Silicon Mac. Keep services on
   loopback and stop only those started for that check.
5. Merge/tag/publish only with maintainer approval. Carry the limitations above
   into the release notes; do not label the entire optimization roadmap done.

Packaging, PR approval, publication and roadmap qualification are separate
decisions. This document is not evidence that a release has already been
published or that all future performance objectives have been met.
