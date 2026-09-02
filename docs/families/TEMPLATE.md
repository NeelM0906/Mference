# `<FAMILY_DISPLAY_NAME>` on Mference

<!-- Copy this file to docs/families/<FAMILY>.md and replace every placeholder.
     Delete rows and sections that do not apply; do not leave a placeholder in
     a published page. Voice: state what was measured or read, and where from.
     Label anything not yet measured as an estimate. -->

Checkpoint selection, architecture contract, memory budget, and measured
results for running [`<REPO_ID>`](https://huggingface.co/<REPO_ID>) with
SSD-streamed experts. This document is the architecture contract for the
`<FAMILY>` family.

| | |
|---|---|
| Family identifier | `<FAMILY>` (`ModelFamily.<FAMILY>`) |
| Source repository | `<REPO_ID>` |
| Pinned revision | `<REVISION_SHA>` |
| Parameters | `<TOTAL_B>`B total, `<ACTIVE_B>`B active |
| Install size | `<INSTALL_BYTES>` bytes |
| Status | `<planned / in port / supported / unmaintained>` |

## Checkpoint selection

<!-- One paragraph: who published the model, under what license, and how many
     conversions exist. Then say which quantization contract the runtime
     enforces and where (ManifestReader.validateQuant). -->

`<FAMILY_DISPLAY_NAME>` was published by `<VENDOR>` under `<LICENSE>` in
`<MONTH YEAR>`. Weights are redistributed by the conversion author, not by
this project; the license note below records what the pinned revision permits.

**License note.** `<LICENSE>` — `<what it permits and any attribution or
use-restriction clause that affects redistribution or benchmarking>`.
Update [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md) if any
reference code is imported alongside the port.

### Rejected candidates

<!-- Every conversion that was examined and why it lost. A one-line "the rest
     were unreadable" row is fine; an empty table is not. -->

| Candidate | Routed experts | Attention / embeddings | Disk | Verdict |
|---|---|---|---|---|
| `<REPO_ID>` | `<affine 4-bit g64>` | `<affine 4-bit g64>` | `<GB>` | **Selected** |
| `<candidate>` | `<format>` | `<format>` | `<GB>` | `<why rejected>` |
| `<candidate>` | `<format>` | `<format>` | `<GB>` | `<why rejected>` |

Verified against revision `<REVISION_SHA>` by reading the safetensors headers
directly: `<tensor groups checked>` are `<quantization contract>`.
`<Any tensor that deviates, and whether the deviation is cosmetic.>`

## Architecture contract

<!-- One row per ArchConfig axis this family sets away from its default.
     "Covered by existing kernel?" is yes / no / partial, and every "no" or
     "partial" must name the gap. Cross-check the axis names against
     docs/FAMILY_CONTRACT.md so this table cannot drift from the enumeration. -->

| Axis | Value | Covered by existing kernel? | Gap |
|---|---|---|---|
| `layerTypes` | `<value>` | yes | — |
| `attention` | `<value>` | `<yes / no / partial>` | `<named kernel work, or ->` |
| `routerScoring` | `<value>` | `<yes / no / partial>` | `<...>` |
| `sharedExpertGating` | `<value>` | `<yes / no / partial>` | `<...>` |
| `<axis>` | `<value>` | `<yes / no / partial>` | `<...>` |

Axes left at their defaults are not listed. Anything the runner branches on
that is *not* an axis is a family quirk: record it here explicitly, because the
quirk count is the health metric for the axis enumeration.

| Quirk | Where it lives | Why it is not an axis |
|---|---|---|
| `<quirk>` | `<file:line>` | `<reason>` |

## Memory budget

<!-- Params and bytes per group, then the resident/on-disk split. Numbers come
     from the produced install, not from the config file, once one exists. -->

| Group | Params | Bytes |
|---|---|---|
| `embed` + `unembed` | `<B>` | `<GB>` |
| attention (`<N>` layers) | `<B>` | `<GB>` |
| shared experts | `<B>` | `<GB>` |
| router + norms | `<B>` | `<GB>` |
| **Resident total** | **`<B>`** | **`<GB>`** |
| routed experts | `<B>` | `<GB>` |
| **On disk** | | **`<GB>`** |

### KV cache

| Context | KV bytes |
|---|---|
| 32 K | `<GB>` |
| 128 K | `<GB>` |

### Expert-cache ladder

| Slots | Resident expert bytes | Notes |
|---|---:|---|
| 16 | `<GB>` | floor; `auto` default for non-Qwen families |
| 32 | `<GB>` | |
| 96 | `<GB>` | |

## Port status

<!-- Check a box only when the artifact exists and is green. bringup-check.sh
     stages 1-3 cover toy parity, repack, and ladder; the last two are manual. -->

- [ ] **Repack** — `SupportedModelSource` entry pinned to `<REVISION_SHA>`;
      `--verify-install` passes on the produced install.
- [ ] **Toy parity** — family toy synthetic and parity suite registered, and
      wired into `bringup-check.sh`'s stage 1 filter table.
- [ ] **Runner** — forward pass matches the reference implementation; per-layer
      logit deltas within the tolerance policy this page records below.
- [ ] **Ladder** — 16 / 32 / auto expert-cache slots produce byte-identical
      greedy output (`bringup-check.sh` stage 3).
- [ ] **Gate** — every step of [`FAMILY_GATE.md`](../FAMILY_GATE.md) green,
      including the full suite three times consecutively.
- [ ] **Protocol bench** — the three frozen `real-generation-v1` cases with one
      discarded warmup and every footer `stop=endOfTurn`.

## Measured results

<!-- Medians across measured repetitions; peak RSS is the maximum across runs.
     Leave the table empty until the protocol has actually been run: an
     estimate must never occupy a measured row. -->

Host: `<Mac model>`, `<chip>`, `<memory>`, macOS `<version>`, Swift
`<version>`, Mference commit `<SHA>`. Protocol:
[`COMMUNITY_BENCHMARKS.md`](../COMMUNITY_BENCHMARKS.md), `<N>` measured
repetitions per case after one discarded warmup, each run a fresh process.

| Case | Prompt / generated | Prefill | Decode | Range | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: |
| short-explanation | | | | | |
| medium-review | | | | | |
| long-synthesis | | | | | |

Decode rate excludes model installation, model loading, and prompt prefill.
Compare rows against [`BENCHMARKS.md`](../BENCHMARKS.md) only when the case,
prompt tokens, generated tokens, settings, and stop reason all match.

### Phases attribution

One `MFERENCE_PHASES=1` decode run, recorded as the baseline that later
optimization A/Bs are judged against ([`FAMILY_GATE.md`](../FAMILY_GATE.md)
step 6).

| Phase | ms/token | Share |
|---|---:|---:|
| `<phase>` | | |

## Known limits

<!-- What does not work, what is approximate, and what is untested. An empty
     section here is a claim that nothing is missing; say so explicitly if so. -->

- `<limit>` — `<consequence for a user, and whether a workaround exists>`.
- Optional or approximate features are opt-in flags with no default-path
  claims: `<list them, or "none">`.
- Untested: `<contexts, hosts, or modes nobody has run>`.

## Reproduction

```bash
# Build once.
swift build -c release

# Install from the pinned revision.
swift run -c release MferenceRepack --model <FAMILY> --output scratch/<FAMILY>.gturbo

# Conformance stages: preflight, toy suite, install verify, ladder smoke,
# protocol scaffold. Green output is the definition of "supported".
./bringup-check.sh <FAMILY> scratch/<FAMILY>.gturbo

# Community protocol benchmark (long; run on a quiet machine).
./run-benchmark.sh <FAMILY> scratch/<FAMILY>.gturbo 3

# Phases attribution baseline.
MFERENCE_PHASES=1 .build/release/MferenceCLI \
  --model scratch/<FAMILY>.gturbo \
  --messages-file docs/benchmark-prompts/real-generation-v1/short-explanation.json \
  --max-new 1024 --max-context 4096 --temperature 0.2 --top-k 64 --top-p 0.95 \
  --seed 20260721
```

Run one model process at a time, and re-read the preconditions in
[`AGENTS.md`](../../AGENTS.md) before any model run.
