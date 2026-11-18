# Source-release support and checkpoint choices

Status reviewed September 21, 2026; PRs #37 and #38 are merged. The 0.1.0 source
baseline is `4ff63ff`; its [final release gate](RELEASE_GATE_0.1.0.md) records
the artifact and extracted-source checks, and
[GitHub Releases](https://github.com/NeelM0906/Mference/releases) records publication.
This table distinguishes established choices from qualification
candidates. It is not a claim that every family has been re-tested on every Mac.
For a first installation and the established general-purpose MoE path, use
**Qwen 3.6**. Families removed since this review are listed in
[retired model families](RETIRED_FAMILIES.md); this runtime no longer loads
their installs.

## Checkpoints

Sizes are decimal GB of installed files, not required RAM or download budgets.
Local receipt totals were inspected September 18; MiniCPM uses the linked
family's planning evidence where no completed current local install exists.
Allow separate room for builds, Open WebUI, and long-context cache spill.

| Install selector | Release role | Installed size | Hardware and context evidence / important limits |
| --- | --- | ---: | --- |
| `qwen36` | First-use recommendation / established general-purpose MoE | 19.55 GB | Historical physical 24 GB M5 / 256 GB M3 Ultra results, frozen 4,096-context protocol. Reduced-working-set experiments are not physical 8 GB qualification. [Evidence](BENCHMARKS.md). |
| `qwen38` | Established dense option | 15.39 GB | Historical 24 GB M5 and 256 GB M3 Ultra measurements; dense weights stay resident. Long-context modes have their own [limits and evidence](QWEN38_LONG_CONTEXT.md). |
| `maple` | Retained preview checkpoint | 6.59 GB | Installed 16/8-slot 512-window boundary, continuation and recovery checks pass on 256 GiB M3 Ultra. [Evidence](RELEASE_VALIDATION_2026-09-20.md). Not newly quality-ranked; approximate FlashHead remains opt-in. |
| `minicpm5` | Smallest download / dense regression option | ~1.43 GB | Historical 24 GB M5 evidence; longer community cases need more than the frozen generation allowance and are disclosed separately. [Evidence](families/MINICPM5.md). |

### Maple known deviation

On Apple7 GPUs (M1), the Maple Q/K norm and RoPE kernel test
(`MapleQKNormRoPETests.swift:74`) measures a maximum absolute difference of
0.0234 against its CPU reference at position 127,999, above the 2^-7
(0.0078) bound the test applies to Apple7 and newer devices. CI's
paravirtualized GPU measures the same value but reports an older GPU family,
so the test's wider 2^-5 bound applies there and CI passes. Maple's generated output is
byte-identical to `main` on these machines; only this kernel tolerance check
fails. The likely cause is lower-precision GPU `sin`/`cos` at very large
positions. It is not fixed: the test and its bound are unchanged, and the
2026-11-15 ruling that a run whose only failure is this test counts as green
still applies.

### Reasoning, tools and context

Qwen 3.8 accepts explicit `xhigh`, `medium`, `low`, and `none` reasoning
efforts for source-template comparisons, while omitted effort preserves its
legacy behavior; its MTP policy is unchanged. Maple and MiniCPM retain their own
thinking policies and reject this effort parameter; reset it before switching
models in the UI.
Tool histories can select a different thinking suffix, which the decoder now
honors. See [runtime controls](RUNTIME_CONTROLS.md) and [the UI guide](OPEN_WEBUI.md).

The server's 16,384-token default is a configuration default, **not** a statement
that this release has freshly qualified that context on every model/hardware
combination. The common performance/screen protocol uses 4,096. Larger contexts
increase state/scratch/cache requirements even when experts are streamed.
Token limits include hidden reasoning; `finish_reason=length` with an empty
visible answer is a truncated generation, not a successful fast response.

### Release boundary

- macOS 26+, Apple Silicon, Swift 6.2+; source build, not a signed application.
- CLI and loopback server are native; Open WebUI 0.11.3 is the only supplied UI.
- One model owner at a time; in-process library switching unloads the old model.
- No application authentication/TLS: never expose either service publicly.
- No weights bundled; checkpoint licensing and substantial first-use downloads
  remain separate from the runtime's license and source archive.
- New kernel paths require installed-model correctness and measured performance
  evidence. The [qualification matrix](PREFILL_QUALIFICATION.md) records gaps;
  the [dated release record](RELEASE_VALIDATION_2026-09-18.md) records actual runs.

Historical numbers remain historical. Neither a low RSS nor reduced expert
slots on a 256 GiB Mac establishes a minimum-RAM recommendation.
