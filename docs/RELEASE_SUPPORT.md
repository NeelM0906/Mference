# Source-release support and checkpoint choices

Status reviewed September 21, 2026; PRs #37 and #38 are merged. The 0.1.0 source
baseline is `4ff63ff`; its [final release gate](RELEASE_GATE_0.1.0.md) records
the artifact and extracted-source checks, and
[GitHub Releases](https://github.com/NeelM0906/Mference/releases) records publication.
This table distinguishes established choices from qualification
candidates. It is not a claim that every family has been re-tested on every Mac.
For a first installation, use **Gemma 4**. For the established general-purpose
MoE path on a 24 GB-class Mac, use **Qwen 3.6**. Keep base Qwen available while
Swift-Qwen is being qualified; shorter reasoning alone does not prove better
quality or faster completed answers.

## Checkpoints

Sizes are decimal GB of installed files, not required RAM or download budgets.
Local receipt totals were inspected September 18; MiniCPM uses the linked
family's planning evidence where no completed current local install exists.
Allow separate room for builds, Open WebUI, and long-context cache spill.

| Install selector | Release role | Installed size | Hardware and context evidence / important limits |
| --- | --- | ---: | --- |
| `gemma4` | First-use recommendation | 14.29 GB | Historical physical 8 GB M2 and 24 GB M5 measurements; bounded experts. [Evidence](BENCHMARKS.md). |
| `qwen36` | Established general-purpose MoE | 19.55 GB | Historical physical 24 GB M5 / 256 GB M3 Ultra results, frozen 4,096-context protocol. Reduced-working-set experiments are not physical 8 GB qualification. [Evidence](BENCHMARKS.md). |
| `qwen38` | Established dense control / alternative to Swift's license | 15.39 GB | Historical 24 GB M5 and 256 GB M3 Ultra measurements; dense weights stay resident. Long-context modes have their own [limits and evidence](QWEN38_LONG_CONTEXT.md). |
| `swiftqwen38` | Optional candidate, not the recommended replacement yet | 15.38 GB | Numerical, state and tool/UI gates on 256 GiB M3 Ultra; [matched-low screen](families/QWEN_MATCHED_QUALIFICATION_2026-09-18.md): 54/60 cases versus base 59/60, 6.77% fewer completion tokens. [Five-seed xhigh and default-policy screens](QWEN_SOURCE_EFFICIENCY_V1.md): 14.992% / 20.088% fewer completion tokens, with quality and policy differences recorded. Default-policy Swift passes 180/180; base's 177/180 reflects one ambiguous punctuation item. Long-form performance and broader quality remain open. No new 24 GB qualification. Distinct model ID and [source license](families/SWIFT_QWEN38.md). |
| `qwen38flashnext` | Advanced flagship MoE | 175.21 GB | INT8-router profile: 4,096-context evidence on 256 GB M3 Ultra; resident/16-slot short and 1,024-chunk sparse-boundary correctness gates pass with observed TensorOps execution. [September 20 measurements](RELEASE_PERFORMANCE_2026-09-20.md): resident `15713e7` reduces prefill-plus-decode time but regresses short/medium prefill; 16-slot `b4325a4` reduces median generation time by 0.5%/1.2%/2.3%, with overlapping short-case and all decode ranges. These are distinct revisions and not smaller-Mac evidence. Packaged MTP weights do **not** imply qualified native MTP decoding. [Family evidence](families/QWEN38_FLASH_NEXT.md). |
| `glm53flash` | Advanced qualification candidate for this update | 180.84 GB | Pinned install and current resident/16-slot sparse-cutover, continuation and cancellation/reset correctness gate pass. [Default-profile warmups](RELEASE_PERFORMANCE_2026-09-20.md#glm-default-profile-warmups-rejected-completion-gate-safety-stop) truncate without visible answers. A separate [matched low-effort resident profile](POSTLAUNCH_QUALIFICATION.md#matched-glm-baseline) completes all 24 processes with identical answers: long generation improves 3.7%, short/medium ranges overlap, no decode gain. Default Max and bounded performance remain open. Resident evidence is from the 256 GiB host; reduced slots do not certify a smaller Mac. [Evidence](families/GLM53_FLASH.md). |
| `deepseekv4flash` | Experimental, retained | 96.69 GB | Sparse-cutover resident/streamed correctness recorded separately from performance. Its 2-bit checkpoint and family-specific constraints remain [explicit](DEEPSEEK_V4_FLASH.md). |
| `inklingsmall` | Retained specialist / large MoE | 148.42 GB | Historical 24 GB M5 measurements; current short regression plus exact installed resident/16-slot 512-window boundary and recovery checks on 256 GiB M3 Ultra. [New evidence](RELEASE_VALIDATION_2026-09-20.md), [family](INKLING_SMALL.md). |
| `maple` | Retained preview checkpoint | 6.59 GB | Installed 16/8-slot 512-window boundary, continuation and recovery checks pass on 256 GiB M3 Ultra. [Evidence](RELEASE_VALIDATION_2026-09-20.md). Not newly quality-ranked; approximate FlashHead remains opt-in. |
| `minicpm5` | Smallest download / dense regression option | ~1.43 GB | Historical 24 GB M5 evidence; longer community cases need more than the frozen generation allowance and are disclosed separately. [Evidence](families/MINICPM5.md). |

### Reasoning, tools and context

Swift's source-faithful default is `xhigh`; `medium`, `low`, and `none` are
explicit request options. MTP stays off by default for Swift. Base Qwen's
existing MTP policy is unchanged. Base Qwen 3.8 accepts the same explicit
efforts for source-template comparisons, while omitted effort preserves its
legacy behavior. Maple, MiniCPM and GLM retain their own thinking policies and
reject this effort parameter; reset it before switching models in the UI.
Tool histories can select a different thinking suffix, which the decoder now
honors. See [runtime controls](RUNTIME_CONTROLS.md) and [the UI guide](OPEN_WEBUI.md).

The server's 16,384-token default is a configuration default, **not** a statement
that this release has freshly qualified that context on every model/hardware
combination. The common performance/screen protocol uses 4,096. Larger contexts
increase state/scratch/cache requirements even when experts are streamed.
Token limits include hidden reasoning; `finish_reason=length` with an empty
visible answer is a truncated generation, not a successful fast response.

### Release boundary

- macOS 15+, Apple Silicon, Swift 6.1+; source build, not a signed application.
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
