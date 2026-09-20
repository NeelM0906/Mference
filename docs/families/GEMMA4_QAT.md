# Gemma 4 QAT aligned

Mference supports the separate
`mlx-community/gemma-4-26B-A4B-it-qat-q4_0-mlx-aligned` checkpoint through the
existing Gemma architecture. It uses its own weights, installed chat template
and generation settings. The original Gemma remains independently selectable.

| Item | QAT checkpoint |
| --- | --- |
| Install selector | `gemma4qat` |
| Final installation directory | `$HOME/llm-models/gemma4qat.gturbo` |
| API model ID | `gemma-4-26b-a4b-it-qat-q4_0-mlx-aligned` |
| Pinned source revision | `745a97a754ed4b7713163c7d0e9c11da41809e0c` |
| Installed bytes, including verified receipt | 15,835,171,794 |
| Resident weight file | 1,512,886,332 bytes |
| Streamed expert pool | 14,281,605,120 bytes |
| Native format | INT4 affine, group 32; BF16 scales/biases and unquantized BF16 routers |

These are storage sizes. They do not measure physical memory use. The installer
copies the supplied native bytes without requantization, streams bounded
ranges, and resumes verified completed work. See the
[installation commands and space accounting](../OPEN_WEBUI.md#gemma-qat-installation).
The same completed installation supports the runtime update; no second
download or repack is needed.

## Use and defaults

Install with `./mference-ui.sh install gemma4qat`. After installation, start
`./mference-ui.sh` and select the QAT API ID in Open WebUI. The existing library
server owns one loaded model and switches it when the selected model changes.
For a CLI session, first follow the project's model-process/resource checks,
then run:

```bash
.build/release/MferenceCLI \
  --model "$HOME/llm-models/gemma4qat.gturbo" --chat --seed 42
```

Omitted sampling controls use temperature 1, Top-K 64, Top-P 0.95, Min-P 0,
repetition penalty 1, and presence/frequency penalties 0. Explicit controls
win, including temperature zero. These defaults apply to raw CLI, chat,
messages files and HTTP, including custom aliases and library swaps.

Thinking is off by default. `--reasoning-effort medium` enables it in the CLI;
the server accepts the existing reasoning-effort or `enable_thinking` controls.
The checkpoint template drops ordinary historical reasoning and replays
tool-call reasoning within the active user turn. `preserve_thinking=true` is
unsupported and rejected before streaming starts. Clients send full histories
with matching tool-call IDs and results. See [QAT controls](../RUNTIME_CONTROLS.md#gemma-qat)
and the [server API](../OPENAI_SERVER.md).

## Qualification and limits

The recorded machine is an Apple M2 Mac14,2 with 16 GiB RAM, macOS 26.6.2
(25G83), and Swift 6.4. Evidence uses an uncommitted working tree based on
`0d0175fe1d0688ea1ede6b76585455c079348e93`. Exact commands, artifact hashes,
errors and timing footers are kept in the author's local validation record,
which is not part of the repository.

- Native scalar/chunked inference matched the pinned FP16 MLX reference's
  top prediction at all 158 qualified full-vocabulary positions, with routing
  and logit errors within the frozen tolerances. This is scoped numerical
  evidence, not bit-identical execution for every prompt.
- The pinned template passed 72 independent cases with exact rendered bytes
  and token IDs. CLI chat/messages, thinking on/off and both real HTTP modes
  passed, including two tool-result rounds, repeated calls and a later user.
- Fresh versus recovered execution matched all logits in 12 full-vocabulary
  rows. Real server responses also matched after cache reuse. Cancellation
  and truncated tool output were followed by clean subsequent requests in
  both server modes.
- Original Gemma → QAT → original Gemma switching passed with one server
  owner and no checkpoint-to-checkpoint cache reuse. The original installation
  and its source profile were preserved.

The raw greedy prompt `The capital of France is` repeats in both Mference and
the independent QAT reference. QAT does not guarantee fewer loops; raw
completion also omits chat framing. No general quality improvement follows
from these integration checks.

The final package run passed all server, installer and Jinja targets. Its core
target retains the pre-existing Maple Q/K norm failure (`0.0234375` against
`0.0078125`) and the known missing optional Flash-Next fixture. The QAT affected
gate passed; the whole repository is not all green.

The tested public runs use a 4,096-token capacity; broader contexts and other
hardware are not qualified by this evidence. The architecture's source context
limit is not a tested runtime limit. KV/activation precision policy remains
unchanged, and this integration provides text inference only.

## M2 generation measurements

The three frozen community cases ran once as discarded warmups and once in
fresh measured processes, using temperature 0.2, Top-K 64, Top-P 0.95, source
Min-P 0, neutral penalties, thinking off, 4,096 context and 1,024 maximum output.
Seeds are 20260721, 20260722 and 20260723. All measured answers reached
`endOfTurn`, matched their warmups byte for byte, and were complete without
repeated blocks. This checks completion, not exhaustive factual correctness.

| Case | Prompt / generated tokens | Prefill | Decode |
| --- | ---: | ---: | ---: |
| Short explanation | 61 / 502 | 12.60 s | 6.642 tokens/s |
| Design review | 430 / 712 | 40.35 s | 6.401 tokens/s |
| Long synthesis | 3015 / 566 | 265.21 s | 4.586 tokens/s |

The Mac was on AC power with Low Power Mode off. Desktop applications and
macOS background services remained active, a deviation from a quiet-machine
benchmark. These are observations under the recorded workload, not medians,
performance ceilings or a speed comparison with original Gemma. Complete
commands, outputs, footers and source/binary hashes are kept in the local
validation record.

A subsequent same-machine original-Gemma comparison
recorded lower original prefill times and higher original decode rates in these
autonomous samples. Input tokens and sampling match, but generated answers
differ, so it is not a matched-output speed ratio. Every prompt and both
measured answers were kept in the local record for human quality review.

## Measured memory

A separate run of the short chat case used `/usr/bin/time -l` and runtime
diagnostics, outside the timing runs. It generated the identical 502-token
answer with 4,096 context capacity, automatic 16 expert slots and FP16 KV.

| Counter | Bytes |
| --- | ---: |
| Peak process physical footprint | 2,322,467,984 (about 2.16 GiB) |
| Peak process RSS | 1,996,816,384 |
| Current Metal allocation after generation | 3,628,318,720 |
| Mapped core buffer capacities | 1,512,804,412 |
| Expert slot buffer capacities | 1,785,200,640 |
| Target KV/state buffer capacities | 306,708,480 |

These counters overlap and must not be added together. Buffer capacity does
not measure physical residency; filesystem-cache attribution and the complete
scratch inventory are unknown. The peak covers this CLI process and workload,
not all contexts or a server peak, and establishes no minimum-RAM guarantee.
The real tool-loop server separately reported a current footprint of
2,338,606,392 bytes and 22,937,600 bytes of recovery-buffer capacity after a
request. Full counters and scope are kept in the local validation record.
