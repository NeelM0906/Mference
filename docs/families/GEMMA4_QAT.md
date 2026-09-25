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
tool-call reasoning within the active user turn. The HTTP server accepts
`preserve_thinking=true` and normalizes it to this source policy, as it does
for ordinary Gemma. It does not change the pinned template. Clients send full histories
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
  evidence, not bit-identical execution for every prompt. That record used
  the source-order prefill: the default build keeps it for decode and
  reproduces the chunked positions only with `MFERENCE_QAT_EXACT_PREFILL=1`;
  see [Prefill arithmetic](#prefill-arithmetic).
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

## Prefill arithmetic

Since 2026-09-21 the default prefill runs QAT's Q/K/V/O projections, shared
expert and routed experts on the kernels original Gemma uses, batches the INT4
shared expert, and runs well-filled routed tiles as grouped matrix products.
Decode, routing, normalization and attention keep the MLX FP16 reduction
order. The source-order prefill kernels spend 32 GPU threads on every
(token, row) dot product and were 3-5x slower on long prompts; these kernels
change only floating-point summation order.

| 3,015-token prompt, M2 MacBook Air 16 GiB | Before | After |
| --- | ---: | ---: |
| QAT, CLI, one chunk | 160.6 s | 38.6 s |
| QAT, server at 16K context | 245.4 s (128-token chunks) | 50.4 s (1,024-token chunks) |
| Original Gemma, CLI, one chunk | 75.6 s | 52.4 s |

These are diagnostic runs with cool-down pauses on a fanless Mac, not
community-protocol benchmarks; decode rates were unchanged. The server's larger
Gemma chunk costs 307 MB of Metal allocation (KV +183.5 MB) and applies to
hosts with at least 16 GiB; see [Runtime controls](../RUNTIME_CONTROLS.md).

What the numbers above do and do not keep:

- Decode is unchanged: all 149 scalar reference positions stay byte-identical
  to the MLX-exact capture.
- Chunked prefill no longer reproduces the scalar MLX oracle. On the raw-text
  reference corpus 4 of 9 chunked positions remain inside the frozen limits and
  5 do not (relative L2 up to 0.44 on the repetitive sequence). Reordered sums
  flip near-tied experts in MoE routing, and the flips cascade; stock MLX also
  changes kernels for batched prompts.
- `MFERENCE_QAT_EXACT_PREFILL=1` reproduces all 158 positions byte for byte
  (verified 2026-09-21). Use it for the MLX reference comparison.
- The default is gated instead on teacher-forced perplexity against that exact
  control, on identical tokens after the frozen community prompts. QAT: 1,000
  predictions, NLL +0.0004 nats/token (95 % -0.0014..+0.0022), same top
  prediction at 99.6 %. Original Gemma, whose shared and routed experts changed
  the same way: +0.0029 (-0.0028..+0.0086), 98.3 %. On the real weights the new
  expert kernels match the old within 5e-4 relative error for every one of
  3,015 layer-0 tokens.

```bash
MFERENCE_GEMMA_PREFILL_GATE="$HOME/llm-models/gemma4qat.gturbo" \
  Scripts/test.sh --filter GemmaPrefillEquivalenceGateTests
```

The gate reads the install, loads the model once and takes about 15 minutes on
the M2; apply the model-process checks first.

## M2 generation measurements

The prefill times in this section predate the prefill change above.

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

## Shadow prefetch during decode (2026-09-21)

Gemma 4 and Gemma 4 QAT keep 16 expert slots per layer, so only 8% of decode
layer steps find all eight routed experts in memory and the GPU waits for the
SSD for roughly half of decode. On hosts from 16 GiB to below 24 GiB both
checkpoints now use shadow prefetch with a budget of four speculative reads per
layer; `--shadow-budget` sets it on the CLI and the server, and `0` turns it
off. It reads into the existing slots, so it costs no memory.

| M2 MacBook Air 16 GiB, 160 decoded tokens, greedy | Off | Budget 1 | Budget 2 | Budget 4 |
| --- | ---: | ---: | ---: | ---: |
| Gemma 4 decode (tok/s) | 6.86 | 7.22 | 7.63 | 7.70 |
| Gemma 4 all-hit layer steps | 7.9% | 14.4% | 19.2% | 24.0% |
| Gemma 4 QAT decode (tok/s) | 6.44 | 6.77 | 7.16 | 7.11 |
| Gemma 4 QAT all-hit layer steps | 7.9% | 15.7% | 21.5% | 27.9% |
| Metal allocation, either checkpoint | unchanged | unchanged | unchanged | unchanged |

Output is byte-identical at every budget. Budget 4 brings no more speed than
budget 2 here while reading about half as much again from the SSD (about 40 GB
against 26 GB of speculative reads for 160 tokens), because with 16 slots a
step still waits whenever any one of its experts is missing. The slot count is
what limits Gemma: with 32 slots the all-hit rate reaches 37% without prefetch
and 55–61% with it, at a cost of about 1.6 GB, which is why 32 slots is not the
default. These are single diagnostic runs with brief pauses on a fanless Mac;
an earlier, hotter session showed no clear gain for Gemma 4 (4.96 / 6.12 / 4.26
tok/s off against 5.35 / 5.20 / 5.44 with budget 2).
