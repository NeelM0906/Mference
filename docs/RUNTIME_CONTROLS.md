# Runtime controls

The CLI and the loopback server expose Mference's generation and runtime
controls; API clients reach the same generation controls through the server's
Chat Completions request fields. Existing families use FP16 KV; Maple uses
native BF16 KV. Generation settings apply to the next request; load-time
settings are fixed for the life of a loaded model.

## CLI modes

The CLI runs in exactly one of three modes, and they are mutually exclusive:

| Mode | Flag | Effect |
| --- | --- | --- |
| Raw completion | `--prompt <string>` | Sends the text to the model with no chat formatting. |
| Single-shot chat | `--messages-file <path>` | Renders a JSON message array through the model's chat template. |
| Interactive chat | `--chat` | Reads turns from standard input and keeps the conversation in memory. |

`--chat` loads the model once and re-renders the whole conversation through the
loaded model's chat template on every turn, so it follows that checkpoint's own
dialect. Type a message and press Return to send it; `/clear` starts a fresh
conversation, `/history` prints the messages held so far, and `/quit` (or
`/exit`, or end-of-file with <kbd>Control</kbd>+<kbd>D</kbd>) exits. Generated
text goes to standard output and prompts, notices, and the timing footer go to
standard error. <kbd>Control</kbd>+<kbd>C</kbd> does nothing while the prompt
waits for input, and ends the whole session rather than one turn while a
response is generating.

Each turn re-prefills the entire conversation from a reset KV cache, so no
state carries between turns and later turns in a long conversation take longer
to start.

`--reuse-prefix` is an opt-in alternative for `--chat`: the KV cache is kept
between turns and a turn prefills only what it adds. It applies the same
single-prefix rules as the loopback server's
[prompt reuse](OPENAI_SERVER.md#prompt-reuse): ChatML checkpoints continue from
the turn as it was generated and append the new user message. A turn that cannot continue resets as above, for example after `/clear`,
when history was dropped to fit the context, when a reply was ended by `--stop`,
or when the continued prompt would no longer fit `--max-context`. A resumed turn
reports `cached=<n>tok` in its timing footer.

Output with `--reuse-prefix` is not guaranteed to be byte-identical to the
default. A resumed turn's new tokens pass through prefill as one short chunk,
and a continued ChatML turn keeps earlier replies as generated rather than
re-rendered. Thinking is unaffected: every turn is rendered with the session's
`--reasoning-effort`.

`--show-reasoning` streams the thoughts of `--chat` turns to standard error,
each thought framed as a `[reasoning]` ... `[/reasoning]` block, while standard
output stays the visible answer only. It shows what the chat loop already
separates from the answer: Qwen 3.6 turns run with `--reasoning-effort`. It does not change what is generated or what enters
the conversation history, and it requires `--chat`.

`--system <string>` sets the system message for `--chat` and is repeatable;
repeated values join with newlines. It requires `--chat`, because the other two
modes carry their own prompt text.

When a conversation no longer fits `--max-context`, the oldest messages are
dropped until it does. The system message and the message just typed are never
dropped; if that pair alone still does not fit, the turn is refused and the
conversation is left untouched.

## Generation controls

The CLI and server expose these generation controls:

| Control | Values | CLI flag | Default | Effect |
| --- | --- | --- | --- | --- |
| Maximum response | 1 up to the remaining context | `--max-new` | 1,024 tokens | Caps generated tokens, including hidden reasoning. A request may use only the context space left after formatting the prompt; a limit reached during reasoning can leave the visible answer empty. |
| Maximum context | 1 up to the model's native context, or `max` | `--max-context` | CLI 4K; server/UI 16K | Sets prompt plus response capacity. Native contexts: 262,144 for Qwen 3.6 and Qwen 3.8; 131,072 for MiniCPM5; 128,000 for Maple. A larger value refuses the load. A selectable context is not a fresh hardware qualification. |
| Reserve full KV | Off or on | `--kv-reserve` | Off | Off: Qwen 3.6 starts full-attention KV at 16,384 tokens and grows it with the conversation (a longer prompt to the prompt plus 16,384, a longer answer by 8,192 at a time), shrinking it for a new conversation. On: reserve `--max-context` at load, as the other families always do. Also on the server. |
| Qwen 3.8 reasoning effort | `xhigh`, `medium`, `low`, `none` | `--reasoning-effort` | Unchanged legacy policy when omitted | Qwen 3.8; chat/messages mode, not raw completion. Server field: `reasoning_effort`. An explicit value selects the installed source template. `medium` adds no effort instruction; `none` closes thinking in the prompt. |
| Qwen 3.6 thinking | `xhigh`, `medium`, `low`, `none` | `--reasoning-effort` | Off | The first three aliases enable the same binary thinking mode; `none` disables it. Chat/messages mode only. The server also accepts `chat_template_kwargs.enable_thinking`; explicit effort wins. CLI stdout contains the visible answer, while interactive history retains reasoning for template replay. `--chat --show-reasoning` streams that reasoning to stderr. CLI turns still reset/re-prefill; CLI completion defaults remain 1,024 tokens. |
| Temperature | 0...2 | `--temperature` | 0.8 | `0` is greedy; positive values sample. |
| Top-K | Off or 1...256 | `--top-k` | 40 | Keeps at most K candidates. CLI `0` turns it off. |
| Top-P | Off or 0.01...1 | `--top-p` | 0.95 | Keeps the nucleus after Top-K, normalized over that candidate set. |
| Min-P | 0...1 | `--min-p` | 0.05 | Removes candidates below this fraction of the peak probability, before temperature. `0` disables it. |
| Repetition penalty | Positive finite | `--repetition-penalty` or `--repeat-penalty` | 1 | Divides positive seen-token logits and multiplies nonpositive ones. |
| Presence penalty | -2...2 | `--presence-penalty` | 0 | Subtracts once per seen token, after repetition penalty. |
| Frequency penalty | -2...2 | `--frequency-penalty` | 0 | Subtracts the penalty times each token's count. |
| Penalty window | -1 or nonnegative | `--repeat-last-n` | 64 | All penalties use the most recent N prompt/generated tokens. `0` disables penalties; `-1` uses all current history. |

Existing checkpoints use llama.cpp's built-in sampling preset. The active chain
is repetition/frequency/presence penalties → Top-K → Top-P → Min-P →
temperature. Seeded output is reproducible within Mference, not guaranteed
identical to llama.cpp's RNG or different weight formats. Thinking and context
or output limits retain their existing defaults.

With positive temperature, a CLI or server Top-P below `1` requires Top-K between `1`
and `256`. To disable all truncation controls, pass `--top-k 0 --top-p 1 --min-p 0`.
Generation controls apply to the next request and do not require a model
reload. They are interactive product settings, not the fixed community
benchmark protocol.

If a CLI chat exhausts its token budget before emitting visible text, stderr
now explains the truncation and points to budget controls. It does not invent an answer, change settings
automatically, or treat a truncated reply as an end-of-turn success.

## Runtime settings

| Control | Values | CLI flag | Production default | Effect |
| --- | --- | --- | --- | --- |
| Expert-cache slots | 8, 16, 24, 32, 64, 96, 128; CLI also accepts resident and auto | `--expert-cache-slots` | CLI/server auto | Qwen 3.6 auto uses 96 slots on hosts with at least 24 GiB, 32 with at least 16 GiB, and 16 otherwise. Other families use 16. `resident` maps every layer file once and skips the slot cache. It lost the Qwen 3.6 community A/B on 24 GiB because of page-cache pressure, so it is not a universal default. More slots retain more routed experts and reduce later reads at the cost of RAM. Ordinary RSS substantially undercounts clean file-backed pages in resident mode. |
| Prompt prefill | On, off | — | On | On requests chunked prefill. Recurrent scans inside a layer-major GPU batch still advance in token order where required. Off selects scalar replay, not skipped prompt processing. |
| RDADVISE | Off, Default, Bounded, Adaptive | `--rdadvise` | Off | Applies experimental read advice. Its effect depends on the workload; it may help a short decode and slow a long one. |
| Prefill chunk tokens | 32, 64, 128, 256, 512, 1024, 2048, 4096, or auto | `--prefill-chunk` | Auto (one-shot); the server's chunk (`--chat`) | Tokens processed per prefill chunk. Larger chunks re-read the routed experts fewer times, which lowers prefill I/O and time. `auto` picks the smallest allowed size that covers a one-shot prompt; interactive `--chat` has no prompt at load time, so auto takes the server's chunk for the model family (below). Maple stages each chunk layer-major but preserves its fixed 512-slot sliding-cache semantics by committing and attending rows in time order. The server takes `--prefill-chunk` as well; their `auto` uses 128 tokens, except on hosts with at least 16 GiB, where Qwen 3.6 uses 2,048. Qwen 3.6 has no sliding-window KV to grow, so its 2,048 costs about 270 MB of scratch. `MFERENCE_SERVER_PREFILL_CHUNK` accepts any listed size; the flag wins over it. |
| Maple FlashHead | Off, on | `--flash-head` | Off | Enables Maple's approximate singleton-decode candidate head when the install carries validated FlashHead tensors. It leaves all non-candidates at negative infinity, so sampling is restricted to selected rows. Prefill and the default decode head remain exact; an install without the data falls back to the exact head. |
| Model verification | Auto, full SHA-256, trusted receipt | `--verify` (CLI and server) | Auto | `auto` checks each routed-expert file's size against the receipt written at install time when that receipt validates (it must exist, match the manifest hash and name the install's current directory), and falls back to `full-sha256` otherwise, for example for an install that was moved or has no receipt; a fallback prints one `notice: --verify auto: …` line on stderr that names the failed receipt check (the server logs it on each load). `full-sha256` re-hashes each routed-expert file on first touch, which for a 145 GB expert pool costs about 59 s inside the first prefill and, on the server, again after every model swap. `trusted-receipt` requires the receipt and fails the load without it. In every mode `manifest.json`, `model_weights.bin` and `layout.json` are hashed at load and the receipt is validated against the manifest hash. The receipt modes trade detection of size-preserving corruption of expert files for that time; choose `full-sha256` when that matters. |
| Library catalog | `product`, `all` | `--catalog` (server with `--library`) | `product` | Which installs library mode serves. `product` lists only the product models (`qwen3.6-35b-a3b`, `minicpm5-2b-int4g64` and `maple-preview-2bit-mlx` among the native installs) under their stable identifiers; every other runnable install, the native Qwen 3.8 27B included, is declined, and a request for it gets HTTP 400 `model_not_runnable` naming `--catalog all`. `all` is a developer switch that lists every runnable install: product models under their stable identifiers, the rest under their discovery identifiers (see [library mode](OPENAI_SERVER.md#library-mode)). GGUF models are listed only when the build has the llama.cpp backend. See [the engine](ENGINE.md). |
| Server prompt cache | `off`, `single-prefix`, `multi-prefix` | `--prompt-cache-mode` (server) | `multi-prefix` | Which finished turns the server keeps for KV prefix reuse. `multi-prefix` keeps every finished turn of every conversation for Qwen 3.6, within `--prompt-cache-bytes`, and is exactly `single-prefix` for every other model. A restored turn continues from exactly the state it ended in, so its output equals what `single-prefix` gives the same conversation without interleaved requests; see [prompt reuse](OPENAI_SERVER.md#prompt-reuse). |
| Server prompt cache budget | Bytes, or auto | `--prompt-cache-bytes` (server) | Auto | RAM ceiling for what `multi-prefix` keeps besides the live conversation: per finished turn, about 61 MiB of recurrent state (Qwen 3.6), plus, for each conversation or branch that is not in the runner, 20 KiB per token of attention KV. Least recently used turns are released first. Under system memory pressure (warning or critical) nothing new is kept, and every copy except the conversation in the runner is released between requests. Swap growth of more than 256 MiB after the copies last grew releases them back to their size before that growth, and nothing new is kept until swap has grown by less than 16 MiB over a minute. `auto` is the smaller of 4 GiB and an eighth of physical memory with at least 24 GiB, 1 GiB from 16 GiB, and 0 below; `0` is exactly `single-prefix`. |
| Speculative expert prefetch | 0 (off), 1–8 | `--shadow-budget` (CLI and server) | For Qwen 3.6: 4 on hosts from 16 to under 24 GiB, 8 from 24 GiB up to 32 GiB; off elsewhere | During decode, predicts the next layer's routed experts and reads up to this many of them per layer into the existing expert slots before the router asks, without ever blocking a real read. It shortens the time the GPU waits for the SSD, costs no memory, and produces byte-identical output at every value; a higher budget reads more from the SSD. `0` turns it off. An explicit value applies on every host and takes precedence over `MFERENCE_SPEC_PREFETCH` and `MFERENCE_SHADOW_BUDGET`. |

The FlashHead switch is a CLI-only control; every other surface uses the default
exact head. `--prefill-chunk` applies to the CLI and the server.
The CLI applies these settings when it loads the model, so each run uses the
values passed on its command line. Setting `MFERENCE_PHASES=1` makes the
CLI print the decode phase report after the timing footer: `cb1` and `cb2`
encode-and-commit time, expert I/O await split into GPU-overlapped and exposed
time, the all-hit layer-step rate, GPU busy/span/gap, speculative-prefetch
counters, and unaccounted GPU waits. After a chunked prefill it also prints
the count, requested bytes and latency histogram of every routed-expert read
the `pread` streamer issued during decode and during the last chunked prefill
call, and the bytes the whole process physically read from storage
(`proc_pid_rusage`; page-cache hits excluded, but mapped core-weight faults
and other file reads included) in each window. For the Qwen 3.6
decode path it adds routed-expert hit and miss counts with the most-missed
layer:expert pairs and misses per layer, and the same all-hit rate recomputed
from those counters; the per-expert counts are on
`RealForwardRunner.expertRoutingCounters`. With resident experts nothing is
read and every routed expert counts as a hit. Failed reads are not
recorded, and a background read still in flight when a window opens is
counted in the new window. `MFERENCE_PREFILL_BREAKDOWN=1` prints, for
Qwen 3.6 chunked prefill, a stage table: GPU time for the attention or
Gated-DeltaNet projections and core, router, shared expert, routed tiles, MoE
tail and head, plus the CPU time spent waiting on routed expert reads. To time
each stage it waits at stage boundaries, so its wall times describe the
diagnostic run, not a production one. Both are diagnostics and do not change
output.

The accepted decode defaults carry environment kill-switches for A/B runs.
`MFERENCE_SLOT_MAP=0` disables the GPU-resident expert-to-slot map that lets
Qwen layers whose eight experts are all cached skip CPU expert planning,
fetching, and routed-command encoding — the router readback itself remains on
the CPU (default on). `MFERENCE_EAGER_ROUTED=0` disables the eager routed commit,
which commits the routed command buffer before its expert fills land, gated on
a shared event; a failed eager fill aborts the decode step with an error
rather than emitting corrupt output (default on). `MFERENCE_ROUTER_EVENT=0`
disables the early mid-buffer router readback. `MFERENCE_SPEC_PREFETCH`
selects the speculative-prefetch mode — shadow prefetch is the default for
Qwen 3.6 on hosts from 16 GiB up to 32 GiB, off elsewhere — and `MFERENCE_SHADOW_BUDGET` caps
its per-layer speculative reads. For Qwen 3.8, `MFERENCE_MTP=0` disables MTP
speculative decoding (on by default for greedy decode when the install
carries the attached MTP tensors)
and `MFERENCE_MTP_K` (1–6, default 3) sets
the draft depth. `MFERENCE_DFLASH2_DIR` points at a DFlash2 drafter
checkpoint and swaps the round's draft source to it (draft depth defaults
to 6); `MFERENCE_DFLASH2_BF16=1` skips its
load-time INT4 quantization for reference runs. All are byte-identical
toggles, not quality controls.

Changing context length, expert-cache slots, RDADVISE, model verification,
prompt-prefill enablement, the prefill chunk size, or FlashHead selection
requires a reload.
Some sampling changes also require a reload because greedy and sampled
generation use different output-head paths.

Multi-turn chat history is fitted with the model tokenizer before generation.
The bounded local compression pass that replaced older turns with a rolling
summary belonged to the removed Mac app; the CLI and server instead drop the
oldest messages, and never silently discard the current user turn.

### macOS interactivity mitigation

Before the first Metal device is created, Mference defaults
`AGX_RELAX_CDM_CTXSTORE_TIMEOUT` to `1`. This relaxes an AGX context-store
deadline that can terminate a long prefill command buffer as
`kIOGPUCommandBufferCallbackErrorImpactingInteractivity` on macOS 26. It is a
mitigation, not a guarantee: upstream reports failures with the setting
enabled too.

Export `AGX_RELAX_CDM_CTXSTORE_TIMEOUT=0` before launching the CLI or server to
restore stock driver behaviour; an explicit value is never overwritten. When a
Qwen 3.6 prefill command buffer still fails, the error names its
phase label (chunk start and size, layer, phase), Metal status, error domain,
code and description, including the IOGPU token, and no prompt or generated
content. A failure reported only through the buffer's status, with no error
object, is caught as well. A failed sampling command buffer also stops the
request. The Qwen 3.6 decode step keeps its existing handling: it
prints a failed command buffer with the same detail and continues.

## Run an experiment

1. Start from 4K context, the automatic expert-cache choice, prefill on, and RDADVISE off.
2. Keep the prompt and generation controls fixed.
3. Record a baseline after a warmup.
4. Change one runtime control and reload the model.
5. Compare prompt prefill, request TTFT, decode rate, peak memory, and I/O per
   token over repeated runs.
6. Restore the production defaults when the experiment ends.

Use the [community benchmark protocol](COMMUNITY_BENCHMARKS.md) for a standard
production result. A run with changed runtime controls is experimental and must
name the changed setting.

## Read the results

- **Decode rate** measures generated tokens per second after prompt prefill.
- **Request TTFT** includes prompt prefill and the wait for the first generated
  token.
- **Peak memory** in Last run is the highest decode-service memory observed
  during the request. The HUD shows the service's current memory instead of the
  much smaller foreground UI process.
- **I/O / token** reports routed-expert read time per generated token.
- **Advanced** shows decode duration and per-token cb1, cb2, and output-head
  time. When RDADVISE runs, it also shows time, calls, data, and skipped advice.

During chunked prefill, the phase label reports exact progress, for example
`Prefill (128/514)`. Errors and unsupported configurations appear only when
they occur. RDADVISE remains experimental and is off by default. A measured
result is a data point, not a performance ceiling.
