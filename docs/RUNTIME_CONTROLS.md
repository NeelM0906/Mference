# Runtime controls

The Mac app exposes generation and runtime controls in its collapsible right
settings pane. Use the right-sidebar button in the status bar or
<kbd>Shift</kbd>+<kbd>Command</kbd>+<kbd>I</kbd> to hide or restore it. FP16 is
the fixed KV format. Generation settings apply to the next request; load-time
settings require a reload.

Chat navigation lives separately in the collapsible left sidebar. Use its
**New chat** button or <kbd>Command</kbd>+<kbd>N</kbd> to create an independent
context. The left-sidebar buttons or
<kbd>Control</kbd>+<kbd>Command</kbd>+<kbd>S</kbd> toggle the chat list without
changing the right settings pane.

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

`--system <string>` sets the system message for `--chat` and is repeatable;
repeated values join with newlines. It requires `--chat`, because the other two
modes carry their own prompt text.

When a conversation no longer fits `--max-context`, the oldest messages are
dropped until it does. The system message and the message just typed are never
dropped; if that pair alone still does not fit, the turn is refused and the
conversation is left untouched.

## Generation controls

The Mac app and CLI expose these generation controls:

| Control | Mac values | CLI flag | Default | Effect |
| --- | --- | --- | --- | --- |
| Maximum response | Automatic | `--max-new` | App: remaining context; CLI: 1,024 tokens | The app can use the context space left after formatting the prompt. The CLI uses its explicit or default `--max-new` limit. |
| Maximum context | 4K, 8K, 16K, 32K, 64K | `--max-context` | 4K | Sets prompt plus response capacity. The app shows the FP16 KV-memory delta. |
| Temperature | 0...2 in 0.05 steps | `--temperature` | 0.2 | `0` is greedy; positive values sample. |
| Top-K | Off or 1...256 | `--top-k` | 64 | Keeps at most K candidates. CLI `0` turns it off. |
| Top-P | Off or 0.01...1 | `--top-p` | 0.95 | Applies nucleus truncation before Top-K and is effective only while Top-K is enabled. |

With positive temperature, a CLI Top-P below `1` requires Top-K between `1`
and `256`. To disable both truncation controls, pass `--top-k 0 --top-p 1`.
Generation controls apply to the next request and do not require a model
reload. They are interactive product settings, not the fixed community
benchmark protocol.

## Runtime settings

| Control | Values | CLI flag | Production default | Effect |
| --- | --- | --- | --- | --- |
| Expert-cache slots | 8, 16, 24, 32 | `--expert-cache-slots` | 16 | More slots can retain more routed experts and reduce later reads, but values above 16 use more RAM. |
| Prompt prefill | On, off | — | On | On processes known prompt tokens through the chunked prefill path. Off disables that path. |
| RDADVISE | Off, Default, Bounded, Adaptive | `--rdadvise` | Off | Applies experimental read advice. Its effect depends on the workload; it may help a short decode and slow a long one. |

The CLI applies these settings when it loads the model, so each run uses the
values passed on its command line. Setting `MFERENCE_PHASES=1` makes the
CLI print the decode phase split (`cb1`, expert I/O await, `cb2`, and GPU
waits) after the timing footer; it is a diagnostic and does not change
behavior.

Changing context length, expert-cache slots, or RDADVISE requires a reload.
Some sampling changes also require a reload because greedy and sampled
generation use different output-head paths. Prompt-prefill settings apply to
each request and do not require a reload.

Multi-turn chat history is fitted with the model tokenizer before generation.
When older complete turns no longer fit, the app runs a bounded local
compression pass and replaces those turns in model context with a rolling
summary. The full transcript stays available in the UI, and each chat keeps a
separate summary. The current user turn is never silently discarded.

## Run an experiment

1. Start from 4K context, 16 expert-cache slots, prefill on, and RDADVISE off.
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
