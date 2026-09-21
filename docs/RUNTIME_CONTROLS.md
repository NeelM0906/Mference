# Runtime controls

The CLI and the loopback server expose Mference's generation and runtime
controls; Open WebUI reaches the same generation controls through the server's
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
the turn as it was generated and append the new user message, and Gemma rewinds
to the point where the freshly rendered conversation diverges from the cached
one. A turn that cannot continue resets as above, for example after `/clear`,
when history was dropped to fit the context, when a reply was ended by `--stop`,
or when the continued prompt would no longer fit `--max-context`. A resumed turn
reports `cached=<n>tok` in its timing footer.

Output with `--reuse-prefix` is not guaranteed to be byte-identical to the
default. A resumed turn's new tokens pass through prefill as one short chunk,
and a continued ChatML turn keeps earlier replies as generated rather than
re-rendered. For Gemma this already applies to the first turn: prefill captures
a recovery point just before the generation suffix, which splits the last chunk
and sends its few remaining tokens through the short-chunk kernels. Thinking
is unaffected: every turn is rendered with the session's `--reasoning-effort`,
and Gemma's template still removes earlier thoughts at a new user message.

`--show-reasoning` streams the thoughts of `--chat` turns to standard error,
each thought framed as a `[reasoning]` ... `[/reasoning]` block, while standard
output stays the visible answer only. It shows what the chat loop already
separates from the answer: Gemma and Swift-Qwen turns, and Qwen 3.6 turns run
with `--reasoning-effort`. It does not change what is generated or what enters
the conversation history, and it requires `--chat`.

`--system <string>` sets the system message for `--chat` and is repeatable;
repeated values join with newlines. It requires `--chat`, because the other two
modes carry their own prompt text.

When a conversation no longer fits `--max-context`, the oldest messages are
dropped until it does. The system message and the message just typed are never
dropped; if that pair alone still does not fit, the turn is refused and the
conversation is left untouched.

### Gemma QAT

The separate `gemma4qat` checkpoint uses the existing
`$HOME/llm-models/gemma4qat.gturbo` installation:

```bash
.build/release/MferenceCLI --model "$HOME/llm-models/gemma4qat.gturbo" \
  --prompt "The capital of France is" --max-new 64 --seed 42
.build/release/MferenceCLI --model "$HOME/llm-models/gemma4qat.gturbo" \
  --chat --reasoning-effort medium --max-new 256 --seed 42
```

Omitted controls use the verified installed `generation_config.json`:
temperature 1, Top-K 64 and Top-P 0.95. Min-P is off, repetition penalty is 1,
and presence/frequency penalties are zero. Explicit flags override these
values, including `--temperature 0` for greedy decoding and `--min-p 0`.
These defaults apply to CLI raw/chat/messages and the selected server model,
including custom aliases and library swaps. The original Gemma retains its
existing defaults. QAT requires its installed
tokenizer and assets; it does not fall back to a remote tokenizer or an
environment override.

Raw completion applies no chat template and can include model-generated
channel text. The qualified raw greedy prompt `The capital of France is`
repeats in both Mference and the pinned FP16 reference; QAT is not inherently
loop-free. Chat uses the pinned checkpoint's own template. Thinking is off
when omitted; `medium`, `low` and `xhigh` all enable the same binary mode.
The source drops ordinary assistant reasoning and replays tool-call reasoning
only within the active user turn. The HTTP server accepts `preserve_thinking`
and normalizes it to this source policy for both Gemma checkpoints; it does
not disable thinking or retain older thoughts. Clients still send complete tool
history, including reasoning and matching tool-call IDs.

No reinstall is needed for this runtime update. Kernel, CLI chat and both server modes
have passed scoped qualification on M2, including tools and cache recovery.
M2 timing observations and a separate short-chat CLI peak-memory measurement
are recorded. General loop reduction, other hardware and wider-context resource
limits remain unverified.

## Generation controls

The CLI and server expose these generation controls:

| Control | Values | CLI flag | Default | Effect |
| --- | --- | --- | --- | --- |
| Maximum response | 1 up to the remaining context | `--max-new` | 1,024 tokens | Caps generated tokens, including hidden reasoning. A request may use only the context space left after formatting the prompt; a limit reached during reasoning can leave the visible answer empty. |
| Maximum context | 4K, 8K, 16K, 32K, 64K, 128K | `--max-context` | CLI 4K; server/UI 16K | Sets prompt plus response capacity. Maple supports 128000 tokens in the runtime, CLI, and server; other family or product limits may differ. A selectable context is not a fresh hardware qualification. |
| Qwen 3.8 reasoning effort | `xhigh`, `medium`, `low`, `none` | `--reasoning-effort` | Swift: `xhigh`; base: unchanged legacy policy when omitted | Base/Swift Qwen 3.8; chat/messages mode, not raw completion. Server field: `reasoning_effort`. An explicit value selects the installed source template for either checkpoint. `medium` adds no effort instruction; `none` closes thinking in the prompt. |
| Gemma 4 / Qwen 3.6 thinking | `xhigh`, `medium`, `low`, `none` | `--reasoning-effort` | Off | The first three aliases enable the same binary thinking mode; `none` disables it. Chat/messages mode only. The server also accepts `chat_template_kwargs.enable_thinking`; explicit effort wins. CLI stdout contains the visible answer, while interactive history retains reasoning for template replay. `--chat --show-reasoning` streams that reasoning to stderr. Gemma's template strips earlier ordinary reasoning at a new user turn. CLI turns still reset/re-prefill; CLI completion defaults remain 1,024 tokens. |
| Temperature | 0...2 | `--temperature` | 0.8; QAT 1 | `0` is greedy; positive values sample. |
| Top-K | Off or 1...256 | `--top-k` | 40; QAT 64 | Keeps at most K candidates. CLI `0` turns it off. |
| Top-P | Off or 0.01...1 | `--top-p` | 0.95 | Keeps the nucleus after Top-K, normalized over that candidate set. |
| Min-P | 0...1 | `--min-p` | 0.05; QAT 0 | Removes candidates below this fraction of the peak probability, before temperature. `0` disables it. |
| Repetition penalty | Positive finite | `--repetition-penalty` or `--repeat-penalty` | 1 | Divides positive seen-token logits and multiplies nonpositive ones. |
| Presence penalty | -2...2 | `--presence-penalty` | 0 | Subtracts once per seen token, after repetition penalty. |
| Frequency penalty | -2...2 | `--frequency-penalty` | 0 | Subtracts the penalty times each token's count. |
| Penalty window | -1 or nonnegative | `--repeat-last-n` | 64 | All penalties use the most recent N prompt/generated tokens. `0` disables penalties; `-1` uses all current history. |

Existing checkpoints use llama.cpp's built-in sampling preset; QAT
uses the source defaults described above. The active chain
is repetition/frequency/presence penalties → Top-K → Top-P → Min-P →
temperature. Seeded output is reproducible within Mference, not guaranteed
identical to llama.cpp's RNG or different weight formats. Thinking and context
or output limits retain their existing defaults. See
[the reference and compatibility details](LLAMA_SAMPLING.md).

With positive temperature, a CLI or server Top-P below `1` requires Top-K between `1`
and `256`. To disable all truncation controls, pass `--top-k 0 --top-p 1 --min-p 0`.
Generation controls apply to the next request and do not require a model
reload. They are interactive product settings, not the fixed community
benchmark protocol.

If a CLI chat exhausts its token budget before emitting visible text, stderr
now explains the truncation and points to budget controls (and Swift's supported
effort controls when applicable). It does not invent an answer, change settings
automatically, or treat a truncated reply as an end-of-turn success.

## Runtime settings

| Control | Values | CLI flag | Production default | Effect |
| --- | --- | --- | --- | --- |
| Expert-cache slots | 8, 16, 24, 32, 64, 96, 128; CLI also accepts resident and auto | `--expert-cache-slots` | CLI/server auto | Qwen 3.6 auto uses 96 slots on hosts with at least 24 GiB, 32 with at least 16 GiB, and 16 otherwise. Flash-Next auto maps the routed-expert pool on hosts with at least 192 GiB when that pool plus the core leaves 32 GiB of headroom; otherwise it uses 16 slots. GLM selects resident when its pool plus core plus 48 GiB of reserve fits physical memory; otherwise it uses 16 slots. Other families use 16. `resident` maps every layer file once and skips the slot cache. This won for Flash-Next on the 256 GiB M3 Ultra but lost the Qwen 3.6 community A/B on 24 GiB because of page-cache pressure, so it is not a universal default. More slots retain more routed experts and reduce later reads at the cost of RAM. Ordinary RSS substantially undercounts clean file-backed pages in resident mode. |
| Prompt prefill | On, off | — | On | On requests chunked prefill. The merged GLM bounded-expert path and DeepSeek sparse-cutover path now batch rather than replaying the full model per prompt token. Real-checkpoint and hardware coverage remains separate: see the [qualification matrix](PREFILL_QUALIFICATION.md). Recurrent scans inside a layer-major GPU batch still advance in token order where required. Off selects scalar replay, not skipped prompt processing. [Runtime diagnostics](RUNTIME_DIAGNOSTICS.md) reports actual per-request counts and separate memory metrics. |
| RDADVISE | Off, Default, Bounded, Adaptive | `--rdadvise` | Off | Applies experimental read advice. Its effect depends on the workload; it may help a short decode and slow a long one. |
| Prefill chunk tokens | 32, 64, 128, 256, 512, 1024, 2048, 4096, or auto | `--prefill-chunk` | Auto (one-shot); the server's chunk (`--chat`) | Tokens processed per prefill chunk. Larger chunks re-read the routed experts fewer times, which lowers prefill I/O and time. `auto` picks the smallest allowed size that covers a one-shot prompt; interactive `--chat` has no prompt at load time, so auto takes the server's chunk for the model family (below). Maple stages each chunk layer-major but preserves its fixed 512-slot sliding-cache semantics by committing and attending rows in time order. The server has no flag: it uses 128 tokens, except on hosts with at least 16 GiB, where Gemma 4 and Gemma 4 QAT use 1,024 (about 309 MB more scratch and sliding-window KV) and Qwen 3.6 uses 2,048 (about 270 MB more scratch; it has no sliding-window KV to grow). `MFERENCE_SERVER_PREFILL_CHUNK` accepts any listed size. |
| Maple FlashHead | Off, on | `--flash-head` | Off | Enables Maple's approximate singleton-decode candidate head when the install carries validated FlashHead tensors. It leaves all non-candidates at negative infinity, so sampling is restricted to selected rows. Prefill and the default decode head remain exact; an install without the data falls back to the exact head. |
| Model verification | Full SHA-256, trusted receipt | `--verify` | Full SHA-256 | `full-sha256` re-hashes each routed-expert file on first touch, which for a 145 GB expert pool costs about 59 s inside the first prefill. `trusted-receipt` instead checks each file's size against the receipt written at install time; the receipt itself is still validated against the manifest hash, and `model_weights.bin` and `layout.json` are still hashed. It trades detection of size-preserving corruption for that time. |

The `--prefill-chunk` flag and FlashHead switch are CLI-only controls; every
other surface uses the default exact head and the server chunk described above.
The CLI applies these settings when it loads the model, so each run uses the
values passed on its command line. Setting `MFERENCE_PHASES=1` makes the
CLI print the decode phase report after the timing footer: `cb1` and `cb2`
encode-and-commit time, expert I/O await split into GPU-overlapped and exposed
time, the all-hit layer-step rate, GPU busy/span/gap, speculative-prefetch
counters, and unaccounted GPU waits. `MFERENCE_PREFILL_BREAKDOWN=1` prints the
Inkling prefill routed-expert split (fetch, encode, drain). Both are
diagnostics and do not change behavior.

The accepted decode defaults carry environment kill-switches for A/B runs.
`MFERENCE_SLOT_MAP=0` disables the GPU-resident expert-to-slot map that lets
Qwen layers whose eight experts are all cached skip CPU expert planning,
fetching, and routed-command encoding — the router readback itself remains on
the CPU (default on). `MFERENCE_EAGER_ROUTED=0` disables the eager routed commit,
which commits the routed command buffer before its expert fills land, gated on
a shared event; a failed eager fill aborts the decode step with an error
rather than emitting corrupt output (default on). `MFERENCE_ROUTER_EVENT=0`
disables the early mid-buffer router readback. `MFERENCE_SPEC_PREFETCH`
selects the speculative-prefetch mode — shadow prefetch is the accepted
DeepSeek-V4-Flash default, off elsewhere — and `MFERENCE_SHADOW_BUDGET` caps
its per-layer speculative reads. For Qwen 3.8, `MFERENCE_MTP=0` disables MTP
speculative decoding (on by default for greedy decode when the install
carries the attached MTP tensors; **off by default for the Swift-Qwen candidate**)
and `MFERENCE_MTP_K` (1–6, default 3) sets
the draft depth. `MFERENCE_DFLASH2_DIR` points at a DFlash2 drafter
checkpoint and swaps the round's draft source to it (draft depth defaults
to 6; see docs/QWEN38_DFLASH2.md); `MFERENCE_DFLASH2_BF16=1` skips its
load-time INT4 quantization for reference runs. All are byte-identical
toggles, not quality controls.

Gemma prefill has two switches that are not byte-identical, because the
defaults reorder floating-point sums. `MFERENCE_GEMMA_PREFILL_LEGACY=1` returns
Gemma 4 and Gemma 4 QAT to the per-token shared expert and per-row routed
experts; the default batches the INT4 shared expert and runs well-filled
routed tiles as grouped matrix products. `MFERENCE_QAT_EXACT_PREFILL=1`
additionally returns QAT's prefill projections, shared expert and routed
experts to the MLX FP16 reduction order, which is several times slower on long
prompts. QAT decode, routing, normalization and attention keep that order in
every mode. Both exist for A/B runs and for the
[prefill equivalence gate](families/GEMMA4_QAT.md#prefill-arithmetic).

Flash-Next's installer carries an MTP sidecar, but native Flash-Next speculative
execution is not implemented. The dense Qwen MTP switches do not activate it.

Changing context length, expert-cache slots, RDADVISE, model verification,
prompt-prefill enablement, the prefill chunk size, or FlashHead selection
requires a reload.
Some sampling changes also require a reload because greedy and sampled
generation use different output-head paths.

Multi-turn chat history is fitted with the model tokenizer before generation.
The bounded local compression pass that replaced older turns with a rolling
summary belonged to the removed Mac app; the CLI and server instead drop the
oldest messages, and never silently discard the current user turn.

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
