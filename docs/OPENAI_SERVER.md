# Local OpenAI-compatible server

`MferenceServer` exposes a local Chat Completions API for one installed
supported model. It binds to `127.0.0.1` by default, or to the machine's exact
Tailscale IPv4 address with `--bind tailnet`. It has no application-level
authentication or TLS; do not expose it through a wildcard interface, proxy,
or tunnel.

By default it serves one model per process, named by `--model`.
[Library mode](#library-mode) (`--library`) instead serves every product model
from one process and swaps the resident model in place.

## Start the server

First, install the model with `MferenceRepack`. Then check that no other
Mference model process is running:

```bash
pgrep -fl 'MferenceServer|MferenceCLI|MferencePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'
```

If the command prints a match, do not start the server.

```bash
swift build -c release --product MferenceServer
.build/release/MferenceServer \
  --model scratch/qwen36.gturbo \
  --port 8080 \
  --max-context 16384
```

The server loads the model before opening the port. Wait for
`MferenceServer ready`, then keep the process running while clients use
it.

To reach the server from other devices in the same Tailnet, let it detect and
bind the machine's Tailscale IPv4 address:

```bash
.build/release/MferenceServer \
  --model scratch/qwen36.gturbo \
  --bind tailnet \
  --port 8080 \
  --max-context 32768 \
  --queue-limit 32
```

This requires the `tailscale` CLI on `PATH`. The server binds only that one
address; it never binds a wildcard interface. If Tailscale is missing, not
running, or reports anything other than a single Tailscale IPv4 address, the
command fails instead of falling back to a broader interface. The startup line
prints the address it actually bound.

`--bind tailnet` is not authentication. Access is governed entirely by the
Tailnet ACL, and every device the ACL admits gets unauthenticated access to the
full API. The server still has no application-level authentication or TLS.

Check the server from another terminal:

```bash
curl --silent --show-error http://127.0.0.1:8080/health
curl --silent --show-error http://127.0.0.1:8080/v1/models
curl --silent --show-error http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-35b-a3b",
    "messages": [{"role": "user", "content": "Reply with exactly READY."}],
    "temperature": 0,
    "max_completion_tokens": 16
  }'
```

By default, the server runs one generation and queues up to four requests. Use
`--queue-limit` to change the queue size. Press Control-C to stop the server.

Model integrity follows `--verify`, on the first load and on every library
swap. The default `auto` checks the routed-expert files against the install
receipt's sizes when that receipt validates and hashes them on first touch
otherwise, logging a `notice: --verify auto: …` line that names the failed
receipt check; `--verify full-sha256` always hashes, which adds that time to
the first prefill after each load. See
[Runtime settings](RUNTIME_CONTROLS.md#runtime-settings) for the trade-off.

## Connect a client

The base URL is `http://127.0.0.1:8080/v1`. Some client libraries require an
API key, but the server ignores it.

Python:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
response = client.chat.completions.create(
    model="qwen3.6-35b-a3b",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)
print(response.choices[0].message.content)
```

OpenCode:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "mference": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Mference",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1",
        "apiKey": "local"
      },
      "models": {
        "qwen3.6-35b-a3b": {
          "name": "Qwen 3.6 35B-A3B"
        }
      }
    }
  }
}
```

Select `mference/qwen3.6-35b-a3b` in OpenCode.

## Prompt reuse

KV prefix reuse is on by default. Send the complete message history with
every request. When a request continues a retained conversation exactly, the
server reuses the verified KV prefix and reports the number of reused tokens in:

```text
usage.prompt_tokens_details.cached_tokens
```

`--prompt-cache-mode` selects how many prefixes are retained:

| Mode | Retains |
| --- | --- |
| `multi-prefix` (default) | Qwen 3.6: the end of every finished turn, across conversations, within `--prompt-cache-bytes`. Every other model, and any runner that cannot snapshot its state: as `single-prefix`. |
| `single-prefix` | The last finished turn. A different or incompatible history replaces it. |
| `off` | Nothing; every request prefills its whole prompt. |

Deep (`qwen3.8-27b`) is the exception: with MTP draft decoding on, every request
prefills its whole prompt, and `cached_tokens` is 0.

With `multi-prefix`, Qwen 3.6 keeps a copy of its recurrent state (about
61 MiB) at the end of each finished turn. While one conversation continues,
its attention KV stays in the runner and nothing else is copied. Before a
request writes over KV that a retained turn still needs (another
conversation, or an edited or regenerated earlier message), the conversation's
KV up to its latest retained turn is copied out, even when only its end will
be overwritten: 20 KiB per token, 640 MiB for a 32K-token conversation. A later request
that continues any retained turn restores it and prefills only what it adds.
`--prompt-cache-bytes <bytes|auto>` caps these copies; the least recently used
are released first, and a copy that does not fit is skipped. `auto` is the
smaller of 4 GiB and an eighth of physical memory on Macs with at least
24 GiB, 1 GiB from 16 GiB, and 0 below. `0` serves exactly as
`single-prefix`. The copies are released when the model is unloaded or
swapped; cancellation and errors release only the conversation that was in
the runner. When macOS reports memory pressure (warning or critical), the
server keeps no new copies, and at the first moment no request is running it
releases every copy except the conversation in the runner. Copies are kept
again once pressure returns to normal. Some Macs swap before they report
pressure, so swap is watched too. If swap grows by more than 256 MiB after
the copies last grew, the server releases copies back to their size before
that growth, and keeps no new ones until swap has grown by less than 16 MiB
over a minute.

A restored prefix continues from exactly the state the turn ended in, so a
request gets the same output it would get from `single-prefix` had no other
request come between its turns. Neither mode is byte-identical to prefilling
the whole history from scratch: a continued turn's new tokens are prefilled
as one short chunk, and earlier replies stay as generated rather than
re-rendered.

## Tool calls

The server can return OpenAI-style function calls, but it cannot authorize or
execute them. The client runs the tool loop:

1. Send function schemas in `tools`.
2. When `finish_reason` is `"tool_calls"`, inspect each function name and JSON
   argument object. Apply the client's normal permission checks before running
   the function.
3. Append the assistant message, including its unchanged `tool_calls` and any
   `reasoning_content`.
4. Append each result as a `role: "tool"` message. Its `tool_call_id` must
   match the call it resolves.
5. Send the complete history and tool schemas again.

The server accepts only function tools. Omit `tool_choice` or set it to `auto`
to allow calls. Set it to `none` to disable them. The server does not support
`required`, named tool selection, or `parallel_tool_calls: false`.

## Library mode

With `--library`, the server scans for completed installs and serves the
product models among them through the [engine](ENGINE.md) (`EngineCatalog` and
`EngineHost`).

- A bare `--library` scans the default roots: the `Mference.libraryRoot` user
  default (or `MFERENCE_LIBRARY_ROOT`) if set, `~/llm-models`, the checkout's
  `scratch/`, and `~/Library/Application Support/Mference`. `--library <dir>`
  scans that root instead, and repeats; add a bare `--library` to keep the
  defaults too. Each root contributes itself and its immediate subdirectories.
- `.gturbo` installs are detected by their own `manifest.json` and install
  receipt, never by directory name. `.gguf` files are found in each root and in
  `~/llm-models/gguf`. Partial installs and installs locked by a running
  `MferenceRepack` are skipped, and every skip is logged once on stderr with
  its reason.
- `--catalog product` (the default) lists only the product models, under their
  stable identifiers: `qwen3.5-4b` (Quick) and `qwen3.8-27b` (Deep) through
  llama.cpp, and `qwen3.6-35b-a3b`, `minicpm5-2b-int4g64`,
  `maple-preview-2bit-mlx` and `inkling-2-mini-ud-q4_k_xl` (on Macs with
  32 GB or more) through the native runtime. A request for any other
  runnable install gets `400 model_not_runnable`. `--catalog all` is a
  developer switch that lists every runnable install; extra installs of one
  family get an `@<directory basename>` suffix.
- `MferenceServer --library --list-models` prints what would be served, with
  engine, size and path, without loading anything or binding a port.
- `--model-id` is refused in library mode. The library is fixed at startup.

A request that names a model other than the resident one swaps it in process.
The running and queued generations finish first, the resident model is
released, and only then does the new one load, so peak memory is one model.
The request waits for the load rather than getting `503`. The first token
after a switch therefore takes seconds to tens of seconds, or longer when the
expert pool has to be hashed. While a model loads, `/health` reports
`{"status":"loading","model":"…"}`. A failed load leaves nothing resident, and
the next request retries.

In library mode, prompt rendering and the context check run in the request's
turn, because the tokenizer belongs to the model being swapped in. A request
that waited behind another streaming generation therefore reports those
failures in-band (see [Errors](#errors)), and one that did not wait still gets
its real status.

## Errors

The prompt is rendered and checked against the context window before any
response is written, so an overlong prompt, an unknown model, an unsupported
parameter, an oversized body, or a full queue comes back as a JSON error
envelope with a real status code — `400`, `404`, `413`, `415`, or `429` — and
no stream is started. Asking for `"stream": true` does not change this.

A failure raised after that point cannot change the status, because a
streaming request already has `200` and the SSE head on the wire. It is
reported in-band instead: one frame carrying an `error` object, then
`data: [DONE]`, then a normal end of the chunked body.

```text
data: {"error":{"message":"generation failed","type":"server_error","code":"internal_error"}}

data: [DONE]

```

The frame carries the same envelope the blocking path would have returned, so
a failure that can only surface once generation is under way still names its
cause. A reused KV prefix does not reach it: the prompt cache is matched only
after the head is committed, and a retained prefix plus the new turn can be
longer than the prompt that was checked. When it no longer fits the context
window, the server drops the cached prefix and runs the checked prompt, as on
a cache miss.

Treat any frame with an `error` key as fatal for that request; no
`finish_reason` chunk precedes it. The stream is never terminated by dropping
the connection, so a client that sees an aborted transport (`TypeError:
terminated` under undici, for example) should look for a dead server process
or its own timeout rather than a generation error.

## Server log

The server writes one line per request to stderr:

```text
[2026-07-31T17:03:10Z] request chatcmpl-f6a02587… started streaming=true
[2026-07-31T17:03:12Z] request chatcmpl-f6a02587… completed in 2.4s prompt=812 cached=768 completion=96 finish=stop
```

The start line is written before the model runs, so a long prefill — which
emits nothing for minutes — is distinguishable from a wedged server. The
completion line reports how much of the prompt the KV prefix supplied in
`cached`, matching `usage.prompt_tokens_details.cached_tokens`. The `prefill`
field reports computed prefill time, including any snapshot capture; request
latency also includes prefix matching and snapshot restoration.

A failed request logs the status it would have carried, whether or not a
stream had already committed `200`, along with the underlying error — which is
more detail than the response carries, since responses deliberately do not
leak runtime internals:

```text
[2026-07-31T17:09:32Z] request chatcmpl-b36dd1ed… failed status=400 streaming=true error=context_length_exceeded: prompt exceeds the configured context
```

## Supported API

Endpoints:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

Each model in `GET /v1/models` carries `max_model_len`, vLLM's field: the
context window for prompt plus completion that the model runs with. That is
the numeric `--max-context`, or with `--max-context max` the model family's
native context. In library mode it comes from the engine catalog, so listing
loads nothing and answers during a model load.

Each model also carries `engine`: what runs it, `mference` (a `.gturbo`
install on Mference's own runtime) or `llama.cpp` (a GGUF file). It is
metadata for a model picker and changes nothing else about the API.

Chat Completions supports JSON and Server-Sent Events responses. Set
`"stream": true` for streaming. Set
`"stream_options": {"include_usage": true}` to receive a final usage chunk.

For ChatML (including Qwen) and MiniCPM, usage also includes
`completion_tokens_details.reasoning_tokens` and the Mference extension
`visible_tokens`. These count generated payload tokens in each channel,
including whitespace and buffered byte tokens, not re-tokenized rendered text.
They exclude channel markers, tool payloads and EOS; their sum need not equal
`completion_tokens`. Visible counts are omitted when a string stop trims an
answer inside a token. Unsupported dialects omit the details object rather
than reporting invented zeros. Reasoning token accounting does not imply that
every checkpoint streams its hidden reasoning text.

Every completion also carries a llama.cpp-style `timings` object, unasked: at
the top level of a JSON response, and on the last chunk of a stream (the usage
chunk when one was requested, otherwise the `finish_reason` chunk). It holds
`prompt_n` (computed prompt tokens), `cache_n` (tokens the KV prefix
supplied), `prompt_ms` and `prompt_per_second` for prefill, and `predicted_n`,
`predicted_ms` and `predicted_per_second` for decode.

Requests may contain system, developer, user, assistant, and tool messages.
Guidance must precede the conversation, and consecutive messages of the same
guidance role are merged into one block separated by a blank line. A
`developer` message is treated as a `system` message — it merges with adjacent
system guidance instead of rendering as its own block.
Exception: base Qwen 3.8 requests with an explicit `reasoning_effort` use the
source template and reject `developer`; use leading `system` guidance instead.
Supported options include `temperature`, `top_p`, `top_k`, `min_p`,
`presence_penalty`, `frequency_penalty`, `repetition_penalty` (alias
`repeat_penalty`), `repeat_last_n`, `seed`, `stop`, `max_tokens`,
`max_completion_tokens`, and function-tool fields. Conflicting repetition
aliases are rejected.

When an install carries the checkpoint's `generation_config.json`, its sampler
values replace these defaults; otherwise the defaults below (temperature `0.8`,
Top-K `40`) apply. Keys the checkpoint leaves out are off (no Top-K, Top-P, or
Min-P, and neutral penalties), and Top-P below 1 without Top-K uses a 64-token
window, the widest the fused sampler handles (full-vocabulary nucleus is not
implemented).

Without a checkpoint `generation_config.json`, all families default to temperature `0.8`, Top-K `40`, Top-P `0.95`, Min-P
`0.05`, repetition penalty `1.0`, presence/frequency penalties `0.0`, and a
64-token penalty window. These match llama.cpp's built-in sampling defaults,
before any GGUF metadata or application overrides. Explicit request values
win. Min-P accepts `0...1`; zero disables it. Presence and frequency each
accept finite values in `-2...2`. `repeat_last_n: 0` disables all history
penalties; `-1` uses the complete effective history. The window includes
cached and newly prefetched prompt tokens plus generated tokens.

The chain is penalties → Top-K → normalized Top-P → Min-P → temperature.
Penalties act on post-softcap logits: positive seen-token logits are divided
by repetition penalty, nonpositive logits multiplied; then subtract
`count * frequency_penalty + presence_penalty`. Counts are restricted to the
penalty window. Nonzero penalties with an enabled window require the logits
path even at temperature zero.
Top-K accepts `0` (off) or `1...256`; with positive temperature, disabling
Top-K requires `top_p: 1`. Full-vocabulary Min-P remains supported.

Qwen 3.8 accepts `reasoning_effort` values `xhigh`, `medium`, `low`, and
`none`. An explicit value uses the source template, retains reasoning in a
separate `reasoning_content` response/history field, and allows only exact
rendered-prefix cache reuse. Preserve that field when replaying history.
Omitted effort retains Qwen 3.8's legacy behavior, which does not stream
hidden reasoning.

The GGUF models in library mode take their thinking controls from the chat
template in the GGUF file:

- Quick (`qwen3.5-4b`) reads only `enable_thinking`. With `reasoning_effort` and
  `chat_template_kwargs.enable_thinking` omitted, its template default applies
  (no thinking). `none` (or `enable_thinking: false`) renders with
  `enable_thinking=false`; `low`, `medium` and `xhigh` (or
  `enable_thinking: true`) render with `enable_thinking=true`. `developer`
  messages are accepted and treated as `system` guidance.
- Deep (`qwen3.8-27b`) carries Qwen 3.8's source template, so the Qwen 3.8 rules
  above apply: it reads `reasoning_effort` (`xhigh`, `medium`, `low`, `none`;
  omitted is `xhigh` with thinking on), and a request with an explicit effort
  rejects `developer` messages. Deep is served with MTP draft decoding
  (llama.cpp's `draft-mtp`). Its greedy output is equivalent to decoding
  without MTP but not guaranteed byte-identical, since verifying drafted
  tokens in a batch differs numerically from single-token decode (8 of 10
  server outputs matched in validation; see [the engine](ENGINE.md)). While
  MTP is on, Deep reuses no prompt prefix (see Prompt reuse).

Qwen 3.6 opts into its source thinking template with
`chat_template_kwargs: {"enable_thinking": true}` or an explicit
`reasoning_effort`. For this checkpoint, `none` disables thinking and the other
accepted effort values enable it; omitted controls retain non-thinking
behavior. Explicit `reasoning_effort` takes precedence over `enable_thinking`.
Preserve `reasoning_content` in replayed assistant turns: Qwen 3.6 can continue
the cached generated turn with a verified source-template suffix, including
tool-result rounds. `preserve_thinking` is accepted for compatibility; source
template history always preserves reasoning. Thinking requests without an
explicit completion cap default to 32,768 tokens, subject to available context.
Other models reject explicit thinking controls. Sampling and MTP defaults
remain unchanged.

For Qwen 3.6's recommended general-thinking sampling profile, send these
explicit options along with `model` and `messages`:

```json
{
  "chat_template_kwargs": {"enable_thinking": true},
  "temperature": 1.0,
  "top_p": 0.95,
  "top_k": 20,
  "min_p": 0.0,
  "presence_penalty": 1.5,
  "repetition_penalty": 1.0
}
```

These options change sampling policy only, not thinking templates, reasoning
history, tool parsing, or prompt-cache continuation. Presence penalty does
not guarantee that every semantic or textual repetition loop is eliminated.

The server supports one model and one choice. It does not support the Responses
API, legacy Completions, embeddings, multimodal input, structured output,
batching, log probabilities, or remote model switching.

Maple's chat template opens a live `<think>` reasoning block at the start of
every completion. The server suppresses the reasoning text from the response,
but those tokens still count as completion tokens, so give Maple requests a
generous allowance — around 2048 `max_completion_tokens` (when neither cap is
set the server uses 4096) — or the reasoning budget swallows the visible
answer and the request finishes with `finish_reason` `"length"`.

`--max-context` takes any length up to the model's native context (default
16K): 262,144 tokens for Qwen 3.6 and Qwen 3.8; 131,072 for MiniCPM5; 128,000
for Maple. `--max-context max` gives every model its own native context, which
suits a library of different models; families that do not grow their KV (all
but Qwen 3.6) then reserve that whole context when they load. A model whose native
context is shorter refuses to load: with `--model` the server exits at
startup, and in library mode the request that asked for it gets HTTP 400
`context_exceeds_model`. Inkling 2 Mini (native 262,144) is the exception on
a Mac under 48 GB, where its expert residency plan cannot hold every layer:
there any larger value, `max` included, runs it at 16,384 tokens (the tested
maximum on 32 GB), and
`/v1/models` lists that.

Qwen 3.6 does not reserve its full-attention KV for
`--max-context` up front. It starts at 16,384 tokens (a context of 16,384 or
less is reserved whole); a prompt that does not fit grows it to the prompt
plus 16,384 tokens, room for the answer, and an answer that outgrows that
adds 8,192 tokens at a time, up to `--max-context`. Each layer reserves
address space for the whole context and grows its Metal buffer over the same
pages, so rows are never copied and the KV is never held twice; a new
conversation starts again from 16,384 and returns the pages. Metal charges a
KV buffer in full once it is bound, so a whole reservation costs memory a
short chat never uses. `--kv-reserve` reserves the whole context at load
instead; the other families always do. Pages the GPU writes into a growing
KV count as wired system memory rather than in the server process's own
footprint, so Activity Monitor shows the process smaller than its KV.

Maple uses native BF16
KV with layer-major chunked prefill; existing families use FP16 KV. On an 8 GB Mac,
run one model process at a time and watch memory pressure.
