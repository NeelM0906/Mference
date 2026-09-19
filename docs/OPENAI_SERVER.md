# Local OpenAI-compatible server

`MferenceServer` exposes a local Chat Completions API for one installed
supported model. It binds to `127.0.0.1` by default, or to the machine's exact
Tailscale IPv4 address with `--bind tailnet`. It has no application-level
authentication or TLS; do not expose it through a wildcard interface, proxy,
or tunnel.

This document is the API: one model per process, named by `--model`. If you want
the **user interface** — a browser, a model picker, chats that persist — start
at [The Mference UI (Open WebUI)](OPEN_WEBUI.md) and run `./mference-ui.sh`,
which does everything below for you.

The `--library` mode that UI runs on serves every installed model from this one
process, listing them all in `/v1/models` and swapping the resident model in
place when a request names a different one. It changes nothing about the mode
described here, which is what runs whenever `--library` is absent. The UI guide
covers library mode, its model identifiers, and the cost of a swap.

## Start the server

First, install the model with `MferenceRepack` or
`./mference-ui.sh install <family>`. Then check that no other Mference model
process is running:

```bash
pgrep -fl 'MferenceServer|MferenceCLI|MferencePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'
```

If the command prints a match, do not start the server.

```bash
swift build -c release --product MferenceServer
.build/release/MferenceServer \
  --model scratch/gemma4.gturbo \
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
  --model scratch/gemma4.gturbo \
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
    "model": "gemma-4-26b-a4b-it",
    "messages": [{"role": "user", "content": "Reply with exactly READY."}],
    "temperature": 0,
    "max_completion_tokens": 16
  }'
```

By default, the server runs one generation and queues up to four requests. Use
`--queue-limit` to change the queue size. Press Control-C to stop the server.

## Connect a client

The base URL is `http://127.0.0.1:8080/v1`. Some client libraries require an
API key, but the server ignores it.

Python:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
response = client.chat.completions.create(
    model="gemma-4-26b-a4b-it",
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
        "gemma-4-26b-a4b-it": {
          "name": "Gemma 4 26B-A4B IT"
        }
      }
    }
  }
}
```

Select `mference/gemma-4-26b-a4b-it` in OpenCode.

## Prompt reuse

Single-prefix KV reuse is on by default. Send the complete message history with
every request. When a request continues the retained conversation exactly, the
server reuses the verified KV prefix and reports the number of reused tokens in:

```text
usage.prompt_tokens_details.cached_tokens
```

The server retains one prefix. A different or incompatible history replaces
it. Use `--prompt-cache-mode off` to disable reuse.

## Tool calls

The server can return OpenAI-style function calls, but it cannot authorize or
execute them. The client runs the tool loop:

1. Send function schemas in `tools`.
2. When `finish_reason` is `"tool_calls"`, inspect each function name and JSON
   argument object. Apply the client's normal permission checks before running
   the function.
3. Append the assistant message, including its unchanged `tool_calls`.
4. Append each result as a `role: "tool"` message. Its `tool_call_id` must
   match the call it resolves.
5. Send the complete history and tool schemas again.

The server accepts only function tools. Omit `tool_choice` or set it to `auto`
to allow calls. Set it to `none` to disable them. The server does not support
`required`, named tool selection, or `parallel_tool_calls: false`.

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
cause. Reusing a KV prefix is the case that reaches it: whether the retained
prefix plus the new turn fits the context window is known only after the
prompt cache has been matched, which happens after the head is committed.

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
`cached`, matching `usage.prompt_tokens_details.cached_tokens`.

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

Chat Completions supports JSON and Server-Sent Events responses. Set
`"stream": true` for streaming. Set
`"stream_options": {"include_usage": true}` to receive a final usage chunk.

For ChatML (including base/Swift Qwen), GLM and MiniCPM, usage also includes
`completion_tokens_details.reasoning_tokens` and the Mference extension
`visible_tokens`. These count generated payload tokens in each channel,
including whitespace and buffered byte tokens, not re-tokenized rendered text.
They exclude channel markers, tool payloads and EOS; their sum need not equal
`completion_tokens`. Visible counts are omitted when a string stop trims an
answer inside a token. Unsupported dialects omit the details object rather
than reporting invented zeros. Reasoning token accounting does not imply that
every checkpoint streams its hidden reasoning text.

Requests may contain system, developer, user, assistant, and tool messages.
Guidance must precede the conversation, and consecutive messages of the same
guidance role are merged into one block separated by a blank line. Only
Gemma's chat template has a distinct `developer` role; with any other family
loaded a `developer` message is treated as a `system` message — it merges
with adjacent system guidance instead of rendering as its own block.
Exception: Swift-Qwen, and base Qwen 3.8 requests with an explicit
`reasoning_effort`, use the source template and reject `developer`; use leading
`system` guidance instead.
Supported options include `temperature`, `top_p`, `top_k`, `min_p`,
`presence_penalty`, `frequency_penalty`, `repetition_penalty` (alias
`repeat_penalty`), `repeat_last_n`, `seed`, `stop`, `max_tokens`,
`max_completion_tokens`, and function-tool fields. Conflicting repetition
aliases are rejected.

All families default to temperature `0.8`, Top-K `40`, Top-P `0.95`, Min-P
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
path even at temperature zero. See [sampling compatibility](LLAMA_SAMPLING.md).
Top-K accepts `0` (off) or `1...256`; with positive temperature, disabling
Top-K requires `top_p: 1`. Full-vocabulary Min-P remains supported.

Base and Swift Qwen 3.8 accept `reasoning_effort` values `xhigh`, `medium`, `low`,
and `none`. An explicit value uses the source template, retains reasoning in a
separate `reasoning_content` response/history field, and allows only exact
rendered-prefix cache reuse. Preserve that field when replaying history.
Omitted effort retains Swift's source-default xhigh and base Qwen's legacy
behavior; the latter does not stream hidden reasoning. Do not treat those two
omitted-effort policies as a matched fine-tune comparison.

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
Other models reject these thinking controls. Sampling and MTP defaults remain
unchanged.

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

Context length can be 4K, 8K, 16K, 32K, 64K, or 128000 tokens; the default is
16K. Maple supports 128000 tokens in the server and uses native BF16 KV with
layer-major chunked prefill; existing families use FP16 KV. On an 8 GB Mac,
run one model process at a time and watch memory pressure.
