# Open WebUI as an Mference frontend

[Open WebUI](https://github.com/open-webui/open-webui) is a browser chat UI that
talks to any OpenAI-compatible backend. Pointed at
[`MferenceServer`](OPENAI_SERVER.md) in library mode, it gives every installed
model a picker entry and streams from whichever one you choose — with one model
resident at a time and one Mference process, which is the project's hard rule.

Open WebUI proxies requests from its own backend, so nothing here depends on
CORS. Both processes stay on `127.0.0.1`.

## Install

```bash
uv tool install --python 3.11 open-webui
```

That puts `open-webui` in `~/.local/bin`. Verify it runs:

```bash
open-webui serve --help
```

## Launch

```bash
swift build -c release --product MferenceServer
Scripts/openwebui.sh
```

The launcher runs the model-process check from
[AGENTS.md](../AGENTS.md) and refuses to start if anything from
`MferenceServer`, `MferenceMac`, `MferenceDecodeService`, `MferenceCLI`, the
package test helper, or `mlx-lm` is already running. It never terminates a
process it did not start. Then it starts `MferenceServer` on
`127.0.0.1:8080` in library mode, waits for `/health`, prints the model list,
and starts Open WebUI on `http://127.0.0.1:3000`. Control-C stops both.

Useful options:

```bash
Scripts/openwebui.sh --dry-run                       # print the plan, start nothing
Scripts/openwebui.sh --library scratch               # scan one root instead of the defaults
Scripts/openwebui.sh --model scratch/qwen36.gturbo   # preload instead of loading lazily
Scripts/openwebui.sh --max-context 32768             # applies to every model
Scripts/openwebui.sh --server-port 8081 --webui-port 3001
```

To run the two halves by hand, start the server yourself and give Open WebUI the
same environment the launcher sets (below):

```bash
.build/release/MferenceServer --library --port 8080
```

## Library mode

`--library` is opt-in. Without it `MferenceServer` behaves exactly as
[the server guide](OPENAI_SERVER.md) describes: one model per process, named by
`--model`.

With `--library`, the server scans for completed installs and serves all of
them:

- `--library` with no value scans the roots the Mac app scans: the
  `Mference.libraryRoot` user default (or the `MFERENCE_LIBRARY_ROOT`
  environment variable) if set, the package checkout's `scratch/`, and
  `~/Library/Application Support/Mference`.
- `--library <dir>` scans that root instead, and repeats. Combine explicit roots
  with the defaults by passing a bare `--library` as well.
- Each root contributes itself and its immediate subdirectories, so a root may
  be a directory of installs or a single install.

Detection goes by each directory's own `manifest.json`, never by its name, using
the same probe the Mac app uses: manifest, family, architecture baseline,
`packed_experts/layout.json`, and an install receipt bound to the manifest. A
directory with no manifest is ignored. A staging directory (`<name>.partial`) or
one whose sibling `<name>.gturbo.install.lock` is held by a running install is
skipped. Every skip is reported once on stderr, with the reason:

```text
[2026-09-10T05:03:39Z] library skipped /path/qwen38flashnext.gturbo: not runnable (qwen3.8-flash-next-int4g64): no runner for qwen38flashnext; missing axes hyperConnectionsLowRank, attentionIndexer, pleNgramEmbedding
[2026-09-10T05:03:39Z] library ready models=gemma-4-26b-a4b-it,qwen3.6-35b-a3b
```

The server refuses to start when no root yields a completed install.

### Model identifiers

An install alone in its family is advertised under that family's identifier —
`gemma-4-26b-a4b-it`, `qwen3.6-35b-a3b`, `qwen3.8-27b-4bit`,
`deepseek-v4-flash-2bit-dq`, `inkling-small-4bit`, `maple-preview-2bit-mlx`.

When two installs share a family, **both** are suffixed with their directory
basename minus `.gturbo`, and neither keeps the bare identifier. So
`qwen36.gturbo` and `qwen36-ourquant.gturbo` under the same root become:

```text
qwen3.6-35b-a3b@qwen36
qwen3.6-35b-a3b@qwen36-ourquant
```

If two installs of one family also share a basename (the same directory name
under two roots), the second and later get a `#2`, `#3`, … suffix in path order.
Assignment depends only on the set of installs found, not on the order the roots
were given, so reordering `--library` does not rename anything.

`--model-id` is refused in library mode: one override cannot name several
models.

### Installs that are not listed

A family the repacker can install but no runner can execute yet is **skipped**,
not listed as unusable. Listing it would put a permanently failing entry in the
model picker. The skip line names it, and the moment its capability gate lifts
the install is listed with no code change and no configuration — library mode
lists whatever the runtime accepts. `qwen3.8-flash-next` is the current example.

### Switching models, and what it costs

Choosing a different model in Open WebUI's picker sends a normal
`POST /v1/chat/completions` naming it. The server swaps **in process**:

1. The generation that is running finishes, and so does everything already
   queued ahead of the request. Nothing is unloaded while a generation is in
   flight.
2. The resident `ServerModelSession` is released — Metal buffers, expert cache,
   and KV with it — *before* the replacement is loaded, so peak memory is one
   model, not two.
3. The requested model loads, and the request is then rendered and served by it.

The swapping request **blocks** until the model is ready rather than returning
`503` with `Retry-After`. Loading runs first-touch SHA-256 verification over the
expert pool and takes tens of seconds to minutes, so the first token after a
switch is slow — sometimes very slow. Open WebUI tolerates the wait; a retry
protocol would be one the API does not describe. The swap is logged:

```text
[2026-09-10T05:04:11Z] swap started from=gemma-4-26b-a4b-it to=qwen3.6-35b-a3b
[2026-09-10T05:05:37Z] swap finished model=qwen3.6-35b-a3b in 86.4s
```

`/health` stays answerable throughout and reports the target:

```json
{"status":"loading","model":"qwen3.6-35b-a3b"}
```

Once a model is resident, `/health` reports `{"status":"ok","model":"…"}`. In
single-model mode `/health` is unchanged: `{"status":"ok"}`.

Switching back and forth costs a full reload each way. Keep a conversation on
one model when you can, and use `--model` to preload the one you start with.

### Error statuses in library mode

Resolving a model identifier needs no load, so an unknown model is still a
synchronous `404` with the same `model_not_found` envelope single-model mode
returns. Everything else — parameter validation, prompt rendering, the context
check — runs inside the request's turn, because the tokenizer and dialect belong
to the model being swapped in. That splits the remaining failures by whether the
request had to wait:

- **First in line.** Nothing is on the wire, so an unsupported parameter or an
  overlong prompt is `400` and a failed load is `500`, exactly as before.
- **Queued behind another generation, streaming.** `200` and the SSE head were
  committed when the request was queued, so the same envelope arrives in-band as
  one `error` frame followed by `[DONE]` — the mechanism
  [the server guide](OPENAI_SERVER.md#errors) already describes for post-commit
  failures.
- **Queued behind another generation, non-streaming.** Nothing was committed, so
  it still gets its real status.

A failed load leaves nothing resident; the next request retries from scratch.

## Environment the launcher sets

| Variable | Value | Why |
| --- | --- | --- |
| `OPENAI_API_BASE_URL` | `http://127.0.0.1:8080/v1` | The Mference server. |
| `OPENAI_API_KEY` | `local` | Required by the client; the server ignores it. |
| `ENABLE_OLLAMA_API` | `false` | No Ollama backend to probe. |
| `WEBUI_AUTH` | `false` | Single local user, no sign-in. |
| `ENABLE_TITLE_GENERATION` | `false` | Would fire an extra generation per chat. |
| `ENABLE_TAGS_GENERATION` | `false` | Would fire an extra generation per chat. |
| `ENABLE_FOLLOW_UP_GENERATION` | `false` | Would fire an extra generation per turn. |
| `ENABLE_AUTOCOMPLETE_GENERATION` | `false` | Would fire generations while typing. |
| `ENABLE_RETRIEVAL_QUERY_GENERATION` | `false` | Would fire an extra generation per turn. |
| `ENABLE_SEARCH_QUERY_GENERATION` | `false` | Would fire an extra generation per turn. |
| `DATA_DIR` | `~/Library/Application Support/Mference/open-webui` | Keeps its database out of the checkout. |

The server runs one generation at a time behind a short queue, so each of those
auxiliary generations would compete with the answer you are waiting for — and a
swap in between would reload a model to produce a chat title.

These are Open WebUI *defaults*, seeded into its persisted configuration. They
take effect on a fresh `DATA_DIR`; once a value has been changed in Open WebUI's
admin settings, the stored value wins and the environment variable no longer
does. Check them under **Admin Panel → Settings** if an unexpected generation
appears. The names are the ones the installed package reads:

```bash
grep -o 'ENABLE_[A-Z_]*GENERATION\|WEBUI_AUTH\|OPENAI_API_BASE_URL\|ENABLE_OLLAMA_API\|DATA_DIR' \
  ~/.local/share/uv/tools/open-webui/lib/python3.11/site-packages/open_webui/{env,config}.py |
  sort -u
```

## Security posture

Both processes bind `127.0.0.1`. Open WebUI's authentication is **disabled**, so
anyone who can reach its port is an administrator of it — never expose it
through a wildcard interface, a proxy, a tunnel, or a port forward.

`MferenceServer` has no application-level authentication or TLS either. If you
run it with `--bind tailnet` so other Tailnet devices can use the API, Open WebUI
still stays on loopback: the Tailnet ACL is the boundary for the API, and it is
not a boundary for an unauthenticated admin UI. The launcher never binds
anything but loopback.

A tool call the local model emits is not authorization to run anything; the
client runs the tool loop under its own permission policy.

## Known limitations

- **One generation at a time.** The server runs a single generation and queues a
  few more. Open a second browser tab and its request waits.
- **A swap is not free.** Every model change is an unload and a full load. See
  [Switching models](#switching-models-and-what-it-costs).
- **One context length for every model.** `--max-context` applies to whichever
  model is resident. Maple's 128000-token window needs `--max-context 128000`,
  which then applies to the others too.
- **Parameters the server rejects.** `n > 1`, `logprobs`, a non-zero
  `presence_penalty` or `frequency_penalty`, `parallel_tool_calls: false`, and
  any `tool_choice` other than `auto` or `none`. Open WebUI's defaults send none
  of these; they can appear from **Chat → Controls → Advanced Params** or from a
  model's own **Advanced Params** in **Workspace → Models**, so that is where to
  look if a request starts coming back `400`.
- **No multimodal input, embeddings, or structured output.** Open WebUI features
  that need them (image input, RAG embedding through this backend, JSON schema
  response formats) will not work against `MferenceServer`.
- **The library is fixed at startup.** Installing a model while the server runs
  does not add it; restart the launcher.

## License and branding

Open WebUI is distributed under its own modified BSD-3-Clause license, which
permits use and modification but requires that its "Open WebUI" name and
branding stay intact in deployments unless you qualify for the exemption or hold
a separate license from its maintainers — Mference neither bundles nor rebrands
it, and this document only describes pointing your own install at the local
server.
