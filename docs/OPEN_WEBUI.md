# The Mference UI (Open WebUI)

Mference's user interface is [Open WebUI](https://github.com/open-webui/open-webui)
in the browser, with [`MferenceServer`](OPENAI_SERVER.md) behind it in library
mode. One Mference process serves every model you have installed, the picker
switches between them, and exactly one model is resident at a time.

`./mference-ui.sh` starts both halves, and is also how models are installed and
listed. Both processes bind `127.0.0.1`; Open WebUI proxies from its own backend,
so nothing here depends on CORS.

## First launch

```bash
git clone <this repository> && cd Mference
swift build -c release
./mference-ui.sh install gemma4   # ~15 GB; see "Installing models" first
./mference-ui.sh                  # starts both halves, opens the browser
```

The fifth step is the browser: the launcher opens
`http://127.0.0.1:3000` once Open WebUI answers.

`./mference-ui.sh` with no arguments:

1. Runs the model-process check from [AGENTS.md](../AGENTS.md) and refuses to
   start if anything already owns a model. It never terminates a process it did
   not start.
2. Installs Open WebUI with `uv` if it is missing, pinned to the version this
   launcher expects. It never installs `uv` itself.
3. Builds `MferenceServer` if `.build/release/MferenceServer` is missing.
4. Starts `MferenceServer` on `127.0.0.1:8080` in library mode and waits for
   `/health`.
5. Starts Open WebUI on `127.0.0.1:3000`, waits for it to answer, and opens it.

Control-C stops both — and only the two it started. Starting with nothing
installed is fine: the server comes up, the picker is empty, and the launcher
says which command installs a model.

Options:

```bash
./mference-ui.sh --dry-run                       # print the plan, start nothing
./mference-ui.sh --library scratch               # scan one root instead of the defaults
./mference-ui.sh --model scratch/qwen36.gturbo   # preload instead of loading lazily
./mference-ui.sh --max-context 32768             # applies to every model
./mference-ui.sh --server-port 8081 --webui-port 3001
```

`--dry-run` works for every subcommand, before or after it, and prints each step
without running any of them.

To run the two halves by hand, start the server yourself and give Open WebUI the
[environment the launcher sets](#environment-the-launcher-sets):

```bash
.build/release/MferenceServer --library --port 8080
```

## Installing models

```bash
./mference-ui.sh install gemma4
./mference-ui.sh install qwen36 --resume   # extra arguments reach MferenceRepack
```

The install runs `MferenceRepack` into `scratch/<family>.gturbo`, which is one
of the roots library mode scans, so a model installed this way appears in the
picker the next time the UI starts. Supported families are `gemma4`, `qwen36`,
`qwen38`, `deepseekv4flash`, `inklingsmall`, `maple`, and `qwen38flashnext`;
the launcher reads that list out of `MferenceRepack`'s own help, so it cannot
drift.

Downloads range from ~15 GB (Gemma 4) to ~148 GB (Inkling-Small). Check disk
first, and read [docs/DEEPSEEK_V4_FLASH.md](DEEPSEEK_V4_FLASH.md) or
[docs/INKLING_SMALL.md](INKLING_SMALL.md) before installing either of those two.
A cancelled download continues with `--resume`.

`qwen38flashnext` installs and verifies but has no runner yet, so it is
deliberately **not** listed in the picker; see
[Installs that are not listed](#installs-that-are-not-listed).

## Listing what the picker will show

```bash
./mference-ui.sh models
```

It runs the server's own discovery — `MferenceServer --library --list-models` —
so there is one source of truth for what counts as an install, and prints:

```text
MODEL                 FAMILY  BYTES        PATH
gemma-4-26b-a4b-it    gemma4  14291921884  /Users/you/Mference/scratch/gemma4.gturbo
qwen3.6-35b-a3b       qwen36  19546491213  /Users/you/Mference/scratch/qwen36.gturbo
```

`BYTES` is the installed size the verified-install receipt records, or `-` when
the receipt carries no sizes. Nothing is loaded and no port is bound. With
nothing installed the output is the single line `no models installed`, and the
launcher adds the command that fixes that. Directories discovery declined —
partial, locked, or gated installs — are reported on stderr with the reason.

## The model picker, and what a swap costs

Choosing a different model in the picker sends a normal
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
expert pool and takes tens of seconds to minutes, so **the first token after a
switch is slow — sometimes very slow**. Open WebUI tolerates the wait; a retry
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

## Builtin tools are off for Mference models

Open WebUI 0.11 defaults every model to "native function calling" and attaches
its own builtin tool schemas — web search, code interpreter, memory, notes,
time, ask-user and more — to every chat request. `MferenceServer` does what an
OpenAI-compatible server must and renders those tools into the prompt. Measured
on this host with Gemma 4, the same one-line question cost 27 prompt tokens
sent to the server directly and 5,445 tokens sent through the UI, turning a
2-second answer into a 45-second prefill.

Open WebUI 0.11.3 has no environment variable for this; it is a per-model
capability. So the launcher runs
`Scripts/openwebui-configure-models.py` after Open WebUI is up, on every
launch: it signs in with the no-auth admin session, lists the models the
Mference server advertises, and registers each one with the `Builtin Tools`
capability unchecked. The step is idempotent, picks up newly installed models,
and is non-fatal — if it cannot reach Open WebUI it says so and chat still
works, just slowly. After the fix the same question costs 17 prompt tokens and
answers in 1.2 s.

To use Open WebUI's own web search or code interpreter with a model, re-enable
its `Builtin Tools` capability in **Admin > Models**; the launcher only sets
the capability when the model has no entry yet or the flag is not already off,
so it will not undo a deliberate change. If you turn authentication on, pass
the admin API token to the script with `OPEN_WEBUI_TOKEN` (or `--token`).

## Where your chats live

Open WebUI keeps its database, uploads, and settings under `DATA_DIR`, which the
launcher sets to:

```text
~/Library/Application Support/Mference/open-webui
```

That is deliberately outside the checkout, so chats survive a `git clean` and a
rebuild, and are not something a repository operation can delete. Back up that
directory to back up your conversations; delete it to start over with fresh
defaults. Nothing in it is sent anywhere — both processes are loopback-only.

## Library mode

`--library` is what the launcher passes. Without it `MferenceServer` behaves
exactly as [the server guide](OPENAI_SERVER.md) describes: one model per process,
named by `--model`.

With `--library`, the server scans for completed installs and serves all of
them:

- `--library` with no value scans the default roots: the `Mference.libraryRoot`
  user default (or the `MFERENCE_LIBRARY_ROOT` environment variable) if set, the
  package checkout's `scratch/`, and `~/Library/Application Support/Mference`.
- `--library <dir>` scans that root instead, and repeats. Combine explicit roots
  with the defaults by passing a bare `--library` as well.
- Each root contributes itself and its immediate subdirectories, so a root may
  be a directory of installs or a single install.

Detection goes by each directory's own `manifest.json`, never by its name:
manifest, family, architecture baseline, `packed_experts/layout.json`, and an
install receipt bound to the manifest. A directory with no manifest is ignored.
A staging directory (`<name>.partial`) or one whose sibling
`<name>.gturbo.install.lock` is held by a running install is skipped. Every skip
is reported once on stderr, with the reason:

```text
[2026-09-10T05:03:39Z] library skipped /path/qwen38flashnext.gturbo: not runnable (qwen3.8-flash-next-int4g64): no runner for qwen38flashnext; missing axes hyperConnectionsLowRank, attentionIndexer, pleNgramEmbedding
[2026-09-10T05:03:39Z] library ready models=gemma-4-26b-a4b-it,qwen3.6-35b-a3b
```

The library is fixed at startup, and an empty one is not an error: the server
starts, `/v1/models` is an empty list, and installing a model takes effect the
next time the launcher runs.

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
run it with `--bind tailnet` so other Tailnet devices can use the API, that
applies to the server only and Open WebUI stays on loopback: the Tailnet ACL is
the boundary for the API, and it is not a boundary for an unauthenticated admin
UI. The launcher never binds anything but loopback.

A tool call the local model emits is not authorization to run anything; the
client runs the tool loop under its own permission policy.

## Troubleshooting

**"another Mference model process is running".** The launcher found something
matching the AGENTS.md check — a server, a CLI run, an install, or a package
test suite — and refused rather than becoming a second model process. It never
kills anything. Wait for that process to finish, or stop it yourself, then
re-run. `./mference-ui.sh models` refuses for the same reason, because listing
still starts a short-lived `MferenceServer`.

**A port is already in use.** `MferenceServer` exits before it answers
`/health`, or Open WebUI exits before it answers. Move either half:
`./mference-ui.sh --server-port 8081 --webui-port 3001`. Open WebUI's stored
`OPENAI_API_BASE_URL` follows the flag only on a fresh `DATA_DIR`; on an
existing one, change the connection under **Admin Panel → Settings →
Connections**.

**`uv` is missing.** The launcher prints the two ways to install it —
`brew install uv`, or the installer line from astral.sh — and exits non-zero. It
will not pipe an installer into a shell on your behalf. Install `uv`, re-run,
and Open WebUI is installed automatically.

**Open WebUI version mismatch.** The launcher pins one version and prints a
one-line notice when a different one is installed. It does not reinstall or
downgrade. To match the pin:
`uv tool install --python 3.11 --force "open-webui==<pinned version>"`, with the
version from the notice or from `OPEN_WEBUI_VERSION` at the top of
`mference-ui.sh`.

**The picker is empty.** Nothing completed an install under the roots being
scanned. Run `./mference-ui.sh models` to see what discovery found and what it
skipped, then `./mference-ui.sh install gemma4`. A model installed while the
server is running does not appear until the launcher restarts.

**A request comes back `400`.** See
[Parameters the server rejects](#known-limitations); Open WebUI's own defaults
send none of them, so the source is usually **Chat → Controls → Advanced
Params**.

## Known limitations

- **One generation at a time.** The server runs a single generation and queues a
  few more. Open a second browser tab and its request waits.
- **A swap is not free.** Every model change is an unload and a full load. See
  [The model picker](#the-model-picker-and-what-a-swap-costs).
- **One context length for every model.** `--max-context` applies to whichever
  model is resident. Maple's 128000-token window needs `--max-context 128000`,
  which then applies to the others too.
- **The library is fixed at startup.** Installing a model while the server runs
  does not add it; restart the launcher.
- **Parameters the server rejects.** `n > 1`, `logprobs`, a non-zero
  `presence_penalty` or `frequency_penalty`, `parallel_tool_calls: false`, and
  any `tool_choice` other than `auto` or `none`. Open WebUI's defaults send none
  of these; they can appear from **Chat → Controls → Advanced Params** or from a
  model's own **Advanced Params** in **Workspace → Models**, so that is where to
  look if a request starts coming back `400`.
- **No multimodal input, embeddings, or structured output.** Open WebUI features
  that need them (image input, RAG embedding through this backend, JSON schema
  response formats) will not work against `MferenceServer`.

## License and branding

Open WebUI is distributed under its own modified BSD-3-Clause license, which
permits use and modification but requires that its "Open WebUI" name and
branding stay intact in deployments unless you qualify for the exemption or hold
a separate license from its maintainers — Mference neither bundles nor rebrands
it, and this document only describes pointing your own install at the local
server.
