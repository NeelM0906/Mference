# Source release: installation and readiness

Mference is a Swift/Metal source release for Apple Silicon, with Open WebUI as
its only UI. It is not a notarized application, a prebuilt binary distribution,
or a model-weight bundle. Model licenses and download sizes are separate from
the runtime. The default server and UI are loopback-only and must not be
exposed through a tunnel, wildcard binding or public proxy.

## First-use path

1. Install Xcode 16.3+ or matching Command Line Tools (Swift 6.1+), on an
   Apple Silicon Mac running macOS 15 or newer. For the UI, install `uv` too.
2. Clone the repository or extract the tagged source archive and enter it.
3. Run `./mference-ui.sh doctor`. This does not build, download, start a model,
   open the UI database, or change settings. It checks platform, toolchain,
   port availability and UI prerequisites, and reports free disk/memory.
4. Run `swift build -c release`.
5. Run `./mference-ui.sh install gemma4` for the established first-use example.
   This downloads/repackages the pinned checkpoint, about 14 GB. Larger models
   need substantially more storage; a model download is not an instant step.
6. Run `./mference-ui.sh`. First UI launch installs pinned Open WebUI 0.11.3
   if absent; it never silently replaces a different installed version.
7. Select the installed model in the browser and send a short request. Initial
   loading and integrity verification take longer than subsequent requests.

Source updates trigger an incremental server build on launch. Ctrl-C stops
only the two services the launcher started. A process already using a model
or either requested port causes a clear refusal, not termination of that process.

For CLI-only use, follow [the README](../README.md#try-it); no UI dependency is
needed. For an existing OpenAI client, see [the server guide](OPENAI_SERVER.md).

## Recovery

- Interrupted install: `./mference-ui.sh install gemma4 --resume`. Verified
  ranges are reused; no full checkpoint staging or second model copy is needed.
- Occupied ports: select both explicitly, for example
  `./mference-ui.sh --server-port 8081 --webui-port 3001`.
- Different Xcode toolchain artifacts: use a separate build directory, e.g.
  `./mference-ui.sh --build-path /tmp/mference-build`. For installation the
  option precedes `install`: `./mference-ui.sh --build-path /tmp/mference-build install gemma4`.
  Do not purge unrelated caches as a first troubleshooting step.
- Isolated UI verification: `./mference-ui.sh --data-dir /tmp/mference-ui-test`.
  This creates a separate UI database and secret without touching existing chats.
- Empty picker: stop the launcher, install a model, then restart. Library
  discovery reads each install's manifest, not the directory name.
- Wrong Open WebUI version: the launcher reports the mismatch before starting
  its model server. Resolve the tool environment explicitly; the launcher
  does not overwrite another installation.

## Release gates

A release recommendation requires all of the following evidence, recorded
against a commit rather than inferred from a merge or fluent output:

- Release build and serial package tests on macOS 15 / Swift 6.1 and macOS 26.
- Launcher/adapter unit checks, source archive completeness, and Markdown links.
- A real installed-model smoke test and browser/server streaming, cancellation,
  history reuse and model-switch recovery with isolated UI data.
- Separate opt-in numerical/state tests for changes to a checkpoint's kernels.
- Performance claims made only from the frozen community protocol; additional
  comparison protocols are identified separately and include failures.

[The qualification matrix](PREFILL_QUALIFICATION.md) tracks per-family coverage.
Unqualified combinations and experimental checkpoints are not promoted merely
to meet a release date. Lower expert-cache settings on a 256 GiB Mac do not
establish support or performance on a physically smaller-memory Mac.

This checklist is a release requirement, not a statement that every gate or
every model/hardware combination has already passed. Dated validation records
and release notes carry the actual results.
