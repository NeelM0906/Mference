# Mference

Swift and Metal inference for pinned checkpoints on Apple Silicon: Gemma 4
26B-A4B, Qwen 3.6 35B-A3B, DeepSeek-V4-Flash 284B-A13B, Inkling-Small
276B-A12B, Maple Preview, Qwen 3.8 27B, Qwen3.8-Flash-Next 180B-A3.5B, and
MiniCPM5-2B. The shared core and KV cache stay resident; routed experts stream
from SSD per token. Qwen 3.8 27B and MiniCPM5-2B are dense and fully
resident.

## Scope

Make only the changes the user asks for, and keep them surgical. Do not start
optimization work, refactors, or new model ports on your own initiative. The
model-run safety rules below always apply, whatever the task.

## Layout and commands

`Sources/Mference/` is the runtime and kernels; `Sources/MferenceRepack/`,
`Sources/MferenceCLI/`, and `Sources/MferenceServer/` contain the installer,
CLI, and loopback server. The four products are `Mference`, `MferenceRepack`,
`MferenceCLI`, and `MferenceServer`. `Tests/` contains focused public tests;
`docs/` contains design, benchmark, and experiment notes.

```bash
swift build -c release
./mference-ui.sh
./mference-ui.sh install qwen36
swift run -c release MferenceRepack --model qwen36 --output scratch/qwen36.gturbo
swift run -c release MferenceRepack --model qwen36 --output scratch/qwen36.gturbo --resume
swift run -c release MferenceCLI \
  --model scratch/qwen36.gturbo \
  --prompt "The capital of France is" \
  --max-new 64
```

The installer streams each pinned checkpoint without staging the full source.
Set `HF_TOKEN` only if requested. Source reads range from ~5 GB (MiniCPM5) to
~360 GB (Flash-Next), producing installs from ~1.4 GB to ~175 GB; check disk
before installing, and read
[docs/DEEPSEEK_V4_FLASH.md](docs/DEEPSEEK_V4_FLASH.md) or
[docs/INKLING_SMALL.md](docs/INKLING_SMALL.md) before touching those two.
Cancellation preserves verified completed ranges; continue with `--resume` or
remove them with `--discard-partial`.

## Models and the library

Install directories take the family's own label (`gemma4.gturbo`,
`qwen36.gturbo`, `minicpm5.gturbo`, and so on for the labels
`MferenceRepack --help` lists), but detection goes by each directory's own
manifest, not its name. `MferenceServer --library`
scans the library roots — the `Mference.libraryRoot` default if set, the
package checkout's `scratch/`, and `~/Library/Application Support/Mference` —
and serves every model it finds there to the UI. The CLI and a non-library
server take an explicit `--model` path; that selection also persists via
`defaults write Mference model qwen36` (or `MFERENCE_MODEL` in the
environment). `MferenceCLI --verify trusted-receipt` skips the first-touch
SHA-256 of the expert pool in favor of the install receipt's size checks; the
strict `full-sha256` mode is the default.

## Local server

Follow the [server guide](docs/OPENAI_SERVER.md) for launch commands, health
checks, client setup, prompt reuse, tool loops, and supported API behavior.
Its opt-in `--library` mode serves every installed model from that one process,
swapping the resident model in place; see the
[Open WebUI guide](docs/OPEN_WEBUI.md).
Apply the model-process checks below first; never start a second model process
or terminate an existing one.

Keep the server on its default `127.0.0.1` binding unless the user explicitly
asks for `--bind tailnet` and intends the Tailnet ACL to be the access
boundary. It has no application-level authentication or TLS, so never bind it
to a wildcard interface or expose it through a proxy or tunnel. A tool call
from the local model never bypasses the client's normal permission policy. Keep
the execution session alive while the server is needed, and stop only a server
you launched.

## Test rules

Before a model run, require macOS 15+, Swift 6.1+, enough disk, acceptable
`memory_pressure -Q`, a completed install of the model the run needs, and no
process from `pgrep -fl 'MferenceServer|MferenceCLI|MferencePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'`.
If a check fails, inform the user and stop; do not terminate apps or delete or
reinstall a model.

Run package tests through `Scripts/test.sh`. Real-model regression suites are
env-gated (for example `MFERENCE_INKLING_GTURBO`) and skip without the gate.
Run only one server, CLI, or model-using test at a time.

For performance results, build release once and follow the [community
benchmark guide](docs/COMMUNITY_BENCHMARKS.md) exactly. Do not enable
experimental controls or profiling. Measured baselines are in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md), and each family's own page under
[docs/families/](docs/families/) carries its bring-up numbers.

Do not download a full checkpoint, duplicate a `.gturbo` model, create a
worktree, or purge caches just to run tests.

Report the commit, hardware and RAM, macOS, Swift version, exact command, exit
code, complete timing footer or error, and every protocol deviation. Treat
results as measurements, not performance ceilings.

## UI

Open WebUI, started by `./mference-ui.sh`, is the only Mference UI. The script
launches `MferenceServer` in library mode and points Open WebUI at it;
`./mference-ui.sh install <family>` runs an install. The server is the only
model owner behind the UI — never start a second model process alongside it.
Generation and runtime controls, and their defaults, are documented in
[The Mference UI](docs/OPEN_WEBUI.md) and [Runtime controls](docs/RUNTIME_CONTROLS.md).
