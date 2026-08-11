# Mference

Swift and Metal inference for pinned mixture-of-experts checkpoints on Apple
Silicon: Gemma 4 26B-A4B, Qwen 3.6 35B-A3B, DeepSeek-V4-Flash 284B-A13B, and
Inkling-Small 276B-A12B. The shared core and KV cache stay resident; routed
experts stream from SSD per token.

## Scope

Make only the changes the user asks for, and keep them surgical. Do not start
optimization work, refactors, or new model ports on your own initiative. The
model-run safety rules below always apply, whatever the task.

## Layout and commands

`Sources/Mference/` is the runtime and kernels; `Sources/MferenceRepack/`,
`Sources/MferenceCLI/`, `Sources/MferenceServer/`, and `Sources/MferenceApp/`
contain the installer, CLI, loopback server, and Mac app.
`Sources/ChatTemplate/` is a standalone SwiftUI chat app (Core/UI/Mac) whose
UI components the Mac app shares. `Tests/` contains focused public tests;
`docs/` contains design, benchmark, and experiment notes.

```bash
swift build -c release
.build/release/MferenceMac
swift run -c release MferenceRepack --model qwen36 --output scratch/qwen36.gturbo
swift run -c release MferenceRepack --model qwen36 --output scratch/qwen36.gturbo --resume
swift run -c release MferenceCLI \
  --model scratch/qwen36.gturbo \
  --prompt "The capital of France is" \
  --max-new 64
```

The installer streams each pinned checkpoint without staging the full source.
Set `HF_TOKEN` only if requested. Downloads range from ~15 GB (Gemma 4) to
~148 GB (Inkling-Small); check disk before installing, and read
[docs/DEEPSEEK_V4_FLASH.md](docs/DEEPSEEK_V4_FLASH.md) or
[docs/INKLING_SMALL.md](docs/INKLING_SMALL.md) before touching those two.
Cancellation preserves verified completed ranges; continue with `--resume` or
remove them with `--discard-partial`.

## Models and the library

Install directories are named `gemma4.gturbo`, `qwen36.gturbo`,
`deepseekv4flash.gturbo`, and `inklingsmall.gturbo`, but detection goes by
each directory's own manifest, not its name. The Mac app scans its library
roots — the `Mference.libraryRoot` default if set, the package checkout's
`scratch/`, and `~/Library/Application Support/Mference` — and auto-adopts
installed models; its toolbar picker switches between families and offers
downloads for missing ones. The CLI and server take an explicit `--model`
path. Non-app selection persists via `defaults write Mference model qwen36`
(or `MFERENCE_MODEL` in the environment). `MferenceCLI --verify
trusted-receipt` skips the first-touch SHA-256 of the expert pool in favor of
the install receipt's size checks; the strict `full-sha256` mode is the
default.

## Local server

Follow the [server guide](docs/OPENAI_SERVER.md) for launch commands, health
checks, client setup, prompt reuse, tool loops, and supported API behavior.
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
process from `pgrep -fl 'MferenceServer|MferenceMac|MferenceDecodeService|MferenceCLI|MferencePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'`.
If a check fails, inform the user and stop; do not terminate apps or delete or
reinstall a model.

Run package tests through `Scripts/test.sh`. Real-model regression suites are
env-gated (for example `MFERENCE_INKLING_GTURBO`) and skip without the gate.
Run only one app, CLI, or model-using test at a time.

For performance results, build release once and follow the [community
benchmark guide](docs/COMMUNITY_BENCHMARKS.md) exactly. Do not enable
experimental controls or profiling. Measured baselines for all four families
are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

Do not download a full checkpoint, duplicate a `.gturbo` model, create a
worktree, or purge caches just to run tests.

Report the commit, hardware and RAM, macOS, Swift version, exact command, exit
code, complete timing footer or error, and every protocol deviation. Treat
results as measurements, not performance ceilings.

## App controls

The Mac app renders each chat through the installed model's own chat format
and template. The UI is a chat-template-style shell: a recency-grouped
sidebar, a toolbar model picker (status dot, whole family, download rows for
missing models), a streaming markdown transcript, and a glass composer with
document attachments. The inspector shows realtime tok/s, token count, and
inference memory, and exposes context length, expert-cache slots, temperature,
Top-K, Top-P, prefill, and RDADVISE. The defaults are temperature `0.2`,
Top-K `64`, and Top-P `0.95`. Responses can use the context space left after
formatting the prompt, and FP16 is the runtime KV format. Build the app with
its sibling `MferenceDecodeService`; it never loads a second in-process model.
See [README](README.md) and [Runtime controls](docs/RUNTIME_CONTROLS.md).
