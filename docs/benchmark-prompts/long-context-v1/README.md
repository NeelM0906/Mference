# Long-context v1 — frozen November 14, 2026

A fixed long-prompt workload for measuring 32K-class prefill, decode after a
long prompt, and a warm follow-up turn. It is a speed workload, not a quality
benchmark. Do not change either file; a correction needs a new version.

| File | Contents | Prompt tokens (Qwen 3.6 chat template) | SHA-256 |
| --- | --- | ---: | --- |
| `long-32k.json` | One user message: five repository documents and a briefing task | 31,480 | `023109c46119c246461cf34a1706908e12dae1de2d3376e6756fa8c093d4b2bd` |
| `followup-1500.json` | One user message: one more document excerpt and a short task | 1,443 alone; 32,916 appended to the first turn without its reply | `82b9ed5d9010a1be931a5fac64b3af615ef377ccc6b4fc4f77310d285b521fa2` |

Token counts come from `MFTokenizer.encodeChat` on the pinned `qwen36` install;
other families tokenize differently, so report the prompt tokens each run
prints.

## Source

Both prompts quote this repository's own public documentation at commit
`06c13ee`, unchanged, each document wrapped in a `<document path="...">` tag:

- `long-32k.json`: System design (`docs/SYSTEM_DESIGN.md`),
  [Family contract](../../FAMILY_CONTRACT.md),
  [Runtime controls](../../RUNTIME_CONTROLS.md),
  [Local OpenAI-compatible server](../../OPENAI_SERVER.md), and
  llama.cpp sampling (`docs/LLAMA_SAMPLING.md`) up to its section
  "Real Gemma entry-point checks", followed by a task asking for a sourced
  briefing under 600 words.
- `followup-1500.json`: Runtime diagnostics (`docs/RUNTIME_DIAGNOSTICS.md`) up
  to its section "Gemma prefix recovery", followed by a task asking for an
  answer under 300 words.

The documents in the prompts are the `06c13ee` text, not whatever the linked
files say today.

## How it is run

- CLI, one turn: `--messages-file long-32k.json --max-context 32768
  --max-new 1024 --temperature 0.2 --top-k 64 --top-p 0.95 --seed 20261114`,
  fresh process per run, one discarded warmup.
- Server, two turns: `MferenceServer --max-context 36864` on loopback, the
  first turn sent cold (no retained prefix), then the first turn, its
  assistant reply and `followup-1500.json` sent as the warm second turn, with
  the same sampling settings, `max_completion_tokens` 1024, through
  [`Scripts/server_stream_bench.py`](../../../Scripts/server_stream_bench.py).
  Report time to first token, decode rate, prompt and cached tokens for both
  turns.

Report medians over five measured runs with min and max, and every finish
reason; a run that stops on the token limit is reported as such, not dropped.
