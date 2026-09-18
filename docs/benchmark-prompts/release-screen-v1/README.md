# Release screen v1 — frozen September 18, 2026

This is a bounded **functional** screen, not a general model-quality ranking,
coding benchmark, BF16 quantization comparison or the community speed protocol.
The 60 short cases comprise ten each of everyday questions, reasoning,
code reading, instruction following, tool-call formation and synthesis of
provided facts. Code and tools are never executed. Short structured tasks do
not measure open-ended writing or sophisticated programming ability.

The corpus and rubrics are fixed before model runs. Keep failed answers,
timeouts, malformed tools, empty output and token-limit truncations in the
denominator. Do not change a question or expected answer after seeing outputs;
any corpus correction requires a new protocol version and complete rerun.

## Rubrics and measurement

- Text: exact trimmed answer. JSON: exact keys and values, no prose/fences.
- Tools: exactly one completed call with the named function and exact JSON
  arguments; numeric values may use equivalent integer/decimal notation, but
  booleans are not numbers. Tool responses are not simulated in this screen.
- Text/JSON requires `finish_reason=stop`; tools require `tool_calls`.
- Primary result: successes out of all measured requests, by category/profile.
  Report the per-case outcomes too; three deterministic repeats are not three
  independent quality questions.
- Secondary: time to first nonempty model text/reasoning/tool delta, first
  visible answer text, completed response, total completion tokens, reasoning
  and visible token counts where the server explicitly reports them. Missing
  counts are null, not zero or character-based estimates. A tool-only answer
  has no first-visible-text time. Failed/truncated requests are not fast wins.
  Payload counts exclude channel delimiters, tool payloads and EOS, so visible
  tokens must not be inferred by subtracting reasoning from total completion.
  Counts describe generated channel tokens (including whitespace/buffered bytes),
  not a re-tokenization of the displayed answer. A string-stop-filtered response
  omits visible counts because the cutoff can fall inside a token. Legacy
  base-Qwen hidden reasoning is not streamed, so its first emitted delta is not
  its first generated token. Explicit base reasoning-effort requests now return
  reasoning deltas like Swift; record which policy was used.
- Latency summaries must distinguish all requests from successful requests;
  report median/range across the three repeats, not the best repeat.

Use one discarded warmup **per case and profile**, then three measured repeats,
one request/model at a time. Disable prefix caching to prevent repeats becoming
cache-hit measurements. Use 4096 context, MTP off, greedy sampling, seed
20260918, and 512 maximum completion tokens. Truncations stay failures: do not
raise the cap after inspecting an answer. No downloads or other demanding work
may run during the measured screen.

Record the engine commit, dirty status, source revisions, hardware/RAM, OS,
Swift, model manifests, server command/policy, corpus hash and all responses.
The script records supplied provenance but cannot certify the server was
launched with those settings; preserve its startup log as evidence.

Run `Scripts/release_task_screen.py` against an already-running loopback server;
it never creates a second model owner. `--profiles` names a JSON list containing
`label`, `model`, `source_revision`, and optional `reasoning_effort`. Required
arguments also include `--engine-commit`, `--machine-record` and `--output`.
Output creation refuses to overwrite existing evidence. Interrupted files stay
partial and must not be called complete results.

For Swift/base comparison use matching supported reasoning policies separately
from each checkpoint's product defaults. The legacy base tool template disables
thinking; those tool rows are not effort-matched. Never attribute fine-tune
token savings or shorter reasoning policy to a kernel speedup. Flash-Next/GLM
comparisons use their own explicit supported policies, with differences shown.
The existing `swift-screen-v1` and frozen community prompts remain unchanged.

The separately selected `matched-low.json` profiles exercise explicit low
effort on both base and Swift with identical rendered inputs. They do not
change this frozen corpus, budget, sampling or repetition protocol, and they
are not a test of source-default xhigh. Summarize a completed or partial run
with `python3 Scripts/release_screen_summary.py OUTPUT.jsonl`; missing/duplicate
records cannot silently become a completed result, and failures remain in the
denominator. Paired token counts conditional on both answers passing are
reported separately from all-request counts.
