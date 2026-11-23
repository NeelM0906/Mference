# Inkling 2 Mini reference fixtures

Small goldens for the Inkling 2 Mini family (`inkling2Loop`), taken from the
shipped weights file itself: `unsloth/Inkling-2-Mini-GGUF` at commit
`45fd08764f2b0bf16b03a5483a80e5c7aaec8772`,
`Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file`, 17,629,760,736 bytes,
SHA-256 `e64276656f072e9b2197a61723667cbdb71da66a716ffd966e572c21088f3cd3`.
The reference is stock upstream llama.cpp at tag `b12641`
(`d2bceda3de6d61429b885f4f012bc983c6dcfd50`, the first tag with the inkling2
tokenizer fix `ignore_merges`). Generators and the build recipe are in
[`Scripts/parity/inkling2/`](../../../../Scripts/parity/inkling2/README.md).

| File | Contents | Made by |
|---|---|---|
| `gguf_tensors.tsv` | All 744 tensors: index, name, ggml type and id, `ne0..ne3`, absolute file offset, bytes | `gguf_header_dump.py` (stdlib GGUF v3 parser using the clone's `gguf-py/gguf/constants.py`); matches llama.cpp's C++ reader on every tensor |
| `gguf_kv.json` | All 63 key/values; long arrays (tokens, merges, token types) and the chat template as length + SHA-256 | same |
| `gguf_summary.json` | Version, counts, alignment, data start, bytes per type | same |
| `tokenizer_goldens.json` | 22 texts (ASCII, whitespace, numbers, code, JSON, accents, CJK, other scripts, emoji, special-token markup, long words, a paragraph): token ids and pieces with `parse_special` false (plain text) and true, no BOS (the model adds none), round-trip flag | `ink-golden tokenize` (llama_tokenize), inputs `inputs/tokenizer_texts.json` |
| `chat_goldens.json` | 16 chat-template renders and their ids: user, system, every `reasoning_effort` name and 0.5, multi-turn, assistant thinking in history, tools declared, a tool call with its result, a named tool result with reasoning, user text containing a special-token string | `ink-golden chat` (llama.cpp's Jinja renderer through `common_chat_templates_apply`, as llama-server renders; the whole string tokenized with `parse_special` true, as llama-server does), inputs `inputs/chat_cases.json` |

Notes for the native side:

- The template ends after the thinking-effort system message (or the last
  message); it adds no assistant header. The model writes
  `<|message_model|>` itself.
- Effort names map to the numbers in the template's `effort_map` (`none` 0,
  `minimal` 0.1, `low` 0.2, `medium` 0.7, `high` 0.9, `xhigh` and `max` 0.99);
  with no effort the template uses 0.9.
- `c09-user-special-text`: llama-server tokenizes the whole render with
  special-token parsing, so a `<|end_message|>` typed by the user becomes the
  special token. That is the reference's behaviour, recorded as it is.

## `generate/`: greedy, log-probability and routing goldens

From `ink-golden generate` with `-ngl 99` (the default offload: NVFP4 expert
products on the CPU, everything else on Metal), n_ctx 4096, n_ubatch 512,
8 threads, F16 KV, flash attention auto, inputs
`Scripts/parity/inkling2/inputs/generate_prompts.json`.

- `<id>/golden.json` for 10 prompts (5 raw, 5 chat: effort none, default,
  low, a tool turn, after a tool result) and `trace01`: prompt text and ids,
  greedy generated ids (argmax, ties to the lower id), text, pieces, stop.
- `g01-capital`, `g05-wetlands`, `g06-chat-none`: `logprobs.json`, the top 20
  log-probabilities (log-softmax over the 200,058 real tokens) at every
  prompt and generated position, `[[id, logprob], ...]` plus the next token.
- `trace01/trace.json`: for each of the 48 layer visits (layers 0-5, the
  core 6-17 three times, 18-23) the cache slot, the physical layer, and per
  position the 8 routed expert ids with their weights (normalized, scaled by
  8). The per-visit hidden states themselves (2.2 MB) are kept with the
  program's results, not here.
- `noise-metal-vs-cpu.json`: the same prompts run CPU-only (no GPU device,
  no op offload, no weight repacking) compared with these: the reference's
  own backend noise, from which the G6 tolerances were proposed. Metal run
  twice is bit-identical.
