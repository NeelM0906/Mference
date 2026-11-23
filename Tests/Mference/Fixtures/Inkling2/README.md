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
| `tokenizer_goldens.json` | 29 texts: 22 general (ASCII, whitespace, numbers, code, JSON, accents, CJK, other scripts, emoji, special-token markup, long words, a paragraph); 5 where `ignore_merges` matters (`t23`-`t26`: `def f(*args, **kwargs):`, `Aleksandr Klebsiella`, `日本について知りません`, `keineswegs anderzijds`; `t27`: all 337 such vocabulary words, one per line); 2 labelled known differences (`t28` `getÜbersicht`, `t29` `ÉTAT`: the vendor tokenizer splits these patterns differently from llama.cpp, so they are excluded from G6 token agreement). Token ids, pieces (readable; bytes that are part of a multi-byte character show as U+FFFD) and `pieces_hex` (exact bytes) with `parse_special` false and true, no BOS (the model adds none), round-trip flag | `ink-golden tokenize` (llama_tokenize), inputs `inputs/tokenizer_texts.json` |
| `ignore_merges_words.json` | The 337 vocabulary entries that plain BPE merges do not rebuild as one token (id, byte-level token, decoded text, the BPE pieces) | `ignore_merges_words.py` (GGUF header only) |
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
  position that predicts a token: each prompt position and each generated
  token except the last (n_prompt + n_gen - 1 rows), `[[id, logprob], ...]`
  plus the next token.
- `trace01/trace.json`: for each of the 48 layer visits (layers 0-5, the
  core 6-17 three times, 18-23) the cache slot, the physical layer, and per
  position the 8 routed expert ids with their weights (normalized, scaled by
  8). The per-visit hidden states (`hidden_after_visit.f32`, 2.2 MB) and
  `result_norm.f32`, which trace.json names, are kept with the program's
  results, not here.
## `generate2/`: the second prompt set

Same settings, inputs `inputs/generate_prompts_2.json`: eight more log-prob
prompts (chat with a thinking block in the history, a tool call and its
result, ~1,300 and ~2,600 raw tokens with log-probs on the last 64 prompt
positions and the generated ones (`logprobs_first_pos` in golden.json), code,
default chat, multilingual, multi-turn) and `trace02`, a chat-formatted
20-token trace with log-probs at every position (so its `result_norm` has
all 20 rows). With `generate/` that makes 11 log-prob prompts and 2 traces.

## `cpu/`: the CPU-only traces (G6a reference)

`trace01/` and `trace02/`: `golden.json` and `trace.json` (and, for
`trace02`, the `logprobs.json` its golden.json names) from the same
prompts run with `-ngl 0`, no GPU device, no op offload and no CPU weight
repacking, so every op runs on the CPU backend with the reference's Q8_0
rounding of the inputs to NVFP4 and Q8_0 products. `trace.json` holds, per
layer visit, the routed expert ids and weights (graph tensors
`ffn_moe_topk-<slot>`, `ffn_moe_weights_scaled-<slot>`). The FP32 hidden
states after each visit (`l_out-<slot>`, `[48 x n_prompt x 2304]`) are with
the program's results (`goldens/generate*/cpu/<trace>/hidden_after_visit.f32`),
as is a third trace of the 300-token tool prompt (2.5 MB of JSON).

## Using the CPU-only trace (G6a)

Feed the runner `cpu/<trace>/golden.json` `prompt_ids` (no BOS) as one
prefill. The hidden state after each layer visit is kept outside the
repository (`goldens/generate*/cpu/<trace>/hidden_after_visit.f32`): raw
IEEE-754 binary32, little-endian, no header, `[48 visits x n_prompt x 2304]`
row-major, so visit `v`, position `t`, channel `c` is at byte
`4 * ((v * n_prompt + t) * 2304 + c)`. Visit `v` is `trace.json`
`visits[v]` (`layer` = physical layer, `slot` = cache slot): visits 0-5 are
layers 0-5 in slots 0-5, 6-17 core pass 1 (layers 6-17, slots 6-17), 18-29
pass 2 (slots 24-35), 30-41 pass 3 (slots 36-47), 42-47 layers 18-23 in slots
18-23. Each record is the layer output `l_out-<slot>` (after the MoE
residual add), the next visit's input; visit 0's input is the token
embedding after the `token_embd_norm` RMS norm. Expert ids per visit and
position are in `trace.json` (compare as sets; weights sum to 8). The CPU-only
run rounds the inputs of NVFP4 and Q8_0 products to Q8_0. The full
definitions are in the program's goldens README ("Trace records").

## `noise/`: the reference against itself

`compare_goldens.py` outputs: Metal vs Metal in a fresh process
(bit-identical), and Metal vs CPU-only for each set (`metal-vs-cpu.json`,
`set2-...`, `set3-...` for the tool-prompt trace). The backend split grows
with context and on structured text (tool JSON); the G6 tolerances are set
per prompt from these numbers. `table-logprobs.md` and `table-traces.md` are
those numbers as tables (`noise_table.py`); p95 is nearest-rank.
