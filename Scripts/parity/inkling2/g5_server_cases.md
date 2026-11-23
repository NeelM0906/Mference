# G5-server cases

`g5_server.py` sends these cases in this order to one running `MferenceServer`.

- **Servers:** one per family with `--model` (MiniCPM5, Qwen 3.6, Maple), and one `--library` server that takes the three families in turn.
- **Settings:** each server runs at `--max-context 4096` with the default prompt cache. Every request uses temperature 0 and seed 7.
- **How runs are compared:** `compare` matches two runs (a control server and a candidate) by model and case. Any difference is a stop-and-report, like a G5 failure.

## Compared fields

- the status, and `error.type`, `error.param` and `error.code` for an error;
- the content and `reasoning_content`, exactly (empty and null differ);
- tool calls, by name and parsed arguments, and also the arguments strings as sent;
- `finish_reason`;
- every integer usage field.

A stream is reduced to its concatenated deltas, the assembled tool calls with their indexes, every finish reason and usage frame, every frame that is not a delta, unparsed lines and the `[DONE]` marker. Ids, `created`, timings and logs are not compared.

`compare` checks every case the families in the files should have, so a case missing on both sides counts as missing. It refuses a file that holds a case twice.

A difference in the arguments string alone, with the parsed JSON equal, is reported on its own line.

## Shared inputs

- **Tools:** `get_weather(city: string, unit: "celsius"|"fahrenheit")`.
- **System:** "You are a concise assistant."

## Cases

| Id | Request | Families | Exercises |
|---|---|---|---|
| P1 | system + user "List three primary colors, one per line.", 48 tokens | all | the plain turn |
| P2 | P1 streaming, `include_usage` | all | the SSE writer |
| P3 | P1 + P1's reply as returned + user "Now name one fruit of each color.", 64 tokens | all | prompt reuse (`cached_tokens`) |
| T1, T2 | user "What is 17 * 23? Answer with the number only.", `chat_template_kwargs.enable_thinking: true`, 384 tokens; non-streaming, streaming | Qwen 3.6, MiniCPM5 | thinking on Qwen 3.6; MiniCPM5 answers 400 (it thinks by default, hidden) |
| E1 | T1's request | Maple | the 400 refusal |
| U1, U2 | user "What is the weather in Paris in celsius? Use the tool." with `tools`, 128 tokens | all | tool calls (non-streaming, streaming) |
| R1, R2 | U1's user message + a fixed assistant turn (`call_0`, `{"city":"Paris","unit":"celsius"}`) + the tool result `{"temp_c": 18, "sky": "clear"}`, 96 tokens | all | tool results fed back |
| L1, L2 | user "Write a 250-word story about a lighthouse keeper.", 400 tokens | all | the detokenizer's window path over hundreds of tokens |
| S1, S2 | P1 with `stop: ["\n\n"]`, 48 tokens | all | the stop matcher on the detokenizer's output |
| S3, S4 | user "Write two short paragraphs about the sea, separated by a blank line." with `stop: ["\n\n"]`, 160 tokens | all | the stop string between visible paragraphs (added after the first run) |

## Coverage seen on 2026-11-23 (main bccd7546 against trunk 9a639656)

Read these cases no wider than they are:
- MiniCPM5's T1 and T2 are refusals (400) on both sides, so they do not exercise thinking.
- S1 and S2 stopped on the string on MiniCPM5 only.
- S3 and S4 were added after this run.


- **Long visible text:** Qwen 3.6 (L: 400 visible tokens) and Maple (332). MiniCPM5's 400 tokens were hidden reasoning.
- **The stop string:** it fired on MiniCPM5 (S: 31 tokens against P1's 37). It did not fire on Qwen 3.6 or Maple.
- **Reasoning:** reasoning content on Qwen 3.6 (T: 226 reasoning tokens).
- **Tool calls:** on all three families.
- **Prompt reuse across conversations:** Qwen 3.6, R1 `cached_tokens` 344.
