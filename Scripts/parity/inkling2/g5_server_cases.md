# G5-server cases

`g5_server.py` sends these cases in this order to one running `MferenceServer`.

- **Servers:** one per family with `--model` (MiniCPM5, Qwen 3.6, Maple), and one `--library` server that takes the three families in turn.
- **Settings:** each server runs at `--max-context 4096` with the default prompt cache. Every request uses temperature 0 and seed 7.
- **How runs are compared:** `compare` matches two runs (a control server and a candidate) by model and case. Any difference is a stop-and-report, like a G5 failure.

## Compared fields

- the status, and `error.type`, `error.param` and `error.code` for an error;
- the content and `reasoning_content` (empty and null are equal);
- tool calls, by name and parsed arguments, and also the arguments strings as sent;
- `finish_reason`;
- every integer usage field.

A stream is reduced to the same fields. Ids, `created`, timings and logs are not compared.

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

## Coverage seen on 2026-11-23 (main bccd7546 against trunk 9a639656)

- **Long visible text:** Qwen 3.6 (L: 400 visible tokens) and Maple (332). MiniCPM5's 400 tokens were hidden reasoning.
- **The stop string:** it fired on MiniCPM5 (S: 31 tokens against P1's 37). It did not fire on Qwen 3.6 or Maple.
- **Reasoning:** reasoning content on Qwen 3.6 (T: 226 reasoning tokens).
- **Tool calls:** on all three families.
- **Prompt reuse across conversations:** Qwen 3.6, R1 `cached_tokens` 344.
