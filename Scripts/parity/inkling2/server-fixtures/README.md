# Inkling 2 server fixtures (acceptance checklist items 1-10)

Request and expected-response pairs for the server behavior the agent-acceptance client
(`../agent_acceptance.py`) relies on, written so each can become a `NIOEmbedded` test in
`Tests/MferenceServer` without guessing. Each file names the checklist items it covers, what trunk
`6a646319` already does (`status_on_trunk_6a646319`: PRESENT, PARTIAL or ABSENT, with the code and
tests that pin it), the request body the client sends, and what the server must answer.

- `model_generates` is the raw generation as text, with Inkling 2's special tokens written literally
  (`<|message_model|>`, `<|content_thinking|>`, `<|content_text|>`, `<|content_invoke_tool_json|>`,
  `<|end_message|>`, `<|content_model_end_sampling|>` = EOS 200006). Tokenize it with the Inkling 2
  tokenizer fixture (`Tests/Mference/Core/Tokenization/Fixtures/Inkling2Tokenizer`, where each special
  token is one id) and drive generation with those ids, so the real `StructuredAssistantDecoder`,
  request validation, template render and SSE writer all run.
- `stream` lists the SSE events in order. Text deltas may be split across chunks; the fixture gives
  their concatenation. `"<...>"` values are rules, not literal strings.
- The model id is the product id `inkling-2-mini-ud-q4_k_xl`.

| File | Items | Status on 6a646319 |
|---|---|---|
| `F01-stream-thinking-then-text.json` | 2, 6, 8 | PARTIAL (parts pinned separately, never through HTTP for Inkling) |
| `F02-stream-two-tool-calls.json` | 2, 3 | PARTIAL |
| `F03-tool-results-back-with-thinking-in-history.json` | 4, 6 | PARTIAL |
| `F04-malformed-tool-call-is-text-and-stop.json` | 5 | ABSENT (today: HTTP 500 or an in-band error frame) |
| `F05-reasoning-effort.json` | 6 | PRESENT for every template name after K1 hand-in 12, tested through the server |
| `F06-sampling-defaults-seed-cap-eos.json` | 7 | PRESENT in parts |
| `F07-startup-switches-and-context.json` | 1, 9, 10 | 1 and 9 PRESENT; 10 PARTIAL (Deep fixed at 32,768 in library mode) |
