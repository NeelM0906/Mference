#!/usr/bin/env python3
"""Byte-exact chat-template fixtures for the ``minicpm5`` dialect.

Renders a fixed set of conversations through the pinned checkpoint's real
``tokenizer.json`` + ``chat_template.jinja`` with ``transformers``
``apply_chat_template`` and writes them, with their inputs, to
``Tests/Mference/Core/Tokenization/Fixtures/MiniCPM5Tokenizer/renders.json``.
``MiniCPM5TemplateTests`` replays the inputs through the hand-ported Swift
dialect and compares the strings byte for byte.

Tool-call arguments are given with keys in sorted order on purpose: the Swift
``JSONValue.object`` is unordered and renders parameters sorted by key, while
Jinja preserves insertion order (recorded on docs/families/MINICPM5.md).

    Scripts/parity/minicpm5_make_template_fixtures.py <dir-with-tokenizer-files>
"""
import json, sys, hashlib
from pathlib import Path
from transformers import AutoTokenizer
import transformers

src = Path(sys.argv[1])
out_dir = Path(__file__).resolve().parents[2] / "Tests/Mference/Core/Tokenization/Fixtures/MiniCPM5Tokenizer"
tok = AutoTokenizer.from_pretrained(str(src))

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"},
                                      "unit": {"type": "string", "enum": ["c", "f"]}},
                       "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "run_code",
        "description": "Run a snippet & return stdout.",
        "parameters": {"type": "object",
                       "properties": {"code": {"type": "string"},
                                      "timeout": {"type": "integer"}},
                       "required": ["code"]}}},
]

def call(name, args, id_):
    return {"id": id_, "type": "function", "function": {"name": name, "arguments": args}}

CASES = [
    ("user_only_thinking_undefined", [{"role": "user", "content": "Hi"}], None, {}),
    ("user_only_thinking_true", [{"role": "user", "content": "Hi"}], None, {"enable_thinking": True}),
    ("user_only_thinking_false", [{"role": "user", "content": "Hi"}], None, {"enable_thinking": False}),
    ("system_user", [{"role": "system", "content": "You are terse."},
                     {"role": "user", "content": "Name a colour."}], None, {"enable_thinking": True}),
    ("multi_turn_plain_assistant", [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Name a colour."},
        {"role": "assistant", "content": "Blue."},
        {"role": "user", "content": "Another?"}], None, {"enable_thinking": True}),
    ("multi_turn_assistant_with_think", [
        {"role": "user", "content": "Name a colour."},
        {"role": "assistant", "content": "<think>\nred or blue\n</think>\n\nBlue."},
        {"role": "user", "content": "Another?"}], None, {"enable_thinking": True}),
    ("multi_turn_last_assistant_keeps_think", [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "<think>\nfirst\n</think>\n\nA1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "<think>\nsecond\n</think>\n\nA2"},
        {"role": "user", "content": "Q3"}], None, {"enable_thinking": False}),
    ("content_untrimmed_with_newlines", [
        {"role": "system", "content": "  spaced  "},
        {"role": "user", "content": "\nleading and trailing\n"},
        {"role": "assistant", "content": "\n\nanswer\n"},
        {"role": "user", "content": "next"}], None, {"enable_thinking": True}),
    ("tools_no_system", [{"role": "user", "content": "Weather in Paris?"}], TOOLS, {"enable_thinking": True}),
    ("tools_with_system", [{"role": "system", "content": "Be brief."},
                           {"role": "user", "content": "Weather in Paris?"}], TOOLS, {"enable_thinking": True}),
    ("tools_system_with_tool_def_sep", [
        {"role": "system", "content": "Intro.\n<tool_def_sep>\nOutro."},
        {"role": "user", "content": "Weather in Paris?"}], TOOLS, {"enable_thinking": True}),
    ("tool_call_and_response", [
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [
            call("get_weather", {"city": "Paris", "unit": "c"}, "call_1")]},
        {"role": "tool", "content": "{\"temp\": 21}", "tool_call_id": "call_1"},
        ], TOOLS, {"enable_thinking": True}),
    ("tool_call_cdata_and_types", [
        {"role": "user", "content": "Run it."},
        {"role": "assistant", "content": "Running now.", "tool_calls": [
            call("run_code", {"code": "print(1 < 2 & 3)\nprint('x')", "timeout": 5}, "call_2")]},
        {"role": "tool", "content": "True\nx", "tool_call_id": "call_2"},
        {"role": "assistant", "content": "Done."},
        {"role": "user", "content": "Thanks"}], TOOLS, {"enable_thinking": False}),
    ("two_calls_two_results", [
        {"role": "user", "content": "Both."},
        {"role": "assistant", "content": "", "tool_calls": [
            call("get_weather", {"city": "Paris"}, "call_3"),
            call("run_code", {"code": "1+1"}, "call_4")]},
        {"role": "tool", "content": "sunny", "tool_call_id": "call_3"},
        {"role": "tool", "content": "2", "tool_call_id": "call_4"},
        ], TOOLS, {"enable_thinking": True}),
    ("assistant_content_with_tool_sep", [
        {"role": "user", "content": "Both."},
        {"role": "assistant", "content": "First<tool_sep>then", "tool_calls": [
            call("get_weather", {"city": "Rome"}, "call_5"),
            call("run_code", {"code": "2*2"}, "call_6")]},
        {"role": "tool", "content": "rain", "tool_call_id": "call_5"},
        {"role": "tool", "content": "4", "tool_call_id": "call_6"},
        ], TOOLS, {"enable_thinking": True}),
    ("assistant_think_with_tool_calls", [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": "<think>\nneed the tool\n</think>\n\nChecking.", "tool_calls": [
            call("get_weather", {"city": "Oslo"}, "call_7")]},
        {"role": "tool", "content": "cold", "tool_call_id": "call_7"},
        ], TOOLS, {"enable_thinking": True}),
]

renders = []
for name, messages, tools, kwargs in CASES:
    text = tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True,
                                   tokenize=False, **kwargs)
    ids = tok.encode(text, add_special_tokens=False)
    renders.append({"name": name, "messages": messages, "tools": tools,
                    "enable_thinking": kwargs.get("enable_thinking", "undefined"),
                    "render": text, "token_ids_real_tokenizer": ids})

special = {t: tok.convert_tokens_to_ids(t) for t in
           ["<s>", "</s>", "<|im_start|>", "<|im_end|>", "<think>", "</think>",
            "<function", "</function>", "<param", "<tool_response>", "</tool_response>",
            "<tools>", "</tools>", "<tool_call>", "</tool_call>", "<unk>"]}
probe = {s: tok.encode(s, add_special_tokens=False) for s in
         ["<think>\n", "</think>\n\n", "<function name=\"get_weather\"><param name=\"city\">Paris</param></function>",
          "<tool_response>\nx\n</tool_response>", "<tool_sep>", "Hello world", "<|im_start|>user\nHi<|im_end|>\n"]}
out = {
    "source": {"repo": "openbmb/MiniCPM5-2B", "revision": "cd199ce3ee67549c42ef7372f809f2c63599a3e9",
               "chat_template_sha256": hashlib.sha256((src / "chat_template.jinja").read_bytes()).hexdigest(),
               "tokenizer_json_sha256": hashlib.sha256((src / "tokenizer.json").read_bytes()).hexdigest(),
               "transformers": transformers.__version__},
    "special_token_ids": special,
    "encoding_probes_real_tokenizer": probe,
    "renders": renders,
}
(out_dir / "renders.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
print("wrote", out_dir / "renders.json", len(renders), "renders")
print(json.dumps(special))
print(json.dumps(probe, ensure_ascii=False))
