#!/usr/bin/env python3
"""Byte-exact chat-template fixtures for the ``glm5`` dialect (GLM-5.3-Flash).

Two outputs under ``Tests/Mference/Core/Tokenization/Fixtures/Glm5Tokenizer/``:

* ``tokenizer.json`` + ``tokenizer_config.json`` — a *synthetic* tokenizer:
  the 256 byte-level base tokens and two merges, with the checkpoint's real
  pre-tokenizer / decoder settings and its 36 added tokens at their **real
  ids** (154820-154855; ``<think>``, ``<tool_call>``, ``<arg_key>`` … are
  added tokens flagged non-special, exactly as shipped). Small enough to
  commit; the ids and the added-token behavior are what the Swift dialect
  keys on.
* ``renders.json`` — a fixed set of conversations rendered through the
  checkpoint's real ``tokenizer.json`` + ``chat_template.jinja`` with
  ``transformers`` ``apply_chat_template``. ``Glm5TemplateTests`` replays the
  inputs through the hand-ported Swift dialect and compares byte for byte.

Tool schemas and call arguments are given with keys in sorted order on
purpose: the Swift ``JSONValue.object`` is unordered and renders keys sorted,
while Jinja preserves insertion order (recorded on docs/families/GLM53_FLASH.md).

    <venv>/bin/python Scripts/parity/glm5_make_template_fixtures.py <dir with tokenizer.json, tokenizer_config.json, chat_template.jinja>
"""
import json, sys, hashlib
from pathlib import Path
from transformers import AutoTokenizer
import transformers

src = Path(sys.argv[1])
out_dir = Path(__file__).resolve().parents[2] / "Tests/Mference/Core/Tokenization/Fixtures/Glm5Tokenizer"
out_dir.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained(str(src))
real = json.loads((src / "tokenizer.json").read_text())

# ---- synthetic tokenizer.json -------------------------------------------------
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b); cs.append(2**8 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

alphabet = [bytes_to_unicode()[b] for b in range(256)]
vocab = {ch: i for i, ch in enumerate(alphabet)}
merges = ["t h", "e r"]
for m in merges:
    vocab[m.replace(" ", "")] = len(vocab)
synthetic = {
    "version": "1.0", "truncation": None, "padding": None,
    "added_tokens": real["added_tokens"],
    "normalizer": real["normalizer"],
    "pre_tokenizer": real["pre_tokenizer"],
    "post_processor": None,
    "decoder": real["decoder"],
    "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": "",
              "end_of_word_suffix": "", "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
              "vocab": vocab, "merges": merges},
}
(out_dir / "tokenizer.json").write_text(json.dumps(synthetic, indent=1, ensure_ascii=False) + "\n")
cfg = json.loads((src / "tokenizer_config.json").read_text())
(out_dir / "tokenizer_config.json").write_text(json.dumps(
    {k: cfg[k] for k in ["tokenizer_class", "eos_token", "pad_token", "extra_special_tokens",
                         "model_max_length", "clean_up_tokenization_spaces"] if k in cfg},
    indent=1, ensure_ascii=False) + "\n")

# ---- renders ------------------------------------------------------------------
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
TOOLS = json.loads(json.dumps(TOOLS, sort_keys=True))

def call(name, args, id_):
    return {"id": id_, "type": "function", "function": {"name": name, "arguments": args}}

CASES = [
    ("user_only", [{"role": "user", "content": "Hi"}], None, {}),
    ("user_only_effort_low", [{"role": "user", "content": "Hi"}], None, {"reasoning_effort": "low"}),
    ("user_only_effort_high", [{"role": "user", "content": "Hi"}], None, {"reasoning_effort": "high"}),
    ("system_user", [{"role": "system", "content": "You are terse."},
                     {"role": "user", "content": "Name a colour."}], None, {}),
    ("multi_turn_plain_assistant", [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Name a colour."},
        {"role": "assistant", "content": "Blue."},
        {"role": "user", "content": "Another?"}], None, {}),
    ("multi_turn_assistant_with_think", [
        {"role": "user", "content": "Name a colour."},
        {"role": "assistant", "content": "<think>red or blue</think>Blue."},
        {"role": "user", "content": "Another?"}], None, {}),
    ("multi_turn_clear_thinking", [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "<think>first</think>A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "<think>second</think>A2"},
        {"role": "user", "content": "Q3"}], None, {"clear_thinking": True}),
    ("content_whitespace", [
        {"role": "system", "content": "  spaced  "},
        {"role": "user", "content": "\nleading and trailing\n"},
        {"role": "assistant", "content": "\n\nanswer\n"},
        {"role": "user", "content": "next"}], None, {}),
    ("trailing_assistant", [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "B"}], None, {}),
    ("later_system", [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "B"},
        {"role": "system", "content": "Now be verbose."},
        {"role": "user", "content": "C"}], None, {}),
    ("tools_no_system", [{"role": "user", "content": "Weather in Paris?"}], TOOLS, {}),
    ("tools_with_system", [{"role": "system", "content": "Be brief."},
                           {"role": "user", "content": "Weather in Paris?"}], TOOLS, {}),
    ("tool_call_and_response", [
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [
            call("get_weather", {"city": "Paris", "unit": "c"}, "call_1")]},
        {"role": "tool", "content": "{\"temp\": 21}", "tool_call_id": "call_1"},
        ], TOOLS, {}),
    ("tool_call_types_and_text", [
        {"role": "user", "content": "Run it."},
        {"role": "assistant", "content": "Running now.", "tool_calls": [
            call("run_code", {"code": "print(1 < 2 & 3)\nprint('x')", "timeout": 5}, "call_2")]},
        {"role": "tool", "content": "True\nx", "tool_call_id": "call_2"},
        {"role": "assistant", "content": "Done."},
        {"role": "user", "content": "Thanks"}], TOOLS, {}),
    ("two_calls_two_results", [
        {"role": "user", "content": "Both."},
        {"role": "assistant", "content": "", "tool_calls": [
            call("get_weather", {"city": "Paris"}, "call_3"),
            call("run_code", {"code": "1+1"}, "call_4")]},
        {"role": "tool", "content": "sunny", "tool_call_id": "call_3"},
        {"role": "tool", "content": "2", "tool_call_id": "call_4"},
        ], TOOLS, {}),
    ("two_results_out_of_order", [
        {"role": "user", "content": "Both."},
        {"role": "assistant", "content": "", "tool_calls": [
            call("get_weather", {"city": "Paris"}, "call_3"),
            call("run_code", {"code": "1+1"}, "call_4")]},
        {"role": "tool", "content": "2", "tool_call_id": "call_4"},
        {"role": "tool", "content": "sunny", "tool_call_id": "call_3"},
        ], TOOLS, {}),
    ("results_without_ids", [
        {"role": "user", "content": "Both."},
        {"role": "assistant", "content": "", "tool_calls": [
            call("get_weather", {"city": "Paris"}, "call_3"),
            call("run_code", {"code": "1+1"}, "call_4")]},
        {"role": "tool", "content": "sunny"},
        {"role": "tool", "content": "2"},
        ], TOOLS, {}),
    ("assistant_think_with_tool_calls", [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": "<think>need the tool</think>Checking.", "tool_calls": [
            call("get_weather", {"city": "Oslo"}, "call_7")]},
        {"role": "tool", "content": "cold", "tool_call_id": "call_7"},
        ], TOOLS, {}),
    ("nested_argument_value", [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "", "tool_calls": [
            call("run_code", {"code": "Zürich \"quoted\"", "opts": {"f": True, "k": [1, 2.5, None]}}, "call_8")]},
        {"role": "tool", "content": "ok", "tool_call_id": "call_8"},
        ], TOOLS, {}),
]

_bench = Path(__file__).resolve().parents[2] / "docs/benchmark-prompts/real-generation-v1"
for _case in ["short-explanation", "medium-review", "long-synthesis"]:
    CASES.append((f"protocol_{_case}", json.loads((_bench / f"{_case}.json").read_text()), None, {}))

renders = []
for name, messages, tools, kwargs in CASES:
    text = tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True,
                                   tokenize=False, **kwargs)
    ids = tok.encode(text, add_special_tokens=False)
    renders.append({"name": name, "messages": messages, "tools": tools,
                    "reasoning_effort": kwargs.get("reasoning_effort", "undefined"),
                    "clear_thinking": kwargs.get("clear_thinking", False),
                    "render": text, "token_ids_real_tokenizer": ids})

special = {t: tok.convert_tokens_to_ids(t) for t in
           ["<|endoftext|>", "[gMASK]", "<sop>", "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>",
            "<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
            "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>", "/nothink"]}
probe = {s: tok.encode(s, add_special_tokens=False) for s in
         ["[gMASK]<sop>", "<|assistant|><think>", "<think></think>", "</think>",
          "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
          "<|observation|><tool_response>x</tool_response>", "Hello world", "<|user|>Hi"]}
out = {
    "source": {"repo": "pipenetwork/GLM-5.3-Flash-MLX-mixed-4_8bit",
               "revision": "d43ea8b407ce4e9c25e6ac9baec3feab70d9f5f3",
               "chat_template_sha256": hashlib.sha256((src / "chat_template.jinja").read_bytes()).hexdigest(),
               "tokenizer_json_sha256": hashlib.sha256((src / "tokenizer.json").read_bytes()).hexdigest(),
               "transformers": transformers.__version__},
    "special_token_ids": special,
    "encoding_probes_real_tokenizer": probe,
    "renders": renders,
}
(out_dir / "renders.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
print("wrote", out_dir / "renders.json", len(renders), "renders;", out_dir / "tokenizer.json")
print(json.dumps(special))
print(json.dumps(probe, ensure_ascii=False))
