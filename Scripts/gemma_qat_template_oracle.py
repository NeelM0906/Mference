# /// script
# requires-python = ">=3.12"
# dependencies = ["jinja2==3.1.6", "tokenizers==0.23.2"]
# ///
"""Frozen QAT source renders; no tokenizer or checkpoint download.

uv run Scripts/gemma_qat_template_oracle.py --output /new/oracle.json
Pass --tokenizer /existing/install/tokenizer/tokenizer.json for native IDs.
Existing expected output is never overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment
import tokenizers

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "Tests/Mference/Core/Tokenization/Fixtures/GemmaQATTokenizer"
TEMPLATE_SHA256 = "94899c0f917d93f6fe81c95744d1e8ddab2d21d39228d2e4aec1fb2a25bff413"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, default=FIXTURE / "tokenizer.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Frozen expectations must not be overwritten"
    source = (FIXTURE / "chat_template.jinja").read_bytes()
    assert hashlib.sha256(source).hexdigest() == TEMPLATE_SHA256
    assert jinja2.__version__ == "3.1.6" and tokenizers.__version__ == "0.23.2"
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

    def fail(message):
        raise ValueError(message)

    environment.globals["raise_exception"] = fail
    render = environment.from_string(source.decode()).render
    tokenizer = tokenizers.Tokenizer.from_file(str(args.tokenizer))
    lookup = {"type": "function", "function": {"name": "lookup", "description": "Lookup",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
    combine = {"type": "function", "function": {"name": "combine", "description": "Combine records",
        "parameters": {"type": "object", "properties": {
            "items": {"type": "array", "items": {"type": "object", "properties": {
                "name": {"type": "string"}, "score": {"type": "number", "nullable": True}}, "required": ["name"]}},
            "ok": {"type": "boolean"}}, "required": ["items"]}}}

    def call(identifier="call_1", name="lookup", arguments=None):
        return {"id": identifier, "type": "function", "function": {
            "name": name, "arguments": {"query": "A"} if arguments is None else arguments}}

    opening = [{"role": "user", "content": "Lookup A"}]
    tool_turn = {"role": "assistant", "content": None, "reasoning_content": "Need lookup.", "tool_calls": [call()]}
    result = {"role": "tool", "content": "Found A", "tool_call_id": "call_1"}
    final = {"role": "assistant", "content": "Done.", "reasoning_content": "Checked."}
    histories = {
        "normal": ([{"role": "user", "content": " Hi "}], []),
        "empty_content": ([{"role": "user", "content": ""}], []),
        "system": ([{"role": "system", "content": " Be terse. "}, {"role": "user", "content": "Hi"}], []),
        "developer": ([{"role": "developer", "content": " Be terse. "}, {"role": "user", "content": "Hi"}], []),
        "later_developer": ([{"role": "user", "content": "A"}, {"role": "developer", "content": "Be terse."},
                             {"role": "user", "content": "B"}], []),
        "system_only": ([{"role": "system", "content": "Be terse."}], []),
        "ordinary_reasoning": ([{"role": "user", "content": "A"},
            {"role": "assistant", "content": " B ", "reasoning_content": "Check A."}], []),
        "ordinary_reasoning_new_user": ([{"role": "user", "content": "A"},
            {"role": "assistant", "content": " B ", "reasoning_content": "Check A."},
            {"role": "user", "content": "C"}], []),
        "inline_reasoning": ([{"role": "user", "content": "A"},
            {"role": "assistant", "content": " Before <|channel>thought\nHidden.<channel|> After "}], []),
        "assistant_continuations": ([{"role": "user", "content": "A"},
            {"role": "assistant", "content": "First."}, {"role": "assistant", "content": "Second."}], []),
        "tool_call": (opening + [tool_turn], [lookup]),
        "tool_result": (opening + [tool_turn, result], [lookup]),
        "tool_final": (opening + [tool_turn, result, final], [lookup]),
        "next_user": (opening + [tool_turn, result, final, {"role": "user", "content": "Again"}], [lookup]),
        "repeated_tools": (opening + [tool_turn, result,
            {"role": "assistant", "content": None, "reasoning_content": "Again.", "tool_calls": [call("call_2"), call("call_3")]},
            {"role": "tool", "content": "Second A", "tool_call_id": "call_2"},
            {"role": "tool", "content": "Third A", "tool_call_id": "call_3"}], [lookup]),
        "tool_with_visible_content": (opening + [{**tool_turn, "content": "Checking."}, result], [lookup]),
        "nested_null_arguments": ([{"role": "user", "content": "Combine"},
            {"role": "assistant", "content": None, "reasoning_content": "Check the records.",
             "tool_calls": [call(name="combine", arguments={"items": [{"name": "A", "score": None},
                {"name": "B", "score": 2.5}], "ok": True, "extra": {"empty": [], "nothing": None, "flag": False}})]},
            {"role": "tool", "tool_call_id": "call_1", "content": "Combined"}], [combine]),
        "empty_messages": ([], []),
    }
    cases = []
    for name, (messages, tools) in histories.items():
        for thinking in (False, True):
            for generate in (False, True):
                case = dict(name=name, messages=messages, tools=tools, thinking=thinking, generate=generate)
                try:
                    text = render(messages=messages, tools=tools, bos_token="<bos>",
                                  enable_thinking=thinking, add_generation_prompt=generate)
                    case.update(render=text, ids=tokenizer.encode(text, add_special_tokens=False).ids)
                    # The source does not implement this control. Mference's
                    # accepted API contract rejects true explicitly; it must
                    # never substitute canonical Gemma's history semantics.
                    assert text == render(messages=messages, tools=tools, bos_token="<bos>",
                        enable_thinking=thinking, preserve_thinking=True, add_generation_prompt=generate)
                except (jinja2.TemplateError, ValueError) as error:
                    case["error"] = {"type": type(error).__name__, "message": str(error)}
                cases.append(case)
    payload = dict(checkpoint="mlx-community/gemma-4-26B-A4B-it-qat-q4_0-mlx-aligned",
        revision="745a97a754ed4b7713163c7d0e9c11da41809e0c", template_sha256=TEMPLATE_SHA256,
        tokenizer_sha256=hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        jinja=jinja2.__version__, tokenizers=tokenizers.__version__, cases=cases)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "renders": sum("render" in c for c in cases),
        "source_errors": sum("error" in c for c in cases), "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
