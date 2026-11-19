#!/usr/bin/env python3
"""Generate the native engine's fixed prompt set (docs/NATIVE_PARITY.md).

    python3 Scripts/native/golden/make_prompt_set.py            # write the fixture
    python3 Scripts/native/golden/make_prompt_set.py --check    # exit 1 if it differs

Writes Tests/Mference/Fixtures/native/prompts/quick-v1.json. The set reuses
frozen prompts already in the repository: the engine program's C4 cases
(Scripts/engine_validate.py: short chat, 2K-token document question, tool call
and tool result, JSON answer), the real-generation-v1 prompts, and
long-context-v1's 32K prompt. The ~4K document question is a longer ledger from
the same deterministic generator as the C4 2K case. Stdlib only, no model.

The set is frozen once goldens are captured from it: a change needs a new
version (quick-v2), never an edit of quick-v1.
"""
import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "Tests/Mference/Fixtures/native/prompts/quick-v1.json"
PROMPTS = ROOT / "docs/benchmark-prompts"


def _engine_validate():
    spec = importlib.util.spec_from_file_location("engine_validate", ROOT / "Scripts/engine_validate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _messages(path):
    return json.loads((PROMPTS / path).read_text(encoding="utf-8"))


def _doc_question(ev, entries, target):
    doc = ev.ledger(entries)
    line = doc[target - 1]
    shelf = re.search(r"shelf (\S+),", line).group(1)
    return ("Here is a warehouse ledger.\n\n" + "\n".join(doc) + "\n\n"
            f"Question: which supplier delivered the goods on shelf {shelf} recorded in "
            f"entry {target}, and how many units were there? Answer in one sentence.")


def build():
    ev = _engine_validate()
    cases = {c["name"]: c for c in ev.prompt_cases()}
    tool = ev.WEATHER_TOOL
    lisbon = "What is the weather in Lisbon right now? Use the tool."
    # capture: rows kept for per-op tensors (head, tail, every stride-th) and
    # decode steps; "all" means every prompt row.
    every_row = {"head": 1 << 20, "tail": 0, "stride": 0, "decode_steps": 8}
    prompts = [
        {"id": "short-chat", "origin": "C4 short-chat (Scripts/engine_validate.py)",
         "messages": cases["short-chat"]["turns"][0]["messages"], "max_new": 256, "capture": every_row},
        {"id": "json-answer", "origin": "C4 json-answer",
         "messages": cases["json-answer"]["turns"][0]["messages"], "max_new": 256, "capture": every_row},
        {"id": "tool-call", "origin": "C4 tool-roundtrip, first turn",
         "messages": cases["tool-roundtrip"]["turns"][0]["messages"], "tools": [tool],
         "max_new": 256, "capture": every_row},
        {"id": "tool-result", "origin": "C4 tool-roundtrip, second turn (fixed tool call)",
         "messages": [
             {"role": "user", "content": lisbon},
             {"role": "assistant", "content": "",
              "tool_calls": [{"id": "call_0", "type": "function",
                              "function": {"name": "get_weather", "arguments": "{\"city\": \"Lisbon\"}"}}]},
             {"role": "tool", "tool_call_id": "call_0",
              "content": json.dumps({"city": "Lisbon", "temperature_c": 18, "conditions": "light rain"})},
         ],
         "tools": [tool], "max_new": 192, "capture": every_row},
        {"id": "explain-thinking", "origin": "real-generation-v1/short-explanation.json, thinking on",
         "messages": _messages("real-generation-v1/short-explanation.json"), "enable_thinking": True,
         "max_new": 512, "capture": every_row},
        {"id": "essay-512", "origin": "quick-v1 (new): a long free-form answer",
         "messages": [{"role": "user", "content":
                       "Explain step by step how a TCP connection is opened with the three-way handshake "
                       "and closed again, what each flag and sequence number means, and what goes wrong "
                       "when a SYN is lost or a SYN flood arrives. Write about 400 words of plain prose."}],
         "max_new": 512, "capture": every_row},
        {"id": "review-512", "origin": "real-generation-v1/medium-review.json (~512 tokens)",
         "messages": _messages("real-generation-v1/medium-review.json"), "max_new": 512,
         "capture": {"head": 64, "tail": 64, "stride": 32, "decode_steps": 8}},
        {"id": "doc-2k", "origin": "C4 doc-2k",
         "messages": cases["doc-2k"]["turns"][0]["messages"], "max_new": 128,
         "capture": {"head": 16, "tail": 32, "stride": 128, "decode_steps": 8}},
        {"id": "synthesis-3k", "origin": "real-generation-v1/long-synthesis.json (~3K tokens, long answer)",
         "messages": _messages("real-generation-v1/long-synthesis.json"), "max_new": 512,
         "capture": {"head": 16, "tail": 32, "stride": 256, "decode_steps": 8}},
        {"id": "doc-4k", "origin": "C4 ledger generator, 120 entries, question on entry 87 (~4K tokens)",
         "messages": [{"role": "user", "content": _doc_question(ev, 120, 87)}], "max_new": 128,
         "capture": {"head": 16, "tail": 32, "stride": 256, "decode_steps": 8}},
        {"id": "long-32k", "origin": "long-context-v1/long-32k.json (~32K tokens)",
         "messages": _messages("long-context-v1/long-32k.json"), "max_new": 128,
         "capture": {"head": 16, "tail": 32, "stride": 2048, "decode_steps": 8}},
    ]
    return {
        "name": "quick-v1",
        "version": 1,
        "about": "Fixed prompt set for native-engine parity (docs/NATIVE_PARITY.md). Frozen; "
                 "generated by Scripts/native/golden/make_prompt_set.py.",
        "defaults": {"enable_thinking": False},
        "prompts": prompts,
    }


def render(obj):
    return json.dumps(obj, indent=1, ensure_ascii=False, sort_keys=False) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if the committed fixture differs")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    text = render(build())
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
        if current != text:
            print(f"{args.out} is out of date; rerun {Path(__file__).name}", file=sys.stderr)
            return 1
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
