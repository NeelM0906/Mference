#!/usr/bin/env python3
"""Bounded local Swift-Qwen diagnostics, NOT a performance/quality benchmark.

Uses an already-running loopback server. Never executes generated tool calls.
Keeps the frozen release-screen inputs and budgets unchanged. Saves complete
request/response evidence, including reasoning, for local inspection only.
"""
import argparse
import json
from pathlib import Path
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18489)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--profile", choices=["xhigh-legacy", "xhigh-source", "low-source", "tool-string"], required=True)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be in 1..65535")
    if not 1 <= args.max_tokens <= 16384:
        parser.error("max-tokens must be in 1..16384")
    request = {
        "model": "swift-qwen3.8-27b-int4g64", "seed": args.seed,
        "top_p": 0.95, "repetition_penalty": 1,
        "max_completion_tokens": args.max_tokens,
        "reasoning_effort": "low" if args.profile == "low-source" else "xhigh",
        "temperature": 0.2 if args.profile == "xhigh-legacy" else 1.0,
        "top_k": 64 if args.profile == "xhigh-legacy" else 20,
        "messages": json.loads((Path(__file__).resolve().parents[1] /
            "docs/benchmark-prompts/real-generation-v1/short-explanation.json").read_text()),
    }
    if args.profile == "tool-string":
        request.update(reasoning_effort="none", temperature=0, top_k=1, top_p=1,
                       max_completion_tokens=512,
                       messages=[{"role": "user", "content": "Call echo with text exactly 123. Do not answer in prose."}],
                       tools=[{"type": "function", "function": {
                           "name": "echo", "description": "Echo the exact text",
                           "parameters": {"type": "object", "properties": {
                               "text": {"type": "string"}}, "required": ["text"]}}}])
    # Refuse accidental evidence overwrite before sending any request.
    with args.output.open("x") as output:
        record = {"profile": args.profile, "request": request,
                  "classification": "diagnostic; concurrent GLM installation; not a benchmark"}
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{args.port}/v1/chat/completions",
                json.dumps(request).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as response:
                record["response"] = json.load(response)
        except Exception as error:
            record["error"] = str(error)
            raise
        finally:
            json.dump(record, output, ensure_ascii=False, indent=2)
            output.write("\n")
    response = record["response"]
    choice = response["choices"][0]
    message = choice["message"]
    print(json.dumps({"profile": args.profile, "finish": choice["finish_reason"],
        "usage": response["usage"], "content": message.get("content"),
        "tool_calls": message.get("tool_calls"),
        "reasoning_characters": len(message.get("reasoning_content") or "")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
