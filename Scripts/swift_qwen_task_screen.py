#!/usr/bin/env python3
"""Frozen bounded functional screen through an already-running loopback library.

One request at a time; never launches a model or executes generated code/tools.
This is not the community performance protocol or a broad quality benchmark.
Run the server with MFERENCE_MTP=0 and 4096 context for matched plain decode.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def score(case, response):
    choices = response.get("choices", [])
    if not choices:
        return False, "missing choice"
    choice = choices[0]
    message = choice.get("message", {})
    if "expected_call" in case:
        calls = message.get("tool_calls", [])
        if choice.get("finish_reason") != "tool_calls" or len(calls) != 1:
            return False, "expected exactly one completed tool call"
        call = calls[0].get("function", {})
        try:
            actual = {"name": call.get("name"), "arguments": json.loads(call.get("arguments", ""))}
        except (TypeError, ValueError):
            return False, "invalid tool arguments"
        return actual == case["expected_call"], "exact tool name/arguments"
    if choice.get("finish_reason") != "stop":
        return False, "incomplete answer"
    content = (message.get("content") or "").strip()
    if "expected_json" in case:
        try:
            actual = json.loads(content)
        except ValueError:
            return False, "invalid JSON or extra prose"
        # JSON serialization avoids Python treating True and 1 as equal.
        return json.dumps(actual, sort_keys=True) == json.dumps(case["expected_json"], sort_keys=True), "exact JSON"
    return content == case["expected"], "exact trimmed text"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18489)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    corpus = Path(__file__).resolve().parents[1] / "docs/benchmark-prompts/swift-screen-v1/cases.json"
    raw = corpus.read_bytes()
    cases = json.loads(raw)
    settings = {"temperature": 0, "top_p": 1, "top_k": 1, "repetition_penalty": 1,
                "max_completion_tokens": 512, "seed": 20260916}
    profiles = [("base", "qwen3.8-27b-4bit", None)] + [
        ("swift-" + effort, "swift-qwen3.8-27b-int4g64", effort)
        for effort in ("medium", "xhigh", "low", "none")]
    # Refuse overwriting evidence; writes are flushed after each completed case.
    with args.output.open("x") as out:
        out.write(json.dumps({"protocol": "swift-screen-v1", "corpus_sha256": hashlib.sha256(raw).hexdigest(),
                              "settings": settings, "context": 4096, "mtp": "off-required"}) + "\n")
        for label, model, effort in profiles:
            for case in cases:
                body = dict(settings, model=model, messages=[{"role": "user", "content": case["prompt"]}])
                if effort is not None:
                    body["reasoning_effort"] = effort
                if "tool" in case:
                    body["tools"] = [{"type": "function", "function": case["tool"]}]
                request = urllib.request.Request(f"http://127.0.0.1:{args.port}/v1/chat/completions",
                    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                started = time.monotonic()
                try:
                    with urllib.request.urlopen(request, timeout=180) as reply:
                        response = json.load(reply)
                    passed, reason = score(case, response)
                except urllib.error.HTTPError as error:
                    response = {"http_status": error.code, "error": error.read().decode()}
                    passed, reason = False, "HTTP error"
                result = {"profile": label, "case": case["id"], "passed": passed, "rubric": reason,
                          "wall_seconds": time.monotonic() - started, "request": body, "response": response}
                out.write(json.dumps(result) + "\n")
                out.flush()
                print(json.dumps({k: result[k] for k in ("profile", "case", "passed", "wall_seconds")}), flush=True)


if __name__ == "__main__":
    main()
