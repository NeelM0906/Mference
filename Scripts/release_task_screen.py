#!/usr/bin/env python3
"""Frozen 60-case functional screen against an existing loopback model server.

Never starts a server or executes generated code or tool calls. Run with no
other model owner, downloads, profiling, or experimental prefill controls.
Use a 4096-token server context, MTP off and prefix cache off. This protocol is
separate from the community performance benchmark. Output is create-only JSONL.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import socket
import time
import urllib.error
import urllib.request

from swift_qwen_task_screen import score as legacy_score


def equivalent(actual, expected):
    """JSON value equality, without Python's True == 1 loophole."""
    if isinstance(expected, bool) or expected is None:
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, (int, float)):
        return type(actual) in (int, float) and math.isfinite(actual) and actual == expected
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and actual.keys() == expected.keys()
                and all(equivalent(actual[k], v) for k, v in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(equivalent(a, b) for a, b in zip(actual, expected)))
    return type(actual) is type(expected) and actual == expected


def score(case, response):
    if "expected_call" not in case:
        return legacy_score(case, response)
    choices = response.get("choices", [])
    if not choices:
        return False, "missing choice"
    choice = choices[0]
    calls = choice.get("message", {}).get("tool_calls", [])
    if choice.get("finish_reason") != "tool_calls" or len(calls) != 1:
        return False, "expected exactly one completed tool call"
    call = calls[0].get("function", {})
    try:
        actual = {"name": call.get("name"), "arguments": json.loads(call.get("arguments", ""))}
    except (ValueError, TypeError):
        return False, "invalid tool arguments"
    return equivalent(actual, case["expected_call"]), "exact tool name and JSON values"


def read_stream(reply, started, clock=time.monotonic):
    """Reconstruct one choice; record text/tool delta times, never role-only time."""
    message = {"role": "assistant", "content": "", "reasoning_content": ""}
    calls = {}
    usage = None
    finish = None
    first_delta = None
    first_visible = None
    done = False
    for raw in reply:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            break
        event = json.loads(data)
        if "error" in event:
            raise ValueError("stream error: " + json.dumps(event["error"]))
        if event.get("usage") is not None:
            usage = event["usage"]
        choices = event.get("choices", [])
        if not choices:
            continue
        if len(choices) != 1 or choices[0].get("index", 0) != 0:
            raise ValueError("protocol requires exactly one choice")
        choice = choices[0]
        delta = choice.get("delta", {})
        elapsed = clock() - started
        content = delta.get("content") or ""
        reasoning = delta.get("reasoning_content") or ""
        tool_deltas = delta.get("tool_calls") or []
        meaningful_tool = any(part.get("function", {}).get("name") or
                              part.get("function", {}).get("arguments") for part in tool_deltas)
        if first_delta is None and (content or reasoning or meaningful_tool):
            first_delta = elapsed
        if first_visible is None and content:
            first_visible = elapsed
        message["content"] += content
        message["reasoning_content"] += reasoning
        for part in tool_deltas:
            index = part["index"]
            call = calls.setdefault(index, {"id": "", "type": "function",
                                            "function": {"name": "", "arguments": ""}})
            if part.get("id"):
                call["id"] = part["id"]
            for field in ("name", "arguments"):
                call["function"][field] += part.get("function", {}).get(field) or ""
        if choice.get("finish_reason") is not None:
            finish = choice["finish_reason"]
    if not done or finish is None:
        raise ValueError("incomplete SSE response: finish reason and [DONE] required")
    if calls:
        message["tool_calls"] = [calls[k] for k in sorted(calls)]
    response = {"choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage}
    details = (usage or {}).get("completion_tokens_details") or {}
    reasoning_tokens = details.get("reasoning_tokens")
    completion_tokens = (usage or {}).get("completion_tokens")
    # Do not estimate token counts from characters or silently call missing data zero.
    visible_tokens = None
    if not calls and reasoning_tokens is not None and completion_tokens is not None:
        visible_tokens = completion_tokens - reasoning_tokens
    return response, {"first_model_delta_seconds": first_delta,
                      "first_visible_answer_seconds": first_visible,
                      "completed_seconds": clock() - started,
                      "reasoning_tokens": reasoning_tokens,
                      "visible_tokens": visible_tokens,
                      "completion_tokens": completion_tokens}


def run_case(url, body, case, timeout):
    started = time.monotonic()
    request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            response, metrics = read_stream(reply, started)
        passed, reason = score(case, response)
        return {"passed": passed, "rubric": reason, "metrics": metrics, "response": response}
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError, ValueError) as error:
        return {"passed": False, "rubric": "transport or protocol error",
                "metrics": {"completed_seconds": time.monotonic() - started},
                "error": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18489)
    parser.add_argument("--profiles", type=Path, required=True,
                        help="JSON list of {label, model, reasoning_effort?, source_revision}")
    parser.add_argument("--engine-commit", required=True)
    parser.add_argument("--machine-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.timeout <= 0:
        parser.error("port and timeout must be positive and valid")
    corpus = Path(__file__).resolve().parents[1] / "docs/benchmark-prompts/release-screen-v1/cases.json"
    raw = corpus.read_bytes()
    cases = json.loads(raw)
    profiles = json.loads(args.profiles.read_text())
    if not profiles or len({p["label"] for p in profiles}) != len(profiles):
        parser.error("profiles must have unique labels")
    for profile in profiles:
        if not profile.get("model") or not profile.get("source_revision"):
            parser.error("each profile needs model and source_revision")
    settings = {"temperature": 0, "top_p": 1, "top_k": 1, "repetition_penalty": 1,
                "max_completion_tokens": 512, "seed": 20260918, "stream": True,
                "stream_options": {"include_usage": True}}
    with args.output.open("x") as output:
        header = {"protocol": "release-screen-v1", "corpus_sha256": hashlib.sha256(raw).hexdigest(),
                  "engine_commit": args.engine_commit, "machine": args.machine_record.read_text(),
                  "profiles": profiles, "settings": settings, "context": 4096,
                  "required_server_policy": {"mtp": "off", "prefix_cache": "off"},
                  "warmups_per_case": 1, "measured_repetitions": 3}
        output.write(json.dumps(header) + "\n")
        output.flush()
        for profile in profiles:
            for case in cases:
                body = dict(settings, model=profile["model"],
                            messages=[{"role": "user", "content": case["prompt"]}])
                if profile.get("reasoning_effort") is not None:
                    body["reasoning_effort"] = profile["reasoning_effort"]
                if "tool" in case:
                    body["tools"] = [{"type": "function", "function": case["tool"]}]
                for repetition in range(4):
                    result = run_case(f"http://127.0.0.1:{args.port}/v1/chat/completions",
                                      body, case, args.timeout)
                    result.update(profile=profile["label"], case=case["id"], category=case["category"],
                                  repetition=repetition, warmup=repetition == 0, request=body)
                    output.write(json.dumps(result) + "\n")
                    output.flush()
                    print(json.dumps({k: result[k] for k in
                                      ("profile", "case", "repetition", "warmup", "passed", "metrics")}), flush=True)


if __name__ == "__main__":
    main()
