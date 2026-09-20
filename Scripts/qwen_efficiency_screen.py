#!/usr/bin/env python3
"""Frozen five-seed Qwen source-policy functional/token screen, not a speed benchmark.

Uses an already-running loopback server, never starts a model or executes tools.
The separate release-screen-v1 corpus and rubrics remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path

from release_task_screen import run_case

PROTOCOL = "qwen-source-efficiency-v1"
SEEDS = [20260920, 20260921, 20260922, 20260923, 20260924]
SETTINGS = {"temperature": 1.0, "top_k": 20, "top_p": 0.95,
            "repetition_penalty": 1.0, "max_completion_tokens": 4096,
            "reasoning_effort": "xhigh", "stream": True,
            "stream_options": {"include_usage": True}}
PROFILES = [
    {"label": "base-xhigh", "model": "qwen3.8-27b-4bit",
     "source_revision": "legacy receipt lacks revision; index matches pin 3e6447f082e89cc7f0bc6e5441afd38dfce760ff"},
    {"label": "swift-xhigh", "model": "swift-qwen3.8-27b-int4g64",
     "source_revision": "1b30aaaf753fe5c1cb51ada2ea0367a53445359c"},
]


def request(profile, case, seed):
    body = dict(SETTINGS, model=profile["model"], seed=seed,
                messages=[{"role": "user", "content": case["prompt"]}])
    if "tool" in case:
        body["tools"] = [{"type": "function", "function": case["tool"]}]
    return body


def summarize(rows, cases):
    expected = {(p["label"], c["id"], seed) for p in PROFILES for c in cases for seed in SEEDS}
    indexed = {}
    for row in rows:
        key = row["profile"], row["case"], row["seed"]
        if key not in expected or key in indexed:
            raise ValueError("unexpected or duplicate result: " + str(key))
        indexed[key] = row
    result = {"protocol": PROTOCOL, "complete": set(indexed) == expected,
              "received": len(indexed), "expected": len(expected), "profiles": {}}
    for p in PROFILES:
        selected = [r for r in rows if r["profile"] == p["label"]]
        totals = {}
        for metric in ("completion_tokens", "reasoning_tokens", "visible_tokens"):
            values = [r.get("metrics", {}).get(metric) for r in selected]
            totals[metric] = sum(values) if values and all(type(v) is int for v in values) else None
        result["profiles"][p["label"]] = {
            "passed": sum(r["passed"] is True for r in selected),
            "observed": len(selected), "expected": len(cases) * len(SEEDS),
            "cases_passing_all_seeds": sum(all(indexed.get((p["label"], c["id"], s), {}).get("passed") is True
                                              for s in SEEDS) for c in cases),
            "all_request_token_totals": totals,
            "failures": [{"case": r["case"], "seed": r["seed"], "rubric": r["rubric"]}
                         for r in selected if r["passed"] is not True],
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18489)
    parser.add_argument("--engine-commit", required=True)
    parser.add_argument("--machine-record", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("invalid port")
    corpus = Path(__file__).resolve().parents[1] / "docs/benchmark-prompts/release-screen-v1/cases.json"
    raw = corpus.read_bytes()
    cases = json.loads(raw)
    header = {"protocol": PROTOCOL, "engine_commit": args.engine_commit,
              "machine": args.machine_record.read_text(), "corpus_sha256": hashlib.sha256(raw).hexdigest(),
              "profiles": PROFILES, "settings": SETTINGS, "seeds": SEEDS, "context": 8192,
              "required_server_policy": {"mtp": "off", "prefix_cache": "off"},
              "interpretation": "functional/token counts only; 60 tasks, not 300 independent questions; not upstream benchmark reproduction"}
    rows = []
    errors = 0
    with args.output.open("x") as output:
        output.write(json.dumps(header) + "\n")
        output.flush()
        for profile in PROFILES:
            for case in cases:
                for seed in SEEDS:
                    body = request(profile, case, seed)
                    row = run_case(f"http://127.0.0.1:{args.port}/v1/chat/completions", body, case, 300)
                    row.update(profile=profile["label"], case=case["id"], category=case["category"],
                               seed=seed, request=body)
                    rows.append(row)
                    output.write(json.dumps(row) + "\n")
                    output.flush()
                    print(json.dumps({k: row[k] for k in ("profile", "case", "seed", "passed", "metrics")}), flush=True)
                    errors = errors + 1 if "error" in row else 0
                    if errors >= 3:
                        raise RuntimeError("three consecutive transport/server errors; evidence remains partial")
    print(json.dumps({"summary": summarize(rows, cases)}), flush=True)


if __name__ == "__main__":
    main()
