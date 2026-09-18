#!/usr/bin/env python3
"""Summarize a frozen release-screen-v1 run without dropping failed requests.

No latency/speed claims: this is functional and token accounting, separate from
the community performance protocol. Repeated greedy runs are not independent
quality questions. An incomplete run is labeled partial and exits 1.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def summarize(header, rows, cases):
    profiles = [profile["label"] for profile in header["profiles"]]
    if not profiles or len(profiles) != len(set(profiles)):
        raise ValueError("profiles must be nonempty and unique")
    case_ids = {case["id"] for case in cases}
    expected = {(profile, case, repeat) for profile in profiles for case in case_ids for repeat in range(4)}
    indexed = {}
    for row in rows:
        key = row["profile"], row["case"], row["repetition"]
        if key not in expected or key in indexed:
            raise ValueError("unexpected or duplicate result: " + str(key))
        if row["warmup"] != (row["repetition"] == 0):
            raise ValueError("incorrect warmup label")
        indexed[key] = row
    result = {"complete": set(indexed) == expected, "expected_records": len(expected),
              "received_records": len(indexed), "profiles": {},
              "interpretation": "functional screen; repeats are not independent tasks; not an upstream benchmark reproduction"}
    for profile in profiles:
        measured = [row for (label, _, repeat), row in indexed.items() if label == profile and repeat > 0]
        categories = {}
        for case in cases:
            bucket = categories.setdefault(case["category"], {"passed": 0, "observed": 0, "expected": 0})
            bucket["expected"] += 3
            for repeat in range(1, 4):
                row = indexed.get((profile, case["id"], repeat))
                if row is not None:
                    bucket["observed"] += 1
                    bucket["passed"] += row["passed"] is True
        finishes = Counter()
        for row in measured:
            choices = row.get("response", {}).get("choices", [])
            finishes[choices[0].get("finish_reason", "missing") if choices else "error"] += 1
        token_totals = {}
        for metric in ("completion_tokens", "reasoning_tokens", "visible_tokens"):
            values = [row["metrics"].get(metric) for row in measured]
            # Unknown usage must not quietly turn into zero token use.
            token_totals[metric] = sum(values) if values and all(type(v) is int for v in values) else None
        failures = [{"case": row["case"], "repeat": row["repetition"], "rubric": row["rubric"]}
                    for row in measured if row["passed"] is not True]
        result["profiles"][profile] = {
            "passed": sum(row["passed"] is True for row in measured),
            "observed": len(measured), "expected": len(cases) * 3,
            "cases_passed_all_repeats": sum(all(indexed.get((profile, case, repeat), {}).get("passed") is True
                for repeat in range(1, 4)) for case in case_ids),
            "categories": categories, "finishes": dict(finishes),
            "all_request_token_totals": token_totals, "failures": failures}
    if len(profiles) == 2:
        common = []
        for case in sorted(case_ids):
            for repeat in range(1, 4):
                pair = [indexed.get((profile, case, repeat)) for profile in profiles]
                if all(row is not None and row["passed"] is True for row in pair):
                    common.append(pair)
        totals = {}
        for side, profile in enumerate(profiles):
            counts = [pair[side]["metrics"].get("completion_tokens") for pair in common]
            totals[profile] = sum(counts) if counts and all(type(v) is int for v in counts) else None
        result["paired_successes"] = {"count": len(common), "completion_token_totals": totals,
            "warning": "conditional on both answers passing; use alongside all-request failure counts"}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.run.read_text().splitlines() if line.strip()]
    header, rows = records[0], records[1:]
    corpus = Path(__file__).resolve().parents[1] / "docs/benchmark-prompts/release-screen-v1/cases.json"
    raw = corpus.read_bytes()
    if (header["protocol"] != "release-screen-v1" or header["measured_repetitions"] != 3
            or header["warmups_per_case"] != 1 or header["corpus_sha256"] != hashlib.sha256(raw).hexdigest()):
        raise ValueError("not the frozen release-screen-v1 protocol")
    result = summarize(header, rows, json.loads(raw))
    print(json.dumps(result, indent=2))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
