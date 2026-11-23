#!/usr/bin/env python3
"""Summarize the Ask about your week log (internal testing).

The app writes one JSON line per question to
~/Library/Logs/Mference/week-ask.jsonl while the WeekAskLog default is on:

    defaults write ai.mference.app WeekAskLog -bool true
    defaults write ai.mference.app WeekAskReview warn   # or off (mode A)

Usage:
    App/Scripts/week_ask_report.py [LOG] [--last N] [--since ISO8601]

Prints totals by model, source and review mode, how often answers carried a
time warning or were cut off, latency and token figures, then the last N
questions with what was shown. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import TypedDict

DEFAULT_LOG = Path.home() / "Library" / "Logs" / "Mference" / "week-ask.jsonl"


class Entry(TypedDict, total=False):
    """One logged question (fields written by WeekAskLogEntry)."""

    time: str
    review: str
    model: str
    question: str
    historyCount: int
    format: str
    schedule: str
    reply: str
    shown: str
    cited: list[str]
    warnings: list[str]
    source: str
    wasCut: bool
    latencyMilliseconds: int
    promptTokens: int | None
    completionTokens: int | None
    error: str | None


def read_entries(path: Path, since: str | None = None) -> tuple[list[Entry], int]:
    """Read the log and its rotated `.1` file, oldest first.

    Returns the entries and the number of lines that could not be parsed.
    """
    entries: list[Entry] = []
    bad = 0
    for candidate in (path.with_name(path.name + ".1"), path):
        if not candidate.exists():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entry: Entry = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if since and entry.get("time", "") < since:
                continue
            entries.append(entry)
    entries.sort(key=lambda entry: entry.get("time", ""))
    return entries, bad


def percentile(values: list[int], fraction: float) -> int:
    """Nearest-rank percentile of a non-empty list."""
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(fraction * len(ordered) + 0.5) - 1))
    return ordered[index]


def summary(entries: list[Entry]) -> list[str]:
    """The totals block, one string per line."""
    total = len(entries)
    lines = [f"Questions: {total}"]
    if total == 0:
        return lines
    for label, key in (("Models", "model"), ("Sources", "source"), ("Review modes", "review"), ("Formats", "format")):
        counts = Counter(str(entry.get(key, "?")) for entry in entries)
        lines.append(f"{label}: " + ", ".join(f"{name} {count}" for name, count in counts.most_common()))
    answered = [entry for entry in entries if entry.get("source") == "model"]
    warned = sum(1 for entry in answered if entry.get("warnings"))
    cut = sum(1 for entry in answered if entry.get("wasCut"))
    errors = sum(1 for entry in entries if entry.get("error"))
    if answered:
        lines.append(f"Model answers with a time warning: {warned}/{len(answered)} ({100 * warned / len(answered):.0f}%)")
        lines.append(f"Model answers cut off: {cut}/{len(answered)}")
    lines.append(f"Errors: {errors}")
    latencies = [int(entry["latencyMilliseconds"]) for entry in answered if "latencyMilliseconds" in entry]
    if latencies:
        lines.append(
            f"Latency ms: median {int(statistics.median(latencies))}, p95 {percentile(latencies, 0.95)}, "
            f"max {max(latencies)}"
        )
    completions = [int(entry["completionTokens"]) for entry in answered if entry.get("completionTokens") is not None]
    prompts = [int(entry["promptTokens"]) for entry in answered if entry.get("promptTokens") is not None]
    if prompts:
        lines.append(f"Prompt tokens: median {int(statistics.median(prompts))}, max {max(prompts)}")
    if completions:
        lines.append(f"Answer tokens: median {int(statistics.median(completions))}, max {max(completions)}")
    return lines


def recent(entries: list[Entry], count: int) -> list[str]:
    """The last `count` questions, each with what was shown and any warning."""
    lines: list[str] = []
    for entry in entries[-count:] if count > 0 else []:
        shown = " ".join(str(entry.get("shown", "")).split())
        if len(shown) > 200:
            shown = shown[:197] + "..."
        thread = f" (follow-up {entry.get('historyCount')})" if entry.get("historyCount") else ""
        lines.append(
            f"- {entry.get('time', '?')} {entry.get('model', '?')} [{entry.get('review', '?')}, {entry.get('format', 'lines')}] "
            f"{entry.get('latencyMilliseconds', '?')} ms{thread}"
        )
        lines.append(f"  Q: {entry.get('question', '')}")
        lines.append(f"  A: {shown}")
        for warning in entry.get("warnings") or []:
            lines.append(f"  ! {warning}")
        if entry.get("error"):
            lines.append(f"  error: {entry.get('error')}")
    return lines


def main(argv: list[str] | None = None) -> int:
    """Print the report; exit 1 when the log does not exist."""
    parser = argparse.ArgumentParser(description="Summarize the Ask about your week log.")
    parser.add_argument("log", nargs="?", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--last", type=int, default=10, help="questions to list (default 10)")
    parser.add_argument("--since", help="only entries at or after this ISO 8601 time")
    args = parser.parse_args(argv)
    path: Path = args.log.expanduser()
    if not path.exists() and not path.with_name(path.name + ".1").exists():
        print(f"No log at {path}. Turn it on: defaults write ai.mference.app WeekAskLog -bool true", file=sys.stderr)
        return 1
    entries, bad = read_entries(path, args.since)
    print("\n".join(summary(entries)))
    if bad:
        print(f"Unreadable lines skipped: {bad}")
    listed = recent(entries, args.last)
    if listed:
        print(f"\nLast {min(args.last, len(entries))}:")
        print("\n".join(listed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
