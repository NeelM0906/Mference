"""Tests for App/Scripts/week_ask_report.py (stdlib unittest)."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import week_ask_report  # noqa: E402


def entry(**fields: object) -> str:
    base = {
        "time": "2026-11-22T21:00:00Z",
        "review": "warn",
        "model": "Quick 4B",
        "question": "what do i have this week?",
        "historyCount": 0,
        "schedule": "{\"now\": \"Sunday 2026-11-22, 9 PM\"}",
        "reply": "{\"answer\": \"Thanksgiving is Thursday.\", \"refs\": [\"E1\"]}",
        "shown": "Thanksgiving is Thursday.",
        "cited": ["E1"],
        "warnings": [],
        "source": "model",
        "route": "answer",
        "jsonParsed": True,
        "wasCut": False,
        "latencyMilliseconds": 900,
        "promptTokens": 700,
        "completionTokens": 20,
        "error": None,
    }
    base.update(fields)
    return json.dumps(base)


class WeekAskReportTests(unittest.TestCase):
    def write(self, directory: Path, lines: list[str], rotated: list[str] | None = None) -> Path:
        path = directory / "week-ask.jsonl"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        if rotated is not None:
            (directory / "week-ask.jsonl.1").write_text("\n".join(rotated) + "\n", encoding="utf-8")
        return path

    def test_reads_current_and_rotated_logs_oldest_first_and_skips_bad_lines(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = self.write(Path(folder), [entry(time="2026-11-22T22:00:00Z"), "not json"],
                              rotated=[entry(time="2026-11-22T20:00:00Z")])
            entries, bad = week_ask_report.read_entries(path)
            self.assertEqual([e["time"] for e in entries], ["2026-11-22T20:00:00Z", "2026-11-22T22:00:00Z"])
            self.assertEqual(bad, 1)

    def test_summary_counts_warnings_cuts_sources_and_latency(self) -> None:
        entries = [json.loads(entry(latencyMilliseconds=1000)),
                   json.loads(entry(warnings=["“7 PM” isn't in your calendar."], latencyMilliseconds=3000)),
                   json.loads(entry(source="modelFailed", model="Quick 4B", error="boom", latencyMilliseconds=5))]
        text = "\n".join(week_ask_report.summary(entries))
        self.assertIn("Questions: 3", text)
        self.assertIn("Sources: model 2, modelFailed 1", text)
        self.assertIn("Model answers with a time warning: 1/2 (50%)", text)
        self.assertIn("Errors: 1", text)
        self.assertIn("Latency ms: median 2000, p95 3000, max 3000", text)

    def test_summary_counts_routes_json_parsing_and_latency_per_route(self) -> None:
        entries = [json.loads(entry(route="free_query", latencyMilliseconds=400)),
                   json.loads(entry(route="free_query", latencyMilliseconds=600)),
                   json.loads(entry(route="answer", latencyMilliseconds=2000)),
                   json.loads(entry(route="prose", jsonParsed=False, latencyMilliseconds=3000)),
                   json.loads(entry(source="failed", route=None, jsonParsed=None, error="emptyReply"))]
        text = "\n".join(week_ask_report.summary(entries))
        self.assertIn("Routes: free_query 2, answer 1, prose 1, None 1", text)
        self.assertIn("Replies parsed as JSON: 3/4 (75%)", text)
        self.assertIn("Latency ms, free_query: median 500 (2 replies)", text)
        self.assertIn("Latency ms, prose: median 3000 (1 replies)", text)
        listed = "\n".join(week_ask_report.recent(entries[:1], 1))
        self.assertIn("[warn, free_query]", listed)

    def test_since_filters_and_recent_lists_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = self.write(Path(folder), [entry(time="2026-11-21T00:00:00Z"),
                                             entry(time="2026-11-23T00:00:00Z", warnings=["note"])])
            entries, _ = week_ask_report.read_entries(path, since="2026-11-22")
            self.assertEqual(len(entries), 1)
            listed = "\n".join(week_ask_report.recent(entries, 5))
            self.assertIn("  ! note", listed)

    def test_main_reports_a_missing_log(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(week_ask_report.main([str(Path(folder) / "none.jsonl")]), 1)

    def test_main_prints_the_report(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = self.write(Path(folder), [entry()])
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(week_ask_report.main([str(path), "--last", "1"]), 0)
            self.assertIn("Q: what do i have this week?", output.getvalue())


if __name__ == "__main__":
    unittest.main()
