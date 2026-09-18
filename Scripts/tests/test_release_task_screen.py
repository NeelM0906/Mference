import collections
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Scripts"))
spec = importlib.util.spec_from_file_location("release_screen", ROOT / "Scripts/release_task_screen.py")
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)


def stream(events, done=True):
    lines = ["data: " + json.dumps(event) + "\n\n" for event in events]
    if done:
        lines.append("data: [DONE]\n\n")
    return io.BytesIO("".join(lines).encode())


def delta(value, finish=None):
    return {"choices": [{"index": 0, "delta": value, "finish_reason": finish}]}


class ReleaseScreenTests(unittest.TestCase):
    def test_corpus_is_sixty_unique_cases_in_six_equal_categories(self):
        cases = json.loads((ROOT / "docs/benchmark-prompts/release-screen-v1/cases.json").read_text())
        self.assertEqual(len(cases), 60)
        self.assertEqual(len({case["id"] for case in cases}), 60)
        counts = collections.Counter(case["category"] for case in cases)
        self.assertEqual(len(counts), 6)
        self.assertEqual(set(counts.values()), {10})
        for case in cases:
            self.assertEqual(sum(key in case for key in ("expected", "expected_json", "expected_call")), 1)
            if "expected_call" in case:
                message = {"tool_calls": [{"function": {"name": case["expected_call"]["name"],
                             "arguments": json.dumps(case["expected_call"]["arguments"])}}]}
                reason = "tool_calls"
            else:
                message = {"content": case.get("expected", json.dumps(case.get("expected_json")))}
                reason = "stop"
            response = {"choices": [{"message": message, "finish_reason": reason}]}
            self.assertTrue(screen.score(case, response)[0], case["id"])
            response["choices"][0]["finish_reason"] = "length"
            self.assertFalse(screen.score(case, response)[0], case["id"])

    def test_stream_separates_reasoning_visible_and_role_only_deltas(self):
        events = [delta({"role": "assistant"}), delta({"reasoning_content": "think"}),
                  delta({"content": "4"}), delta({}, "stop"),
                  {"choices": [], "usage": {"completion_tokens": 7,
                    "completion_tokens_details": {"reasoning_tokens": 6}}}]
        times = iter([1, 2, 3, 4, 5])
        result, metrics = screen.read_stream(stream(events), 0, lambda: next(times))
        self.assertEqual(result["choices"][0]["message"]["content"], "4")
        self.assertEqual(metrics["first_model_delta_seconds"], 2)
        self.assertEqual(metrics["first_visible_answer_seconds"], 3)
        self.assertEqual(metrics["completed_seconds"], 5)
        self.assertEqual(metrics["visible_tokens"], 1)
        self.assertEqual(metrics["reasoning_tokens"], 6)

    def test_fragmented_tool_call_and_missing_usage(self):
        events = [delta({"tool_calls": [{"index": 0, "id": "call-1", "function": {"name": "add", "arguments": '{"a":'}}]}),
                  delta({"tool_calls": [{"index": 0, "function": {"arguments": '2}'}}]}), delta({}, "tool_calls")]
        response, metrics = screen.read_stream(stream(events), 0, lambda: 1)
        call = response["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual(call["function"], {"name": "add", "arguments": '{"a":2}'})
        self.assertIsNone(metrics["first_visible_answer_seconds"])
        self.assertIsNone(metrics["reasoning_tokens"])
        self.assertIsNone(metrics["visible_tokens"])

    def test_incomplete_and_error_streams_are_not_successes(self):
        for reply in [stream([delta({"content": "4"}, "stop")], done=False),
                      stream([delta({"content": "4"})]), stream([{"error": "cancelled"}])]:
            with self.assertRaises(ValueError):
                screen.read_stream(reply, 0, lambda: 1)

    def test_bool_not_number_but_decimal_number_equivalent(self):
        self.assertFalse(screen.equivalent({"enabled": 0}, {"enabled": False}))
        self.assertFalse(screen.equivalent({"count": True}, {"count": 1}))
        self.assertTrue(screen.equivalent({"amount": 12.0}, {"amount": 12}))
        self.assertFalse(screen.equivalent({"a": 1, "extra": 0}, {"a": 1}))

    def test_timeout_is_recorded_failure(self):
        with patch.object(screen.urllib.request, "urlopen", side_effect=TimeoutError("timeout")):
            result = screen.run_case("http://127.0.0.1:18489/v1/chat/completions", {}, {"expected": "4"}, 1)
        self.assertFalse(result["passed"])
        self.assertIn("timeout", result["error"])


if __name__ == "__main__":
    unittest.main()
