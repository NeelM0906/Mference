import importlib.util
from pathlib import Path
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("screen", SCRIPTS / "qwen_efficiency_screen.py")
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)


class EfficiencyScreenTests(unittest.TestCase):
    def test_matched_request_controls(self):
        case = {"id": "tool", "prompt": "Echo", "tool": {"name": "echo"}}
        a, b = [screen.request(p, case, screen.SEEDS[0]) for p in screen.PROFILES]
        self.assertNotEqual(a.pop("model"), b.pop("model"))
        self.assertEqual(a, b)
        self.assertEqual(a["max_completion_tokens"], 4096)
        self.assertEqual(a["reasoning_effort"], "xhigh")
        self.assertEqual(a["tools"][0]["function"], case["tool"])

    def test_complete_failures_and_unknown_usage(self):
        cases = [{"id": "one"}]
        rows = [{"profile": p["label"], "case": "one", "seed": s, "passed": False,
                 "rubric": "truncated", "metrics": {"completion_tokens": 4096}}
                for p in screen.PROFILES for s in screen.SEEDS]
        summary = screen.summarize(rows, cases)
        self.assertTrue(summary["complete"])
        for result in summary["profiles"].values():
            self.assertEqual(result["passed"], 0)
            self.assertEqual(len(result["failures"]), 5)
            self.assertEqual(result["all_request_token_totals"]["completion_tokens"], 20480)
            self.assertIsNone(result["all_request_token_totals"]["reasoning_tokens"])
        self.assertFalse(screen.summarize(rows[:-1], cases)["complete"])
        with self.assertRaises(ValueError):
            screen.summarize(rows + [rows[0]], cases)


if __name__ == "__main__":
    unittest.main()
