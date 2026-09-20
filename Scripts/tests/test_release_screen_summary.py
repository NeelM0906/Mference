import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from release_screen_summary import summarize


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.header = {"profiles": [{"label": "base"}, {"label": "swift"}]}
        self.cases = [{"id": "one", "category": "reasoning"}]
        self.rows = [{"profile": profile, "case": "one", "category": "reasoning", "repetition": repeat,
                      "warmup": repeat == 0, "passed": True, "rubric": "exact",
                      "metrics": {"completion_tokens": 10, "reasoning_tokens": 8, "visible_tokens": 1},
                      "response": {"choices": [{"finish_reason": "stop"}]}}
                     for profile in ("base", "swift") for repeat in range(4)]

    def test_complete_counts_exclude_warmup_not_failures(self):
        self.rows[-1].update(passed=False, rubric="truncated", response={"choices": [{"finish_reason": "length"}]})
        result = summarize(self.header, self.rows, self.cases)
        self.assertTrue(result["complete"])
        swift = result["profiles"]["swift"]
        self.assertEqual((swift["passed"], swift["observed"], swift["expected"]), (2, 3, 3))
        self.assertEqual(swift["all_request_token_totals"]["completion_tokens"], 30)
        self.assertEqual(swift["cases_passed_all_repeats"], 0)
        self.assertEqual(result["paired_successes"]["count"], 2)

    def test_partial_is_not_complete_and_keeps_expected_denominator(self):
        result = summarize(self.header, self.rows[:-1], self.cases)
        self.assertFalse(result["complete"])
        self.assertEqual(result["profiles"]["swift"]["expected"], 3)

    def test_unknown_usage_not_zero(self):
        self.rows[-1]["metrics"] = {}
        result = summarize(self.header, self.rows, self.cases)
        self.assertIsNone(result["profiles"]["swift"]["all_request_token_totals"]["completion_tokens"])
        self.assertIsNone(result["paired_successes"]["completion_token_totals"]["swift"])

    def test_duplicate_and_mislabeled_rows_rejected(self):
        with self.assertRaises(ValueError):
            summarize(self.header, self.rows + [self.rows[0]], self.cases)
        bad = copy.deepcopy(self.rows)
        bad[0]["warmup"] = False
        with self.assertRaises(ValueError):
            summarize(self.header, bad, self.cases)


if __name__ == "__main__":
    unittest.main()
