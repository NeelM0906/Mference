#!/usr/bin/env python3
"""Self-test for g5_server.py (stdlib unittest, no model, no server):
python3 -I Scripts/parity/inkling2/test_g5_server.py"""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g5_server as g  # noqa: E402

USAGE = {"prompt_tokens": 20, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 3},
         "completion_tokens_details": {"reasoning_tokens": 2}}


def nonstream(content="Red", reasoning=None, calls=None, finish="stop"):
    m = {"role": "assistant", "content": content}
    if reasoning is not None:
        m["reasoning_content"] = reasoning
    if calls:
        m["tool_calls"] = calls
    return json.dumps({"id": "x", "created": 1, "choices": [{"message": m, "finish_reason": finish}], "usage": USAGE})


def sse(events):
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"


class NormalizeTests(unittest.TestCase):
    def test_stream_against_its_twin(self):
        call = [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{\"a\": 1, \"b\": 2}"}}]
        a = g.normalize(200, nonstream(content="", reasoning="think", calls=call, finish="tool_calls"), False)
        b = g.normalize(200, sse([
            {"choices": [{"delta": {"reasoning_content": "th"}}]},
            {"choices": [{"delta": {"reasoning_content": "ink"}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "z", "function": {"name": "f", "arguments": "{\"a\": 1,"}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": " \"b\": 2}"}}]}, "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": USAGE}]), True)
        self.assertEqual(g.loose(a), g.loose(b))
        self.assertEqual(a["content"], "")  # kept exactly for A-against-B
        self.assertIsNone(b["content"])
        self.assertTrue(b["done"])
        self.assertEqual(b["tool_calls"][0]["index"], 0)
        self.assertEqual(a["tool_calls_raw"], b["tool_calls_raw"])

    def test_stream_details_that_must_not_compare_equal(self):
        base_events = [{"choices": [{"delta": {"content": "Hi"}, "finish_reason": "stop"}]}]
        ok = g.normalize(200, sse(base_events), True)
        no_done = g.normalize(200, "".join(f"data: {json.dumps(e)}\n\n" for e in base_events), True)
        self.assertNotEqual(ok, no_done)
        frame = g.normalize(200, sse(base_events + [{"object": "error", "message": "x"}]), True)
        self.assertNotEqual(ok, frame)
        self.assertEqual(ok, g.normalize(200, ": ping\n\n" + sse(base_events) + ": ping\n\n", True))  # keepalives are timing
        junk = g.normalize(200, sse(base_events).replace("data: [DONE]", "data: {not json\ndata: [DONE]"), True)
        self.assertEqual(junk["unparsed_lines"], ["{not json"])
        tc = lambda extra: [{"choices": [{"delta": {"tool_calls": [dict({"function": {"name": "f", "arguments": "{}"}}, **extra)]}}]}]
        self.assertNotEqual(g.normalize(200, sse(tc({"index": 0})), True), g.normalize(200, sse(tc({})), True))
        # U+2028 inside a JSON string must not split the line
        sep = g.normalize(200, "data: " + json.dumps({"choices": [{"delta": {"content": "a\u2028b"}}]}, ensure_ascii=False) + "\n\ndata: [DONE]\n\n", True)
        self.assertEqual(sep["content"], "a\u2028b")
        self.assertEqual(sep["unparsed_lines"], [])

    def test_errors(self):
        n = g.normalize(400, json.dumps({"error": {"type": "invalid_request_error", "param": "p", "code": "c"}}), False)
        self.assertEqual(n, {"status": 400, "error_type": "invalid_request_error", "error_param": "p", "error_code": "c"})
        self.assertEqual(g.normalize(500, "not json", False), {"status": 500, "error_raw": "not json"})
        self.assertEqual(g.normalize(500, json.dumps({"error": None}), False), {"status": 500, "error_raw": {"error": None}})
        self.assertNotEqual(g.normalize(500, json.dumps({"error": "a"}), False), g.normalize(500, json.dumps({"error": "b"}), False))

    def test_cases_per_family(self):
        ids = lambda f: [c for c, _ in g.cases(f)]
        self.assertIn("T1", ids("qwen36"))
        self.assertIn("E1", ids("maple"))
        self.assertNotIn("T1", ids("maple"))
        for f in ("qwen36", "maple", "minicpm5"):
            self.assertTrue({"P1", "P3", "K1", "K2", "U1", "R2", "L1", "L2", "S1", "S2"} <= set(ids(f)))
        k2 = dict(g.cases("qwen36"))["K2"]({"K1": {"content": "A river", "reasoning": "think"}})
        self.assertEqual(k2["messages"][-2], {"role": "assistant", "content": "A river", "reasoning_content": "think"})
        self.assertEqual(dict(g.cases("qwen36"))["K1"]["max_completion_tokens"], 16)


class CompareTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, name, rows):
        p = os.path.join(self.tmp, name)
        with open(p, "w") as f:
            for case, norm in rows:
                f.write(json.dumps({"model": "m", "family": "qwen36", "case": case, "request": {"x": 1}, "norm": norm}) + "\n")
        return p

    def full(self, name, override=None):
        n = g.normalize(200, nonstream(), False)
        return self.write(name, [(c, (override or {}).get(c, n)) for c, _ in g.cases("qwen36")])

    def run_compare(self, a, b, fams=("qwen36",)):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = g.compare(a, b, list(fams))
        return rc, out.getvalue()

    def test_same_different_missing(self):
        n = g.normalize(200, nonstream(), False)
        a = self.full("a.jsonl")
        self.assertEqual(self.run_compare(a, self.full("b.jsonl"))[0], 0)
        rc, out = self.run_compare(a, self.full("c.jsonl", {"P2": dict(n, content="Blue")}))
        self.assertEqual(rc, 1)
        self.assertIn("| m | P2 | DIFFER | content |", out)
        part = self.write("d.jsonl", [("P1", n)])
        self.assertEqual(self.run_compare(a, part)[0], 2)
        # a case missing on both sides still counts: both runs stopped after P1
        rc, out = self.run_compare(part, self.write("e.jsonl", [("P1", n)]))
        self.assertEqual(rc, 2)
        self.assertIn("MISSING on both sides", out)
        # no rows at all is not a pass
        self.assertEqual(self.run_compare(self.write("f.jsonl", []), self.write("g.jsonl", []))[0], 2)

    def test_a_planned_family_with_no_rows_counts(self):
        a, b = self.full("a.jsonl"), self.full("b.jsonl")
        rc, out = self.run_compare(a, b, ("qwen36", "maple"))
        self.assertEqual(rc, 2)
        self.assertIn("families with no rows: maple", out)
        with self.assertRaises(SystemExit):
            self.run_compare(a, b, ("qwen99",))

    def test_odd_frames_do_not_crash(self):
        ev = [{"choices": [{"delta": {"tool_calls": [{"index": None, "function": {"name": "f", "arguments": "{}"}},
                                                     {"index": 0, "function": {"name": "g", "arguments": "{}"}}]}}]},
              {"choices": [], "usage": [1, 2]}, {"choices": [], "error": {"type": "x"}}]
        n = g.normalize(200, sse(ev), True)
        self.assertEqual(len(n["tool_calls"]), 2)
        self.assertIn({"choices": [], "error": {"type": "x"}}, n["other_frames"])
        self.assertIn({"<not an object>": "[1, 2]"}, n["usage_frames"])
        self.assertEqual(g.normalize(200, json.dumps({"choices": [{"message": {"content": "x"}}], "usage": "u"}), False)["usage"],
                         {"<not an object>": "\"u\""})

    def test_duplicate_rows_are_refused(self):
        n = g.normalize(200, nonstream(), False)
        with self.assertRaises(SystemExit):
            self.run_compare(self.write("a.jsonl", [("P1", n), ("P1", n)]), self.full("b.jsonl"))

    def test_arguments_string_only_is_its_own_line(self):
        call = lambda args: [{"id": "c", "type": "function", "function": {"name": "f", "arguments": args}}]
        x = g.normalize(200, nonstream(content=None, calls=call("{\"a\":1,\"b\":2}"), finish="tool_calls"), False)
        y = g.normalize(200, nonstream(content=None, calls=call("{\"b\": 2, \"a\": 1}"), finish="tool_calls"), False)
        rc, out = self.run_compare(self.full("a.jsonl", {"U1": x}), self.full("b.jsonl", {"U1": y}))
        self.assertEqual(rc, 1)
        self.assertIn("DIFFER (arguments string only; parsed JSON equal)", out)
        self.assertIn("1 differ (1 in the tool-call arguments string only)", out)


if __name__ == "__main__":
    unittest.main()
