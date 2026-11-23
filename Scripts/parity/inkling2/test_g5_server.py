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
    def test_stream_reduces_to_the_non_stream_fields(self):
        call = [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{\"a\": 1, \"b\": 2}"}}]
        a = g.normalize(200, nonstream(content="", reasoning="think", calls=call, finish="tool_calls"), False)
        b = g.normalize(200, sse([
            {"choices": [{"delta": {"reasoning_content": "th"}}]},
            {"choices": [{"delta": {"reasoning_content": "ink"}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "z", "function": {"name": "f", "arguments": "{\"a\": 1,"}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": " \"b\": 2}"}}]}, "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": USAGE}]), True)
        b.pop("stream_errors")
        self.assertEqual(a, b)
        self.assertIsNone(a["content"])  # "" and null both mean no text
        self.assertEqual(a["tool_calls_raw"], ["{\"a\": 1, \"b\": 2}"])
        self.assertEqual(a["usage"]["prompt_tokens_details.cached_tokens"], 3)

    def test_errors(self):
        n = g.normalize(400, json.dumps({"error": {"type": "invalid_request_error", "param": "p", "code": "c"}}), False)
        self.assertEqual(n, {"status": 400, "error_type": "invalid_request_error", "error_param": "p", "error_code": "c"})
        self.assertEqual(g.normalize(500, "not json", False)["error_type"], "<not JSON>")

    def test_cases_per_family(self):
        ids = lambda f: [c for c, _ in g.cases(f)]
        self.assertIn("T1", ids("qwen36"))
        self.assertIn("E1", ids("maple"))
        self.assertNotIn("T1", ids("maple"))
        for f in ("qwen36", "maple", "minicpm5"):
            self.assertTrue({"P1", "P3", "U1", "R2", "L1", "L2", "S1", "S2"} <= set(ids(f)))


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

    def run_compare(self, a, b):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = g.compare(a, b)
        return rc, out.getvalue()

    def test_same_different_missing(self):
        n = g.normalize(200, nonstream(), False)
        a = self.write("a.jsonl", [("P1", n), ("P2", n)])
        self.assertEqual(self.run_compare(a, self.write("b.jsonl", [("P1", n), ("P2", n)]))[0], 0)
        m = dict(n, content="Blue")
        rc, out = self.run_compare(a, self.write("c.jsonl", [("P1", n), ("P2", m)]))
        self.assertEqual(rc, 1)
        self.assertIn("| m | P2 | DIFFER | content |", out)
        self.assertEqual(self.run_compare(a, self.write("d.jsonl", [("P1", n)]))[0], 2)

    def test_arguments_string_only_is_its_own_line(self):
        call = lambda args: [{"id": "c", "type": "function", "function": {"name": "f", "arguments": args}}]
        x = g.normalize(200, nonstream(content=None, calls=call("{\"a\":1,\"b\":2}"), finish="tool_calls"), False)
        y = g.normalize(200, nonstream(content=None, calls=call("{\"b\": 2, \"a\": 1}"), finish="tool_calls"), False)
        rc, out = self.run_compare(self.write("a.jsonl", [("U1", x)]), self.write("b.jsonl", [("U1", y)]))
        self.assertEqual(rc, 1)
        self.assertIn("DIFFER (arguments string only; parsed JSON equal)", out)
        self.assertIn("1 differ (1 in the tool-call arguments string only)", out)


if __name__ == "__main__":
    unittest.main()
