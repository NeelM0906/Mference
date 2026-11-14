import importlib.util
import json
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "server_stream_bench.py"
spec = importlib.util.spec_from_file_location("server_stream_bench", SCRIPT)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def sse(chunks):
    lines = [b": keep-alive\n", b"\n"]
    for c in chunks:
        lines.append(b"data: " + json.dumps(c).encode() + b"\n")
        lines.append(b"\n")
    lines.append(b"data: [DONE]\n")
    lines.append(b"data: {\"ignored\": true}\n")
    return lines


class FakeClock:
    def __init__(self, times):
        self.times = list(times)

    def __call__(self):
        return self.times.pop(0)


class ServerStreamBenchTests(unittest.TestCase):
    def test_parse_stops_at_done(self):
        events = list(bench.parse_sse_lines(sse([{"a": 1}, {"b": 2}])))
        self.assertEqual(events, [{"a": 1}, {"b": 2}])

    def test_summarize_measures_first_delta_and_decode_window(self):
        chunks = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"reasoning_content": "think"}}]},
            {"choices": [{"delta": {"content": "Hel"}}]},
            {"choices": [{"delta": {"content": "lo"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 5,
                                      "prompt_tokens_details": {"cached_tokens": 0}},
             "timings": {"prompt_n": 100}},
        ]
        # One clock read per chunk, then one at end of stream.
        clock = FakeClock([10.5, 12.0, 13.0, 14.0, 14.1, 14.2, 14.3])
        rec = bench.summarize_stream(bench.parse_sse_lines(sse(chunks)), 10.0, clock)
        self.assertAlmostEqual(rec["ttft_s"], 2.0)
        self.assertAlmostEqual(rec["decode_window_s"], 2.0)
        self.assertAlmostEqual(rec["decode_tok_s"], 2.0)
        self.assertAlmostEqual(rec["total_s"], 4.3)
        self.assertEqual(rec["prompt_tokens"], 100)
        self.assertEqual(rec["cached_tokens"], 0)
        self.assertEqual(rec["finish_reason"], "stop")
        self.assertEqual(rec["content"], "Hello")
        self.assertEqual(rec["server_timings"], {"prompt_n": 100})

    def test_summarize_rejects_empty_stream(self):
        with self.assertRaises(RuntimeError):
            bench.summarize_stream(iter([]), 0.0, FakeClock([1.0]))

    def test_summarize_raises_on_error_chunk(self):
        with self.assertRaises(RuntimeError):
            bench.summarize_stream(iter([{"error": {"message": "x"}}]), 0.0,
                                   FakeClock([1.0, 2.0]))

    def test_refuses_non_loopback_host(self):
        with self.assertRaises(SystemExit):
            bench.main(["--base-url", "http://192.0.2.1:8080", "--model", "m",
                        "--case", "x.json", "--out", "/dev/null"])

    def test_stats_median_min_max(self):
        self.assertEqual(bench.stats([3, 1, 2, None]),
                         {"median": 2, "min": 1, "max": 3, "n": 3})
        self.assertIsNone(bench.stats([None]))


if __name__ == "__main__":
    unittest.main()
