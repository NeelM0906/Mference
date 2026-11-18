"""Tests for Scripts/engine_validate.py and Scripts/engine-validate.sh.

No model, no network beyond a loopback fake server, no machine lock.
"""
import http.server
import importlib.util
import json
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("engine_validate", SCRIPTS / "engine_validate.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def product_list():
    return {"object": "list", "data": [
        {"id": ev.model(k)["id"], "object": "model", "created": 0, "owned_by": "mference",
         "engine": ev.model(k)["engine"], "max_model_len": 32768}
        for k in ev.PRODUCT_ORDER]}


class FakeServer:
    """A loopback OpenAI-shaped server with scripted replies and a swap log."""

    def __init__(self, log_path=None, stream_differs=False):
        self.log_path = log_path
        self.stream_differs = stream_differs
        self.resident = None
        self.requests = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _json(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/v1/models":
                    self._json(200, product_list())
                else:
                    self._json(200, {"status": "ok"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                outer._log_swap(body["model"])
                reply = outer.reply(body)
                if not body.get("stream"):
                    self._json(200, reply)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                message = reply["choices"][0]["message"]
                content = message.get("content") or ""
                if outer.stream_differs:
                    content += "!"
                chunks = [{"choices": [{"index": 0, "delta": {"role": "assistant"}}]}]
                for piece in [content[:3], content[3:]]:
                    if piece:
                        chunks.append({"choices": [{"index": 0, "delta": {"content": piece}}]})
                for i, call in enumerate(message.get("tool_calls") or []):
                    args = call["function"]["arguments"]
                    chunks.append({"choices": [{"index": 0, "delta": {"tool_calls": [
                        {"index": i, "id": call["id"], "type": "function",
                         "function": {"name": call["function"]["name"], "arguments": args[:5]}}]}}]})
                    chunks.append({"choices": [{"index": 0, "delta": {"tool_calls": [
                        {"index": i, "function": {"arguments": args[5:]}}]}}]})
                chunks.append({"choices": [{"index": 0, "delta": {},
                                            "finish_reason": reply["choices"][0]["finish_reason"]}]})
                chunks.append({"choices": [], "usage": reply["usage"],
                               "timings": {"prompt_per_second": 500.0,
                                           "predicted_per_second": 40.0}})
                for c in chunks:
                    self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def _log_swap(self, model_id):
        if not self.log_path or model_id == self.resident:
            return
        with open(self.log_path, "a") as f:
            if self.resident:
                f.write(f"[t] swap started from={self.resident} to={model_id}\n")
                f.write(f"[t] model released model={self.resident}\n")
            else:
                f.write(f"[t] swap started to={model_id}\n")
            f.write(f"[t] swap finished model={model_id} in 2.5s\n")
        self.resident = model_id

    def reply(self, body):
        text = body["messages"][-1].get("content") or ""
        prompt_tokens = 10 + len(text) // 4
        if body.get("tools") and body["messages"][-1]["role"] == "user":
            message = {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_x", "type": "function",
                 "function": {"name": "get_weather", "arguments": "{\"city\": \"Lisbon\"}"}}]}
            finish = "tool_calls"
        elif "JSON object" in text:
            message = {"role": "assistant",
                       "content": "{\"city\": \"Tokyo\", \"population_millions\": 37.4}"}
            finish = "stop"
        else:
            message = {"role": "assistant", "content": "A hash table maps keys to values."}
            finish = "stop"
        return {"id": "x", "object": "chat.completion", "model": body["model"],
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 7,
                          "total_tokens": prompt_tokens + 7}}

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


class CanonicalTests(unittest.TestCase):
    def test_stream_and_message_agree_on_split_tool_arguments(self):
        server = FakeServer()
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}],
                "tools": [ev.WEATHER_TOOL]}
        message = ev.canonical_from_message(server.reply(body))
        chunks = [
            {"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "function": {"name": "get_weather", "arguments": "{\"ci"}}]}}]},
            {"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": "ty\": \"Lisbon\"}"}}]}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 7}},
        ]
        stream = ev.canonical_from_chunks(chunks)
        self.assertEqual(ev.disagreements(message, stream), [])
        self.assertEqual(ev.digest(message), ev.digest(stream))

    def test_digest_changes_with_content(self):
        a = {"content": "x", "reasoning_content": "", "tool_calls": [], "finish_reason": "stop",
             "completion_tokens": 1}
        b = dict(a, content="y")
        self.assertNotEqual(ev.digest(a), ev.digest(b))
        self.assertEqual(ev.disagreements(a, b), ["content"])

    def test_error_chunk_is_not_agreement(self):
        stream = ev.canonical_from_chunks([{"error": {"message": "boom"}}])
        self.assertIn("error", ev.disagreements({"content": ""}, stream))


class ModelListTests(unittest.TestCase):
    def expected(self):
        return [ev.model(k) for k in ev.PRODUCT_ORDER]

    def test_exact_product_list_passes(self):
        self.assertEqual(ev.check_model_list(product_list(), self.expected()), [])

    def test_flash_next_and_missing_engine_fail(self):
        body = product_list()
        body["data"].append({"id": "qwen3.8-flash-next-int4g64", "engine": "mference",
                             "max_model_len": 1})
        del body["data"][0]["engine"]
        problems = ev.check_model_list(body, self.expected())
        self.assertTrue(any("forbidden" in p for p in problems))
        self.assertTrue(any("unexpected" in p for p in problems))
        self.assertTrue(any("engine=None" in p for p in problems))

    def test_registry_override(self):
        entry = ev.model("deep", environ={"ENGINE_VALIDATE_DEEP_ID": "deep-x"})
        self.assertEqual(entry["id"], "deep-x")
        with self.assertRaises(KeyError):
            ev.model("flashnext")


class PromptSetTests(unittest.TestCase):
    def test_ledger_is_deterministic(self):
        self.assertEqual(ev.ledger(80), ev.ledger(80))
        self.assertEqual(len(ev.ledger(80)), 80)

    def test_cases_cover_section_8(self):
        names = [c["name"] for c in ev.prompt_cases()]
        self.assertEqual(names, ["short-chat", "doc-2k", "tool-roundtrip", "json-answer"])

    def test_json_answer_check(self):
        self.assertTrue(ev.json_answer_ok('{"city": "Tokyo", "population_millions": 37}'))
        self.assertTrue(ev.json_answer_ok('```json\n{"city": "Tokyo", "population_millions": 37}\n```'))
        self.assertFalse(ev.json_answer_ok('```json\n{"city": "Tokyo"}\n```'))
        self.assertFalse(ev.json_answer_ok("not JSON"))


class EndToEndTests(unittest.TestCase):
    def test_prompts_models_compare_summary(self):
        with tempfile.TemporaryDirectory() as tmp, FakeServer() as server:
            root = Path(tmp)
            run_a = root / "quick" / "engine"
            run_b = root / "quick" / "upstream"
            for out in (run_a, run_b):
                rc = ev.main(["prompts", "--port", str(server.port), "--model", "qwen3.5-4b",
                              "--out", str(out)])
                self.assertEqual(rc, 0)
            report = json.loads((run_a / "prompts.json").read_text())
            self.assertTrue(report["all_agree"])
            tool = report["cases"]["tool-roundtrip"]
            self.assertEqual(len(tool), 2)
            self.assertTrue(tool[0]["tool_call_emitted"])
            follow = json.loads((run_a / "tool-roundtrip.turn2.request.json").read_text())
            self.assertEqual(follow["messages"][1]["tool_calls"][0]["id"], "call_0")
            self.assertTrue(report["cases"]["json-answer"][0]["json_valid"])
            self.assertTrue((run_a / "short-chat.turn1.stream.raw.sse").exists())

            self.assertEqual(ev.main(["models", "--port", str(server.port),
                                      "--out", str(root / "models.json")]), 0)
            self.assertEqual(ev.main(["compare", str(run_a), str(run_b), "--label",
                                      "quick-engine-vs-upstream",
                                      "--out", str(root / "identity-quick.json")]), 0)
            (run_a / "run.json").write_text(json.dumps({"load_s": 2.5, "swap_delta_mib": 0.0}))
            ev.main(["summary", str(root)])
            summary = (root / "summary.md").read_text()
            self.assertIn("| quick | engine | yes | yes | yes | 2.5 |", summary)
            self.assertIn("quick-engine-vs-upstream | yes", summary)

    def test_stream_disagreement_fails(self):
        with tempfile.TemporaryDirectory() as tmp, FakeServer(stream_differs=True) as server:
            rc = ev.main(["prompts", "--port", str(server.port), "--model", "m",
                          "--out", tmp])
            self.assertEqual(rc, 1)
            report = json.loads((Path(tmp) / "prompts.json").read_text())
            self.assertEqual(report["cases"]["short-chat"][0]["differs_in"], ["content"])

    def test_swap_chain_requires_release_before_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "server.log"
            log.write_text("")
            with FakeServer(log_path=log) as server:
                rc = ev.main(["swap", "--port", str(server.port), "--log", str(log),
                              "--out", str(Path(tmp) / "swap.json")])
            self.assertEqual(rc, 0)
            result = json.loads((Path(tmp) / "swap.json").read_text())
            self.assertEqual(len(result["steps"]), 6)
            self.assertEqual(result["steps"][1]["from"], "qwen3.5-4b")
            self.assertEqual(result["steps"][1]["load_s"], 2.5)

    def test_readings_scale_prompt_and_report_timings(self):
        with tempfile.TemporaryDirectory() as tmp, FakeServer() as server:
            out = Path(tmp) / "readings.json"
            rc = ev.main(["readings", "--port", str(server.port), "--model", "m",
                          "--runs", "1", "--targets", "512", "--out", str(out)])
            self.assertEqual(rc, 0)
            reading = json.loads(out.read_text())["readings"][0]
            self.assertEqual(reading["prefill_tok_s"], 500.0)
            self.assertEqual(reading["decode_tok_s"], 40.0)
            self.assertLess(abs(reading["prompt_tokens"] - 512), 64)


class GateTests(unittest.TestCase):
    def write_prompts(self, directory, md5, status=200):
        turn = {"turn": 1, "md5_nonstream": md5, "md5_stream": md5,
                "nonstream": {"http_status": status}, "stream": {"http_status": status}}
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "prompts.json").write_text(json.dumps({"cases": {"c": [turn]}}))

    def test_missing_side_is_not_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_prompts(Path(tmp) / "a", "x")
            out = Path(tmp) / "id.json"
            rc = ev.main(["identity", str(Path(tmp) / "a"), str(Path(tmp) / "b"),
                          "--label", "l", "--out", str(out)])
            self.assertEqual(rc, 1)
            self.assertEqual(json.loads(out.read_text())["missing"], ["b"])

    def test_identical_errors_are_not_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_prompts(Path(tmp) / "a", "x", status=400)
            self.write_prompts(Path(tmp) / "b", "x", status=400)
            rc = ev.main(["identity", str(Path(tmp) / "a"), str(Path(tmp) / "b"),
                          "--out", str(Path(tmp) / "id.json")])
            self.assertEqual(rc, 1)

    def test_speed_bar(self):
        engine = [{"target_tokens": 512, "prefill_tok_s": 96.0, "decode_tok_s": 40.0}]
        ref = [{"target_tokens": 512, "prefill_tok_s": 100.0, "decode_tok_s": 43.0}]
        rows = {r["metric"]: r for r in ev.speed_rows(engine, ref)}
        self.assertTrue(rows["prefill_tok_s"]["ok"])    # 0.96
        self.assertFalse(rows["decode_tok_s"]["ok"])    # 0.93
        with tempfile.TemporaryDirectory() as tmp:
            rc = ev.main(["speed", tmp, tmp, "--out", str(Path(tmp) / "s.json")])
            self.assertEqual(rc, 1)  # no readings on either side

    def test_run_record_and_swap_growth(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "server.log").write_text(
                "[t] swap finished model=m in 4.0s\n  52428800  maximum resident set size\n")
            for after, grew in (("10.00", 1), ("10.25", 0)):
                ev.main(["runrecord", tmp, "--key", "quick", "--variant", "engine",
                         "--swap-before", "10.00", "--swap-after", after, "--status", "0"])
                record = json.loads((Path(tmp) / "run.json").read_text())
                self.assertEqual(record["load_s"], 4.0)
                self.assertEqual(record["max_rss_bytes"], 52428800)
                self.assertEqual(ev.main(["swapgrew", str(Path(tmp) / "run.json")]), grew)
            ev.main(["runrecord", tmp, "--swap-before", "", "--swap-after", "1"])
            self.assertEqual(ev.main(["swapgrew", str(Path(tmp) / "run.json")]), 0)

    def test_maxrss_reads_engine_runs_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            for run, value in (("quick/engine", 2 << 30), ("deep/engine", 3 << 30),
                               ("deep/upstream", 9 << 30)):
                (Path(tmp) / run).mkdir(parents=True)
                (Path(tmp) / run / "run.json").write_text(json.dumps({"max_rss_bytes": value}))
            self.assertEqual(ev.max_single_rss_mib(tmp), 3072.0)

    def test_invalid_json_answer_fails_even_when_modes_agree(self):
        class Prose(FakeServer):
            def reply(self, body):
                reply = FakeServer.reply(self, body)
                if "JSON object" in body["messages"][-1].get("content", ""):
                    reply["choices"][0]["message"]["content"] = "not JSON"
                return reply
        with tempfile.TemporaryDirectory() as tmp, Prose() as server:
            rc = ev.main(["prompts", "--port", str(server.port), "--model", "m", "--out", tmp])
            self.assertEqual(rc, 1)
            case = json.loads((Path(tmp) / "prompts.json").read_text())["cases"]["json-answer"][0]
            self.assertEqual(case["differs_in"], ["invalid JSON answer"])

    def test_cache_served_readings_fail_the_speed_bar(self):
        engine = [{"target_tokens": 512, "prefill_tok_s": 900.0, "decode_tok_s": 50.0,
                   "cold_violation": True},
                  {"target_tokens": 512, "prefill_tok_s": 100.0, "decode_tok_s": 40.0,
                   "cold_violation": False}]
        ref = [{"target_tokens": 512, "prefill_tok_s": 100.0, "decode_tok_s": 40.0}]
        rows = ev.speed_rows(engine, ref)
        self.assertTrue(all(not r["ok"] and r["cold_violation"] for r in rows))
        self.assertEqual(rows[0]["engine"], 100.0)

    def test_tool_round_trip_without_a_tool_call_fails(self):
        class NoTools(FakeServer):
            def reply(self, body):
                body = dict(body, tools=None)
                return FakeServer.reply(self, body)
        with tempfile.TemporaryDirectory() as tmp, NoTools() as server:
            rc = ev.main(["prompts", "--port", str(server.port), "--model", "m", "--out", tmp])
            self.assertEqual(rc, 1)
            report = json.loads((Path(tmp) / "prompts.json").read_text())
            tool = report["cases"]["tool-roundtrip"]
            self.assertIn("no tool call", tool[0]["differs_in"])
            self.assertIn("skipped", tool[1])

    def test_swap_rss_bound(self):
        import os
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "server.log"
            log.write_text("")
            with FakeServer(log_path=log) as server:
                rc = ev.main(["swap", "--port", str(server.port), "--log", str(log),
                              "--pid", str(os.getpid()), "--chain", "quick,deep",
                              "--max-single-rss-mib", "1", "--out", str(Path(tmp) / "s.json")])
            self.assertEqual(rc, 1)
            step = json.loads((Path(tmp) / "s.json").read_text())["steps"][1]
            self.assertTrue(any("above one model's bound" in p for p in step["problems"]))


class TraceParityTests(unittest.TestCase):
    def test_cases_requests_and_comparison(self):
        with tempfile.TemporaryDirectory() as tmp, FakeServer() as server:
            root = Path(tmp)
            cases, trace = root / "cases", root / "trace"
            cases.mkdir()
            (cases / "m.json").write_text(json.dumps([{"role": "user", "content": "hi"}]))
            (cases / "tools.json").write_text(json.dumps([ev.WEATHER_TOOL]))
            (cases / "cases.tsv").write_text("chat\tm.json\t-\ttrue\ntool\tm.json\ttools.json\tfalse\n")
            rows = ev.trace_cases(cases)
            body = ev.trace_request("q", rows[0], 64)
            self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": True})
            self.assertNotIn("top_k", body)
            self.assertNotIn("tools", body)
            for name, completion in (
                    ("chat", {"content": "A hash table maps keys to values.", "reasoning": None,
                              "tool_calls": [], "finish_reason": "stop", "completion_tokens": 7}),
                    ("tool", {"content": "", "reasoning": None, "finish_reason": "tool_calls",
                              "completion_tokens": 7,
                              "tool_calls": [{"name": "get_weather",
                                              "arguments": "{\"city\": \"Lisbon\"}"}]})):
                (trace / name).mkdir(parents=True)
                (trace / name / "completion.json").write_text(json.dumps(completion))
            rc = ev.main(["traceparity", "--port", str(server.port), "--model", "q",
                          "--cases", str(cases), "--trace", str(trace), "--out", str(root / "out")])
            self.assertEqual(rc, 0)
            (trace / "chat" / "completion.json").write_text(json.dumps(
                {"content": "different", "finish_reason": "stop", "completion_tokens": 7}))
            rc = ev.main(["traceparity", "--port", str(server.port), "--model", "q", "--case", "chat",
                          "--cases", str(cases), "--trace", str(trace), "--out", str(root / "out2")])
            self.assertEqual(rc, 1)
            report = json.loads((root / "out2" / "traceparity.json").read_text())
            self.assertEqual(report["rows"][0]["differs_in"], ["content"])


class SwapLogTests(unittest.TestCase):
    def test_release_before_start_fails(self):
        events = ev.swap_events(
            "[t] model released model=a\n[t] swap started from=a to=b\n"
            "[t] swap finished model=b in 1.0s\n")
        self.assertFalse(ev.verify_swap(events, "a", "b")["ok"])

    def test_release_after_finish_fails(self):
        events = ev.swap_events(
            "[t] swap started from=a to=b\n[t] swap finished model=b in 1.0s\n"
            "[t] model released model=a\n")
        verdict = ev.verify_swap(events, "a", "b")
        self.assertFalse(verdict["ok"])
        self.assertIn("old model released after the new one finished loading", verdict["problems"])

    def test_missing_release_line_can_be_waived(self):
        events = ev.swap_events("[t] swap started from=a to=b\n[t] swap finished model=b in 1.0s\n")
        self.assertFalse(ev.verify_swap(events, "a", "b")["ok"])
        self.assertTrue(ev.verify_swap(events, "a", "b", require_release=False)["ok"])

    def test_failed_swap(self):
        events = ev.swap_events("[t] swap started to=b\n[t] swap failed model=b error=x\n")
        self.assertFalse(ev.verify_swap(events, None, "b")["ok"])


class ParsingTests(unittest.TestCase):
    def test_swapusage(self):
        self.assertEqual(ev.parse_swapusage(
            "total = 2048.00M  used = 1024.50M  free = 1023.50M  (encrypted)"), 1024.5)
        self.assertEqual(ev.parse_swapusage("total = 4.00G  used = 1.50G  free = 2.50G"), 1536.0)
        self.assertIsNone(ev.parse_swapusage("garbage"))

    def test_time_l(self):
        text = ("        5.00 real         1.00 user         0.50 sys\n"
                "          1234567890  maximum resident set size\n"
                "           987654321  peak memory footprint\n")
        self.assertEqual(ev.parse_time_l(text),
                         {"max_rss_bytes": 1234567890, "peak_footprint_bytes": 987654321})


class ShellTests(unittest.TestCase):
    SCRIPT = SCRIPTS / "engine-validate.sh"

    def test_syntax(self):
        subprocess.run(["bash", "-n", str(self.SCRIPT)], check=True)

    def test_dry_run_plans_locked_runs_without_running_anything(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = subprocess.run(
                ["bash", str(self.SCRIPT), "--dry-run", "--out", tmp,
                 "--baseline-server", "/nonexistent/main-server",
                 "--upstream-llama-server", "/nonexistent/llama-server",
                 "--upstream-mtp-args", "--spec-type draft-mtp",
                 "--mtp-off-env", "X=0"],
                check=True, capture_output=True, text=True).stdout
            self.assertEqual(list(Path(tmp).iterdir()), [])
        runs = [line for line in out.splitlines() if " run --quiet --wait 3600 ws3 " in line]
        self.assertEqual(len(runs), 13)
        self.assertTrue(all("_run" in line for line in runs))
        self.assertIn("deep engine-nomtp", out.replace("\\ ", " "))
        self.assertIn("qwen36 main", out.replace("\\ ", " "))
        self.assertNotIn("quick upstream-nomtp", out.replace("\\ ", " "))

    def run_script(self, *args, env=None):
        import os
        return subprocess.run(["bash", str(self.SCRIPT), *args], capture_output=True, text=True,
                              env=dict(os.environ, **(env or {})))

    def test_identity_without_references_refuses_to_start(self):
        result = self.run_script("--dry-run", "--models", "quick,qwen36")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--baseline-server (for qwen36)", result.stderr)
        self.assertIn("--upstream-llama-server (for quick)", result.stderr)
        ok = self.run_script("--dry-run", "--models", "quick,qwen36", "prompts")
        self.assertEqual(ok.returncode, 0, ok.stderr)

    def test_passthrough_cannot_leave_loopback(self):
        for flag, value in (("--server-args", "--bind tailnet"),
                            ("--upstream-args", "-c 4096 --host 0.0.0.0")):
            result = self.run_script("--dry-run", "prompts", flag, value)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("may not set --bind/--host", result.stderr)

    def test_summary_only_rebuild_keeps_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "meta.json").write_text(json.dumps({"commit": "original"}))
            result = self.run_script("--out", tmp, "summary",
                                     env={"ENGINE_VALIDATE_LOCK": "/usr/bin/true"})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((Path(tmp) / "meta.json").read_text()),
                             {"commit": "original"})

    def test_existing_results_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {"ENGINE_VALIDATE_LOCK": "/usr/bin/true"}
            first = self.run_script("--results-root", tmp, "summary", env=env)
            self.assertEqual(first.returncode, 0, first.stderr)
            second = self.run_script("--results-root", tmp, "summary", env=env)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("already exists", second.stderr)

    def test_swap_chain_option(self):
        ok = self.run_script("--dry-run", "swap", "--swap-chain", "qwen36,minicpm5,qwen36")
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertIn("swap\\ chain", ok.stdout)
        bad = self.run_script("--dry-run", "swap", "--swap-chain", "qwen36,flashnext")
        self.assertNotEqual(bad.returncode, 0)

    def test_trace_phase_plans_one_cold_server_per_case_and_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            cases = Path(tmp)
            (cases / "cases.tsv").write_text("a\tm.json\t-\tfalse\nb-c\tm.json\t-\ttrue\n")
            result = self.run_script("--dry-run", "--models", "quick", "--trace-cases", tmp,
                                     "--trace", f"quick={tmp}/trace", "trace")
            self.assertEqual(result.returncode, 0, result.stderr)
            runs = [l for l in result.stdout.splitlines() if "_run quick trace-" in l]
            self.assertEqual(len(runs), 5)
            self.assertTrue(any("trace-b-c-stream" in l for l in runs))
            self.assertTrue(runs[-1].rstrip().endswith("trace-warm"))
        self.assertNotEqual(self.run_script("--dry-run", "trace").returncode, 0)

    def test_unknown_model_is_rejected(self):
        result = subprocess.run(["bash", str(self.SCRIPT), "--dry-run", "--models", "flashnext"],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
