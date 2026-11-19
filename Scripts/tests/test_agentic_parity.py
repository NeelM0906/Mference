"""Tests for Scripts/native/agentic/agentic_parity.py: the checker and the
parity verdict. No model, no network, no machine lock."""
import importlib.util
import json
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("agentic_parity", SCRIPTS / "native/agentic/agentic_parity.py")
ap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ap)

TASK = {t["id"]: t for t in ap.TASKS}


def call(name, **args):
    return {"content": "", "tool_calls": [{"name": name, "arguments": json.dumps(args)}]}


def say(text):
    return {"content": text, "tool_calls": []}


class SuiteTests(unittest.TestCase):
    def test_suite_is_frozen(self):
        self.assertEqual(ap.SUITE, "agentic-v1")
        ids = [t["id"] for t in ap.TASKS]
        self.assertEqual(len(ids), 32)
        self.assertEqual(len(set(ids)), 32)
        kinds = {t["check"]["kind"] for t in ap.TASKS}
        self.assertEqual(kinds, {"tool_call", "no_tool", "json", "content"})

    def test_bodies_are_greedy(self):
        b = ap.body_for(TASK["weather-plain"], "quick")
        self.assertEqual((b["temperature"], b["top_k"]), (0, 1))
        self.assertIn("tools", b)
        self.assertNotIn("tools", ap.body_for(TASK["json-city"], "quick"))


class ScoreTests(unittest.TestCase):
    def test_tool_call(self):
        t = TASK["convert-decimal"]
        self.assertTrue(ap.score(t, call("convert_currency", amount=19.99, **{"from": "gbp", "to": "JPY"}))[0])
        self.assertFalse(ap.score(t, call("convert_currency", amount=20, **{"from": "GBP", "to": "JPY"}))[0])
        self.assertFalse(ap.score(t, call("get_weather", city="x"))[0])
        self.assertFalse(ap.score(t, say("about 3700 yen"))[0])
        bad = {"content": "", "tool_calls": [{"name": "convert_currency", "arguments": "{oops"}]}
        self.assertEqual(ap.score(t, bad), (False, "arguments are not JSON"))
        for not_object in ['["GBP"]', "null", "3"]:
            r = {"content": "", "tool_calls": [{"name": "convert_currency", "arguments": not_object}]}
            self.assertEqual(ap.score(t, r), (False, "arguments are not a JSON object"))

    def test_optional_arguments_and_contains(self):
        self.assertTrue(ap.score(TASK["weather-plain"], call("get_weather", city="Oslo"))[0])
        email = TASK["email-basic"]
        self.assertTrue(ap.score(email, call("send_email", to="ana@example.com", subject="Lunch",
                                             body="See you at noon!"))[0])
        self.assertFalse(ap.score(email, call("send_email", to="ana@example.com", subject="Lunch",
                                              body="See you later"))[0])

    def test_calculator_checks_the_value(self):
        t = TASK["calc-basic"]
        self.assertTrue(ap.score(t, call("calculate", expression="(17*23)+4"))[0])
        self.assertTrue(ap.score(t, call("calculate", expression="17 * 23 + 4"))[0])
        self.assertFalse(ap.score(t, call("calculate", expression="17 * (23 + 4)"))[0])
        self.assertFalse(ap.score(t, call("calculate", expression="__import__('os')"))[0])

    def test_safe_eval(self):
        self.assertEqual(ap._safe_eval("1200 / 16"), 75)
        self.assertEqual(ap._safe_eval("-(2+3)*4"), -20)
        self.assertIsNone(ap._safe_eval("2 ** 8"))
        self.assertIsNone(ap._safe_eval("1/0"))

    def test_no_tool(self):
        self.assertTrue(ap.score(TASK["no-tool-fact"], say("Au"))[0])
        self.assertFalse(ap.score(TASK["no-tool-fact"], call("search_files", pattern="gold"))[0])
        self.assertTrue(ap.score(TASK["result-weather"], say("It is 18 °C with light rain in Lisbon."))[0])
        self.assertFalse(ap.score(TASK["result-weather"], say("It is mild."))[0])
        self.assertFalse(ap.score(TASK["no-tool-greeting"], say(""))[0])

    def test_json_and_thinking(self):
        t = TASK["json-extract"]
        self.assertTrue(ap.score(t, say('{"item": "Oak dowels", "qty": 12, "rush": false}'))[0])
        self.assertTrue(ap.score(t, say('<think>hmm</think>```json\n{"item": "oak dowels", "qty": 12, "rush": false}\n```'))[0])
        self.assertFalse(ap.score(t, say('{"item": "oak dowels", "qty": 12, "rush": true}'))[0])
        self.assertFalse(ap.score(t, say("12 oak dowels"))[0])

    def test_content(self):
        self.assertTrue(ap.score(TASK["exact-math"], say("12"))[0])
        self.assertTrue(ap.score(TASK["exact-math"], say("12."))[0])
        self.assertFalse(ap.score(TASK["exact-math"], say("The answer is 12"))[0])
        self.assertTrue(ap.score(TASK["exact-letter"], say("'g'"))[0])

    def test_error(self):
        self.assertEqual(ap.score(TASK["exact-math"], {"error": "boom"}), (False, "error: boom"))


class RunTests(unittest.TestCase):
    """run_tasks against a fake post_chat with engine_validate's real shape."""

    def test_unpacks_post_chat_and_scores_every_task(self):
        ev_spec = importlib.util.spec_from_file_location("engine_validate", SCRIPTS / "engine_validate.py")
        ev = importlib.util.module_from_spec(ev_spec)
        ev_spec.loader.exec_module(ev)
        body = {"choices": [{"message": {"content": "12", "tool_calls": []}, "finish_reason": "stop"}]}

        def post(_body):
            return 200, body, json.dumps(body), 0.01  # (status, parsed, raw, elapsed)

        report = ap.run_tasks(post, ev.canonical_from_message, "quick")
        self.assertEqual(len(report), 32)
        self.assertTrue(report["exact-math"]["passed"])
        self.assertFalse(report["weather-plain"]["passed"])

    def test_http_error_and_bad_arguments_fail_their_task_only(self):
        def post(b):
            if b["messages"][0]["content"].startswith("What's the weather in Oslo"):
                return 500, None, "", 0.0
            return 200, {"choices": [{"message": {"content": "", "tool_calls": [
                {"function": {"name": "get_weather", "arguments": '["Oslo"]'}}]}}]}, "", 0.0

        canonical = lambda parsed: {"content": "", "tool_calls": [
            {"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
            for c in parsed["choices"][0]["message"]["tool_calls"]]}
        saved = []
        report = ap.run_tasks(post, canonical, "quick", on_task=lambda r: saved.append(len(r)))
        self.assertEqual(saved, list(range(1, 33)))  # a partial report after every task
        self.assertEqual(report["weather-plain"]["reason"], "error: HTTP 500")
        self.assertEqual(report["weather-unit"]["reason"], "arguments are not a JSON object")
        self.assertEqual(len(report), 32)


    def test_a_request_error_fails_its_task_only(self):
        def post(b):
            if b["messages"][0]["content"].startswith("What's the weather in Oslo"):
                raise TimeoutError("timed out")
            return 200, {"choices": [{"message": {"content": "12", "tool_calls": []}}]}, "", 0.0

        report = ap.run_tasks(post, lambda p: {"content": "12", "tool_calls": []}, "quick")
        self.assertTrue(report["weather-plain"]["reason"].startswith("error: request failed"))
        self.assertTrue(report["exact-math"]["passed"])


class CompareTests(unittest.TestCase):
    def report(self, failing=(), tokens=None):
        return {t["id"]: {"passed": t["id"] not in failing,
                          "result": {"prompt_tokens": (tokens or {}).get(t["id"], 100)}} for t in ap.TASKS}

    def test_identical(self):
        v = ap.compare(self.report(["exact-math"]), self.report(["exact-math"]))
        self.assertTrue(v["parity"])
        self.assertEqual((v["regressions"], v["improvements"]), ([], []))

    def test_regressions_are_bounded_not_net(self):
        self.assertTrue(ap.compare(self.report(), self.report(["exact-math"]))["parity"])
        v = ap.compare(self.report(), self.report(["exact-math", "json-bool"]))
        self.assertFalse(v["parity"])
        self.assertEqual(sorted(v["regressions"]), ["exact-math", "json-bool"])
        # two regressions are not cancelled by two improvements
        v = ap.compare(self.report(["calc-basic", "email-basic"]), self.report(["exact-math", "json-bool"]))
        self.assertEqual(v["golden_passed"], v["candidate_passed"])
        self.assertFalse(v["parity"])
        self.assertEqual(sorted(v["improvements"]), ["calc-basic", "email-basic"])
        # the candidate doing better is never a failure
        self.assertTrue(ap.compare(self.report(["exact-math", "json-bool"]), self.report())["parity"])

    def test_missing_task_fails(self):
        c = self.report()
        del c["exact-math"]
        self.assertFalse(ap.compare(self.report(), c)["parity"])

    def test_request_errors_are_setup_errors_never_parity(self):
        # golden server down: every golden task errored, so nothing was measured
        down = {t["id"]: {"passed": False, "result": {"error": "request failed: ConnectionRefusedError()"}}
                for t in ap.TASKS}
        v = ap.compare(down, down)
        self.assertFalse(v["parity"])
        self.assertEqual(len(v["setup_errors"]["golden"]), 32)
        # a single HTTP error on the candidate side also blocks the verdict
        c = self.report()
        c["exact-math"] = {"passed": False, "result": {"error": "HTTP 500"}}
        v = ap.compare(self.report(), c)
        self.assertEqual(v["setup_errors"], {"golden": [], "candidate": ["exact-math"]})
        self.assertEqual(v["regressions"], [])
        self.assertFalse(v["parity"])
        # an "error" key, even null, is not a model answer
        c["exact-math"] = {"passed": False, "result": {"error": None}}
        self.assertEqual(ap.compare(self.report(), c)["setup_errors"]["candidate"], ["exact-math"])

    def test_render_mismatch_is_a_setup_error_not_a_flip(self):
        v = ap.compare(self.report(), self.report(["json-bool"], tokens={"json-bool": 101}))
        self.assertEqual(v["render_mismatch"], ["json-bool"])
        self.assertEqual(v["regressions"], [])
        self.assertFalse(v["parity"])


if __name__ == "__main__":
    unittest.main()
