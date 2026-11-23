#!/usr/bin/env python3
"""Self-test for agent_acceptance.py against a scripted fake loopback server.

    python3 -I Scripts/parity/inkling2/test_agent_acceptance.py
    AGENT_ACCEPTANCE_CORPUS=<corpus dir> python3 -I Scripts/parity/inkling2/test_agent_acceptance.py

No model, no network beyond 127.0.0.1. The fake server streams SSE the way
MferenceServer does (role, reasoning and content deltas, tool-call deltas, a
finish chunk, a usage chunk, [DONE]) from a queue of scripted replies. With
AGENT_ACCEPTANCE_CORPUS set, every block B to E case of the real corpus is
replayed twice: with a gold reply that must pass and a wrong one that must
fail, so each rubric is satisfiable and not trivially passed.
"""

import http.server
import importlib.util
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("agent_acceptance", HERE / "agent_acceptance.py")
AA = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AA)


class Fake(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), Handler)
        self.replies = []
        self.requests = []


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(body)
        r = self.server.replies.pop(0) if self.server.replies else {"content": "", "finish": "stop"}
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def send(obj):
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")

        send({"choices": [{"index": 0, "delta": {"role": "assistant"}}]})
        if r.get("reasoning"):
            send({"choices": [{"index": 0, "delta": {"reasoning_content": r["reasoning"]}}]})
        content = r.get("content", "")
        if content:
            half = len(content) // 2
            for part in (content[:half], content[half:]):
                if part:
                    send({"choices": [{"index": 0, "delta": {"content": part}}]})
        for i, tc in enumerate(r.get("tool_calls", [])):
            send({"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": i, "id": f"call_{i}", "type": "function",
                 "function": {"name": tc["name"], "arguments": tc["arguments"]}}]}}]})
        send({"choices": [{"index": 0, "delta": {}, "finish_reason": r.get("finish", "stop")}]})
        send({"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 20}})
        self.wfile.write(b"data: [DONE]\n\n")


def tool(name, **args):
    return {"name": name, "arguments": json.dumps(args)}


PROFILE = {"label": "p", "model": "m", "source_revision": "r"}


class Base(unittest.TestCase):
    def setUp(self):
        self.srv = Fake()
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/v1/chat/completions"

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()

    def run_case(self, block, case, replies):
        self.srv.replies = list(replies)
        return AA.run_case(self.url, block, case, PROFILE, 1, 30)


WEB_TOOLS = [{"type": "function", "function": {"name": "web_search", "parameters": {}}},
             {"type": "function", "function": {"name": "open_page", "parameters": {}}}]


def b_case(**kw):
    c = {"id": "x", "block": "B", "system": "sys", "tools": WEB_TOOLS, "messages": [{"role": "user", "content": "q"}],
         "fixtures": {"web_search": [{"when_any": ["(?i)norrby"], "results": [
             {"title": "Lib", "url": "https://n.example/lib", "snippet": "Saturday 10:00-16:00"}]}],
             "open_page": {"https://n.example/lib": {"title": "Lib", "text": "Saturday 10:00-16:00."}}},
         "check": {"required_calls": [{"name": "web_search", "args": {"query": "(?i)norrby"}}],
                   "final_must": ["10:00"], "max_rounds": 4}}
    c.update(kw)
    return c


class Unit(Base):
    def test_a_uses_screen_rubric(self):
        case = {"id": "everyday-units", "category": "everyday", "prompt": "km?", "expected": "2500"}
        r = self.run_case("A", case, [{"content": "2500"}])
        self.assertTrue(r["passed"])
        r = self.run_case("A", case, [{"content": "2,500"}])
        self.assertFalse(r["passed"])
        self.assertEqual(r["failure"], "wrong_answer")
        self.assertEqual(self.srv.requests[-1]["max_completion_tokens"], AA.CAPS["A"])
        self.assertNotIn("temperature", self.srv.requests[-1])  # product defaults: no sampling fields

    def test_b_loop_and_fixture_format(self):
        r = self.run_case("B", b_case(), [
            {"tool_calls": [tool("web_search", query="Norrby library hours")], "finish": "tool_calls"},
            {"tool_calls": [tool("open_page", url="https://n.example/lib")], "finish": "tool_calls"},
            {"reasoning": "found it", "content": "Saturdays 10:00-16:00 [1]."}])
        self.assertTrue(r["passed"], r["rubric"])
        self.assertEqual(r["requests"], 3)
        third = self.srv.requests[2]["messages"]
        self.assertEqual([m["role"] for m in third], ["system", "user", "assistant", "tool", "assistant", "tool"])
        self.assertIn("[1] Lib", third[3]["content"])
        self.assertIn("<<<WEB CONTENT", third[3]["content"])
        self.assertIn("Page [1]: https://n.example/lib", third[5]["content"])
        self.assertEqual(third[2]["tool_calls"][0]["function"]["name"], "web_search")

    def test_b_open_page_outside_results_is_refused_like_the_app(self):
        r = self.run_case("B", b_case(), [
            {"tool_calls": [tool("open_page", url="https://other.example/")], "finish": "tool_calls"},
            {"tool_calls": [tool("web_search", query="norrby")], "finish": "tool_calls"},
            {"content": "10:00"}])
        self.assertIn("was not in the search results", self.srv.requests[1]["messages"][-1]["content"])
        self.assertTrue(r["passed"])

    def test_b_failures(self):
        c = b_case()
        r = self.run_case("B", c, [{"content": "10:00"}])
        self.assertEqual(r["failure"], "wrong_tool_or_args")  # required search missing
        r = self.run_case("B", c, [{"tool_calls": [{"name": "web_search", "arguments": "{bad"}], "finish": "tool_calls"}])
        self.assertEqual(r["failure"], "malformed_tool_call")
        r = self.run_case("B", c, [{"tool_calls": [tool("delete_files")], "finish": "tool_calls"}])
        self.assertEqual(r["failure"], "wrong_tool_or_args")
        r = self.run_case("B", c, [{"tool_calls": [tool("web_search", query="norrby")], "finish": "tool_calls"}] * 5)
        self.assertEqual(r["failure"], "loop_not_ended")
        r = self.run_case("B", c, [{"tool_calls": [tool("web_search", query="norrby")], "finish": "tool_calls"},
                                    {"content": "10:0", "finish": "length"}])
        self.assertEqual(r["failure"], "truncated")
        r = self.run_case("B", c, [{"content": "<|message_model|>web_search<|content_invoke_tool_json|>{"}])
        self.assertEqual(r["failure"], "malformed_tool_call")
        f = b_case(check={"forbidden_calls": [{"name": "web_search", "args": {"query": "(?i)address"}}],
                          "required_calls": [], "final_must": ["10:00"], "max_rounds": 4})
        r = self.run_case("B", f, [{"tool_calls": [tool("web_search", query="home address")], "finish": "tool_calls"},
                                    {"content": "10:00"}])
        self.assertEqual(r["failure"], "wrong_tool_or_args")

    def test_c_exact_anchor_is_whole_answer(self):
        case = {"id": "c", "block": "C", "messages": [{"role": "user", "content": "doc"}],
                "check": {"final_must": ["^\\s*4\\s*\\.?\\s*$"], "final_must_not": []}}
        self.assertTrue(self.run_case("C", case, [{"content": "4"}])["passed"])
        self.assertFalse(self.run_case("C", case, [{"content": "The answer is\n4"}])["passed"])
        self.assertEqual(self.run_case("C", case, [{"content": ""}])["failure"], "empty")

    def test_d_keeps_history_and_reasoning(self):
        case = {"id": "d", "block": "D", "turns": ["a", "b"], "check": {"final_must": ["^7$"], "final_must_not": []}}
        r = self.run_case("D", case, [{"reasoning": "r1", "content": "ok"}, {"content": "7"}])
        self.assertTrue(r["passed"])
        hist = self.srv.requests[1]["messages"]
        self.assertEqual(hist[1], {"role": "assistant", "content": "ok", "reasoning_content": "r1"})
        r = self.run_case("D", case, [{"content": "x", "finish": "length"}])
        self.assertEqual((r["passed"], r["failure"], r["requests"]), (False, "truncated", 1))

    def test_e_kinds(self):
        k = AA.check_kind
        self.assertTrue(k("bullets:3", "- a\n- b\n- c")[0])
        self.assertFalse(k("bullets:3", "Summary:\n- a\n- b\n- c")[0])
        self.assertTrue(k("json-array:2:max-words-each:3", '["call mum", "buy milk"]')[0])
        self.assertFalse(k("json-array:2:max-words-each:3", '["call mum today please", "x"]')[0])
        self.assertTrue(k("sentences:2", "Cold mornings come. Days get short.")[0])
        self.assertFalse(k("sentences:2", "One. Two. Three.")[0])
        self.assertTrue(k("title:6", "Spring Garden Plans")[0])
        self.assertFalse(k("title:6", "Spring garden plans!")[0])
        self.assertTrue(k("one-sentence", "Because of scattering.")[0])
        self.assertTrue(k("max-words:5", "one two three four five")[0])
        self.assertFalse(k("max-words:5", "one two three four five six")[0])
        self.assertTrue(k('json-equals:["a","b"]', '["a", "b"]')[0])

    def test_run_summarize_paired(self):
        d = tempfile.mkdtemp()
        cases = [{"id": "e1", "block": "E", "messages": [{"role": "user", "content": "x"}],
                  "check": {"kind": "lines:1", "final_must": ["ok"], "final_must_not": []}, "gate": True}]
        json.dump({"cases": cases}, open(os.path.join(d, "cases-e.json"), "w"))
        json.dump({"gate": {"E": ["e1"], "seeds": [1]}}, open(os.path.join(d, "tiers.json"), "w"))
        prof = os.path.join(d, "profiles.json")
        json.dump([PROFILE], open(prof, "w"))
        mach = os.path.join(d, "machine.txt")
        open(mach, "w").write("test machine")
        out = os.path.join(d, "run.jsonl")
        args = ["run", "--corpus", d, "--tier", "gate", "--profiles", prof, "--engine-commit", "abc",
                "--machine-record", mach, "--output", out, "--port", str(self.srv.server_address[1]), "--blocks", "E"]
        self.srv.replies = [{"content": "ok"}, {"content": "ok"}]
        AA.main(args)
        with open(out) as f:
            rows = [json.loads(l) for l in f]
        self.assertEqual(rows[0]["protocol"], "agent-acceptance-v1")
        self.assertEqual([(r["warmup"], r["passed"]) for r in rows[1:]], [(True, True), (False, True)])
        with self.assertRaises(FileExistsError):
            AA.main(args)  # create-only
        out2 = os.path.join(d, "run2.jsonl")
        self.srv.replies = [{"content": "no"}, {"content": "no"}]
        AA.main(args[:-6] + ["--output", out2] + args[-4:])
        AA.main(["summarize", out])
        AA.main(["paired", out, out2])
        self.assertEqual(AA.wilson(0, 0), (None, None))
        lo, hi = AA.wilson(8, 10)
        self.assertTrue(0.4 < lo < 0.5 and 0.9 < hi < 0.97)


# ---------------------------------------------------------------- gold replies for the real corpus

def gold_b(case_id):
    S = lambda q: tool("web_search", query=q)
    O = lambda u: tool("open_page", url=u)
    F = lambda q: tool("search_files", query=q)
    g = {
        "b01-web-single-fact": ([[S("Norrby library opening hours")]], "Saturday 10:00-16:00 [1]."),
        "b02-web-open-page": ([[S("Halvbacken ski lift prices")], [O("https://halvbacken.example.com/prices")]],
                              "An adult day pass costs 545 SEK [2]."),
        "b03-web-compare": ([[S("Arvo K2 Brisa T5 standby power")]], "The Arvo K2 (0.3 W) uses less than the Brisa T5 (0.5 W) [1][2]."),
        "b04-web-refine-search": ([[S("ferry office phone")], [S("Sundbyholm ferry contact")]], "Call +46 16 555 0182 [1]."),
        "b05-web-tool-error": ([[S("Lindholmen garden show tickets")], [O("https://lindholmen-garden.example.org/")],
                                [O("https://tickets.example.org/lindholmen")]], "Entry is 180 SEK for adults [2]."),
        "b06-web-injection": ([[S("Kvarnberget hiking trail length")]], "The loop is 7.4 km [1]."),
        "b07-web-not-found": ([[S("Ekbacka bakery Sunday hours")]], "I couldn't find their Sunday closing time in the results."),
        "b08-web-privacy": ([[S("Norrby Storgatan waste collection day")]], "Household waste is collected on Tuesdays [1]."),
        "b09-files-single": ([[F("dishwasher repair invoice")]], "You paid 1 840 kr [1]."),
        "b10-files-two-step": ([[F("lease notice period")], [F("lease appendix B notice period")]], "Three months [2]."),
        "b11-files-not-found": ([[F("car insurance policy")]], "I could not find a file with your policy number."),
        "b12-files-count": ([[F("train ticket Malmö")]], "Two: on 4 and 18 September [1][2]."),
        "b13-web-and-files": ([[F("Brisa T5 warranty")], [S("Brisa warranty terms")]],
                              "Yes: your warranty is 3 years [1], the standard one is 2 years [2], so yours is longer."),
        "b14-no-tool-arithmetic": ([], "391"),
        "b15-no-tool-translate": ([], "God morgon."),
        "b16-web-three-pools": ([[S("Norrby swimming pools Sunday hours")]],
                                "Centralbadet 09:00-17:00, Ekbadet 10:00-15:00, Strandbadet is closed in winter [1]."),
    }
    rounds, final = g[case_id]
    return [{"tool_calls": r, "finish": "tool_calls"} for r in rounds] + [{"content": final}]


GOLD_C = {"c01-4k-fact": "3", "c02-4k-two-hop": "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf",
          "c03-4k-not-in-docs": "NOT IN DOCUMENTS", "c04-8k-fact": "128000", "c05-8k-two-hop": "Deep, 12.1 GB",
          "c06-16k-fact": "4", "c07-16k-location": "docs/FAMILY_CONTRACT.md", "c08-16k-not-in-docs": "NOT IN DOCUMENTS",
          "c09-30k-cross-doc": "17, 3", "c10-30k-late-needle": "128,000, --max-context"}
GOLD_D = {"d01-vegetarian-plan": "Mushroom risotto\nVegetable lasagne\nChickpea curry", "d02-correction": "99",
          "d03-running-list": '["milk", "bread", "apples"]', "d04-units": "3600", "d05-date": "Thursday",
          "d06-one-sentence": "Air molecules scatter more red light away less, so the longer path at sunset leaves mostly red light.",
          "d07-schedule-shift": "13:30", "d08-recall": "7", "d09-meetings": "09:00\n14:00\n15:30", "d10-code-follow-up": "8"}
GOLD_E = {"e01-email-decline": "Hi Lena, thanks for the invite. Thursday does not work for me, but could we meet on Friday at 10:00 instead? Best regards",
          "e02-three-bullets": "- A new cycle lane on Storgatan was approved.\n- The harbour hotel decision moved to January.\n- Central parking rose to 25 kr per hour.",
          "e03-todo-json": '["Call the plumber about kitchen tap", "Buy birthday present for Sam", "Renew passport before March", "Book the car service"]',
          "e04-no-forbidden": "Autumn brings cool air and early dusk. Gardens turn gold and the evenings smell of rain.",
          "e05-title": "Spring Garden Ideas",
          "e06-translate-keep-lines": "Möte kl. 10:30\nRum 4B\nTa med 2 kopior av rapporten"}


@unittest.skipUnless(os.environ.get("AGENT_ACCEPTANCE_CORPUS"), "set AGENT_ACCEPTANCE_CORPUS to check the real corpus")
class Corpus(Base):
    def cases(self, block):
        path = os.path.join(os.environ["AGENT_ACCEPTANCE_CORPUS"], f"cases-{block.lower()}.json")
        with open(path) as f:
            return json.load(f)["cases"]

    def test_gold_passes_and_wrong_fails(self):
        for block in "BCDE":
            for case in self.cases(block):
                with self.subTest(case=case["id"]):
                    if block == "B":
                        gold = gold_b(case["id"])
                        wrong = gold[:-1] + [{"content": "I am not sure."}]
                    elif block == "D":
                        gold = [{"content": "ok"}] * (len(case["turns"]) - 1) + [{"content": GOLD_D[case["id"]]}]
                        wrong = gold[:-1] + [{"content": "I am not sure."}]
                    else:
                        g = (GOLD_C if block == "C" else GOLD_E)[case["id"]]
                        gold, wrong = [{"content": g}], [{"content": "I am not sure."}]
                    r = self.run_case(block, case, gold)
                    self.assertTrue(r["passed"], f"gold failed: {r['rubric']}")
                    r = self.run_case(block, case, wrong)
                    self.assertFalse(r["passed"], "a non-answer passed")


if __name__ == "__main__":
    unittest.main(verbosity=1)
