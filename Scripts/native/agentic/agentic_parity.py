#!/usr/bin/env python3
"""Agentic parity for the native engine (docs/NATIVE_PARITY.md, D1).

    agentic_parity.py run --port 8080 --model <id> --out <dir>      # one backend, model run
    agentic_parity.py compare <golden-dir> <candidate-dir>          # parity verdict, no model
    agentic_parity.py list                                          # the frozen task set

The suite agentic-v1 is 32 deterministic tool-use and structured-output tasks:
choosing among several tools, extracting arguments (numbers, enums, units,
names), knowing when not to call a tool, using a tool result in the final
answer across turns, and answering in a required JSON shape. Each task is sent to an
OpenAI-compatible server on loopback at temperature 0 and scored by a
model-free checker on the parsed tool calls or content. `run` scores one
backend; `compare` puts the golden and the candidate side by side.

Both sides go through the same MferenceServer build in --library mode: the
golden is the llama.cpp Quick entry (LlamaBackend at the pin), the candidate
the native Quick entry. The server renders the chat template and parses tool
calls the same way for both, so only the backend differs. A task whose prompt
renders to a different token count on the two sides is a setup error,
reported apart and never counted as a parity result. A request, HTTP or server
error on either side is a setup error too: the run measured nothing for that
task, so the comparison fails until it is re-run. Parity holds when at
most PARITY_SLACK tasks regress (pass on the golden, fail on the candidate),
no task is missing and no prompt rendered differently; tasks the candidate
newly passes are listed, never counted against it.

The suite is frozen: a change is agentic-v2. Stdlib only; the checker and the
verdict are unit-tested without a model (Scripts/tests/test_agentic_parity.py).
"""
import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2]
SUITE = "agentic-v1"
PARITY_SLACK = 1


def _ev():
    spec = importlib.util.spec_from_file_location("engine_validate", SCRIPTS / "engine_validate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tool(name, description, props, required):
    # keys sorted, so the schema renders identically on every engine
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"properties": props, "required": required, "type": "object"}}}


WEATHER = _tool("get_weather", "Current weather for a city.",
                {"city": {"description": "City name", "type": "string"},
                 "unit": {"description": "Temperature unit", "enum": ["celsius", "fahrenheit"], "type": "string"}},
                ["city"])
CONVERT = _tool("convert_currency", "Convert an amount between two ISO 4217 currencies.",
                {"amount": {"description": "Amount to convert", "type": "number"},
                 "from": {"description": "Source currency code", "type": "string"},
                 "to": {"description": "Target currency code", "type": "string"}},
                ["amount", "from", "to"])
EVENT = _tool("create_event", "Create a calendar event.",
              {"date": {"description": "Date as YYYY-MM-DD", "type": "string"},
               "time": {"description": "Start time as HH:MM, 24-hour", "type": "string"},
               "title": {"description": "Event title", "type": "string"}},
              ["date", "time", "title"])
SEARCH = _tool("search_files", "Search the project's files for a text pattern.",
               {"pattern": {"description": "Text to look for", "type": "string"},
                "path": {"description": "Directory to search", "type": "string"}},
               ["pattern"])
EMAIL = _tool("send_email", "Send an email.",
              {"body": {"description": "Message body", "type": "string"},
               "subject": {"description": "Subject line", "type": "string"},
               "to": {"description": "Recipient address", "type": "string"}},
              ["to", "subject", "body"])
CALC = _tool("calculate", "Evaluate an arithmetic expression exactly.",
             {"expression": {"description": "Expression using + - * / and parentheses", "type": "string"}},
             ["expression"])
TOOLS = [WEATHER, CONVERT, EVENT, SEARCH, EMAIL, CALC]


def _u(text):
    return [{"role": "user", "content": text}]


def _after_tool(question, name, arguments, result):
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "call_0", "type": "function",
                         "function": {"name": name, "arguments": json.dumps(arguments)}}]},
        {"role": "tool", "tool_call_id": "call_0", "content": json.dumps(result)},
    ]


def _call(name, **args):
    return {"kind": "tool_call", "name": name, "arguments": args}


TASKS = [
    # tool choice and argument extraction
    {"id": "weather-plain", "messages": _u("What's the weather in Oslo right now?"), "tools": TOOLS,
     "check": _call("get_weather", city="oslo")},
    {"id": "weather-unit", "messages": _u("Tell me the current temperature in Phoenix in Fahrenheit."),
     "tools": TOOLS, "check": _call("get_weather", city="phoenix", unit="fahrenheit")},
    {"id": "weather-celsius", "messages": _u("How warm is it in Nairobi? Use Celsius please."), "tools": TOOLS,
     "check": _call("get_weather", city="nairobi", unit="celsius")},
    {"id": "convert-basic", "messages": _u("Convert 250 US dollars to euros."), "tools": TOOLS,
     "check": _call("convert_currency", amount=250, **{"from": "USD", "to": "EUR"})},
    {"id": "convert-decimal", "messages": _u("How many Japanese yen is 19.99 British pounds?"), "tools": TOOLS,
     "check": _call("convert_currency", amount=19.99, **{"from": "GBP", "to": "JPY"})},
    {"id": "convert-words", "messages": _u("I have three thousand Swiss francs. What is that in Canadian dollars?"),
     "tools": TOOLS, "check": _call("convert_currency", amount=3000, **{"from": "CHF", "to": "CAD"})},
    {"id": "event-basic", "messages": _u("Put 'Dentist' on my calendar for 2026-12-03 at 14:30."), "tools": TOOLS,
     "check": _call("create_event", date="2026-12-03", time="14:30", title="dentist")},
    {"id": "event-24h", "messages": _u("Schedule a team sync called Planning on 2027-01-15 at 9 in the morning."),
     "tools": TOOLS, "check": _call("create_event", date="2027-01-15", time="09:00", title="planning")},
    {"id": "event-pm", "messages": _u("Create an event 'Board review' on 2026-11-30 at 4:15 pm."), "tools": TOOLS,
     "check": _call("create_event", date="2026-11-30", time="16:15", title="board review")},
    {"id": "search-basic", "messages": _u("Find every file that mentions TODO(perf)."), "tools": TOOLS,
     "check": _call("search_files", pattern="TODO(perf)")},
    {"id": "search-path", "messages": _u("Search the Sources/Mference directory for the text makeBuffer."),
     "tools": TOOLS, "check": _call("search_files", pattern="makeBuffer", path="sources/mference")},
    {"id": "email-basic", "messages": _u("Email ana@example.com with subject 'Lunch' and say: see you at noon."),
     "tools": TOOLS, "check": {"kind": "tool_call", "name": "send_email",
                               "arguments": {"to": "ana@example.com", "subject": "lunch"},
                               "contains": {"body": "noon"}}},
    {"id": "calc-basic", "messages": _u("Use the calculator to work out (17 * 23) + 4."), "tools": TOOLS,
     "check": {"kind": "tool_call", "name": "calculate", "expression_value": 395}},
    {"id": "calc-words", "messages": _u("What is 1200 divided by 16? Use the calculator tool."), "tools": TOOLS,
     "check": {"kind": "tool_call", "name": "calculate", "expression_value": 75}},
    # no tool needed
    {"id": "no-tool-greeting", "messages": _u("Hi! Just saying hello, no need to do anything."), "tools": TOOLS,
     "check": {"kind": "no_tool"}},
    {"id": "no-tool-fact", "messages": _u("What is the chemical symbol for gold? Answer in one word."),
     "tools": TOOLS, "check": {"kind": "no_tool", "contains_any": ["au"]}},
    {"id": "no-tool-capital", "messages": _u("What is the capital of Canada? One word."), "tools": TOOLS,
     "check": {"kind": "no_tool", "contains_any": ["ottawa"]}},
    # using a tool result
    {"id": "result-weather", "messages": _after_tool("What's the weather in Lisbon?", "get_weather",
                                                     {"city": "Lisbon"},
                                                     {"city": "Lisbon", "temperature_c": 18, "conditions": "light rain"}),
     "tools": TOOLS, "check": {"kind": "no_tool", "contains_all": ["18", "rain"]}},
    {"id": "result-convert", "messages": _after_tool("Convert 100 EUR to USD.", "convert_currency",
                                                     {"amount": 100, "from": "EUR", "to": "USD"},
                                                     {"amount": 108.37, "currency": "USD"}),
     "tools": TOOLS, "check": {"kind": "no_tool", "contains_all": ["108.37"]}},
    {"id": "result-search", "messages": _after_tool("Which files mention makeBuffer?", "search_files",
                                                    {"pattern": "makeBuffer"},
                                                    {"matches": ["Model.swift", "Buffers.swift"]}),
     "tools": TOOLS, "check": {"kind": "no_tool", "contains_all": ["model.swift", "buffers.swift"]}},
    {"id": "result-calc", "messages": _after_tool("What is 37 times 41?", "calculate",
                                                  {"expression": "37 * 41"}, {"result": 1517}),
     "tools": TOOLS, "check": {"kind": "no_tool", "contains_all": ["1517"]}},
    {"id": "result-chain", "messages": _after_tool(
        "What's the weather in Reykjavik, and then convert 50 EUR to ISK?", "get_weather",
        {"city": "Reykjavik"}, {"city": "Reykjavik", "temperature_c": -2, "conditions": "snow"}),
     "tools": TOOLS, "check": _call("convert_currency", amount=50, **{"from": "EUR", "to": "ISK"})},
    # structured output
    {"id": "json-city", "messages": _u("Return only a JSON object with keys \"city\" (string) and \"country\" "
                                       "(string) for the capital of France. No prose, no code fence."),
     "check": {"kind": "json", "equals": {"city": "paris", "country": "france"}}},
    {"id": "json-numbers", "messages": _u("Return only a JSON object with integer keys \"sum\" and \"product\" "
                                          "for the numbers 6 and 7. No prose."),
     "check": {"kind": "json", "equals": {"sum": 13, "product": 42}}},
    {"id": "json-list", "messages": _u("Return only a JSON object with a key \"primes\" holding the first five "
                                       "prime numbers as an array. No prose."),
     "check": {"kind": "json", "equals": {"primes": [2, 3, 5, 7, 11]}}},
    {"id": "json-bool", "messages": _u("Return only a JSON object {\"even\": <bool>} saying whether 1234 is even. "
                                       "No prose."),
     "check": {"kind": "json", "equals": {"even": True}}},
    {"id": "json-extract", "messages": _u("Extract the order as JSON with keys \"item\" (string), \"qty\" (integer) "
                                          "and \"rush\" (boolean) from: 'Please send 12 oak dowels, no hurry.' "
                                          "Return only the JSON object."),
     "check": {"kind": "json", "equals": {"item": "oak dowels", "qty": 12, "rush": False}}},
    {"id": "json-nested", "messages": _u("Return only a JSON object {\"user\": {\"name\": \"Mira\", \"age\": 31}} "
                                         "with age increased by one. No prose."),
     "check": {"kind": "json", "equals": {"user": {"name": "mira", "age": 32}}}},
    # exact short answers
    {"id": "exact-math", "messages": _u("What is 144 divided by 12? Reply with the number only."),
     "check": {"kind": "content", "regex": r"^\s*12(\.0+)?\s*\.?\s*$"}},
    {"id": "exact-letter", "messages": _u("What is the third letter of the word 'engine'? Reply with the letter only."),
     "check": {"kind": "content", "regex": r"^\s*[\"']?g[\"']?\s*\.?\s*$"}},
    {"id": "exact-yesno", "messages": _u("Is 97 a prime number? Reply yes or no only."),
     "check": {"kind": "content", "regex": r"^\s*yes\s*\.?\s*$"}},
    {"id": "exact-count", "messages": _u("How many words are in the sentence 'the quick brown fox jumps'? "
                                         "Reply with the number only."),
     "check": {"kind": "content", "regex": r"^\s*5\s*\.?\s*$"}},
]


# ---------------------------------------------------------------------------
# Scoring (model-free)

def _norm(v):
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, list):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {str(k).lower(): _norm(x) for k, x in v.items()}
    return v


def _safe_eval(expression):
    """Evaluate + - * / and parentheses over numbers, nothing else."""
    if not re.fullmatch(r"[\d\s.+\-*/()]+", expression or ""):
        return None
    tokens = re.findall(r"\d+(?:\.\d+)?|[+\-*/()]", expression)
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def take():
        nonlocal pos
        pos += 1
        return tokens[pos - 1]

    def atom():
        t = take()
        if t == "(":
            v = expr()
            if take() != ")":
                raise ValueError("paren")
            return v
        if t == "-":
            return -atom()
        return float(t)

    def term():
        v = atom()
        while peek() in ("*", "/"):
            v = v * atom() if take() == "*" else v / atom()
        return v

    def expr():
        v = term()
        while peek() in ("+", "-"):
            v = v + term() if take() == "+" else v - term()
        return v

    try:
        value = expr()
        return value if pos == len(tokens) else None
    except (ValueError, ZeroDivisionError, IndexError, TypeError):
        return None


def _strip_think(content):
    return re.sub(r"<think>.*?</think>", "", content or "", flags=re.S).strip()


def _json_body(content):
    text = _strip_think(content)
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text)
    except ValueError:
        return None


def score(task, result):
    """(passed, reason) for one canonical result {content, tool_calls}."""
    check = task["check"]
    if result.get("error"):
        return False, f"error: {result['error']}"
    calls = result.get("tool_calls") or []
    content = _strip_think(result.get("content"))
    kind = check["kind"]
    if kind == "tool_call":
        if len(calls) < 1:
            return False, "no tool call"
        call = calls[0]
        if call.get("name") != check["name"]:
            return False, f"called {call.get('name')}"
        try:
            args = json.loads(call.get("arguments") or "{}") if isinstance(call.get("arguments"), str) else call.get("arguments") or {}
        except ValueError:
            return False, "arguments are not JSON"
        if not isinstance(args, dict):
            return False, "arguments are not a JSON object"
        args = _norm(args)
        for k, v in (check.get("arguments") or {}).items():
            if args.get(k.lower()) != _norm(v):
                return False, f"argument {k}={args.get(k.lower())!r}, expected {v!r}"
        for k, sub in (check.get("contains") or {}).items():
            if sub.lower() not in str(args.get(k.lower(), "")):
                return False, f"argument {k} lacks {sub!r}"
        if "expression_value" in check:
            v = _safe_eval(str(args.get("expression", "")))
            if v is None or abs(v - check["expression_value"]) > 1e-9:
                return False, f"expression {args.get('expression')!r} = {v}"
        return True, "ok"
    if kind == "no_tool":
        if calls:
            return False, f"called {calls[0].get('name')}"
        text = content.lower()
        if not text:
            return False, "empty answer"
        if check.get("contains_all") and not all(s in text for s in check["contains_all"]):
            return False, "answer lacks the tool result"
        if check.get("contains_any") and not any(s in text for s in check["contains_any"]):
            return False, "answer lacks the expected word"
        return True, "ok"
    if kind == "json":
        body = _json_body(result.get("content"))
        if not isinstance(body, dict):
            return False, "not a JSON object"
        if _norm(body) != _norm(check["equals"]):
            return False, f"JSON {body!r}"
        return True, "ok"
    if kind == "content":
        if calls:
            return False, "unexpected tool call"
        return (bool(re.match(check["regex"], content, flags=re.I)), "ok" if re.match(check["regex"], content, flags=re.I) else f"answer {content[:40]!r}")
    raise ValueError(f"unknown check kind {kind}")


def _prompt_tokens(entry):
    return ((entry or {}).get("result") or {}).get("prompt_tokens")


def compare(golden, candidate, slack=PARITY_SLACK):
    """Verdict from two `run` reports ({task id: {passed, result, ...}})."""
    ids = [t["id"] for t in TASKS]
    missing = [i for i in ids if i not in golden or i not in candidate]
    both = [i for i in ids if i in golden and i in candidate]
    # a request, HTTP or server error measured nothing: on either side it is a
    # setup error that blocks the verdict, never a scored failure
    errored = lambda run: [i for i in both if ((run[i].get("result") or {}).get("error"))]
    setup_errors = {"golden": errored(golden), "candidate": errored(candidate)}
    render_mismatch = [i for i in both
                       if _prompt_tokens(golden[i]) is not None and _prompt_tokens(candidate[i]) is not None
                       and _prompt_tokens(golden[i]) != _prompt_tokens(candidate[i])]
    scored = [i for i in both if i not in render_mismatch
              and i not in setup_errors["golden"] and i not in setup_errors["candidate"]]
    regressions = [i for i in scored if golden[i]["passed"] and not candidate[i]["passed"]]
    improvements = [i for i in scored if candidate[i]["passed"] and not golden[i]["passed"]]
    g = sum(1 for i in scored if golden[i]["passed"])
    c = sum(1 for i in scored if candidate[i]["passed"])
    passed = (not missing and not render_mismatch and not setup_errors["golden"]
              and not setup_errors["candidate"] and len(regressions) <= slack)
    return {"suite": SUITE, "tasks": len(ids), "golden_passed": g, "candidate_passed": c,
            "slack": slack, "regressions": regressions, "improvements": improvements,
            "render_mismatch": render_mismatch, "setup_errors": setup_errors, "missing": missing,
            "parity": passed}


def body_for(task, model):
    body = {"model": model, "messages": task["messages"], "temperature": 0, "top_k": 1, "top_p": 1.0,
            "seed": 0, "max_tokens": 512}
    if task.get("tools"):
        body["tools"] = task["tools"]
    return body


def run_tasks(post, canonical, model, on_task=None):
    """Score every task. `post(body)` returns engine_validate.post_chat's
    `(status, parsed, raw, elapsed)`; `canonical(parsed)` its canonical
    result. A task whose request or scoring raises is recorded as failed,
    never aborts the suite. `on_task(report)` runs after every task, so
    partial results are saved as they come."""
    report = {}
    for task in TASKS:
        try:
            status, parsed, *_ = post(body_for(task, model))
            result = canonical(parsed) if status == 200 else {"error": f"HTTP {status}"}
        except Exception as error:  # a timeout or connection error fails this task only
            result = {"error": f"request failed: {error!r}"}
        try:
            passed, reason = score(task, result)
        except Exception as error:  # a malformed response fails its task only
            passed, reason = False, f"scoring error: {error!r}"
        report[task["id"]] = {"passed": passed, "reason": reason, "result": result}
        print(f"{task['id']}: {'PASS' if passed else 'FAIL'} {reason}", file=sys.stderr)
        if on_task:
            on_task(report)
    return report


def cmd_run(args):
    ev = _ev()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def save(report, complete=False):
        tmp = out / "agentic.json.tmp"
        tmp.write_text(json.dumps({"suite": SUITE, "model": args.model, "complete": complete,
                                   "tasks": report}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(out / "agentic.json")

    report = run_tasks(lambda body: ev.post_chat(args.host, args.port, body, args.timeout),
                       ev.canonical_from_message, args.model, on_task=save)
    save(report, complete=True)
    print(f"{sum(r['passed'] for r in report.values())}/{len(report)} passed")
    return 0


def cmd_compare(args):
    load = lambda d: json.loads((Path(d) / "agentic.json").read_text(encoding="utf-8"))["tasks"]
    verdict = compare(load(args.golden), load(args.candidate), args.slack)
    print(json.dumps(verdict, indent=1))
    return 0 if verdict["parity"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--host", default="127.0.0.1")
    r.add_argument("--port", type=int, default=8080)
    r.add_argument("--model", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--timeout", type=float, default=600)
    c = sub.add_parser("compare")
    c.add_argument("golden")
    c.add_argument("candidate")
    c.add_argument("--slack", type=int, default=PARITY_SLACK)
    sub.add_parser("list")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "compare":
        return cmd_compare(args)
    for t in TASKS:
        print(f"{t['id']}\t{t['check']['kind']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
