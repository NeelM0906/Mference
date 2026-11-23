#!/usr/bin/env python3
"""agent-acceptance-v1 client: Inkling 2 Mini vs Deep on Mference's own tasks.

Runs the frozen corpus (blocks A to E) against an ALREADY-RUNNING loopback
OpenAI-compatible server (MferenceServer --library). It never starts a model,
never runs generated code, never touches the network beyond 127.0.0.1, and
plays the tool side itself from the corpus fixtures. Scoring is deterministic
(no model judge); failures stay in the denominator. Stdlib only, Python 3.9.

    run:        agent_acceptance.py run --corpus <dir> --tier gate|full --profiles <json>
                    --engine-commit <sha> --machine-record <file> --output <new.jsonl> [--port 18489]
                    [--blocks A,B,C,D,E] [--seeds 1,2,3] [--timeout 900]
    summarize:  agent_acceptance.py summarize <run.jsonl>
    paired:     agent_acceptance.py paired <run-a.jsonl> <run-b.jsonl>

--profiles: JSON list of {"label", "model", "source_revision", optional
"reasoning_effort", optional "sampling" (extra request fields)}. A profile
with no sampling fields sends none: the server's product defaults apply.
Block A uses the release screen's own scoring (Scripts/release_task_screen.py,
imported), at the acceptance caps instead of its 512-token screen cap.
Output is create-only JSONL: one header line, then one line per request
sequence (a case, including every tool round or turn), warmups marked.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent.parent
REPO = SCRIPTS.parent
CAPS = {"A": 2048, "B": 2048, "C": 4096, "D": 4096, "E": 2048}
MAX_PAGES = 3


def load_screen():
    spec = importlib.util.spec_from_file_location("release_task_screen", SCRIPTS / "release_task_screen.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SCREEN = load_screen()


# ------------------------------------------------------------------ transport

def post(url, body, timeout):
    """One streamed chat completion: (response, metrics) or raises."""
    started = time.monotonic()
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as reply:
        return SCREEN.read_stream(reply, started)


def choice(response):
    return response["choices"][0]


def visible(response):
    return (choice(response).get("message", {}).get("content") or "").strip()


MARKUP = re.compile(r"<\|(message_model|content_invoke_tool_json|content_thinking|content_text|end_message)\|>")


# ------------------------------------------------------------------ tool fixtures (block B)

class Fixtures:
    """Plays web_search, open_page and search_files from a case's fixtures, in the app's formats
    (WebToolbox.swift, FindChatTools.swift), including its failure messages."""

    def __init__(self, fx, offered):
        self.fx = fx
        self.offered = offered
        self.calls = {}
        self.allowed_urls = set()
        self.pages_read = 0
        self.numbers = {}
        self.code = fx.get("marker_code", "00000000")

    def number(self, key):
        if key not in self.numbers:
            self.numbers[key] = len(self.numbers) + 1
        return self.numbers[key]

    def pick(self, tool, query):
        n = self.calls.get(tool, 0)
        for entry in self.fx.get(tool, []):
            if "calls" in entry and n not in entry["calls"]:
                continue
            if "when_any" in entry and not any(re.search(p, query) for p in entry["when_any"]):
                continue
            return entry
        return None

    def run(self, name, args):
        if name not in self.offered:
            return f"There is no tool named {name}."
        self.calls[name] = self.calls.get(name, 0) + 1
        if name == "web_search":
            query = str(args.get("query", "")).strip()
            if not query:
                return "web_search needs a non-empty \"query\"."
            entry = self.pick(name, query)
            results = (entry or {}).get("results", [])
            if not results:
                return f"No results for “{query}”."
            lines = [f"Results for “{query}”:", f"<<<WEB CONTENT {self.code}>>>"]
            for r in results:
                self.allowed_urls.add(r["url"])
                lines += ["", f"[{self.number(r['url'])}] {r['title']}", r["url"], r["snippet"][:300]]
            lines.append(f"<<<END WEB CONTENT {self.code}>>>")
            return "\n".join(lines)
        if name == "open_page":
            url = str(args.get("url", ""))
            if not url:
                return "open_page needs the \"url\" of a search result."
            if url not in self.allowed_urls:
                return f"{url} was not in the search results; open only the URL of a result."
            if self.pages_read >= MAX_PAGES:
                return f"Already read {MAX_PAGES} pages for this answer. Answer with what you have."
            self.pages_read += 1
            page = self.fx.get("open_page", {}).get(url)
            if page is None:
                return f"The page could not be read ({url})."
            if "error" in page:
                return page["error"]
            return (f"Page [{self.number(url)}]: {url}\n<<<WEB CONTENT {self.code}>>>\n{page['title']}\n\n"
                    f"{page['text']}\n<<<END WEB CONTENT {self.code}>>>")
        if name == "search_files":
            query = str(args.get("query", "")).strip()
            if not query:
                return "search_files needs a non-empty \"query\"."
            entry = self.pick(name, query)
            files = (entry or {}).get("files", [])
            if not files:
                return f"No file on this Mac is a close match for “{query}”."
            lines = [f"Files matching “{query}”:", "<<<FILE CONTENT>>>"]
            for f in files:
                lines += ["", f"[{self.number(f['name'])}] {f['name']} ({f['location']}), modified {f['modified']}",
                          f["passage"]]
            lines.append("<<<END FILE CONTENT>>>")
            return "\n".join(lines)
        return f"There is no tool named {name}."


def call_matches(spec, call):
    if spec["name"] != call["name"]:
        return False
    return all(re.search(p, str(call["args"].get(k, ""))) for k, p in spec.get("args", {}).items())


def check_trace(check, calls):
    """Required calls as an ordered subsequence (or any order), forbidden calls, call limit."""
    if "max_calls" in check and len(calls) > check["max_calls"]:
        return False, "wrong_tool_or_args", f"{len(calls)} tool calls, at most {check['max_calls']} allowed"
    for spec in check.get("forbidden_calls", []):
        for c in calls:
            if call_matches(spec, c):
                return False, "wrong_tool_or_args", f"forbidden call {c['name']} {c['args']}"
    req = check.get("required_calls", [])
    if check.get("any_order"):
        used = set()
        for spec in req:
            hit = next((i for i, c in enumerate(calls) if i not in used and call_matches(spec, c)), None)
            if hit is None:
                return False, "wrong_tool_or_args", f"missing call {spec}"
            used.add(hit)
    else:
        i = 0
        for c in calls:
            if i < len(req) and call_matches(req[i], c):
                i += 1
        if i < len(req):
            return False, "wrong_tool_or_args", f"missing call {req[i]}"
    return True, None, "trace ok"


def check_text(check, text):
    for p in check.get("final_must", []):
        if not re.search(p, text):
            return False, f"answer lacks /{p}/"
    for p in check.get("final_must_not", []):
        if re.search(p, text):
            return False, f"answer matches forbidden /{p}/"
    kind = check.get("kind")
    if kind:
        ok, why = check_kind(kind, text)
        if not ok:
            return False, why
    return True, "answer ok"


def words(s):
    return len(re.findall(r"\S+", s))


def sentences(s):
    s = s.strip()
    parts = [p for p in re.split(r"(?<=[.!?])\s+", s) if p.strip()]
    return len(parts) if s and s[-1] in ".!?" else -1


def check_kind(kind, text):
    lines = [l for l in text.strip().split("\n") if l.strip()]
    if kind.startswith("max-words:"):
        n = int(kind.split(":")[1])
        return words(text) <= n, f"{words(text)} words (max {n})"
    if kind.startswith("bullets:"):
        n = int(kind.split(":")[1])
        return len(lines) == n and all(l.startswith("- ") for l in lines), f"{len(lines)} lines, bullets required"
    if kind.startswith("json-array:"):
        _, n, _, m = kind.split(":")
        try:
            arr = json.loads(text)
        except ValueError:
            return False, "not a JSON array (or extra text)"
        ok = isinstance(arr, list) and len(arr) == int(n) and all(isinstance(x, str) and words(x) <= int(m) for x in arr)
        return ok, f"JSON array of {n} strings of at most {m} words"
    if kind.startswith("json-equals:"):
        try:
            return json.loads(text) == json.loads(kind.split(":", 1)[1]), "exact JSON"
        except ValueError:
            return False, "not JSON (or extra text)"
    if kind.startswith("sentences:"):
        n = int(kind.split(":")[1])
        return sentences(text) == n and len(lines) <= n, f"{sentences(text)} sentences (need {n})"
    if kind == "one-sentence":
        return sentences(text) == 1 and len(lines) == 1, f"{sentences(text)} sentences (need 1)"
    if kind == "three-lines":
        return len(lines) == 3, f"{len(lines)} lines (need 3)"
    if kind.startswith("title:"):
        n = int(kind.split(":")[1])
        ws = text.strip().split()
        ok = 0 < len(ws) <= n and not re.search(r"[^\w\s'’-]", text.strip()) and all(w[0].isupper() for w in ws)
        return ok, f"title of {len(ws)} words, capitalised, no punctuation"
    if kind.startswith("lines:"):
        n = int(kind.split(":")[1])
        return len(lines) == n, f"{len(lines)} lines (need {n})"
    return False, f"unknown check kind {kind}"


def finished(response):
    """(ok, category) for a reply that should end the case with an answer."""
    fin = choice(response).get("finish_reason")
    text = visible(response)
    if fin == "length":
        return False, "truncated"
    if MARKUP.search(text):
        return False, "malformed_tool_call"
    if fin == "tool_calls":
        return False, "wrong_tool_or_args"
    if fin != "stop":
        return False, "truncated"
    if not text:
        return False, "empty"
    return True, None


# ------------------------------------------------------------------ cases

def base_body(profile, block, seed):
    body = {"model": profile["model"], "stream": True, "stream_options": {"include_usage": True},
            "max_completion_tokens": CAPS[block], "seed": seed}
    body.update(profile.get("sampling", {}))
    if profile.get("reasoning_effort") is not None:
        body["reasoning_effort"] = profile["reasoning_effort"]
    return body


def assistant_turn(response):
    m = choice(response)["message"]
    out = {"role": "assistant", "content": m.get("content") or ""}
    if m.get("reasoning_content"):
        out["reasoning_content"] = m["reasoning_content"]
    if m.get("tool_calls"):
        out["tool_calls"] = m["tool_calls"]
    return out


def run_a(url, case, profile, seed, timeout):
    body = base_body(profile, "A", seed)
    body["messages"] = [{"role": "user", "content": case["prompt"]}]
    if "tool" in case:
        body["tools"] = [{"type": "function", "function": case["tool"]}]
    response, metrics = post(url, body, timeout)
    passed, rubric = SCREEN.score(case, response)
    cat = None
    if not passed:
        fin = choice(response).get("finish_reason")
        cat = "truncated" if fin == "length" else ("empty" if not visible(response) and fin == "stop" else "wrong_answer")
    return passed, cat, rubric, [{"request": body, "response": response, "metrics": metrics}]


def run_b(url, case, profile, seed, timeout):
    offered = {t["function"]["name"] for t in case["tools"]}
    fx = Fixtures(case["fixtures"], offered)
    messages = [{"role": "system", "content": case["system"]}] + list(case["messages"])
    calls, rounds = [], []
    check = case["check"]
    for _ in range(check.get("max_rounds", 6)):
        body = base_body(profile, "B", seed)
        body.update(messages=list(messages), tools=case["tools"])
        response, metrics = post(url, body, timeout)
        rounds.append({"request": body, "response": response, "metrics": metrics})
        fin = choice(response).get("finish_reason")
        if fin != "tool_calls":
            ok, cat = finished(response)
            if not ok:
                return False, cat, "no usable final answer", rounds
            ok, cat, why = check_trace(check, calls)
            if not ok:
                return False, cat, why, rounds
            ok, why = check_text(check, visible(response))
            return ok, None if ok else "wrong_answer", why, rounds
        messages.append(assistant_turn(response))
        for tc in choice(response)["message"].get("tool_calls", []):
            name = tc.get("function", {}).get("name", "")
            try:
                args = json.loads(tc.get("function", {}).get("arguments") or "{}")
                if not isinstance(args, dict):
                    raise ValueError("arguments are not an object")
            except ValueError:
                return False, "malformed_tool_call", f"invalid arguments for {name}", rounds
            if name not in offered:
                return False, "wrong_tool_or_args", f"call to a tool not offered: {name}", rounds
            calls.append({"name": name, "args": args})
            result = fx.run(name, args)
            rounds[-1].setdefault("tool_results", []).append(result)
            messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": result})
    return False, "loop_not_ended", f"no final answer within {check.get('max_rounds', 6)} rounds", rounds


def run_single(url, case, profile, seed, timeout, block):
    body = base_body(profile, block, seed)
    body["messages"] = list(case["messages"])
    response, metrics = post(url, body, timeout)
    ok, cat = finished(response)
    if not ok:
        return False, cat, "no usable final answer", [{"request": body, "response": response, "metrics": metrics}]
    text = visible(response)
    ok, why = check_text(case["check"], text)
    return ok, None if ok else "wrong_answer", why, [{"request": body, "response": response, "metrics": metrics}]


def run_d(url, case, profile, seed, timeout):
    messages, rounds = [], []
    for turn in case["turns"]:
        messages.append({"role": "user", "content": turn})
        body = base_body(profile, "D", seed)
        body["messages"] = list(messages)
        response, metrics = post(url, body, timeout)
        rounds.append({"request": body, "response": response, "metrics": metrics})
        ok, cat = finished(response)
        if not ok:
            return False, cat, f"turn {len(rounds)}: no usable answer", rounds
        messages.append(assistant_turn(response))
    ok, why = check_text(case["check"], visible(rounds[-1]["response"]))
    return ok, None if ok else "wrong_answer", why, rounds


def run_case(url, block, case, profile, seed, timeout):
    started = time.monotonic()
    try:
        if block == "A":
            r = run_a(url, case, profile, seed, timeout)
        elif block == "B":
            r = run_b(url, case, profile, seed, timeout)
        elif block == "D":
            r = run_d(url, case, profile, seed, timeout)
        else:
            r = run_single(url, case, profile, seed, timeout, block)
        passed, cat, why, rounds = r
        error = None
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError, ValueError, KeyError, IndexError) as e:
        passed, cat, why, rounds, error = False, "transport_error", "transport or protocol error", [], str(e)
    usage = [ro["response"].get("usage") for ro in rounds]
    return {"passed": passed, "failure": cat, "rubric": why, "rounds": rounds, "error": error,
            "requests": len(rounds), "seconds": time.monotonic() - started,
            "first_delta_seconds": (rounds[0]["metrics"].get("first_model_delta_seconds") if rounds else None),
            "completion_tokens": sum((u or {}).get("completion_tokens") or 0 for u in usage) if usage and all(u for u in usage) else None,
            "prompt_tokens": sum((u or {}).get("prompt_tokens") or 0 for u in usage) if usage and all(u for u in usage) else None}


# ------------------------------------------------------------------ run, summarize, paired

def load_corpus(corpus, tier_name, blocks):
    tiers = read_json(os.path.join(corpus, "tiers.json"))
    tier = tiers[tier_name]
    screen_path = REPO / "docs/benchmark-prompts/release-screen-v1/cases.json"
    screen = {c["id"]: c for c in read_json(screen_path)}
    out = []
    for b in blocks:
        if b == "A":
            out += [("A", screen[i]) for i in tier["A"]]
        else:
            cases = {c["id"]: c for c in read_json(os.path.join(corpus, f"cases-{b.lower()}.json"))["cases"]}
            out += [(b, cases[i]) for i in tier[b]]
    return tiers, tier, out, screen_path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cmd_run(a):
    blocks = [b.strip().upper() for b in a.blocks.split(",")]
    tiers, tier, cases, screen_path = load_corpus(a.corpus, a.tier, blocks)
    profiles = read_json(a.profiles)
    if not profiles or len({p["label"] for p in profiles}) != len(profiles):
        sys.exit("profiles must have unique labels")
    for p in profiles:
        if not p.get("model") or not p.get("source_revision"):
            sys.exit("each profile needs model and source_revision")
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else tier["seeds"]
    url = f"http://127.0.0.1:{a.port}/v1/chat/completions"
    manifest = {n: sha256(os.path.join(a.corpus, n)) for n in sorted(os.listdir(a.corpus)) if n.endswith(".json")}
    with open(a.output, "x") as out:
        header = {"protocol": "agent-acceptance-v1", "tier": a.tier, "blocks": blocks, "seeds": seeds,
                  "warmups_per_case": a.warmups, "caps": CAPS, "profiles": profiles,
                  "engine_commit": a.engine_commit, "machine": Path(a.machine_record).read_text(),
                  "corpus_sha256": manifest, "release_screen_sha256": sha256(screen_path), "port": a.port}
        out.write(json.dumps(header) + "\n")
        out.flush()
        for profile in profiles:
            for block, case in cases:
                for si, seed in enumerate(seeds):
                    # warmups: per case and profile (before the first seed only), as the release screen
                    for rep in range((a.warmups if si == 0 else 0) + 1):
                        r = run_case(url, block, case, profile, seed, a.timeout)
                        r.update(profile=profile["label"], block=block, case=case["id"], seed=seed,
                                 warmup=si == 0 and rep < a.warmups)
                        out.write(json.dumps(r, ensure_ascii=False) + "\n")
                        out.flush()
                        print(json.dumps({k: r[k] for k in ("profile", "block", "case", "seed", "warmup", "passed",
                                                            "failure", "seconds")}), flush=True)


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def measured(path):
    with open(path) as f:
        header = json.loads(f.readline())
        rows = [json.loads(l) for l in f if l.strip()]
    return header, [r for r in rows if not r["warmup"]]


def cmd_summarize(a):
    header, rows = measured(a.run)
    groups = {}
    for r in rows:
        for key in ((r["profile"], r["block"]), (r["profile"], "all")):
            g = groups.setdefault(key, {"n": 0, "k": 0, "fail": {}, "secs": []})
            g["n"] += 1
            g["k"] += r["passed"]
            g["secs"].append(r["seconds"])
            if not r["passed"]:
                g["fail"][r["failure"]] = g["fail"].get(r["failure"], 0) + 1
    print("| Profile | Block | Passed | Rate | 95% Wilson | Median s/case | Failures |")
    print("|---|---|---:|---:|---|---:|---|")
    for (p, b), g in sorted(groups.items()):
        s = sorted(g["secs"])
        med = s[len(s) // 2] if s else None
        lo, hi = wilson(g["k"], g["n"])
        print(f"| {p} | {b} | {g['k']}/{g['n']} | {g['k'] / g['n']:.3f} | {lo}-{hi} | {med:.1f} | "
              + ", ".join(f"{k} {v}" for k, v in sorted(g["fail"].items())) + " |")


def cmd_paired(a):
    _, ra = measured(a.run_a)
    _, rb = measured(a.run_b)
    key = lambda r: (r["block"], r["case"], r["seed"])
    ia = {key(r): r["passed"] for r in ra}
    ib = {key(r): r["passed"] for r in rb}
    common = sorted(set(ia) & set(ib))
    table = {}
    for k in common:
        t = table.setdefault(k[0], [0, 0, 0, 0])
        t[(0 if ia[k] else 2) + (0 if ib[k] else 1)] += 1
    print(f"| Block | Both pass | Only A ({os.path.basename(a.run_a)}) | Only B ({os.path.basename(a.run_b)}) | Neither |")
    print("|---|---:|---:|---:|---:|")
    for b, t in sorted(table.items()):
        print(f"| {b} | {t[0]} | {t[1]} | {t[2]} | {t[3]} |")
    print(f"\n{len(common)} paired cases; {len(set(ia) ^ set(ib))} present in only one run.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--corpus", required=True)
    r.add_argument("--tier", choices=["gate", "full"], required=True)
    r.add_argument("--profiles", required=True)
    r.add_argument("--engine-commit", required=True)
    r.add_argument("--machine-record", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--port", type=int, default=18489)
    r.add_argument("--blocks", default="A,B,C,D,E")
    r.add_argument("--seeds", default=None)
    r.add_argument("--warmups", type=int, default=1)
    r.add_argument("--timeout", type=float, default=900)
    s = sub.add_parser("summarize")
    s.add_argument("run")
    p = sub.add_parser("paired")
    p.add_argument("run_a")
    p.add_argument("run_b")
    a = ap.parse_args(argv)
    {"run": cmd_run, "summarize": cmd_summarize, "paired": cmd_paired}[a.cmd](a)


if __name__ == "__main__":
    main()
