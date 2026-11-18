#!/usr/bin/env python3
"""Checker for Scripts/engine-validate.sh: the engine program's section 8 matrix.

Standard library only. Talks to an OpenAI-compatible server that is already
running on loopback (MferenceServer, or upstream llama-server for the GGUF
reference); it never starts, stops or signals a model process. The shell
script owns process lifetime and the machine lock.

Subcommands:

  registry   print a field of the product-model table (used by the shell)
  models     GET /v1/models: exactly the expected ids, each with `engine`
  prompts    the fixed prompt set, non-streaming and streaming, at
             temperature 0; raw responses, canonical results and an
             agreement verdict per case
  readings   cold 512- and 4K-token prompts: TTFT, prefill and decode tok/s,
             RSS sampled while each request runs
  swap       the swap chain through one server; per step, release-before-load
             from the server log, wall time and sampled peak RSS
  compare    canonical results of two `prompts` runs: byte/token identity
  timeparse  peak RSS and footprint from `/usr/bin/time -l` output
  summary    one summary.md table over a results directory

Every subcommand writes JSON next to its raw outputs and exits non-zero when
its check fails, so the shell can record the verdict without parsing text.
"""
import argparse
import hashlib
import http.client
import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

LOOPBACK = {"127.0.0.1", "localhost", "::1"}
HERE = Path(__file__).resolve().parent

_bench_spec = importlib.util.spec_from_file_location(
    "server_stream_bench", HERE / "server_stream_bench.py")
_bench = importlib.util.module_from_spec(_bench_spec)
_bench_spec.loader.exec_module(_bench)
parse_sse_lines = _bench.parse_sse_lines

# The five product models (spec D1/D2), installed under ~/llm-models. `id` is
# the API id the engine catalog assigns; override any field with ENGINE_VALIDATE_<KEY>_<FIELD> (for example
# ENGINE_VALIDATE_DEEP_ID) if the merged catalog spells one differently.
MODELS = {
    "quick": {
        "id": "qwen3.5-4b",
        "engine": "llama.cpp",
        "kind": "gguf",
        "path": "~/llm-models/gguf/Qwen3.5-4B-UD-Q4_K_XL.gguf",
        "bytes": 2912109728,
    },
    "deep": {
        "id": "qwen3.8-27b",
        "engine": "llama.cpp",
        "kind": "gguf",
        "path": "~/llm-models/gguf/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf",
        "bytes": 12120016960,
    },
    "qwen36": {
        "id": "qwen3.6-35b-a3b",
        "engine": "mference",
        "kind": "gturbo",
        "path": "~/llm-models/qwen36.gturbo",
        "bytes": None,
    },
    "minicpm5": {
        "id": "minicpm5-2b-int4g64",
        "engine": "mference",
        "kind": "gturbo",
        "path": "~/llm-models/minicpm5.gturbo",
        "bytes": None,
    },
    "maple": {
        "id": "maple-preview-2bit-mlx",
        "engine": "mference",
        "kind": "gturbo",
        "path": "~/llm-models/maple.gturbo",
        "bytes": None,
    },
}
PRODUCT_ORDER = ["quick", "deep", "qwen36", "minicpm5", "maple"]
SWAP_CHAIN = ["quick", "deep", "qwen36", "minicpm5", "maple", "quick"]
# Installed but never offered (spec D1): its id must not appear.
FORBIDDEN_IDS = {"qwen3.8-flash-next-int4g64", "unsloth-qwen3.8-flash-next-ud-iq3_xxs"}


def model(key, environ=os.environ):
    if key not in MODELS:
        raise KeyError(f"unknown model key {key!r}; known: {', '.join(PRODUCT_ORDER)}")
    entry = dict(MODELS[key])
    for field in list(entry):
        override = environ.get(f"ENGINE_VALIDATE_{key.upper()}_{field.upper()}")
        if override is not None:
            entry[field] = int(override) if field == "bytes" else override
    if isinstance(entry.get("path"), str):
        entry["path"] = os.path.expanduser(entry["path"])
    entry["key"] = key
    return entry


# ---------------------------------------------------------------------------
# HTTP


def _check_host(host):
    if host not in LOOPBACK:
        raise SystemExit(f"refusing non-loopback host {host!r}")


def get_json(host, port, path, timeout=30.0):
    _check_host(host)
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        body = resp.read().decode("utf-8", "replace")
        return resp.status, (json.loads(body) if body.strip() else None)
    finally:
        conn.close()


def post_chat(host, port, body, timeout=3600.0):
    """One non-streaming request. Returns (status, parsed body, raw text, seconds)."""
    _check_host(host)
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    payload = json.dumps(dict(body, stream=False)).encode("utf-8")
    started = time.perf_counter()
    try:
        conn.request("POST", "/v1/chat/completions", body=payload,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8", "replace")
    finally:
        conn.close()
    elapsed = time.perf_counter() - started
    try:
        parsed = json.loads(raw)
    except ValueError:
        parsed = None
    return resp.status, parsed, raw, elapsed


def post_chat_stream(host, port, body, timeout=3600.0, clock=time.perf_counter):
    """One streaming request. Returns (status, chunks, raw text, ttft, total)."""
    _check_host(host)
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    body = dict(body, stream=True)
    body.setdefault("stream_options", {"include_usage": True})
    payload = json.dumps(body).encode("utf-8")
    started = clock()
    raw_lines = []
    chunks = []
    ttft = None
    try:
        conn.request("POST", "/v1/chat/completions", body=payload,
                     headers={"Content-Type": "application/json",
                              "Accept": "text/event-stream"})
        resp = conn.getresponse()
        if resp.status != 200:
            raw = resp.read().decode("utf-8", "replace")
            return resp.status, [], raw, None, clock() - started

        def lines():
            for line in iter(resp.readline, b""):
                raw_lines.append(line.decode("utf-8", "replace"))
                yield line

        for chunk in parse_sse_lines(lines()):
            if ttft is None and _chunk_has_payload(chunk):
                ttft = clock() - started
            chunks.append(chunk)
    finally:
        conn.close()
    return 200, chunks, "".join(raw_lines), ttft, clock() - started


def _chunk_has_payload(chunk):
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        if delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"):
            return True
    return False


# ---------------------------------------------------------------------------
# Canonical results


def canonical_from_message(body):
    """Canonical result of a non-streaming response."""
    if not isinstance(body, dict) or "error" in body:
        return {"error": (body or {}).get("error") if isinstance(body, dict) else "unparseable"}
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    calls = [{"name": (c.get("function") or {}).get("name"),
              "arguments": (c.get("function") or {}).get("arguments")}
             for c in message.get("tool_calls") or []]
    usage = body.get("usage") or {}
    return {
        "content": message.get("content") or "",
        "reasoning_content": message.get("reasoning_content") or "",
        "tool_calls": calls,
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
    }


def canonical_from_chunks(chunks):
    """Canonical result of a stream: deltas concatenated, tool calls merged by index."""
    content, reasoning = [], []
    calls = {}
    finish = None
    usage = {}
    for chunk in chunks:
        if "error" in chunk:
            return {"error": chunk["error"]}
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning.append(delta["reasoning_content"])
            for call in delta.get("tool_calls") or []:
                slot = calls.setdefault(call.get("index", len(calls)),
                                        {"name": None, "arguments": ""})
                function = call.get("function") or {}
                if function.get("name"):
                    slot["name"] = (slot["name"] or "") + function["name"]
                if function.get("arguments"):
                    slot["arguments"] += function["arguments"]
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
        if chunk.get("usage"):
            usage = chunk["usage"]
    return {
        "content": "".join(content),
        "reasoning_content": "".join(reasoning),
        "tool_calls": [calls[i] for i in sorted(calls)],
        "finish_reason": finish,
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
    }


IDENTITY_FIELDS = ("content", "reasoning_content", "tool_calls", "finish_reason",
                   "completion_tokens")


def digest(result):
    """md5 over the fields that must be byte-identical between two runs. The
    HTTP status is part of it, and an error result never counts as an output
    (`comparable`), so two identical rejections are not identity."""
    material = {k: result.get(k) for k in IDENTITY_FIELDS}
    material["http_status"] = result.get("http_status")
    if "error" in result:
        material = {"error": result["error"], "http_status": result.get("http_status")}
    blob = json.dumps(material, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.md5(blob).hexdigest()


def comparable(result):
    return "error" not in result and result.get("http_status", 200) == 200


def disagreements(a, b):
    return [k for k in IDENTITY_FIELDS if a.get(k) != b.get(k)] + (
        ["error"] if ("error" in a) != ("error" in b) else [])


# ---------------------------------------------------------------------------
# Prompt set (fixed; deterministic; no files to drift)


def _lcg(seed):
    state = seed & 0xFFFFFFFF
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state


ITEMS = ["brass hinges", "copper wire", "oak dowels", "steel brackets", "glass jars",
         "wool blankets", "ceramic tiles", "rubber gaskets", "linen sacks", "pine planks",
         "tin buckets", "cotton rope", "iron nails", "clay pots", "felt pads"]
SUPPLIERS = ["Halden Supply", "Brightwater Trading", "Orsk & Daughters", "Kestrel Freight",
             "Marlow Goods", "Pellam Works", "Vantor Imports", "Ashby Mercantile"]


def ledger(entries, seed=20261118):
    """A deterministic warehouse ledger, about 25 tokens per entry."""
    rng = _lcg(seed)
    lines = []
    for n in range(1, entries + 1):
        item = ITEMS[next(rng) % len(ITEMS)]
        supplier = SUPPLIERS[next(rng) % len(SUPPLIERS)]
        shelf = f"{'ABCDEFGH'[next(rng) % 8]}-{1 + next(rng) % 12}"
        count = 10 + next(rng) % 990
        day = 1 + next(rng) % 360
        lines.append(f"Entry {n}: {count} units of {item} on shelf {shelf}, "
                     f"received from {supplier} on day {day}.")
    return lines


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}


def prompt_cases():
    """The section 8 item 2 set. Each case is a list of turns; a turn is either
    a request body fragment or a callable building one from the previous turn's
    canonical result (the tool round-trip)."""
    doc = ledger(80)
    target = doc[56]  # Entry 57
    shelf = re.search(r"shelf (\S+),", target).group(1)
    doc_question = (
        "Here is a warehouse ledger.\n\n" + "\n".join(doc) + "\n\n"
        f"Question: which supplier delivered the goods on shelf {shelf} recorded in "
        "entry 57, and how many units were there? Answer in one sentence.")

    def tool_followup(first):
        calls = first.get("tool_calls") or []
        if not calls:
            return None
        call = calls[0]
        return {
            "messages": [
                {"role": "user", "content": "What is the weather in Lisbon right now? Use the tool."},
                {"role": "assistant", "content": "",
                 "tool_calls": [{"id": "call_0", "type": "function",
                                 "function": {"name": call["name"],
                                              "arguments": call["arguments"]}}]},
                {"role": "tool", "tool_call_id": "call_0",
                 "content": json.dumps({"city": "Lisbon", "temperature_c": 18,
                                        "conditions": "light rain"})},
            ],
            "tools": [WEATHER_TOOL],
            "max_tokens": 256,
        }

    return [
        {"name": "short-chat", "turns": [{
            "messages": [{"role": "user",
                          "content": "In two sentences, explain what a hash table is."}],
            "max_tokens": 256}]},
        {"name": "doc-2k", "turns": [{
            "messages": [{"role": "user", "content": doc_question}],
            "max_tokens": 256}]},
        {"name": "tool-roundtrip", "turns": [
            {"messages": [{"role": "user",
                           "content": "What is the weather in Lisbon right now? Use the tool."}],
             "tools": [WEATHER_TOOL], "max_tokens": 512},
            tool_followup]},
        {"name": "json-answer", "turns": [{
            "messages": [{"role": "user", "content":
                          "Return only a JSON object with the keys \"city\" (string) and "
                          "\"population_millions\" (number) for the most populous city in "
                          "Japan. No prose, no code fence."}],
            "max_tokens": 256}]},
    ]


def base_body(model_id, extra):
    body = {"model": model_id, "temperature": 0, "top_k": 1, "top_p": 1.0, "seed": 0}
    body.update(extra or {})
    return body


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def json_answer_ok(content):
    """The answer parses as the requested object. A Markdown code fence around
    it is tolerated: the case checks structured output, not instruction-following."""
    text = content.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S)
    if fence:
        text = fence.group(1)
    try:
        value = json.loads(text)
    except ValueError:
        return False
    return isinstance(value, dict) and {"city", "population_millions"} <= set(value)


def cmd_prompts(args):
    out = Path(args.out)
    extra = json.loads(args.extra) if args.extra else {}
    report = {"model": args.model, "cases": {}}
    ok = True
    for case in prompt_cases():
        name = case["name"]
        previous = {"nonstream": None, "stream": None}
        turns = []
        for index, turn in enumerate(case["turns"], start=1):
            # A follow-up turn is built once, from the non-streaming result, so
            # both modes send byte-identical history.
            if callable(turn):
                turn = turn(previous["nonstream"]) if previous["nonstream"] else None
                if turn is None:
                    turns.append({"turn": index, "skipped": "previous turn made no tool call"})
                    continue
            body = base_body(args.model, extra)
            body.update(turn)
            stem = f"{name}.turn{index}"
            _write(out / f"{stem}.request.json", json.dumps(body, indent=2, sort_keys=True))

            status, parsed, raw, seconds = post_chat(args.host, args.port, body, args.timeout)
            _write(out / f"{stem}.nonstream.raw.json", raw)
            nonstream = canonical_from_message(parsed)
            nonstream["http_status"] = status
            nonstream["seconds"] = seconds

            status_s, chunks, raw_s, ttft, total = post_chat_stream(
                args.host, args.port, body, args.timeout)
            _write(out / f"{stem}.stream.raw.sse", raw_s)
            stream = canonical_from_chunks(chunks) if status_s == 200 else {"error": raw_s}
            stream["http_status"] = status_s
            stream["ttft_s"] = ttft
            stream["seconds"] = total

            diff = disagreements(nonstream, stream)
            record = {
                "turn": index,
                "nonstream": nonstream, "stream": stream,
                "md5_nonstream": digest(nonstream), "md5_stream": digest(stream),
                "agree": not diff and status == 200 and status_s == 200,
                "differs_in": diff,
            }
            if name == "json-answer":
                # Agreement on output that is not the requested JSON exercised nothing.
                record["json_valid"] = json_answer_ok(nonstream.get("content", ""))
                if not record["json_valid"]:
                    record["agree"] = False
                    record["differs_in"] = diff + ["invalid JSON answer"]
            if name == "tool-roundtrip" and index == 1:
                # A round-trip that never calls the tool exercised nothing.
                record["tool_call_emitted"] = bool(nonstream.get("tool_calls"))
                if not record["tool_call_emitted"]:
                    record["agree"] = False
                    record["differs_in"] = diff + ["no tool call"]
            ok = ok and record["agree"]
            turns.append(record)
            previous = {"nonstream": nonstream, "stream": stream}
            print(f"[{args.model}] {stem}: agree={record['agree']} "
                  f"md5={record['md5_nonstream']} finish={nonstream.get('finish_reason')} "
                  f"tokens={nonstream.get('completion_tokens')}", file=sys.stderr, flush=True)
        report["cases"][name] = turns
    report["all_agree"] = ok
    _write(out / "prompts.json", json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Comparison between two runs (G5 native vs main, G6 engine vs upstream, MTP on/off)


def load_digests(directory):
    """{case: md5 or None}; None marks an output that cannot count as identical
    (skipped turn, HTTP error, error chunk). Missing prompts.json -> None."""
    path = Path(directory) / "prompts.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    digests = {}
    for name, turns in report["cases"].items():
        for turn in turns:
            key = f"{name}.turn{turn['turn']}"
            if "skipped" in turn:
                digests[key] = None
                continue
            digests[key] = turn["md5_nonstream"] if comparable(turn["nonstream"]) else None
            digests[key + ".stream"] = turn["md5_stream"] if comparable(turn["stream"]) else None
    return digests


def compare_digests(a, b):
    if a is None or b is None:
        return {"identical": False, "rows": [],
                "missing": [side for side, d in (("a", a), ("b", b)) if d is None]}
    keys = sorted(set(a) | set(b))
    rows = [{"case": k, "a": a.get(k), "b": b.get(k),
             "equal": a.get(k) is not None and a.get(k) == b.get(k)} for k in keys]
    return {"identical": bool(rows) and all(r["equal"] for r in rows), "rows": rows}


def cmd_compare(args):
    result = compare_digests(load_digests(args.a), load_digests(args.b))
    result.update({"a": str(args.a), "b": str(args.b), "label": args.label})
    for side in result.get("missing", []):
        print(f"MISSING {side}: no prompts.json in {getattr(args, side)}", file=sys.stderr)
    _write(Path(args.out), json.dumps(result, indent=2, sort_keys=True))
    for row in result["rows"]:
        mark = "ok  " if row["equal"] else "DIFF"
        print(f"{mark} {row['case']}: {row['a']} {row['b']}", file=sys.stderr)
    return 0 if result["identical"] else 1


# ---------------------------------------------------------------------------
# /v1/models


def check_model_list(body, expected, forbidden=FORBIDDEN_IDS):
    """Problems with a /v1/models body against the expected product entries."""
    problems = []
    data = (body or {}).get("data")
    if not isinstance(data, list):
        return ["response has no `data` list"]
    ids = [m.get("id") for m in data]
    want = {m["id"]: m for m in expected}
    if sorted(ids) != sorted(want):
        missing = sorted(set(want) - set(ids))
        extra = sorted(set(ids) - set(want))
        problems.append(f"ids differ: missing={missing} unexpected={extra}")
    for entry in data:
        if entry.get("id") in forbidden:
            problems.append(f"forbidden model listed: {entry.get('id')}")
        expect = want.get(entry.get("id"))
        if expect is None:
            continue
        if entry.get("engine") != expect["engine"]:
            problems.append(f"{entry.get('id')}: engine={entry.get('engine')!r}, "
                            f"want {expect['engine']!r}")
        if not isinstance(entry.get("max_model_len"), int):
            problems.append(f"{entry.get('id')}: max_model_len missing")
    return problems


def cmd_models(args):
    expected = [model(k) for k in args.expect.split(",") if k]
    status, body = get_json(args.host, args.port, "/v1/models")
    problems = [f"HTTP {status}"] if status != 200 else check_model_list(body, expected)
    result = {"status": status, "body": body, "problems": problems, "ok": not problems}
    _write(Path(args.out), json.dumps(result, indent=2, sort_keys=True))
    for p in problems:
        print(f"FAIL {p}", file=sys.stderr)
    return 0 if not problems else 1


# ---------------------------------------------------------------------------
# Memory sampling


def rss_kb(pid):
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        return int(out) if out else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def parse_swapusage(text):
    """`used` from `sysctl -n vm.swapusage`, in MiB."""
    m = re.search(r"used\s*=\s*([0-9.]+)([MG])", text or "")
    if not m:
        return None
    value = float(m.group(1))
    return value * 1024 if m.group(2) == "G" else value


def swap_used_mib():
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_swapusage(out)


class Sampler:
    """Samples the server's RSS and system swap while a block runs."""

    def __init__(self, pid, interval=0.25):
        self.pid = pid
        self.interval = interval
        self.peak_rss_kb = 0
        self.swap_start = None
        self.swap_peak = None
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        self.swap_start = swap_used_mib()
        self.swap_peak = self.swap_start
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        ticks = 0
        while not self._stop.is_set():
            if self.pid:
                value = rss_kb(self.pid)
                if value:
                    self.peak_rss_kb = max(self.peak_rss_kb, value)
            if ticks % 8 == 0:
                swap = swap_used_mib()
                if swap is not None:
                    self.swap_peak = swap if self.swap_peak is None else max(self.swap_peak, swap)
            ticks += 1
            self._stop.wait(self.interval)

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        return False

    @property
    def swap_grew(self):
        # Any growth trips the guard (spec section 6); no tolerance.
        return (self.swap_start is not None and self.swap_peak is not None
                and self.swap_peak > self.swap_start)

    def report(self):
        return {"peak_rss_mib": round(self.peak_rss_kb / 1024, 1) if self.peak_rss_kb else None,
                "swap_start_mib": self.swap_start, "swap_peak_mib": self.swap_peak,
                "swap_grew": self.swap_grew}


SWAP_TRIP = 90  # exit status: swap guard tripped, stop everything


def parse_time_l(text):
    """Peak figures from `/usr/bin/time -l` (macOS) output, in bytes."""
    result = {}
    for key, pattern in (("max_rss_bytes", r"(\d+)\s+maximum resident set size"),
                         ("peak_footprint_bytes", r"(\d+)\s+peak memory footprint")):
        m = re.search(pattern, text or "")
        result[key] = int(m.group(1)) if m else None
    return result


def cmd_swapusage(args):
    used = swap_used_mib()
    if used is None:
        print("swap usage unavailable", file=sys.stderr)
        return 1
    print(f"{used:.2f}")
    return 0


def cmd_probe(args):
    """One-token request: makes library mode load the model, records the time."""
    body = base_body(args.model, json.loads(args.extra) if args.extra else {})
    body.update({"messages": FLUSH, "max_tokens": 1})
    status, _, raw, seconds = post_chat(args.host, args.port, body, args.timeout)
    _write(Path(args.out), json.dumps({"http_status": status, "seconds": seconds,
                                       "body": raw[:2000]}, indent=2, sort_keys=True))
    if status != 200:
        print(f"probe failed: HTTP {status}: {raw[:300]}", file=sys.stderr)
    return 0 if status == 200 else 1


def cmd_timeparse(args):
    text = Path(args.file).read_text(encoding="utf-8", errors="replace")
    print(json.dumps(parse_time_l(text), sort_keys=True))
    return 0


# ---------------------------------------------------------------------------
# Readings


FLUSH = [{"role": "user", "content": "Reply with the single word OK."}]


def filler(paragraphs, nonce):
    lines = ledger(paragraphs, seed=7 + paragraphs)
    return (f"Session {nonce}. Read this ledger.\n\n" + "\n".join(lines)
            + "\n\nIn one sentence, which supplier appears most often?")


def prompt_tokens_for(args, content):
    status, body, raw, _ = post_chat(args.host, args.port, base_body(args.model, {
        "messages": [{"role": "user", "content": content}], "max_tokens": 1}), args.timeout)
    if status != 200:
        raise RuntimeError(f"probe failed: HTTP {status}: {raw[:300]}")
    return body["usage"]["prompt_tokens"]


def entries_for_target(args, target):
    """Ledger entries giving about `target` prompt tokens, from two probes."""
    small = prompt_tokens_for(args, filler(4, "probe-a"))
    large = prompt_tokens_for(args, filler(64, "probe-b"))
    per_entry = max((large - small) / 60.0, 1.0)
    return max(1, round((target - small) / per_entry) + 4)


def cmd_readings(args):
    extra = json.loads(args.extra) if args.extra else {}
    results = {"model": args.model, "readings": []}
    tripped = False
    for target in [int(t) for t in args.targets.split(",")]:
        entries = entries_for_target(args, target)
        for run in range(args.runs):
            post_chat(args.host, args.port, base_body(args.model, {
                "messages": FLUSH, "max_tokens": 1}), args.timeout)
            nonce = f"{target}-{run}-{time.time_ns()}"
            body = base_body(args.model, extra)
            body.update({"messages": [{"role": "user", "content": filler(entries, nonce)}],
                         "max_tokens": args.max_tokens})
            with Sampler(args.pid) as sampler:
                status, chunks, raw, ttft, total = post_chat_stream(
                    args.host, args.port, body, args.timeout)
            if status != 200:
                raise RuntimeError(f"reading failed: HTTP {status}: {raw[:300]}")
            usage, timings = {}, {}
            for chunk in chunks:
                usage = chunk.get("usage") or usage
                timings = chunk.get("timings") or timings
            reading = {
                "target_tokens": target, "run": run,
                "prompt_tokens": usage.get("prompt_tokens"),
                "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                "completion_tokens": usage.get("completion_tokens"),
                "ttft_s": ttft, "total_s": total,
                "prefill_tok_s": timings.get("prompt_per_second"),
                "decode_tok_s": timings.get("predicted_per_second"),
                "timings": timings,
            }
            reading.update(sampler.report())
            reading["cold_violation"] = bool(reading["cached_tokens"])
            results["readings"].append(reading)
            print(f"[{args.model}] {target} run{run}: prompt={reading['prompt_tokens']} "
                  f"ttft={ttft and round(ttft, 2)}s prefill={reading['prefill_tok_s']} "
                  f"decode={reading['decode_tok_s']} rss={reading['peak_rss_mib']}MiB",
                  file=sys.stderr, flush=True)
            if sampler.swap_grew:
                tripped = True
                break
        if tripped:
            break
    results["swap_trip"] = tripped
    _write(Path(args.out), json.dumps(results, indent=2, sort_keys=True))
    if tripped:
        return SWAP_TRIP
    # A reading the prefix cache served is not a cold measurement.
    return 1 if any(r["cold_violation"] for r in results["readings"]) else 0


# ---------------------------------------------------------------------------
# Swap chain


LOG_LINE = re.compile(r"^\[[^\]]*\]\s+(.*)$")


def swap_events(log_text):
    """Ordered (kind, fields) events from the server log."""
    events = []
    for raw in (log_text or "").splitlines():
        m = LOG_LINE.match(raw.strip())
        line = m.group(1) if m else raw.strip()
        if line.startswith("swap started"):
            fields = dict(re.findall(r"(\w+)=(\S+)", line))
            events.append(("started", fields))
        elif line.startswith("swap finished"):
            fields = dict(re.findall(r"(\w+)=(\S+)", line))
            d = re.search(r" in ([0-9.]+)s", line)
            fields["seconds"] = float(d.group(1)) if d else None
            events.append(("finished", fields))
        elif line.startswith("swap failed"):
            events.append(("failed", dict(re.findall(r"(\w+)=(\S+)", line))))
        elif line.startswith("model released"):
            events.append(("released", dict(re.findall(r"(\w+)=(\S+)", line))))
    return events


def verify_swap(events, previous, target, require_release=True):
    """Release-before-load check for one step, over the events that step logged."""
    problems = []
    kinds = [k for k, _ in events]
    if "failed" in kinds:
        problems.append("swap failed")
    try:
        start = next(i for i, (k, f) in enumerate(events) if k == "started" and f.get("to") == target)
    except StopIteration:
        return {"ok": False, "problems": [f"no 'swap started to={target}'"], "load_s": None}
    if previous is not None and events[start][1].get("from") != previous:
        problems.append(f"swap started from={events[start][1].get('from')}, want {previous}")
    finish = next((i for i, (k, f) in enumerate(events)
                   if i > start and k == "finished" and f.get("model") == target), None)
    if finish is None:
        problems.append(f"no 'swap finished model={target}'")
    if previous is not None and require_release:
        # The server logs `swap started`, then `model released` for the old
        # model once its memory is gone, then loads and logs `swap finished`.
        released = next((i for i, (k, f) in enumerate(events)
                         if k == "released" and f.get("model") == previous), None)
        if released is None:
            problems.append(f"no 'model released model={previous}'")
        elif released < start:
            problems.append("'model released' logged before 'swap started'")
        elif finish is not None and released > finish:
            problems.append("old model released after the new one finished loading")
    return {"ok": not problems, "problems": problems,
            "load_s": events[finish][1].get("seconds") if finish is not None else None}


RSS_MARGIN = 0.10


def max_single_rss_mib(results):
    """Largest peak RSS of any per-model engine run under `results`, in MiB."""
    peaks = []
    for run in Path(results).glob("*/engine*/run.json"):
        value = (_load(run) or {}).get("max_rss_bytes")
        if value:
            peaks.append(value / 1048576)
    return max(peaks) if peaks else None


def cmd_maxrss(args):
    value = max_single_rss_mib(args.results)
    if value is None:
        return 1
    print(f"{value:.1f}")
    return 0


def cmd_swap(args):
    chain = [model(k) for k in args.chain.split(",")]
    log_path = Path(args.log)
    results = {"chain": [m["id"] for m in chain], "steps": []}
    ok, tripped = True, False
    previous = None
    for step, entry in enumerate(chain):
        offset = log_path.stat().st_size if log_path.exists() else 0
        body = base_body(entry["id"], {"messages": [{"role": "user", "content": "Say hello."}],
                                       "max_tokens": 8})
        with Sampler(args.pid) as sampler:
            status, parsed, raw, seconds = post_chat(args.host, args.port, body, args.timeout)
        time.sleep(0.5)  # let the log flush
        with open(log_path, "rb") as f:
            f.seek(offset)
            new_text = f.read().decode("utf-8", "replace")
        verdict = verify_swap(swap_events(new_text), previous, entry["id"],
                              require_release=not args.no_release_line)
        if status != 200:
            verdict["ok"] = False
            verdict["problems"].append(f"HTTP {status}: {raw[:200]}")
        # Two models resident at once would show as a step peak well above the
        # largest single-model peak measured in the per-model runs.
        if args.max_single_rss_mib and previous is not None:
            bound = args.max_single_rss_mib * (1 + RSS_MARGIN)
            peak = sampler.report()["peak_rss_mib"]
            verdict["rss_bound_mib"] = round(bound, 1)
            if peak is None:
                verdict["ok"] = False
                verdict["problems"].append("no RSS samples for the bound")
            elif peak > bound:
                verdict["ok"] = False
                verdict["problems"].append(
                    f"peak RSS {peak} MiB above one model's bound {bound:.1f} MiB")
        record = {"step": step, "from": previous, "to": entry["id"], "http_status": status,
                  "request_s": seconds, **verdict, **sampler.report(), "log": new_text}
        results["steps"].append(record)
        ok = ok and verdict["ok"]
        print(f"swap {previous} -> {entry['id']}: ok={verdict['ok']} load={verdict['load_s']}s "
              f"peak_rss={record['peak_rss_mib']}MiB {verdict['problems']}",
              file=sys.stderr, flush=True)
        if sampler.swap_grew:
            tripped = True
            break
        previous = entry["id"]
    results["ok"] = ok and not tripped
    results["swap_trip"] = tripped
    _write(Path(args.out), json.dumps(results, indent=2, sort_keys=True))
    if tripped:
        return SWAP_TRIP
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Speed bar (section 8 item 4): engine no more than 5% slower than reference


SPEED_TOLERANCE = 0.05


def speed_rows(engine, reference):
    """Median rates per prompt size. Cache-served (`cold_violation`) readings
    never count; a size with any of them fails rather than passing on the rest."""
    rows = []
    for target in sorted({r["target_tokens"] for r in engine + reference}):
        mine = [r for r in engine if r["target_tokens"] == target]
        theirs = [r for r in reference if r["target_tokens"] == target]
        contaminated = any(r.get("cold_violation") for r in mine + theirs)
        for field in ("prefill_tok_s", "decode_tok_s"):
            e = _median([r.get(field) for r in mine if not r.get("cold_violation")])
            ref = _median([r.get(field) for r in theirs if not r.get("cold_violation")])
            ratio = e / ref if e and ref else None
            rows.append({"target_tokens": target, "metric": field, "engine": e,
                         "reference": ref, "ratio": ratio, "cold_violation": contaminated,
                         "ok": (not contaminated and ratio is not None
                                and ratio >= 1 - SPEED_TOLERANCE)})
    return rows


def cmd_speed(args):
    a = (_load(Path(args.a) / "readings.json") or {}).get("readings")
    b = (_load(Path(args.b) / "readings.json") or {}).get("readings")
    if not a or not b:
        result = {"label": args.label, "ok": False, "rows": [],
                  "missing": [side for side, r in (("a", a), ("b", b)) if not r]}
    else:
        rows = speed_rows(a, b)
        result = {"label": args.label, "ok": bool(rows) and all(r["ok"] for r in rows),
                  "rows": rows}
    result.update({"a": str(args.a), "b": str(args.b), "tolerance": SPEED_TOLERANCE})
    _write(Path(args.out), json.dumps(result, indent=2, sort_keys=True))
    for row in result["rows"]:
        print(f"{'ok  ' if row['ok'] else 'SLOW'} {row['target_tokens']} {row['metric']}: "
              f"engine={row['engine']} reference={row['reference']} ratio={row['ratio']}",
              file=sys.stderr)
    return 0 if result["ok"] else 1


# ---------------------------------------------------------------------------
# Run records (written by the shell's exit handler)


def _num(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def run_record(run_dir, key, variant, started, ready_at, swap_before, swap_after, status):
    log_path = Path(run_dir) / "server.log"
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    loads = [float(x) for x in re.findall(r"swap finished model=\S+ in ([0-9.]+)s", log)]
    load = loads[0] if loads else None
    if load is None and variant.startswith("upstream") and _num(ready_at) and _num(started):
        load = _num(ready_at) - _num(started)
    before, after = _num(swap_before), _num(swap_after)
    return {"key": key, "variant": variant, "status": int(status or 0), "load_s": load,
            "swap_before_mib": before, "swap_after_mib": after,
            "swap_delta_mib": after - before if before is not None and after is not None else None,
            **parse_time_l(log)}


def cmd_runrecord(args):
    record = run_record(args.run_dir, args.key, args.variant, args.started, args.ready_at,
                        args.swap_before, args.swap_after, args.status)
    _write(Path(args.run_dir) / "run.json", json.dumps(record, indent=2, sort_keys=True))
    return 0


def cmd_swapgrew(args):
    """Exit 0 when the run recorded any swap growth (or could not measure it)."""
    record = _load(args.run_json) or {}
    if record.get("swap_before_mib") is None or record.get("swap_after_mib") is None:
        return 0
    return 0 if record["swap_delta_mib"] > 0 else 1


def cmd_meta(args):
    def sh(*cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
        except (OSError, subprocess.SubprocessError) as error:
            return f"unavailable: {error}"

    def stamp(path):
        if not path or not os.path.exists(path):
            return None
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(os.path.getmtime(path)))

    swift = sh("swift", "--version")
    meta = {
        "commit": sh("git", "-C", args.repo, "rev-parse", "HEAD"),
        "branch": sh("git", "-C", args.repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(sh("git", "-C", args.repo, "status", "--porcelain")),
        "hardware": sh("sysctl", "-n", "machdep.cpu.brand_string"),
        "memory_bytes": sh("sysctl", "-n", "hw.memsize"),
        "macos": sh("sw_vers", "-productVersion") + " " + sh("sw_vers", "-buildVersion"),
        "swift": swift.splitlines()[0] if swift else None,
        "server": args.server, "server_mtime": stamp(args.server),
        "baseline_server": args.baseline or None, "baseline_mtime": stamp(args.baseline),
        "upstream_llama_server": args.upstream or None,
        "argv": args.argv, "phases": args.phases,
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write(Path(args.out), json.dumps(meta, indent=2, sort_keys=True))
    return 0


# ---------------------------------------------------------------------------
# Summary


def _load(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _median(values):
    values = sorted(v for v in values if isinstance(v, (int, float)))
    if not values:
        return None
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


def _fmt(value, digits=1):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "NO"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def summarize(root):
    root = Path(root)
    lines = [f"# Engine validation summary: {root.name}", ""]
    meta = _load(root / "meta.json") or {}
    if meta:
        lines += ["| Field | Value |", "| --- | --- |"]
        lines += [f"| {k} | {str(v).replace('|', '/')} |" for k, v in sorted(meta.items())]
        lines.append("")
    models = _load(root / "models.json")
    if models:
        lines.append(f"`/v1/models`: {'PASS' if models.get('ok') else 'FAIL'} "
                     + "; ".join(models.get("problems") or []))
        lines.append("")

    lines += ["## Per model run", "",
              "| Model | Run | Agree (stream = non-stream) | JSON valid | Tool call | Load s | "
              "512 TTFT s | 512 prefill tok/s | 512 decode tok/s | 4K TTFT s | "
              "4K prefill tok/s | 4K decode tok/s | Peak RSS MiB | Peak footprint MiB | "
              "Swap delta MiB |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for run_dir in sorted(p for p in root.glob("*/*") if (p / "run.json").exists()):
        run = _load(run_dir / "run.json") or {}
        prompts = _load(run_dir / "prompts.json") or {}
        readings = (_load(run_dir / "readings.json") or {}).get("readings", [])
        cases = prompts.get("cases", {})
        json_valid = next((t.get("json_valid") for t in cases.get("json-answer", [])), None)
        tool = next((t.get("tool_call_emitted") for t in cases.get("tool-roundtrip", [])), None)

        def pick(target, field):
            return _median([r.get(field) for r in readings if r.get("target_tokens") == target])

        footprint = run.get("peak_footprint_bytes")
        lines.append("| " + " | ".join([
            run_dir.parent.name, run_dir.name,
            _fmt(prompts.get("all_agree")) if prompts else "-",
            _fmt(json_valid), _fmt(tool), _fmt(run.get("load_s")),
            _fmt(pick(512, "ttft_s"), 2), _fmt(pick(512, "prefill_tok_s")),
            _fmt(pick(512, "decode_tok_s")), _fmt(pick(4096, "ttft_s"), 2),
            _fmt(pick(4096, "prefill_tok_s")), _fmt(pick(4096, "decode_tok_s")),
            _fmt(max([r.get("peak_rss_mib") or 0 for r in readings], default=None) or None),
            _fmt(footprint / 1048576 if footprint else None),
            _fmt(run.get("swap_delta_mib")),
        ]) + " |")
    lines.append("")

    speeds = sorted(root.glob("speed-*.json"))
    if speeds:
        lines += ["## Speed bar (engine >= 95% of reference)", "",
                  "| Comparison | Pass | Rows below the bar |", "| --- | --- | --- |"]
        for path in speeds:
            c = _load(path) or {}
            slow = [f"{r['target_tokens']} {r['metric']} {r['ratio']:.3f}" if r.get("ratio")
                    else f"{r['target_tokens']} {r['metric']} n/a"
                    for r in c.get("rows", []) if not r.get("ok")]
            slow += [f"missing {m}" for m in c.get("missing", [])]
            lines.append(f"| {c.get('label') or path.stem} | {_fmt(bool(c.get('ok')))} | "
                         f"{', '.join(slow) or '-'} |")
        lines.append("")

    comparisons = sorted(root.glob("identity-*.json"))
    if comparisons:
        lines += ["## Identity", "", "| Comparison | Identical | Differing cases |",
                  "| --- | --- | --- |"]
        for path in comparisons:
            c = _load(path) or {}
            bad = [r["case"] for r in c.get("rows", []) if not r.get("equal")]
            bad += [f"missing {m}" for m in c.get("missing", [])]
            lines.append(f"| {c.get('label') or path.stem} | {_fmt(bool(c.get('identical')))} | "
                         f"{', '.join(bad) or '-'} |")
        lines.append("")

    swap = _load(root / "swap" / "swap.json")
    if swap:
        lines += ["## Swap chain", "", "| From | To | OK | Load s | Peak RSS MiB | Swap grew | Problems |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for s in swap.get("steps", []):
            lines.append(f"| {s.get('from') or '-'} | {s.get('to')} | {_fmt(s.get('ok'))} | "
                         f"{_fmt(s.get('load_s'))} | {_fmt(s.get('peak_rss_mib'))} | "
                         f"{'YES' if s.get('swap_grew') else 'no'} | {'; '.join(s.get('problems') or []) or '-'} |")
        lines.append("")
    trips = sorted(str(p.relative_to(root)) for p in root.rglob("SWAP_TRIP"))
    if trips:
        lines += ["**SWAP GUARD TRIPPED**: " + ", ".join(trips), ""]
    return "\n".join(lines) + "\n"


def cmd_summary(args):
    text = summarize(args.results)
    _write(Path(args.results) / "summary.md", text)
    print(text)
    return 0


def cmd_registry(args):
    if args.key == "all":
        print(",".join(PRODUCT_ORDER))
        return 0
    if args.key == "chain":
        print(" ".join(dict.fromkeys(SWAP_CHAIN)))
        return 0
    try:
        value = model(args.key)[args.field]
    except KeyError as error:
        print(f"engine_validate: {error.args[0]}", file=sys.stderr)
        return 2
    print("" if value is None else value)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def server(sp):
        sp.add_argument("--host", default="127.0.0.1")
        sp.add_argument("--port", type=int, required=True)
        sp.add_argument("--timeout", type=float, default=3600.0)

    sp = sub.add_parser("registry")
    sp.add_argument("key")
    sp.add_argument("field", nargs="?", default="id")
    sp.set_defaults(func=cmd_registry)

    sp = sub.add_parser("models")
    server(sp)
    sp.add_argument("--expect", default=",".join(PRODUCT_ORDER))
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_models)

    sp = sub.add_parser("prompts")
    server(sp)
    sp.add_argument("--model", required=True)
    sp.add_argument("--extra", help="JSON object merged into every request body")
    sp.add_argument("--out", required=True, help="directory")
    sp.set_defaults(func=cmd_prompts)

    sp = sub.add_parser("readings")
    server(sp)
    sp.add_argument("--model", required=True)
    sp.add_argument("--pid", type=int, default=0, help="server PID to sample RSS from")
    sp.add_argument("--targets", default="512,4096")
    sp.add_argument("--runs", type=int, default=2)
    sp.add_argument("--max-tokens", type=int, default=128)
    sp.add_argument("--extra")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_readings)

    sp = sub.add_parser("swap")
    server(sp)
    sp.add_argument("--chain", default=",".join(SWAP_CHAIN))
    sp.add_argument("--log", required=True, help="the server's stderr log file")
    sp.add_argument("--pid", type=int, default=0)
    sp.add_argument("--no-release-line", action="store_true",
                    help="do not require a 'model released' log line per swap")
    sp.add_argument("--max-single-rss-mib", type=float, default=None,
                    help="largest single-model peak RSS; a step above it +10%% fails")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_swap)

    for name in ("compare", "identity"):
        sp = sub.add_parser(name)
        sp.add_argument("a")
        sp.add_argument("b")
        sp.add_argument("--label")
        sp.add_argument("--out", required=True)
        sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser("speed")
    sp.add_argument("a", help="engine run directory")
    sp.add_argument("b", help="reference run directory")
    sp.add_argument("--label")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_speed)

    sp = sub.add_parser("maxrss")
    sp.add_argument("results")
    sp.set_defaults(func=cmd_maxrss)

    sp = sub.add_parser("runrecord")
    sp.add_argument("run_dir")
    for flag in ("--key", "--variant", "--started", "--ready-at", "--swap-before",
                 "--swap-after", "--status"):
        sp.add_argument(flag, default="")
    sp.set_defaults(func=cmd_runrecord)

    sp = sub.add_parser("swapgrew")
    sp.add_argument("run_json")
    sp.set_defaults(func=cmd_swapgrew)

    sp = sub.add_parser("meta")
    sp.add_argument("out")
    for flag in ("--repo", "--server", "--baseline", "--upstream", "--argv", "--phases"):
        sp.add_argument(flag, default="")
    sp.set_defaults(func=cmd_meta)

    sp = sub.add_parser("probe")
    server(sp)
    sp.add_argument("--model", required=True)
    sp.add_argument("--extra")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_probe)

    sp = sub.add_parser("swapusage")
    sp.set_defaults(func=cmd_swapusage)

    sp = sub.add_parser("timeparse")
    sp.add_argument("file")
    sp.set_defaults(func=cmd_timeparse)

    sp = sub.add_parser("summary")
    sp.add_argument("results")
    sp.set_defaults(func=cmd_summary)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
