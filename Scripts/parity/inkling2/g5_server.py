#!/usr/bin/env python3
"""G5-server: fixed chat requests against a running MferenceServer, and a
comparison of two such runs (stdlib only).

    python3 -I Scripts/parity/inkling2/g5_server.py run --port N --model ID --family minicpm5|qwen36|maple --out FILE.jsonl
    python3 -I Scripts/parity/inkling2/g5_server.py compare A.jsonl B.jsonl --families F[,F...]

`run` sends the cases ($INK/inbox/k0-notice-g5-server-cases.md) in a fixed
order to 127.0.0.1:N and appends one JSON row per case to FILE.jsonl: the
request, the HTTP status, the raw body (or SSE text) and the normalized
fields. `compare` matches rows by (model, case) and compares the normalized
fields and the request against every case of the planned families
(`--families`: one for a --model server, all three for --library), so a
family whose model wrote no rows on either side counts as missing; it exits 1 on any difference, 2 when a case is missing on either side
(or there are no rows), 0 when every case is the same. A file with a
duplicate (model, case) is refused.

Normalized: status; for an error, its type, param and code (the whole body
when it has no error object); message content and reasoning_content exactly
("" and null stay distinct); tool calls as (name, parsed arguments), ids
dropped, and the arguments strings as sent (a difference in the string alone,
parsed JSON equal, is reported on its own line); finish_reason; every integer
usage field (nested ones as `parent.field`). A stream keeps its concatenated
deltas, the assembled tool calls with their indexes, every finish_reason and
usage frame, every frame that is not a delta (in-band errors included), the
lines it could not parse and whether `data: [DONE]` came; SSE comment lines
(the timed `: ping` keepalives) are skipped. Ids, `created`,
timings and logs are not compared. Each stream is also checked against its
non-stream twin on the same server, loosely, as a printed note that does not
gate.
"""

import http.client
import json
import os
import sys
import urllib.error
import urllib.request

SYSTEM = "You are a concise assistant."
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["city", "unit"]}}}]
TOOL_USER = "What is the weather in Paris in celsius? Use the tool."
FIXED_CALL = {"role": "assistant", "content": "", "tool_calls": [{
    "id": "call_0", "type": "function",
    "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\",\"unit\":\"celsius\"}"}}]}
TOOL_RESULT = {"role": "tool", "tool_call_id": "call_0", "content": "{\"temp_c\": 18, \"sky\": \"clear\"}"}
# T1/T2 send chat_template_kwargs.enable_thinking; E1 sends it where the case list expected a refusal.
# Observed on main bccd7546 and trunk 9a639656: Qwen 3.6 thinks; MiniCPM5 refuses it with 400 (its
# thinking is the template default and hidden), so its T1/T2 record that refusal; Maple refuses (E1).
THINKING = {"minicpm5": True, "qwen36": True, "maple": False}
STREAM = {"stream": True, "stream_options": {"include_usage": True}}


def base(messages, max_tokens, **extra):
    body = {"messages": messages, "temperature": 0, "seed": 7, "max_completion_tokens": max_tokens}
    body.update(extra)
    return body


def cases(family):
    """[(case id, request body, or a function of the earlier results giving one)]."""
    p1 = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "List three primary colors, one per line."}]
    out = [("P1", base(p1, 48)),
           ("P2", base(p1, 48, **STREAM)),
           ("P3", lambda got: base(p1 + [{"role": "assistant", "content": got["P1"].get("content") or ""},
                                         {"role": "user", "content": "Now name one fruit of each color."}], 64))]
    t1 = [{"role": "user", "content": "What is 17 * 23? Answer with the number only."}]
    think = {"chat_template_kwargs": {"enable_thinking": True}}
    if THINKING[family]:
        out += [("T1", base(t1, 384, **think)), ("T2", base(t1, 384, **think, **STREAM))]
    else:
        out += [("E1", base(t1, 384, **think))]
    u = [{"role": "user", "content": TOOL_USER}]
    out += [("U1", base(u, 128, tools=TOOLS)),
            ("U2", base(u, 128, tools=TOOLS, **STREAM)),
            ("R1", base(u + [FIXED_CALL, TOOL_RESULT], 96, tools=TOOLS)),
            ("R2", base(u + [FIXED_CALL, TOOL_RESULT], 96, tools=TOOLS, **STREAM))]
    # a long generation: the detokenizer's window path for hundreds of tokens
    story = [{"role": "user", "content": "Write a 250-word story about a lighthouse keeper."}]
    out += [("L1", base(story, 400)), ("L2", base(story, 400, **STREAM))]
    # a stop string that can straddle a token boundary (the stop matcher reads the detokenizer's output)
    out += [("S1", base(p1, 48, stop=["\n\n"])), ("S2", base(p1, 48, stop=["\n\n"], **STREAM))]
    # the same stop string inside visible text: two paragraphs, so it fires between them
    two = [{"role": "user", "content": "Write two short paragraphs about the sea, separated by a blank line."}]
    out += [("S3", base(two, 160, stop=["\n\n"])), ("S4", base(two, 160, stop=["\n\n"], **STREAM))]
    return out


def usage_ints(u):
    if u is not None and not isinstance(u, dict):
        return {"<not an object>": json.dumps(u)}
    out = {}
    for k, v in (u or {}).items():
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            out[k] = v
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, int) and not isinstance(v2, bool):
                    out[f"{k}.{k2}"] = v2
    return out


def tool_calls_raw(calls):
    """The arguments strings exactly as sent (key order and spacing matter to some clients)."""
    return [(c.get("function") or {}).get("arguments") for c in calls or []]


def tool_calls_norm(calls):
    out = []
    for c in calls or []:
        f = c.get("function") or {}
        try:
            args = json.loads(f.get("arguments") or "null")
        except ValueError:
            args = {"<unparsed>": f.get("arguments")}
        out.append({"name": f.get("name"), "arguments": args})
    return out


def post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
        # no (complete) answer: status 0, recorded and compared
        return 0, json.dumps({"error": {"type": "transport_error", "message": f"{type(e).__name__}: {e}"}})


def normalize(status, raw, stream):
    """The compared fields. Non-stream values are kept exactly ("" and null, [] and absent stay
    distinct); a stream keeps every frame that is not a plain delta, every finish_reason and usage
    frame, the [DONE] marker and the count of lines it could not parse."""
    if status != 200:
        try:
            body = json.loads(raw)
        except ValueError:
            return {"status": status, "error_raw": raw}
        err = body.get("error") if isinstance(body, dict) else None
        if not isinstance(err, dict):
            return {"status": status, "error_raw": body}
        return {"status": status, "error_type": err.get("type"), "error_param": err.get("param"),
                "error_code": err.get("code")}
    if not stream:
        try:
            d = json.loads(raw)
            ch = d["choices"][0]
            m = ch.get("message") or {}
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            return {"status": status, "unparsed_body": raw}
        return {"status": status, "content": m.get("content"), "reasoning": m.get("reasoning_content"),
                "tool_calls": tool_calls_norm(m.get("tool_calls")), "tool_calls_raw": tool_calls_raw(m.get("tool_calls")),
                "tool_calls_present": "tool_calls" in m, "finish_reason": ch.get("finish_reason"),
                "usage": usage_ints(d.get("usage")), "usage_present": d.get("usage") is not None}
    content, reasoning, calls, finishes, usages, other, unparsed, done = [], [], {}, [], [], [], [], False
    for line in raw.split("\n"):  # not splitlines(): U+2028 and friends may sit unescaped inside JSON strings
        line = line.rstrip("\r")
        if not line.strip() or line.startswith(":"):
            continue  # blank separators and SSE comments (the server's timed ": ping" keepalives)
        if not line.startswith("data:"):
            other.append({"line": line})
            continue
        payload = line[5:].lstrip(" ")
        if payload == "[DONE]":
            done = True
            continue
        try:
            ev = json.loads(payload)
        except ValueError:
            unparsed.append(payload)
            continue
        if not isinstance(ev, dict) or "choices" not in ev or "error" in ev:
            other.append(ev)  # error frames and anything else that is not a plain delta, in order
            if not isinstance(ev, dict) or "choices" not in ev:
                continue
        if not isinstance(ev.get("choices"), list):
            continue
            continue
        if ev.get("usage") is not None:
            usages.append(usage_ints(ev["usage"]))
        for ch in ev.get("choices") or []:
            dl = ch.get("delta") or {}
            if dl.get("content"):
                content.append(dl["content"])
            if dl.get("reasoning_content"):
                reasoning.append(dl["reasoning_content"])
            for tc in dl.get("tool_calls") or []:
                idx = tc.get("index", "<no index>")  # a missing index is a difference, not slot 0
                key = idx if isinstance(idx, int) and not isinstance(idx, bool) else repr(idx)
                slot = calls.setdefault(key, {"name": None, "arguments": ""})
                f = tc.get("function") or {}
                if f.get("name"):
                    slot["name"] = f["name"]
                slot["arguments"] += f.get("arguments") or ""
            if ch.get("finish_reason"):
                finishes.append(ch["finish_reason"])
    order = sorted(calls, key=lambda k: (isinstance(k, str), str(k) if isinstance(k, str) else k))
    assembled = [{"type": "function", "function": calls[i], "index": i} for i in order]
    return {"status": status, "content": "".join(content) or None, "reasoning": "".join(reasoning) or None,
            "tool_calls": [dict(t, index=c["index"]) for t, c in zip(tool_calls_norm(assembled), assembled)],
            "tool_calls_raw": tool_calls_raw(assembled), "finish_reasons": finishes, "usage_frames": usages,
            "other_frames": other, "unparsed_lines": unparsed, "done": done}


def run(args):
    opt = lambda k: args[args.index(k) + 1]
    port, model, family, out = opt("--port"), opt("--model"), opt("--family"), opt("--out")
    if family not in THINKING:
        sys.exit(f"unknown family {family}")
    if os.path.exists(out):  # one file may hold several models (library mode), never one model twice
        with open(out) as f:
            if any(json.loads(l).get("model") == model for l in f if l.strip()):
                sys.exit(f"{out} already has rows for {model}; use a new file")
    got = {}
    with open(out, "a") as f:
        for cid, body in cases(family):
            if callable(body):
                body = body(got)
            body = dict(body, model=model)
            status, raw = post(port, body)
            norm = normalize(status, raw, bool(body.get("stream")))
            got[cid] = norm
            f.write(json.dumps({"model": model, "family": family, "case": cid, "request": body,
                                "status": status, "raw": raw, "norm": norm}) + "\n")
            f.flush()
            print(f"{model} {cid}: {status} finish={norm.get('finish_reason', norm.get('finish_reasons'))}", flush=True)
    return 0


def loose(n):
    """For the stream-against-twin note only: the stream's view of a non-stream row."""
    return {"content": n.get("content") or None, "reasoning": n.get("reasoning") or None,
            "tool_calls": [{k: v for k, v in t.items() if k != "index"} for t in n.get("tool_calls") or []],
            "tool_calls_raw": n.get("tool_calls_raw") or [],
            "finish_reason": n.get("finish_reason") or (n.get("finish_reasons") or [None])[-1],
            "usage": n.get("usage") or ((n.get("usage_frames") or [{}])[-1])}


def compare(a_path, b_path, families):
    """families: the families the run planned (for library mode all three), so a family whose
    model has no rows on either side still counts as missing."""
    def load(p):
        rows = {}
        with open(p) as f:
            for l in f:
                if not l.strip():
                    continue
                r = json.loads(l)
                k = (r["model"], r["case"])
                if k in rows:
                    sys.exit(f"{p}: {k} appears twice; refusing to compare")
                rows[k] = r
        return rows
    a, b = load(a_path), load(b_path)
    # every case each model's family should have, whether or not either side wrote it
    fams = {r["model"]: r["family"] for r in list(a.values()) + list(b.values())}
    unknown = sorted({f for f in list(fams.values()) + families if f not in THINKING})
    if unknown:
        sys.exit(f"unknown families: {', '.join(unknown)}")
    expected = {(m, c) for m, f in fams.items() for c, _ in cases(f)}
    absent = [f for f in families if f not in fams.values()]
    keys = sorted(expected | set(a) | set(b))
    diffs = missing = raw_only_n = 0
    for f in absent:
        missing += len(cases(f))
        print(f"| ({f}) | all {len(cases(f))} cases | MISSING on both sides: no rows for this family | |")
    print("| Model | Case | Result | Fields that differ |")
    print("|---|---|---|---|")
    for k in keys:
        if k not in a or k not in b:
            missing += 1
            where = "both sides" if k not in a and k not in b else ("A" if k not in a else "B")
            print(f"| {k[0]} | {k[1]} | MISSING on {where} | |")
            continue
        na, nb = a[k]["norm"], b[k]["norm"]
        bad = [f for f in sorted(set(na) | set(nb)) if na.get(f) != nb.get(f)]
        if a[k]["request"] != b[k]["request"]:
            bad.append("request")
        diffs += bool(bad)
        raw_only = bad == ["tool_calls_raw"]  # the arguments string alone, parsed JSON equal
        raw_only_n += raw_only
        label = "DIFFER (arguments string only; parsed JSON equal)" if raw_only else ("DIFFER" if bad else "same")
        print(f"| {k[0]} | {k[1]} | {label} | {', '.join(bad)} |")
        for f in bad:
            va = a[k]["request"] if f == "request" else na.get(f)
            vb = b[k]["request"] if f == "request" else nb.get(f)
            print(f"    {k[1]} {f}: A={json.dumps(va)[:400]}")
            print(f"    {k[1]} {f}: B={json.dumps(vb)[:400]}")
    for side, rows in (("A", a), ("B", b)):  # stream against its non-stream twin, same side: printed, not gated
        for (m, c), r in sorted(rows.items()):
            twin = {"P2": "P1", "T2": "T1", "U2": "U1", "R2": "R1", "L2": "L1", "S2": "S1", "S4": "S3"}.get(c)
            if twin and (m, twin) in rows:
                l1, l2 = loose(rows[(m, twin)]["norm"]), loose(r["norm"])
                fs = [f for f in l1 if l1[f] != l2[f]]
                if fs:
                    print(f"note (not gated): {side} {m} {c} against {twin}: {', '.join(fs)}")
    print(f"\n{len(keys)} cases expected; {diffs} differ ({raw_only_n} in the tool-call arguments string only); "
          f"{missing} missing")
    if not keys:
        print("no rows at all")
        return 2
    if absent:
        print(f"families with no rows: {', '.join(absent)}")
    return 1 if diffs else (2 if missing else 0)


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] not in ("run", "compare"):
        sys.exit(__doc__)
    if a[0] == "run":
        sys.exit(run(a[1:]))
    if len(a) != 5 or a[3] != "--families":
        sys.exit("usage: g5_server.py compare A.jsonl B.jsonl --families minicpm5,qwen36,maple")
    sys.exit(compare(a[1], a[2], a[4].split(",")))
