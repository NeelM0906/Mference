#!/usr/bin/env python3
"""Mode S client: stream chat completions from a loopback MferenceServer and
record time to first token, decode rate and prompt token counts.

Standard library only. The server must already be running (see
docs/OPENAI_SERVER.md); this script never starts or stops a model process.

Each case is a JSON array of chat messages (the same format as
docs/benchmark-prompts/real-generation-v1/*.json). For every measured run the
script writes one JSON line with:

  ttft_s             request sent -> first streamed content or reasoning delta
  total_s            request sent -> end of stream
  decode_tok_s       (completion_tokens - 1) / (last delta - first delta)
  prompt_tokens, cached_tokens, completion_tokens, finish_reason
  server_timings     the server's llama.cpp-style `timings` object

Cold runs must not hit the prompt prefix cache. With --flush (default on) a
one-token request with an unrelated prompt is sent before every cold run, so
the server's single retained prefix no longer matches; a cold run that still
reports cached tokens is marked `cold_violation`.

With --followup, every run is a two-turn case: the cold first turn, then the
first turn's assistant reply plus the follow-up user message, sent warm (no
flush between them). Both turns are recorded.

usage:
  server_stream_bench.py --port 8080 --model qwen3.6-35b-a3b \
      --case docs/benchmark-prompts/real-generation-v1/long-synthesis.json \
      --runs 5 --seed 20260723 --out results.jsonl
"""
import argparse
import http.client
import json
import statistics
import sys
import time
from urllib.parse import urlparse

LOOPBACK = {"127.0.0.1", "localhost", "::1"}

FLUSH_MESSAGES = [{"role": "user", "content": "Reply with the single word OK."}]


def parse_sse_lines(lines):
    """Yield decoded JSON payloads from an iterable of SSE byte lines.

    Stops at `data: [DONE]`. Ignores comments, blank lines and non-data fields.
    """
    for raw in lines:
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        line = line.rstrip("\r\n")
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        if data:
            yield json.loads(data)


def summarize_stream(events, t_sent, clock):
    """Fold streamed chunks into one run record.

    `events` yields parsed chunks; `clock()` is read as each chunk arrives, so
    tests can drive it with a fake clock.
    """
    t_first = None
    t_last = None
    finish = None
    usage = None
    timings = None
    text_parts = []
    reasoning_parts = []
    for chunk in events:
        now = clock()
        if "error" in chunk:
            raise RuntimeError(f"server error in stream: {chunk['error']}")
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            content = delta.get("content")
            reasoning = delta.get("reasoning_content")
            if content or reasoning:
                if t_first is None:
                    t_first = now
                t_last = now
                if content:
                    text_parts.append(content)
                if reasoning:
                    reasoning_parts.append(reasoning)
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
        if chunk.get("usage"):
            usage = chunk["usage"]
        if chunk.get("timings"):
            timings = chunk["timings"]
    t_end = clock()
    if t_first is None:
        raise RuntimeError("stream ended without any content or reasoning delta")
    usage = usage or {}
    completion = usage.get("completion_tokens")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    decode = None
    if completion and completion > 1 and t_last > t_first:
        decode = (completion - 1) / (t_last - t_first)
    return {
        "ttft_s": t_first - t_sent,
        "total_s": t_end - t_sent,
        "decode_window_s": t_last - t_first,
        "decode_tok_s": decode,
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": cached,
        "completion_tokens": completion,
        "finish_reason": finish,
        "server_timings": timings,
        "content": "".join(text_parts),
        "reasoning_chars": sum(len(p) for p in reasoning_parts),
    }


def stream_completion(host, port, body, timeout):
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    payload = json.dumps(body).encode("utf-8")
    t_sent = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", body=payload,
                 headers={"Content-Type": "application/json",
                          "Accept": "text/event-stream"})
    resp = conn.getresponse()
    if resp.status != 200:
        detail = resp.read().decode("utf-8", "replace")
        conn.close()
        raise RuntimeError(f"HTTP {resp.status}: {detail}")
    try:
        record = summarize_stream(parse_sse_lines(iter(resp.readline, b"")),
                                  t_sent, time.perf_counter)
    finally:
        conn.close()
    return record


def request_body(args, messages, max_tokens, seed):
    body = {
        "model": args.model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_completion_tokens": max_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
    }
    if seed is not None:
        body["seed"] = seed
    return body


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"median": statistics.median(values), "min": min(values),
            "max": max(values), "n": len(values)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default=None,
                   help="loopback base URL, e.g. http://127.0.0.1:8080 (overrides --port)")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--model", required=True, help="API model id")
    p.add_argument("--case", required=True, help="JSON messages file for the (first) turn")
    p.add_argument("--followup", help="JSON messages file appended as a warm second turn")
    p.add_argument("--label", default=None, help="case label written into every record")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1, help="discarded runs before measuring")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--followup-max-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-k", type=int, default=64)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-flush", action="store_true",
                   help="do not send the prefix-flush request before cold runs")
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--idle", type=float, default=0.0,
                   help="seconds to sleep between runs")
    p.add_argument("--out", required=True, help="JSON lines output file")
    args = p.parse_args(argv)

    if args.base_url:
        url = urlparse(args.base_url)
        host, port = url.hostname, url.port or 80
    else:
        host, port = "127.0.0.1", args.port
    if host not in LOOPBACK:
        p.error(f"refusing non-loopback host {host!r}")

    with open(args.case, encoding="utf-8") as f:
        first_messages = json.load(f)
    followup_messages = None
    if args.followup:
        with open(args.followup, encoding="utf-8") as f:
            followup_messages = json.load(f)
    label = args.label or args.case
    fmax = args.followup_max_tokens or args.max_tokens

    records = []
    total_runs = args.warmup + args.runs
    with open(args.out, "a", encoding="utf-8") as out:
        for i in range(total_runs):
            measured = i >= args.warmup
            if not args.no_flush:
                stream_completion(host, port,
                                  request_body(args, FLUSH_MESSAGES, 1, None),
                                  args.timeout)
            rec = stream_completion(host, port,
                                    request_body(args, first_messages, args.max_tokens, args.seed),
                                    args.timeout)
            rec.update({"label": label, "turn": 1, "run": i - args.warmup + 1,
                        "measured": measured,
                        "cold_violation": (not args.no_flush) and rec["cached_tokens"] > 0})
            turns = [rec]
            if followup_messages is not None:
                history = list(first_messages)
                history.append({"role": "assistant", "content": rec["content"]})
                history.extend(followup_messages)
                rec2 = stream_completion(host, port,
                                         request_body(args, history, fmax, args.seed),
                                         args.timeout)
                rec2.update({"label": label, "turn": 2, "run": rec["run"],
                             "measured": measured, "cold_violation": False})
                turns.append(rec2)
            for r in turns:
                line = dict(r)
                line["content_chars"] = len(line.pop("content"))
                out.write(json.dumps(line, sort_keys=True) + "\n")
                out.flush()
                tag = "measured" if measured else "warmup"
                dec = r["decode_tok_s"]
                print(f"[{label}] {tag} run{r['run']} turn{r['turn']}: "
                      f"ttft={r['ttft_s']:.2f}s decode="
                      f"{dec if dec is None else round(dec, 2)} tok/s "
                      f"prompt={r['prompt_tokens']} cached={r['cached_tokens']} "
                      f"gen={r['completion_tokens']} finish={r['finish_reason']}"
                      + (" COLD-VIOLATION" if r["cold_violation"] else ""),
                      file=sys.stderr, flush=True)
            if measured:
                records.extend(turns)
            if args.idle and i + 1 < total_runs:
                time.sleep(args.idle)

    summary = {"label": label}
    for turn in sorted({r["turn"] for r in records}):
        rs = [r for r in records if r["turn"] == turn]
        summary[f"turn{turn}"] = {
            "ttft_s": stats([r["ttft_s"] for r in rs]),
            "total_s": stats([r["total_s"] for r in rs]),
            "decode_tok_s": stats([r["decode_tok_s"] for r in rs]),
            "prompt_tokens": sorted({r["prompt_tokens"] for r in rs}),
            "cached_tokens": sorted({r["cached_tokens"] for r in rs}),
            "completion_tokens": [r["completion_tokens"] for r in rs],
            "finish_reasons": [r["finish_reason"] for r in rs],
            "cold_violations": sum(1 for r in rs if r["cold_violation"]),
        }
    print(json.dumps(summary, indent=2, sort_keys=True))
    bad = any(r["cold_violation"] for r in records)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
