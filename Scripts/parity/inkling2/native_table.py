#!/usr/bin/env python3
"""Native side of the K0 baseline table (stdlib only).

    python3 -I Scripts/parity/inkling2/native_table.py <native-dir> <baseline.md>

Reads, per native_stage.sh stage directory (native-512, native-4k,
native-16k): run.json (guarded_run.sh: exit, wall, swap, peak footprint, max
RSS), jobs.log (other swift jobs at start), checks.log (app-open check, the
CLI's error line on failure) and stderr.log (MferenceCLI's footer
`[stop=... prefill=<N>tok/<s>s new=<N>tok decode=<s>s tok/s=<x>]`). Prints a
Markdown table with exactly baseline.md's columns plus one ratio column
against the llama.cpp text-* row of the same prompt, then the app-open
checks, any CLI errors and every raw footer.

Definitions, matched to baseline.md's text-* rows:
- Prefill tok/s = prompt tokens / prefill seconds.
- TTFT s = prefill seconds. The first token is sampled from the prefill
  logits, as in llama-completion; model load is excluded on both sides.
- Decode tok/s = (new tokens - 1) / decode seconds. n new tokens take n - 1
  forward steps on both sides; llama-completion's "eval time" counts those
  n - 1 runs. The CLI footer's own tok/s divides n by the same seconds; it is
  kept in the raw lines.
- Ratio = native / llama.cpp for prefill and decode (above 1 is faster) and
  for TTFT (below 1 is faster).
"""

import json
import os
import re
import sys

STAGES = [("native-512", "text-512"), ("native-4k", "text-4k"), ("native-16k", "text-16k")]
FOOTER = re.compile(r"^\[stop=(\S+) prefill=(\d+)tok/([\d.]+)s(?:[^\]]*?) new=(\d+)tok decode=([\d.]+)s tok/s=([\d.]+)\]$")
HEADER = ("| Stage | Prompt tok | Prefill tok/s | Decode tok/s | TTFT s | Peak footprint GB | Max RSS GB "
          "| Swap before / after MB | Delta | Exit | Wall s | Other swift jobs at start "
          "| vs llama.cpp (prefill / decode / TTFT) |")
RULE = "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|"


def read(path, default=""):
    return open(path, errors="replace").read() if os.path.exists(path) else default


def baseline_rows(path):
    """{stage: (prefill, decode, ttft)} from baseline.md's results table."""
    out = {}
    for line in read(path).splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 5 and cells[0].startswith("text-"):
            num = lambda s: float(s.split("±")[0]) if re.match(r"^[\d.]+", s) else None
            out[cells[0]] = (num(cells[2]), num(cells[3]), num(cells[4]))
    return out


def ratio(a, b):
    return f"{a / b:.2f}x" if a is not None and b else "-"


def stage_row(d, stage, base):
    p = os.path.join(d, stage)
    rj = json.loads(read(os.path.join(p, "run.json"), "{}"))
    jobs = read(os.path.join(p, "jobs.log")).splitlines()
    # jobs.log is appended to: the last "at start" line belongs to the run in run.json
    jstart = next((l.split(": ")[-1] for l in reversed(jobs) if "at start" in l), "?")
    footers = [m for m in map(FOOTER.match, read(os.path.join(p, "stderr.log")).splitlines()) if m]
    ptok = pp = tg = ttft = None
    if footers:
        f = footers[-1]
        ptok, pre_s, new, dec_s = int(f.group(2)), float(f.group(3)), int(f.group(4)), float(f.group(5))
        pp = ptok / pre_s if pre_s > 0 else None
        tg = (new - 1) / dec_s if dec_s > 0 and new > 1 else None
        ttft = pre_s
    fmt = lambda x, n=2: f"{x:.{n}f}" if x is not None else "-"
    gb = lambda x: f"{x / 1e9:.2f}" if isinstance(x, (int, float)) else "?"
    b = base.get(stage)
    vs = " / ".join(ratio(x, y) for x, y in zip((pp, tg, ttft), b)) if b else "no baseline row"
    cells = [stage, str(ptok) if ptok is not None else "?", fmt(pp), fmt(tg), fmt(ttft),
             gb(rj.get("peak_footprint_bytes")), gb(rj.get("max_rss_bytes")),
             f"{rj.get('swap_start_mb', '?')} / {rj.get('swap_end_mb', '?')}", str(rj.get("swap_delta_mb", "?")),
             str(rj.get("exit", "?")), str(rj.get("wall_s", "?")), jstart, vs]
    return "| " + " | ".join(cells) + " |", [f.group(0) for f in footers]


def main():
    d, baseline = sys.argv[1], sys.argv[2]
    base = baseline_rows(baseline)
    rows, raw, notes = [], [], []
    for stage, text in STAGES:
        p = os.path.join(d, stage)
        if not os.path.isdir(p):
            continue
        row, footers = stage_row(d, stage, {stage: base[text]} if text in base else {})
        rows.append(row)
        raw += [f"{stage}: {f}" for f in footers]
        for line in read(os.path.join(p, "checks.log")).splitlines():
            if "app open check" in line or "REFUSED" in line or "CLI: error:" in line:
                notes.append(f"{stage}: {line.split(' ', 1)[-1]}")
    print(HEADER)
    print(RULE)
    for r in rows:
        print(r)
    print("\nChecks and errors:\n")
    for n in notes:
        print(f"- {n}")
    print("\nRaw footers:\n")
    print("```")
    for l in raw:
        print(l)
    print("```")


if __name__ == "__main__":
    main()
