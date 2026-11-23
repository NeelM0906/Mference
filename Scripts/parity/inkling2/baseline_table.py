#!/usr/bin/env python3
"""Summary table for the K0 baseline from the stage directories (stdlib).

    python3 -I Scripts/parity/inkling2/baseline_table.py <baseline-dir>

Reads, per stage, run.json (guarded_run.sh: exit, wall, swap, peak footprint),
jobs.log (swift jobs at start and end), stdout.log (llama-bench markdown) and
stderr.log (llama-completion's common_perf_print lines) and prints a
Markdown table plus every raw timing line.
"""

import json
import os
import re
import sys

STAGES = ["bench-512", "text-512", "bench-4k", "text-4k", "bench-16k", "text-16k"]


def main():
    d = sys.argv[1]
    rows = []
    raw = []
    for st in STAGES:
        p = os.path.join(d, st)
        if not os.path.isdir(p):
            continue
        rj = json.load(open(os.path.join(p, "run.json"))) if os.path.exists(os.path.join(p, "run.json")) else {}
        jobs = open(os.path.join(p, "jobs.log")).read().strip().splitlines() if os.path.exists(os.path.join(p, "jobs.log")) else []
        jstart = next((l.split(": ")[-1] for l in jobs if "at start" in l), "?")
        out = open(os.path.join(p, "stdout.log"), errors="replace").read() if os.path.exists(os.path.join(p, "stdout.log")) else ""
        err = open(os.path.join(p, "stderr.log"), errors="replace").read() if os.path.exists(os.path.join(p, "stderr.log")) else ""
        pp = tg = ttft = ptok = None
        for m in re.finditer(r"\|\s*(pp\d+|tg\d+)\s*\|\s*([\d.]+) ± ([\d.]+)", out):
            raw.append(f"{st}: {m.group(0).strip()}")
            if m.group(1).startswith("pp"):
                pp = f"{m.group(2)} ± {m.group(3)}"
                ptok = m.group(1)[2:]
            else:
                tg = f"{m.group(2)} ± {m.group(3)}"
        for line in err.splitlines():
            if "common_perf_print" in line:
                raw.append(f"{st}: {line.split('common_perf_print:')[-1].strip()}")
        m = re.search(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens \(.*?([\d.]+) tokens per second\)", err)
        if m:
            ptok = m.group(2)
            pp = m.group(3)
            pe_ms = float(m.group(1))
            m2 = re.search(r"\beval time =\s*([\d.]+) ms /\s*(\d+) runs\s*\(\s*([\d.]+) ms per token,\s*([\d.]+) tokens per second\)", err)
            if m2:
                tg = m2.group(4)
                ttft = f"{(pe_ms + float(m2.group(3))) / 1000:.2f}"
        rows.append((st, ptok or "?", pp or "-", tg or "-", ttft or "-", rj.get("peak_footprint_bytes"), rj.get("max_rss_bytes"),
                     rj.get("swap_start_mb"), rj.get("swap_end_mb"), rj.get("swap_delta_mb"), rj.get("exit"), rj.get("wall_s"), jstart))
    print("| Stage | Prompt tok | Prefill tok/s | Decode tok/s | TTFT s | Peak footprint GB | Max RSS GB | Swap before / after MB | Delta | Exit | Wall s | Other swift jobs at start |")
    print("|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|")
    for r in rows:
        gb = lambda x: f"{x / 1e9:.2f}" if isinstance(x, (int, float)) else "?"
        print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {gb(r[5])} | {gb(r[6])} | {r[7]} / {r[8]} | {r[9]} | {r[10]} | {r[11]} | {r[12]} |")
    print("\nRaw timing lines:\n")
    print("```")
    for l in raw:
        print(l)
    print("```")


if __name__ == "__main__":
    main()
