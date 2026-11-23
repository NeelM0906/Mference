#!/usr/bin/env python3
"""Native side of the K0 baseline table (stdlib only).

    python3 -I Scripts/parity/inkling2/native_table.py <native-dir> <baseline.md> [--metal-baseline <md>]

Reads, per native_stage.sh stage directory (native-512, native-4k,
native-16k): run.json (guarded_run.sh: exit, wall, swap, peak footprint, max
RSS), jobs.log (other swift jobs at start), checks.log (app-open check, CLI
source, the stage's exit and the CLI's error line), run.log (guarded_run.sh's
refusal reasons) and stderr.log (MferenceCLI's footer
`[stop=... prefill=<N>tok/<s>s new=<N>tok decode=<s>s tok/s=<x>]`). Prints a
Markdown table with exactly baseline.md's columns plus two ratio columns:
against baseline.md, and against stock llama.cpp on an all-Metal GGUF
(`--metal-baseline`: a table with baseline.md's columns and its text-*
rows; "pending" until that measurement exists), each against the text-*
row of the same prompt. After the table: the app-open checks, any CLI
errors and every raw footer.

Definitions, matched to baseline.md's text-* rows:
- Prefill tok/s = prompt tokens / prefill seconds.
- TTFT s = prefill seconds. The first token is sampled from the prefill
  logits, as in llama-completion. Model load is excluded on both sides, but
  llama-completion's warm-up decode is inside its (excluded) load time while
  the native CLI has none: native prefill includes first-use costs (first
  expert reads, pipeline builds), most visible in the 512-token row.
- Decode tok/s = (new tokens - 1) / decode seconds. n new tokens take n - 1
  forward steps on both sides; llama-completion's "eval time" counts those
  n - 1 runs. The CLI footer's own tok/s divides n by the same seconds; it is
  kept in the raw lines.
- Ratio = native / llama.cpp for prefill and decode (above 1 is faster) and
  for TTFT (below 1 is faster).
- Exit = the stage's exit (checks.log); when it differs from the CLI's own
  exit (run.json) both show, e.g. "0 / stage 70" for swap growth found after
  a clean CLI exit.
- Peak footprint counts anonymous memory: the native runner's resident
  experts and staging slabs are anonymous, llama.cpp's experts file-mapped
  (RSS only), so that column is not like for like.
"""

import json
import os
import re
import sys

STAGES = [("native-512", "text-512"), ("native-4k", "text-4k"), ("native-16k", "text-16k")]
FOOTER = re.compile(r"^\[stop=(\S+) prefill=(\d+)tok/([\d.]+)s(?:[^\]]*?) new=(\d+)tok decode=([\d.]+)s tok/s=([\d.]+)\]$")
HEADER = ("| Stage | Prompt tok | Prefill tok/s | Decode tok/s | TTFT s | Peak footprint GB | Max RSS GB "
          "| Swap before / after MB | Delta | Exit | Wall s | Other swift jobs at start "
          "| vs llama.cpp (prefill / decode / TTFT) | vs llama.cpp, all-Metal file (prefill / decode / TTFT) |")
RULE = "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---|"


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


def stage_row(d, stage, base, metal=None):
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
    checks = read(os.path.join(p, "checks.log"))
    st = re.findall(r"stage exit (\d+)", checks)
    cli_exit = rj.get("exit")
    if st and str(cli_exit) != st[-1]:
        exit_cell = f"{cli_exit if cli_exit is not None else '-'} / stage {st[-1]}"
    else:
        exit_cell = str(cli_exit if cli_exit is not None else (f"stage {st[-1]}" if st else "?"))
    b = base.get(stage)
    vs = " / ".join(ratio(x, y) for x, y in zip((pp, tg, ttft), b)) if b else "no baseline row"
    if metal is None:
        vs_metal = "pending"
    else:
        m = metal.get(stage)
        vs_metal = " / ".join(ratio(x, y) for x, y in zip((pp, tg, ttft), m)) if m else "no baseline row"
    cells = [stage, str(ptok) if ptok is not None else "?", fmt(pp), fmt(tg), fmt(ttft),
             gb(rj.get("peak_footprint_bytes")), gb(rj.get("max_rss_bytes")),
             f"{rj.get('swap_start_mb', '?')} / {rj.get('swap_end_mb', '?')}", str(rj.get("swap_delta_mb", "?")),
             exit_cell, str(rj.get("wall_s", "?")), jstart, vs, vs_metal]
    stop = (footers[-1].group(1), int(footers[-1].group(4))) if footers else None
    return "| " + " | ".join(cells) + " |", [f.group(0) for f in footers], stop


def main():
    args, mb = sys.argv[1:], None
    usage = "usage: native_table.py <native-dir> <baseline.md> [--metal-baseline <md>]"
    for i, a in enumerate(list(args)):
        if a == "--metal-baseline" or a.startswith("--metal-baseline="):
            mb = a.split("=", 1)[1] if "=" in a else (args[i + 1] if i + 1 < len(args) else "")
            del args[i:i + (1 if "=" in a else 2)]
            if not mb:
                sys.exit(usage + "\n--metal-baseline needs a file")
            break
    if len(args) != 2 or any(x.startswith("--") for x in args):
        sys.exit(usage)
    d, baseline = args
    for path, what in ((d, "native dir"), (baseline, "baseline"), (mb, "metal baseline")):
        if path is not None and not os.path.exists(path):
            sys.exit(f"native_table.py: {what} {path} does not exist")
    base = baseline_rows(baseline)
    metal_rows = baseline_rows(mb) if mb else None
    rows, raw, notes = [], [], []
    for stage, text in STAGES:
        p = os.path.join(d, stage)
        if not os.path.isdir(p):
            continue
        metal = None if metal_rows is None else ({stage: metal_rows[text]} if text in metal_rows else {})
        row, footers, stop = stage_row(d, stage, {stage: base[text]} if text in base else {}, metal)
        rows.append(row)
        raw += [f"{stage}: {f}" for f in footers]
        if stop:
            want = {"native-512": 128, "native-4k": 64, "native-16k": 32}[stage]
            flag = "" if stop == ("maxTokens", want) else \
                f" (NOT comparable: llama.cpp generated {want}; decode covers {stop[1] - 1} steps)"
            notes.append(f"{stage}: stop={stop[0]}, {stop[1]} new tokens{flag}")
        for line in read(os.path.join(p, "checks.log")).splitlines():
            if any(k in line for k in ("app open check", "cli source", "REFUSED", "stage exit")):
                notes.append(f"{stage}: {line.split(' ', 1)[-1]}")
        for line in read(os.path.join(p, "run.log")).splitlines():
            if "REFUSED" in line or "SWAP GUARD" in line:
                notes.append(f"{stage}: guarded_run: {line.split(' ', 1)[-1]}")
    print(HEADER)
    print(RULE)
    for r in rows:
        print(r)
    print("\nNative prefill and TTFT include first-use costs (no warm-up run; llama-completion's"
          " warm-up is in its excluded load time). Peak footprint: the native experts are anonymous"
          " memory, llama.cpp's file-mapped (RSS only).")
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
