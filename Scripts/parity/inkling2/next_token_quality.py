#!/usr/bin/env python3
"""How well three runs predict real text: the log-probability each assigns to
the token that actually comes next, at the same positions (stdlib only).

    python3 -I Scripts/parity/inkling2/next_token_quality.py <run-a> <run-b> <run-c>
        [--names A,B,C] [--goldens <goldens-dir>] [--prompts id,id,...] [--boot 10000] [--json out]

Each run is an ink-golden `generate` directory of the teacher-forced set
(`generate-tf/metal`, `generate-tf/cpu`, a runner's `g6c/cand/tf`): per prompt
golden.json `prompt_ids` and logprobs.json rows {pos, top, [next, next_logprob]}.
None is treated as the reference; the text is.

Positions: every recorded position p whose next token, prompt_ids[p + 1], is
part of the ORIGINAL prompt text (p + 1 < the free-running golden's prompt
length, read from `generate/metal` or `generate2/metal`). Later positions of a
teacher-forced prompt predict the Metal golden's own greedy tokens, which would
favour Metal, so they are left out.

Default prompts: the raw natural-text ones, g05-wetlands (prose), lp06-raw-1300
and lp07-raw-2600 (Markdown documentation), lp08-code (Python), lp10-multilingual.
Chat prompts are left out (their framing tokens are deterministic, not text);
g01-capital adds only 4 positions.

Two measures:
- top20 (the comparison asked for): positions where the true next token is in
  the top 20 of all three runs; its top-20 log-probability in each.
- exact (supplementary, when all three rows carry `next_logprob` for that
  token): every natural position, the full-vocabulary log-probability.

Reported: per run the mean log-probability; per pair (A - B, A - C, B - C, so pass
the runner first, then Metal, then CPU) the mean paired difference
(first minus second; positive = the first predicts the text better) with a 95% percentile bootstrap over positions (seeded),
and per prompt the mean difference (positions within a prompt are correlated;
read the interval together with the per-prompt signs).
"""

import json
import os
import random
import subprocess
import sys


def load(d, pid, name):
    p = os.path.join(d, pid, name)
    return json.load(open(p)) if os.path.exists(p) else None


def ink_dir():
    if os.environ.get("INK"):
        return os.environ["INK"]
    here = os.path.dirname(os.path.abspath(__file__))
    common = subprocess.run(["git", "-C", here, "rev-parse", "--path-format=absolute", "--git-common-dir"],
                            capture_output=True, text=True, check=True).stdout.strip()
    return os.path.join(os.path.dirname(common), "scratch", "inkling-program")


DEFAULT = ["g05-wetlands", "lp06-raw-1300", "lp07-raw-2600", "lp08-code", "lp10-multilingual"]


def original_prompt_length(goldens, pid):
    for gd in ("generate/metal", "generate2/metal"):
        g = load(os.path.join(goldens, gd), pid, "golden.json")
        if g:
            return len(g["prompt_ids"])
    raise SystemExit(f"{pid}: no free-running golden in generate/metal or generate2/metal")


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def boot(diffs, reps, rng):
    n = len(diffs)
    if n == 0:
        return (float("nan"), float("nan"))
    ms = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return (ms[int(0.025 * reps)], ms[int(0.975 * reps) - 1])


def collect(runs, goldens, prompts):
    """Rows per measure, [(pid, pos, lp_run0, lp_run1, lp_run2)], with the prompts
    used and the prompts not yet in every run."""
    rows = {"top20": [], "exact": []}
    used, missing = [], []
    for pid in prompts:
        gs = [load(r, pid, "golden.json") for r in runs]
        ls = [load(r, pid, "logprobs.json") for r in runs]
        if not all(gs) or not all(ls):
            missing.append(pid)
            continue
        ids = gs[0]["prompt_ids"]
        if any(g["prompt_ids"] != ids for g in gs):
            raise SystemExit(f"{pid}: the three runs' prompt ids differ")
        n0 = original_prompt_length(goldens, pid)
        by = [{r["pos"]: r for r in l} for l in ls]
        used.append(pid)
        for p in sorted(set(by[0]) & set(by[1]) & set(by[2])):
            if p + 1 >= n0:
                continue
            t = ids[p + 1]
            tops = [dict((i, v) for i, v in b[p]["top"]) for b in by]
            if all(t in tp for tp in tops):
                rows["top20"].append((pid, p) + tuple(tp[t] for tp in tops))
            if all(b[p].get("next") == t and b[p].get("next_logprob") is not None for b in by):
                rows["exact"].append((pid, p) + tuple(b[p]["next_logprob"] for b in by))
    return rows, used, missing


def main():
    args = sys.argv[1:]
    opt = lambda k, d=None: args[args.index(k) + 1] if k in args else d
    runs = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or not args[i - 1].startswith("--"))]
    if len(runs) != 3:
        sys.exit(__doc__)
    names = opt("--names", "A,B,C").split(",")
    goldens = opt("--goldens") or os.path.join(ink_dir(), "goldens")
    prompts = opt("--prompts", ",".join(DEFAULT)).split(",")
    reps = int(opt("--boot", "10000"))
    rng = random.Random(20261123)

    rows, used, missing = collect(runs, goldens, prompts)

    pairs = [(0, 1), (0, 2), (1, 2)]  # A - B, A - C, B - C: pass the runner first for runner - Metal, runner - CPU, Metal - CPU
    report = {"runs": dict(zip(names, runs)), "prompts_used": used, "prompts_missing": missing, "measures": {}}
    print(f"prompts used: {', '.join(used)}" + (f"; not yet in every run: {', '.join(missing)}" if missing else ""))
    for measure in ("top20", "exact"):
        rs = rows[measure]
        m = {"positions": len(rs), "mean_logprob": {}, "pairs": {}, "per_prompt": {}}
        if not rs:
            report["measures"][measure] = m
            print(f"\n{measure}: no positions")
            continue
        for k in range(3):
            m["mean_logprob"][names[k]] = mean([r[2 + k] for r in rs])
        print(f"\n{measure}: {len(rs)} positions; mean log-prob of the true next token: "
              + ", ".join(f"{names[k]} {m['mean_logprob'][names[k]]:.4f}" for k in range(3)))
        print("| Pair | Mean difference | 95% bootstrap | Positions first better / worse | Per prompt |")
        print("|---|---:|---|---|---|")
        for a, b in pairs:
            d = [r[2 + a] - r[2 + b] for r in rs]
            lo, hi = boot(d, reps, rng)
            per = {}
            for r, x in zip(rs, d):
                per.setdefault(r[0], []).append(x)
            pp = {k: mean(v) for k, v in per.items()}
            label = f"{names[a]} - {names[b]}"
            m["pairs"][label] = {"mean": mean(d), "ci95": [lo, hi], "better": sum(x > 0 for x in d),
                                 "worse": sum(x < 0 for x in d)}
            m["per_prompt"][label] = pp
            print(f"| {label} | {mean(d):+.4f} | [{lo:+.4f}, {hi:+.4f}] | {sum(x > 0 for x in d)} / {sum(x < 0 for x in d)} | "
                  + ", ".join(f"{k.split('-')[0]} {v:+.3f}" for k, v in pp.items()) + " |")
        report["measures"][measure] = m
    if opt("--json"):
        json.dump(report, open(opt("--json"), "w"), indent=1)


if __name__ == "__main__":
    main()
