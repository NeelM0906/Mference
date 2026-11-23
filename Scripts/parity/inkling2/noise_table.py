#!/usr/bin/env python3
"""The reference-noise tables for the G6 proposal, from compare_goldens.py
--json outputs (stdlib only).

    python3 -I Scripts/parity/inkling2/noise_table.py <noise.json> [<noise.json> ...]

Prints a Markdown table per prompt with log-probs (top-1 agreement overall and
at a run-A margin >= 0.2 nats; mean / nearest-rank p95 / max abs delta of the
run-A top-1 log-prob; mean / nearest-rank p95 approximate top-20 KL), the
pooled top-1 agreement over every prompt of every file (overall and at margins
0.2, 0.5, 2.0), and a table per trace (hidden-state relative L2 at the worst
and the last visit; routed-expert set agreement, mean and worst visit).
Positions are those where both runs saw the same context (compare_goldens.py).
"""

import json
import sys


def main():
    lp_rows, tr_rows = [], []
    pooled = {"positions": 0, "agree": 0, "m": {}}
    for f in sys.argv[1:]:
        d = json.load(open(f))
        for p in d["prompts"]:
            l = p.get("logprobs")
            if l:
                m02 = l["top1_at_margin"]["0.2"]
                lp_rows.append((p["id"], l["positions"], l["top1_agreement"], m02["rate"], l["mean_abs_dlogprob_top1"],
                                l["p95_abs_dlogprob_top1"], l["max_abs_dlogprob_top1"], l["mean_kl_top20_approx"],
                                l["p95_kl_top20_approx"]))
                pooled["positions"] += l["positions"]
                pooled["agree"] += round(l["top1_agreement"] * l["positions"])
                for k, c in l["top1_at_margin"].items():
                    pm = pooled["m"].setdefault(k, [0, 0])
                    pm[0] += c["positions"]
                    pm[1] += c["agree"]
            t = p.get("trace")
            if t:
                tr_rows.append((p["id"], t["max_rel_l2"], t["final_visit_rel_l2"], t["mean_expert_set_agreement"],
                                t["min_expert_set_agreement"]))
    f3 = lambda x: "n/a" if x is None else f"{x:.3f}"
    f4 = lambda x: "n/a" if x is None else f"{x:.4f}"
    print("| Prompt | Positions | Top-1 | Top-1 at margin >= 0.2 | mean / p95 / max abs delta lp | mean / p95 KL (top-20, approx.) |")
    print("|---|---:|---:|---:|---|---|")
    for r in lp_rows:
        print(f"| {r[0]} | {r[1]} | {f3(r[2])} | {f3(r[3])} | {f3(r[4])} / {f3(r[5])} / {f3(r[6])} | {f4(r[7])} / {f4(r[8])} |")
    n = pooled["positions"]
    ms = ", ".join(f"{c[1]}/{c[0]} = {c[1] / c[0]:.3f} at >= {k}" for k, c in sorted(pooled["m"].items(), key=lambda x: float(x[0])))
    print(f"| **pooled** | {n} | {pooled['agree'] / n:.3f} | {ms} | | |")
    print()
    print("| Trace | Hidden rel L2, worst visit / last visit | Expert-set agreement, mean / worst visit |")
    print("|---|---|---|")
    for r in tr_rows:
        print(f"| {r[0]} | {f3(r[1])} / {f4(r[2])} | {f3(r[3])} / {f3(r[4])} |")


if __name__ == "__main__":
    main()
