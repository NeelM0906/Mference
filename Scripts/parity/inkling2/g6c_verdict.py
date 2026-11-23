#!/usr/bin/env python3
"""G6c verdicts for a candidate run against the Inkling 2 reference (stdlib).

    python3 -I Scripts/parity/inkling2/g6c_verdict.py <candidate-dir> [--traces <dir>]
        [--goldens <goldens-dir>] [--json out.json]

<candidate-dir> holds the runner's output, in ink-golden's `generate` layout,
for the teacher-forced prompt set (`inputs/generate_prompts_tf.json`: each
log-prob golden's prompt plus all but its last generated id as one ids prompt,
log-probs at every recorded position). `--traces <dir>` optionally holds the
runner's traces of the trace prompts (trace01, trace02, trace03-tool-result),
same layout (golden.json, trace.json, hidden_after_visit.f32).

Two verdicts from one run (tokenizer and chat renders, rule 1, are exact and
tested elsewhere):

A. v2 as adopted (spec v1.7, section 8): log-probs against the free-running
   Metal goldens (`generate/metal`, `generate2/metal`), teacher-forced; per
   prompt each metric <= max(1.5 x the measured Metal-vs-CPU-only value,
   floor) with the measured values from `noise/metal-vs-cpu.json` and
   `noise/set2-metal-vs-cpu.json` and the floors: disagreement rates 0.03
   (top-1, and top-1 at a reference margin >= 0.2), mean |delta lp| 0.02, p95
   |delta lp| 0.06, mean KL 0.002, p95 KL 0.006. Pooled top-1 >= 0.93 and
   >= 0.99 at margin >= 2.0. Traces: hidden rel L2 at every visit <= 1.5 x
   the measured worst visit, at the last visit <= 1.5 x the measured last
   visit; mean expert-set agreement >= measured mean - 0.10.
B. v2 with the addendum (a separate proposal, NOT adopted): log-probs against
   the untraced teacher-forced Metal run (`generate-tf/metal`: the same
   contexts in one batch), measured values from `noise/tf-metal-vs-tfcpu.json`,
   the disagreement floors max(0.03, 2/n) with n the positions the rate is
   over; everything else as in A.

A failure under A stays a failure. Where llama.cpp's own Metal run, teacher-
forced, fails A on the same prompt, the report says so: g01-capital (its
golden's generated positions were decode steps, the candidate's are one
batch) and trace02 (its golden was made with the trace's eval callback).
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_goldens as cg  # noqa: E402

FLOORS = {"top1": 0.03, "top1_m02": 0.03, "mean_dlp": 0.02, "p95_dlp": 0.06, "mean_kl": 0.002, "p95_kl": 0.006}
TRACE_GOLDENS = {"trace01": "generate/metal", "trace02": "generate2/metal", "trace03-tool-result": "generate3/metal"}
TRACE_NOISE = {"trace01": "metal-vs-cpu.json", "trace02": "set2-metal-vs-cpu.json", "trace03-tool-result": "set3-metal-vs-cpu.json"}
# Prompts where llama.cpp's untraced Metal run, teacher-forced, fails verdict A
# against its own goldens (goldens README, "Teacher-forced noise").
METAL_SELF_FAILS_A = {
    "g01-capital": "llama.cpp's Metal fails A here too: golden decode steps vs one batch",
    "trace02": "llama.cpp's Metal fails A here too: the golden was made with the trace callback",
}


def metrics(l):
    """The rule-2 metrics of one compare_goldens log-prob record; disagreement
    rates with the positions they are over."""
    m02 = l["top1_at_margin"]["0.2"]
    return {
        "top1": (1 - l["top1_agreement"], l["positions"]),
        "top1_m02": ((1 - m02["rate"]) if m02["rate"] is not None else 0.0, m02["positions"]),
        "mean_dlp": (l["mean_abs_dlogprob_top1"], None),
        "p95_dlp": (l["p95_abs_dlogprob_top1"], None),
        "mean_kl": (l["mean_kl_top20_approx"], None),
        "p95_kl": (l["p95_kl_top20_approx"], None),
    }


def measured(goldens, files):
    out = {}
    for f in files:
        for p in json.load(open(os.path.join(goldens, "noise", f)))["prompts"]:
            if p.get("logprobs"):
                out[p["id"]] = metrics(p["logprobs"])
    return out


def bounds(meas, count_floor):
    b = {}
    for k, (v, n) in meas.items():
        floor = FLOORS[k]
        if count_floor and k in ("top1", "top1_m02") and n:
            floor = max(floor, 2 / n)
        b[k] = max(1.5 * v, floor)
    return b


def judge(cand_lp, meas, count_floor):
    """Per-prompt rule 2: (pass, {metric: (value, bound)} for failures, bounds)."""
    got = metrics(cand_lp)
    if count_floor:
        # the count floor uses the candidate comparison's own n
        meas = {k: (v, got[k][1]) if k in ("top1", "top1_m02") else (v, n) for k, (v, n) in meas.items()}
    b = bounds(meas, count_floor)
    fails = {k: (got[k][0], b[k]) for k in b if got[k][0] is not None and got[k][0] > b[k] + 1e-12}
    return not fails, fails, b


def pooled_ok(prompts):
    lps = [p["logprobs"] for p in prompts if p.get("logprobs")]
    n = sum(l["positions"] for l in lps)
    a = sum(l["top1_agreement"] * l["positions"] for l in lps)
    n2 = sum(l["top1_at_margin"]["2.0"]["positions"] for l in lps)
    a2 = sum(l["top1_at_margin"]["2.0"]["agree"] for l in lps)
    top1, top1_2 = (a / n if n else None), (a2 / n2 if n2 else None)
    return {"positions": n, "top1": top1, "top1_at_2": top1_2,
            "pass": top1 is not None and top1 >= 0.93 and (top1_2 is None or top1_2 >= 0.99)}


def traces(goldens, tdir):
    res = {}
    if not tdir:
        return res
    for pid, gd in TRACE_GOLDENS.items():
        if not os.path.isdir(os.path.join(tdir, pid)):
            res[pid] = {"pass": False, "why": f"no {pid}/ in {tdir}"}
            continue
        t = cg.trace_metrics(os.path.join(goldens, gd), tdir, pid)
        ref = next(p["trace"] for p in json.load(open(os.path.join(goldens, "noise", TRACE_NOISE[pid])))["prompts"]
                   if p["id"] == pid)
        if t is None:
            res[pid] = {"pass": False, "why": "no trace.json in the candidate"}
            continue
        worst_bound, last_bound = 1.5 * ref["max_rel_l2"], 1.5 * ref["final_visit_rel_l2"]
        exp_bound = ref["mean_expert_set_agreement"] - 0.10
        fails = []
        if t["max_rel_l2"] is not None and t["max_rel_l2"] > worst_bound:
            fails.append(f"worst-visit rel L2 {t['max_rel_l2']:.4g} > {worst_bound:.4g}")
        if t["final_visit_rel_l2"] is not None and t["final_visit_rel_l2"] > last_bound:
            fails.append(f"last-visit rel L2 {t['final_visit_rel_l2']:.4g} > {last_bound:.4g}")
        if t["mean_expert_set_agreement"] < exp_bound:
            fails.append(f"mean expert-set agreement {t['mean_expert_set_agreement']:.3f} < {exp_bound:.3f}")
        res[pid] = {"pass": not fails, "fails": fails}
    return res


def ink_dir():
    """$INK, else the program directory under the main checkout's scratch/
    (as paths.sh finds it)."""
    if os.environ.get("INK"):
        return os.environ["INK"]
    common = subprocess.run(["git", "-C", HERE, "rev-parse", "--path-format=absolute", "--git-common-dir"],
                            capture_output=True, text=True, check=True).stdout.strip()
    return os.path.join(os.path.dirname(common), "scratch", "inkling-program")


def main():
    args = sys.argv[1:]
    cand = args[0]
    opt = lambda name, default=None: args[args.index(name) + 1] if name in args else default
    goldens = opt("--goldens") or os.path.join(ink_dir(), "goldens")
    tdir, out_json = opt("--traces"), opt("--json")

    # A: v2 as adopted, against the free-running Metal goldens, teacher-forced
    meas_a = measured(goldens, ["metal-vs-cpu.json", "set2-metal-vs-cpu.json"])
    prompts_a = []
    for gd in ("generate/metal", "generate2/metal"):
        prompts_a += [p for p in cg.compare(os.path.join(goldens, gd), cand, teacher_forced=True)["prompts"]]
    # B: the addendum, against the untraced teacher-forced Metal run
    meas_b = measured(goldens, ["tf-metal-vs-tfcpu.json"])
    prompts_b = cg.compare(os.path.join(goldens, "generate-tf", "metal"), cand, teacher_forced=False)["prompts"]
    tr = traces(goldens, tdir)

    rows, report = [], {"candidate": cand, "goldens": goldens, "prompts": {}, "traces": tr}
    by_b = {p["id"]: p for p in prompts_b}
    for pa in prompts_a:
        pid = pa["id"]
        pb = by_b.get(pid, {})
        ra = rb = None
        if not pa.get("context_equal") or not pa.get("all_positions_compared"):
            ra = (False, {"context": "candidate context differs from the golden's, or positions are missing"}, {})
        elif pa.get("logprobs") and pid in meas_a:
            ra = judge(pa["logprobs"], meas_a[pid], count_floor=False)
        if not pb.get("prompt_ids_equal"):
            rb = (False, {"context": "candidate ids differ from generate-tf/metal's"}, {})
        elif pb.get("logprobs") and pid in meas_b:
            rb = judge(pb["logprobs"], meas_b[pid], count_floor=True)
        note = ""
        if ra and not ra[0] and pid in METAL_SELF_FAILS_A:
            note = METAL_SELF_FAILS_A[pid]
        report["prompts"][pid] = {
            "A_pass": ra[0] if ra else None, "A_fails": ra[1] if ra else None, "A_bounds": ra[2] if ra else None,
            "B_pass": rb[0] if rb else None, "B_fails": rb[1] if rb else None, "B_bounds": rb[2] if rb else None,
            "note": note}
        rows.append((pid, ra, rb, note))
    report["pooled_A"] = pooled_ok([p for p in prompts_a if p.get("context_equal")])
    report["pooled_B"] = pooled_ok([p for p in prompts_b if p.get("prompt_ids_equal")])
    tr_ok = all(t["pass"] for t in tr.values())
    report["verdict_A"] = all(r[1] and r[1][0] for r in rows) and report["pooled_A"]["pass"] and tr_ok
    report["verdict_B"] = all(r[2] and r[2][0] for r in rows) and report["pooled_B"]["pass"] and tr_ok
    fmt = lambda r: "-" if r is None else ("pass" if r[0] else "FAIL: " + "; ".join(
        f"{k} {v[0]:.4g} > {v[1]:.4g}" if isinstance(v, tuple) else f"{k}: {v}" for k, v in r[1].items()))
    print("| Prompt | A: v2 as adopted | B: v2 + addendum (not adopted) | Note |")
    print("|---|---|---|---|")
    for pid, ra, rb, note in rows:
        print(f"| {pid} | {fmt(ra)} | {fmt(rb)} | {note} |")
    for name in ("A", "B"):
        p = report[f"pooled_{name}"]
        t2 = "n/a" if p["top1_at_2"] is None else f"{p['top1_at_2']:.4f}"
        print(f"\nPooled {name}: {p['positions']} positions, top-1 {p['top1']:.4f} (>= 0.93), "
              f"at margin >= 2.0 {t2} (>= 0.99): {'pass' if p['pass'] else 'FAIL'}")
    for pid, t in tr.items():
        print(f"Trace {pid}: {'pass' if t['pass'] else 'FAIL: ' + '; '.join(t.get('fails') or [t.get('why', '')])}")
    if not tr:
        print("Traces: none given (--traces); rule 4 not checked")
    scope = "" if tr else " (log-prob rules only: no --traces, rule 4 not checked)"
    report["traces_checked"] = bool(tr)
    print(f"\nVerdict A (v2 as adopted, spec v1.7): {'PASS' if report['verdict_A'] else 'FAIL'}{scope}")
    print(f"Verdict B (v2 + addendum, not adopted): {'PASS' if report['verdict_B'] else 'FAIL'}{scope}")
    if out_json:
        json.dump(report, open(out_json, "w"), indent=1)
    sys.exit(0 if report["verdict_A"] else 1)


if __name__ == "__main__":
    main()
