#!/usr/bin/env python3
"""Tier G6c for a candidate run against the Inkling 2 reference (stdlib).

    python3 -I Scripts/parity/inkling2/g6c_verdict.py <candidate-dir> [--traces <dir>]
        [--goldens <goldens-dir>] [--json out.json]

<candidate-dir> holds the runner's output, in ink-golden's `generate` layout,
for the teacher-forced prompt set (`inputs/generate_prompts_tf.json`: each
log-prob golden's prompt plus all but its last generated id as one ids
prompt, log-probs at every recorded position). `--traces <dir>` holds the
runner's traces of trace01, trace02 and trace03-tool-result, same layout
(golden.json, trace.json, hidden_after_visit.f32). Rule 1 (tokenizer and chat
renders, exact) is tested elsewhere.

The gate (spec v1.8, section 8: v2 with its addendum; goldens README, "G6
thresholds v2 with its addendum"):
- Rule 2, per prompt, against `generate-tf/metal` (llama.cpp's Metal on the
  same contexts in one batch, untraced): each metric <= max(1.5 x the value
  measured for that prompt in `noise/tf-metal-vs-tfcpu.json`, floor), with
  top-1 and top-1-at-margin-0.2 as disagreement rates and floors
  max(0.03, 2/n) for those (n = the reference positions the rate is over),
  0.02 mean |delta lp|, 0.06 p95 |delta lp|, 0.002 mean KL, 0.006 p95 KL.
  Every expected prompt must be present with every position compared.
- Rule 3, pooled over those prompts: top-1 >= 0.93, and >= 0.99 where the
  reference margin is >= 2.0.
- Rule 4, per trace: hidden rel L2 <= 1.5 x the measured worst visit at every
  visit and <= 1.5 x the measured last visit at the last; mean expert-set
  agreement >= the measured mean - 0.10 (measured: the Metal trace goldens
  against the CPU-only ones, `noise/{metal,set2,set3}-metal-vs-cpu.json`).
Any non-finite metric fails.

For the record only (does not gate): plain v2 as first adopted (spec v1.7),
rule 2 against the free-running Metal goldens (`generate/metal`,
`generate2/metal`) with the free-running measurement
(`noise/metal-vs-cpu.json`, `noise/set2-metal-vs-cpu.json`) and floor 0.03.
llama.cpp's own Metal fails it on g01-capital (decode steps in the golden)
and trace02 (traced golden); the report notes those.

Exit: 0 the gate passes with traces checked; 1 the gate fails; 3 the
log-prob rules pass but no --traces was given (rule 4 unchecked).
"""

import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_goldens as cg  # noqa: E402

FLOORS = {"top1": 0.03, "top1_m02": 0.03, "mean_dlp": 0.02, "p95_dlp": 0.06, "mean_kl": 0.002, "p95_kl": 0.006}
TRACE_GOLDENS = {"trace01": "generate/metal", "trace02": "generate2/metal", "trace03-tool-result": "generate3/metal"}
TRACE_NOISE = {"trace01": "metal-vs-cpu.json", "trace02": "set2-metal-vs-cpu.json",
               "trace03-tool-result": "set3-metal-vs-cpu.json"}
# Prompts where llama.cpp's untraced Metal run, teacher-forced, fails plain v2
# against its own free-running goldens (goldens README, "Teacher-forced noise").
METAL_FAILS_PLAIN_V2 = {
    "g01-capital": "llama.cpp's Metal fails plain v2 here too: golden decode steps vs one batch",
    "trace02": "llama.cpp's Metal fails plain v2 here too: the golden was traced",
}


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def within(value, bound):
    """value <= bound, false for a missing or non-finite value."""
    return finite(value) and value <= bound + 1e-12


def metrics(l):
    """The rule-2 metrics of one compare_goldens log-prob record; the
    disagreement rates with the positions they are over."""
    m02 = l["top1_at_margin"]["0.2"]
    top1 = 1 - l["top1_agreement"] if finite(l.get("top1_agreement")) else float("nan")
    m = (1 - m02["rate"]) if m02["rate"] is not None else (0.0 if m02["positions"] == 0 else float("nan"))
    return {
        "top1": (top1, l["positions"]),
        "top1_m02": (m, m02["positions"]),
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


def judge(rec, meas, count_floor, ref_positions):
    """Rule 2 for one prompt. `rec` is compare_goldens' record of the
    candidate; `meas` the measured metrics; `ref_positions` the number of
    reference positions that must all be compared. The count floor uses the
    reference's n (from `meas`), never the candidate's. Returns
    (pass, failures {name: text}, bounds)."""
    fails, b = {}, {}
    l = rec.get("logprobs") if rec else None
    if l is None:
        return False, {"logprobs": "no log-probs compared"}, b
    if l["positions"] != ref_positions:
        fails["positions"] = f"{l['positions']} compared of {ref_positions}"
    got = metrics(l)
    for k, (v, n) in meas.items():
        floor = FLOORS[k]
        if count_floor and k in ("top1", "top1_m02") and n:
            floor = max(floor, 2 / n)
        b[k] = max(1.5 * v, floor)
        if not within(got[k][0], b[k]):
            fails[k] = f"{got[k][0]:.4g} > {b[k]:.4g}" if finite(got[k][0]) else f"{got[k][0]} (not finite)"
    return not fails, fails, b


def pooled(records):
    lps = [r["logprobs"] for r in records if r and r.get("logprobs")]
    n = sum(l["positions"] for l in lps)
    a = sum(l["top1_agreement"] * l["positions"] for l in lps)
    n2 = sum(l["top1_at_margin"]["2.0"]["positions"] for l in lps)
    a2 = sum(l["top1_at_margin"]["2.0"]["agree"] for l in lps)
    top1 = a / n if n else float("nan")
    top1_2 = a2 / n2 if n2 else None
    ok = finite(top1) and top1 >= 0.93 and (top1_2 is None or (finite(top1_2) and top1_2 >= 0.99))
    return {"positions": n, "top1": top1, "top1_at_2": top1_2, "pass": ok}


def traces(goldens, tdir):
    res = {}
    for pid, gd in TRACE_GOLDENS.items():
        gdir = os.path.join(goldens, gd)
        if not os.path.isdir(os.path.join(tdir, pid)):
            res[pid] = {"pass": False, "fails": [f"no {pid}/ in {tdir}"]}
            continue
        g = cg.load(gdir, pid, "golden.json")
        c, ct = cg.load(tdir, pid, "golden.json"), cg.load(tdir, pid, "trace.json")
        gt = cg.load(gdir, pid, "trace.json")
        if not c or not ct:
            res[pid] = {"pass": False, "fails": ["no golden.json or trace.json in the candidate"]}
            continue
        if c.get("prompt_ids") != g["prompt_ids"]:
            res[pid] = {"pass": False, "fails": ["candidate prompt ids differ from the golden's"]}
            continue
        hb = os.path.join(tdir, pid, ct.get("hidden_file", ""))
        if ct.get("hidden_shape") != gt["hidden_shape"] or not os.path.isfile(hb) or \
                os.path.getsize(hb) != 4 * math.prod(gt["hidden_shape"]):
            res[pid] = {"pass": False, "fails": [f"hidden states not {gt['hidden_shape']} F32"]}
            continue
        t = cg.trace_metrics(gdir, tdir, pid)
        ref = next(p["trace"] for p in json.load(open(os.path.join(goldens, "noise", TRACE_NOISE[pid])))["prompts"]
                   if p["id"] == pid)
        worst_bound, last_bound = 1.5 * ref["max_rel_l2"], 1.5 * ref["final_visit_rel_l2"]
        exp_bound = ref["mean_expert_set_agreement"] - 0.10
        fails = []
        for v in t["visits"]:
            # a visit whose reference state is all zeros has no relative error
            if v["rel_l2"] is not None and not within(v["rel_l2"], worst_bound):
                fails.append(f"visit {v['visit']} rel L2 {v['rel_l2']:.4g} > {worst_bound:.4g}")
        if not within(t["final_visit_rel_l2"], last_bound):
            fails.append(f"last-visit rel L2 {t['final_visit_rel_l2']} > {last_bound:.4g}")
        if not finite(t["mean_expert_set_agreement"]) or t["mean_expert_set_agreement"] < exp_bound:
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
    opt = lambda name: args[args.index(name) + 1] if name in args else None
    goldens = opt("--goldens") or os.path.join(ink_dir(), "goldens")
    tdir, out_json = opt("--traces"), opt("--json")
    gate_golden = os.path.join(goldens, "generate-tf", "metal")

    # The gate: against the untraced teacher-forced Metal run
    meas_g = measured(goldens, ["tf-metal-vs-tfcpu.json"])
    expected = list(meas_g)  # every prompt the gate has a measurement for
    gate_recs = {p["id"]: p for p in cg.compare(gate_golden, cand, teacher_forced=False)["prompts"]}
    # The record: plain v2 against the free-running goldens, teacher-forced
    meas_p = measured(goldens, ["metal-vs-cpu.json", "set2-metal-vs-cpu.json"])
    plain_recs = {}
    for gd in ("generate/metal", "generate2/metal"):
        for p in cg.compare(os.path.join(goldens, gd), cand, teacher_forced=True)["prompts"]:
            plain_recs[p["id"]] = p

    rows, report = [], {"candidate": cand, "goldens": goldens, "prompts": {}}
    for pid in expected:
        ref_n = meas_g[pid]["top1"][1]
        g = gate_recs.get(pid)
        if not g or g.get("missing_in_b"):
            rg = (False, {"missing": "prompt not in the candidate"}, {})
        elif not g.get("prompt_ids_equal"):
            rg = (False, {"context": "candidate ids differ from generate-tf/metal's"}, {})
        else:
            rg = judge(g, meas_g[pid], True, ref_n)
        p = plain_recs.get(pid)
        if not p:
            rp = (False, {"missing": "prompt not in the candidate"}, {})
        elif not p.get("context_equal"):
            rp = (False, {"context": "candidate context differs from the golden's"}, {})
        else:
            rp = judge(p, meas_p[pid], False, ref_n)
        # the note explains a metric failure only, not a missing prompt or position
        structural = {"missing", "context", "positions", "logprobs"} & set(rp[1])
        note = METAL_FAILS_PLAIN_V2.get(pid, "") if not rp[0] and not structural else ""
        report["prompts"][pid] = {"gate_pass": rg[0], "gate_fails": rg[1], "gate_bounds": rg[2],
                                  "plain_v2_pass": rp[0], "plain_v2_fails": rp[1], "plain_v2_bounds": rp[2],
                                  "note": note}
        rows.append((pid, rg, rp, note))
    report["pooled_gate"] = pooled([gate_recs.get(pid) for pid in expected])
    report["pooled_plain_v2"] = pooled([plain_recs.get(pid) for pid in expected])
    tr = traces(goldens, tdir) if tdir else {}
    report["traces"], report["traces_checked"] = tr, bool(tdir)
    tr_ok = all(t["pass"] for t in tr.values())
    logprob_gate = all(r[1][0] for r in rows) and report["pooled_gate"]["pass"]
    report["gate_pass"] = logprob_gate and tr_ok and bool(tdir)
    report["plain_v2_pass"] = all(r[2][0] for r in rows) and report["pooled_plain_v2"]["pass"] and tr_ok

    fmt = lambda r: "pass" if r[0] else "FAIL: " + "; ".join(f"{k} {v}" for k, v in r[1].items())
    print("| Prompt | Gate (spec v1.8) | Plain v2, record only | Note |")
    print("|---|---|---|---|")
    for pid, rg, rp, note in rows:
        print(f"| {pid} | {fmt(rg)} | {fmt(rp)} | {note} |")
    for name, key in (("gate", "pooled_gate"), ("plain v2", "pooled_plain_v2")):
        p = report[key]
        num = lambda x: "n/a" if x is None else f"{x:.4f}"
        print(f"\nPooled, {name}: {p['positions']} positions, top-1 {num(p['top1'])} (>= 0.93), "
              f"at margin >= 2.0 {num(p['top1_at_2'])} (>= 0.99): {'pass' if p['pass'] else 'FAIL'}")
    for pid, t in tr.items():
        print(f"Trace {pid}: {'pass' if t['pass'] else 'FAIL: ' + '; '.join(t['fails'][:5])}")
    if not tdir:
        print("Traces: no --traces given; rule 4 NOT checked")
    gate = "PASS" if report["gate_pass"] else ("FAIL" if not (logprob_gate and tr_ok) else "INCOMPLETE (rule 4 unchecked)")
    print(f"\nG6c gate (spec v1.8): {gate}")
    print(f"Plain v2, record only: {'pass' if report['plain_v2_pass'] else 'FAIL'}"
          + ("" if tdir else " (traces unchecked)"))
    if out_json:
        json.dump(report, open(out_json, "w"), indent=1, default=str)
    sys.exit(0 if report["gate_pass"] else (3 if logprob_gate and tr_ok and not tdir else 1))


if __name__ == "__main__":
    main()
