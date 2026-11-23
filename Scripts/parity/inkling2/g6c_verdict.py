#!/usr/bin/env python3
"""Tier G6c for a candidate run against the Inkling 2 reference (stdlib).

    python3 -I Scripts/parity/inkling2/g6c_verdict.py <candidate-dir> [--traces <dir>]
        [--goldens <goldens-dir>] [--json out.json] [--precision-root <dir>]

<candidate-dir> holds the runner's output, in ink-golden's `generate` layout,
for the teacher-forced prompt set (`inputs/generate_prompts_tf.json`: each
log-prob golden's prompt plus all but its last generated id as one ids
prompt, log-probs at every recorded position). `--traces <dir>` holds the
runner's traces of trace01, trace02 and trace03-tool-result, same layout
(golden.json, trace.json, hidden_after_visit.f32). Rule 1 (tokenizer and chat
renders, exact) is tested elsewhere. `--precision-root` (default
$INK/results/k4/precision) holds K4's reference runs:
<root>/<prompt>/<mode>/tf/<prompt>/{golden,logprobs}.json and
<root>/<prompt>/<mode>/traces/<prompt>/..., mode `fp32` (exact), `llamaCPU` or
`llamaExperts` (llama.cpp's rounding).

The gate (spec v1.12, section 8: G6c re-based on the exact model; goldens
README, "G6c re-based"). Rules 1, 3 and 4 as in v1.8; rule 2 is (a)-(d):
- (a) the runner against `.fp32`, teacher-forced, on every teacher-forced
  prompt of at most 347 tokens (all but lp06 and lp07) and the three traces:
  mean KL <= 0.0015, p95 KL <= 0.005, max |delta lp| <= 0.1, top-1 agreement
  100% where the reference margin is >= 0.2; routed-expert sets differing on
  at most 2% of token-visits (mean per-visit set agreement >= 0.98), each flip
  at a selection margin below 0.0258 where the traces record selection
  margins (trace.json carries none today: printed as not checkable). The
  report also prints the bounds re-derived as max(3 x the measured worst,
  today's bound), for the hand-in that records them.
- (b) the reference in llama.cpp's rounding mode against `generate-tf/metal`
  under the addendum bounds (below): `.llamaCPU` for prompts under 32 tokens,
  `.llamaExperts` from 32 up; required on trace02, g06-chat-none and
  lp09-chat-default, judged wherever that mode's run exists.
- (c1) the runner against `generate-tf/cpu` within 2 x the addendum bounds,
  on every prompt.
- (c2) the runner no farther from `generate-tf/cpu` than `.fp32` is, per
  prompt and metric, within 10% (bound max(1.1 x fp32's value, the
  addendum's floor)), on every teacher-forced prompt (a) covers.
- (d) next-token quality (next_token_quality.py) over its five natural-text
  prompts: runner minus llama.cpp Metal and minus CPU, each with mean >=
  -0.005 nats and a 95% bootstrap lower bound no lower than llama.cpp's own
  Metal-minus-CPU lower bound on the same positions (full-vocabulary
  log-probabilities; top-20 only if a run lacks them).
A rule whose required inputs are missing makes the verdict INCOMPLETE, not
pass. Any non-finite metric fails.

Printed for the record, not gating:
- as written, v1.8 (v2 with its addendum): rule 2 per prompt against
  `generate-tf/metal`, each metric <= max(1.5 x the value measured for that
  prompt in `noise/tf-metal-vs-tfcpu.json`, floor), floors max(0.03, 2/n)
  for the two disagreement rates (n = the reference positions the rate is
  over), 0.02 mean |delta lp|, 0.06 p95 |delta lp|, 0.002 mean KL, 0.006 p95
  KL; every expected prompt with every position compared. These are the
  "addendum bounds" (b) and (c1) use.
- plain v2 as first adopted (spec v1.7): rule 2 against the free-running
  Metal goldens with the free-running measurement and floor 0.03. llama.cpp's
  own Metal fails it on g01-capital and trace02; the report notes those.
Rule 3, pooled over the prompts against `generate-tf/metal`: top-1 >= 0.93,
and >= 0.99 where the reference margin is >= 2.0. Rule 4, per trace: hidden
rel L2 <= 1.5 x the measured worst visit at every visit and <= 1.5 x the
measured last visit at the last; mean expert-set agreement >= the measured
mean - 0.10 (`noise/{metal,set2,set3}-metal-vs-cpu.json`).

Exit: 0 the gate passes; 1 a rule fails; 3 everything checked passes but no
--traces was given (rule 4 unchecked); 4 everything checked passes but a
rule's required inputs are missing (coverage incomplete).
"""

import json
import math
import os
import random
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


# ---------------------------------------------------------------- the gate: re-based tier (spec v1.12)
EXACT_BOUNDS = {"mean_kl": 0.0015, "p95_kl": 0.005, "max_dlp": 0.1}  # (a); 5 x K4's trace02 runner-vs-fp32
EXPERT_AGREEMENT_MIN = 0.98   # (a): sets differ on at most 2% of token-visits
FLIP_MARGIN_MAX = 0.0258      # (a): each flip below Metal's largest flip margin, where margins are recorded
A_MAX_TOKENS = 347            # (a), (c2): teacher-forced prompts of at most this many tokens (lp05)
B_REQUIRED = ("trace02", "g06-chat-none", "lp09-chat-default")
B_SHORT = 32                  # (b): .llamaCPU under this many tokens, .llamaExperts from it up
QUALITY_MEAN_MIN = -0.005     # (d), nats
QUALITY_MARGIN = 0.01         # (d) first form (lower bound >= -0.01), superseded: printed for the record


def fmt4(x):
    return f"{x:.4g}" if finite(x) else str(x)


def pair(ref_dir, cand_dir, pid):
    """compare_goldens' log-prob record of cand against ref for one prompt, or None."""
    ga, gb = cg.load(ref_dir, pid, "golden.json"), cg.load(cand_dir, pid, "golden.json")
    la, lb = cg.load(ref_dir, pid, "logprobs.json"), cg.load(cand_dir, pid, "logprobs.json")
    if not (ga and gb and la and lb) or ga["prompt_ids"] != gb["prompt_ids"]:
        return None
    return cg.lp_metrics(la, lb, len(ga["prompt_ids"]))


def has_run(root, pid, mode, kind="tf"):
    return bool(root) and os.path.isdir(os.path.join(root, pid, mode, kind, pid))


def flips(ref_traces, cand_traces, pid):
    """(a)'s margin clause: the selection margins of the token-visits whose expert
    set differs, when the reference's trace.json records them (per visit
    `margins`, one per token). None when it does not."""
    ta, tb = cg.load(ref_traces, pid, "trace.json"), cg.load(cand_traces, pid, "trace.json")
    if not ta or not tb or any("margins" not in v for v in ta["visits"]):
        return None
    out = []
    for va, vb in zip(ta["visits"], tb["visits"]):
        for t, (x, y) in enumerate(zip(va["experts"], vb["experts"])):
            if set(x) != set(y):
                out.append(va["margins"][t])
    return out


def expert_rule(root, pid, tdir):
    """(a)'s expert-set part for one trace prompt: (fails, agreement, flip margins) or None
    when the runner's or fp32's trace is missing."""
    ref = os.path.join(root, pid, "fp32", "traces") if root else None
    if not (ref and os.path.isdir(os.path.join(ref, pid)) and tdir and os.path.isdir(os.path.join(tdir, pid))):
        return None
    t = cg.trace_metrics(ref, tdir, pid)
    ex = t["mean_expert_set_agreement"] if t else None
    fails = {}
    if not (finite(ex) and ex >= EXPERT_AGREEMENT_MIN):
        fails["expert_sets"] = f"agreement {fmt4(ex)} < {EXPERT_AGREEMENT_MIN} (more than 2% of token-visits differ)"
    fm = flips(ref, tdir, pid)
    if fm is not None and any(not (finite(m) and m < FLIP_MARGIN_MAX) for m in fm):
        fails["flip_margin"] = f"a flip at margin {fmt4(max(fm))} >= {FLIP_MARGIN_MAX}"
    return fails, ex, fm


def addendum_bound(k, v, n):
    floor = max(FLOORS[k], 2 / n) if k in ("top1", "top1_m02") and n else FLOORS[k]
    return max(1.5 * v, floor), floor


def rebased(goldens, cand, tdir, root, meas_g, expected):
    """Rule 2 of the v1.12 gate. Each part: {prompt: {pass, fails, ...}} plus
    `missing` (required inputs not present)."""
    out = {k: {"prompts": {}, "missing": []} for k in ("a", "b", "c1", "c2")}
    out["d"] = None
    cpu = os.path.join(goldens, "generate-tf", "cpu")
    metal = os.path.join(goldens, "generate-tf", "metal")
    tokens = {}
    for pid in expected:
        g = cg.load(metal, pid, "golden.json")
        tokens[pid] = len(g["prompt_ids"]) if g else None
    a_cover = [p for p in expected if tokens[p] is not None and tokens[p] <= A_MAX_TOKENS]
    out["a"]["covered"] = a_cover + [p for p in TRACE_GOLDENS if p not in a_cover]
    worst = {k: 0.0 for k in EXACT_BOUNDS}
    for pid in expected:
        # (a) the runner against the exact reference, teacher-forced
        if pid in a_cover:
            if not has_run(root, pid, "fp32"):
                out["a"]["missing"].append(f"{pid}: no fp32 tf run")
            else:
                l = pair(os.path.join(root, pid, "fp32", "tf"), cand, pid)
                if l is None:
                    out["a"]["prompts"][pid] = {"pass": False, "fails": {"missing": "runner output, or ids differ"}}
                else:
                    got = {"mean_kl": l["mean_kl_top20_approx"], "p95_kl": l["p95_kl_top20_approx"],
                           "max_dlp": l["max_abs_dlogprob_top1"]}
                    fails = {k: f"{fmt4(got[k])} > {b}" for k, b in EXACT_BOUNDS.items() if not within(got[k], b)}
                    for k in worst:
                        worst[k] = max(worst[k], got[k]) if finite(got[k]) else float("nan")
                    m02 = l["top1_at_margin"]["0.2"]
                    if m02["positions"] and m02["agree"] != m02["positions"]:
                        fails["top1_at_margin_0.2"] = f"{m02['agree']}/{m02['positions']}"
                    out["a"]["prompts"][pid] = {"pass": not fails, "fails": fails, "metrics": got}
        # (b) the reference in llama.cpp's rounding mode against tf Metal, addendum bounds
        mode = "llamaCPU" if tokens[pid] is not None and tokens[pid] < B_SHORT else "llamaExperts"
        if has_run(root, pid, mode):
            l = pair(metal, os.path.join(root, pid, mode, "tf"), pid)
            if l is None:
                out["b"]["prompts"][pid] = {"mode": mode, "pass": False, "fails": {"missing": "reference output, or ids differ"}}
            else:
                ok, fails, _ = judge({"logprobs": l}, meas_g[pid], True, meas_g[pid]["top1"][1])
                out["b"]["prompts"][pid] = {"mode": mode, "pass": ok, "fails": fails}
        elif pid in B_REQUIRED:
            out["b"]["missing"].append(f"{pid}: no {mode} tf run")
        # (c1) within 2 x the addendum bounds of tf CPU
        l = pair(cpu, cand, pid)
        if l is None:
            out["c1"]["prompts"][pid] = {"pass": False, "fails": {"missing": "runner output, or ids differ"}}
            continue
        got = metrics(l)
        fails = {}
        for k, (v, n) in meas_g[pid].items():
            bound = 2 * addendum_bound(k, v, n)[0]
            if not within(got[k][0], bound):
                fails[k] = f"{fmt4(got[k][0])} > {fmt4(bound)}"
        out["c1"]["prompts"][pid] = {"pass": not fails, "fails": fails}
        # (c2) no farther from tf CPU than fp32 is (+10%, addendum floors), where (a) applies
        if pid in a_cover:
            le = pair(cpu, os.path.join(root, pid, "fp32", "tf"), pid) if has_run(root, pid, "fp32") else None
            if le is None:
                out["c2"]["missing"].append(f"{pid}: no fp32 tf run")
            else:
                ge = metrics(le)
                fails = {}
                for k in got:
                    floor = addendum_bound(k, 0.0, meas_g[pid][k][1])[1]
                    bound = max(1.1 * ge[k][0], floor) if finite(ge[k][0]) else float("nan")
                    if not within(got[k][0], bound):
                        fails[k] = f"{fmt4(got[k][0])} > {fmt4(bound)} (fp32 {fmt4(ge[k][0])})"
                out["c2"]["prompts"][pid] = {"pass": not fails, "fails": fails}
    # (a), expert sets on the three traces (and trace02's, which is also a tf prompt)
    for pid in TRACE_GOLDENS:
        r = expert_rule(root, pid, tdir)
        if r is None:
            out["a"]["missing"].append(f"{pid}: no runner or fp32 trace")
            continue
        fails, ex, fm = r
        rec = out["a"]["prompts"].setdefault(pid, {"pass": True, "fails": {}})
        rec["fails"].update(fails)
        rec["pass"] = rec["pass"] and not fails
        rec["expert_set_agreement"] = ex
        rec["flip_margins"] = "not recorded" if fm is None else {"flips": len(fm), "max": max(fm) if fm else None}
    out["a"]["measured_worst"] = worst
    out["a"]["rederived"] = {k: max(3 * worst[k], b) if finite(worst[k]) else float("nan")
                             for k, b in EXACT_BOUNDS.items()}
    # (d) quality: the true next token's log-probability, runner against both llama.cpp runs
    import next_token_quality as ntq
    rows, used, missing = ntq.collect([cand, metal, cpu], goldens, ntq.DEFAULT)
    measure = "exact" if rows["exact"] else "top20"
    rs = rows[measure]
    d = {"measure": measure, "positions": len(rs), "prompts": used, "missing": missing, "pairs": {}}
    if rs:
        rng = random.Random(20261123)
        mc = [r[3] - r[4] for r in rs]  # llama.cpp against itself on the same positions: the yardstick
        mc_lo, mc_hi = ntq.boot(mc, 10000, rng)
        d["llama_metal_minus_cpu"] = {"mean": ntq.mean(mc), "ci95": [mc_lo, mc_hi]}
        for k, name in ((1, "metal"), (2, "cpu")):
            diffs = [r[2] - r[2 + k] for r in rs]
            lo, hi = ntq.boot(diffs, 10000, rng)
            mean_d = ntq.mean(diffs)
            d["pairs"][f"runner - {name}"] = {
                "mean": mean_d, "ci95": [lo, hi],
                "pass": finite(mean_d) and mean_d >= QUALITY_MEAN_MIN and finite(lo) and lo >= mc_lo,
                "pass_superseded": finite(lo) and lo >= -QUALITY_MARGIN}
        d["pass"] = all(v["pass"] for v in d["pairs"].values())
        d["pass_superseded"] = all(v["pass_superseded"] for v in d["pairs"].values())
        d["llama_passes_superseded"] = finite(mc_lo) and mc_lo >= -QUALITY_MARGIN
    else:
        d["pass"] = False
    out["d"] = d
    return out


def rebased_state(rb):
    """'pass', 'fail' or 'incomplete' for rule 2 as a whole, with the failing and missing parts."""
    failed = [k for k in ("a", "b", "c1", "c2") if any(not v["pass"] for v in rb[k]["prompts"].values())]
    missing = [k for k in ("a", "b", "c1", "c2") if rb[k]["missing"]]
    d = rb["d"]
    if d["missing"] or not d["positions"]:
        missing.append("d")
    elif not d["pass"]:
        failed.append("d")
    return ("fail" if failed else "incomplete" if missing else "pass"), failed, missing


def print_rebased(rb):
    titles = (("a", "(a) runner vs the exact reference (fp32)"),
              ("b", "(b) llama.cpp-mode reference vs tf Metal, addendum bounds"),
              ("c1", "(c1) runner vs tf CPU within 2x the addendum bounds"),
              ("c2", "(c2) runner no farther from tf CPU than fp32 (+10%)"))
    print("\nRule 2, re-based (spec v1.12; gates):")
    print("| Part | Prompts judged | Result | Missing (required) |")
    print("|---|---|---|---|")
    for key, title in titles:
        part = rb[key]["prompts"]
        bad = {p: v["fails"] for p, v in part.items() if not v["pass"]}
        res = ("pass" if part else "no data") if not bad else \
            "FAIL " + "; ".join(f"{p}: {', '.join(f'{k} {x}' for k, x in f.items())}" for p, f in bad.items())
        judged = ", ".join(f"{p}" + (f" ({v['mode']})" if "mode" in v else "") for p, v in part.items())
        print(f"| {title} | {len(part) - len(bad)}/{len(part)}: {judged} | {res} | {'; '.join(rb[key]['missing']) or '-'} |")
    a = rb["a"]
    for pid, v in a["prompts"].items():
        if "expert_set_agreement" in v:
            fm = v["flip_margins"]
            fm_s = "selection margins not recorded in trace.json: margin clause not checkable" if fm == "not recorded" \
                else f"{fm['flips']} flips, max margin {fmt4(fm['max'])} (< {FLIP_MARGIN_MAX})"
            print(f"  (a) {pid} expert sets: agreement {fmt4(v['expert_set_agreement'])} (>= {EXPERT_AGREEMENT_MIN}); {fm_s}")
    print("  (a) measured worst over the prompts judged: "
          + ", ".join(f"{k} {fmt4(v)}" for k, v in a["measured_worst"].items())
          + "; re-derived bounds max(3 x worst, today's): "
          + ", ".join(f"{k} {fmt4(v)}" for k, v in a["rederived"].items())
          + ("" if not a["missing"] else " (coverage incomplete: not yet the re-derivation)"))
    d = rb["d"]
    if not d["positions"]:
        print(f"(d) quality: no positions; missing {', '.join(d['missing']) or '-'}")
    else:
        pairs = "; ".join(f"{k} {v['mean']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]" for k, v in d["pairs"].items())
        mc = d["llama_metal_minus_cpu"]
        print(f"(d) quality ({d['measure']}, {d['positions']} positions, {', '.join(d['prompts'])}"
              + (f"; missing {', '.join(d['missing'])}" if d["missing"] else "") + f"): {pairs}; "
              + f"llama.cpp Metal - CPU {mc['mean']:+.4f} [{mc['ci95'][0]:+.4f}, {mc['ci95'][1]:+.4f}]: "
              + ("pass" if d["pass"] else "FAIL")
              + f" (mean >= {QUALITY_MEAN_MIN}, lower bound >= llama.cpp's own)")
        print(f"    first form, superseded (lower bound >= -{QUALITY_MARGIN}; llama.cpp Metal - CPU "
              + ("passes" if d["llama_passes_superseded"] else "fails") + " it too): "
              + ("pass" if d["pass_superseded"] else "fail"))


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
    if not args or args[0].startswith("--"):
        sys.exit(__doc__)
    cand = args[0]
    opt = lambda name: args[args.index(name) + 1] if name in args else None
    goldens = opt("--goldens") or os.path.join(ink_dir(), "goldens")
    tdir, out_json = opt("--traces"), opt("--json")
    root = opt("--precision-root") or os.path.join(os.path.dirname(goldens), "results", "k4", "precision")
    root = root if os.path.isdir(root) else None
    gate_golden = os.path.join(goldens, "generate-tf", "metal")

    # Rule 2 as written in v1.8 (record only), and rule 3: against the untraced teacher-forced Metal run
    meas_g = measured(goldens, ["tf-metal-vs-tfcpu.json"])
    expected = list(meas_g)  # every prompt the gate has a measurement for
    gate_recs = {p["id"]: p for p in cg.compare(gate_golden, cand, teacher_forced=False)["prompts"]}
    # Plain v2 against the free-running goldens, teacher-forced (record only)
    meas_p = measured(goldens, ["metal-vs-cpu.json", "set2-metal-vs-cpu.json"])
    plain_recs = {}
    for gd in ("generate/metal", "generate2/metal"):
        for p in cg.compare(os.path.join(goldens, gd), cand, teacher_forced=True)["prompts"]:
            plain_recs[p["id"]] = p

    rows, report = [], {"candidate": cand, "goldens": goldens, "precision_root": root, "prompts": {}}
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
        report["prompts"][pid] = {"as_written_v18_pass": rg[0], "as_written_v18_fails": rg[1],
                                  "as_written_v18_bounds": rg[2], "plain_v2_pass": rp[0],
                                  "plain_v2_fails": rp[1], "plain_v2_bounds": rp[2], "note": note}
        rows.append((pid, rg, rp, note))
    report["pooled"] = pooled([gate_recs.get(pid) for pid in expected])
    report["pooled_plain_v2"] = pooled([plain_recs.get(pid) for pid in expected])
    tr = traces(goldens, tdir) if tdir else {}
    report["traces"], report["traces_checked"] = tr, bool(tdir)
    tr_ok = all(t["pass"] for t in tr.values())
    # Rule 2, re-based: the gate
    rb = rebased(goldens, cand, tdir, root, meas_g, expected)
    state, failed, missing = rebased_state(rb)
    report["rule2_rebased"], report["rule2_state"] = rb, {"state": state, "failed": failed, "missing": missing}
    checked_ok = state != "fail" and report["pooled"]["pass"] and tr_ok
    report["gate_pass"] = checked_ok and state == "pass" and bool(tdir)
    report["as_written_v18_pass"] = all(r[1][0] for r in rows) and report["pooled"]["pass"] and tr_ok and bool(tdir)
    report["plain_v2_pass"] = all(r[2][0] for r in rows) and report["pooled_plain_v2"]["pass"] and tr_ok

    fmt = lambda r: "pass" if r[0] else "FAIL: " + "; ".join(f"{k} {v}" for k, v in r[1].items())
    print_rebased(rb)
    p = report["pooled"]
    num = lambda x: "n/a" if x is None else f"{x:.4f}"
    print(f"\nRule 3, pooled against tf Metal: {p['positions']} positions, top-1 {num(p['top1'])} (>= 0.93), "
          f"at margin >= 2.0 {num(p['top1_at_2'])} (>= 0.99): {'pass' if p['pass'] else 'FAIL'}")
    for pid, t in tr.items():
        print(f"Rule 4, trace {pid}: {'pass' if t['pass'] else 'FAIL: ' + '; '.join(t['fails'][:5])}")
    if not tdir:
        print("Rule 4: no --traces given; NOT checked")
    if not checked_ok:
        gate = "FAIL"
    elif state == "incomplete":
        gate = "INCOMPLETE (rule 2 inputs missing: " + ", ".join(missing) + ")"
    elif not tdir:
        gate = "INCOMPLETE (rule 4 unchecked)"
    else:
        gate = "PASS"
    print(f"\nG6c gate (spec v1.12, re-based): {gate}"
          + (f"; failing: rule 2 {', '.join(failed)}" if failed else "")
          + ("; rule 3" if not report["pooled"]["pass"] else "") + ("; rule 4" if not tr_ok else ""))

    print("\nFor the record (does not gate): rule 2 as written in v1.8, and plain v2 (v1.7)")
    print("| Prompt | As written, v1.8 | Plain v2 | Note |")
    print("|---|---|---|---|")
    for pid, rg, rp, note in rows:
        print(f"| {pid} | {fmt(rg)} | {fmt(rp)} | {note} |")
    p = report["pooled_plain_v2"]
    print(f"Pooled, plain v2: {p['positions']} positions, top-1 {num(p['top1'])}, at margin >= 2.0 "
          f"{num(p['top1_at_2'])}: {'pass' if p['pass'] else 'FAIL'}")
    print(f"As written, v1.8: {'PASS' if report['as_written_v18_pass'] else 'FAIL'}"
          + ("" if tdir else " (traces unchecked)"))
    print(f"Plain v2: {'pass' if report['plain_v2_pass'] else 'FAIL'}" + ("" if tdir else " (traces unchecked)"))
    if out_json:
        json.dump(report, open(out_json, "w"), indent=1, default=str)
    if report["gate_pass"]:
        sys.exit(0)
    if not checked_ok:
        sys.exit(1)
    sys.exit(4 if state == "incomplete" else 3)


if __name__ == "__main__":
    main()
