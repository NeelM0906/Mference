#!/usr/bin/env python3
"""Agreement metrics between two ink-golden `generate` outputs (stdlib only).

    python3 -I Scripts/parity/inkling2/compare_goldens.py <run-a> <run-b> [--json out.json] [--teacher-forced]

Used twice: on the reference against itself (Metal vs Metal in a fresh process,
Metal vs CPU) to measure the reference's own noise, from which the G6
thresholds are set; and later on the native runner's output (same format)
against the reference.

Per prompt:
- prompt tokens identical (they must be: same tokenizer and template);
- greedy continuation: length of the common prefix of the generated ids, and
  whether the two runs are identical;
- log-probabilities, at every position where both runs saw the same context
  (all recorded prompt positions, and generated positions up to and including
  the first divergence): top-1 agreement (overall and where run A's top-1
  leads its top-2 by >= 0.2, 0.5, 2.0 nats), |delta| of the log-probability
  each run gives to run A's top-1 token (mean, nearest-rank p95, p99, max),
  top-20 id overlap, and an approximate KL(A||B) over the union of the two
  top-20 lists (a token missing from one list gets that list's 20th
  log-probability, an upper bound on its true value; mean, nearest-rank p95,
  max); pooled over prompts, top-1 agreement overall and by margin;
- trace: per layer visit, the relative L2 error of the hidden states
  (||a - b|| / ||a|| over all positions) and the share of (token, visit) whose
  eight routed expert ids agree as a set.
"""

import json
import math
import os
import struct
import sys


def load(d, pid, name):
    p = os.path.join(d, pid, name)
    return json.load(open(p)) if os.path.exists(p) else None


def f32(path):
    b = open(path, "rb").read()
    return struct.unpack("<%df" % (len(b) // 4), b)


MARGINS = (0.2, 0.5, 2.0)


def p95(values):
    """Nearest-rank 95th percentile: the ceil(0.95 n)-th smallest value."""
    s = sorted(values)
    return s[max(0, math.ceil(0.95 * len(s)) - 1)] if s else None


def lp_metrics(la, lb, aligned_upto):
    rows_a = {r["pos"]: r for r in la}
    rows_b = {r["pos"]: r for r in lb}
    n = agree = 0
    by_margin = {m: [0, 0] for m in MARGINS}  # margin of run A's top-1 over its top-2: [positions, top-1 agreements]
    dl = []
    overlap = []
    kls = []
    for pos in sorted(rows_a):
        if pos > aligned_upto or pos not in rows_b:
            continue
        a, b = rows_a[pos], rows_b[pos]
        ta = {i: lp for i, lp in a["top"]}
        tb = {i: lp for i, lp in b["top"]}
        top1a, top1b = a["top"][0][0], b["top"][0][0]
        n += 1
        agree += top1a == top1b
        margin = a["top"][0][1] - a["top"][1][1]
        for m in MARGINS:
            if margin >= m:
                by_margin[m][0] += 1
                by_margin[m][1] += top1a == top1b
        floor_a, floor_b = a["top"][-1][1], b["top"][-1][1]
        dl.append(abs(ta[top1a] - tb.get(top1a, floor_b)))
        overlap.append(len(set(ta) & set(tb)) / len(ta))
        kl = 0.0
        for i in set(ta) | set(tb):
            pa = ta.get(i, floor_a)
            pb = tb.get(i, floor_b)
            kl += math.exp(pa) * (pa - pb)
        kls.append(max(kl, 0.0))
    s = sorted(dl)
    return {
        "positions": n,
        "top1_agreement": agree / n if n else None,
        "max_abs_dlogprob_top1": max(dl) if dl else None,
        "p99_abs_dlogprob_top1": s[min(len(s) - 1, int(0.99 * len(s)))] if s else None,
        "mean_abs_dlogprob_top1": sum(dl) / len(dl) if dl else None,
        "min_top20_overlap": min(overlap) if overlap else None,
        "mean_kl_top20_approx": sum(kls) / len(kls) if kls else None,
        "max_kl_top20_approx": max(kls) if kls else None,
        "p95_abs_dlogprob_top1": p95(dl),
        "p95_kl_top20_approx": p95(kls),
        "top1_at_margin": {str(m): {"positions": c[0], "agree": c[1], "rate": (c[1] / c[0] if c[0] else None)}
                           for m, c in by_margin.items()},
    }


def trace_metrics(da, db, pid):
    ta, tb = load(da, pid, "trace.json"), load(db, pid, "trace.json")
    if not ta or not tb:
        return None
    nv, nt, ne = ta["hidden_shape"]
    ha = f32(os.path.join(da, pid, ta["hidden_file"]))
    hb = f32(os.path.join(db, pid, tb["hidden_file"]))
    per = []
    for v in range(nv):
        off = v * nt * ne
        num = den = 0.0
        for i in range(off, off + nt * ne):
            d = ha[i] - hb[i]
            num += d * d
            den += ha[i] * ha[i]
        ea, eb = ta["visits"][v]["experts"], tb["visits"][v]["experts"]
        same = sum(set(x) == set(y) for x, y in zip(ea, eb))
        per.append({"visit": v, "layer": ta["visits"][v]["layer"],
                    "rel_l2": math.sqrt(num / den) if den else None,
                    "expert_set_agreement": same / len(ea)})
    # a visit whose reference state is all zeros has no relative error (None)
    rels = [p["rel_l2"] for p in per if p["rel_l2"] is not None]
    return {
        "visits": per,
        "zero_norm_visits": sum(1 for p in per if p["rel_l2"] is None),
        "max_rel_l2": max(rels) if rels else None,
        "final_visit_rel_l2": per[-1]["rel_l2"],
        "min_expert_set_agreement": min(p["expert_set_agreement"] for p in per),
        "mean_expert_set_agreement": sum(p["expert_set_agreement"] for p in per) / len(per),
    }


def compare(da, db, teacher_forced=False):
    """The comparison of run B with run A as one dict (what --json writes)."""
    ia = json.load(open(os.path.join(da, "index.json")))
    result = {"a": da, "b": db, "prompts": []}
    for p in ia["prompts"]:
        pid = p["id"]
        ga, gb = load(da, pid, "golden.json"), load(db, pid, "golden.json")
        if gb is None:
            # teacher-forced sets cover only the log-prob prompts: skip the rest quietly
            if not teacher_forced:
                result["prompts"].append({"id": pid, "missing_in_b": True})
            continue
        if teacher_forced:
            # B ran A's prompt + generated ids (but the last) as one prompt:
            # every recorded position has the same context in both
            ctx = ga["prompt_ids"] + ga["generated_ids"][:-1]
            r = {"id": pid, "teacher_forced": True, "context_equal": gb["prompt_ids"] == ctx,
                 "prompt_ids_equal": gb["prompt_ids"] == ctx}
            if r["context_equal"] and ga.get("logprobs_file") and gb.get("logprobs_file"):
                la = load(da, pid, "logprobs.json")
                r["logprobs"] = lp_metrics(la, load(db, pid, "logprobs.json"), len(ctx))
                # every position of A must have been compared
                r["all_positions_compared"] = r["logprobs"]["positions"] == len(la)
            result["prompts"].append(r)
            continue
        r = {"id": pid, "prompt_ids_equal": ga["prompt_ids"] == gb["prompt_ids"]}
        xa, xb = ga["generated_ids"], gb["generated_ids"]
        k = 0
        while k < min(len(xa), len(xb)) and xa[k] == xb[k]:
            k += 1
        r["generated_identical"] = xa == xb
        r["common_prefix"] = k
        r["generated_a"] = len(xa)
        # positions with the same context: every prompt position (pos <= n_prompt - 2
        # predicts a prompt token, n_prompt - 1 the first generated one) and generated
        # positions up to the first divergence
        n_prompt = len(ga["prompt_ids"])
        aligned = n_prompt - 1 + k
        # different prompt tokens: no position has the same context, so nothing else is comparable
        if not r["prompt_ids_equal"]:
            result["prompts"].append(r)
            continue
        if ga.get("logprobs_file") and gb.get("logprobs_file"):
            r["logprobs"] = lp_metrics(load(da, pid, "logprobs.json"), load(db, pid, "logprobs.json"), aligned)
        t = trace_metrics(da, db, pid)
        if t:
            r["trace"] = t
        result["prompts"].append(r)
    lps = [p["logprobs"] for p in result["prompts"] if p.get("logprobs")]
    pooled_n = sum(l["positions"] for l in lps)
    pooled = {
        "positions": pooled_n,
        "top1_agreement": (sum(l["top1_agreement"] * l["positions"] for l in lps) / pooled_n) if pooled_n else None,
    }
    for m in MARGINS:
        k = str(m)
        pn = sum(l["top1_at_margin"][k]["positions"] for l in lps)
        pa = sum(l["top1_at_margin"][k]["agree"] for l in lps)
        pooled[f"top1_at_margin_{k}"] = pa / pn if pn else None
    result["pooled_logprobs"] = pooled
    if teacher_forced:
        result["summary_teacher_forced"] = {
            "prompts": len(result["prompts"]),
            "all_contexts_equal": all(p.get("context_equal") for p in result["prompts"]),
            "all_positions_compared": all(p.get("all_positions_compared") for p in result["prompts"]),
        }
    result["summary"] = {
        "prompts": len(result["prompts"]),
        "all_prompt_ids_equal": all(p.get("prompt_ids_equal") for p in result["prompts"]),
        "greedy_identical": sum(1 for p in result["prompts"] if p.get("generated_identical")),
        "min_common_prefix": min((p["common_prefix"] for p in result["prompts"] if "common_prefix" in p), default=None),
        "top1_agreement_min": min((l["top1_agreement"] for l in lps), default=None),
        "max_abs_dlogprob_top1": max((l["max_abs_dlogprob_top1"] for l in lps), default=None),
        "max_kl_top20_approx": max((l["max_kl_top20_approx"] for l in lps), default=None),
    }
    return result


def main():
    da, db = sys.argv[1], sys.argv[2]
    out_json = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    # --teacher-forced: run B is make_tf_prompts.py's set run on run A's goldens
    result = compare(da, db, teacher_forced="--teacher-forced" in sys.argv)
    txt = json.dumps(result, indent=1)
    if out_json:
        open(out_json, "w").write(txt)
    print(json.dumps(result.get("summary_teacher_forced", result["summary"]), indent=1))
    for p in result["prompts"]:
        if p.get("teacher_forced"):
            line = f"{p['id']}: teacher-forced context_equal={p.get('context_equal')} all_positions={p.get('all_positions_compared')}"
        else:
            line = f"{p['id']}: ids_equal={p.get('prompt_ids_equal')} identical={p.get('generated_identical')} prefix={p.get('common_prefix')}/{p.get('generated_a')}"
        if p.get("logprobs"):
            l = p["logprobs"]
            line += f" top1={l['top1_agreement']:.4f} max|dlp|={l['max_abs_dlogprob_top1']:.4g} maxKL~={l['max_kl_top20_approx']:.3g}"
        if p.get("trace"):
            t = p["trace"]
            mr = "n/a" if t["max_rel_l2"] is None else f"{t['max_rel_l2']:.3g}"
            line += f" trace max_rel_l2={mr} experts_min={t['min_expert_set_agreement']:.3f}"
        print(line)


if __name__ == "__main__":
    main()
