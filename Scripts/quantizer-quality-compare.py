#!/usr/bin/env python3
"""W2.1b offline comparator: KL divergence between two logit dumps.

Consumes the dumps written by `QuantizerQualityMeasurement` (each model run
alone, per AGENTS.md's one-model-process rule) and reports the model-level
quality numbers for docs/QUANTIZER_QUALITY.md.

    Scripts/quantizer-quality-compare.py <dump-root> \
        --reference trusted --candidate ours [--noise trusted-repeat]

Every reported KL divergence is exact: the dumps hold the FULL vocabulary at
every position in the runner's own FP16, so nothing is truncated to a top-K and
no log-sum-exp correction is needed. FP16 is widened to FP32 before the softmax.

D_KL(P_reference || Q_candidate) is measured in nats.
"""
import argparse, json, os, sys
import numpy as np


def load_meta(root, label):
    with open(os.path.join(root, label, "meta.json")) as f:
        return json.load(f)


def logits(root, label, item):
    path = os.path.join(root, label, item["file"])
    a = np.memmap(path, dtype=np.float16, mode="r")
    expected = item["positions"] * item["vocab"]
    if a.size != expected:
        raise SystemExit(f"{path}: {a.size} values, expected {expected}")
    return a.reshape(item["positions"], item["vocab"])


def kl_rows(p_logits, q_logits, chunk=32):
    """Exact per-position D_KL(P||Q) in nats, plus agreement diagnostics."""
    out = {"kl": [], "top1": [], "top5": [], "maxdlogit": []}
    n = p_logits.shape[0]
    for start in range(0, n, chunk):
        p = np.asarray(p_logits[start:start + chunk], dtype=np.float32)
        q = np.asarray(q_logits[start:start + chunk], dtype=np.float32)
        out["maxdlogit"].extend(np.abs(p - q).max(axis=1).tolist())
        # log-softmax in a numerically safe form
        lp = p - p.max(axis=1, keepdims=True)
        lp -= np.log(np.exp(lp).sum(axis=1, keepdims=True))
        lq = q - q.max(axis=1, keepdims=True)
        lq -= np.log(np.exp(lq).sum(axis=1, keepdims=True))
        pp = np.exp(lp)
        out["kl"].extend((pp * (lp - lq)).sum(axis=1).tolist())
        out["top1"].extend((p.argmax(axis=1) == q.argmax(axis=1)).tolist())
        kp = np.argpartition(-p, 5, axis=1)[:, :5]
        kq = np.argpartition(-q, 5, axis=1)[:, :5]
        for i in range(p.shape[0]):
            out["top5"].append(len(set(kp[i].tolist()) & set(kq[i].tolist())) / 5.0)
    return out


def summarize(name, values):
    a = np.asarray(values, dtype=np.float64)
    return (f"{name:24} n={a.size:6d} mean={a.mean():.6g} median={np.median(a):.6g} "
            f"p99={np.percentile(a, 99):.6g} max={a.max():.6g}")


def compare(root, ref, cand, tag):
    ref_meta, cand_meta = load_meta(root, ref), load_meta(root, cand)
    cand_items = {i["name"]: i for i in cand_meta["items"]}
    all_kl, all_top1, all_top5, all_dl = [], [], [], []
    print(f"\n=== {tag}: D_KL({ref} || {cand}) ===")
    identical = True
    for item in ref_meta["items"]:
        other = cand_items.get(item["name"])
        if other is None:
            print(f"  {item['name']}: MISSING in {cand}")
            continue
        if item["positions"] != other["positions"]:
            raise SystemExit(f"{item['name']}: position count differs — "
                             "the two runs did not teacher-force the same sequence")
        p, q = logits(root, ref, item), logits(root, cand, other)
        if not np.array_equal(np.asarray(p), np.asarray(q)):
            identical = False
        r = kl_rows(p, q)
        cs = item["continuationStart"]
        klc = r["kl"][cs:]
        print(f"  {item['name']:20} pos={item['positions']:4d} "
              f"KL mean={np.mean(r['kl']):.6g} p99={np.percentile(r['kl'], 99):.6g} "
              f"max={np.max(r['kl']):.6g} | continuation-only mean="
              f"{(np.mean(klc) if klc else float('nan')):.6g} | "
              f"top1={np.mean(r['top1']):.4f} top5={np.mean(r['top5']):.4f} "
              f"maxdlogit={np.max(r['maxdlogit']):.4g}")
        all_kl += r["kl"]; all_top1 += r["top1"]
        all_top5 += r["top5"]; all_dl += r["maxdlogit"]
    print("  " + summarize("KL (nats)", all_kl))
    print("  " + summarize("max |dlogit|", all_dl))
    print(f"  top-1 agreement {np.mean(all_top1):.6f}   "
          f"top-5 overlap {np.mean(all_top5):.6f}")
    print(f"  dumps byte-identical: {identical}")
    return {"kl": all_kl, "top1": all_top1, "top5": all_top5,
            "maxdlogit": all_dl, "identical": identical}


def sanity(root, label, teacher_label=None):
    """Prove the dump holds REAL logits before any number is read off it.

    This is not paranoia. The default fused head skips the logits write
    entirely and leaves only a greedy argmax in `lastGreedyToken`
    (`RealForwardRunner.swift`), so a harness built without
    `RuntimeConfiguration(forceLogitsHead: true)` dumps never-written memory
    and every KL number computed from it is confident nonsense. Two
    independent checks catch that:

      1. **Distribution shape.** log-sum-exp must be finite and within a few
         tens of nats of the max logit; a real distribution over 248k tokens
         has an entropy that keeps those close. Uninitialized memory gives
         NaN/inf, a zero page gives logsumexp = log(vocab) with max = 0.
      2. **Argmax agrees with the greedy token.** The dump teacher-forces a
         sequence whose continuation IS the trusted run's own greedy output,
         so within the continuation the argmax at position p must equal
         sequence[p+1]. Memory that was never written cannot satisfy that.
    """
    meta = load_meta(root, label)
    tokens_path = os.path.join(root, teacher_label or label, "tokens.json")
    seqs = {}
    if os.path.exists(tokens_path):
        with open(tokens_path) as f:
            seqs = {i["name"]: i for i in json.load(f)["items"]}
    print(f"\n=== dump sanity: {label} ===")
    ok = True
    for item in meta["items"]:
        a = logits(root, label, item)
        head = np.asarray(a[: min(64, a.shape[0])], dtype=np.float32)
        finite = np.isfinite(head).all()
        mx = head.max(axis=1)
        lse = mx + np.log(np.exp(head - mx[:, None]).sum(axis=1))
        gap = float(np.max(lse - mx))
        degenerate = bool(np.allclose(head, 0))
        agree = float("nan")
        info = seqs.get(item["name"])
        if info is not None:
            seq = info["sequence"]
            cs = item["continuationStart"]
            # Position p predicts sequence[p+1]; score the continuation only.
            lo, hi = max(cs - 1, 0), min(item["positions"] - 1, len(seq) - 1)
            if hi > lo:
                block = np.asarray(a[lo:hi], dtype=np.float32)
                pred = block.argmax(axis=1)
                want = np.asarray(seq[lo + 1:hi + 1], dtype=np.int64)
                agree = float((pred == want).mean())
        bad = (not finite) or degenerate or not (0.0 <= gap < 60.0)
        ok = ok and not bad
        print(f"  {item['name']:20} finite={finite} all-zero={degenerate} "
              f"logsumexp-max gap={gap:.4g} argmax==greedy={agree:.4f}"
              + ("   <-- SUSPECT" if bad else ""))
    print(f"  dump looks real: {ok}")
    return ok


def rollouts(root, ref, cand):
    def read(label, name):
        with open(os.path.join(root, label, name)) as f:
            return json.load(f)
    try:
        a, b = read(ref, "rollout.json"), read(cand, "rollout.json")
        ma = read(ref, "rollout_margins.json")
    except FileNotFoundError:
        return
    print(f"\n=== greedy rollouts: {ref} vs {cand} ===")
    for name in sorted(a):
        if name not in b:
            continue
        x, y = a[name], b[name]
        n = min(len(x), len(y))
        same = [i for i in range(n) if x[i] == y[i]]
        first = next((i for i in range(n) if x[i] != y[i]), None)
        exact = len(same) / n if n else float("nan")
        if first is None:
            print(f"  {name:20} {n:3d} tokens: IDENTICAL")
        else:
            margin = ma.get(name, [])
            m = margin[first] if first < len(margin) else float("nan")
            print(f"  {name:20} {n:3d} tokens: token-exact={exact:.3f} "
                  f"first divergence at {first} "
                  f"(reference top-2 logit margin there = {m:.4g})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--reference", default="trusted")
    ap.add_argument("--candidate", default="ours")
    ap.add_argument("--noise", default=None,
                    help="a second run of the REFERENCE install; its divergence "
                         "from the reference is the floor every other number is "
                         "read against")
    args = ap.parse_args()

    # Authenticity first: a KL number computed from never-written memory is
    # worse than no number at all.
    real = sanity(args.root, args.reference)
    real &= sanity(args.root, args.candidate, teacher_label=args.reference)
    if args.noise:
        real &= sanity(args.root, args.noise, teacher_label=args.reference)
    if not real:
        print("\nAT LEAST ONE DUMP IS SUSPECT — check forceLogitsHead before "
              "reading any number below.")

    floor = None
    if args.noise:
        floor = compare(args.root, args.reference, args.noise, "NOISE FLOOR")
    signal = compare(args.root, args.reference, args.candidate, "SIGNAL")
    rollouts(args.root, args.reference, args.candidate)

    print("\n=== verdict inputs ===")
    if floor is not None:
        f = np.asarray(floor["kl"]); s = np.asarray(signal["kl"])
        print(f"  noise-floor KL: mean={f.mean():.6g} max={f.max():.6g} "
              f"(byte-identical dumps: {floor['identical']})")
        print(f"  signal    KL: mean={s.mean():.6g} max={s.max():.6g}")
        if f.max() > 0:
            print(f"  signal/noise (mean) = {s.mean() / max(f.mean(), 1e-30):.4g}")
        else:
            print("  noise floor is exactly zero: decode is deterministic, so every "
                  "part of the signal is attributable to the quantizer.")
    else:
        print("  no noise-floor run supplied; the signal cannot be read against one.")


if __name__ == "__main__":
    main()
