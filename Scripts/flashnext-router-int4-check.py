#!/usr/bin/env python3
"""Does Flash-Next's INT4 router change expert selection?

The `qwen38flashnext.gturbo` install quantizes the MoE router
(`mlp.gate.weight`, [512, 2560], top-10) and the shared-expert gate
(`mlp.shared_expert_gate.weight`, [1, 2560]) at INT4 affine group-64 like
everything else. The independent `mlx-community/Qwen3.8-Flash-Next-4bit`
conversion keeps those 96 tensors at INT8 group-64. This script measures what
that costs, per layer, against the vendor BF16 source:

  1. reconstruction error (relative Frobenius, max-abs) — ours vs control;
  2. routing agreement on N probe vectors: exact top-10 set agreement, mean
     Jaccard, top-1 agreement, mean swapped experts, and the KL between the
     renormalized routing weights over the BF16-selected set;
  3. the shared-expert gate's `sigmoid(w . x)` drift;
  4. a norm-fold byte check: vendor vs control vs install bytes for the
     zero-centered RMSNorm gains (`FlashNextResident.zeroCenteredNormSuffixes`).

Our side is decoded from the install itself (resident index in
`model_weights.bin`; `ResidentIndexReader` layout), and cross-checked by
re-encoding the BF16 source with the transcribed `Int4AffineEncoder` — the
install bytes must reproduce bit-for-bit, which is what proves the decode.

    Scripts/flashnext-router-int4-check.py --cache <dir> \
        --install scratch/qwen38flashnext.gturbo [--layers 0,4,...] \
        [--probes 4096] [--activations hidden.f32]

Reads the install read-only; never starts a model process. Results:
docs/experiments/2026-09-10-flashnext-router-int4-check.md.
"""
import argparse, importlib.util, json, os, struct, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "weight_gate", os.path.join(HERE, "quantizer-weight-gate.py"))
wg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wg)

FAMILY = wg.FAMILIES["qwen38flashnext"]
O, C = FAMILY["prefixes"]
GROUP = wg.GROUP

NORM_SUFFIXES = [  # FlashNextResident.zeroCenteredNormSuffixes, verbatim
    ".self_attn.q_norm.weight", ".self_attn.k_norm.weight",
    ".self_attn.indexer.q_layernorm.weight", ".self_attn.indexer.k_layernorm.weight",
    ".attn_hyper_connection.hc_norm.weight", ".mlp_hyper_connection.hc_norm.weight",
    ".hyper_connection_mixer.hc_norm.weight",
    ".ple.norm_conv.weight", ".ple.norm_key.weight", ".ple.norm_query.weight",
]


# ----------------------------------------------------------- our install

def read_resident_index(path):
    """`ResidentIndexReader.load`: 24-byte header, 72-byte entries, names in
    a string table addressed by absolute file offset."""
    with open(path, "rb") as f:
        index_size, _, count = struct.unpack("<QQQ", f.read(24))
        f.seek(0)
        buf = f.read(index_size)
    entries = {}
    for i in range(count):
        p = 24 + i * 72
        name_off, name_len, dtype = struct.unpack_from("<IHB", buf, p)
        file_off, size = struct.unpack_from("<QQ", buf, p + 8)
        shape = struct.unpack_from("<IIII", buf, p + 24)
        s_off, s_size, b_off, b_size = struct.unpack_from("<QQQQ", buf, p + 40)
        entries[buf[name_off:name_off + name_len].decode()] = dict(
            dtype=dtype, offset=file_off, size=size,
            shape=tuple(d for d in shape if d), scale_offset=s_off,
            scale_size=s_size, bias_offset=b_off, bias_size=b_size)
    return entries


class Install:
    def __init__(self, root):
        self.root = root
        self.path = os.path.join(root, "model_weights.bin")
        self.entries = read_resident_index(self.path)
        self.manifest = json.load(open(os.path.join(root, "manifest.json")))

    def raw(self, off, size, dtype):
        with open(self.path, "rb") as f:
            f.seek(off)
            return np.frombuffer(f.read(size), dtype=dtype)

    def int4(self, name):
        """Decode an INT4 affine g64 resident tensor -> (float32 [R, C],
        packed bytes, scale bits, bias bits)."""
        e = self.entries[name]
        assert e["dtype"] == 0, f"{name} is dtype {e['dtype']}, not INT4 affine"
        rows, cols = e["shape"]
        packed = self.raw(e["offset"], e["size"], np.uint8).reshape(-1, GROUP // 2)
        s = self.raw(e["scale_offset"], e["scale_size"], np.uint16)
        b = self.raw(e["bias_offset"], e["bias_size"], np.uint16)
        assert packed.shape[0] == s.size == b.size == rows * cols // GROUP
        w = wg.dequant(wg.unpack(packed), s, b).reshape(rows, cols)
        return w, packed, s, b

    def bf16(self, name):
        e = self.entries[name]
        assert e["dtype"] == 1, f"{name} is dtype {e['dtype']}, not BF16"
        return self.raw(e["offset"], e["size"], np.uint16)


# --------------------------------------------------------------- control

def control_int8(mlx, cname, tag):
    """MLX affine INT8 g64: `.weight` U32 [R, C/4] little-endian (element i at
    byte i), `.scales`/`.biases` BF16 [R, C/64]."""
    w = mlx.tensor(cname + ".weight", "w_" + tag)
    s = mlx.tensor(cname + ".scales", "s_" + tag)
    b = mlx.tensor(cname + ".biases", "b_" + tag)
    rows = w.shape[0]
    cols = w.shape[1] * 4
    assert s.shape == (rows, cols // GROUP), (s.shape, rows, cols)
    q = w.view(np.uint8).reshape(-1, GROUP)
    return wg.dequant(q, s.reshape(-1), b.reshape(-1)).reshape(rows, cols)


# --------------------------------------------------------------- metrics

def recon(ref, approx):
    err = approx - ref
    return (float(np.sqrt((err ** 2).sum()) / np.sqrt((ref ** 2).sum())),
            float(np.abs(err).max()))


def topk_sets(logits, k):
    """Indices of the k largest logits per row, sorted descending."""
    idx = np.argpartition(-logits, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(logits, idx, axis=1), axis=1)
    return np.take_along_axis(idx, order, axis=1)


def routing_agreement(l_ref, l_alt, k):
    """Top-k of softmax(logits), renormalized (the model's router: softmax is
    monotone so top-k of the probs is top-k of the logits). KL is
    D_KL(ref || alt) between the renormalized weights over the REFERENCE's
    selected set, i.e. what weights the alternative router would hand the
    experts the BF16 router chose — well defined even when the sets differ."""
    s_ref, s_alt = topk_sets(l_ref, k), topk_sets(l_alt, k)
    n = l_ref.shape[0]
    inter = np.array([len(set(a) & set(b)) for a, b in zip(s_ref, s_alt)])
    exact = float((inter == k).mean())
    jaccard = float((inter / (2 * k - inter)).mean())
    top1 = float((s_ref[:, 0] == s_alt[:, 0]).mean())
    swapped = float((k - inter).mean())
    rl = np.take_along_axis(l_ref, s_ref, axis=1).astype(np.float64)
    al = np.take_along_axis(l_alt, s_ref, axis=1).astype(np.float64)
    lp = rl - rl.max(1, keepdims=True); lp -= np.log(np.exp(lp).sum(1, keepdims=True))
    lq = al - al.max(1, keepdims=True); lq -= np.log(np.exp(lq).sum(1, keepdims=True))
    kl = (np.exp(lp) * (lp - lq)).sum(1)
    # Margin the reference had at the top-k boundary, for context.
    sorted_ref = -np.sort(-l_ref, axis=1)
    margin = sorted_ref[:, k - 1] - sorted_ref[:, k]
    return dict(exact=exact, jaccard=jaccard, top1=top1, swapped=swapped,
                kl_mean=float(kl.mean()), kl_p99=float(np.percentile(kl, 99)),
                margin_median=float(np.median(margin)), n=n)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="/tmp/mference-flashnext-router")
    ap.add_argument("--install", default=os.path.join(HERE, "..", "scratch",
                                                      "qwen38flashnext.gturbo"))
    ap.add_argument("--layers", default="0,4,9,13,17,22,26,30,35,39,43,47",
                    help="comma-separated MoE layer indices")
    ap.add_argument("--probes", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--activations", default=None,
                    help="optional raw float32 [N, 2560] file of REAL router "
                         "inputs (hidden states); used as a second probe set")
    ap.add_argument("--norm-layers", default="3,19,40",
                    help="layers whose zero-centered norms get the byte check")
    ap.add_argument("--json", default=None, help="write all numbers here")
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)

    install = Install(args.install)
    arch = install.manifest["arch"]
    hidden, n_experts = arch["hiddenSize"], install.manifest["expertsPerLayer"]
    top_k = arch.get("topKExperts", 10)
    print(f"install {args.install}: modelID {install.manifest['modelID']}, "
          f"{install.manifest['numLayers']} layers, {n_experts} experts, "
          f"top-{top_k}, hidden {hidden}, router slot "
          f"{install.manifest['quant']['router']}")

    orig = wg.Repo(wg.resolve_url(FAMILY["orig"]), args.cache, "orig_flashnext")
    mlx = wg.Repo(wg.resolve_url(FAMILY["control"]), args.cache, "ctrl_flashnext")
    ctrl_cfg_path = os.path.join(args.cache, "ctrl.config.json")
    if not os.path.exists(ctrl_cfg_path):
        wg.curl(f"{mlx.base}/config.json", ctrl_cfg_path)
    ccfg = json.load(open(ctrl_cfg_path))
    q = ccfg["quantization"]
    int8 = sorted(k for k, v in q.items() if isinstance(v, dict) and v.get("bits") == 8)
    print(f"control: base {q['bits']}-bit g{q['group_size']}; {len(int8)} INT8 "
          f"overrides, all g64: "
          f"{all(q[k]['group_size'] == 64 for k in int8)}; suffixes "
          f"{sorted({k.split('.mlp.')[-1] for k in int8})}; "
          f"num_experts_per_tok {ccfg.get('num_experts_per_tok')}")
    assert ccfg.get("num_experts_per_tok", top_k) == top_k

    rng = np.random.default_rng(args.seed)
    probes = {"gaussian": rng.standard_normal((args.probes, hidden)).astype(np.float32)}
    if args.activations:
        acts = np.fromfile(args.activations, dtype=np.float32).reshape(-1, hidden)
        probes["real"] = acts[:args.probes]
        print(f"real activations: {probes['real'].shape[0]} vectors from "
              f"{args.activations}")
    else:
        print("no Flash-Next activation dump given (--activations); probe set "
              "(b) skipped — Gaussian probes only")

    layers = [int(x) for x in args.layers.split(",")]
    results = {"layers": [], "probes": args.probes, "seed": args.seed}
    for L in layers:
        row = {"layer": L}
        for short, tname in (("gate", "mlp.gate.weight"),
                             ("segate", "mlp.shared_expert_gate.weight")):
            oname = f"{O}layers.{L}.{tname}"
            cname = wg.control_name(oname, FAMILY["prefixes"])
            tag = f"L{L}_{short}"
            ref = wg.bf16_to_f32(orig.tensor(oname, "o_" + tag)).astype(np.float32)
            ours, packed, s, b = install.int4(oname)
            ctrl = control_int8(mlx, cname, tag)
            assert ref.shape == ours.shape == ctrl.shape, (ref.shape, ours.shape, ctrl.shape)
            # Decode proof: the install's bytes are exactly what the transcribed
            # encoder produces from this BF16 source.
            ep, es, eb, _ = wg.encode_ours(ref.reshape(-1, GROUP), 15)
            bitid = bool((ep == packed).all() and (es == s).all() and (eb == b).all())
            ro, mo = recon(ref, ours)
            rc, mc = recon(ref, ctrl)
            row[short] = dict(bitid=bitid, rel_ours=ro, rel_ctrl=rc, max_ours=mo,
                              max_ctrl=mc, shape=list(ref.shape))
            if short == "gate":
                for pname, X in probes.items():
                    l_ref = X @ ref.T
                    row[short][pname] = dict(
                        ours=routing_agreement(l_ref, X @ ours.T, top_k),
                        ctrl=routing_agreement(l_ref, X @ ctrl.T, top_k))
            else:
                for pname, X in probes.items():
                    g_ref = sigmoid(X @ ref[0])
                    row[short][pname] = dict(
                        ours=float(np.abs(sigmoid(X @ ours[0]) - g_ref).mean()),
                        ctrl=float(np.abs(sigmoid(X @ ctrl[0]) - g_ref).mean()),
                        ours_max=float(np.abs(sigmoid(X @ ours[0]) - g_ref).max()),
                        ctrl_max=float(np.abs(sigmoid(X @ ctrl[0]) - g_ref).max()))
        results["layers"].append(row)
        g = row["gate"]["gaussian"]
        print(f"layer {L:2d}  router rel {row['gate']['rel_ours']:.4f}/"
              f"{row['gate']['rel_ctrl']:.4f}  bitid {row['gate']['bitid']}  "
              f"top10 exact {g['ours']['exact']:.3f}/{g['ctrl']['exact']:.3f}  "
              f"top1 {g['ours']['top1']:.3f}/{g['ctrl']['top1']:.3f}  "
              f"swapped {g['ours']['swapped']:.2f}/{g['ctrl']['swapped']:.2f}  "
              f"KL {g['ours']['kl_mean']:.2e}/{g['ctrl']['kl_mean']:.2e}")

    # ---------------------------------------------------------- summary
    def agg(short, key, side, pname="gaussian"):
        return np.array([r[short][pname][side][key] for r in results["layers"]])
    print("\n=== aggregate over", len(layers), "layers (Gaussian probes) ===")
    for short in ("gate", "segate"):
        ro = np.array([r[short]["rel_ours"] for r in results["layers"]])
        rc = np.array([r[short]["rel_ctrl"] for r in results["layers"]])
        mo = np.array([r[short]["max_ours"] for r in results["layers"]])
        mc = np.array([r[short]["max_ctrl"] for r in results["layers"]])
        bit = sum(r[short]["bitid"] for r in results["layers"])
        print(f"{short:7} install bytes == re-encoded BF16: {bit}/{len(layers)}; "
              f"rel Frobenius ours {ro.mean():.5f} ctrl {rc.mean():.5f} "
              f"ratio {ro.mean()/rc.mean():.2f}; max-abs ours {mo.mean():.5f} "
              f"ctrl {mc.mean():.5f} ratio {mo.mean()/mc.mean():.2f}")
    for pname in probes:
        print(f"-- probes: {pname}")
        for key in ("exact", "jaccard", "top1", "swapped", "kl_mean", "kl_p99"):
            a, c = agg("gate", key, "ours", pname), agg("gate", key, "ctrl", pname)
            print(f"  router {key:8} ours {a.mean():.4f} [{a.min():.4f}, {a.max():.4f}]"
                  f"   ctrl {c.mean():.4f} [{c.min():.4f}, {c.max():.4f}]")
        so = np.array([r["segate"][pname]["ours"] for r in results["layers"]])
        sc = np.array([r["segate"][pname]["ctrl"] for r in results["layers"]])
        print(f"  segate mean |d sigmoid| ours {so.mean():.5f} ctrl {sc.mean():.5f}")

    # ------------------------------------------------- norm-fold byte check
    print("\n=== zero-centered norm byte check (vendor w / control / install) ===")
    norm_rows = []
    names = ["model.language_model.hyper_connection_mixer.hc_norm.weight",
             "model.language_model.layers.1.ple.norm_key.weight",
             "model.language_model.layers.1.ple.norm_query.weight",
             "model.language_model.layers.1.ple.norm_conv.weight"]
    for L in [int(x) for x in args.norm_layers.split(",")]:
        for suf in NORM_SUFFIXES[:6]:
            names.append(f"{O}layers.{L}{suf}")
    for oname in names:
        if oname not in install.entries or oname not in orig.weight_map:
            continue
        cname = wg.control_name(oname, FAMILY["prefixes"]) + ".weight"
        if cname not in mlx.weight_map:
            print(f"  {oname}: control has no such tensor")
            continue
        tag = "n_" + oname.replace(".", "_")
        v = orig.tensor(oname, tag).reshape(-1)
        c = mlx.tensor(cname, "c" + tag).reshape(-1)
        i = install.bf16(oname)
        v1 = wg.bf16_bits(wg.bf16_to_f32(v) + np.float32(1))
        norm_rows.append(dict(
            name=oname, n=int(v.size),
            install_eq_vendor=bool(np.array_equal(i, v)),
            control_eq_vendor=bool(np.array_equal(c, v)),
            control_eq_vendor_plus1=bool(np.array_equal(c, v1)),
            vendor_absmax=float(np.abs(wg.bf16_to_f32(v)).max()),
            vendor_mean=float(wg.bf16_to_f32(v).mean())))
        r = norm_rows[-1]
        print(f"  {oname[21:]:60} n={r['n']:5d} install==vendor {r['install_eq_vendor']!s:5} "
              f"control==vendor {r['control_eq_vendor']!s:5} control==bf16(1+w) "
              f"{r['control_eq_vendor_plus1']!s:5} |w|max {r['vendor_absmax']:.3f}")
    print(f"  {sum(r['install_eq_vendor'] for r in norm_rows)}/{len(norm_rows)} install "
          f"bytes == vendor bare w; {sum(r['control_eq_vendor'] for r in norm_rows)}/"
          f"{len(norm_rows)} control == vendor bare w; "
          f"{sum(r['control_eq_vendor_plus1'] for r in norm_rows)}/{len(norm_rows)} "
          f"control == bf16(1 + w)")
    results["norms"] = norm_rows
    if args.json:
        json.dump(results, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
