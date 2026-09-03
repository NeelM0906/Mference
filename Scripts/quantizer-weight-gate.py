#!/usr/bin/env python3
"""W2.1b weight-level gate: our INT4 affine group-64 encoder vs the
mlx-community conversion of the SAME BF16 source rows.

Both sides quantize `Qwen/Qwen3.6-35B-A3B` rev 995ad96e. Group-64 quantization
never straddles a row (every quantizable last dimension is a multiple of 64),
so a *row slice* is self-contained: the BF16 rows fully determine the packed
nibbles, scales and biases for those same rows. That is what makes this gate
cheap — ~11 MB of HTTP range requests instead of a 72 GB download — and it is
why it can be re-run for any future family before committing to an install.

    Scripts/quantizer-weight-gate.py --cache <dir>

The encoder here is a transcription of
`Sources/MferenceRepack/Core/Quantization/Int4AffineEncoder.swift:encodeGroup`;
`Int4AffineEncoderConventionTests` locks the Swift original to the same
properties this script relies on. See docs/QUANTIZER_QUALITY.md.
"""
import argparse, json, os, struct, subprocess, time
import numpy as np

ORIG = ("https://huggingface.co/Qwen/Qwen3.6-35B-A3B/resolve/"
        "995ad96eacd98c81ed38be0c5b274b04031597b0")
MLX = ("https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit/resolve/"
       "38740b847e4cb78f352aba30aa41c76e08e6eb46")
GROUP = 64
EPS = np.float32(1e-7)
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "U32": 4, "I32": 4, "U8": 1, "I8": 1}


# --------------------------------------------------------------- fetching

def curl(url, out, rng=None, retries=6):
    for attempt in range(retries):
        cmd = ["curl", "-sSL", "--fail", "-o", out]
        if rng:
            cmd += ["-r", rng]
        r = subprocess.run(cmd + [url], capture_output=True)
        if r.returncode == 0:
            return
        time.sleep(2 ** attempt)
    raise SystemExit(f"curl failed: {url} {rng}")


class Repo:
    def __init__(self, base, cache, tag):
        self.base, self.cache, self.tag = base, cache, tag
        self._headers = {}
        idx = os.path.join(cache, f"{tag}.index.json")
        if not os.path.exists(idx):
            curl(f"{base}/model.safetensors.index.json", idx)
        self.weight_map = json.load(open(idx))["weight_map"]

    def header(self, shard):
        if shard in self._headers:
            return self._headers[shard]
        path = os.path.join(self.cache, f"{self.tag}.{shard}.hdr.json")
        if not os.path.exists(path):
            lenp = path + ".len"
            curl(f"{self.base}/{shard}", lenp, rng="0-7")
            n = struct.unpack("<Q", open(lenp, "rb").read(8))[0]
            curl(f"{self.base}/{shard}", path, rng=f"8-{8 + n - 1}")
            open(path + ".n", "w").write(str(n))
        n = int(open(path + ".n").read())
        self._headers[shard] = (json.load(open(path)), 8 + n)
        return self._headers[shard]

    def rows(self, name, row_start, row_count, tag):
        """Rows of `name`, counting over the leading dimensions flattened
        against the last one: for [E, R, C] row `e*R + r`."""
        shard = self.weight_map[name]
        head, data_base = self.header(shard)
        info = head[name]
        esz = DTYPE_BYTES[info["dtype"]]
        last = info["shape"][-1]
        row_bytes = last * esz
        start = data_base + info["data_offsets"][0] + row_start * row_bytes
        out = os.path.join(self.cache, f"{tag}.bin")
        if not os.path.exists(out):
            curl(f"{self.base}/{shard}", out,
                 rng=f"{start}-{start + row_count * row_bytes - 1}")
        return out, info["shape"], info["dtype"], last


# ------------------------------------------------------------- quantizers

def bf16_bits(x):
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    return ((bits + (np.uint32(0x7FFF) + lsb)) >> np.uint32(16)).astype(np.uint16)


def bf16_to_f32(bits):
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32)
            << np.uint32(16)).view(np.float32)


def _pack(q, n_bins):
    """Nibble-pack at INT4; INT8 is already one byte per weight."""
    if n_bins == 15:
        return (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    return q.astype(np.uint8)


def encode_ours(vals, n_bins=15):
    """Int4AffineEncoder.encodeGroup, and — at n_bins=255 — Int8AffineEncoder's
    identical grid: plain min/max affine, scale and bias rounded through BF16
    *before* index quantization, no zero-point snap.

    One function serves both widths on purpose. The two Swift encoders are
    deliberately the same convention (`Int8AffineEncoderConventionTests` locks
    that), so a second transcription here could only drift."""
    wmin, wmax = vals.min(axis=1), vals.max(axis=1)
    const = wmax == wmin
    s_bits = bf16_bits(np.where(const, np.float32(1), (wmax - wmin) / np.float32(n_bins)))
    b_bits = bf16_bits(wmin)
    s, b = bf16_to_f32(s_bits), bf16_to_f32(b_bits)
    inv = np.where(s == 0, np.float32(0), np.float32(1) / s)
    t = ((vals - b[:, None]) * inv[:, None]).astype(np.float32)
    # Swift's Float.rounded() is round-half-away-from-zero.
    q = np.clip(np.trunc(t + np.copysign(np.float32(0.5), t)), 0, n_bins).astype(np.uint8)
    return _pack(q, n_bins), s_bits, b_bits, q


def encode_mlx_model(vals, n_bins=15):
    """Model of MLX's affine_quantize: anchor the grid on the larger-magnitude
    endpoint (so the scale may be negative) and rescale so an integer bin lands
    exactly on 0.0. Used only to ATTRIBUTE the difference, never to gate."""
    wmin, wmax = vals.min(axis=1), vals.max(axis=1)
    raw = np.maximum((wmax - wmin) / np.float32(n_bins), EPS).astype(np.float32)
    mask = np.abs(wmax) > np.abs(wmin)
    scale = np.where(mask, -raw, raw).astype(np.float32)
    edge = np.where(mask, wmax, wmin).astype(np.float32)
    rq = np.rint((-edge / scale).astype(np.float32))
    do = (rq > 0) & (rq < n_bins)
    scale = np.where(do, -edge / np.where(do, rq, np.float32(1)), scale).astype(np.float32)
    s_bits, b_bits = bf16_bits(scale), bf16_bits(edge)
    s, b = bf16_to_f32(s_bits), bf16_to_f32(b_bits)
    inv = np.where(s == 0, np.float32(0), np.float32(1) / s)
    q = np.clip(np.rint(((vals - b[:, None]) * inv[:, None]).astype(np.float32)),
                0, n_bins).astype(np.uint8)
    return _pack(q, n_bins), s_bits, b_bits, q


def unpack(packed, n_bins=15):
    if n_bins != 15:
        return packed
    out = np.empty((packed.shape[0], GROUP), dtype=np.uint8)
    out[:, 0::2] = packed & 0x0F
    out[:, 1::2] = packed >> 4
    return out


def degenerate(vals, n_bins=15):
    return ((vals.max(axis=1) - vals.min(axis=1)) / np.float32(n_bins)) <= EPS


# ------------------------------------------------------------ sample plan

def sample_plan():
    O, M = "model.language_model.", "language_model.model."
    plan = []

    def add(tag, orig, mlx, ors, mrs, rc):
        plan.append(dict(tag=tag, orig=orig, mlx=mlx,
                         orig_row=ors, mlx_row=mrs, rows=rc))

    add("embed", O + "embed_tokens.weight", M + "embed_tokens", 0, 0, 64)
    add("embed_mid", O + "embed_tokens.weight", M + "embed_tokens", 120000, 120000, 64)
    add("lmhead", "lm_head.weight", "language_model.lm_head", 0, 0, 64)
    add("lmhead_tail", "lm_head.weight", "language_model.lm_head", 248000, 248000, 64)
    for l in (0, 5, 17, 29, 38):                       # linear-attention layers
        for t in ("in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z", "out_proj"):
            add(f"l{l}_{t}", f"{O}layers.{l}.linear_attn.{t}.weight",
                f"{M}layers.{l}.linear_attn.{t}", 0, 0, 8)
    for l in (3, 7, 19, 27, 39):                       # full-attention layers
        for t in ("q_proj", "k_proj", "v_proj", "o_proj"):
            add(f"l{l}_{t}", f"{O}layers.{l}.self_attn.{t}.weight",
                f"{M}layers.{l}.self_attn.{t}", 0, 0, 8)
    for l in (0, 7, 19, 27, 39):
        for t in ("gate_proj", "up_proj", "down_proj"):
            add(f"l{l}_se_{t}", f"{O}layers.{l}.mlp.shared_expert.{t}.weight",
                f"{M}layers.{l}.mlp.shared_expert.{t}", 0, 0, 32)
        # The two INT8 tensors of the control's mixture. The router is
        # [numExperts, hidden] = [256, 2048]; the shared-expert gate is the
        # single row [1, 2048], which is also the narrowest shape the streaming
        # path ever sees.
        add(f"l{l}_router", f"{O}layers.{l}.mlp.gate.weight",
            f"{M}layers.{l}.mlp.gate", 0, 0, 32)
        add(f"l{l}_segate", f"{O}layers.{l}.mlp.shared_expert_gate.weight",
            f"{M}layers.{l}.mlp.shared_expert_gate", 0, 0, 1)
        # Routed experts. gate_up_proj is [E, 2*I, H] with the gate half first,
        # so expert e's gate rows start at e*2I and its up rows at e*2I + I.
        for e in (0, 5, 137, 255):
            add(f"l{l}_e{e}_gate", f"{O}layers.{l}.mlp.experts.gate_up_proj",
                f"{M}layers.{l}.mlp.switch_mlp.gate_proj", e * 1024, e * 512, 16)
            add(f"l{l}_e{e}_up", f"{O}layers.{l}.mlp.experts.gate_up_proj",
                f"{M}layers.{l}.mlp.switch_mlp.up_proj", e * 1024 + 512, e * 512, 16)
            add(f"l{l}_e{e}_down", f"{O}layers.{l}.mlp.experts.down_proj",
                f"{M}layers.{l}.mlp.switch_mlp.down_proj", e * 2048, e * 2048, 16)
    return plan


# ------------------------------------------------------------- comparison

def compare(item, opath, mpath, mlx, n_bins):
    """One sampled tensor: our encoder vs the control's stored bytes, both
    measured against the same BF16 source rows.

    Identical methodology at either width (§3-§5 of docs/QUANTIZER_QUALITY.md).
    Neither grid is 'the' right answer, so the only meaningful score is
    reconstruction error against the BF16 source; bit-identity is reported but
    is not expected and is not a gate."""
    spath, _, _, _ = mlx.rows(item["mlx"] + ".scales", item["mlx_row"],
                              item["rows"], "s_" + item["tag"])
    bpath, _, _, _ = mlx.rows(item["mlx"] + ".biases", item["mlx_row"],
                              item["rows"], "b_" + item["tag"])
    src = bf16_to_f32(np.fromfile(opath, dtype=np.uint16)).reshape(-1, GROUP)
    keep = ~degenerate(src, n_bins)
    op, os_, ob, qo = encode_ours(src, n_bins)
    width = GROUP if n_bins == 255 else GROUP // 2
    tp = np.fromfile(mpath, dtype=np.uint8).reshape(-1, width)
    ts = np.fromfile(spath, dtype=np.uint16)
    tb = np.fromfile(bpath, dtype=np.uint16)
    qt = unpack(tp, n_bins)
    mp, _, _, _ = encode_mlx_model(src, n_bins)

    do = bf16_to_f32(os_)[:, None] * qo.astype(np.float32) + bf16_to_f32(ob)[:, None]
    dt = bf16_to_f32(ts)[:, None] * qt.astype(np.float32) + bf16_to_f32(tb)[:, None]
    G, EO, ET = src[keep], (do - src)[keep], (dt - src)[keep]
    den = np.sqrt((G ** 2).sum())
    return dict(
        tag=item["tag"],
        rel_ours=float(np.sqrt((EO ** 2).sum()) / den),
        rel_mlx=float(np.sqrt((ET ** 2).sum()) / den),
        max_ours=float(np.abs(EO).max()), max_mlx=float(np.abs(ET).max()),
        bitid=bool((op == tp).all() and (os_ == ts).all() and (ob == tb).all()),
        nibble_diff=float((qo[keep] != qt[keep]).mean()),
        attributed=float((mp[keep] == tp[keep]).mean()),
        degenerate=float(np.mean(~keep)))


def summarize(rows, label):
    if not rows:
        return
    print(f"\n{'tensor':22} {'bitid':>5} {'codeDiff':>8} {'relOurs':>8} {'relMLX':>8} "
          f"{'ratio':>6} {'maxOurs':>9} {'maxMLX':>9} {'attrib':>7}")
    for r in rows:
        print(f"{r['tag']:22} {str(r['bitid']):>5} {r['nibble_diff']:8.4f} "
              f"{r['rel_ours']:8.5f} {r['rel_mlx']:8.5f} "
              f"{r['rel_ours']/r['rel_mlx']:6.4f} {r['max_ours']:9.6f} "
              f"{r['max_mlx']:9.6f} {r['attributed']:7.4f}")
    ro = np.array([r["rel_ours"] for r in rows])
    rm = np.array([r["rel_mlx"] for r in rows])
    mo = np.array([r["max_ours"] for r in rows])
    mm = np.array([r["max_mlx"] for r in rows])
    print(f"\n{len(rows)} {label} tensors compared (degenerate groups excluded)")
    print(f"  bit-identical to the control: {sum(r['bitid'] for r in rows)}/{len(rows)}")
    print(f"  relative Frobenius error  ours mean {ro.mean():.6f} median {np.median(ro):.6f}")
    print(f"  relative Frobenius error   mlx mean {rm.mean():.6f} median {np.median(rm):.6f}")
    print(f"  ours strictly better on {(ro < rm).sum()}/{len(rows)}; "
          f"worst ratio {(ro/rm).max():.4f} on "
          f"{rows[int(np.argmax(ro/rm))]['tag']}")
    print(f"  max-abs error  ours better on {(mo < mm).sum()}/{len(rows)}; "
          f"mean ratio {np.mean(mo/mm):.4f} worst {np.max(mo/mm):.4f}")
    print(f"  attribution: the MLX-convention model reproduces "
          f"{np.mean([r['attributed'] for r in rows]):.4f} of the control's "
          f"packed bytes in non-degenerate groups")


# ------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/mference-quant-gate")
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)
    orig, mlx = Repo(ORIG, args.cache, "orig"), Repo(MLX, args.cache, "mlx")

    rows, rows8, skipped = [], [], []
    for item in sample_plan():
        opath, _, _, cols = orig.rows(item["orig"], item["orig_row"], item["rows"],
                                      "o_" + item["tag"])
        wname = item["mlx"] + ".weight"
        mpath, mshape, _, mlast = mlx.rows(wname, item["mlx_row"], item["rows"],
                                           "w_" + item["tag"])
        bits = (mlast * 4 * 8) // cols
        if bits == 8:
            rows8.append(compare(item, opath, mpath, mlx, n_bins=255))
            continue
        if bits != 4:
            skipped.append((item["tag"], bits, item["mlx"]))
            continue
        rows.append(compare(item, opath, mpath, mlx, n_bins=15))

    summarize(rows, "INT4")
    summarize(rows8, "INT8 (the control's per-tensor overrides: routers and "
                     "shared-expert gates)")
    if skipped:
        print("\n  neither 4- nor 8-bit on the control side (no bitwise "
              "comparison is possible):")
        for tag, bits, name in skipped:
            print(f"    {tag:22} control is {bits}-bit  ({name})")


if __name__ == "__main__":
    main()
