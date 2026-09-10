#!/usr/bin/env python3
"""W2.1b weight-level gate: our INT4 affine group-64 encoder vs the
mlx-community conversion of the SAME BF16 source rows.

Both sides quantize one vendor BF16 checkpoint. Group quantization never
straddles a row (every quantizable last dimension is a multiple of the group),
so a *row slice* is self-contained: the BF16 rows fully determine the packed
nibbles, scales and biases for those same rows. That is what makes this gate
cheap — a few MB of HTTP range requests instead of a 72 GB download — and it is
why it can be re-run for any future family before committing to an install.

    # The W2.1b result in docs/QUANTIZER_QUALITY.md (Qwen 3.6, default):
    Scripts/quantizer-weight-gate.py --cache <dir>

    # Any other family, any tensors: sample every tensor whose vendor name
    # matches PATTERN (a regex), full rows unless --rows caps them.
    Scripts/quantizer-weight-gate.py --cache <dir> --family qwen38flashnext \
        --tensors 'layers\\.(3|19|40)\\.mlp\\.(gate|shared_expert_gate)\\.weight'

    # Or name the repos yourself (REPO[@REV]; REV defaults to main):
    Scripts/quantizer-weight-gate.py --orig Qwen/Qwen3.8-Flash-Next@de4b8e4d \
        --control mlx-community/Qwen3.8-Flash-Next-4bit --tensors '...'

The encoder here is a transcription of
`Sources/MferenceRepack/Core/Quantization/Int4AffineEncoder.swift:encodeGroup`;
`Int4AffineEncoderConventionTests` locks the Swift original to the same
properties this script relies on. See docs/QUANTIZER_QUALITY.md.

The module is importable (see Scripts/flashnext-router-int4-check.py): `Repo`
does the safetensors-header-parse + byte-range fetch, and the `bf16_*`,
`encode_ours`, `unpack`, `dequant` helpers are the decoders.
"""
import argparse, json, os, re, struct, subprocess, sys, time
import numpy as np

# Per-family repo pins. `orig` is the vendor BF16 upload, `control` the
# independent conversion whose packed bytes we compare against. Names on the
# two sides differ by a prefix (`prefixes`) and the control drops `.weight`
# in favour of `.weight/.scales/.biases` triples.
FAMILIES = {
    "qwen36": dict(
        orig="Qwen/Qwen3.6-35B-A3B@995ad96eacd98c81ed38be0c5b274b04031597b0",
        control="mlx-community/Qwen3.6-35B-A3B-4bit@38740b847e4cb78f352aba30aa41c76e08e6eb46",
        prefixes=("model.language_model.", "language_model.model.")),
    "qwen38flashnext": dict(
        # Vendor BF16 at the revision the install's manifest was built from;
        # the control landed 2026-09-02 (INT4 g32 base, INT8 g64 on the 96
        # `mlp.gate` / `mlp.shared_expert_gate` tensors).
        orig="Qwen/Qwen3.8-Flash-Next@de4b8e4d",
        control="mlx-community/Qwen3.8-Flash-Next-4bit@main",
        prefixes=("model.language_model.", "language_model.model.")),
}

def resolve_url(spec):
    """`owner/repo[@rev]` -> `https://huggingface.co/owner/repo/resolve/rev`."""
    if spec.startswith("http"):
        return spec.rstrip("/")
    repo, _, rev = spec.partition("@")
    return f"https://huggingface.co/{repo}/resolve/{rev or 'main'}"

# Kept for readers of docs/QUANTIZER_QUALITY.md: the Qwen 3.6 pins by name.
ORIG = resolve_url(FAMILIES["qwen36"]["orig"])
MLX = resolve_url(FAMILIES["qwen36"]["control"])
GROUP = 64
EPS = np.float32(1e-7)
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "U32": 4, "I32": 4, "U8": 1, "I8": 1}


# --------------------------------------------------------------- fetching

def curl(url, out, rng=None, retries=8):
    """One HTTP range request through the `resolve` endpoint, following the
    CDN redirect. The CDN rate-limits bulk paths (429) and occasionally 5xxs;
    both back off exponentially. Anything else fails fast."""
    for attempt in range(retries):
        cmd = ["curl", "-sSL", "-o", out, "-w", "%{http_code}"]
        if rng:
            cmd += ["-r", rng]
        r = subprocess.run(cmd + [url], capture_output=True, text=True)
        code = r.stdout.strip()[-3:]
        if r.returncode == 0 and code in ("200", "206"):
            return
        transient = code in ("429", "500", "502", "503", "504") or r.returncode != 0
        if not transient:
            raise SystemExit(f"curl failed: HTTP {code} for {url} {rng}")
        wait = min(2 ** attempt, 60)
        print(f"  [curl] HTTP {code or r.returncode} on {os.path.basename(url)} "
              f"{rng or ''}; retry in {wait}s", file=sys.stderr)
        time.sleep(wait)
    raise SystemExit(f"curl failed after {retries} attempts: {url} {rng}")


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

    def info(self, name):
        shard = self.weight_map[name]
        head, _ = self.header(shard)
        return head[name]

    def rows(self, name, row_start, row_count, tag):
        """Rows of `name`, counting over the leading dimensions flattened
        against the last one: for [E, R, C] row `e*R + r`. Returns the cache
        path of the raw bytes plus (shape, dtype, last-dim)."""
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

    def tensor(self, name, tag):
        """The whole tensor as a numpy array in its stored dtype."""
        info = self.info(name)
        shape = info["shape"]
        n_rows = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
        path, _, dtype, _ = self.rows(name, 0, n_rows, tag)
        np_dtype = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32,
                    "U32": np.uint32, "I32": np.int32, "U8": np.uint8,
                    "I8": np.int8}[dtype]
        return np.fromfile(path, dtype=np_dtype).reshape(shape)


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
    *before* index quantization, no zero-point snap. `vals` is [groups, G].

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


def unpack(packed, n_bins=15, group=GROUP):
    """Packed bytes [groups, G/2] (INT4, low nibble = even index) or
    [groups, G] (INT8) -> indices [groups, G]. Both our gturbo layout and
    MLX's little-endian uint32 packing put element i of a group at bit
    offset i*bits, so one unpacker serves both sides."""
    if n_bins != 15:
        return packed
    out = np.empty((packed.shape[0], group), dtype=np.uint8)
    out[:, 0::2] = packed & 0x0F
    out[:, 1::2] = packed >> 4
    return out


def dequant(q, s_bits, b_bits):
    """Affine decode `s*q + b` for [groups, G] indices and per-group BF16
    scale/bias bit patterns; returns float32 [groups, G]."""
    return (bf16_to_f32(s_bits)[:, None] * q.astype(np.float32)
            + bf16_to_f32(b_bits)[:, None])


def degenerate(vals, n_bins=15):
    return ((vals.max(axis=1) - vals.min(axis=1)) / np.float32(n_bins)) <= EPS


# ------------------------------------------------------------ sample plan

def control_name(orig_name, prefixes):
    """Vendor tensor name -> control module name (without .weight/.scales)."""
    o, c = prefixes
    base = orig_name[:-len(".weight")] if orig_name.endswith(".weight") else orig_name
    if base == "lm_head":
        return "language_model.lm_head"
    if base.startswith(o):
        return c + base[len(o):]
    return base


def sample_plan():
    """The W2.1b Qwen 3.6 plan: the tensors and rows behind §4/§5 of
    docs/QUANTIZER_QUALITY.md. Unchanged so those numbers stay reproducible."""
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


def pattern_plan(orig, mlx, pattern, prefixes, max_rows):
    """Every vendor tensor matching `pattern` that the control also carries
    as a quantized `.weight/.scales/.biases` triple; full rows, or the first
    `max_rows` when given. Tensors the control keeps unquantized (norms) are
    listed and skipped: there is nothing to compare a quantizer against."""
    rx = re.compile(pattern)
    plan, unquantized = [], []
    for name in sorted(orig.weight_map):
        if not rx.search(name):
            continue
        cname = control_name(name, prefixes)
        if cname + ".scales" not in mlx.weight_map:
            unquantized.append((name, cname))
            continue
        shape = orig.info(name)["shape"]
        n_rows = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
        rows = min(n_rows, max_rows) if max_rows else n_rows
        tag = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
        plan.append(dict(tag=tag, orig=name, mlx=cname,
                         orig_row=0, mlx_row=0, rows=rows))
    return plan, unquantized


# ------------------------------------------------------------- comparison

def compare(item, opath, mpath, mlx, n_bins, group=GROUP):
    """One sampled tensor: our encoder vs the control's stored bytes, both
    measured against the same BF16 source rows.

    Identical methodology at either width (§3-§5 of docs/QUANTIZER_QUALITY.md).
    Neither grid is 'the' right answer, so the only meaningful score is
    reconstruction error against the BF16 source; bit-identity is reported but
    is not expected and is not a gate. `group` is the CONTROL's group size —
    ours is always 64 — so when they differ the two error columns are still
    against the same source but the byte columns are meaningless and are
    reported as such."""
    spath, _, _, _ = mlx.rows(item["mlx"] + ".scales", item["mlx_row"],
                              item["rows"], "s_" + item["tag"])
    bpath, _, _, _ = mlx.rows(item["mlx"] + ".biases", item["mlx_row"],
                              item["rows"], "b_" + item["tag"])
    flat = bf16_to_f32(np.fromfile(opath, dtype=np.uint16))
    src = flat.reshape(-1, GROUP)
    keep = ~degenerate(src, n_bins)
    op, os_, ob, qo = encode_ours(src, n_bins)
    width = group if n_bins == 255 else group // 2
    tp = np.fromfile(mpath, dtype=np.uint8).reshape(-1, width)
    ts = np.fromfile(spath, dtype=np.uint16)
    tb = np.fromfile(bpath, dtype=np.uint16)
    qt = unpack(tp, n_bins, group)
    mp, _, _, _ = encode_mlx_model(src.reshape(-1, group) if group != GROUP else src, n_bins)

    do = dequant(qo, os_, ob).reshape(-1)
    dt = dequant(qt, ts, tb).reshape(-1)
    keep_flat = np.repeat(keep, GROUP)
    G, EO, ET = flat[keep_flat], (do - flat)[keep_flat], (dt - flat)[keep_flat]
    den = np.sqrt((G ** 2).sum())
    same_group = group == GROUP
    return dict(
        tag=item["tag"],
        rel_ours=float(np.sqrt((EO ** 2).sum()) / den),
        rel_mlx=float(np.sqrt((ET ** 2).sum()) / den),
        max_ours=float(np.abs(EO).max()), max_mlx=float(np.abs(ET).max()),
        bitid=bool(same_group and (op == tp).all() and (os_ == ts).all()
                   and (ob == tb).all()),
        nibble_diff=(float((qo[keep] != qt[keep]).mean()) if same_group
                     else float("nan")),
        attributed=(float((mp[keep] == tp[keep]).mean()) if same_group
                    else float((mp == tp).mean())),
        degenerate=float(np.mean(~keep)),
        control_group=group)


def summarize(rows, label):
    if not rows:
        return
    print(f"\n{'tensor':22} {'bitid':>5} {'codeDiff':>8} {'relOurs':>8} {'relMLX':>8} "
          f"{'ratio':>6} {'maxOurs':>9} {'maxMLX':>9} {'attrib':>7}")
    for r in rows:
        print(f"{r['tag'][-22:]:22} {str(r['bitid']):>5} {r['nibble_diff']:8.4f} "
              f"{r['rel_ours']:8.5f} {r['rel_mlx']:8.5f} "
              f"{r['rel_ours']/r['rel_mlx']:6.4f} {r['max_ours']:9.6f} "
              f"{r['max_mlx']:9.6f} {r['attributed']:7.4f}")
    ro = np.array([r["rel_ours"] for r in rows])
    rm = np.array([r["rel_mlx"] for r in rows])
    mo = np.array([r["max_ours"] for r in rows])
    mm = np.array([r["max_mlx"] for r in rows])
    print(f"\n{len(rows)} {label} tensors compared (degenerate groups excluded)")
    groups = sorted({r["control_group"] for r in rows})
    if groups != [GROUP]:
        print(f"  control group size(s) {groups} vs ours {GROUP}: byte columns "
              f"are not comparable, error columns are")
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

def open_repos(args):
    fam = FAMILIES[args.family]
    orig_spec = args.orig or fam["orig"]
    ctrl_spec = args.control or fam["control"]
    cache = args.cache
    os.makedirs(cache, exist_ok=True)
    # Cache tags carry the repo so two families never share a header file.
    def tag(kind, spec):
        return kind + "_" + re.sub(r"[^A-Za-z0-9]+", "_", spec)[:80]
    orig = Repo(resolve_url(orig_spec), cache, "orig" if args.family == "qwen36"
                and not args.orig else tag("orig", orig_spec))
    mlx = Repo(resolve_url(ctrl_spec), cache, "mlx" if args.family == "qwen36"
               and not args.control else tag("ctrl", ctrl_spec))
    return orig, mlx, fam


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="/tmp/mference-quant-gate")
    ap.add_argument("--family", choices=sorted(FAMILIES), default="qwen36",
                    help="repo pins and name mapping (default: qwen36, the "
                         "W2.1b result)")
    ap.add_argument("--orig", help="vendor BF16 repo as owner/repo[@rev]; "
                                   "overrides the family pin")
    ap.add_argument("--control", help="control conversion as owner/repo[@rev]; "
                                      "overrides the family pin")
    ap.add_argument("--tensors", help="regex over VENDOR tensor names; when "
                    "given, sample these (full rows) instead of the built-in "
                    "Qwen 3.6 plan")
    ap.add_argument("--rows", type=int, default=0,
                    help="with --tensors: cap rows per tensor (0 = all)")
    args = ap.parse_args()
    orig, mlx, fam = open_repos(args)

    if args.tensors:
        plan, unquantized = pattern_plan(orig, mlx, args.tensors,
                                         fam["prefixes"], args.rows)
        if unquantized:
            print(f"{len(unquantized)} matching tensors are unquantized on the "
                  f"control side (no .scales); skipped:")
            for n, c in unquantized[:12]:
                print(f"    {n}  ->  {c}")
            if len(unquantized) > 12:
                print(f"    ... {len(unquantized) - 12} more")
        if not plan:
            raise SystemExit("no quantized tensor matched --tensors")
    elif args.family != "qwen36" or args.orig or args.control:
        raise SystemExit("the built-in sample plan is Qwen 3.6-specific; pass "
                         "--tensors PATTERN for other repos")
    else:
        plan = sample_plan()

    rows, rows8, skipped = [], [], []
    for item in plan:
        opath, _, _, cols = orig.rows(item["orig"], item["orig_row"], item["rows"],
                                      "o_" + item["tag"])
        wname = item["mlx"] + ".weight"
        mpath, mshape, _, mlast = mlx.rows(wname, item["mlx_row"], item["rows"],
                                           "w_" + item["tag"])
        bits = (mlast * 4 * 8) // cols
        group = cols // mlx.info(item["mlx"] + ".scales")["shape"][-1]
        if bits == 8:
            rows8.append(compare(item, opath, mpath, mlx, n_bins=255, group=group))
            continue
        if bits != 4:
            skipped.append((item["tag"], bits, item["mlx"]))
            continue
        rows.append(compare(item, opath, mpath, mlx, n_bins=15, group=group))

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
