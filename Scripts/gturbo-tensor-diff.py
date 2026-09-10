#!/usr/bin/env python3
"""Compare two .gturbo installs tensor by tensor, on disk, no model process.

    Scripts/gturbo-tensor-diff.py CONTROL.gturbo OURS.gturbo

For every resident tensor this dequantizes both sides and reports the relative
Frobenius difference. Two legitimate INT4 group-64 grids of the same BF16
source differ by ~0.13; anything far above that is a structural defect, not a
quantizer effect, and the tensor's name says immediately which subsystem owns
it.

This is the tool that closed W2.1b's model-level half. The gate's first run
reported *zero* top-1 agreement over 882 teacher-forced positions — impossible
for a quantizer measured better than the control on 128 of 134 sampled tensors,
so the divergence had to be structural. Running this localized it in one shot:
100 tensors at relative difference ~1.0, all of them RMSNorm gains, all
unquantized BF16 passthrough where quantization cannot explain any difference
at all. The cause was the `1 + w` norm convention (see
`FlashNextPlanner.foldsNormBias`).

Keep reaching for it whenever two installs of the same checkpoint disagree at
the model level: it needs no GPU, loads no model, and answers "which tensors"
before anyone has to guess "which subsystem". Names are normalized, because an
mlx-lm conversion writes `language_model.model.<...>` while a vendor's original
repo writes `model.language_model.<...>`.
"""
import json
import os
import struct
import sys
import numpy as np

HDR = 24
ENT = 72


def index(path):
    with open(path, "rb") as f:
        index_size, resident_size, count = struct.unpack("<QQQ", f.read(HDR))
        f.seek(0)
        blob = f.read(index_size)
    out = {}
    for i in range(count):
        rec = blob[HDR + i * ENT: HDR + (i + 1) * ENT]
        name_off, name_len = struct.unpack("<IH", rec[0:6])
        dtype = rec[6]
        file_off, size = struct.unpack("<QQ", rec[8:24])
        dims = [d for d in struct.unpack("<IIII", rec[24:40]) if d > 0]
        s_off, s_size, b_off, b_size = struct.unpack("<QQQQ", rec[40:72])
        name = blob[name_off:name_off + name_len].decode()
        out[name] = dict(dtype=dtype, off=file_off, size=size, dims=dims,
                         s_off=s_off, s_size=s_size, b_off=b_off, b_size=b_size)
    return out


def normalize(name):
    for p in ("language_model.model.", "model.language_model.",
              "language_model.", "model."):
        if name.startswith(p):
            return name[len(p):]
    return name


def bf16(u16):
    return (np.asarray(u16, np.uint16).astype(np.uint32) << 16).view(np.float32)


def read(path, off, n):
    with open(path, "rb") as f:
        f.seek(off)
        return f.read(n)


def values(path, e):
    """Dequantized float32 values, or the raw BF16 values when unquantized."""
    if e["s_size"] == 0:
        raw = np.frombuffer(read(path, e["off"], e["size"]), np.uint16)
        return bf16(raw) if e["dtype"] == 1 else None
    count = 1
    for d in e["dims"]:
        count *= d
    bits = round(8 * e["size"] / count)
    packed = np.frombuffer(read(path, e["off"], e["size"]), np.uint8)
    if bits == 4:
        q = np.empty(count, np.uint8)
        q[0::2] = packed & 0x0F
        q[1::2] = packed >> 4
    elif bits == 8:
        q = packed.copy()
    else:
        return None
    scales = bf16(np.frombuffer(read(path, e["s_off"], e["s_size"]), np.uint16))
    biases = bf16(np.frombuffer(read(path, e["b_off"], e["b_size"]), np.uint16))
    g = q.reshape(-1, 64).astype(np.float32)
    return (g * scales[:, None] + biases[:, None]).reshape(-1)


def main():
    ctrl_dir, ours_dir = sys.argv[1], sys.argv[2]
    cw = os.path.join(ctrl_dir, "model_weights.bin")
    ow = os.path.join(ours_dir, "model_weights.bin")
    ci = {normalize(k): v for k, v in index(cw).items()}
    oi = {normalize(k): v for k, v in index(ow).items()}
    rows = []
    for name in sorted(set(ci) & set(oi)):
        a, b = values(cw, ci[name]), values(ow, oi[name])
        if a is None or b is None or a.size != b.size:
            rows.append((name, float("nan"), ci[name], oi[name]))
            continue
        den = np.sqrt((a.astype(np.float64) ** 2).sum())
        rel = float(np.sqrt(((a - b).astype(np.float64) ** 2).sum()) / den) if den else 0.0
        rows.append((name, rel, ci[name], oi[name]))
    rows.sort(key=lambda r: (-1 if np.isnan(r[1]) else r[1]), reverse=True)
    print(f"{'tensor':62} {'relDiff':>9}  dims")
    for name, rel, c, _ in rows[:40]:
        print(f"{name:62} {rel:9.5f}  {c['dims']}")
    finite = [r[1] for r in rows if not np.isnan(r[1])]
    print(f"\n{len(rows)} tensors; median relDiff {np.median(finite):.5f}")
    bad = [r for r in rows if np.isnan(r[1]) or r[1] > 0.5]
    print(f"tensors with relDiff > 0.5 or unreadable: {len(bad)}")
    for name, rel, c, o in bad[:30]:
        print(f"  {name:58} {rel:8.4f} ctrl{c['dims']} dtype{c['dtype']} "
              f"size{c['size']} | ours{o['dims']} dtype{o['dtype']} size{o['size']}")


if __name__ == "__main__":
    main()
