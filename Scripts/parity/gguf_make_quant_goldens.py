#!/usr/bin/env python3
"""Generate GGUF quant goldens (WS4 U3): blocks paired with llama.cpp's own
dequantization of them, from the pinned llama.cpp's `mf-dequant`
(ggml_get_type_traits()->to_float, b1b770d9).

    python3 -I Scripts/parity/gguf_make_quant_goldens.py synthetic <mf-dequant>
        Seeded synthetic blocks for the 7 types of the Unsloth UD-IQ3_XXS
        install, built to hit the edge cases (every IQ3_XXS / IQ2_S grid
        index, every sign index and scale nibble, IQ4_XS ls = 32 / 0 / 63,
        signed zeros, fp16 d at +-0, subnormal, min normal and +-max).
        Writes the committed fixture Tests/Mference/Fixtures/gguf-quant/.
        Contains no model data.

    python3 -I Scripts/parity/gguf_make_quant_goldens.py real <mf-dequant> <gguf-shard-dir> <out-dir>
        12 real blocks per type (start, middle and end of the first tensor of
        that type) from the pinned GGUF. These are model weights: <out-dir>
        must be outside the repository (a local, uncommitted fixture for the
        env-gated MFERENCE_GGUF_REAL_GOLDENS test).

Stdlib only. Both modes write goldens.safetensors and goldens-manifest.json.
"""

import hashlib
import json
import os
import pathlib
import random
import struct
import subprocess
import sys
import tempfile

SOURCE_REPO = "unsloth/Qwen3.8-Flash-Next-GGUF"
SOURCE_COMMIT = "3f3ba9cb826b49c665510ce1702fd513a393ce39"
LLAMA_CPP_COMMIT = "b1b770d90d94cad9e9ea5b993bdbc383a522fb28"
SEED = 0x4747_5546

# ggml type id -> (name, weights per block, bytes per block, offset of fp16 d)
TYPES = {
    8: ("q8_0", 32, 34, 0),
    14: ("q6_K", 256, 210, 208),
    18: ("iq3_xxs", 256, 98, 0),
    20: ("iq4_nl", 32, 18, 0),
    22: ("iq2_s", 256, 82, 0),
    23: ("iq4_xs", 256, 136, 0),
    42: ("q2_0", 64, 18, 0),
}
# fp16 d edge cases: +0, -0, min subnormal (+-), max subnormal, min normal,
# max finite (+-), 1.0, -1.0.
D_EDGES = [0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x7BFF, 0xFBFF, 0x3C00, 0xBC00]

REPO = pathlib.Path(__file__).resolve().parents[2]
SYNTHETIC_OUT = REPO / "Tests/Mference/Fixtures/gguf-quant"


def random_finite_d(rng):
    while True:
        bits = rng.getrandbits(16)
        if bits & 0x7C00 != 0x7C00:
            return bits


def synthetic_blocks(t, rng):
    """Seeded blocks of type t, edge cases first, then random ones."""
    name, weights, nbytes, d_at = TYPES[t]
    count = {"iq2_s": 40}.get(name, 24)
    blocks = []
    for b in range(count):
        blk = bytearray(rng.getrandbits(8) for _ in range(nbytes))
        d = D_EDGES[b] if b < len(D_EDGES) else random_finite_d(rng)
        blk[d_at:d_at + 2] = struct.pack("<H", d)
        blocks.append(blk)

    if name == "iq3_xxs":
        # Blocks 0-3 cover all 256 grid indices, all 128 sign indices and all
        # 16 scale nibbles (twice).
        for b in range(4):
            blk = blocks[b]
            blk[2:66] = bytes(range(64 * b, 64 * b + 64))
            for g in range(8):
                word = 0
                for l in range(4):
                    word |= ((32 * b + 4 * g + l) % 128) << (7 * l)
                word |= ((8 * b + g) % 16) << 28
                blk[66 + 4 * g:70 + 4 * g] = struct.pack("<I", word)
    elif name == "iq2_s":
        # Blocks 0-31 cover all 1024 10-bit grid indices; scales cover all nibbles.
        for b in range(32):
            blk = blocks[b]
            qh = [0] * 8
            for k in range(32):
                g, l = k // 4, k % 4
                idx = 32 * b + k
                blk[2 + 4 * g + l] = idx & 0xFF
                qh[g] |= (idx >> 8) << (2 * l)
            blk[66:74] = bytes(qh)
            blk[74:82] = bytes(((b + 2 * i) % 16) | (((b + 2 * i + 1) % 16) << 4) for i in range(8))
    elif name == "iq4_xs":
        # ls = scales_l nibble | 2 bits of scales_h << 4. Block 0 (d = +0) and a
        # normal-d block with ls = 32 everywhere (M = 0: llama.cpp's -0 for
        # negative kvalues); ls = 0 (M = -32); ls = 63 (M = 31).
        for b, (low, high) in {0: (0, 2), 8: (0, 2), 9: (0, 0), 10: (0xF, 3)}.items():
            blk = blocks[b]
            blk[2:4] = struct.pack("<H", sum(high << (2 * g) for g in range(8)))
            blk[4:8] = bytes([low | low << 4] * 4)
    elif name == "q2_0":
        # q = 1 everywhere: (1 - 1) * d is -0 for the negative-d edge blocks.
        for b in (1, 3, 7, 9):
            blocks[b][2:18] = bytes([0x55] * 16)
        blocks[8][2:18] = bytes([0x00] * 8 + [0xFF] * 8)  # q = 0 and q = 3
    elif name == "q6_K":
        # Extreme sub-block scales and the extreme 6-bit codes.
        blocks[8][192:208] = struct.pack("<16b", -128, 127, 0, 1, -1, -64, 64, 2, -2, 3, -3, 100, -100, 50, -50, 7)
        blocks[8][0:192] = bytes([0x00] * 192)   # q = -32
        blocks[9][0:192] = bytes([0xFF] * 192)   # q = 31
        blocks[9][192:208] = struct.pack("<16b", *([127] * 8 + [-128] * 8))
    elif name == "q8_0":
        blocks[8][2:34] = struct.pack("<32b", *([-128, 127, 0, -1, 1] * 6 + [-128, 127]))
    return b"".join(blocks)


def run_dequant(dequant, t, raw, tmp):
    name, weights, nbytes, _ = TYPES[t]
    src, dst = os.path.join(tmp, f"{name}.blocks"), os.path.join(tmp, f"{name}.f32")
    pathlib.Path(src).write_bytes(raw)
    subprocess.run([str(dequant), str(t), src, dst], check=True, stdout=subprocess.DEVNULL)
    values = pathlib.Path(dst).read_bytes()
    if len(values) != len(raw) // nbytes * weights * 4:
        sys.exit(f"{name}: mf-dequant wrote {len(values)} B for {len(raw) // nbytes} blocks")
    return values


# ---- GGUF header reading (real mode) ----

class Reader:
    def __init__(self, f):
        self.f = f

    def take(self, fmt):
        size = struct.calcsize(fmt)
        data = self.f.read(size)
        if len(data) != size:
            raise EOFError("truncated GGUF header")
        return struct.unpack("<" + fmt, data)

    def u32(self):
        return self.take("I")[0]

    def u64(self):
        return self.take("Q")[0]

    def string(self):
        return self.f.read(self.u64()).decode("utf-8")

    def value(self, t):
        scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        if t in scalar:
            return self.take(scalar[t])[0]
        if t == 8:
            return self.string()
        if t == 9:
            inner, n = self.u32(), self.u64()
            return [self.value(inner) for _ in range(n)]
        raise ValueError(f"unknown GGUF value type {t}")


def read_shard(path):
    with open(path, "rb") as f:
        r = Reader(f)
        if f.read(4) != b"GGUF" or r.u32() != 3:
            raise ValueError(f"{path}: not a GGUF v3 file")
        n_tensors, n_kv = r.u64(), r.u64()
        alignment = 32
        for _ in range(n_kv):
            key = r.string()
            value = r.value(r.u32())
            if key == "general.alignment":
                alignment = int(value)
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            dims = [r.u64() for _ in range(r.u32())]
            tensors.append((name, dims, r.u32(), r.u64()))
        pos = f.tell()
        return (pos + alignment - 1) // alignment * alignment, tensors


def real_blocks(shard_dir):
    shards = sorted(shard_dir.glob("Qwen3.8-Flash-Next-UD-IQ3_XXS-0000?-of-00004.gguf"))
    if len(shards) != 4:
        sys.exit(f"expected 4 shards in {shard_dir}, found {len(shards)}")
    picked = {}
    for shard in shards:
        data_start, tensors = read_shard(shard)
        for name, dims, t, offset in tensors:
            if t in TYPES and t not in picked:
                picked[t] = (shard, data_start, name, dims, offset)
    if set(picked) != set(TYPES):
        sys.exit(f"no tensor of ggml type(s) {sorted(set(TYPES) - set(picked))}")
    out = {}
    for t, (shard, data_start, name, dims, offset) in picked.items():
        _, weights, nbytes, _ = TYPES[t]
        n = 1
        for d in dims:
            n *= d
        nb = n // weights
        spots = [0, nb // 2, nb - 4]
        raw = bytearray()
        with open(shard, "rb") as f:
            for s in spots:
                f.seek(data_start + offset + s * nbytes)
                chunk = f.read(4 * nbytes)
                if len(chunk) != 4 * nbytes:
                    sys.exit(f"{name}: short read at block {s}")
                raw += chunk
        out[t] = (bytes(raw), {"tensor": name, "shape": dims, "shard": shard.name,
                               "block_indices": [s + i for s in spots for i in range(4)]})
    return out


def write(out_dir, dequant, blocks_by_type, kind):
    dequant_sha = hashlib.sha256(dequant.read_bytes()).hexdigest()
    tensors, types = {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        for t in sorted(blocks_by_type):
            raw, info = blocks_by_type[t]
            name = TYPES[t][0]
            values = run_dequant(dequant, t, raw, tmp)
            tensors[f"{name}.blocks"] = ("U8", [len(raw)], raw)
            tensors[f"{name}.dequant"] = ("F32", [len(values) // 4], values)
            types[name] = dict(info, ggml_type=t, blocks=len(raw) // TYPES[t][2],
                               blocks_sha256=hashlib.sha256(raw).hexdigest(),
                               dequant_sha256=hashlib.sha256(values).hexdigest())
    header, blob = {}, bytearray()
    for key in sorted(tensors):
        dtype, shape, data = tensors[key]
        header[key] = {"dtype": dtype, "shape": shape, "data_offsets": [len(blob), len(blob) + len(data)]}
        blob += data
    head = json.dumps(header, separators=(",", ":")).encode()
    head += b" " * ((8 - len(head) % 8) % 8)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "goldens.safetensors").write_bytes(struct.pack("<Q", len(head)) + head + blob)
    manifest = {
        "schema": "mference.gguf-quant.goldens/1",
        "generated_by": "Scripts/parity/gguf_make_quant_goldens.py",
        "kind": kind,
        "reference": {"llama_cpp_commit": LLAMA_CPP_COMMIT,
                      "tool": "mf-dequant (ggml_get_type_traits()->to_float)", "tool_sha256": dequant_sha},
        "types": types,
    }
    if kind == "synthetic":
        manifest["seed"] = SEED
    else:
        manifest["source"] = {"repo": SOURCE_REPO, "commit": SOURCE_COMMIT}
    (out_dir / "goldens-manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {out_dir} ({kind}, {len(blob)} data bytes, {len(types)} types)")


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "synthetic" and len(sys.argv) == 3:
        rng = random.Random(SEED)
        blocks = {t: (synthetic_blocks(t, rng), {}) for t in sorted(TYPES)}
        write(SYNTHETIC_OUT, pathlib.Path(sys.argv[2]), blocks, "synthetic")
    elif len(sys.argv) == 5 and sys.argv[1] == "real":
        out = pathlib.Path(sys.argv[4]).resolve()
        if out == REPO or REPO in out.parents:
            sys.exit(f"{out} is inside the repository; real blocks are model weights and must stay local")
        write(out, pathlib.Path(sys.argv[2]), real_blocks(pathlib.Path(sys.argv[3])), "real")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
