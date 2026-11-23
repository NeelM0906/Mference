#!/usr/bin/env python3
"""Dump a GGUF file's key/values and tensor table without loading weights.

Usage (stdlib only; run with -I):

    python3 -I Scripts/parity/inkling2/gguf_header_dump.py \
        --constants <llama.cpp checkout>/gguf-py/gguf/constants.py \
        --gguf <file.gguf> --out <dir>

The type names, value types and block sizes come from the given llama.cpp
checkout's own `gguf-py/gguf/constants.py` (loaded by path, so its package
`__init__` and its numpy dependency are not imported). The parser follows the
GGUF v3 layout in that checkout's `ggml/src/gguf.cpp`.

Writes <out>/kv.json (every key; long arrays summarized with their length and
the SHA-256 of their canonical JSON), <out>/tensors.tsv (name, type, ggml
shape ne0..ne3, absolute file offset, bytes) and <out>/summary.json.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import struct
import sys


def load_constants(path):
    spec = importlib.util.spec_from_file_location("gguf_constants_ref", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class Reader:
    def __init__(self, f):
        self.f = f

    def read(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError("truncated GGUF header")
        return b

    def u32(self):
        return struct.unpack("<I", self.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.read(8))[0]

    def string(self):
        n = self.u64()
        return self.read(n).decode("utf-8", errors="surrogateescape")


def read_value(r, vt, C):
    T = C.GGUFValueType
    scalar = {
        T.UINT8: "<B", T.INT8: "<b", T.UINT16: "<H", T.INT16: "<h",
        T.UINT32: "<I", T.INT32: "<i", T.FLOAT32: "<f", T.BOOL: "<?",
        T.UINT64: "<Q", T.INT64: "<q", T.FLOAT64: "<d",
    }
    if vt in scalar:
        fmt = scalar[vt]
        return struct.unpack(fmt, r.read(struct.calcsize(fmt)))[0]
    if vt == T.STRING:
        return r.string()
    if vt == T.ARRAY:
        et = T(r.u32())
        n = r.u64()
        if et in scalar:
            fmt = scalar[et]
            size = struct.calcsize(fmt)
            raw = r.read(size * n)
            return {"__array__": et.name, "values": list(struct.unpack("<" + fmt[1:] * n, raw))}
        return {"__array__": et.name, "values": [read_value(r, et, C) for _ in range(n)]}
    raise ValueError(f"unknown GGUF value type {vt}")


def summarize(value, limit):
    if isinstance(value, dict) and "__array__" in value:
        vals = value["values"]
        canon = json.dumps(vals, ensure_ascii=False, separators=(",", ":")).encode("utf-8", errors="surrogateescape")
        out = {"type": f"array[{value['__array__']}]", "length": len(vals),
               "sha256": hashlib.sha256(canon).hexdigest()}
        if len(vals) <= limit:
            out["values"] = vals
        else:
            out["head"] = vals[:8]
        return out
    if isinstance(value, str) and len(value) > 400:
        return {"type": "string", "length": len(value),
                "sha256": hashlib.sha256(value.encode("utf-8", errors="surrogateescape")).hexdigest(),
                "head": value[:200]}
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--constants", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--array-limit", type=int, default=128)
    a = ap.parse_args()
    C = load_constants(a.constants)
    os.makedirs(a.out, exist_ok=True)
    file_size = os.path.getsize(a.gguf)

    with open(a.gguf, "rb") as f:
        r = Reader(f)
        magic = r.read(4)
        if magic != b"GGUF":
            raise SystemExit(f"bad magic {magic!r}")
        version = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()
        kv = {}
        kv_types = {}
        for _ in range(n_kv):
            key = r.string()
            vt = C.GGUFValueType(r.u32())
            kv[key] = read_value(r, vt, C)
            kv_types[key] = vt.name
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            nd = r.u32()
            ne = [r.u64() for _ in range(nd)]
            t = C.GGMLQuantizationType(r.u32())
            off = r.u64()
            tensors.append((name, t, ne, off))
        header_end = f.tell()

    alignment = kv.get("general.alignment", C.GGUF_DEFAULT_ALIGNMENT)
    data_start = (header_end + alignment - 1) // alignment * alignment

    rows = []
    by_type = {}
    for name, t, ne, off in tensors:
        block, tsize = C.GGML_QUANT_SIZES[t]
        n = 1
        for d in ne:
            n *= d
        if ne[0] % block != 0:
            raise SystemExit(f"{name}: ne0 {ne[0]} not a multiple of block {block}")
        nbytes = n // block * tsize
        rows.append((name, t.name, int(t), ne, data_start + off, nbytes))
        s = by_type.setdefault(t.name, {"tensors": 0, "bytes": 0})
        s["tensors"] += 1
        s["bytes"] += nbytes

    # Sanity: tensors tile the data section in offset order with alignment padding.
    ordered = sorted(rows, key=lambda x: x[4])
    gaps = []
    for prev, cur in zip(ordered, ordered[1:]):
        end = prev[4] + prev[5]
        pad = cur[4] - end
        if pad < 0 or pad >= alignment:
            gaps.append({"after": prev[0], "next": cur[0], "gap": pad})
    last_end = ordered[-1][4] + ordered[-1][5]

    with open(os.path.join(a.out, "tensors.tsv"), "w") as o:
        o.write("index\tname\ttype\ttype_id\tne0\tne1\tne2\tne3\toffset\tbytes\n")
        for i, (name, tn, tid, ne, off, nb) in enumerate(rows):
            ne4 = list(ne) + [1] * (4 - len(ne))
            o.write(f"{i}\t{name}\t{tn}\t{tid}\t{ne4[0]}\t{ne4[1]}\t{ne4[2]}\t{ne4[3]}\t{off}\t{nb}\n")

    out_kv = {k: {"type": kv_types[k], "value": summarize(v, a.array_limit)} for k, v in kv.items()}
    with open(os.path.join(a.out, "kv.json"), "w") as o:
        json.dump(out_kv, o, indent=1, ensure_ascii=False)

    summary = {
        "file_bytes": file_size, "gguf_version": version, "n_tensors": n_tensors, "n_kv": n_kv,
        "alignment": alignment, "header_end": header_end, "data_start": data_start,
        "data_end_of_last_tensor": last_end, "trailing_bytes": file_size - last_end,
        "tensor_bytes_total": sum(x[5] for x in rows), "by_type": by_type,
        "layout_gaps": gaps,
    }
    with open(os.path.join(a.out, "summary.json"), "w") as o:
        json.dump(summary, o, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
