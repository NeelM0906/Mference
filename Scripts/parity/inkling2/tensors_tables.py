#!/usr/bin/env python3
"""Markdown tables for TENSORS.md from a gguf_header_dump.py output directory.

    python3 -I Scripts/parity/inkling2/tensors_tables.py <dump-dir> > tables.md

Prints: the key/value table (long arrays summarized), the per-layer layout
(attention kind, expert type, per-expert strides, extra tensors), and the
full tensor table (name, type, ggml shape ne0..ne3, absolute offset, bytes).
"""

import json
import os
import re
import sys

QK = {"NVFP4": (64, 36), "Q8_0": (32, 34), "F32": (1, 4), "BF16": (1, 2)}


def main():
    d = sys.argv[1]
    kv = json.load(open(os.path.join(d, "kv.json")))
    rows = []
    with open(os.path.join(d, "tensors.tsv")) as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            rows.append({"i": int(p[0]), "name": p[1], "type": p[2], "ne": [int(x) for x in p[4:8]],
                         "off": int(p[8]), "bytes": int(p[9])})

    print("### Key/values\n")
    print("| Key | Type | Value |")
    print("|---|---|---|")
    for k, v in kv.items():
        val = v["value"]
        if isinstance(val, dict) and "length" in val:
            s = f"{val['type']}, length {val['length']:,}, sha256 `{val['sha256']}`"
        elif isinstance(val, list):
            s = "`" + json.dumps(val) + "`"
        else:
            s = f"`{val}`"
        print(f"| `{k}` | {v['type']} | {s} |")

    print("\n### Per-layer layout\n")
    swa = kv["inkling2_loop.attention.sliding_window_pattern"]["value"]
    if isinstance(swa, dict):
        swa = swa["values"]
    start = kv["inkling2_loop.loop.start"]["value"]
    end = kv["inkling2_loop.loop.end"]["value"]
    base = None
    layers = {}
    for r in rows:
        m = re.match(r"blk\.(\d+)\.(.+)$", r["name"])
        if m:
            layers.setdefault(int(m.group(1)), {})[m.group(2)] = r
    common = set.intersection(*(set(t) for t in layers.values()))
    print("Tensors every layer has: " + ", ".join(f"`{n}`" for n in sorted(common)) + ".\n")
    print("| Layer | Attention | Loop core | Expert type | gate_up bytes/expert | down bytes/expert | Layer bytes | Extra tensors |")
    print("|---:|---|---|---|---:|---:|---:|---|")
    for il in sorted(layers):
        t = layers[il]
        gu, dn = t["ffn_gate_up_exps.weight"], t["ffn_down_exps.weight"]
        e = gu["ne"][2]
        extra = sorted(set(t) - common)
        lb = sum(x["bytes"] for x in t.values())
        att = "sliding 1024" if swa[il] else "global"
        core = "yes (3 passes)" if start <= il < end else ""
        print(f"| {il} | {att} | {core} | {gu['type']} | {gu['bytes'] // e:,} | {dn['bytes'] // e:,} | {lb:,} | "
              + ", ".join(f"`{x}`" for x in extra) + " |")

    print("\n### Full tensor table\n")
    print("Shapes are ggml `ne` order (ne0 is the contiguous dimension; a weight `[in, out, experts]`). "
          "Offsets are absolute file offsets (data section start + tensor offset).\n")
    print("| # | Name | Type | ne0 | ne1 | ne2 | Offset | Bytes |")
    print("|---:|---|---|---:|---:|---:|---:|---:|")
    for r in rows:
        ne = r["ne"]
        print(f"| {r['i']} | `{r['name']}` | {r['type']} | {ne[0]} | {ne[1]} | {ne[2]} | {r['off']} | {r['bytes']} |")


if __name__ == "__main__":
    main()
