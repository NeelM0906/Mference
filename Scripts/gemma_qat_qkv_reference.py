# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx-lm==0.31.3", "mlx==0.32.2", "mlx-metal==0.32.2", "numpy==2.5.3"]
# ///
"""Freeze QAT projection, per-head normalization and rotary intermediates.

Run alone: uv run Scripts/gemma_qat_qkv_reference.py INSTALL TRACE
Uses saved native layer inputs and installed resident weights. No expert reads.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models import rope_utils

import gemma_qat_mlx_reference as reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("install", type=Path)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    assert {n: importlib.metadata.version(n) for n in reference.VERSIONS} == reference.VERSIONS
    sources = {inspect.getfile(reference.gemma): reference.GEMMA_SOURCE_SHA256,
        inspect.getfile(rope_utils): "9f68c938c040fa111d13f2ed95c70e8261515fb3b54f8a0a474c096baf4e087a"}
    for name, digest in sources.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
    hashes = {"trace.json": "f7f848996c3672a5f2396df62c2a0f735096acafe16a573d4a57b091c9d3b375",
        "native.f32": "80426c775806f87165a03385cf968f00a82318f6a5b77b7a1a396867138385a8"}
    for name, digest in hashes.items():
        assert hashlib.sha256((args.trace / name).read_bytes()).hexdigest() == digest
    for name in ["qkv-reference.f16", "qkv-cases.json", "qkv-witness.json"]:
        if (args.trace / name).exists():
            raise FileExistsError(f"refusing to replace frozen witness {args.trace / name}")
    manifest = json.loads((args.install / "manifest.json").read_text())
    assert manifest["modelID"] == reference.MODEL_ID
    assert manifest["sourceSnapshotHash"] == "sha256:" + reference.SOURCE
    meta = json.loads((args.trace / "trace.json").read_text())
    entries = {(e["position"], e["layer"], e["stage"]): e for e in meta["entries"]}
    raw = np.memmap(args.trace / "native.f32", mode="r", dtype="<f4")
    mx.set_cache_limit(128 * 1024 * 1024)
    reader = reference.InstalledWeights(args.install)
    model, _ = reference.installed_model(reader, json.loads((args.install / "tokenizer/config.json").read_text()))
    cases, payload, differences = [], bytearray(), []

    def snapshot(position, layer, stage):
        item = entries[position, layer, stage]
        return raw[item["offset"] // 4:item["offset"] // 4 + item["count"]].astype(np.float16)

    for layer, block in enumerate(model.layers):
        attention = block.self_attn
        positions = list(range(len(meta["sequence"])))
        if layer in [0, 5]:
            positions += [31, 128, 1024, 4097, 131071]
        for position in positions:
            input_position = min(position, len(meta["sequence"]) - 1)
            x = mx.array(snapshot(input_position, layer, "input_norm"))[None, None, :]
            hd = attention.head_dim
            q = attention.q_proj(x).reshape(1, 1, attention.n_heads, hd)
            k = attention.k_proj(x).reshape(1, 1, attention.n_kv_heads, hd)
            v = k if attention.use_k_eq_v else attention.v_proj(x).reshape(1, 1, attention.n_kv_heads, hd)
            qn, kn, vn = attention.q_norm(q), attention.k_norm(k), attention.v_norm(v)
            qr = attention.rope(qn.transpose(0, 2, 1, 3), offset=mx.array(position))
            kr = attention.rope(kn.transpose(0, 2, 1, 3), offset=mx.array(position))
            values = {"input": x, "q_raw": q, "k_raw": k, "v_raw": v,
                "q_norm": qn, "k_norm": kn, "v_norm": vn, "query": qr, "key": kr}
            mx.eval(*values.values())
            case = {"position": position, "input_position": input_position, "layer": layer,
                "head_dim": hd, "kv_heads": attention.n_kv_heads,
                "theta": 10000 if hd == 256 else 1000000,
                "rotated_pairs": 128 if hd == 256 else 64, "values": {}}
            for stage, value in values.items():
                data = np.asarray(value).astype("<f2").tobytes()
                case["values"][stage] = {"offset": len(payload), "count": len(data) // 2}
                payload.extend(data)
            cases.append(case)
            if position < len(meta["sequence"]):
                for stage, value in [("query", qr), ("key", kr), ("value", vn)]:
                    a = np.asarray(value).reshape(-1)
                    b = snapshot(position, layer, stage)
                    differences.append({"position": position, "layer": layer, "stage": stage,
                        "count": a.size, "different": int(np.count_nonzero(a != b)),
                        "max_absolute": float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())})
    files = {"qkv-reference.f16": payload,
        "qkv-cases.json": (json.dumps(cases, sort_keys=True, indent=2) + "\n").encode()}
    for name, data in files.items():
        (args.trace / name).write_bytes(data)
    summary = {"cases": len(cases), "count": len(payload) // 2, "dependencies": reference.VERSIONS,
        "sources": {Path(n).name: d for n, d in sources.items()}, "inputs": hashes,
        "sha256": {n: hashlib.sha256(data).hexdigest() for n, data in files.items()},
        "native_differences": differences}
    (args.trace / "qkv-witness.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "native_differences"}), flush=True)
    for stage in ["query", "key", "value"]:
        print(stage, sum(r["different"] for r in differences if r["stage"] == stage), flush=True)


if __name__ == "__main__":
    main()
