# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx-lm==0.31.3", "mlx==0.32.2", "mlx-metal==0.32.2", "numpy==2.5.3"]
# ///
"""Isolate QAT attention using identical saved native inputs and KV history.

Run alone after tracing: uv run Scripts/gemma_qat_attention_probe.py INSTALL TRACE
No expert reads, weight copies or changes to the full-model numerical oracle.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

import gemma_qat_mlx_reference as reference
from gemma_qat_trace import Observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("install", type=Path)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    assert {n: importlib.metadata.version(n) for n in reference.VERSIONS} == reference.VERSIONS
    assert hashlib.sha256(Path(inspect.getfile(reference.gemma)).read_bytes()).hexdigest() == reference.GEMMA_SOURCE_SHA256
    manifest = json.loads((args.install / "manifest.json").read_text())
    assert manifest["modelID"] == reference.MODEL_ID
    assert manifest["sourceSnapshotHash"] == "sha256:" + reference.SOURCE
    meta = json.loads((args.trace / "trace.json").read_text())
    entries = {(e["position"], e["layer"], e["stage"]): e for e in meta["entries"]}
    raw = np.memmap(args.trace / "native.f32", mode="r", dtype="<f4")
    mx.set_cache_limit(128 * 1024 * 1024)
    reader = reference.InstalledWeights(args.install)
    model, _ = reference.installed_model(reader, json.loads((args.install / "tokenizer/config.json").read_text()))
    assert len(meta["sequence"]) < model.args.sliding_window
    caches = model.make_cache()
    results, output_entries, output_bytes = [], [], bytearray()

    for index, layer in enumerate(model.layers):
        attention = layer.self_attn
        snapshots = {}

        def capture(stage, value):
            mx.eval(value)
            snapshots[stage] = np.asarray(value).reshape(-1).copy()

        attention.rope = Observed(attention.rope,
            after=lambda value: capture("query" if value.shape[1] == attention.n_heads else "key", value))
        attention.v_norm = Observed(attention.v_norm, after=lambda value: capture("value", value))
        attention.o_proj = Observed(attention.o_proj, before=lambda value: capture("attention", value))
        native_keys, native_values = [], []

        for position in range(len(meta["sequence"])):
            def snapshot(stage):
                e = entries[position, index, stage]
                return raw[e["offset"] // 4:e["offset"] // 4 + e["count"]].astype(np.float16)

            def check(stage, value, label=None):
                mx.eval(value)
                a = np.asarray(value).reshape(-1)
                b = snapshot(stage)
                assert a.size == b.size
                af, bf = a.astype(np.float64), b.astype(np.float64)
                results.append({"position": position, "layer": index, "stage": label or stage,
                    "count": int(a.size), "equal": int(np.count_nonzero(a == b)),
                    "relative_l2": float(np.linalg.norm(af - bf) / max(np.linalg.norm(af), 1e-12)),
                    "max_absolute": float(np.abs(af - bf).max())})
                output_entries.append({"position": position, "layer": index, "stage": label or stage,
                    "offset": len(output_bytes), "count": int(a.size)})
                output_bytes.extend(a.astype("<f2").tobytes())

            result = attention(mx.array(snapshot("input_norm"))[None, None, :], mask=None, cache=caches[index])
            mx.eval(result[0])
            for stage in ["query", "key", "value", "attention"]:
                check(stage, snapshots[stage])
            hd, heads = attention.head_dim, attention.n_kv_heads
            native_keys.append(snapshot("key").reshape(heads, hd))
            native_values.append(snapshot("value").reshape(heads, hd))
            query = mx.array(snapshot("query").reshape(1, attention.n_heads, 1, hd))
            keys = mx.array(np.stack(native_keys, axis=1)[None])
            values = mx.array(np.stack(native_values, axis=1)[None])
            direct = reference.gemma.scaled_dot_product_attention(
                query, keys, values, cache=caches[index], scale=attention.scale, mask=None)
            check("attention", direct, "attention_native_qkv")

    files = {"attention-reference.f16": output_bytes,
        "attention-cases.json": (json.dumps(output_entries, sort_keys=True, indent=2) + "\n").encode()}
    for name, data in files.items():
        path = args.trace / name
        if path.exists():
            raise FileExistsError(f"refusing to replace frozen witness {path}")
        path.write_bytes(data)
    summary = {"results": results, "dependencies": reference.VERSIONS,
        "reference_source_sha256": reference.GEMMA_SOURCE_SHA256,
        "trace_sha256": {n: hashlib.sha256((args.trace / n).read_bytes()).hexdigest() for n in ["trace.json", "native.f32"]},
        "sha256": {n: hashlib.sha256(data).hexdigest() for n, data in files.items()}}
    (args.trace / "attention-probe.json").write_text(json.dumps(summary, indent=2) + "\n")
    for stage in dict.fromkeys(r["stage"] for r in results):
        rows = [r for r in results if r["stage"] == stage]
        print(json.dumps({"stage": stage, "equal": sum(r["equal"] for r in rows),
            "count": sum(r["count"] for r in rows), "worst": max(rows, key=lambda r: r["relative_l2"])}), flush=True)


if __name__ == "__main__":
    main()
