# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx-lm==0.31.3", "mlx==0.32.2", "mlx-metal==0.32.2", "numpy==2.5.3"]
# ///
"""Evaluate projection-order candidates against unchanged pinned MLX math."""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

import gemma_qat_mlx_reference as reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("install", type=Path)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    cases = json.loads((args.trace / "projection-cases.json").read_text())
    metadata = json.loads((args.trace / "trace.json").read_text())
    entries = {(e["position"], e["layer"], e["stage"]): e for e in metadata["entries"]}
    inputs = np.memmap(args.trace / "native.f32", mode="r", dtype="<f4")
    native = np.memmap(args.trace / "projection-native.f16", mode="r", dtype="<f2")
    candidate = np.memmap(args.trace / "projection-source-order.f16", mode="r", dtype="<f2")
    safe = np.memmap(args.trace / "projection-safe-order.f16", mode="r", dtype="<f2")
    reader = reference.InstalledWeights(args.install)
    mx.set_cache_limit(128 * 1024 * 1024)
    results = []
    with (args.trace / "projection-reference.f16").open("wb") as output:
        for case in cases:
            entry = entries[case["position"], case["layer"], case["stage"]]
            x = inputs[entry["offset"] // 4:entry["offset"] // 4 + entry["count"]]
            role = case["role"]
            suffix = "self_attn.o_proj.weight" if role == "o" else f"mlp.{role}_proj.weight"
            projection = reader.resident(f"language_model.model.layers.{case['layer']}.{suffix}")
            value = projection(mx.array(x, dtype=mx.float16)[None, None, :])
            mx.eval(value)
            expected = np.asarray(value).reshape(-1)
            assert output.tell() == case["offset"]
            output.write(expected.astype("<f2").tobytes())
            start, count = case["offset"] // 2, case["count"]
            result = {**case}
            for name, data in [("native", native), ("candidate", candidate), ("safe", safe)]:
                actual = data[start:start + count]
                diff = actual.astype(np.float64) - expected.astype(np.float64)
                result[name] = {"equal": int(np.count_nonzero(actual == expected)),
                    "relative_l2": float(np.linalg.norm(diff) / max(np.linalg.norm(expected.astype(np.float64)), 1e-12)),
                    "maximum_absolute": float(np.max(np.abs(diff)))}
            results.append(result)
    (args.trace / "projection-comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    for role in dict.fromkeys(r["role"] for r in results):
        rows = [r for r in results if r["role"] == role]
        print(json.dumps({"role": role, "values": sum(r["count"] for r in rows),
                          "native_equal": sum(r["native"]["equal"] for r in rows),
                          "candidate_equal": sum(r["candidate"]["equal"] for r in rows),
                          "safe_equal": sum(r["safe"]["equal"] for r in rows),
                          "worst_candidate": max(rows, key=lambda r: r["candidate"]["relative_l2"])}), flush=True)


if __name__ == "__main__":
    main()
