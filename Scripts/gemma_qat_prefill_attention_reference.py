# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx-lm==0.31.3", "mlx==0.32.2", "mlx-metal==0.32.2", "numpy==2.5.3"]
# ///
"""Freeze attention expectations for the actual v14 prefill Q/K/V inputs.

Run alone: uv run Scripts/gemma_qat_prefill_attention_reference.py TRACE
Uses saved activations only. Supplemental repeated histories cover boundaries;
the original six-row prompt remains the real-reproducer case for every layer.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

import gemma_qat_mlx_reference as reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    root = parser.parse_args().trace
    assert {n: importlib.metadata.version(n) for n in reference.VERSIONS} == reference.VERSIONS
    hashes = {
        "trace.json": "176709159a8174ffaafc51f29e8cf749c6d805ac07c47de956a34f4b9391873c",
        "native.f32": "272545a9817f292134db91fd68b14bba779e6042ef6ea8b3d7ecd4a71d7e3be6",
    }
    for name, digest in hashes.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    for name in ["prefill-attention.f16", "prefill-attention.json"]:
        if (root / name).exists():
            raise FileExistsError(f"refusing to replace frozen witness {root / name}")
    meta = json.loads((root / "trace.json").read_text())
    assert meta["sequence"] == [2, 818, 5279, 529, 7001, 563]
    entries = {(e["position"], e["layer"], e["stage"]): e for e in meta["entries"]}
    raw = np.memmap(root / "native.f32", mode="r", dtype="<f4")

    def snapshot(position, layer, stage):
        e = entries[position, layer, stage]
        data = raw[e["offset"] // 4:e["offset"] // 4 + e["count"]]
        result = data.astype("<f2")
        assert np.array_equal(result.astype(np.float32), data)
        return result

    cases, payload, observed = [], bytearray(), []
    shapes = [(layer, 0, 6) for layer in range(30)]
    shapes += [(layer, start, count) for layer in [0, 5]
               for start, count in [(27, 6), (123, 9), (1019, 7), (1149, 7), (2299, 7)]]
    shapes += [(5, 4093, 7)]
    mx.set_cache_limit(32 * 1024 * 1024)
    for layer, start, count in shapes:
        hd = snapshot(0, layer, "query").size // 16
        heads, window = (8, 1024) if hd == 256 else (2, start + count)
        rows = {stage: np.stack([snapshot(p, layer, stage) for p in range(6)])
                for stage in ["query", "key", "value"]}
        query = rows["query"][np.arange(start, start + count) % 6]
        histories = [rows[stage][np.arange(start + count) % 6].reshape(-1, heads, hd)
                     for stage in ["key", "value"]]
        result = []
        for row, position in enumerate(range(start, start + count)):
            first = max(0, position + 1 - window)
            k, v = [mx.array(np.swapaxes(h[first:position + 1], 0, 1)[None]) for h in histories]
            q = mx.array(query[row].reshape(1, 16, 1, hd))
            output = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0)
            mx.eval(output)
            result.append(np.asarray(output).reshape(-1).astype("<f2"))
        result = np.stack(result)
        cases.append({"layer": layer, "head_dim": hd, "start": start, "query_count": count,
                      "window": window, "offset": len(payload), "count": result.size,
                      "input_sha256": [hashlib.sha256(x.tobytes()).hexdigest() for x in [query, *histories]]})
        payload.extend(result.tobytes())
        if start == 0:
            actual = np.stack([snapshot(p, layer, "attention") for p in range(6)])
            observed.append({"layer": layer, "different": int(np.count_nonzero(actual != result)),
                             "count": int(result.size),
                             "max_absolute": float(np.max(np.abs(actual.astype(float) - result.astype(float))))})
    files = {"prefill-attention.f16": payload,
             "prefill-attention.json": (json.dumps(cases, sort_keys=True, indent=2) + "\n").encode()}
    for name, data in files.items():
        (root / name).write_bytes(data)
    summary = {"cases": len(cases), "values": len(payload) // 2, "dependencies": reference.VERSIONS,
               "inputs": {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in ["trace.json", "native.f32"]},
               "sha256": {n: hashlib.sha256(data).hexdigest() for n, data in files.items()},
               "original_prefill_differences": observed}
    (root / "prefill-attention-reference-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
