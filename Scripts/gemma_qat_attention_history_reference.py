# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx-lm==0.31.3", "mlx==0.32.2", "mlx-metal==0.32.2", "numpy==2.5.3"]
# ///
"""Bounded attention boundary fixtures derived from frozen real Q/K/V rows.

Run alone: uv run Scripts/gemma_qat_attention_history_reference.py TRACE
These repeated histories supplement, and never replace, the actual trace.
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
    for name, digest in {
        "trace.json": "f7f848996c3672a5f2396df62c2a0f735096acafe16a573d4a57b091c9d3b375",
        "native.f32": "4285c0af675a3b28874bc3196824675f25e38258a990e4ff7c72ebf0afac1469",
    }.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    for name in ["attention-history.f16", "attention-history.json"]:
        if (root / name).exists():
            raise FileExistsError(f"refusing to replace frozen witness {root / name}")
    entries = {(e["position"], e["layer"], e["stage"]): e
        for e in json.loads((root / "trace.json").read_text())["entries"]}
    raw = np.memmap(root / "native.f32", mode="r", dtype="<f4")
    cases, payload = [], bytearray()

    def snapshot(position, layer, stage):
        entry = entries[position, layer, stage]
        return raw[entry["offset"] // 4:entry["offset"] // 4 + entry["count"]].astype("<f2")

    mx.set_cache_limit(32 * 1024 * 1024)
    for layer, hd, heads, lengths in [
        (0, 256, 8, [31, 32, 33, 127, 128, 129, 1024, 1025, 1153, 2305]),
        (5, 512, 2, [31, 32, 33, 127, 128, 129, 1024, 1025, 4096, 4097]),
    ]:
        q = snapshot(4, layer, "query").reshape(1, 16, 1, hd)
        seed = [np.stack([snapshot(p, layer, stage).reshape(heads, hd)
            for p in range(5)]) for stage in ["key", "value"]]
        for length in lengths:
            histories = [rows[np.arange(length) % 5] for rows in seed]
            start = max(0, length - 1024) if hd == 256 else 0
            keys, values = [mx.array(np.swapaxes(history[start:], 0, 1)[None]) for history in histories]
            output = mx.fast.scaled_dot_product_attention(mx.array(q), keys, values, scale=1.0)
            mx.eval(output)
            data = np.asarray(output).astype("<f2").tobytes()
            cases.append({"layer": layer, "head_dim": hd, "length": length, "window": 1024 if hd == 256 else length,
                "offset": len(payload), "count": len(data) // 2,
                "input_sha256": [hashlib.sha256(a.tobytes()).hexdigest() for a in [q, *histories]]})
            payload.extend(data)
    files = {"attention-history.f16": payload,
        "attention-history.json": (json.dumps(cases, sort_keys=True, indent=2) + "\n").encode()}
    for name, data in files.items():
        (root / name).write_bytes(data)
    print(json.dumps({"cases": len(cases), "count": len(payload) // 2,
        "dependencies": reference.VERSIONS,
        "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}), flush=True)


if __name__ == "__main__":
    main()
