# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx-lm==0.31.3", "mlx==0.32.2", "mlx-metal==0.32.2", "numpy==2.5.3"]
# ///
"""Freeze full-attention intermediates from the pinned MLX fallback formula.

Run alone: uv run Scripts/gemma_qat_attention_stages.py TRACE
Uses saved native Q/K/V only; no model loading or checkpoint copies.
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
        "trace.json": "f7f848996c3672a5f2396df62c2a0f735096acafe16a573d4a57b091c9d3b375",
        "native.f32": "4285c0af675a3b28874bc3196824675f25e38258a990e4ff7c72ebf0afac1469",
        "attention-cases.json": "1e31903aa64cdb2ec92039a4928d7cb2306ddb15d2acac440fcc142ecb5a9a2a",
        "attention-reference.f16": "aeb5f0d449e3cfd13f55cbdc12bebfbc4874094b3062e0eb5271c6dda7fa89ec",
    }
    for name, digest in hashes.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    entries = {(e["position"], e["layer"], e["stage"]): e
        for e in json.loads((root / "trace.json").read_text())["entries"]}
    raw = np.memmap(root / "native.f32", mode="r", dtype="<f4")
    expected = np.memmap(root / "attention-reference.f16", mode="r", dtype="<f2")
    cases, payload = [], bytearray()
    compared = 0

    def snapshot(position, layer, stage):
        entry = entries[position, layer, stage]
        return raw[entry["offset"] // 4:entry["offset"] // 4 + entry["count"]].astype(np.float16)

    for item in json.loads((root / "attention-cases.json").read_text()):
        if item["stage"] != "attention_native_qkv" or item["count"] != 16 * 512:
            continue
        position, layer = item["position"], item["layer"]
        q = mx.array(snapshot(position, layer, "query").reshape(1, 2, 8, 1, 512))
        histories = [np.stack([snapshot(p, layer, stage).reshape(2, 512)
            for p in range(position + 1)], axis=1)[None, :, None]
            for stage in ("key", "value")]
        k, v = (mx.array(value) for value in histories)
        # mlx/fast.cpp v0.32.2 fallback: FP16 matmul, precise softmax
        # retaining the input dtype, then FP16 matmul. Source scale is 1.
        scores = q @ mx.swapaxes(k, -1, -2)
        probabilities = mx.softmax(scores, axis=-1, precise=True)
        output = probabilities @ v
        mx.eval(scores, probabilities, output)
        frozen = expected[item["offset"] // 2:item["offset"] // 2 + item["count"]]
        assert np.array_equal(np.asarray(output).reshape(-1), frozen), (position, layer)
        compared += item["count"]
        for name, values in [("scores", scores), ("probabilities", probabilities), ("output", output)]:
            data = np.asarray(values).astype("<f2").tobytes()
            cases.append({"position": position, "layer": layer, "stage": name,
                "count": len(data) // 2, "offset": len(payload)})
            payload.extend(data)

    files = {"attention-stages.f16": payload,
        "attention-stages.json": (json.dumps(cases, sort_keys=True, indent=2) + "\n").encode()}
    for name in files:
        if (root / name).exists():
            raise FileExistsError(f"refusing to replace frozen witness {root / name}")
    for name, data in files.items():
        (root / name).write_bytes(data)
    print(json.dumps({"compared": compared, "equal": compared, "dependencies": reference.VERSIONS,
        "sha256": {n: hashlib.sha256(data).hexdigest() for n, data in files.items()}}), flush=True)


if __name__ == "__main__":
    main()
