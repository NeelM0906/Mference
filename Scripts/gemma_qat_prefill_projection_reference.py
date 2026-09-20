# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx-lm==0.31.3", "mlx==0.32.2", "mlx-metal==0.32.2", "numpy==2.5.3"]
# ///
"""Freeze resident projections of the actual v15 long-prefill inputs.

Run alone: uv run Scripts/gemma_qat_prefill_projection_reference.py INSTALL TRACE
Reads only installed resident weights, with no checkpoint download or rewrite.
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
    parser.add_argument("install", type=Path)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    root = args.trace
    assert {n: importlib.metadata.version(n) for n in reference.VERSIONS} == reference.VERSIONS
    for name, digest in {
        "trace.json": "c4fc161029429ae9015d72cf1a96618532c3edbf8590531685bcc6b7ddd13b8e",
        "native.f32": "9675dfc958ef0244fb0375240df075f513ac17508e9d9a04270b7ac1c4042691",
    }.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    names = ["prefill-projection-inputs.f16", "prefill-projection-reference.f16", "prefill-projection-cases.json"]
    for name in names:
        if (root / name).exists():
            raise FileExistsError(f"refusing to replace frozen witness {root / name}")
    meta = json.loads((root / "trace.json").read_text())
    entries = {(e["position"], e["layer"], e["stage"]): e for e in meta["entries"]}
    positions = sorted({e["position"] for e in meta["entries"]})
    native = np.memmap(root / "native.f32", mode="r", dtype="<f4")
    reader = reference.InstalledWeights(args.install)
    mx.set_cache_limit(32 * 1024 * 1024)
    cases, inputs, expected = [], bytearray(), bytearray()
    for layer in [0, 5, 29]:
        for role in ["q", "k", "v", "o"]:
            stage = "attention" if role == "o" else "input_norm"
            weight_role = "k" if layer in [5, 29] and role == "v" else role
            weight = f"language_model.model.layers.{layer}.self_attn.{weight_role}_proj.weight"
            projection = reader.resident(weight)
            input_offset, output_offset = len(inputs), len(expected)
            for position in positions:
                e = entries[position, layer, stage]
                raw = native[e["offset"] // 4:e["offset"] // 4 + e["count"]]
                x = raw.astype("<f2")
                assert np.array_equal(x.astype(np.float32), raw)
                output = projection(mx.array(x)[None, None])
                mx.eval(output)
                y = np.asarray(output).reshape(-1).astype("<f2")
                inputs.extend(x.tobytes())
                expected.extend(y.tobytes())
            cases.append({"layer": layer, "role": role, "weight": weight,
                          "positions": positions, "columns": len(x), "rows": len(y),
                          "input_offset": input_offset, "output_offset": output_offset})
    payloads = [inputs, expected, (json.dumps(cases, sort_keys=True, indent=2) + "\n").encode()]
    for name, data in zip(names, payloads, strict=True):
        (root / name).write_bytes(data)
    print(json.dumps({"cases": len(cases), "rows_per_case": len(positions),
                      "values": len(expected) // 2, "dependencies": reference.VERSIONS,
                      "sha256": {n: hashlib.sha256(b).hexdigest() for n, b in zip(names, payloads, strict=True)}}), flush=True)


if __name__ == "__main__":
    main()
