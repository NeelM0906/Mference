# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx-lm==0.31.3", "mlx==0.32.2", "mlx-metal==0.32.2", "numpy==2.5.3"]
# ///
"""Freeze source post-attention normalization on saved native inputs."""
import argparse
import hashlib
import importlib.metadata
import inspect
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
    inputs, outputs, cases = bytearray(), bytearray(), []

    def append(target, values):
        mx.eval(values)
        array = np.asarray(values).astype("<f2").reshape(-1)
        spec = {"offset": len(target), "count": int(array.size)}
        target.extend(array.tobytes())
        return spec

    for position in [0, 2, 4]:
        for index, layer in enumerate(model.layers):
            def snapshot(layer_id, stage):
                e = entries[position, layer_id, stage]
                return mx.array(raw[e["offset"] // 4:e["offset"] // 4 + e["count"]], dtype=mx.float16)[None, None, :]

            previous = snapshot(index - 1, "layer_output") if index else (
                model.model.embed_tokens(mx.array([[meta["sequence"][position]]])) * model.model.embed_scale)
            attention = snapshot(index, "attention_projection")
            hidden = previous + layer.post_attention_layernorm(attention)
            dense = layer.pre_feedforward_layernorm(hidden)
            routed = layer.pre_feedforward_layernorm_2(hidden)
            router = mx.fast.rms_norm(hidden, None, model.args.rms_norm_eps)
            cases.append({"position": position, "layer": index,
                "hidden_input": append(inputs, previous), "attention_input": append(inputs, attention),
                "hidden": append(outputs, hidden), "dense": append(outputs, dense),
                "routed": append(outputs, routed), "router": append(outputs, router)})
    files = {"norm-inputs.f16": inputs, "norm-reference.f16": outputs,
             "norm-cases.json": (json.dumps(cases, sort_keys=True, indent=2) + "\n").encode()}
    for name, data in files.items():
        path = args.trace / name
        if path.exists():
            raise FileExistsError(f"refusing to replace frozen witness {path}")
        path.write_bytes(data)
    print(json.dumps({"cases": len(cases), "dependencies": reference.VERSIONS,
        "reference_source_sha256": reference.GEMMA_SOURCE_SHA256,
        "trace_sha256": {n: hashlib.sha256((args.trace / n).read_bytes()).hexdigest() for n in ["trace.json", "native.f32"]},
        "sha256": {n: hashlib.sha256(data).hexdigest() for n, data in files.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
