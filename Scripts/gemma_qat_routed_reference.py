# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx-lm==0.31.3", "mlx==0.32.2", "mlx-metal==0.32.2", "numpy==2.5.3"]
# ///
"""Freeze source routed-FFN results for exact current native trace inputs."""
import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

import gemma_qat_mlx_reference as reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("install", type=Path)
    parser.add_argument("native_dump", type=Path)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    meta = json.loads((args.trace / "trace.json").read_text())
    entries = {(e["position"], e["layer"], e["stage"]): e for e in meta["entries"]}
    raw = np.memmap(args.trace / "native.f32", mode="r", dtype="<f4")
    corpus = json.loads((args.native_dump / "meta.json").read_text())
    item = next(x for x in corpus["items"] if x["name"] == meta["name"])
    assert item["sequence"][:len(meta["sequence"])] == meta["sequence"]
    mx.set_cache_limit(128 * 1024 * 1024)
    reader = reference.InstalledWeights(args.install)
    inputs, outputs, cases = bytearray(), bytearray(), []

    def append(target, values):
        array = np.asarray(values).astype("<f2").reshape(-1)
        spec = {"offset": len(target), "count": int(array.size)}
        target.extend(array.tobytes())
        return spec

    for position, layer in [(0, 0), (2, 17), (4, 29)]:
        def snapshot(stage):
            e = entries[position, layer, stage]
            return raw[e["offset"] // 4:e["offset"] // 4 + e["count"]].astype(np.float16)

        ids = item["routes"][position][layer]
        x = snapshot("routed_input")
        acts = snapshot("routed_activations").reshape(8, -1)
        weights = snapshot("routing_weights")
        order = np.lexsort((ids, snapshot("router_logits")[ids]))
        assert order.tolist() == list(range(7, -1, -1))
        source_acts, native_act_downs, source_act_downs = [], [], []
        for rank, expert in enumerate(ids):
            projections = reader.expert(layer, expert)
            value = mx.array(x)[None, None, :]
            act = reference.gemma.geglu(projections["gate"](value), projections["up"](value))
            native_down = projections["down"](mx.array(acts[rank])[None, None, :])
            source_down = projections["down"](act)
            mx.eval(act, native_down, source_down)
            source_acts.append(act)
            native_act_downs.append(native_down)
            source_act_downs.append(source_down)

        def reduce(downs):
            value = (mx.stack([downs[i] for i in order], axis=-2)
                     * mx.array(weights[order])[:, None]).sum(axis=-2)
            mx.eval(value)
            return value

        act_result = mx.stack(source_acts, axis=-2)
        down_result, full_result = reduce(native_act_downs), reduce(source_act_downs)
        mx.eval(act_result)
        cases.append({"position": position, "layer": layer, "experts": ids,
            "input": append(inputs, x), "native_activations": append(inputs, acts),
            "weights": append(inputs, weights), "expected_activations": append(outputs, act_result),
            "expected_down": append(outputs, down_result), "expected_full": append(outputs, full_result),
            "native_activation_mismatches": int(np.count_nonzero(np.asarray(act_result).reshape(8, -1) != acts)),
            "native_output_mismatches": int(np.count_nonzero(np.asarray(full_result).reshape(-1) != snapshot("routed_output")))})
    files = {"routed-inputs.f16": inputs, "routed-reference.f16": outputs,
             "routed-cases.json": (json.dumps(cases, sort_keys=True, indent=2) + "\n").encode()}
    for name, data in files.items():
        (args.trace / name).write_bytes(data)
    print(json.dumps({"cases": cases, "maximum_expert_read_bytes": reader.maximum_expert_read,
                      "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
