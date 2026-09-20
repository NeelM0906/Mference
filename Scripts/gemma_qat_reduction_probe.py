# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx-lm==0.31.3", "mlx==0.32.2", "mlx-metal==0.32.2", "numpy==2.5.3"]
# ///
"""Isolate source expert-weighting/reduction on installed-weight projections."""
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
    metadata = json.loads((args.trace / "trace.json").read_text())
    entries = {(e["position"], e["layer"], e["stage"]): e for e in metadata["entries"]}
    raw = np.memmap(args.trace / "native.f32", mode="r", dtype="<f4")
    corpus = json.loads((args.native_dump / "meta.json").read_text())
    item = next(x for x in corpus["items"] if x["name"] == metadata["name"])
    mx.set_cache_limit(128 * 1024 * 1024)
    reader = reference.InstalledWeights(args.install)

    def snapshot(stage):
        e = entries[0, 0, stage]
        return raw[e["offset"] // 4:e["offset"] // 4 + e["count"]].astype(np.float16)

    activations = snapshot("routed_activations").reshape(8, -1)
    weights = snapshot("routing_weights")
    downs = []
    for rank, expert in enumerate(item["routes"][0][0]):
        projection = reader.expert(0, expert)["down"]
        down = projection(mx.array(activations[rank])[None, None, :])
        mx.eval(down)
        downs.append(down)
    partials = np.stack([np.asarray(x).reshape(-1) for x in downs])
    products = (partials * weights[:, None]).astype(np.float16)
    # Native rank is descending score; the source argpartition returns the
    # same selected set in stable ascending order. Preserve ties by expert ID.
    ids = np.array(item["routes"][0][0])
    scores = snapshot("router_logits")[ids]
    order = np.lexsort((ids, scores))
    source = (mx.stack([downs[i] for i in order], axis=-2)
              * mx.array(weights[order])[:, None]).sum(axis=-2)
    mx.eval(source)
    expected = np.asarray(source).reshape(-1)
    sequential = np.zeros(partials.shape[1], dtype=np.float16)
    for rank in order:
        sequential = (sequential + products[rank]).astype(np.float16)
    float_sum = products.astype(np.float32).sum(axis=0).astype(np.float16)
    result = {"position": 0, "layer": 0, "native_ids": ids.tolist(),
              "source_order_in_native_slots": order.tolist(), "values": int(expected.size),
              "half_sequential_mismatches": int(np.count_nonzero(expected != sequential)),
              "float_sum_mismatches": int(np.count_nonzero(expected != float_sum)),
              "native_output_mismatches": int(np.count_nonzero(expected != snapshot("routed_output"))),
              "files": {}}
    # Recheck the earlier synthetic boundary fixture's sum assumption against
    # the actual source operation; this does not alter the full-model oracle.
    factors = 1 + np.arange(32) / 32
    fixture_partials = (factors[None, :] * (1 + 3 * (np.arange(8) + 1)[:, None] / 4096)).astype(np.float16)
    result["boundary_fixture"] = []
    for values in [[0.462890625, 0, 0, 0, 0, 0, 0, 0],
                   [0.462890625, 0.257080078125, 0.0535888671875, 0.05419921875,
                    0.046661376953125, 0.04547119140625, 0.0396728515625, 0.03839111328125]]:
        fixture_weights = np.array(values, dtype=np.float16)
        fixture_source = (mx.array(fixture_partials[::-1].copy()) * mx.array(fixture_weights[::-1].copy())[:, None]).sum(0)
        mx.eval(fixture_source)
        result["boundary_fixture"].append(np.asarray(fixture_source).astype(float).tolist())
    for name, values in [("reduction-partials.f16", partials), ("reduction-weights.f16", weights),
                         ("reduction-reference.f16", expected)]:
        data = values.astype("<f2").tobytes()
        (args.trace / name).write_bytes(data)
        result["files"][name] = hashlib.sha256(data).hexdigest()
    (args.trace / "reduction-probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    assert result["half_sequential_mismatches"] == 0


if __name__ == "__main__":
    main()
