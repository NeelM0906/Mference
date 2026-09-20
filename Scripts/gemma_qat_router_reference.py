# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx-lm==0.31.3", "mlx==0.32.2", "mlx-metal==0.32.2", "numpy==2.5.3"]
# ///
"""Freeze source router stages for the unchanged v12 long-prompt trace.

Run alone: uv run Scripts/gemma_qat_router_reference.py INSTALL TRACE
Reads resident weights and saved inputs only; never reads expert weights.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("install", type=Path)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    assert {n: importlib.metadata.version(n) for n in reference.VERSIONS} == reference.VERSIONS
    assert hashlib.sha256(Path(inspect.getfile(reference.gemma)).read_bytes()).hexdigest() == reference.GEMMA_SOURCE_SHA256
    hashes = {"trace.json": "bb9dca0f7ec69b8f4d2c0365e5dd5a2ca1330aa53d52913ea831d87b533fdcdb",
              "native.f32": "a5a13fe619c61f81e836139dcf777ec303c056f66b0892d05aead98fa440ae5a"}
    for name, digest in hashes.items():
        assert hashlib.sha256((args.trace / name).read_bytes()).hexdigest() == digest
    for name in ("router-reference.f16", "router-cases.json", "router-witness.json"):
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
        e = entries[position, layer, stage]
        return raw[e["offset"] // 4:e["offset"] // 4 + e["count"]].astype(np.float16)

    for layer, block in enumerate(model.layers):
        router = block.router
        effective = router.scale * router._root_size
        for position in range(len(meta["sequence"])):
            x = mx.array(snapshot(position, layer, "post_attention"))[None, None, :]
            normalized = mx.array(snapshot(position, layer, "router_input"))[None, None, :]
            scaled = mx.fast.rms_norm(x, effective, router.eps)
            decomposed = normalized * effective
            mx.eval(scaled, decomposed)
            assert np.array_equal(np.asarray(scaled), np.asarray(decomposed)), (position, layer, "scaled input")
            scores = router.proj(scaled)
            ids, weights = router(x)
            mx.eval(scores, ids, weights)
            # Native slots descend by score/ID; source slots ascend. Preserve
            # the upstream computed weights and only reverse their slot order.
            indices = np.asarray(ids).reshape(-1)[::-1].copy()
            gains = np.asarray(weights).reshape(-1)[::-1].copy()
            case = {"position": position, "layer": layer, "indices": indices.tolist(), "values": {}}
            for name, value in {"input": normalized, "scaled_input": scaled,
                                "logits": scores, "weights": gains}.items():
                data = np.asarray(value).astype("<f2").tobytes()
                case["values"][name] = {"offset": len(payload), "count": len(data) // 2}
                payload.extend(data)
            cases.append(case)
            for stage, actual in (("router_logits", np.asarray(scores).reshape(-1)),
                                  ("routing_weights", gains)):
                native = snapshot(position, layer, stage)
                differences.append({"position": position, "layer": layer, "stage": stage,
                    "different": int(np.count_nonzero(actual != native)), "count": actual.size,
                    "max_absolute": float(np.abs(actual.astype(np.float64) - native.astype(np.float64)).max())})
    files = {"router-reference.f16": payload,
             "router-cases.json": (json.dumps(cases, sort_keys=True, indent=2) + "\n").encode()}
    for name, data in files.items():
        (args.trace / name).write_bytes(data)
    summary = {"cases": len(cases), "dependencies": reference.VERSIONS,
        "source_sha256": reference.GEMMA_SOURCE_SHA256, "inputs": hashes,
        "sha256": {n: hashlib.sha256(data).hexdigest() for n, data in files.items()},
        "native_differences": differences}
    (args.trace / "router-witness.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "native_differences"}), flush=True)
    for stage in ("router_logits", "routing_weights"):
        print(stage, sum(r["different"] for r in differences if r["stage"] == stage), flush=True)


if __name__ == "__main__":
    main()
