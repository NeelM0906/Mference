# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx-lm==0.31.3", "mlx==0.32.2", "mlx-metal==0.32.2", "numpy==2.5.3"]
# ///
"""Diagnose the frozen QAT numerical failure; does not change its oracle/limits.

Run alone after the native trace test exits:
  uv run Scripts/gemma_qat_trace.py MODEL.gturbo TRACE_DIRECTORY
"""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import gemma_qat_mlx_reference as reference


class Observed(nn.Module):
    def __init__(self, target, before=None, after=None):
        super().__init__()
        self.target, self.before, self.after = target, before, after
        if hasattr(target, "layer_type"):
            self.layer_type = target.layer_type

    def __call__(self, *args, **kwargs):
        if self.before:
            self.before(args[0])
        value = self.target(*args, **kwargs)
        if self.after:
            self.after(value)
        return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("install", type=Path)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    metadata = json.loads((args.trace / "trace.json").read_text())
    native = np.memmap(args.trace / "native.f32", mode="r", dtype="<f4")
    mx.set_cache_limit(128 * 1024 * 1024)
    reader = reference.InstalledWeights(args.install)
    model, _ = reference.installed_model(reader, json.loads((args.install / "tokenizer/config.json").read_text()))
    snapshots = {}
    position = 0

    def capture(layer, stage, value):
        mx.eval(value)
        snapshots[position, layer, stage] = np.asarray(value.astype(mx.float32)).reshape(-1).copy()

    def after(layer, stage):
        return lambda value: capture(layer, stage, value)

    for layer_id, layer in enumerate(model.model.layers):
        layer.input_layernorm = Observed(layer.input_layernorm,
            before=after(layer_id, "layer_input"), after=after(layer_id, "input_norm"))
        layer.self_attn.o_proj = Observed(layer.self_attn.o_proj,
            before=after(layer_id, "attention"), after=after(layer_id, "attention_projection"))
        layer.self_attn.rope = Observed(layer.self_attn.rope,
            after=lambda value, i=layer_id: capture(i, "query", value)
                if value.shape[1] == model.args.num_attention_heads else None)
        layer.pre_feedforward_layernorm = Observed(layer.pre_feedforward_layernorm,
            before=after(layer_id, "post_attention"), after=after(layer_id, "dense_input"))
        layer.pre_feedforward_layernorm_2 = Observed(layer.pre_feedforward_layernorm_2,
            after=after(layer_id, "routed_input"))
        layer.post_feedforward_layernorm_1 = Observed(layer.post_feedforward_layernorm_1,
            after=after(layer_id, "shared_output"))
        layer.router.proj = Observed(layer.router.proj, after=after(layer_id, "router_logits"))
        layer.router = Observed(layer.router,
            before=lambda value, i=layer_id: capture(i, "router_input", mx.fast.rms_norm(value, None, model.args.rms_norm_eps)))
        layer.experts = Observed(layer.experts, after=after(layer_id, "routed_output"))
        model.model.layers[layer_id] = Observed(layer,
            after=lambda value, i=layer_id: capture(i, "layer_output", value[0]))

    cache = model.make_cache()
    logits = []
    for position, token in enumerate(metadata["sequence"]):
        value = model(mx.array([[token]], dtype=mx.int32), cache=cache)
        mx.eval(value)
        logits.append(np.asarray(value).reshape(-1).copy())
    np.stack(logits).astype("<f2").tofile(args.trace / "reference-logits.f16")
    results = []
    with (args.trace / "reference.f32").open("wb") as output:
        entries = []
        for entry in metadata["entries"]:
            key = entry["position"], entry["layer"], entry["stage"]
            if key not in snapshots:
                continue
            actual = native[entry["offset"] // 4:entry["offset"] // 4 + entry["count"]].astype(np.float64)
            expected = snapshots[key]
            entries.append({**entry, "offset": output.tell()})
            output.write(expected.astype("<f4").tobytes())
            expected = expected.astype(np.float64)
            error = actual - expected
            result = {"position": key[0], "layer": key[1], "stage": key[2],
                      "relative_l2": float(np.linalg.norm(error) / max(np.linalg.norm(expected), 1e-12)),
                      "max_absolute": float(np.abs(error).max())}
            results.append(result)
        for key, value in snapshots.items():
            if key[2] == "layer_input":
                entries.append({"position": key[0], "layer": key[1], "stage": key[2],
                                "offset": output.tell(), "count": len(value)})
                output.write(value.astype("<f4").tobytes())
        (args.trace / "reference-trace.json").write_text(json.dumps({"entries": entries}))
    (args.trace / "comparison.json").write_text(json.dumps(results, indent=2))
    for row in results:
        if row["stage"] == "layer_output" and row["relative_l2"] > 0.005:
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
