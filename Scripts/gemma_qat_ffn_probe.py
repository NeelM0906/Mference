# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx-lm==0.31.3", "mlx==0.32.2", "mlx-metal==0.32.2", "numpy==2.5.3"]
# ///
"""Compare routed FFN operations on identical native inputs and selected experts.

Diagnostic only: the full teacher-forced gate and its limits remain unchanged.
Run alone after the native trace exits; source reads stay bounded per expert.
"""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

import gemma_qat_mlx_reference as reference


def metrics(expected, actual):
    a, b = np.asarray(expected, np.float64), np.asarray(actual, np.float64)
    return {"relative_l2": float(np.linalg.norm(a-b)/max(np.linalg.norm(a), 1e-12)),
            "max_absolute": float(np.max(np.abs(a-b))),
            "equal": int(np.count_nonzero(a == b)), "count": a.size}


def cpu_projection(projection, x, source_bias_sum=False):
    packed = np.asarray(projection.weight)
    q = ((packed[..., None] >> (np.arange(8, dtype=np.uint32) * 4)) & 15).reshape(packed.shape[0], -1)
    s = np.asarray(projection.scales, np.float64).repeat(32, axis=-1)
    b = np.asarray(projection.biases, np.float64).repeat(32, axis=-1)
    result = (q * s + b) @ x.astype(np.float64)
    if source_bias_sum:
        quad = x.reshape(-1, 4).astype(np.float16)
        source_sum = ((quad[:, 0] + quad[:, 1]) + quad[:, 2]) + quad[:, 3]
        correction = source_sum.astype(np.float64) - quad.astype(np.float64).sum(-1)
        result += b[:, ::4] @ correction
    return result


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("install", type=Path)
    parser.add_argument("native_dump", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--first-only", action="store_true")
    parser.add_argument("--router-only", action="store_true")
    args = parser.parse_args()
    reader = reference.InstalledWeights(args.install)
    mx.set_cache_limit(128 * 1024 * 1024)
    metadata = json.loads((args.trace / "trace.json").read_text())
    raw = np.memmap(args.trace / "native.f32", mode="r", dtype="<f4")
    corpus = json.loads((args.native_dump / "meta.json").read_text())
    item_name = metadata.get("name", "capital-scalar")
    routes = next(x for x in corpus["items"] if x["name"] == item_name)["routes"]

    def snapshot(position, layer, stage):
        e = next(e for e in metadata["entries"]
                 if (e["position"], e["layer"], e["stage"]) == (position, layer, stage))
        return raw[e["offset"]//4:e["offset"]//4+e["count"]].copy()

    if args.router_only:
        rows = []
        for entry in metadata["entries"]:
            if entry["stage"] != "router_logits":
                continue
            position, layer = entry["position"], entry["layer"]
            scores = mx.array(snapshot(position, layer, "router_logits"), dtype=mx.float16)
            ids = mx.argpartition(scores, kth=-8)[-8:]
            prefix = f"model.layers.{layer}"
            if prefix + ".router.per_expert_scale" not in reader.entries:
                prefix = "language_model." + prefix
            gains = reader.resident(prefix + ".router.per_expert_scale")
            weights = mx.softmax(scores[ids]) * gains[ids]
            mx.eval(ids, weights)
            source_ids = np.asarray(ids).tolist()
            by_id = dict(zip(source_ids, np.asarray(weights).tolist(), strict=True))
            native_ids = routes[position][layer]
            expected = np.array([by_id.get(i, 0) for i in native_ids], dtype=np.float16)
            actual = snapshot(position, layer, "routing_weights")
            row = {"position": position, "layer": layer,
                   "source_ids": source_ids, "native_ids": native_ids,
                   "same_selected_set": set(source_ids) == set(native_ids),
                   "weights": metrics(expected, actual)}
            rows.append(row)
        summary = {"rows": len(rows),
                   "selection_mismatches": sum(not r["same_selected_set"] for r in rows),
                   "weight_mismatches": sum(r["weights"]["count"] - r["weights"]["equal"] for r in rows),
                   "maximum_expert_read_bytes": reader.maximum_expert_read}
        summary["passed"] = bool(rows) and summary["selection_mismatches"] == 0 and summary["weight_mismatches"] == 0
        (args.trace / "router-probe.json").write_text(json.dumps({"rows": rows, "summary": summary}, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
        if not summary["passed"]:
            raise SystemExit(1)
        return

    rows = []
    for position, layer in ([(0, 0)] if args.first_only else [(0, 0), (3, 2)]):
        x = snapshot(position, layer, "routed_input").astype(np.float16)
        actual_acts = snapshot(position, layer, "routed_activations").reshape(8, -1)
        actual_output = snapshot(position, layer, "routed_output")
        ids = routes[position][layer]
        prefix = f"model.layers.{layer}"
        if prefix + ".router.per_expert_scale" not in reader.entries:
            prefix = "language_model." + prefix
        gain = reader.resident(prefix + ".router.per_expert_scale")
        scores = mx.array(snapshot(position, layer, "router_logits")[ids], dtype=mx.float16)
        weights = mx.softmax(scores) * gain[mx.array(ids)]
        # Keep the original native-order diagnostic above. The real source
        # argpartition uses ascending sort, which changes half-softmax sums.
        ascending = mx.argsort(scores)
        ascending_weights = mx.softmax(scores[ascending]) * gain[mx.array(ids)[ascending]]
        mx.eval(ascending, ascending_weights)
        source_order_weights = np.empty(8, dtype=np.float16)
        source_order_weights[np.asarray(ascending)] = np.asarray(ascending_weights)
        observed_weights = next((e for e in metadata["entries"]
            if (e["position"], e["layer"], e["stage"]) == (position, layer, "routing_weights")), None)
        native_weights = snapshot(position, layer, "routing_weights").astype(np.float16) if observed_weights else None
        source_down = []
        unrounded_down = []
        source_bias_down = []
        for rank, expert in enumerate(ids):
            q = reader.expert(layer, expert)
            gx, ux = q["gate"](mx.array(x)), q["up"](mx.array(x))
            act = reference.gemma.geglu(gx, ux)
            # Hold native activation inputs fixed for a separate phase-2 probe.
            native_act = actual_acts[rank].astype(np.float16)
            down = q["down"](mx.array(native_act))
            mx.eval(act, down, weights)
            source_down.append(down)
            g, u = cpu_projection(q["gate"], x), cpu_projection(q["up"], x)
            fused = (gelu(g) * u).astype(np.float16)
            staged = (gelu(g.astype(np.float16).astype(np.float64))
                      * u.astype(np.float16).astype(np.float64)).astype(np.float16)
            half_g, half_u = np.asarray(gx), np.asarray(ux)
            f32_pointwise = (gelu(half_g.astype(np.float64)) * half_u.astype(np.float64)).astype(np.float16)
            # Explicit activation-type operations, mirroring the source graph.
            cube = (half_g.astype(np.float64)**3).astype(np.float16)
            angle = np.float16(np.sqrt(2/np.pi)) * (half_g + np.float16(0.044715) * cube)
            half_pointwise = np.float16(0.5) * half_g * (np.float16(1) + np.tanh(angle)) * half_u
            bad = np.flatnonzero(~np.isfinite(actual_acts[rank]))
            rows.append({"position": position, "layer": layer, "expert": expert,
                "native_nonfinite_indices": bad.tolist(),
                "source_gate_at_nonfinite": half_g[bad].astype(float).tolist(),
                "source_up_at_nonfinite": half_u[bad].astype(float).tolist(),
                "source_act_at_nonfinite": np.asarray(act)[bad].astype(float).tolist(),
                "cpu_source_bias_sum_vs_source_gate": metrics(np.asarray(gx), cpu_projection(q["gate"], x, True).astype(np.float16)),
                "cpu_source_bias_sum_vs_source_up": metrics(np.asarray(ux), cpu_projection(q["up"], x, True).astype(np.float16)),
                "cpu_half_projection_vs_source_gate": metrics(np.asarray(gx), g.astype(np.float16)),
                "cpu_half_projection_vs_source_up": metrics(np.asarray(ux), u.astype(np.float16)),
                "f32_pointwise_vs_source_activation": metrics(np.asarray(act), f32_pointwise),
                "half_pointwise_vs_source_activation": metrics(np.asarray(act), half_pointwise),
                "source_vs_native_activation": metrics(np.asarray(act), actual_acts[rank]),
                "fused_cpu_vs_native_activation": metrics(fused, actual_acts[rank]),
                "staged_cpu_vs_source_activation": metrics(np.asarray(act), staged)})
            unrounded_down.append(cpu_projection(q["down"], native_act))
            if native_weights is not None:
                source_bias_down.append(cpu_projection(q["down"], native_act, True))
        output = (weights[:, None] * mx.stack(source_down)).sum(0)
        mx.eval(output)
        w = np.asarray(weights, np.float64)
        fused = (w[:, None] * np.stack(unrounded_down)).sum(0).astype(np.float16)
        rows.append({"position": position, "layer": layer,
            "source_vs_native_output_same_activations": metrics(np.asarray(output), actual_output),
            "fused_cpu_vs_native_output_same_activations": metrics(fused, actual_output)})
        if native_weights is not None:
            down_half = np.stack([np.asarray(value) for value in source_down])
            source_product = (native_weights[:, None] * down_half).astype(np.float16)
            source_result = source_product.astype(np.float32).sum(0).astype(np.float16)
            unrounded_product = (native_weights.astype(np.float32)[:, None]
                                 * down_half.astype(np.float32)).sum(0).astype(np.float16)
            unrounded_projection = (native_weights.astype(np.float64)[:, None]
                                    * np.stack(source_bias_down)).sum(0).astype(np.float16)
            rows.append({"position": position, "layer": layer,
                "source_vs_native_routing_weights": metrics(np.asarray(weights), native_weights),
                "source_routing_weights": np.asarray(weights).astype(float).tolist(),
                "native_routing_weights": native_weights.astype(float).tolist(),
                "ascending_source_routing_weights": source_order_weights.astype(float).tolist(),
                "ascending_source_vs_native_routing_weights": metrics(source_order_weights, native_weights),
                "source_vs_native_output_same_activations_and_weights": metrics(source_result, actual_output),
                "unrounded_product_vs_native_output": metrics(unrounded_product, actual_output),
                "unrounded_projection_and_product_vs_native_output": metrics(unrounded_projection, actual_output)})
    result = {"results": rows, "maximum_expert_read_bytes": reader.maximum_expert_read}
    (args.trace / "ffn-probe.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
