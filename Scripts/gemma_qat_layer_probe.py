# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx-lm==0.31.3", "mlx==0.32.2", "mlx-metal==0.32.2", "numpy==2.5.3"]
# ///
"""Compare source layer operations on identical native trace inputs.

Diagnostic only; never changes the frozen full-model oracle or tolerances.
Run alone after native tracing: uv run Scripts/gemma_qat_layer_probe.py INSTALL TRACE
"""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

import gemma_qat_mlx_reference as reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("install", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--positions", type=int, nargs="+", default=[0, 2, 4])
    parser.add_argument("--native-dump", type=Path,
                        help="Also compare routed operations on recorded native experts and weights")
    args = parser.parse_args()
    meta = json.loads((args.trace / "trace.json").read_text())
    raw = np.memmap(args.trace / "native.f32", mode="r", dtype="<f4")
    entries = {(e["position"], e["layer"], e["stage"]): e for e in meta["entries"]}
    mx.set_cache_limit(128 * 1024 * 1024)
    reader = reference.InstalledWeights(args.install)
    model, _ = reference.installed_model(reader, json.loads((args.install / "tokenizer/config.json").read_text()))
    routes = None
    if args.native_dump:
        corpus = json.loads((args.native_dump / "meta.json").read_text())
        item = next(x for x in corpus["items"] if x["name"] == meta["name"])
        assert item["sequence"][:len(meta["sequence"])] == meta["sequence"]
        routes = item["routes"]

    def snapshot(position, layer, stage):
        e = entries[position, layer, stage]
        return raw[e["offset"] // 4:e["offset"] // 4 + e["count"]]

    def tensor(position, layer, stage):
        return mx.array(snapshot(position, layer, stage), dtype=mx.float16)[None, None, :]

    results = []
    for position in args.positions:
        for index, layer in enumerate(model.layers):
            def check(stage, expected, label=None):
                mx.eval(expected)
                a = np.asarray(expected).astype(np.float64).reshape(-1)
                b = snapshot(position, index, stage).astype(np.float64)
                result = {"position": position, "layer": index, "stage": label or stage,
                          "relative_l2": float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12)),
                          "max_absolute": float(np.abs(a - b).max()),
                          "equal": int(np.count_nonzero(a == b)), "count": a.size}
                results.append(result)

            previous = tensor(position, index - 1, "layer_output") if index else (
                model.model.embed_tokens(mx.array([[meta["sequence"][position]]])) * model.model.embed_scale)
            check("input_norm", layer.input_layernorm(previous))
            queries = layer.self_attn.q_proj(tensor(position, index, "input_norm"))
            queries = queries.reshape(1, 1, layer.self_attn.n_heads, -1).transpose(0, 2, 1, 3)
            check("query", layer.self_attn.rope(layer.self_attn.q_norm(queries), offset=position))
            check("attention_projection", layer.self_attn.o_proj(tensor(position, index, "attention")))
            check("post_attention", previous + layer.post_attention_layernorm(tensor(position, index, "attention_projection")))
            h = tensor(position, index, "post_attention")
            check("dense_input", layer.pre_feedforward_layernorm(h))
            check("routed_input", layer.pre_feedforward_layernorm_2(h))
            check("router_input", mx.fast.rms_norm(h, None, model.args.rms_norm_eps))
            router_scaled = mx.fast.rms_norm(h, layer.router.scale * layer.router._root_size, layer.router.eps)
            check("router_logits", layer.router.proj(router_scaled))
            check("shared_output", layer.post_feedforward_layernorm_1(layer.mlp(tensor(position, index, "dense_input"))))
            if (position, index, "shared_activations") in entries:
                dense = tensor(position, index, "dense_input")
                check("shared_activations", reference.gemma.geglu(layer.mlp.gate_proj(dense), layer.mlp.up_proj(dense)))
                check("shared_output", layer.post_feedforward_layernorm_1(
                    layer.mlp.down_proj(tensor(position, index, "shared_activations"))), label="shared_down_norm")
            h1, h2 = tensor(position, index, "shared_output"), tensor(position, index, "routed_output")
            tail = layer.post_feedforward_layernorm(h1 + layer.post_feedforward_layernorm_2(h2))
            check("layer_output", (h + tail) * layer.layer_scalar, label="layer_tail")
            if routes is not None:
                acts, downs = [], []
                routed = tensor(position, index, "routed_input")
                native_acts = snapshot(position, index, "routed_activations").reshape(8, -1)
                for rank, expert in enumerate(routes[position][index]):
                    projections = reader.expert(index, expert)
                    act = reference.gemma.geglu(projections["gate"](routed), projections["up"](routed))
                    down = projections["down"](mx.array(native_acts[rank], dtype=mx.float16)[None, None, :])
                    mx.eval(act, down)
                    acts.append(act)
                    downs.append(down)
                check("routed_activations", mx.stack(acts, axis=-2))
                weights = mx.array(snapshot(position, index, "routing_weights"), dtype=mx.float16)
                check("routed_output", (mx.stack(downs, axis=-2) * weights[:, None]).sum(axis=-2),
                      label="routed_down_native_order_sum")
                ids = np.array(routes[position][index])
                order = np.lexsort((ids, snapshot(position, index, "router_logits")[ids]))
                check("routed_output", (mx.stack([downs[i] for i in order], axis=-2)
                      * weights[mx.array(order)][:, None]).sum(axis=-2),
                      label="routed_down_source_order_sum")
    (args.trace / "layer-probe.json").write_text(json.dumps(results, indent=2) + "\n")
    for stage in dict.fromkeys(r["stage"] for r in results):
        rows = [r for r in results if r["stage"] == stage]
        worst = max(rows, key=lambda r: r["relative_l2"])
        print(json.dumps({"stage": stage, "exact_values": sum(r["equal"] for r in rows),
                          "values": sum(r["count"] for r in rows), "worst": worst}), flush=True)


if __name__ == "__main__":
    main()
