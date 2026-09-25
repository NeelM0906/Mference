# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx-lm==0.31.3", "mlx==0.32.2", "mlx-metal==0.32.2", "numpy==2.5.3"]
# ///
"""Independent, bounded Gemma QAT execution from an existing .gturbo install.

Run alone, after the native env-gated tests have exited and after the usual
model-process/memory checks:
  uv run Scripts/gemma_qat_mlx_reference.py MODEL.gturbo NATIVE_DUMP_DIRECTORY

No checkpoint download or output weights. Uses MLX-LM 0.31.3 Gemma's unmodified
layer, attention, router, norm, residual and cache math, with FP16 activation
compute. Already-converted BF16 parameters are cast to FP16; no second
sanitize or quantization pass. Only SwitchGLU's storage access is adapted: each
selected expert is read and evaluated separately, then the upstream Experts
class performs its weighting/reduction. Never constructs an all-expert array.

Native dumps are pre-softcap FP16. Both sides receive the same Float64 final
softcap before the full-vocabulary comparison. This is an execution check of
these installed quantized weights, not a QAT quality or loop-reduction claim.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import struct

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import gemma4_text as gemma
import numpy as np

MODEL_ID = "gemma-4-26b-a4b-it-qat-q4_0-mlx-aligned"
SOURCE = "7dbbeef0345505798abcf0ac54434116a48c2f1e7aad828071c17a7a871adfe7"
GEMMA_SOURCE_SHA256 = "77f46bc3f162a0b9513157dade4be2c381d4df3295262c69034a53d46111370f"
VERSIONS = {"mlx-lm": "0.31.3", "mlx": "0.32.2", "mlx-metal": "0.32.2", "numpy": "2.5.3"}
# Frozen before any native/reference model comparison. FP16 reduction order
# permits small logit differences; a high-margin routing/winner change fails.
LIMITS = {"relative_l2": 0.025, "rmse": 0.15, "kl_nats": 0.01,
          "winner_reference_gap": 0.15, "routing_boundary_gap": 0.02,
          "routing_slot_agreement": 0.98}


def module(cls, **fields):
    """Bind installed arrays without initializing giant random parameters."""
    value = cls.__new__(cls)
    nn.Module.__init__(value)
    for name, field in fields.items():
        setattr(value, name, field)
    return value


def bf16(raw, shape):
    bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    return mx.array((bits << 16).view(np.float32).reshape(shape)).astype(mx.float16)


class InstalledWeights:
    def __init__(self, path):
        self.path = path
        self.payload = np.memmap(path / "model_weights.bin", dtype=np.uint8, mode="r")
        index_size, resident_size, count = struct.unpack_from("<QQQ", self.payload)
        if index_size + resident_size != len(self.payload) or 24 + count * 72 > index_size:
            raise ValueError("invalid resident index")
        self.index_size = index_size
        self.entries = {}
        for i in range(count):
            fields = struct.unpack_from("<I H B x Q Q 4I Q Q Q Q", self.payload, 24 + i * 72)
            no, ns = fields[:2]
            if no + ns > index_size:
                raise ValueError("invalid tensor name span")
            name = bytes(self.payload[no:no + ns]).decode()
            if name in self.entries:
                raise ValueError("duplicate tensor name")
            self.entries[name] = fields[2:]
        self.layout = json.loads((path / "packed_experts/layout.json").read_text())
        self.maximum_expert_read = 0

    def span(self, offset, size):
        if offset < self.index_size or size <= 0 or offset + size > len(self.payload):
            raise ValueError("resident tensor out of bounds")
        return self.payload[offset:offset + size]

    def resident(self, name, embedding=False):
        dtype, offset, size, *tail = self.entries[name]
        shape = tuple(d for d in tail[:4] if d)
        so, ss, bo, bs = tail[4:]
        if dtype == 1:
            if size != np.prod(shape) * 2 or any((so, ss, bo, bs)):
                raise ValueError(f"invalid BF16 tensor {name}")
            return bf16(self.span(offset, size), shape)
        rows, cols = shape
        if dtype != 0 or cols % 32 or size != rows * cols // 2 or ss != bs or ss != rows * cols // 16:
            raise ValueError(f"invalid group-32 tensor {name}")
        weight = mx.array(np.frombuffer(self.span(offset, size), dtype="<u4").reshape(rows, cols // 8))
        fields = dict(weight=weight, scales=bf16(self.span(so, ss), (rows, cols // 32)),
                      biases=bf16(self.span(bo, bs), (rows, cols // 32)),
                      group_size=32, bits=4, mode="affine")
        if embedding:
            fields.update(num_embeddings=rows, dims=cols)
        return module(nn.QuantizedEmbedding if embedding else nn.QuantizedLinear, **fields)

    def expert(self, layer, expert):
        info = self.layout["layers"][layer]
        entry = next(e for e in info["experts"] if e["expert"] == expert)
        if entry["size"] > 4 * 1024 * 1024:
            raise ValueError("unexpected expert size")
        with (self.path / "packed_experts" / info["file"]).open("rb") as handle:
            handle.seek(entry["offset"])
            blob = handle.read(entry["size"])
        if len(blob) != entry["size"]:
            raise ValueError("short expert read")
        self.maximum_expert_read = max(self.maximum_expert_read, len(blob))
        result = {}
        for role in ("gate", "up", "down"):
            values = {}
            for suffix, key in (("", "weight"), ("_scales", "scales"), ("_biases", "biases")):
                spec = entry["tensors"][role + suffix]
                start, size = spec["offset"], spec["size"]
                raw = memoryview(blob)[start:start + size]
                if len(raw) != size:
                    raise ValueError("expert component out of bounds")
                rows, cols = spec["shape"]
                if suffix:
                    if spec["dtype"] != "BF16" or size != rows * cols * 2:
                        raise ValueError("invalid expert companion")
                    values[key] = bf16(raw, (rows, cols))
                else:
                    if spec["dtype"] != "U32" or spec["bits"] != 4 or size != rows * cols // 2:
                        raise ValueError("invalid expert weights")
                    values[key] = mx.array(np.frombuffer(raw, dtype="<u4").reshape(rows, cols // 8))
            result[role] = module(nn.QuantizedLinear, **values, group_size=32, bits=4, mode="affine")
        return result


class BoundedSwitchGLU(nn.Module):
    def __init__(self, reader, layer):
        super().__init__()
        self.reader, self.layer = reader, layer

    def __call__(self, x, indices):
        # Scalar teacher forcing keeps source reads bounded even for long
        # contexts. Native chunked results compare at the same token positions.
        if x.shape[:2] != (1, 1):
            raise ValueError("reference expects one teacher-forced token")
        mx.eval(indices)
        outputs = []
        for expert_id in np.asarray(indices).reshape(-1):
            weights = self.reader.expert(self.layer, int(expert_id))
            gate, up = weights["gate"](x), weights["up"](x)
            output = weights["down"](gemma.geglu(gate, up))
            mx.eval(output)  # release this expert's arrays before the next read
            outputs.append(output)
        return mx.stack(outputs, axis=-2)


class RecordedProjection(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
        self.last_scores = None

    def __call__(self, x):
        scores = x @ self.weight.T
        mx.eval(scores)
        self.last_scores = np.asarray(scores.astype(mx.float32)).reshape(-1).copy()
        return scores


def installed_model(reader, config):
    args = gemma.ModelArgs.from_dict(config["text_config"])
    if (args.hidden_size_per_layer_input or args.num_kv_shared_layers or
            not args.enable_moe_block or not args.tie_word_embeddings):
        raise ValueError("checkpoint graph differs from the pinned 26B text graph")
    root = "language_model.model"
    def norm(name):
        return module(nn.RMSNorm, weight=reader.resident(name), eps=args.rms_norm_eps)
    layers = []
    for i in range(args.num_hidden_layers):
        p = f"{root}.layers.{i}"
        attention = gemma.Attention(args, i)
        for role in ("q", "k", "v", "o"):
            name = f"{p}.self_attn.{role}_proj.weight"
            if name in reader.entries:
                setattr(attention, role + "_proj", reader.resident(name))
        attention.q_norm = norm(p + ".self_attn.q_norm.weight")
        attention.k_norm = norm(p + ".self_attn.k_norm.weight")
        mlp = module(gemma.MLP, **{role + "_proj": reader.resident(f"{p}.mlp.{role}_proj.weight")
                                  for role in ("gate", "up", "down")})
        router = gemma.Router(args)
        router.proj = RecordedProjection(reader.resident(p + ".router.proj.weight"))
        router.scale = reader.resident(p + ".router.scale")
        router.per_expert_scale = reader.resident(p + ".router.per_expert_scale")
        layer = module(gemma.DecoderLayer, config=args, layer_idx=i, layer_type=args.layer_types[i],
                       self_attn=attention, mlp=mlp, enable_moe=True, router=router,
                       experts=module(gemma.Experts, switch_glu=BoundedSwitchGLU(reader, i)),
                       hidden_size_per_layer_input=0, per_layer_input_gate=None,
                       per_layer_projection=None, post_per_layer_input_norm=None,
                       layer_scalar=reader.resident(p + ".layer_scalar"))
        for name in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                     "pre_feedforward_layernorm_2", "post_feedforward_layernorm",
                     "post_feedforward_layernorm_1", "post_feedforward_layernorm_2"):
            setattr(layer, name, norm(p + "." + name + ".weight"))
        layers.append(layer)
    trunk = module(gemma.Gemma4TextModel, config=args, vocab_size=args.vocab_size,
                   window_size=args.sliding_window, sliding_window_pattern=args.sliding_window_pattern,
                   num_hidden_layers=args.num_hidden_layers,
                   embed_tokens=reader.resident(root + ".embed_tokens.weight", embedding=True),
                   embed_scale=args.hidden_size ** 0.5, layers=layers, norm=norm(root + ".norm.weight"),
                   hidden_size_per_layer_input=0, previous_kvs=list(range(args.num_hidden_layers)))
    model = module(gemma.Model, args=args, model_type=args.model_type, model=trunk,
                   final_logit_softcapping=None, tie_word_embeddings=True)
    mx.eval(model.parameters())
    return model, args.final_logit_softcapping


def compare(reference, candidate, softcap):
    a, b = np.asarray(reference, np.float64), np.asarray(candidate, np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite logits")
    a, b = np.tanh(a / softcap) * softcap, np.tanh(b / softcap) * softcap
    error = a - b
    lp, lq = a - a.max(), b - b.max()
    lp -= np.log(np.exp(lp).sum())
    lq -= np.log(np.exp(lq).sum())
    metrics = {"relative_l2": float(np.linalg.norm(error) / max(np.linalg.norm(a), 1e-12)),
               "rmse": float(np.sqrt(np.mean(error * error))),
               "kl_nats": float(np.sum(np.exp(lp) * (lp - lq))),
               "winner_reference_gap": float(a.max() - a[b.argmax()]),
               "max_absolute_error": float(np.abs(error).max()),
               "top1_equal": bool(a.argmax() == b.argmax())}
    metrics["passed"] = all(metrics[k] <= LIMITS[k]
                            for k in ("relative_l2", "rmse", "kl_nats", "winner_reference_gap"))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("install", type=Path)
    parser.add_argument("native_dump", type=Path)
    args = parser.parse_args()
    versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    if versions != VERSIONS:
        raise ValueError(f"unexpected reference dependencies: {versions}")
    source_hash = hashlib.sha256(Path(inspect.getfile(gemma)).read_bytes()).hexdigest()
    if source_hash != GEMMA_SOURCE_SHA256:
        raise ValueError("reference Gemma source differs from the pinned implementation")
    manifest_bytes = (args.install / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest["modelID"] != MODEL_ID or manifest["sourceSnapshotHash"] != "sha256:" + SOURCE:
        raise ValueError("not the pinned QAT checkpoint")
    native = json.loads((args.native_dump / "meta.json").read_text())
    if native["manifest_sha256"] != manifest_hash or native["logit_stage"] != "pre_softcap_fp16":
        raise ValueError("native dump identity/stage mismatch")
    mx.set_cache_limit(128 * 1024 * 1024)
    reader = InstalledWeights(args.install)
    model, softcap = installed_model(reader, json.loads((args.install / "tokenizer/config.json").read_text()))
    results = []
    matches = slots = 0
    routing_ok = True
    output_path = args.native_dump / "reference.jsonl"
    with output_path.open("w") as output:
        for item in native["items"]:
            logits = np.memmap(args.native_dump / item["file"], mode="r", dtype="<f2").reshape(-1, native["vocab"])
            positions = {position: row for row, position in enumerate(item["positions"])}
            cache = model.make_cache()
            for position, token in enumerate(item["sequence"]):
                value = model(mx.array([[token]], dtype=mx.int32), cache=cache)[0, -1]
                mx.eval(value)
                if position not in positions:
                    continue
                row = positions[position]
                result = {"name": item["name"], "position": position,
                          **compare(np.asarray(value), logits[row], softcap)}
                if item.get("routes"):
                    routes = item["routes"][row]
                    route_matches = 0
                    max_boundary_gap = 0.0
                    for layer, selected in zip(model.layers, routes, strict=True):
                        scores = layer.router.proj.last_scores
                        top = np.argsort(scores)[-8:]
                        route_matches += len(set(top) & set(selected))
                        boundary_gap = float(max(0, scores[top].min() - scores[selected].min()))
                        max_boundary_gap = max(max_boundary_gap, boundary_gap)
                    matches += route_matches
                    slots += len(routes) * 8
                    routing_ok &= max_boundary_gap <= LIMITS["routing_boundary_gap"]
                    result.update(routing_matches=route_matches, routing_slots=len(routes) * 8,
                                  maximum_routing_boundary_gap=max_boundary_gap)
                results.append(result)
                line = json.dumps(result)
                print(line, flush=True)
                output.write(line + "\n")
                output.flush()
        agreement = matches / slots if slots else 0.0
        passed = bool(results) and all(r["passed"] for r in results) and routing_ok and agreement >= LIMITS["routing_slot_agreement"]
        summary = {"passed": passed, "limits": LIMITS, "dependencies": versions,
                   "reference_source_sha256": source_hash,
                   "manifest_sha256": manifest_hash, "routing_slot_agreement": agreement,
                   "maximum_expert_read_bytes": reader.maximum_expert_read,
                   "peak_mlx_bytes": mx.get_peak_memory(), "compared_positions": len(results)}
        output.write(json.dumps(summary) + "\n")
        print(json.dumps(summary), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
