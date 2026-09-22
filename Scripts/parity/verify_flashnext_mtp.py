#!/usr/bin/env python3
"""Compare a Swift MTP capture to pinned upstream CPU components.

Uses the actual SGLang fusion method (AST-extracted, not reimplemented) and
Transformers' decoder, QSA cache, HC mixer and RMSNorm on identical dequantized
weights, including an optional completed local install. This is component
parity, not a CUDA/SGLang serving run or a performance benchmark.
"""
import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from transformers import Qwen4ExpTextConfig
from transformers.cache_utils import DynamicCache
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextDecoderLayer, Qwen4ExpTextGatedResidual,
    Qwen4ExpTextRMSNorm, Qwen4ExpTextRotaryEmbedding,
)

TRANSFORMERS_COMMIT = "4da05482135896a529d5536c3c003102d36528a2"
SGLANG_SHA256 = "81610c54803cc45d3093c1d2db285fb0b9031c2ee896282ddda18f62eb50c0f2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--sglang-source", type=Path, required=True)
    parser.add_argument("--indexer-ties", choices=("upstream", "lowest-index"), default="upstream",
                        help="Explicit diagnostic adaptation of exactly tied QSA boundaries only")
    parser.add_argument("--write-golden", type=Path,
                        help="Write passing upstream outputs for the Swift regression test; refuses overwrite")
    parser.add_argument("--install", type=Path,
                        help="Read the verified local sidecar/head instead of capture weights (no download)")
    args = parser.parse_args()
    direct = json.loads(importlib.metadata.distribution("transformers").read_text("direct_url.json"))
    if direct["vcs_info"]["commit_id"] != TRANSFORMERS_COMMIT:
        raise RuntimeError("wrong Transformers reference revision")
    source = args.sglang_source.read_bytes()
    if hashlib.sha256(source).hexdigest() != SGLANG_SHA256:
        raise RuntimeError("wrong SGLang reference source")
    tree = ast.parse(source)
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen4ExpForCausalLMMTP")
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "_fuse_residual_linear_shared")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(args.sglang_source), "exec"), namespace)
    fuse = namespace[method.name]
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    capture = json.loads(args.capture.read_text())
    if not isinstance(capture.get("rows"), list) or not capture["rows"]:
        raise ValueError("capture must contain at least one aligned row")
    if args.install:
        from mtp_install_weights import load_install
        arch, weights = load_install(args.install)
    else:
        arch = None
        weights = {name: torch.tensor(t["values"], dtype=torch.float32).reshape(t["shape"])
                   for name, t in capture["weights"].items()}
    cfg = Qwen4ExpTextConfig(
        hidden_size=64, vocab_size=1024, num_hidden_layers=1,
        layer_types=["full_attention"], ple_layer_ids=[],
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        hc_count=4, hc_lowrank=64, num_experts=8, num_experts_per_tok=6,
        moe_intermediate_size=64, shared_expert_intermediate_size=64,
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=32,
        indexer_budget=32, indexer_compress_ratio=4,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000.0,
                         "partial_rotary_factor": 0.25, "mrope_section": [2, 1, 1]},
    )
    if arch is not None:
        cfg = Qwen4ExpTextConfig(
            hidden_size=arch["hiddenSize"], vocab_size=arch["vocabSize"], num_hidden_layers=1,
            layer_types=["full_attention"], ple_layer_ids=[],
            num_attention_heads=arch["numHeads"], num_key_value_heads=arch["numFullKVHeads"], head_dim=arch["fullHeadDim"],
            hc_count=arch["hcCount"], hc_lowrank=arch["hcLowRank"], num_experts=arch["numExperts"], num_experts_per_tok=arch["topKExperts"],
            moe_intermediate_size=arch["moeIntermediateSize"], shared_expert_intermediate_size=arch["ffnIntermediate"],
            indexer_n_heads=arch["indexerNumHeads"], indexer_kv_heads=arch["indexerNumKVHeads"], indexer_head_dim=arch["indexerHeadDim"],
            indexer_budget=arch["indexerBudget"], indexer_compress_ratio=arch["indexerCompressRatio"],
            rope_parameters={"rope_type": "default", "rope_theta": arch["fullRopeTheta"],
                             "partial_rotary_factor": arch["partialRotaryFactor"], "mrope_section": [11, 11, 10]},
        )
    cfg._attn_implementation = "eager"
    d, bundle = cfg.hidden_size, cfg.hidden_size * cfg.hc_count
    with torch.device("meta"):
        layer = Qwen4ExpTextDecoderLayer(cfg, 0).eval()
        mixer = Qwen4ExpTextGatedResidual(cfg, use_combine=False).eval()
    rotary = Qwen4ExpTextRotaryEmbedding(cfg)
    with torch.device("meta"):
        fusion = SimpleNamespace(hc_count=cfg.hc_count, hidden_size=d,
            pre_fc_norm_embedding=Qwen4ExpTextRMSNorm(d),
            pre_fc_norm_hidden=Qwen4ExpTextRMSNorm(bundle),
            fc_embedding=nn.Linear(d, d, bias=False), fc_hidden=nn.Linear(d, d, bias=False))
    for name in ("pre_fc_norm_embedding", "pre_fc_norm_hidden", "fc_embedding", "fc_hidden"):
        getattr(fusion, name).load_state_dict({"weight": weights[f"mtp.{name}.weight"]}, strict=True, assign=True)
    state = {name.removeprefix("mtp.layers.0."): value for name, value in weights.items()
             if name.startswith("mtp.layers.0.") and ".experts." not in name}
    state["mlp.experts.gate_up_proj"] = torch.stack([
        torch.cat([weights[f"mtp.layers.0.mlp.experts.{i}.{p}_proj.weight"] for p in ("gate", "up")])
        for i in range(cfg.num_experts)])
    state["mlp.experts.down_proj"] = torch.stack([
        weights[f"mtp.layers.0.mlp.experts.{i}.down_proj.weight"] for i in range(cfg.num_experts)])
    layer.load_state_dict(state, strict=True, assign=True)
    mixer.load_state_dict({name.removeprefix("mtp.hyper_connection_mixer."): value
                          for name, value in weights.items() if name.startswith("mtp.hyper_connection_mixer.")}, strict=True, assign=True)
    for name in list(weights):
        if ".experts." in name:
            del weights[name]
    cache = DynamicCache(config=cfg)
    stages = {}
    tied_rows = []
    original_topk = torch.Tensor.topk
    def observe_topk(values, *topk_args, **kwargs):
        result = original_topk(values, *topk_args, **kwargs)
        if values.ndim == 1:
            stages["index_scores"] = values.detach().clone()
            k = len(result.indices)
            ranked = values.sort(descending=True).values
            if k < len(values) and ranked[k - 1] == ranked[k]:
                tied_rows.append(pos)
                if args.indexer_ties == "lowest-index":
                    indices = torch.argsort(values, descending=True, stable=True)[:k]
                    # Only ordering among exactly equal scores can change.
                    if not torch.equal(values[indices], result.values):
                        raise RuntimeError("tie adaptation changed selected scores")
                    result = torch.return_types.topk((values[indices], indices))
            stages["index_blocks"] = result.indices.detach().clone()
        return result
    torch.Tensor.topk = observe_topk
    def capture_stage(name, item=None):
        def hook(module, inputs, output):
            stages[name] = (output if item is None else output[item]).detach().flatten()
        return hook
    layer.attn_hyper_connection.register_forward_hook(capture_stage("attention_mix", 0))
    layer.self_attn.register_forward_hook(capture_stage("attention", 0))
    layer.mlp_hyper_connection.register_forward_hook(capture_stage("mlp_mix", 0))
    layer.mlp.gate.register_forward_hook(capture_stage("router", 0))
    layer.mlp.register_forward_hook(capture_stage("moe"))
    failures = []
    golden_rows = []
    worst = {"hidden": 0.0, "logits": 0.0}
    with torch.no_grad():
        for pos, row in enumerate(capture["rows"]):
            e = torch.tensor(row["embedding"]).reshape(1, 1, d)
            h = torch.tensor(row["hidden"]).reshape(1, 1, bundle)
            fused = fuse(fusion, e, h)
            positions = torch.arange(pos + 1).reshape(1, -1)
            hidden = layer(fused, position_embeddings=rotary(fused, positions),
                           attention_mask=torch.zeros(1, 1, 1, pos + 1), past_key_values=cache)
            logits = nn.functional.linear(mixer(hidden), weights["lm_head.weight"])
            golden_rows.append({"embedding": row["embedding"], "hidden": row["hidden"],
                                "output_hidden": hidden.flatten().tolist(), "logits": logits.flatten().tolist()})
            for name, expected, actual in (("hidden", hidden, row["output_hidden"]),
                                            ("logits", logits, row["logits"])):
                expected = expected.flatten()
                actual = torch.tensor(actual)
                relative = float((expected - actual).abs().max() / expected.abs().max())
                worst[name] = max(worst[name], relative)
                print(f"row={pos} {name} maxAbs/scale={relative:.8f}")
                if not torch.isfinite(expected).all() or not relative <= 0.05:
                    failures.append(f"row={pos} {name}: {relative}")
                    for stage, reference in stages.items():
                        if stage.startswith("index_"):
                            print(f"  {stage}: {reference.tolist()}")
                            continue
                        if stage not in row.get("stages", {}):
                            continue
                        actual_stage = torch.tensor(row["stages"][stage])
                        print(f"  {stage}: error={float((reference-actual_stage).abs().max())} scale={float(reference.abs().max())}")
                        if stage == "router":
                            expected_routes = torch.topk(reference, cfg.num_experts_per_tok).indices.tolist()
                            actual_routes = torch.topk(actual_stage, cfg.num_experts_per_tok).indices.tolist()
                            print(f"  router IDs upstream={expected_routes} metal={actual_routes}")
            if int(logits.argmax()) != int(torch.tensor(row["logits"]).argmax()):
                failures.append(f"row={pos} greedy token mismatch")
                print(f"row={pos} greedy upstream={int(logits.argmax())} metal={int(torch.tensor(row['logits']).argmax())}")
                candidates = logits.flatten().topk(4).indices
                print(f"  top IDs={candidates.tolist()} upstream={logits.flatten()[candidates].tolist()} metal={torch.tensor(row['logits'])[candidates].tolist()}")
                print(f"  upstream FP16-logit argmax={int(logits.half().argmax())}")
    print(json.dumps({"rows": len(capture["rows"]), "worst_relative": worst,
                      "indexer_tie_policy": args.indexer_ties, "tied_rows": tied_rows,
                      "torch": torch.__version__, "transformers_commit": TRANSFORMERS_COMMIT,
                      "capture_sha256": hashlib.sha256(args.capture.read_bytes()).hexdigest(),
                      "failures": failures}, indent=2))
    torch.Tensor.topk = original_topk
    if failures:
        raise SystemExit(1)
    if args.write_golden:
        with args.write_golden.open("x") as output:
            json.dump({"transformers_commit": TRANSFORMERS_COMMIT, "sglang_sha256": SGLANG_SHA256,
                       "torch": torch.__version__, "indexer_tie_policy": args.indexer_ties,
                       "rows": golden_rows}, output, separators=(",", ":"))
            output.write("\n")


if __name__ == "__main__":
    main()
