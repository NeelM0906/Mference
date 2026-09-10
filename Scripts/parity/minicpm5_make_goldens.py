#!/usr/bin/env python3
"""Reference-parity golden harness for the ``minicpm5`` family (bring-up kit W3.2).

Builds a toy ``LlamaConfig`` with the production checkpoint's *structure* (plain
pre-norm llama, GQA, full-head NeoX RoPE at theta 5e6, no q/k norm, untied head,
two EOS ids), instantiates ``LlamaForCausalLM`` from it with a fixed seed, emits
the toy checkpoint, and captures per-layer goldens from the **installed**
``transformers`` package (pinned to the version the real checkpoint declares).

Everything is float32 on CPU, single-threaded, ``model.eval()`` under
``torch.no_grad()``. Goldens are byte-reproducible.

Two weight sets are captured, in two fixture directories:

* ``Tests/Mference/Fixtures/minicpm5-bf16/`` — every parameter rounded through
  bfloat16, i.e. exactly what the emitted checkpoint carries. The record of the
  reference's own arithmetic on this checkpoint.
* ``Tests/Mference/Fixtures/minicpm5/`` — the **gate set**: every rank-2
  projection whose last dim is a multiple of 64 (embedding, lm_head, q/k/v/o,
  gate/up/down — all of them, at this shape) additionally replaced by its
  INT4 affine group-64 reconstruction under a transcription of
  ``Int4AffineEncoder.encodeGroup`` (the same transcription
  ``Scripts/quantizer-weight-gate.py`` uses, locked to the Swift original by
  ``Int4AffineEncoderConventionTests``). This is the weight set the Metal runner
  actually sees after the repacker quantizes the checkpoint in flight, so a
  Swift forward compared against it measures the *port*, not the quantizer.

Usage
-----
    Scripts/parity/minicpm5_make_goldens.py               # write goldens + toy ckpt
    Scripts/parity/minicpm5_make_goldens.py --print-hashes

See Scripts/parity/README.md for the venv (torch CPU + transformers==5.6.2).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

SEED = 20260910
REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_GATE = REPO_ROOT / "Tests" / "Mference" / "Fixtures" / "minicpm5"
FIXTURES_BF16 = REPO_ROOT / "Tests" / "Mference" / "Fixtures" / "minicpm5-bf16"
CKPT_DIR = FIXTURES_GATE / "toy-ckpt"      # committed: 330 KB, so the Swift gates never skip
GROUP = 64

# The toy config. Same *shape class* as ArchConfig.miniCPM5Toy in the Swift tests
# (the two are cross-checked by MiniCPM5ReferenceParityTests). initializer_range
# is raised from the 0.02 default so the logits carry usable top-1 margins for
# the token-exact rollout gate (METH-01: a flip inside the arithmetic noise is
# not evidence); the margins actually observed are recorded in the manifest.
TOY_CONFIG = dict(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    hidden_act="silu",
    max_position_embeddings=512,
    initializer_range=0.08,
    rms_norm_eps=1e-6,
    rope_theta=5_000_000.0,
    rope_scaling=None,
    tie_word_embeddings=False,
    attention_bias=False,
    mlp_bias=False,
    bos_token_id=0,
    eos_token_id=[1, 3],
    pad_token_id=1,
    use_cache=True,
)

SHORT_LEN, LONG_LEN, DECODE_STEPS = 12, 48, 16
MIN_MARGIN = 5e-3   # smallest top-1/top-2 logit gap accepted on any rollout step


# ---------------------------------------------------------------------- INT4 g64

def bf16_bits(x):
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    return ((bits + (np.uint32(0x7FFF) + lsb)) >> np.uint32(16)).astype(np.uint16)


def bf16_to_f32(bits):
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def int4_g64_reconstruct(w: np.ndarray) -> np.ndarray:
    """Int4AffineEncoder.encodeGroup + Quantization.dequantizeInt4Affine, row-major
    over groups of 64 along the last dim. Plain min/max affine; scale and bias
    rounded through BF16 before index quantization; round-half-away-from-zero;
    reconstruction is float32 ``q * scale + bias``."""
    rows, cols = w.shape
    assert cols % GROUP == 0, (rows, cols)
    vals = w.astype(np.float32).reshape(-1, GROUP)
    wmin, wmax = vals.min(axis=1), vals.max(axis=1)
    const = wmax == wmin
    s_bits = bf16_bits(np.where(const, np.float32(1), (wmax - wmin) / np.float32(15)))
    b_bits = bf16_bits(wmin)
    s, b = bf16_to_f32(s_bits), bf16_to_f32(b_bits)
    inv = np.where(s == 0, np.float32(0), np.float32(1) / s)
    t = ((vals - b[:, None]) * inv[:, None]).astype(np.float32)
    q = np.clip(np.trunc(t + np.copysign(np.float32(0.5), t)), 0, 15).astype(np.float32)
    return (q * s[:, None] + b[:, None]).astype(np.float32).reshape(rows, cols)


def quantizable(name: str, t: torch.Tensor) -> bool:
    return t.ndim == 2 and name.endswith(".weight") and "norm" not in name and t.shape[1] % GROUP == 0


# ------------------------------------------------------------------------ model

def build_model():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(SEED)
    cfg = LlamaConfig(**TOY_CONFIG)
    cfg._attn_implementation = "eager"
    model = LlamaForCausalLM(cfg)
    model.eval()
    # Round every parameter through bfloat16: that is what the checkpoint carries.
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(p.to(torch.bfloat16).to(torch.float32))
    return cfg, model


def apply_int4(model):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if quantizable(name, p):
                p.copy_(torch.from_numpy(int4_g64_reconstruct(p.numpy())))


def prompts():
    rng = np.random.default_rng(SEED)
    # Never the special ids 0..3 (bos, eos, tool markers in the real vocab).
    short = rng.integers(4, TOY_CONFIG["vocab_size"], size=SHORT_LEN).tolist()
    long = rng.integers(4, TOY_CONFIG["vocab_size"], size=LONG_LEN).tolist()
    return {"short": short, "long": long}


# ---------------------------------------------------------------------- capture

class Capture:
    """Forward hooks on every decoder layer: attention-branch output (after
    o_proj, before the residual add), MLP-branch output, and the layer output."""

    def __init__(self, model):
        self.rows = {}
        self.handles = []
        for i, layer in enumerate(model.model.layers):
            key = f"layer{i:02d}"
            self.handles.append(layer.self_attn.register_forward_hook(
                self._hook(f"{key}.attn_out", pick=0)))
            self.handles.append(layer.mlp.register_forward_hook(
                self._hook(f"{key}.mlp_out")))
            self.handles.append(layer.register_forward_hook(
                self._hook(f"{key}.hidden_out", pick=0)))
        self.handles.append(model.model.norm.register_forward_hook(
            self._hook("final_norm_out")))
        self.handles.append(model.model.embed_tokens.register_forward_hook(
            self._hook("embed_out")))

    def _hook(self, key, pick=None):
        def fn(_module, _inputs, output):
            t = output[pick] if (pick is not None and isinstance(output, (tuple, list))) else output
            self.rows.setdefault(key, []).append(t.detach()[0].clone())   # drop batch
        return fn

    def take(self):
        out = {k: torch.cat(v, dim=0).contiguous() for k, v in self.rows.items()}
        self.rows = {}
        return out

    def remove(self):
        for h in self.handles:
            h.remove()


def run_prefill(model, ids):
    cap = Capture(model)
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), use_cache=False)
    tensors = cap.take()
    cap.remove()
    logits = out.logits[0].detach().contiguous()
    tensors["logits"] = logits
    return tensors


def margin(row: torch.Tensor):
    top2 = torch.topk(row, 2).values
    return float(top2[0] - top2[1])


def run_decode(model, ids):
    """Prefill with a cache, then DECODE_STEPS greedy steps. Step 0's logits are
    the cached prefill's last row (captured without hooks); steps 1.. are hooked,
    so per-layer decode tensors have DECODE_STEPS - 1 rows (Flash-Next's
    convention, kept so one reader serves both families)."""
    from transformers import DynamicCache
    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), past_key_values=cache, use_cache=True)
    step_logits = [out.logits[0, -1].detach().clone()]
    generated = [int(torch.argmax(step_logits[-1]))]
    margins = [margin(step_logits[-1])]
    cap = Capture(model)
    for _ in range(DECODE_STEPS - 1):
        with torch.no_grad():
            out = model(input_ids=torch.tensor([[generated[-1]]]),
                        past_key_values=cache, use_cache=True)
        step_logits.append(out.logits[0, -1].detach().clone())
        generated.append(int(torch.argmax(step_logits[-1])))
        margins.append(margin(step_logits[-1]))
    tensors = cap.take()
    cap.remove()
    tensors["step_logits"] = torch.stack(step_logits).contiguous()
    # Uncached rollout: re-prefill the whole sequence every step.
    seq = list(ids)
    uncached = []
    for _ in range(DECODE_STEPS):
        with torch.no_grad():
            out = model(input_ids=torch.tensor([seq]), use_cache=False)
        nxt = int(torch.argmax(out.logits[0, -1]))
        uncached.append(nxt)
        seq.append(nxt)
    return tensors, generated, uncached, margins


# ----------------------------------------------------------------------- output

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_set(out_dir: Path, model, label: str, print_only: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    findings = {"rollout_margins": {}, "cache_equivalence": {}}
    for name, ids in prompts().items():
        pre = run_prefill(model, ids)
        dec, generated, uncached, margins = run_decode(model, ids)
        assert generated == uncached, (name, generated, uncached)
        assert min(margins) >= MIN_MARGIN, (name, margins)
        findings["rollout_margins"][name] = {
            "min": min(margins), "per_step": margins}
        findings["cache_equivalence"][name] = "cached == uncached rollout, asserted"
        ints_prefill = {
            "argmax_all_positions": [int(i) for i in torch.argmax(pre["logits"], dim=-1)],
            "next_token": int(torch.argmax(pre["logits"][-1])),
        }
        ints_decode = {
            "generated_token_ids": generated,
            "uncached_rollout_token_ids": uncached,
            "top2_margin_per_step": margins,
        }
        for fname, payload in [
            (f"prefill_{name}.safetensors", pre),
            (f"decode_{name}.safetensors", dec),
        ]:
            if not print_only:
                save_file({k: v.to(torch.float32) for k, v in payload.items()},
                          str(out_dir / fname), metadata={"format": "pt"})
            files[fname] = {"tensors": sorted(payload.keys()),
                            "shapes": {k: list(v.shape) for k, v in sorted(payload.items())}}
        for fname, payload in [
            (f"integers_prefill_{name}.json", ints_prefill),
            (f"integers_decode_{name}.json", ints_decode),
        ]:
            if not print_only:
                (out_dir / fname).write_text(json.dumps(payload, indent=1) + "\n")
            files[fname] = {"keys": sorted(payload.keys())}
    return files, findings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-hashes", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    import transformers
    cfg, model = build_model()

    # --- Toy checkpoint (bf16, HF llama layout: model.layers.N.*, lm_head.weight).
    if not args.print_hashes:
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().to(torch.bfloat16).contiguous()
                 for k, v in model.state_dict().items()}
        save_file(state, str(CKPT_DIR / "model.safetensors"), metadata={"format": "pt"})
        config_json = json.loads(cfg.to_json_string())
        config_json["architectures"] = ["LlamaForCausalLM"]
        config_json["model_type"] = "llama"
        (CKPT_DIR / "config.json").write_text(json.dumps(config_json, indent=2, sort_keys=True) + "\n")
        # Round-trip proof: the emitted bytes reload to the same tensors.
        reloaded = load_file(str(CKPT_DIR / "model.safetensors"))
        for k, v in state.items():
            assert torch.equal(reloaded[k], v), k
        names = sorted(state)
        assert "lm_head.weight" in names and "model.embed_tokens.weight" in names
        assert not any(".q_norm." in n or ".k_norm." in n for n in names)

    sets = {}
    files_bf16, findings_bf16 = write_set(FIXTURES_BF16, model, "bf16", args.print_hashes)
    apply_int4(model)
    files_gate, findings_gate = write_set(FIXTURES_GATE, model, "int4", args.print_hashes)

    def manifest(out_dir, files, findings, weights):
        return {
            "schema": "mference.minicpm5.goldens/1",
            "generated_by": "Scripts/parity/minicpm5_make_goldens.py",
            "family": "minicpm5",
            "reference": {
                "package": "transformers", "version": transformers.__version__,
                "model_class": "LlamaForCausalLM", "model_type": "llama",
                "torch_version": torch.__version__, "device": "cpu",
                "attn_implementation": "eager", "deterministic_algorithms": True,
                "num_threads": 1, "seed": SEED,
            },
            "dtype_policy": {
                "weights": weights,
                "forward": "float32",
                "checkpoint": "bfloat16 (Tests/Mference/Fixtures/minicpm5/toy-ckpt)",
                "rmsnorm": "LlamaRMSNorm: weight * (x.float() * rsqrt(mean(x^2) + eps)).to(input_dtype); plain w, no (1 + w) fold",
                "rope": "rotate_half over the whole head_dim; inv_freq = 1/theta^(arange(0, head_dim, 2)/head_dim); rope_theta 5e6; rope_scaling null",
                "attention": "eager, scaling = head_dim ** -0.5, no q/k norm, no output gate, GQA repeat_kv",
                "int4": "Int4AffineEncoder.encodeGroup transcription: min/max affine over 64-groups, BF16 scale/bias, round-half-away-from-zero, reconstruction q*scale+bias in float32",
            },
            "config": json.loads(cfg.to_json_string()),
            "prompts": {k: {"ids": v, "length": len(v)} for k, v in prompts().items()},
            "decode_steps": DECODE_STEPS,
            "decode_row_convention": "row i of a decode_* per-layer tensor is decode step i+1; step_logits has DECODE_STEPS rows, row 0 from the cached prefill leg",
            "golden_files": {
                name: dict(info, size=(out_dir / name).stat().st_size, sha256=sha256(out_dir / name))
                for name, info in files.items()
            } if not args.print_hashes else files,
            "findings": dict(findings, tolerances={
                "note": "The gate set is compared against a Metal runner that computes activations in FP16 with FP32 accumulation; the tolerance tiers are stated per gate in MiniCPM5ReferenceParityTests and recorded on docs/families/MINICPM5.md.",
                "min_rollout_margin_asserted": MIN_MARGIN,
            }),
        }

    for out_dir, files, findings, weights in [
        (FIXTURES_BF16, files_bf16, findings_bf16, "bfloat16-rounded, then float32"),
        (FIXTURES_GATE, files_gate, findings_gate,
         "bfloat16-rounded, then every rank-2 projection (last dim % 64 == 0) replaced by its INT4 g64 reconstruction"),
    ]:
        m = manifest(out_dir, files, findings, weights)
        if args.print_hashes:
            print(json.dumps(m["golden_files"], indent=1))
        else:
            (out_dir / "goldens-manifest.json").write_text(json.dumps(m, indent=1) + "\n")
            print(f"wrote {out_dir}")
            for k, v in findings["rollout_margins"].items():
                print(f"  {k}: min top-1/top-2 margin {v['min']:.4f}")
    if not args.print_hashes:
        total = sum(p.stat().st_size for d in (FIXTURES_BF16, FIXTURES_GATE) for p in d.rglob("*") if p.is_file())
        print(f"fixture bytes total: {total}")
        assert total < 8 * 1024 * 1024


if __name__ == "__main__":
    main()
