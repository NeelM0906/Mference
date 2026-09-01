#!/usr/bin/env python3
"""Reference-parity golden harness for the ``qwen38flashnext`` family (bring-up kit W3.2).

Builds a *toy* ``Qwen4ExpTextConfig`` that exercises every module the Swift port has to
reproduce (hyper-connections, PLE n-gram hashing, the QSA indexer, MoE routing, GDN and
gated full attention), instantiates ``Qwen4ExpForCausalLM`` from it with a fixed seed, and
captures per-module goldens from the *installed* ``transformers`` package.

Everything is float32 on CPU, single-threaded, ``model.eval()`` under ``torch.no_grad()``.
Goldens are byte-reproducible: running the script twice produces identical output files.

Usage
-----
    Scripts/parity/qwen4exp_make_goldens.py                    # write goldens
    Scripts/parity/qwen4exp_make_goldens.py --emit-checkpoint  # ... and the toy bf16 ckpt
    Scripts/parity/qwen4exp_make_goldens.py --print-hashes     # hash outputs, write nothing

See Scripts/parity/README.md for the venv setup and the meaning of each golden file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

# ---------------------------------------------------------------------------------------
# Determinism: must be configured before torch does any work.
# ---------------------------------------------------------------------------------------
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

SEED = 1234
REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "Tests" / "Mference" / "Fixtures" / "qwen4exp"
FIXTURES_BF16 = REPO_ROOT / "Tests" / "Mference" / "Fixtures" / "qwen4exp-bf16"
CKPT_DIR = REPO_ROOT / "scratch" / "qwen4exp-toy-ckpt"
CKPT_PROD_DIR = REPO_ROOT / "scratch" / "qwen4exp-toy-ckpt-prodlayout"

# ---------------------------------------------------------------------------------------
# Toy configuration.
#
# Deviations from the bring-up-kit suggestion, and why:
#   * num_hidden_layers 4 -> 6 with layer_types [lin, lin, lin, qsa, lin, qsa].  The runtime
#     design doc requires ">=2 attention layers" so the per-layer indexer key cache and the
#     layer_idx plumbing are actually exercised.  PLE still lands on one-indexed layer 2
#     (= layers[1]), a linear_attention layer, as the validator demands.
#   * SHORT prompt 12 -> 11 tokens.  With indexer_budget 8 / compress_ratio 4 the block_topk
#     is 2, so a 12-token prompt already has 3 complete blocks at the last query position and
#     is NOT dense-equivalent.  11 tokens gives at most 2 complete blocks (+3 tail) at every
#     query, which is the dense-equivalent regime the A/B gate needs.
#   * rope_parameters.mrope_section [1, 1, 0].  rotary_dim is 4, so there are only 2 frequency
#     pairs and the transformers default [11, 11, 10] cannot be expressed; the sections
#     collapse for text-only positions anyway.
# Every other value is exactly as specified.  No validator rejected anything.
# ---------------------------------------------------------------------------------------
TOY_CONFIG = dict(
    vocab_size=128,
    hidden_size=64,
    num_hidden_layers=6,
    layer_types=[
        "linear_attention",
        "linear_attention",  # <- PLE layer (one-indexed id 2)
        "linear_attention",
        "qwen_sparse_attention",
        "linear_attention",
        "qwen_sparse_attention",
    ],
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    hidden_act="silu",
    output_gate_type="sigmoid",
    rms_norm_eps=1e-6,
    max_position_embeddings=512,
    rope_parameters={
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.25,
        "mrope_section": [1, 1, 0],
    },
    linear_key_head_dim=8,
    linear_num_key_heads=2,
    linear_value_head_dim=8,
    linear_num_value_heads=4,
    linear_conv_kernel_dim=4,
    hc_count=4,
    hc_lowrank=8,
    ple_layer_ids=[2],
    ple_embed_dim=64,
    ple_conv_kernel_size=4,
    ngram_size=3,
    heads_per_ngram=8,
    ngram_vocab_size_base=97,
    make_ngram_vocab_size_divisible_by=128,
    split_ngram_parts=2,
    seed=SEED,
    indexer_n_heads=2,
    indexer_kv_heads=1,
    indexer_head_dim=8,
    indexer_budget=8,
    indexer_compress_ratio=4,
    num_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    norm_topk_prob=True,
    eos_token_id=0,
    bos_token_id=0,
    pad_token_id=None,
    tie_word_embeddings=False,
    initializer_range=0.02,
    attention_bias=False,
    attention_dropout=0.0,
    use_cache=True,
)

SHORT_LEN = 11
LONG_LEN = 48
DECODE_STEPS = 16
# LONG_EOS_AT places an eos_token_id inside the long prompt so the PLE n-gram shift's
# EOS-segmentation path (`_shift_right_ignore_eos`) is exercised, not just the prefix pad.
LONG_EOS_AT = 20


# ---------------------------------------------------------------------------------------
# Independent (pure-python) reimplementation of the PLE hash path, per the runtime design
# doc.  Asserted against the model's own buffers -- if these ever diverge the port is
# reading the spec wrong, or the reference changed.
# ---------------------------------------------------------------------------------------
_MASK64 = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15
_M1 = 0xBF58476D1CE4E5B9
_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def splitmix64(value: int) -> int:
    value = (value + _GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _M1) & _MASK64
    value = ((value ^ (value >> 27)) * _M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def derive_layer_multipliers(unigram_vocab_size: int, ngram_size: int, ple_layer_index: int, seed: int) -> list[int]:
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    out = []
    for i in range(ngram_size):
        out.append(2 * (splitmix64((base_seed + _GAMMA * (i + 1)) & _MASK64) % half_bound) + 1)
    return out


def is_prime(v: int) -> bool:
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    for d in range(3, math.isqrt(v) + 1, 2):
        if v % d == 0:
            return False
    return True


def nth_prime_after(start: int, count: int) -> int:
    p = start
    for _ in range(count):
        p += 1
        while not is_prime(p):
            p += 1
    return p


def derive_head_tables(ngram_vocab_size_base: int, ngram_heads: int, ple_layer_index: int) -> tuple[list[int], list[int]]:
    sizes, offsets, total = [], [], 0
    for head_idx in range(ngram_heads):
        global_head_idx = ple_layer_index * ngram_heads + head_idx
        size = nth_prime_after(ngram_vocab_size_base - 1, global_head_idx + 1)
        sizes.append(size)
        offsets.append(total)
        total += size
    return sizes, offsets


def shift_right_ignore_eos(ids: list[int], shift: int, eos: int) -> list[int]:
    """Pure-python mirror of Qwen4ExpTextNGramEmbedding._shift_right_ignore_eos."""
    if shift == 0:
        return list(ids)
    n = len(ids)
    prev_eos_inclusive, cur = [], -1
    for i, t in enumerate(ids):
        if t == eos:
            cur = i
        prev_eos_inclusive.append(cur)
    previous_eos = [-1] + prev_eos_inclusive[:-1]
    out = []
    for i in range(n):
        position_in_segment = i - (previous_eos[i] + 1)
        source = i - shift
        out.append(ids[source] if (position_in_segment >= shift and source >= 0) else eos)
    return out


def derive_ngram_row_ids(
    token_ids: list[int],
    *,
    ngram_size: int,
    heads_per_ngram: int,
    multipliers: list[int],
    head_vocab_sizes: list[int],
    head_offsets: list[int],
    eos: int,
) -> list[list[int]]:
    """Pure-python mirror of Qwen4ExpTextNGramEmbedding.forward -> final embedding row ids.

    Returns [seq_len][ngram_heads] absolute row indices into the padded n-gram table.
    Prefixes the sequence with (ngram_size - 1) eos tokens, exactly like the fresh-cache path.
    """
    context_len = ngram_size - 1
    history = [eos] * context_len + list(token_ids)
    shifted = [shift_right_ignore_eos(history, s, eos) for s in range(ngram_size)]

    blocks: list[list[int]] = []
    for ngram in range(2, ngram_size + 1):
        start = (ngram - 2) * heads_per_ngram
        mixed = [t * multipliers[0] for t in shifted[0]]
        for position in range(1, ngram):
            mixed = [m ^ (t * multipliers[position]) for m, t in zip(mixed, shifted[position])]
        block = [
            [(m % head_vocab_sizes[start + h]) + head_offsets[start + h] for h in range(heads_per_ngram)]
            for m in mixed
        ]
        blocks.append(block)

    rows = [sum((blk[i] for blk in blocks), []) for i in range(len(history))]
    return rows[-len(token_ids):]


# ---------------------------------------------------------------------------------------
# Capture plumbing.
# ---------------------------------------------------------------------------------------
class Capture:
    """Collects float tensors (-> safetensors) and integer structures (-> json)."""

    def __init__(self) -> None:
        self.floats: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self.ints: "OrderedDict[str, object]" = OrderedDict()
        self.prefix = ""

    def put_float(self, name: str, tensor: torch.Tensor) -> None:
        self.floats[self.prefix + name] = tensor.detach().to(torch.float32).contiguous().clone()

    def put_int(self, name: str, value: object) -> None:
        self.ints[self.prefix + name] = value

    def append_float(self, name: str, tensor: torch.Tensor) -> None:
        """Accumulate one decode step; stacked on flush."""
        key = self.prefix + name
        self.floats.setdefault(key, [])
        assert isinstance(self.floats[key], list), f"{key} already flushed as a tensor"
        self.floats[key].append(tensor.detach().to(torch.float32).contiguous().clone())

    def append_int(self, name: str, value: object) -> None:
        key = self.prefix + name
        self.ints.setdefault(key, [])
        self.ints[key].append(value)

    def stack_pending(self) -> None:
        for key, value in list(self.floats.items()):
            if isinstance(value, list):
                self.floats[key] = torch.cat(value, dim=0).contiguous()


def selected_sets_from_mask(mask: torch.Tensor) -> list[list[int]]:
    """Indexer mask -> per-query sorted list of selected kv indices.

    `mask` is (batch, 1, q_len, kv_len); bool for sdpa, float (0 / dtype-min) for eager.
    """
    keep = mask if mask.dtype == torch.bool else (mask == 0)
    keep = keep[0, 0]  # batch 0, broadcast head dim
    return [torch.nonzero(row, as_tuple=False).flatten().tolist() for row in keep]


def visible_sets_from_causal(mask: torch.Tensor) -> list[list[int]]:
    keep = mask if mask.dtype == torch.bool else (mask == 0)
    keep = keep[0, 0]
    return [torch.nonzero(row, as_tuple=False).flatten().tolist() for row in keep]


class Recorder:
    """Registers hooks on every module whose output the Swift port must reproduce."""

    def __init__(self, model, cap: Capture, *, streaming: bool) -> None:
        self.model = model
        self.cap = cap
        self.streaming = streaming  # True for the decode loop: append instead of assign
        self.handles: list = []
        self._wire()

    def _emit_float(self, name: str, tensor: torch.Tensor) -> None:
        if self.streaming:
            self.cap.append_float(name, tensor[0])  # drop batch dim; steps concat on dim 0
        else:
            self.cap.put_float(name, tensor[0])

    def _emit_int(self, name: str, value: object) -> None:
        if self.streaming:
            self.cap.append_int(name, value)
        else:
            self.cap.put_int(name, value)

    def _hook(self, module, fn):
        # A forward hook that returns non-None REPLACES the module output, so every hook
        # here is wrapped to swallow its return value.
        def wrapped(mod, inputs, output, _fn=fn):
            _fn(mod, inputs, output)
            return None

        self.handles.append(module.register_forward_hook(wrapped))

    def _pre_hook(self, module, fn):
        def wrapped(mod, inputs, _fn=fn):
            _fn(mod, inputs)
            return None

        self.handles.append(module.register_forward_pre_hook(wrapped))

    def _wire(self) -> None:
        inner = self.model.model

        self._hook(inner.embed_tokens, lambda m, i, o: self._emit_float("embed_out", o))
        self._hook(
            inner.hyper_connection_mixer,
            lambda m, i, o: self._emit_float("last_hidden_state", o),
        )

        for layer_idx, layer in enumerate(inner.layers):
            p = f"layer{layer_idx:02d}."

            def on_layer(m, i, o, pfx=p):
                self._emit_float(pfx + "stream_out", o)

            def on_attn_hc(m, i, o, pfx=p):
                self._emit_float(pfx + "attn_hc_mixed", o[0])
                self._emit_float(pfx + "attn_hc_stream_in", o[1])
                self._emit_float(pfx + "attn_hc_inject", o[2])

            def on_mlp_hc(m, i, o, pfx=p):
                self._emit_float(pfx + "mlp_hc_mixed", o[0])
                self._emit_float(pfx + "mlp_hc_stream_in", o[1])
                self._emit_float(pfx + "mlp_hc_inject", o[2])

            def on_moe(m, i, o, pfx=p):
                self._emit_float(pfx + "moe_out", o)

            def on_router(m, i, o, pfx=p):
                # router returns (logits, renormalized top-k weights, top-k indices)
                self._emit_float(pfx + "router_weights", o[1].unsqueeze(0))
                self._emit_int(pfx + "router_indices", o[2].tolist())

            self._hook(layer, on_layer)
            self._hook(layer.attn_hyper_connection, on_attn_hc)
            self._hook(layer.mlp_hyper_connection, on_mlp_hc)
            self._hook(layer.mlp, on_moe)
            self._hook(layer.mlp.gate, on_router)

            if layer.layer_type == "linear_attention":

                def on_gdn(m, i, o, pfx=p):
                    self._emit_float(pfx + "block_out", o)

                self._hook(layer.linear_attn, on_gdn)
            else:

                def on_attn(m, i, o, pfx=p):
                    self._emit_float(pfx + "block_out", o[0])

                def on_indexer(m, i, o, pfx=p):
                    self._emit_int(pfx + "indexer_selected", selected_sets_from_mask(o))
                    self._emit_int(pfx + "indexer_visible", visible_sets_from_causal(i[2]))

                self._hook(layer.self_attn, on_attn)
                self._hook(layer.self_attn.indexer, on_indexer)

            if layer.ple is not None:

                def on_ple(m, i, o, pfx=p):
                    self._emit_float(pfx + "ple_out", o)

                def on_ngram(m, i, o, pfx=p):
                    self._emit_float(pfx + "ple_ngram_embeds", o)

                def on_ngram_ids(m, i, pfx=p):
                    self._emit_int(pfx + "ple_ngram_row_ids", i[0][0].tolist())

                self._hook(layer.ple, on_ple)
                self._hook(layer.ple.ple_embedding, on_ngram)
                self._pre_hook(layer.ple.ple_embedding.ngram_embedding, on_ngram_ids)

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ---------------------------------------------------------------------------------------
# Model construction.
# ---------------------------------------------------------------------------------------
def build_config(attn_implementation: str = "eager"):
    from transformers import Qwen4ExpTextConfig

    cfg = Qwen4ExpTextConfig(**TOY_CONFIG)
    cfg._attn_implementation = attn_implementation
    return cfg


def build_model(attn_implementation: str = "eager", weight_dtype: str = "fp32"):
    """The toy model, in float32.

    ``weight_dtype="bf16"`` rounds every parameter through bfloat16 and back to
    float32 **before** any golden is captured.  That is not a cosmetic option:
    the checkpoint this harness emits is bfloat16, so with the default
    ``fp32`` weights the goldens describe a model whose weights **no consumer
    can obtain**.  A port that loads the emitted checkpoint sees the rounded
    weights, and the resulting logit spread is 1.2e-3 (SHORT) / 3.1e-2 (LONG)
    max-abs -- 10x to 300x the 1e-4 fp32 gate this manifest recommends.

    So the two sets have distinct jobs:

    * ``fp32`` (``Tests/Mference/Fixtures/qwen4exp``) pins the *reference's own*
      arithmetic and is the artifact the design contract was read against.
    * ``bf16`` (``Tests/Mference/Fixtures/qwen4exp-bf16``) pins what a port that
      loads the shipped checkpoint must reproduce, and is the one a Swift
      parity suite can actually gate on at 1e-4.

    The forward is float32 in both cases; only the stored weight values differ.
    Rounding is idempotent (bf16 -> fp32 -> bf16 is exact), so
    ``emit_checkpoint`` writes byte-identical checkpoints either way.
    """
    from transformers import Qwen4ExpForCausalLM

    cfg = build_config(attn_implementation)
    torch.manual_seed(SEED)
    model = Qwen4ExpForCausalLM(cfg)
    model = model.to(torch.float32).eval()
    if weight_dtype == "bf16":
        model = model.to(torch.bfloat16).to(torch.float32).eval()
    elif weight_dtype != "fp32":
        raise ValueError(f"unknown weight dtype {weight_dtype!r}")
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def make_prompt(length: int, *, rng_seed: int, eos_at: int | None) -> list[int]:
    g = torch.Generator().manual_seed(rng_seed)
    ids = torch.randint(1, TOY_CONFIG["vocab_size"], (length,), generator=g).tolist()
    if eos_at is not None:
        ids[eos_at] = TOY_CONFIG["eos_token_id"]
    return ids


# ---------------------------------------------------------------------------------------
# Runs.
# ---------------------------------------------------------------------------------------
@torch.no_grad()
def run_prefill(model, token_ids: list[int], cap: Capture, prefix: str):
    ids = torch.tensor([token_ids], dtype=torch.long)
    attn = torch.ones_like(ids)
    cap.prefix = prefix
    rec = Recorder(model, cap, streaming=False)
    try:
        out = model(input_ids=ids, attention_mask=attn, use_cache=False)
    finally:
        rec.close()
        cap.prefix = ""
    cap.prefix = prefix
    cap.put_float("logits", out.logits[0])
    cap.put_int("argmax_all_positions", out.logits[0].argmax(-1).tolist())
    cap.put_int("next_token", int(out.logits[0, -1].argmax()))
    cap.prefix = ""
    return out


@torch.no_grad()
def run_decode(model, token_ids: list[int], steps: int, cap: Capture, prefix: str):
    """Greedy decode with a DynamicCache; per-step goldens captured through the same hooks."""
    from transformers import DynamicCache

    cache = DynamicCache(config=model.config)
    ids = torch.tensor([token_ids], dtype=torch.long)
    attn = torch.ones_like(ids)

    # Prefill leg (hooks off -- the prefill goldens come from run_prefill's no-cache run).
    out = model(input_ids=ids, attention_mask=attn, past_key_values=cache, use_cache=True)
    cache = out.past_key_values
    next_id = int(out.logits[0, -1].argmax())
    generated = [next_id]

    cap.prefix = prefix
    cap.put_float("cached_prefill_logits_last", out.logits[0, -1:])
    rec = Recorder(model, cap, streaming=True)
    try:
        for _ in range(steps - 1):
            step_ids = torch.tensor([[next_id]], dtype=torch.long)
            attn = torch.cat([attn, torch.ones((1, 1), dtype=torch.long)], dim=1)
            out = model(
                input_ids=step_ids,
                attention_mask=attn,
                past_key_values=cache,
                use_cache=True,
            )
            cache = out.past_key_values
            cap.append_float("step_logits", out.logits[0])
            next_id = int(out.logits[0, -1].argmax())
            generated.append(next_id)
    finally:
        rec.close()

    cap.stack_pending()
    cap.put_int("generated_token_ids", generated)
    cap.prefix = ""
    return generated


@torch.no_grad()
def run_uncached_rollout(model, token_ids: list[int], steps: int) -> list[int]:
    """Same greedy rollout with no cache at all -- re-prefills the whole prefix each step."""
    context = list(token_ids)
    generated: list[int] = []
    for _ in range(steps):
        ids = torch.tensor([context], dtype=torch.long)
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        nxt = int(out.logits[0, -1].argmax())
        generated.append(nxt)
        context.append(nxt)
    return generated


# ---------------------------------------------------------------------------------------
# Assertions the harness must not ship without.
# ---------------------------------------------------------------------------------------
def assert_ple_hash_derivation(model, report: dict) -> None:
    cfg = model.config
    ple = model.model.layers[cfg.ple_layer_ids[0] - 1].ple
    emb = ple.ple_embedding
    ple_layer_index = emb.ple_layer_index

    want_mult = derive_layer_multipliers(cfg.vocab_size, cfg.ngram_size, ple_layer_index, cfg.seed)
    got_mult = emb.layer_multipliers.tolist()
    assert got_mult == want_mult, f"layer_multipliers mismatch: model={got_mult} derived={want_mult}"

    ngram_heads = (cfg.ngram_size - 1) * cfg.heads_per_ngram
    want_sizes, want_offsets = derive_head_tables(cfg.ngram_vocab_size_base, ngram_heads, ple_layer_index)
    assert emb.ngram_heads_vocab_sizes.tolist() == want_sizes, "ngram_heads_vocab_sizes mismatch"
    assert emb.ngram_heads_offsets.tolist() == want_offsets, "ngram_heads_offsets mismatch"

    # No int64 overflow is possible with these multipliers: max token id * max multiplier.
    headroom = (cfg.vocab_size - 1) * max(want_mult)
    assert headroom < (1 << 63), "int64 overflow in the n-gram mix -- multipliers/vocab inconsistent"

    report["ple_hash_derivation"] = {
        "ple_layer_index": ple_layer_index,
        "layer_multipliers": [str(m) for m in want_mult],
        "ngram_heads_vocab_sizes": want_sizes,
        "ngram_heads_offsets": want_offsets,
        "total_vocab_size": sum(want_sizes),
        "padded_vocab_size": int(emb.ngram_embedding.weight.shape[0]),
        "head_dim_per_ngram": int(emb.ngram_embedding.weight.shape[1]),
        "max_mixed_id_headroom_bits": 63 - headroom.bit_length(),
        "verified_against": "pure-python splitmix64 + prime search in this script",
    }


def assert_indexer_regimes(cap: Capture, short_prefix: str, long_prefix: str, report: dict) -> None:
    def gather(prefix: str):
        return {k: v for k, v in cap.ints.items() if k.startswith(prefix) and k.endswith("indexer_selected")}

    short_sel = gather(short_prefix)
    long_sel = gather(long_prefix)
    assert short_sel, "no indexer selections captured for the SHORT prompt"
    assert long_sel, "no indexer selections captured for the LONG prompt"

    # SHORT: dense-equivalent -- every visible position selected, at every attention layer.
    for key, sel in short_sel.items():
        vis_key = key.replace("indexer_selected", "indexer_visible")
        for q, (s, v) in enumerate(zip(sel, cap.ints[vis_key])):
            assert set(s) == set(v), (
                f"SHORT prompt is not dense-equivalent at {key} query {q}: "
                f"selected {sorted(s)} != visible {sorted(v)}"
            )

    # LONG: genuinely sparse -- at least one query drops a visible position, at every layer.
    long_stats = {}
    for key, sel in long_sel.items():
        vis_key = key.replace("indexer_selected", "indexer_visible")
        sparse_queries = [
            q for q, (s, v) in enumerate(zip(sel, cap.ints[vis_key])) if set(s) != set(v)
        ]
        assert sparse_queries, f"LONG prompt selection is dense at {key} -- lower the budget or lengthen the prompt"
        assert set(sel[-1]).issubset(set(cap.ints[vis_key][-1])), f"{key} selected a non-visible position"
        last = len(sel) - 1
        long_stats[key] = {
            "first_sparse_query": sparse_queries[0],
            "num_sparse_queries": len(sparse_queries),
            "last_query_selected": len(sel[last]),
            "last_query_visible": len(cap.ints[vis_key][last]),
            "last_query_sees_itself": last in sel[last],
        }
    report["indexer_regimes"] = {
        "short_dense_equivalent": True,
        "long": long_stats,
    }


# ---------------------------------------------------------------------------------------
# Checkpoint emission.
# ---------------------------------------------------------------------------------------
def emit_checkpoint(model, report: dict) -> None:
    """Write the toy checkpoint twice: HF-native layout and production-name layout.

    Also round-trips the HF-native copy through ``from_pretrained`` to prove the
    ``split_ngram_parts`` shard concatenation reassembles the table correctly, and checks
    empirically which half of the fused ``gate_up_proj`` rows is gate and which is up.
    """
    import copy

    from safetensors import safe_open
    from transformers import Qwen4ExpForCausalLM

    bf16 = copy.deepcopy(model).to(torch.bfloat16)
    if CKPT_DIR.exists():
        shutil.rmtree(CKPT_DIR)
    bf16.save_pretrained(CKPT_DIR, safe_serialization=True)

    native: "OrderedDict[str, tuple[list[int], str]]" = OrderedDict()
    for path in sorted(CKPT_DIR.glob("*.safetensors")):
        with safe_open(path, framework="pt") as f:
            for k in sorted(f.keys()):
                sl = f.get_slice(k)
                native[k] = (list(sl.get_shape()), sl.get_dtype())

    # Production layout: `model.language_model.*` prefix and FUSED expert tensors, matching
    # docs/families/qwen38flashnext.tensors.json.  The text-only ForCausalLM cannot emit this
    # itself, so the repacker's real patterns get a fixture too.
    prod: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    runtime_sd = bf16.state_dict()
    for k, v in runtime_sd.items():
        if k.startswith("model."):
            prod["model.language_model." + k[len("model."):]] = v.contiguous().clone()
        else:
            prod[k] = v.contiguous().clone()
    # Split the n-gram table back into shards, as the real checkpoint stores it.
    shard_keys = [k for k in prod if k.endswith("ple_embedding.ngram_embedding.weight")]
    n_shards = model.config.split_ngram_parts
    for k in shard_keys:
        w = prod.pop(k)
        assert w.shape[0] % n_shards == 0, "padded n-gram vocab not divisible by split_ngram_parts"
        for s, chunk in enumerate(w.chunk(n_shards, dim=0)):
            prod[k[: -len("weight")] + f"shard_{s}.weight"] = chunk.contiguous().clone()

    if CKPT_PROD_DIR.exists():
        shutil.rmtree(CKPT_PROD_DIR)
    CKPT_PROD_DIR.mkdir(parents=True)
    save_file(dict(sorted(prod.items())), str(CKPT_PROD_DIR / "model.safetensors"))
    shutil.copy(CKPT_DIR / "config.json", CKPT_PROD_DIR / "config.json")

    # --- round trip: shards must reassemble, and the reloaded bf16 model must run ---
    reloaded = Qwen4ExpForCausalLM.from_pretrained(CKPT_DIR, dtype=torch.bfloat16)
    reloaded.eval()
    emb_ref = bf16.model.layers[model.config.ple_layer_ids[0] - 1].ple.ple_embedding.ngram_embedding.weight
    emb_rt = reloaded.model.layers[model.config.ple_layer_ids[0] - 1].ple.ple_embedding.ngram_embedding.weight
    assert emb_rt.shape == emb_ref.shape, f"ngram table shape changed on reload: {emb_rt.shape} vs {emb_ref.shape}"
    assert torch.equal(emb_rt, emb_ref), "ngram shard concatenation did not reassemble the table"
    rt_missing = [k for k, v in bf16.state_dict().items() if not torch.equal(v, reloaded.state_dict()[k])]
    assert not rt_missing, f"checkpoint round trip changed tensors: {rt_missing[:5]}"

    # --- fused gate|up row split, verified rather than assumed ---
    experts = bf16.model.layers[0].mlp.experts
    inter = model.config.moe_intermediate_size
    fused = experts.gate_up_proj[0]
    with safe_open(next(CKPT_DIR.glob("*.safetensors")), framework="pt") as f:
        saved_gate = f.get_tensor("model.layers.0.mlp.experts.0.gate_proj.weight")
        saved_up = f.get_tensor("model.layers.0.mlp.experts.0.up_proj.weight")
    gate_is_first = torch.equal(fused[:inter], saved_gate) and torch.equal(fused[inter:], saved_up)
    assert gate_is_first, "fused gate_up_proj row order is NOT gate-then-up -- update the repacker"

    report["checkpoint"] = {
        "round_trip_verified": True,
        "fused_gate_up_row_split": f"gate = rows [0, {inter}), up = rows [{inter}, {2 * inter})",
        "hf_native_dir": str(CKPT_DIR.relative_to(REPO_ROOT)),
        "hf_native_prefix": "model.",
        "hf_native_files": sorted(p.name for p in CKPT_DIR.iterdir()),
        "hf_native_tensors": {k: {"shape": s, "dtype": d} for k, (s, d) in native.items()},
        "prod_layout_dir": str(CKPT_PROD_DIR.relative_to(REPO_ROOT)),
        "prod_layout_prefix": "model.language_model.",
        "prod_layout_tensors": {
            k: {"shape": list(v.shape), "dtype": "BF16"} for k, v in sorted(prod.items())
        },
        "sha256": {
            **{f"hf_native/{p.name}": sha256_file(p) for p in sorted(CKPT_DIR.iterdir()) if p.is_file()},
            **{f"prod_layout/{p.name}": sha256_file(p) for p in sorted(CKPT_PROD_DIR.iterdir()) if p.is_file()},
        },
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--emit-checkpoint",
        action="store_true",
        help=(
            "write the toy bf16 checkpoints to scratch/ (DEFAULT: always on -- the manifest "
            "embeds their tensor names, shapes and sha256, so it cannot be built without them; "
            "the flag is kept for the documented bring-up-kit interface)"
        ),
    )
    ap.add_argument("--print-hashes", action="store_true", help="print sha256 of every output and exit non-writing")
    ap.add_argument(
        "--weight-dtype",
        choices=["fp32", "bf16"],
        default="fp32",
        help=(
            "storage dtype the weights are rounded to before capture (the forward is float32 "
            "either way). fp32 pins the reference's own arithmetic; bf16 pins what a port "
            "loading the emitted bfloat16 checkpoint must reproduce. See build_model."
        ),
    )
    ap.add_argument("--out", default=None, help="fixture output directory")
    args = ap.parse_args()
    if args.out is None:
        args.out = str(FIXTURES if args.weight_dtype == "fp32" else FIXTURES_BF16)

    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    deterministic_algorithms = True
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as exc:  # pragma: no cover - recorded, not fatal
        deterministic_algorithms = False
        print(f"[warn] torch.use_deterministic_algorithms(True) refused: {exc}", file=sys.stderr)

    import transformers

    report: dict = {}
    model = build_model("eager", args.weight_dtype)

    short_ids = make_prompt(SHORT_LEN, rng_seed=11, eos_at=None)
    long_ids = make_prompt(LONG_LEN, rng_seed=48, eos_at=LONG_EOS_AT)

    assert_ple_hash_derivation(model, report)

    cap_pre = Capture()
    run_prefill(model, short_ids, cap_pre, "short.")
    run_prefill(model, long_ids, cap_pre, "long.")
    assert_indexer_regimes(cap_pre, "short.", "long.", report)

    # PLE row ids: assert the reference against the independent pure-python derivation.
    cfg = model.config
    ple_prefix = f"layer{cfg.ple_layer_ids[0] - 1:02d}.ple_ngram_row_ids"
    mult = derive_layer_multipliers(cfg.vocab_size, cfg.ngram_size, 0, cfg.seed)
    ngram_heads = (cfg.ngram_size - 1) * cfg.heads_per_ngram
    sizes, offsets = derive_head_tables(cfg.ngram_vocab_size_base, ngram_heads, 0)
    for name, ids in (("short", short_ids), ("long", long_ids)):
        got = cap_pre.ints[f"{name}.{ple_prefix}"]
        want = derive_ngram_row_ids(
            ids,
            ngram_size=cfg.ngram_size,
            heads_per_ngram=cfg.heads_per_ngram,
            multipliers=mult,
            head_vocab_sizes=sizes,
            head_offsets=offsets,
            eos=cfg.eos_token_id,
        )
        assert got == want, f"PLE n-gram row ids diverge from the independent derivation ({name})"
    report["ple_row_ids_match_independent_derivation"] = True

    cap_dec = Capture()
    gen_short = run_decode(model, short_ids, DECODE_STEPS, cap_dec, "short.")
    gen_long = run_decode(model, long_ids, DECODE_STEPS, cap_dec, "long.")

    # Cache equivalence inside the reference itself.
    unc_short = run_uncached_rollout(model, short_ids, DECODE_STEPS)
    unc_long = run_uncached_rollout(model, long_ids, DECODE_STEPS)
    assert gen_short == unc_short, f"cached decode != uncached rollout (short): {gen_short} vs {unc_short}"
    assert gen_long == unc_long, f"cached decode != uncached rollout (long): {gen_long} vs {unc_long}"
    cap_dec.ints["short.uncached_rollout_token_ids"] = unc_short
    cap_dec.ints["long.uncached_rollout_token_ids"] = unc_long
    report["cache_equivalence_in_reference"] = {"short": True, "long": True}

    # Tolerance observation: sdpa vs eager on identical fp32 inputs.
    model_sdpa = build_model("sdpa", args.weight_dtype)
    with torch.no_grad():
        ids = torch.tensor([long_ids], dtype=torch.long)
        lo_eager = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        lo_sdpa = model_sdpa(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
    report["tolerances"] = {
        "dtype": "float32",
        "eager_vs_sdpa_logits_max_abs": float((lo_eager - lo_sdpa).abs().max()),
        "eager_vs_sdpa_logits_max_rel": float(
            ((lo_eager - lo_sdpa).abs() / lo_eager.abs().clamp_min(1e-6)).max()
        ),
        "eager_vs_sdpa_argmax_identical": bool(
            (lo_eager.argmax(-1) == lo_sdpa.argmax(-1)).all()
        ),
        "recommended_swift_gate_fp32": {"atol": 1e-4, "rtol": 1e-4},
        "note": (
            "Goldens are produced with attn_implementation='eager' (float additive masks). "
            "sdpa uses boolean masks; the deltas above are the reference's own fp32 spread "
            "between the two kernels and are the floor for any port tolerance."
        ),
    }

    # Checkpoint emission is unconditional: the manifest embeds its names/shapes/hashes.
    emit_checkpoint(model, report)

    # ---- assemble outputs -------------------------------------------------------------
    out_dir = Path(args.out)
    files: "OrderedDict[str, bytes]" = OrderedDict()

    def split(cap: Capture, want: str) -> dict:
        return {k[len(want):]: v for k, v in cap.floats.items() if k.startswith(want)}

    def split_i(cap: Capture, want: str) -> dict:
        return {k[len(want):]: v for k, v in cap.ints.items() if k.startswith(want)}

    payloads = {
        "prefill_short.safetensors": split(cap_pre, "short."),
        "prefill_long.safetensors": split(cap_pre, "long."),
        "decode_short.safetensors": split(cap_dec, "short."),
        "decode_long.safetensors": split(cap_dec, "long."),
    }
    for name, tensors in payloads.items():
        tmp = out_dir / (name + ".tmp")
        out_dir.mkdir(parents=True, exist_ok=True)
        save_file({k: v for k, v in sorted(tensors.items())}, str(tmp))
        files[name] = tmp.read_bytes()
        tmp.unlink()

    int_payloads = {
        "integers_prefill_short.json": split_i(cap_pre, "short."),
        "integers_prefill_long.json": split_i(cap_pre, "long."),
        "integers_decode_short.json": split_i(cap_dec, "short."),
        "integers_decode_long.json": split_i(cap_dec, "long."),
    }
    for name, obj in int_payloads.items():
        files[name] = (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode()

    tf_commit = getattr(transformers, "__commit__", None) or os.environ.get("QWEN4EXP_TRANSFORMERS_COMMIT", "")
    manifest = OrderedDict(
        schema="mference.qwen4exp.goldens/1",
        generated_by="Scripts/parity/qwen4exp_make_goldens.py",
        family="qwen38flashnext",
        reference=OrderedDict(
            package="transformers",
            version=transformers.__version__,
            git_commit=tf_commit or resolve_transformers_commit(),
            model_class="Qwen4ExpForCausalLM",
            model_type="qwen4_exp_text",
            torch_version=torch.__version__,
            device="cpu",
            attn_implementation="eager",
            deterministic_algorithms=deterministic_algorithms,
            num_threads=1,
            seed=SEED,
        ),
        dtype_policy=OrderedDict(
            model_weights=(
                "float32"
                if args.weight_dtype == "fp32"
                else "bfloat16 values held in float32 (every parameter rounded through bf16 "
                "before capture, so these goldens describe the weights the emitted checkpoint "
                "actually carries)"
            ),
            weight_dtype=args.weight_dtype,
            forward="float32",
            checkpoint="bfloat16",
            rmsnorm=(
                "Qwen3_5RMSNorm (and therefore every Qwen4ExpTextRMSNorm) UPCASTS to float32 "
                "internally: output = _norm(x.float()) * (1.0 + weight.float()), then .type_as(x). "
                "The (1 + w) zero-centered convention is confirmed; the port's `+1` weight bake is "
                "correct, but the fp32 upcast must be preserved for bf16 runtimes."
            ),
            rmsnorm_gated=(
                "Qwen3_5RMSNormGated upcasts hidden_states to fp32 for the variance, applies the "
                "weight in the INPUT dtype (weight * hs.to(input_dtype)), then multiplies by "
                "act(gate.to(float32)) and casts back. Its weight is ones-initialized and is NOT "
                "zero-centered -- do not bake +1 into linear_attn.norm.weight."
            ),
            indexer=(
                "Pooling is float32 (key_groups.float().mean(1)) then cast BACK to the key dtype "
                "before k_layernorm; scoring is float32 (q.float() @ k.float()). RoPE on pooled "
                "keys is applied at the block's FIRST position."
            ),
            router="softmax in float32 -> topk of the probs -> renormalize -> cast to logits dtype",
        ),
        config=json_safe_config(model.config),
        prompts=OrderedDict(
            short=OrderedDict(
                token_ids=short_ids,
                length=SHORT_LEN,
                regime="dense-equivalent (every visible position selected at every attention layer)",
            ),
            long=OrderedDict(
                token_ids=long_ids,
                length=LONG_LEN,
                eos_token_at=LONG_EOS_AT,
                regime="genuinely sparse (selection excludes visible positions)",
            ),
            decode_steps=DECODE_STEPS,
            greedy_rollout_short=gen_short,
            greedy_rollout_long=gen_long,
        ),
        deviations_from_bringup_kit=[
            "num_hidden_layers 4 -> 6, layer_types [lin,lin,lin,qsa,lin,qsa]: the runtime design "
            "doc requires >=2 attention layers so the per-layer indexer key cache is exercised. "
            "PLE still sits on one-indexed layer 2 (layers[1]), a linear_attention layer.",
            "SHORT prompt 12 -> 11 tokens: with indexer_budget 8 / compress_ratio 4 (block_topk 2) "
            "a 12-token prompt already has 3 complete blocks at the last query and is NOT "
            "dense-equivalent. 11 tokens is the largest dense-equivalent length.",
            "rope_parameters.mrope_section set to [1,1,0]: rotary_dim is 4 so there are only 2 "
            "frequency pairs; the transformers default [11,11,10] cannot be expressed. Text-only "
            "positions collapse the sections, so this is semantically inert.",
            "No config value was rejected by the @strict validators.",
        ],
        checkpoint_naming=OrderedDict(
            toy_forcausallm_prefix="model.",
            production_prefix="model.language_model.",
            prefix_warning=(
                "IMPORTANT: text-only Qwen4ExpForCausalLM.save_pretrained emits `model.layers.*` "
                "with NO `language_model` segment, because production Qwen4-Exp ships the "
                "multimodal Qwen4ExpForConditionalGeneration wrapper. The repacker's production "
                "pattern is `model.language_model.layers.{L}.*`. The harness therefore also emits "
                "a production-name copy at scratch/qwen4exp-toy-ckpt-prodlayout/."
            ),
            expert_layout_warning=(
                "IMPORTANT: save_pretrained DE-FUSES the MoE experts back to the source layout, "
                "`mlp.experts.{e}.gate_proj/up_proj/down_proj.weight` (one tensor per expert), "
                "via the qwen2_moe WeightConverter. Production ships FUSED "
                "`mlp.experts.gate_up_proj` [E, 2*I, H] and `mlp.experts.down_proj` [E, H, I]. "
                "The prod-layout copy carries the fused runtime tensors. Row split confirmed: "
                "gate = rows [0, I), up = rows [I, 2I) of gate_up_proj."
            ),
            ngram_shards=(
                "ngram_embedding is written as `...ngram_embedding.shard_{S}.weight`, "
                "split_ngram_parts shards concatenated on dim 0 at load. layer_multipliers, "
                "ngram_heads_vocab_sizes and ngram_heads_offsets stay I64 in a bf16 checkpoint."
            ),
        ),
        port_hazards=[
            "The n-gram hash mix consumes the FULL positive int64 range by construction: "
            "(vocab_size - 1) * max(layer_multipliers) has bit length 63, i.e. zero headroom. "
            "The Swift port must do this in Int64/UInt64 exact integer arithmetic. Float64 "
            "cannot represent these products and Int32 overflows immediately. This holds for "
            "the production vocab too -- multiplier_max is derived as (2^63-1)//vocab_size.",
            "A query position is NOT guaranteed to attend to itself. When the query completes a "
            "block and that block loses the top-k, the query's own key is absent from the "
            "selected set (observed at LONG query 47 on both attention layers: 8 selected out of "
            "48 visible, self excluded). Do not add a 'always keep self' shortcut.",
            "Only COMPLETE compress_ratio blocks participate; the 1..ratio-1 tail is always kept. "
            "The tail is taken from the visible index list, not from raw positions, so a padded "
            "batch changes which tokens are 'tail'.",
            "PLE segmentation uses eos_token_id INCLUSIVE: a shift may not cross an EOS, and the "
            "EOS position itself starts the next segment. Positions that would read across the "
            "boundary read eos_token_id instead. The toy LONG prompt has an EOS at index 20 to "
            "pin this. Production eos_token_id is 248044 (text_config).",
            "Qwen4ExpTextRMSNorm is zero-centered (1 + w) AND upcasts to fp32; "
            "Qwen3_5RMSNormGated (linear_attn.norm) is ones-centered and must NOT get the +1 "
            "bake. Confirmed by reading both classes in the installed package.",
            "Attention q_proj packs query and gate PER HEAD: the output is viewed as "
            "(.., num_heads, 2 * head_dim) and chunked in 2 on the last dim, so head h occupies "
            "rows [h*2*head_dim, (h+1)*2*head_dim) with q first and gate second WITHIN the head. "
            "It is not a global first-half/second-half split.",
        ],
        golden_files=OrderedDict(),
        findings=report,
    )

    manifest["golden_files"] = OrderedDict(
        (name, OrderedDict(bytes=len(blob), sha256=sha256_bytes(blob), contents=describe(name)))
        for name, blob in files.items()
    )

    manifest_blob = (json.dumps(manifest, indent=1) + "\n").encode()

    if args.print_hashes:
        for name, blob in files.items():
            print(f"{sha256_bytes(blob)}  {name}  ({len(blob)} bytes)")
        print(f"{sha256_bytes(manifest_blob)}  goldens-manifest.json  ({len(manifest_blob)} bytes)")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, blob in files.items():
        (out_dir / name).write_bytes(blob)
    (out_dir / "goldens-manifest.json").write_bytes(manifest_blob)

    total = sum(len(b) for b in files.values()) + len(manifest_blob)
    print(f"wrote {len(files) + 1} files, {total / 1024:.1f} KiB total -> {out_dir}")
    for name, blob in files.items():
        print(f"  {name:34s} {len(blob):>9d}  {sha256_bytes(blob)}")
    print(f"  {'goldens-manifest.json':34s} {len(manifest_blob):>9d}  {sha256_bytes(manifest_blob)}")

    print(f"wrote toy checkpoints -> {CKPT_DIR} and {CKPT_PROD_DIR}")
    assert total < 8 * 1024 * 1024, f"committed goldens exceed the 8 MB budget: {total} bytes"
    return 0


def describe(name: str) -> str:
    kind = "prefill" if "prefill" in name else "decode"
    prompt = "SHORT (dense-equivalent)" if "short" in name else "LONG (sparse)"
    if name.endswith(".json"):
        return (
            f"{kind} integer goldens for the {prompt} prompt: per-attention-layer indexer "
            f"selected/visible token index sets (sorted, per query position), per-MoE router "
            f"top-k expert indices, PLE n-gram embedding row ids (16 per token), greedy token ids."
        )
    return (
        f"{kind} float32 goldens for the {prompt} prompt: embed_out; per layer "
        f"stream_out / attn_hc_{{mixed,stream_in,inject}} / mlp_hc_{{mixed,stream_in,inject}} / "
        f"block_out / moe_out / router_weights; PLE layer ple_out + ple_ngram_embeds; "
        f"last_hidden_state; logits."
    )


def json_safe_config(cfg) -> dict:
    d = cfg.to_dict()
    return json.loads(json.dumps(d, default=str, sort_keys=True))


def resolve_transformers_commit() -> str:
    """Read the git commit pip/uv recorded in the dist-info ``direct_url.json``.

    Populated because the venv installs transformers from
    ``git+https://github.com/huggingface/transformers``; a wheel install leaves it blank.
    """
    import importlib.metadata as md

    try:
        dist = md.distribution("transformers")
        root = getattr(dist, "_path", None)
        candidates = []
        if root is not None:
            candidates.append(Path(root) / "direct_url.json")
        site = Path(md.distribution("transformers").locate_file(""))
        candidates.extend(sorted(site.glob("transformers-*.dist-info/direct_url.json")))
        for path in candidates:
            if path.is_file():
                info = json.loads(path.read_text() or "{}")
                commit = info.get("vcs_info", {}).get("commit_id", "")
                if commit:
                    return commit
    except Exception:  # pragma: no cover - the manifest records "" and the README carries it
        pass
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
