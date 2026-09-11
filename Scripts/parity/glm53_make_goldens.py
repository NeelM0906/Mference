#!/usr/bin/env python3
"""Reference-parity golden harness for the ``glm53flash`` family.

Runs PipeNetwork's GLM-5.3-Flash MLX runtime (``glm53_flash_mlx``, pinned commit
in ``REFERENCE_COMMIT``; it reproduces ``transformers`` 5.16 at 1e-6) as the
golden generator, on the CPU in float32, over a toy ``glm5_next`` configuration
that keeps every axis of the production model live at small size:

* four layers — three Kimi-Delta-Attention layers and one NoPE latent
  sparse-attention layer — so the per-channel-decay recurrence, the depthwise
  conv, the sigmoid-gated output norm, the absorbed MLA fold / unfold and the
  pooled lightning indexer all run;
* ``index_topk 4`` with ``index_kpool 2`` so the pooled selection (two pools
  plus the always-selected tail) is active from the fifth token of a 48-token
  prompt, and the dense bypass is exercised on the first four;
* one leading dense layer, then 16 routed experts top-8 (the production
  width, the one the INT4 expert reduce implements) plus one shared expert,
  sigmoid routing with a selection-only correction bias, weights renormalized
  after selection and scaled by 2.5, and ``swiglu_limit 0.5`` so the clamp bites;
* a 4-stream mHC residual (20 Sinkhorn sweeps) collapsed by the stream mean.

Weights are drawn from a fixed seed and then **rounded through the storage
PipeNetwork's conversion uses** — INT8 affine group-64 for every projection
the conversion's ``quantization`` map names (attention, indexer, shared and
dense FFNs, embedding, head), INT4 for the stacked routed experts, BF16 for the
router gate, the mHC ``fn`` arrays, the depthwise conv, the indexer pooling
gate / ape and every norm, FP32 for the mHC ``base`` / ``scale``, the KDA
``A_log`` / ``dt_bias`` and the router bias. The reference forward runs on the
dequantized values, and the checkpoint emitted next to the goldens carries
exactly those stored bytes, so a Swift forward over the installed checkpoint
measures the *port*, not the quantizer.

Every discrete decision the port has to reproduce exactly — pooled-indexer
selections, router top-8, greedy argmax — is audited for boundary ties and the
smallest margin is recorded in the manifest; the seed is chosen so no boundary
is tied.

Captures are taken on the **per-token** path (one token per forward, cached),
which is the shape the Mference runner takes, and the batched single-forward
logits are recorded beside them; the reference's own agreement between the
two is measured, not assumed.

Usage
-----
    <venv>/bin/python Scripts/parity/glm53_make_goldens.py --reference <clone of glm53-flash-mlx>
    <venv>/bin/python Scripts/parity/glm53_make_goldens.py --scan --seed N

See Scripts/parity/README.md ("glm53flash") for the venv and the clone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np  # noqa: E402

SEED = 12   # chosen by `--scan` over seeds 1-20 at unit weight scale with 16 experts top-8: no indexer boundary below its floor, the fewest router boundaries below the floor (2 of 276, narrowest 1.6e-3), 19 zero-score indexer ties
REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "Tests" / "Mference" / "Fixtures" / "glm53"
CKPT_DIR = FIXTURES / "toy-ckpt"
DEFAULT_REFERENCE = REPO_ROOT / "scratch" / "glm53-flash-mlx"
REFERENCE_REPO = "https://github.com/PipeNetwork/glm53-flash-mlx"
REFERENCE_COMMIT = "a61a7c7d2fbdf3d218a9909365a24bd794f3a247"
GROUP = 64

SHORT_LEN, LONG_LEN, DECODE_STEPS = 12, 48, 16
MIN_ARGMAX_MARGIN = 0.05       # logits: top-1 minus top-2
MIN_ROUTER_MARGIN = 2e-3       # biased sigmoid score: k-th minus (k+1)-th
MIN_INDEXER_MARGIN = 2e-3      # pooled index score: k-th minus (k+1)-th pool
FLOORS = {"argmax": MIN_ARGMAX_MARGIN, "router": MIN_ROUTER_MARGIN, "indexer": MIN_INDEXER_MARGIN}

TOY = dict(
    model_type="glm5_next_text", vocab_size=256, hidden_size=128, intermediate_size=128,
    moe_intermediate_size=64, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=2,
    n_shared_experts=1, n_routed_experts=16, routed_scaling_factor=2.5, kv_lora_rank=64,
    q_lora_rank=64, qk_rope_head_dim=0, v_head_dim=64, qk_nope_head_dim=64, qk_head_dim=64,
    n_group=1, topk_group=1, num_experts_per_tok=8, norm_topk_prob=True, hidden_act="silu",
    max_position_embeddings=4096, rms_norm_eps=1e-5, first_k_dense_replace=1,
    index_topk=4, index_head_dim=64, index_n_heads=2, head_dim=0, index_kpool=2,
    index_kpool_compress=True, index_kpool_always_select_tail=True, indexer_rope_interleave=True,
    index_share_for_mtp_iteration=True,
    layer_types=["linear_attention"] * 3 + ["deepseek_sparse_attention"],
    indexer_types=["full"] * 4, mlp_layer_types=["dense"] + ["sparse"] * 3,
    linear_attn_config={"num_heads": 2, "head_dim": 64, "short_conv_kernel_size": 4,
                        "gate_lower_bound": -5.0, "kda_layers": [0, 1, 2], "full_attn_layers": [3]},
    swiglu_limit=0.5, hc_mult=4, hc_eps=1e-6, hc_sinkhorn_iters=20, mhc=True, mla_use_nope=True,
    moe_router_dtype="float32", num_nextn_predict_layers=0, scoring_func="sigmoid",
    topk_method="noaux_tc", attention_bias=False, tie_word_embeddings=False,
    pad_token_id=0, eos_token_id=[1],
)

LM = "language_model."


# ----------------------------------------------------------------------------
# reference import
# ----------------------------------------------------------------------------

def import_reference(path: Path):
    if not (path / "glm53_flash_mlx" / "glm5_next" / "language.py").exists():
        sys.exit(f"reference not found at {path}; clone {REFERENCE_REPO} there "
                 f"(commit {REFERENCE_COMMIT}) or pass --reference")
    try:
        head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        head = "unknown"
    if head != REFERENCE_COMMIT:
        print(f"WARNING: reference HEAD {head} != pinned {REFERENCE_COMMIT}", file=sys.stderr)
    sys.path.insert(0, str(path))
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    import mlx.nn as nn
    import glm53_flash_mlx.glm5_next.language as LANG
    from glm53_flash_mlx.glm5_next.config import TextConfig
    from mlx_vlm.models import gated_delta as GD
    from mlx_vlm.models.deepseek_v4 import hyper_connection as HC
    return mx, dict(nn=nn, LANG=LANG, TextConfig=TextConfig, GD=GD, HC=HC, head=head)


# ----------------------------------------------------------------------------
# weights
# ----------------------------------------------------------------------------

class Stored:
    """Tensors the toy checkpoint carries, in PipeNetwork's conversion layout."""

    def __init__(self, mx):
        self.mx = mx
        self.main = {}        # checkpoint name -> mx.array (stored dtype)
        self.modules = {}     # quantization per-module overrides

    def bf16(self, name, w):
        a = self.mx.array(np.asarray(w, np.float32)).astype(self.mx.bfloat16)
        self.main[LM + name] = a
        return a.astype(self.mx.float32)

    def f32(self, name, w):
        a = self.mx.array(np.asarray(w, np.float32))
        self.main[LM + name] = a
        return a

    def quant(self, base, w, bits):
        """MLX affine group-64 at `bits` from the BF16-rounded weight; return the
        fp32 dequantized values (fp32 scales, so the reconstruction is exact)."""
        mx = self.mx
        a = mx.array(np.asarray(w, np.float32)).astype(mx.bfloat16)
        wq, s, b = mx.quantize(a, group_size=GROUP, bits=bits)
        mx.eval(wq, s, b)
        self.main[LM + base + ".weight"] = wq
        self.main[LM + base + ".scales"] = s
        self.main[LM + base + ".biases"] = b
        if bits != 4:
            self.modules[LM + base] = {"group_size": GROUP, "bits": bits}
        deq = mx.dequantize(wq, s.astype(mx.float32), b.astype(mx.float32),
                            group_size=GROUP, bits=bits)
        mx.eval(deq)
        return deq


def build_weights(mx, model, rng):
    """Seeded weights at unit fan-in scale (mHC arrays and KDA gates randomized
    as in PipeNetwork's parity test), rounded through PipeNetwork's storage
    policy and assigned to `model`."""
    from mlx.utils import tree_flatten, tree_unflatten
    st = Stored(mx)
    params = dict(tree_flatten(model.parameters()))
    assigned = {}

    def w(shape, s=None):
        # Unit-fan-in scale. PipeNetwork's own parity test scales projections
        # x3 to push activations past the clamp; at swiglu_limit 0.5 the clamp
        # already bites on ~60 % of pre-activations at 1x, and the 3x toy
        # amplifies fp32 accumulation-order noise ~1e3-fold (its own batched
        # vs per-token logits disagree at 5e-4), which would hide a real
        # oracle defect behind a loose tolerance.
        fan = shape[-1]
        s = s if s is not None else fan ** -0.5
        return (rng.standard_normal(shape) * s).astype(np.float32)

    for name, p in params.items():
        shape = tuple(p.shape)
        base = name[:-len(".weight")] if name.endswith(".weight") else name
        if name.endswith((".attn_hc.fn", ".ffn_hc.fn")):
            v = st.bf16(name, rng.standard_normal(shape) * 0.5)
        elif name.endswith((".attn_hc.base", ".ffn_hc.base")):
            v = st.f32(name, rng.standard_normal(shape) * 1.0)
        elif name.endswith((".attn_hc.scale", ".ffn_hc.scale")):
            v = st.f32(name, rng.uniform(0.5, 1.5, shape))
        elif name.endswith((".A_log", ".dt_bias", ".e_score_correction_bias")):
            v = st.f32(name, rng.standard_normal(shape) * 1.0)
        elif name.endswith(".conv1d.weight"):
            v = st.bf16(name, rng.standard_normal(shape) * 0.5)
        elif name.endswith((".index_kpool_compress_ape", ".index_kpool_compress_gate")):
            v = st.bf16(name, rng.standard_normal(shape) * 0.5)
        elif name.endswith(".mlp.gate.weight"):
            v = st.bf16(name, w(shape))
        elif name.endswith(".k_norm.bias"):
            v = st.bf16(name, 0.3 * rng.standard_normal(shape))
        elif "norm" in name and len(shape) == 1:
            v = st.bf16(name, 1.0 + 0.3 * rng.standard_normal(shape))
        elif ".switch_mlp." in name:
            v = st.quant(base, w(shape), 4)
        elif name.endswith(".weight") and len(shape) >= 2:
            v = st.quant(base, w(shape), 8)
        else:
            raise RuntimeError(f"unclassified parameter {name} {shape}")
        assigned[name] = v
    model.update(tree_unflatten(list(assigned.items())))
    mx.eval(model.parameters())
    return st


# ----------------------------------------------------------------------------
# recording
# ----------------------------------------------------------------------------

class Recorder:
    def __init__(self):
        self.phase = "seq.pos000"
        self.layer = -1
        self.tensors = {}
        self.ints = {}
        self.margins = {"argmax": [], "router": [], "indexer": []}
        self.near = {"argmax": [], "router": [], "indexer": []}
        self.ties = []
        self.enabled = True
        self.names = {}
        self.zero_score_ties = 0

    def key(self, name, layer=True):
        if layer:
            return f"{self.phase}.layer{self.layer:02d}.{name}"
        return f"{self.phase}.{name}"

    def put(self, name, arr, layer=True):
        if not self.enabled:
            return
        self.tensors[self.key(name, layer)] = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))

    def put_ints(self, name, value, layer=True):
        if not self.enabled:
            return
        self.ints[self.key(name, layer)] = value

    def margin(self, kind, value, where):
        if not self.enabled:
            return
        self.margins[kind].append(float(value))
        if value == 0:
            self.ties.append(where)
        elif value < FLOORS[kind]:
            self.near[kind].append({"where": where, "margin": float(value)})


REC = Recorder()


def install_hooks(mx, R, model):
    nn, LANG, GD, HC = R["nn"], R["LANG"], R["GD"], R["HC"]
    for name, module in model.named_modules():
        REC.names[id(module)] = name
    for layer in model.model.layers:
        layer.compile_ffn = False     # keep the Python hooks live at (1, 1)

    # -- decoder layer: layer index, stream in / out ------------------------
    orig_layer = LANG.Glm5NextDecoderLayer.__call__

    def layer_call(self, x, mask=None, cache=None):
        REC.layer = next(i for i, l in enumerate(model.model.layers) if l is self)
        REC.put("stream_in", np.asarray(x, np.float32)[0, -1])
        out = orig_layer(self, x, mask, cache)
        REC.put("stream_out", np.asarray(out, np.float32)[0, -1])
        return out
    LANG.Glm5NextDecoderLayer.__call__ = layer_call

    # -- mHC: pre / post / comb and the collapsed input -----------------------
    orig_hc = HC.HyperConnection.__call__
    orig_split = HC._hc_split_sinkhorn_ops

    def split_rec(mixes, scale, base, hc_mult, iters, eps):
        pre, post, comb = orig_split(mixes, scale, base, hc_mult, iters, eps)
        split_rec.last = (pre, post, comb)
        return pre, post, comb
    HC._hc_split_sinkhorn_ops = split_rec

    def hc_call(self, x):
        collapsed, post, comb = orig_hc(self, x)
        site = "attn" if REC.names[id(self)].endswith("attn_hc") else "ffn"
        pre = split_rec.last[0]
        REC.put(f"{site}_hc_pre", np.asarray(pre, np.float32)[0, -1])
        REC.put(f"{site}_hc_post", np.asarray(post, np.float32)[0, -1])
        REC.put(f"{site}_hc_comb", np.asarray(comb, np.float32)[0, -1])
        REC.put(f"{site}_collapsed", np.asarray(collapsed, np.float32)[0, -1])
        return collapsed, post, comb
    HC.HyperConnection.__call__ = hc_call

    # -- every RMSNorm / LayerNorm output, by module name -----------------------
    for cls in (nn.RMSNorm, nn.LayerNorm):
        orig = cls.__call__

        def norm_call(self, x, _orig=orig):
            out = _orig(self, x)
            name = REC.names.get(id(self))
            if name is not None:
                short = name.split(".layers.")[-1]
                short = short.split(".", 1)[1] if ".layers." in name else name
                REC.put(short.replace(".", "_") + "_out", np.asarray(out, np.float32)[0, -1],
                        layer=(".layers." in name))
            return out
        cls.__call__ = norm_call

    # -- KDA: the recurrence's inputs and outputs, g / beta, the state --------
    orig_gdu = LANG.gated_delta_update

    def gdu_rec(q, k, v, a, b, A_log, dt_bias, state=None, lower_bound=None, **kw):
        out, new_state = orig_gdu(q, k, v, a, b, A_log, dt_bias, state=state,
                                  lower_bound=lower_bound, **kw)
        mx.eval(out, new_state)
        beta = mx.sigmoid(b)
        g = GD.compute_g_safe(A_log, a, dt_bias, lower_bound)
        REC.put("kda_q", np.asarray(q, np.float32)[0, -1])          # [H, D], normed & scaled
        REC.put("kda_k", np.asarray(k, np.float32)[0, -1])
        REC.put("kda_v", np.asarray(v, np.float32)[0, -1])
        REC.put("kda_decay", np.asarray(g, np.float32)[0, -1])      # exp(g): [H, D]
        REC.put("kda_beta", np.asarray(beta, np.float32)[0, -1])    # [H]
        REC.put("kda_y", np.asarray(out, np.float32)[0, -1])        # [H, Dv]
        gdu_rec.last_state = new_state
        return out, new_state
    LANG.gated_delta_update = gdu_rec

    orig_lin = LANG.Glm5NextLinearAttention.__call__

    def lin_call(self, inputs, mask=None, cache=None):
        # Instrumented re-derivation of the conv stage (the reference does not
        # expose it), cross-checked against the reference's own output.
        conv_state = cache[0] if (cache is not None and cache[0] is not None) else None
        out = orig_lin(self, inputs, mask, cache)
        x = inputs
        mixed = mx.concatenate([self.q_proj(x), self.k_proj(x), self.v_proj(x)], axis=-1)
        if conv_state is None:
            conv_state = mx.zeros((1, self.conv_kernel_size - 1, self.conv_dim), dtype=x.dtype)
        conv_in = mx.concatenate([conv_state, mixed], axis=1)
        conv_out = nn.silu(self.conv1d(conv_in))
        REC.put("kda_mixed", np.asarray(mixed, np.float32)[0, -1])
        REC.put("kda_conv_out", np.asarray(conv_out, np.float32)[0, -1])
        REC.put("kda_gate", np.asarray(self.g_b_proj(self.g_a_proj(x)), np.float32)[0, -1])
        REC.put("attn_out", np.asarray(out, np.float32)[0, -1])
        return out
    LANG.Glm5NextLinearAttention.__call__ = lin_call

    # -- sparse attention: q, latent, fold, selection, output -------------------
    orig_sparse = LANG.Glm5NextSparseAttention.__call__

    def sparse_call(self, x, mask=None, cache=None):
        out = orig_sparse(self, x, mask, cache)
        qr = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(qr).reshape(1, x.shape[1], self.num_heads, self.q_head_dim)
        latent = self.kv_a_layernorm(self.kv_a_proj_with_mqa(x))
        REC.put("dsa_qr", np.asarray(qr, np.float32)[0, -1])
        REC.put("dsa_q", np.asarray(q, np.float32)[0, -1])                 # [H, 64]
        REC.put("dsa_latent_new", np.asarray(latent, np.float32)[0, -1])   # [kv_lora]
        q_lat = self.embed_q(q.transpose(0, 2, 1, 3))                       # [1, H, S, kv_lora]
        REC.put("dsa_q_latent", np.asarray(q_lat, np.float32)[0, :, -1])   # [H, kv_lora]
        REC.put("attn_out", np.asarray(out, np.float32)[0, -1])
        return out
    LANG.Glm5NextSparseAttention.__call__ = sparse_call

    orig_idx = LANG.Glm5NextIndexer.__call__

    def idx_call(self, x, qr, mask, cache=None):
        topk = orig_idx(self, x, qr, mask, cache)
        S = x.shape[1]
        packed = cache.keys[:, 0] if (cache is not None and cache.keys is not None) else None
        if packed is not None:
            T = cache.offset
            k_full, gate_full, valid_ch = mx.split(packed[:, :T], [self.head_dim, 2 * self.head_dim], axis=-1)
            REC.put("idx_k_new", np.asarray(k_full, np.float32)[0, -1])
            REC.put("idx_gate_new", np.asarray(gate_full, np.float32)[0, -1])
            REC.put_ints("idx_visible", int(T))
            if topk is None:
                REC.put_ints("idx_selected", "dense")
            else:
                valid = valid_ch[..., 0] > 0
                pool_keys, pool_indices, pool_valid = self._pooled_states(k_full, gate_full, valid)
                q = self.wq_b(qr).reshape(1, S, self.n_heads, self.head_dim)[:, -1:]
                scores = mx.maximum((q @ pool_keys[:, None].swapaxes(-1, -2)) * self.softmax_scale, 0.0)
                weights = self.weights_proj(x[:, -1:]) * (self.n_heads ** -0.5)
                index_scores = mx.sum(weights[..., None] * scores, axis=2)[0, 0]   # [P]
                P = pool_keys.shape[1]
                pool_end = np.asarray(pool_indices, np.int64)[0, :, -1]
                visible = (pool_end <= T - 1) & np.asarray(pool_valid, bool)[0]
                sc = np.asarray(index_scores, np.float32)
                sc_masked = np.where(visible, sc, -np.inf)
                select_k = min(self.index_topk // self.index_kpool, P)
                order = np.argsort(-sc_masked, kind="stable")
                chosen = [int(i) for i in order[:select_k] if visible[i]]
                if len(chosen) < visible.sum() and len(chosen) < visible.sum():
                    pass
                if visible.sum() > select_k:
                    kth, nxt = float(sc_masked[order[select_k - 1]]), float(sc_masked[order[select_k]])
                    if kth == 0.0 and nxt == 0.0:
                        # Both pools scored exactly zero (every head's relu
                        # clamped): the reference's stable argsort keeps the
                        # lower pool index, a rule the port reproduces, so this
                        # is a deterministic zero-score tie, not a boundary.
                        REC.zero_score_ties += 1
                    else:
                        REC.margin("indexer", kth - nxt, f"{REC.phase}.layer{REC.layer:02d}.indexer")
                REC.put("idx_pool_keys", np.asarray(pool_keys, np.float32)[0])
                REC.put("idx_q", np.asarray(q, np.float32)[0, 0])
                REC.put("idx_weights", np.asarray(weights, np.float32)[0, 0])
                REC.put("idx_scores", sc)
                REC.put_ints("idx_pool_visible", [bool(v) for v in visible])
                sel = [int(t) for t in np.asarray(topk, np.int64)[0, 0, -1] if t >= 0]
                REC.put_ints("idx_selected", sorted(sel))
                ref_set = set(sel)
                mine = set()
                for pidx in chosen:
                    mine.update(int(t) for t in np.asarray(pool_indices, np.int64)[0, pidx] if t >= 0)
                tail_count = T - (T // self.index_kpool) * self.index_kpool
                mine.update(range(T - tail_count, T))
                assert mine == ref_set, (REC.phase, REC.layer, sorted(mine), sorted(ref_set))
        return topk
    LANG.Glm5NextIndexer.__call__ = idx_call

    # -- router and MoE ---------------------------------------------------------
    orig_gate = LANG.Glm5NextMoEGate.__call__

    def gate_call(self, x):
        inds, scores = orig_gate(self, x)
        logits = (x.astype(mx.float32) @ self.weight.astype(mx.float32).T)[0, -1]
        sig = mx.sigmoid(logits)
        biased = np.asarray(sig + self.e_score_correction_bias, np.float64)
        srt = -np.sort(-biased)
        k = self.top_k
        REC.margin("router", float(srt[k - 1] - srt[k]), f"{REC.phase}.layer{REC.layer:02d}.router")
        REC.put("router_logits", np.asarray(logits, np.float32))
        REC.put("router_scores", np.asarray(sig, np.float32))
        REC.put_ints("router_indices", [int(i) for i in np.asarray(inds, np.int64)[0, -1]])
        REC.put("router_weights", np.asarray(scores, np.float32)[0, -1])
        return inds, scores
    LANG.Glm5NextMoEGate.__call__ = gate_call

    orig_moe = LANG.Glm5NextMoE.__call__

    def moe_call(self, x):
        out = orig_moe(self, x)
        REC.put("shared_out", np.asarray(self.shared_experts(x), np.float32)[0, -1])
        REC.put("mlp_out", np.asarray(out, np.float32)[0, -1])
        return out
    LANG.Glm5NextMoE.__call__ = moe_call

    orig_dense = LANG.ClampedMLP.__call__

    def dense_call(self, x):
        out = orig_dense(self, x)
        name = REC.names.get(id(self), "")
        if name.endswith(".mlp"):
            REC.put("mlp_out", np.asarray(out, np.float32)[0, -1])
        return out
    LANG.ClampedMLP.__call__ = dense_call

    orig_model = LANG.Glm5NextModel.__call__

    def model_call(self, inputs, cache=None, inputs_embeds=None):
        out = orig_model(self, inputs, cache=cache, inputs_embeds=inputs_embeds)
        REC.put("embed_out", np.asarray(self.embed_tokens(inputs), np.float32)[0, -1], layer=False)
        REC.put("final_norm_out", np.asarray(out, np.float32)[0, -1], layer=False)
        return out
    LANG.Glm5NextModel.__call__ = model_call


# ----------------------------------------------------------------------------
# running
# ----------------------------------------------------------------------------

def argmax_margins(logits):
    srt = -np.sort(-np.asarray(logits, np.float64), axis=-1)
    return srt[..., 0] - srt[..., 1]


def snapshot_state(mx, model, cache, phase):
    for L, layer in enumerate(model.model.layers):
        c = cache[L]
        if layer.is_linear:
            REC.tensors[f"{phase}.layer{L:02d}.kda_conv_state"] = np.asarray(c[0], np.float32)[0]   # [K-1, 3*qkv]
            REC.tensors[f"{phase}.layer{L:02d}.kda_state"] = np.asarray(c[1], np.float32)[0]        # [H, Dv, Dk]
        else:
            kv = c[0]
            REC.tensors[f"{phase}.layer{L:02d}.dsa_latent_cache"] = np.asarray(kv.keys, np.float32)[0, 0, :kv.offset]
            ic = c[1]
            REC.tensors[f"{phase}.layer{L:02d}.idx_packed_cache"] = np.asarray(ic.keys, np.float32)[0, 0, :ic.offset]


def forward(mx, model, ids, cache):
    return model(mx.array([ids]), cache=cache).logits


def run_prompt(mx, model, ids, name):
    """Per-token prompt pass (recorded), then DECODE_STEPS greedy steps."""
    REC.tensors.clear()
    REC.ints.clear()
    cache = model.make_cache()
    seq_logits = []
    for p, tok in enumerate(ids):
        REC.phase = f"seq.pos{p:03d}"
        REC.layer = -1
        lg = np.asarray(forward(mx, model, [tok], cache), np.float32)[0, 0]
        seq_logits.append(lg)
        REC.tensors[f"{REC.phase}.logits"] = lg
        REC.margin("argmax", float(argmax_margins(lg)), f"{name}.{REC.phase}.argmax")
    snapshot_state(mx, model, cache, "seq.final")
    seq_logits = np.stack(seq_logits)
    prompt_ints = dict(REC.ints)
    prompt_ints["argmax_all_positions"] = [int(v) for v in seq_logits.argmax(-1)]
    prompt_ints["argmax_margins"] = [float(v) for v in argmax_margins(seq_logits)]
    prompt_tensors = dict(REC.tensors)
    prompt_tensors["seq.logits"] = seq_logits

    REC.tensors.clear()
    REC.ints.clear()
    rollout, step_logits = [], []
    token = int(seq_logits[-1].argmax())
    for s in range(DECODE_STEPS):
        REC.phase = f"decode.step{s:02d}"
        REC.layer = -1
        rollout.append(token)
        row = np.asarray(forward(mx, model, [token], cache), np.float32)[0, 0]
        step_logits.append(row)
        REC.tensors[f"{REC.phase}.logits"] = row
        REC.margin("argmax", float(argmax_margins(row)), f"{name}.{REC.phase}.argmax")
        token = int(row.argmax())
    snapshot_state(mx, model, cache, "decode.final")
    decode_tensors = dict(REC.tensors)
    decode_tensors["decode.step_logits"] = np.stack(step_logits)
    decode_ints = dict(REC.ints)
    decode_ints["greedy_rollout"] = rollout
    decode_ints["greedy_next"] = [int(r.argmax()) for r in step_logits]
    decode_ints["step_margins"] = [float(argmax_margins(r)) for r in step_logits]

    # The reference's own consistency: one batched prefill, chunked prefill,
    # and an uncached recompute of every decode step.
    REC.enabled = False
    single = np.asarray(forward(mx, model, list(ids), model.make_cache()), np.float32)[0]
    prompt_tensors["prefill.logits"] = single
    prompt_ints["sequential_vs_single_max_abs"] = float(np.abs(single - seq_logits).max())
    chunks = [(0, 5), (5, 6), (6, min(13, len(ids)))]
    if len(ids) > 13:
        chunks += [(13, 31), (31, len(ids))]
    fresh = model.make_cache()
    parts = [np.asarray(forward(mx, model, list(ids[a:b]), fresh), np.float32)[0] for a, b in chunks]
    prompt_ints["chunked_vs_single_max_abs"] = float(np.abs(np.concatenate(parts) - single).max())
    prompt_ints["chunks"] = [list(c) for c in chunks]
    worst = 0.0
    seq = list(ids)
    for s in range(DECODE_STEPS):
        seq.append(rollout[s])
        full = np.asarray(forward(mx, model, seq, model.make_cache()), np.float32)[0, -1]
        worst = max(worst, float(np.abs(full - step_logits[s]).max()))
        assert int(full.argmax()) == decode_ints["greedy_next"][s], "uncached rollout diverged"
    decode_ints["uncached_vs_cached_max_abs"] = worst
    REC.enabled = True
    return prompt_tensors, prompt_ints, decode_tensors, decode_ints


# ----------------------------------------------------------------------------
# checkpoint
# ----------------------------------------------------------------------------

def write_checkpoint(mx, st: Stored):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    for old in CKPT_DIR.glob("*"):
        old.unlink()
    main_name = "model-00001.safetensors"
    extras = {   # the vision tower the planner drops
        "vision_model.patch_embed.proj.weight": mx.zeros((32, 12), dtype=mx.bfloat16),
        "vision_model.blocks.0.attn.qkv.weight": mx.zeros((96, 32), dtype=mx.bfloat16),
        "vision_model.merger.proj.weight": mx.zeros((TOY["hidden_size"], 32), dtype=mx.bfloat16),
    }
    main = dict(sorted({**st.main, **extras}.items()))
    mx.save_safetensors(str(CKPT_DIR / main_name), main)
    total = os.path.getsize(CKPT_DIR / main_name)
    (CKPT_DIR / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": {k: main_name for k in main}},
        indent=1, sort_keys=True) + "\n")
    quantization = {"group_size": GROUP, "bits": 4, **dict(sorted(st.modules.items()))}
    config = {
        "architectures": ["Glm5NextForConditionalGeneration"],
        "model_type": "glm5_next",
        "text_config": TOY,
        "vision_config": {"model_type": "glm5_next_vision", "depth": 1, "hidden_size": 32},
        "tie_word_embeddings": False,
        "quantization": quantization,
    }
    (CKPT_DIR / "config.json").write_text(json.dumps(config, indent=1, sort_keys=True) + "\n")
    (CKPT_DIR / "tokenizer_config.json").write_text(json.dumps(
        {"note": "toy checkpoint for the glm53flash parity gates; no tokenizer"}) + "\n")


def save_tensors(path: Path, tensors: dict):
    from safetensors.numpy import save_file
    ordered = {k: np.ascontiguousarray(tensors[k], dtype=np.float32) for k in sorted(tensors)}
    save_file(ordered, str(path))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    global SEED
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    ap.add_argument("--print-hashes", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--scan", action="store_true", help="audit margins only; write nothing")
    args = ap.parse_args()
    SEED = args.seed

    mx, R = import_reference(args.reference)
    model = R["LANG"].LanguageModel(R["TextConfig"].from_dict(TOY))
    rng = np.random.default_rng(SEED)
    st = build_weights(mx, model, rng)
    install_hooks(mx, R, model)

    prompts = {}
    for name, n in (("short", SHORT_LEN), ("long", LONG_LEN)):
        prompts[name] = [int(v) for v in rng.integers(2, TOY["vocab_size"], size=n)]

    FIXTURES.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, ids in prompts.items():
        pt, pi, dt, di = run_prompt(mx, model, ids, name)
        outputs[f"prompt_{name}.safetensors"] = pt
        outputs[f"decode_{name}.safetensors"] = dt
        if not args.scan:
            (FIXTURES / f"integers_prompt_{name}.json").write_text(json.dumps(pi, sort_keys=True))
            (FIXTURES / f"integers_decode_{name}.json").write_text(json.dumps(di, sort_keys=True))
        print(f"  {name}: {len(ids)} prompt tokens, {DECODE_STEPS} decode steps; "
              f"sequential-vs-single {pi['sequential_vs_single_max_abs']:.2e}, "
              f"chunked-vs-single {pi['chunked_vs_single_max_abs']:.2e}, "
              f"uncached-vs-cached {di['uncached_vs_cached_max_abs']:.2e}")
    margins = {k: (min(v) if v else None) for k, v in REC.margins.items()}
    counts = {k: len(v) for k, v in REC.margins.items()}
    below = {k: sum(1 for m in v if m < FLOORS[k]) for k, v in REC.margins.items()}
    print(f"  seed {SEED}: smallest boundary margins",
          {k: (f"{v:.3e}" if v is not None else None) for k, v in margins.items()},
          "decisions", counts, "below floor", below, "ties", len(REC.ties),
          "zero-score indexer ties", REC.zero_score_ties)
    if args.scan:
        return
    if REC.ties:
        sys.exit(f"boundary ties at {REC.ties[:8]}; choose another seed")
    for fname, tensors in outputs.items():
        save_tensors(FIXTURES / fname, tensors)
    write_checkpoint(mx, st)
    manifest = {
        "family": "glm53Flash",
        "reference": {"repo": REFERENCE_REPO, "commit": R["head"], "pinned_commit": REFERENCE_COMMIT,
                      "mlx": mx.__version__, "device": "cpu", "dtype": "float32"},
        "seed": SEED,
        "config": TOY,
        "prompts": {k: {"ids": v} for k, v in prompts.items()},
        "decode_steps": DECODE_STEPS,
        "capture_layout": {
            "prompt": "seq.pos{p:03d}.* per prompt token on the one-token cached path; "
                      "seq.logits [S, V]; prefill.logits [S, V] from one batched forward",
            "decode": "decode.step{s:02d}.*; decode.step_logits [steps, V]",
            "state": "seq.final.* and decode.final.*: kda_conv_state [K-1, 3*qkv], "
                     "kda_state [H, Dv, Dk], dsa_latent_cache [T, kv_lora], idx_packed_cache [T, 2*idx_dim+1]",
        },
        "margins": {"min": margins, "decisions": counts, "below_floor": below, "floors": FLOORS,
                    "below_floor_locations": REC.near,
                    "indexer_zero_score_ties": REC.zero_score_ties,
                    "note": "exact-zero ties are rejected except where both boundary pools score "
                            "exactly zero, where the stable ascending-index order is the rule"},
        "storage_policy": {
            "int8_g64": "every Linear the conversion's quantization map names: KDA q/k/v/o, f_a/f_b, "
                        "g_a/g_b, b_proj; q_a/q_b/kv_a/embed_q/unembed_out/o_proj; indexer wq_b/wk/"
                        "weights_proj; shared and dense FFNs; embed_tokens; lm_head",
            "int4_g64": "mlp.switch_mlp.{gate,up,down}_proj (stacked)",
            "bf16": "mlp.gate.weight, attn_hc/ffn_hc fn, conv1d, indexer k_norm weight/bias, "
                    "index_kpool_compress_gate/ape, every RMSNorm weight",
            "f32": "attn_hc/ffn_hc base and scale, A_log, dt_bias, e_score_correction_bias",
        },
        "files": {},
    }
    for p in sorted(FIXTURES.glob("*.safetensors")) + sorted(FIXTURES.glob("integers_*.json")):
        manifest["files"][p.name] = {"size": p.stat().st_size, "sha256": sha256(p)}
    for p in sorted(CKPT_DIR.glob("*")):
        manifest["files"][f"toy-ckpt/{p.name}"] = {"size": p.stat().st_size, "sha256": sha256(p)}
    (FIXTURES / "goldens-manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    if args.print_hashes:
        for k, v in manifest["files"].items():
            print(f"{v['sha256']}  {k}")
    total = sum(v["size"] for v in manifest["files"].values())
    print(f"  wrote {len(manifest['files'])} files, {total / 1e6:.2f} MB, to {FIXTURES}")


if __name__ == "__main__":
    main()
