# Qwen 3.8 27B performance notes

Qwen 3.8-27B-4bit (`mlx-community/Qwen3.8-27B-4bit`, text stack of the
multimodal checkpoint) is Mference's first dense family: 64 layers — 48
gated-DeltaNet linear-attention and 16 full-attention — with one SwiGLU MLP
per layer, no routed experts, and every weight resident. Per decoded token
the runtime touches ~13 GB of weights, so plain decode is memory-bandwidth
bound; the family's headline speed comes from MTP speculative decoding,
which verifies several drafted tokens against one read of the weights.

All numbers below: 24 GB M5 MacBook Pro, macOS 26.5, `--temperature 0`,
128 generated tokens, fresh process, measured 2026-08-14 at commit
`66c0f10`. The reference baseline is mlx-vlm 0.6.8 on the identical
checkpoint: **6.41 tok/s decode, 40.5 tok/s prefill, 18.6 GB peak** (it
warns it is near the host's working-set ceiling; Mference's text-only
install is ~3.5 GB leaner because the vision tower is excluded at repack).

## Decode

| Configuration | tok/s | vs mlx-vlm |
| --- | ---: | ---: |
| Plain decode | 7.9–8.0 | +25% |
| MTP speculative, k=3 (default) | **15.0–15.1** | **2.35×** |
| MTP speculative, chat mode | 16.9 | 2.64× |

Speculative output is byte-identical to plain greedy decode at every k —
that identity is the acceptance gate, enforced by toy tests with forced
full/partial/zero acceptance and verified on the real model. Accept rate
63.6% at k=3 on prose (80% on chat/math; long essays ~50%, still 1.5×).

The plain-decode path took two accepted optimizations on the way:

- **Chunked prefill** (the qwen36 prefill kernel set; the DeltaNet scan's
  FP32 state is exactly the token-by-token state): prefill 6.0 →
  10.2 tok/s, byte-identical.
- **Fused Hv=48 GDN decode kernel** (`gdn_delta_gated_decode_qwen38`):
  recurrence + gated norm in one dispatch, bit-identical to the generic
  two-dispatch path; 48 fewer dispatches per token. Decode 7.26 → 7.80.
- **tensorops.metal GPUCompiler-32023 fix**: the MPP tensor-ops prefill
  GEMMs had been silently unavailable on macOS 26.5 (naive-QMM fallback).
  Re-enabled: prefill 10.2 → **~60 tok/s** at a 373-token prompt — past
  mlx-vlm's 40.5. This fix also restores the MPP prefill path for the
  other families on macOS 26.5 hosts.

## MTP speculative decoding

The checkpoint ships a native multi-token-prediction draft: one
full-attention transformer layer plus a `fc([norm(embed); norm(hidden)])`
fusion, sharing the target's embedding and lm_head. The mlx 4-bit repo
drops these tensors; `MferenceRepack --attach-mtp <shard>` quantizes them
(INT4 affine group-64 via `Int4AffineEncoder`; norms BF16) from the
original repo's final shard and appends them to an installed
`qwen38.gturbo` in place (+239 MB, ~8 s).

Per round the drafter proposes k tokens; the target verifies all of them
in one pass through decode-exact "multix" kernels (one weight read applied
to k+1 token rows with per-token arithmetic replicating the decode GEMV
and fused-head order bitwise — the prefill QMM/MPP paths are *not*
ulp-identical to decode and cannot be used for verification). DeltaNet
FP32 states and conv tails are blit-captured per verified position; a
partial acceptance restores the arena slot and rewinds the KV cursor, so
a partial round costs the same single weight read as a full one.

Controls: on by default for greedy decode when MTP tensors are present;
`MFERENCE_MTP=0` disables; `MFERENCE_MTP_K` (1–6, default 3) sets the
draft depth. k=3 measured best; k=4+ loses more to accept-rate decay than
it gains in depth.

`MFERENCE_PHASES=1` snapshot (128-token run above): 44 rounds, drafted
132 / accepted 84 (63.6%), 26 rollbacks; draft 875 ms, verify 8391 ms,
accept 503 ms — verify dominates, as it should: it is the one full read
of the weights per round.

## Memory

~15.1 GB install (+239 MB with MTP attached). The fully-resident region
exceeds this host's 13.3 GB `maxBufferLength`, so the loader wraps it as
multiple chunked buffers cut between tensor spans. Decode working set is
the whole install plus ~144 MB of GDN FP32 state and a ~450 MB capture
arena during speculative verify (k=3).

## Measuring

```bash
.build/release/MferenceCLI \
  --model scratch/qwen38-mtp.gturbo \
  --prompt "..." --max-new 128 --temperature 0
```

`MFERENCE_MTP=0` A/Bs the speculative path against plain decode;
outputs must be byte-identical.
