# Scoping note: the MiniCPM5-2B DSpark drafter on a `RoundDrafter` protocol

Status: scoping only, written after the `minicpm5` family gate went green
(2026-09-10). No code. Follow-on project, not part of the bring-up.

## What the vendor ships

[`openbmb/MiniCPM5-2B-DSpark`](https://huggingface.co/openbmb/MiniCPM5-2B-DSpark)
(revision `114a20fd`, one BF16 `model.safetensors`, 62 tensors,
323,776,001 params) is a DSpark block-diffusion draft model trained for exact
pairing with MiniCPM5-2B. Read from its `config.json` and safetensors header:

- **A 5-layer `qwen3`-architecture stack** at the target's width (hidden 2048,
  16 heads / 2 KV heads of 128, intermediate 6144, SiLU, RoPE theta 5e6,
  `rms_norm_eps` 1e-6) — **with per-head `q_norm` / `k_norm`** (`[128]` each),
  i.e. the `qkNorm = true` attention this family's own runner does not run.
- **Conditioning by KV injection from the target:** `target_layer_ids
  [1, 10, 20, 30, 39]`; the five tapped hidden states (5 × 2048 = 10,240)
  are concatenated and projected by `fc` `[2048, 10240]` after
  `hidden_norm` `[2048]` (`projector_type "dspark"`).
- **Block diffusion:** `block_size 7` (anchor + 6 masked slots),
  `mask_token_id 75982`; embedding and `lm_head` are the target's own
  (`draft_vocab_size 130560`, no embedding tensors in the drafter).
- **A confidence head** (`confidence_head.proj` `[1, 2304]` + bias) and a
  **Markov head** (`markov_w1` / `markov_w2` `[130560, 256]`,
  `markov_rank 256`, `markov_head_type "vanilla"`,
  `confidence_head_with_markov true`) that together decide how much of the
  block to trust before verification.
- Vendor evaluation (natural EOS, `max_new_tokens 4096`): acceptance length
  **5.52 at T=0** and **4.05 at T=1.0** aggregate (math 6.05 / code 6.11 /
  general 4.16 at T=0).

## What the tree already has

Two draft sources for Qwen 3.8, both hard-wired to `Qwen38ForwardRunner`:

- `Qwen38MTPSpeculator` — owns the round: `probe`, `canRunRound`,
  `runRound(bonus:position:)`, `consumePending`, `prepareForContinuation`
  (rewind inside the last verify span), `reset`, and the verify/accept/rollback
  against the target's KV. It draws drafts from the install's MTP layer, or —
  when `dflash2` is set — from the drafter.
- `Qwen38DFlash2Drafter` — the block-diffusion drafter: tap capture from the
  target's layer outputs (`isTapLayer`, `encodeTapCapture`, `commitTapRows`),
  a rotating context cache, `propose(anchor:maxDrafts:)`, position
  realignment on continuity breaks, INT4-at-load weights, and a parity harness
  against an fp32 reference ([QWEN38_DFLASH2.md](../../QWEN38_DFLASH2.md)).

The verify/accept machinery is what makes the output byte-identical to plain
decode regardless of draft quality; the drafter only trades acceptance length.

## The protocol to extract

A `RoundDrafter` protocol that both existing drafters already satisfy in
shape, so the speculator becomes generic over the draft source and the target
runner:

```
protocol RoundDrafter {
    var contextWindowRows: Int { get }          // how many trailing tap rows warm it
    func isTapLayer(_ layer: Int) -> Bool
    func encodeTapCapture(commandBuffer:layerIndex:src:srcOffset:rows:)
    func commitTapRows(_ rows: Int)
    func alignPositionBase(_ base: Int)
    func reset()
    func propose(anchor: Int32, maxDrafts: Int) throws -> [Int32]
}
```

plus a target-side `SpeculativeTarget` the speculator drives (`verifyRows`,
`rewind(to:)`, `lastHiddenBuf`), which `Qwen38ForwardRunner` provides today
through internal members and `MiniCPM5ForwardRunner` would provide the same
way. The speculator's verify pass is family-specific only in *which runner's*
batched prefill it reuses.

## What is new for MiniCPM5 + DSpark, sized

| Piece | Reuse | New | Size |
|---|---|---|---|
| Draft stack forward (5 llama/qwen3 layers) | DFlash2's block attention, multi-x INT4 GEMV, tap gather | **q/k norm inside the draft stack** — the standalone per-head RMSNorm kernels exist (`RMSNorm.encodeBF16WPerHead`); it is wiring, not kernels | small |
| Conditioning | DFlash2's `fc` + `hidden_norm` projection and rotating context cache | tap layers `[1, 10, 20, 30, 39]` of a 42-layer target instead of `{5,19,33,47,61}` of 64; same shape of plumbing | small |
| Block proposal | DFlash2 `propose` (anchor + masks, bidirectional block) | no dynamic conv, no top-16 path selector (DSpark has neither) — *less* than DFlash2 | small |
| Confidence + Markov heads | none | `confidence_head` on the 2304-wide feature (2048 hidden + 256 Markov rank?) and the two `[130560, 256]` Markov factors; the exact feature layout must be read from the vendor's `dspark.py` (the config's `auto_map` points at it) before anything is ported | medium; the only genuinely unknown math |
| Speculator on `MiniCPM5ForwardRunner` | `Qwen38MTPSpeculator`'s verify/accept/rollback | the runner needs a batched verify pass over `k+1` rows (it has chunked prefill, which is that pass) and a KV rewind (the KV manager already supports it for Qwen 3.8) | small–medium |
| Weights at load | `DFlash2Int4Slab` INT4-at-load cache | a 0.65 GB BF16 checkpoint quantizes to ~0.2 GB; the slab format is reusable as is | small |
| Parity harness | `Qwen38DFlash2DrafterParityTests` shape | fp32 reference from the vendor's `dspark.py` under the pinned `transformers` in the family's venv; gate: block proposals token-exact for two rounds, confidence scores within tolerance | medium |

Expected payoff on this host: decode is 92 % GPU-bound at 13.9 ms/token
reading 1.42 GB per step; a draft round of 7 rows through 42 layers costs about
one extra weight read, so at the vendor's 4–5.5 accepted tokens per round the
ceiling is roughly 3× on greedy chat, and lower under the protocol's sampling.
DFlash2 on Qwen 3.8 measured +3.7 % over MTP on this host because MTP was
already there; MiniCPM5 has no MTP layer, so DSpark would be the *only*
speculative path and the comparison is against plain decode.

## Risks and the order to attack them

1. **The confidence/Markov math is unread.** Nothing else is worth starting
   before `dspark.py` has been read at the pinned revision and the 2304-wide
   feature and the Markov factorization are written down in a dossier, the way
   `modeling_llama.py` was for the base model.
2. **Draft-stack q/k norms** must use the same `(w · x̂)` convention as the
   target (`Qwen3RMSNorm` in `transformers` 5.x is plain `w`, unlike
   `Qwen3_5RMSNorm`'s `(1 + w)`); verify from source, do not assume.
3. **Thinking-mode traces**: the vendor's acceptance numbers were measured on
   its own generations; the family's greedy think-loop behaviour means
   speculation will spend most rounds inside `<think>`, where acceptance is
   what matters for tok/s, not the visible answer.
4. Keep every emitted byte identical to plain decode (the existing structural
   guarantee); add the same greedy A/B gate DFlash2 uses.

Estimated effort once the dossier exists: comparable to the DFlash2 port minus
its kernels (no dynamic conv, no path selector), plus the two heads.
