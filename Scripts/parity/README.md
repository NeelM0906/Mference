# qwen38flashnext reference-parity golden harness (bring-up kit W3.2)

Goldens for the `qwen38flashnext` (upstream `qwen4_exp`) port, captured from the **installed
`transformers` package** running a toy `Qwen4ExpForCausalLM` on CPU in float32.

Contract being pinned: [`docs/superpowers/specs/2026-09-01-qwen38flashnext-runtime-design.md`](../../docs/superpowers/specs/2026-09-01-qwen38flashnext-runtime-design.md).
Family dossier: [`docs/families/QWEN38_FLASH_NEXT.md`](../../docs/families/QWEN38_FLASH_NEXT.md).

- Generator: [`qwen4exp_make_goldens.py`](qwen4exp_make_goldens.py) (committed)
- Goldens: `Tests/Mference/Fixtures/qwen4exp/` (committed, 2.5 MiB) — captured from the
  float32 init weights
- Goldens: `Tests/Mference/Fixtures/qwen4exp-bf16/` (committed, 2.5 MiB) — captured from the
  same weights rounded to bfloat16, i.e. the ones the emitted checkpoint carries. **This is
  the set a port gates on**; see "Two golden sets" below
- Toy checkpoints: `scratch/qwen4exp-toy-ckpt{,-prodlayout}/` (**not** committed, regenerable)
- venv: `scratch/qwen4exp-parity-venv/` (**not** committed)

## Recorded reference version

| | |
|---|---|
| `transformers` | `5.16.0.dev0` |
| `transformers` git commit | **`4da05482135896a529d5536c3c003102d36528a2`** (`refs/heads/main` at capture time) |
| `torch` | `2.13.0` (CPU) |
| Python | 3.12 |
| attn implementation | `eager` |
| dtype | float32 forward, bfloat16 checkpoint |

The commit is also carried in `goldens-manifest.json` under `reference.git_commit`, read from
the dist-info `direct_url.json` that pip/uv writes for a VCS install.

> The runtime design doc quotes `v5.8.0.dev0` (the version the pinned production checkpoint
> declares). The installed `main` is newer, but `models/qwen4_exp/modeling_qwen4_exp.py` and
> `configuration_qwen4_exp.py` are **byte-identical** to the reference copies the doc was read
> from, so there is no semantic drift. Re-check with `diff` after any reinstall.

## Environment setup

```bash
cd /path/to/Mference            # repo root; scratch/ is gitignored

uv venv --python 3.12 scratch/qwen4exp-parity-venv
VIRTUAL_ENV=scratch/qwen4exp-parity-venv \
  uv pip install torch --index-url https://download.pytorch.org/whl/cpu
VIRTUAL_ENV=scratch/qwen4exp-parity-venv \
  uv pip install numpy safetensors "git+https://github.com/huggingface/transformers"
```

`python -m venv` + `pip install` works identically; `uv` is just faster. To pin the exact
commit these goldens were captured against rather than tracking `main`:

```bash
VIRTUAL_ENV=scratch/qwen4exp-parity-venv uv pip install \
  "git+https://github.com/huggingface/transformers@4da05482135896a529d5536c3c003102d36528a2"
```

Confirm the model classes are present before anything else — if this fails, stop:

```bash
./scratch/qwen4exp-parity-venv/bin/python -c \
  "from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig; print('ok')"
```

## Regenerating the goldens

```bash
./scratch/qwen4exp-parity-venv/bin/python Scripts/parity/qwen4exp_make_goldens.py --emit-checkpoint
```

Output is **byte-reproducible**: same seed, single-threaded,
`torch.use_deterministic_algorithms(True)`. Verify:

```bash
find Tests/Mference/Fixtures/qwen4exp scratch/qwen4exp-toy-ckpt scratch/qwen4exp-toy-ckpt-prodlayout \
  -type f | sort | xargs shasum -a 256 > /tmp/r1.sha
rm -rf scratch/qwen4exp-toy-ckpt scratch/qwen4exp-toy-ckpt-prodlayout Tests/Mference/Fixtures/qwen4exp
./scratch/qwen4exp-parity-venv/bin/python Scripts/parity/qwen4exp_make_goldens.py --emit-checkpoint
find Tests/Mference/Fixtures/qwen4exp scratch/qwen4exp-toy-ckpt scratch/qwen4exp-toy-ckpt-prodlayout \
  -type f | sort | xargs shasum -a 256 | diff - /tmp/r1.sha && echo OK
```

`--print-hashes` prints the golden hashes without writing them. Checkpoint emission is
unconditional (the manifest embeds the checkpoints' tensor names, shapes and sha256), so
`--emit-checkpoint` is retained for the documented interface rather than gating anything.

## Two golden sets: `--weight-dtype fp32` and `--weight-dtype bf16`

```bash
# the original set, Tests/Mference/Fixtures/qwen4exp/           (default)
./scratch/qwen4exp-parity-venv/bin/python Scripts/parity/qwen4exp_make_goldens.py
# the checkpoint-faithful set, Tests/Mference/Fixtures/qwen4exp-bf16/
./scratch/qwen4exp-parity-venv/bin/python Scripts/parity/qwen4exp_make_goldens.py \
  --weight-dtype bf16
```

The default set is captured from the **float32 weights `Qwen4ExpForCausalLM(cfg)`
initialized**. The checkpoint this harness emits is a **lossy bfloat16 copy** of those
weights, so no consumer of the checkpoint can reproduce them. Measured by running the
reference twice, once with each weight dtype:

| | SHORT | LONG |
|---|---|---|
| logits max-abs, bf16 weights vs the fp32 goldens | `1.16e-3` | `3.09e-2` |

against the `atol = rtol = 1e-4` gate this manifest recommends — 10x to 300x over. And it
is not only a tolerance problem, because the discrete gates move too:

* `layer02.router_indices` flips at LONG query 18 (`[2,1]` → `[1,2]`),
* `layer03.indexer_selected` changes at LONG query 28 (`[8,9,10,11,…]` → `[0,1,2,3,…]`),
* the LONG cached-decode greedy rollout diverges at token 7: `…28, 48, 36, 14, 41…`
  becomes `…28, 2, 56, 43, 86…`.

`--weight-dtype bf16` rounds every parameter through bfloat16 and back to float32 **before**
any golden is captured, so the goldens describe the weights the checkpoint carries. The
forward is float32 in both cases; only the stored weight values differ. Rounding is
idempotent, so `emit_checkpoint` writes byte-identical checkpoints either way, and the
eight fp32 golden files still reproduce byte-for-byte.

Use the `fp32` set as the record of the reference's own arithmetic — it is what the design
contract was read against. Use the `bf16` set to gate a port: it is the one
`FlashNextReferenceParityTests` runs, because that suite loads an install built from the
emitted checkpoint.

The script **fails loudly** rather than writing bad goldens. It asserts, in-process:

1. `layer_multipliers`, `ngram_heads_vocab_sizes` and `ngram_heads_offsets` equal an
   independent pure-python splitmix64 / prime-search derivation from the spec.
2. Every PLE n-gram embedding row id equals an independent pure-python reimplementation of
   `_shift_right_ignore_eos` + the XOR-mix + per-head modulo.
3. The SHORT prompt's indexer selection equals the visible set at **every** query position of
   **every** attention layer (dense-equivalent regime).
4. The LONG prompt's selection excludes visible positions at **every** attention layer
   (genuinely sparse regime), and never selects a non-visible position.
5. The cached decode rollout equals a no-cache re-prefill rollout, token for token.
6. The bf16 checkpoint round-trips through `from_pretrained` with every tensor unchanged
   (this is what proves the `split_ngram_parts` shard concatenation is correct).
7. The fused `gate_up_proj` row order is gate-then-up.
8. Committed goldens stay under 8 MiB.

## Toy configuration

Full config is in `goldens-manifest.json` under `config`. Shape summary: `hidden_size` 64,
6 layers `[lin, lin(PLE), lin, qsa, lin, qsa]`, 4 attention heads / 2 KV heads / `head_dim` 16,
`partial_rotary_factor` 0.25 (rotary_dim 4, θ 1e4), GDN `Hk=2 dk=8 / Hv=4 dv=8` conv 4,
`hc_count` 4 / `hc_lowrank` 8 (stream width 256), PLE on one-indexed layer 2 with
`ngram_size` 3 / `heads_per_ngram` 8 / `ngram_vocab_size_base` 97 (padded table 2176×4),
QSA `2` query heads / `1` key head / `head_dim` 8 / `budget` 8 / `compress_ratio` 4
(`block_topk` 2), MoE 8 experts top-2, `vocab_size` 128, `eos_token_id` 0.

Three deviations from the bring-up-kit suggestion (also listed in the manifest under
`deviations_from_bringup_kit`); **no config value was rejected by the `@strict` validators**:

| Suggested | Used | Why |
|---|---|---|
| 4 layers, one attention layer | **6 layers**, two attention layers | The runtime design doc requires ≥2 attention layers so the per-layer indexer key cache and `layer_idx` plumbing are exercised. PLE still lands on one-indexed layer 2 (`layers[1]`), a `linear_attention` layer, as the validator demands. |
| SHORT = 12 tokens | **11 tokens** | With `budget` 8 / `ratio` 4 the `block_topk` is 2, so a 12-token prompt already has 3 complete blocks at its last query and is **not** dense-equivalent. 11 is the largest dense-equivalent length. |
| (unspecified) `mrope_section` | **`[1, 1, 0]`** | `rotary_dim` is 4 → only 2 frequency pairs, so the transformers default `[11, 11, 10]` cannot be expressed. Text-only positions collapse the sections, so this is semantically inert. |

Prompts (exact ids in the manifest under `prompts`): SHORT is 11 non-EOS tokens; LONG is 48
tokens with `eos_token_id` at index 20 so the PLE EOS-segmentation path is exercised. Both get
a 16-step greedy decode continuation.

Selection-regime coverage across the four golden files:

| | prefill | cached decode |
|---|---|---|
| SHORT | **dense-equivalent** (`selected == visible` everywhere) — this is the byte-gate A/B | sparse from step 0 (context passes 11 tokens immediately) |
| LONG | sparse from query 11 onward on both attention layers | sparse throughout |

So the goldens cover dense prefill, sparse prefill, and sparse cached decode. There is no
dense-equivalent *decode* case by construction: with `block_topk` 2 any decode step past a
12-token context is already sparse.

## Golden file contents

All float tensors are float32, batch dimension dropped.
`T` = 11 (short) / 48 (long); `S` = 256 (`hc_count × hidden`); `H` = 64.

### `prefill_{short,long}.safetensors` — 65 tensors

Captured from a single `use_cache=False` forward over the whole prompt.

| Key | Shape | Contents |
|---|---|---|
| `embed_out` | `[T, H]` | `embed_tokens` output, **before** the `repeat(1,1,hc_count)` |
| `layer{NN}.attn_hc_stream_in` | `[T, S]` | hyper stream entering the attention HC (post-PLE on layer 01) |
| `layer{NN}.attn_hc_mixed` | `[T, H]` | HC mix → attention/GDN block input |
| `layer{NN}.attn_hc_inject` | `[T, 4]` | `2·sigmoid(W_inject·h_n / hc_count)` injection weights |
| `layer{NN}.block_out` | `[T, H]` | GDN output (linear layers) or attention output after `o_proj` and the sigmoid gate (attention layers) |
| `layer{NN}.mlp_hc_stream_in` | `[T, S]` | hyper stream entering the MLP HC |
| `layer{NN}.mlp_hc_mixed` | `[T, H]` | HC mix → MoE block input |
| `layer{NN}.mlp_hc_inject` | `[T, 4]` | MLP injection weights |
| `layer{NN}.router_weights` | `[T, 2]` | top-k probs **after** renormalization |
| `layer{NN}.moe_out` | `[T, H]` | routed experts + gated shared expert |
| `layer{NN}.stream_out` | `[T, S]` | hyper stream leaving the layer |
| `layer01.ple_ngram_embeds` | `[T, 64]` | 16 concatenated n-gram head rows |
| `layer01.ple_out` | `[T, S]` | PLE contribution added to the stream, pre-attention |
| `last_hidden_state` | `[T, H]` | global `hyper_connection_mixer` output (there is no final norm) |
| `logits` | `[T, 128]` | `lm_head(last_hidden_state)` |

### `decode_{short,long}.safetensors` — 66 tensors

Greedy decode with a `DynamicCache`. **Row `i` is decode step `i+1`**: the first generated
token comes from the cached prefill leg (`cached_prefill_logits_last`, `[1, 128]`), which is
captured without hooks, so the per-layer decode tensors have `16 - 1 = 15` rows. Keys are the
same as above with `T = 15`, plus `step_logits` `[15, 128]`.

### `integers_prefill_{short,long}.json`, `integers_decode_{short,long}.json`

| Key | Contents |
|---|---|
| `layer{03,05}.indexer_selected` | Per query position, the **sorted list of selected KV token indices** — the QSA selection set. This is the gate the Swift indexer must match *exactly as an integer set*, before any attention math. |
| `layer{03,05}.indexer_visible` | Per query position, the causal-mask visible set. `selected ⊆ visible` always; `selected == visible` iff dense-equivalent. |
| `layer{NN}.router_indices` | Per token, the top-2 expert indices (paired with `router_weights`). |
| `layer01.ple_ngram_row_ids` | Per token, the 16 **absolute row indices** into the padded n-gram table (already offset by `ngram_heads_offsets`). Subtract the offset to recover the per-head modulo. |
| `argmax_all_positions`, `next_token` | Prefill only. |
| `generated_token_ids` | Decode only — the 16 greedy tokens from the cached loop. |
| `uncached_rollout_token_ids` | Decode only — the same 16 tokens from a full re-prefill each step. Equality is asserted in-script; it is the reference's own cache-equivalence proof. |

In the **prefill** files a per-layer value is indexed `[query_position][...]`. In the **decode**
files it gains one outer level, `[decode_step][query_position][...]`, with 15 steps (same
off-by-one as the decode tensors) and exactly one query position per step. So
`integers_decode_long.json["layer03.indexer_selected"][0][0]` is the selected KV set for the
first hooked decode step. `generated_token_ids` and `uncached_rollout_token_ids` are flat lists
of 16 — they include the token produced by the cached prefill leg.

### `goldens-manifest.json`

`reference` (versions/commit/flags), `dtype_policy`, `config` (the full toy config),
`prompts`, `deviations_from_bringup_kit`, `checkpoint_naming`, `port_hazards`,
`golden_files` (size + sha256 + description of each file), and `findings` (the results of
every in-script assertion, including the PLE hash derivation and the indexer regime stats).

## Toy checkpoint (uncommitted, in `scratch/`)

Two copies, both bf16, both regenerated deterministically:

- **`scratch/qwen4exp-toy-ckpt/`** — what `Qwen4ExpForCausalLM.save_pretrained` actually emits.
  Loadable with `from_pretrained`; the harness round-trips it.
- **`scratch/qwen4exp-toy-ckpt-prodlayout/`** — the same weights renamed and re-fused into the
  **production** checkpoint layout, for the repacker.

⚠️ **The two layouts differ in two ways the repacker cares about**, both recorded in the
manifest under `checkpoint_naming`:

1. **Prefix.** The text-only `ForCausalLM` emits `model.layers.{L}.*` with **no
   `language_model` segment**, because production Qwen4-Exp ships the multimodal
   `Qwen4ExpForConditionalGeneration` wrapper and therefore uses
   `model.language_model.layers.{L}.*` (see `docs/families/qwen38flashnext.tensors.json`).
2. **Experts.** `save_pretrained` **de-fuses** the MoE experts back to the source layout,
   `mlp.experts.{e}.{gate,up,down}_proj.weight` — one tensor per expert — via the `qwen2_moe`
   `WeightConverter`. Production ships them **fused**: `mlp.experts.gate_up_proj`
   `[E, 2·I, H]` and `mlp.experts.down_proj` `[E, H, I]`.

The n-gram table is written as `...ngram_embedding.shard_{S}.weight` (`split_ngram_parts`
shards, concatenated on dim 0 at load) in both copies. `layer_multipliers`,
`ngram_heads_vocab_sizes` and `ngram_heads_offsets` stay `I64` even in a bf16 checkpoint.

## Reference dtype behaviour the port must match

Determined by reading the installed package, and reflected in `dtype_policy`:

- **`Qwen3_5RMSNorm` (hence every `Qwen4ExpTextRMSNorm`) upcasts to fp32 internally**:
  `_norm(x.float()) * (1.0 + weight.float())`, then `.type_as(x)`. This resolves the open
  question in the runtime design doc. The zero-centered `(1 + w)` convention is confirmed, so
  the planned `+1` bake at repack is correct — but the fp32 upcast must be preserved.
- **`Qwen3_5RMSNormGated` (`linear_attn.norm`) is different**: it is **ones**-initialized, not
  zero-centered — *do not* bake `+1` into it. It upcasts for the variance, applies the weight
  in the *input* dtype, then multiplies by `act(gate.to(float32))`.
- **Indexer**: block pooling is `key_groups.float().mean(1)` and then cast **back to the key
  dtype** before `k_layernorm`; scoring is fp32 (`q.float() @ k.float()`). RoPE on the pooled
  key is applied at the **block's first position**.
- **Router**: `softmax(logits, dtype=float32)` → `topk` **of the probs** → renormalize → cast
  back to the logits dtype.
- **eager vs sdpa** on identical fp32 inputs: max abs logit delta `7.45e-8`, argmax identical.
  That is the reference's own kernel spread and the floor for any port tolerance; the manifest
  recommends `atol = rtol = 1e-4` for the fp32 Swift gates.

## Port hazards (manifest: `port_hazards`)

- The n-gram hash mix uses the **full positive int64 range by construction**:
  `(vocab_size − 1) × max(layer_multipliers)` has bit length 63 — **zero headroom**. Exact
  `Int64`/`UInt64` arithmetic only; `Double` cannot represent these products and `Int32`
  overflows immediately. This holds for the production vocab too, since `multiplier_max` is
  derived as `(2^63 − 1) // vocab_size`.
- **A query is not guaranteed to attend to itself.** When the query completes a block and that
  block loses the top-k, the query's own key is absent from the selected set — observed at LONG
  query 47 on both attention layers (8 selected of 48 visible, self excluded). Do not add an
  "always keep self" shortcut.
- Only **complete** `compress_ratio` blocks participate; the 1..`ratio`−1 tail is always kept,
  and the tail is taken from the *visible index list*, so padding changes which tokens are tail.
- PLE segmentation treats `eos_token_id` as **inclusive**: a shift may not cross an EOS and the
  EOS position itself starts the next segment; positions that would read across the boundary
  read `eos_token_id`. Production `eos_token_id` is `248044` (`text_config`).
- Attention `q_proj` packs query and gate **per head**: the output is viewed as
  `(.., num_heads, 2 · head_dim)` and chunked in 2 on the last dim, so within each head the
  first `head_dim` is query and the second is gate. It is **not** a global half/half split.
- **`torch.topk`'s tie-break is `std::nth_element`'s, and it is not lowest-index-first.**
  PyTorch's CPU topk takes the `nth_element` branch whenever `k * 64 > n` (always, at indexer
  widths), and libc++'s `__nth_element` short-circuits by length: `2` swaps if out of order,
  `3` runs `__sort3`, `<= 7` runs `__selection_sort` (first maximum, swapped into place — the
  swap displaces whatever was there), larger ranges run a median-of-3 quickselect. So for
  block scores `[0, 0, 0.385, 0]` with `k = 2` it returns blocks `{2, 1}`, not `{2, 0}`.

  This bites only on *bit-equal* scores. Across all four golden runs there are exactly **four**
  boundary ties, all in the LONG prefill, all with the tied score exactly `0.0` after the ReLU:
  `layer03 q11`, `layer05 q11`, `layer05 q17`, `layer05 q18` (3 and 4 candidate blocks). A
  lowest-index-first port fails `layer05.indexer_selected` at query 17 and, through the
  changed indexer key at token 11, cascades into `layer03`/`layer05` selections at queries 33,
  40 and 47 and into `logits` at `3.6e-2`. `FlashNextReferenceRunner.descendingTopK`
  reproduces the length-2/3/<=7 branches and records every boundary tie it sees, so a future
  failure can be attributed rather than guessed at.
