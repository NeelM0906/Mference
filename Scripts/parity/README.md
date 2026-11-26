# Reference-parity golden generators

Generators for the committed reference-parity goldens under
`Tests/Mference/Fixtures/`. Generators for the
retired families (listed in AGENTS.md), and the GGUF quant-table
generator whose runtime source went with them, were removed.

## minicpm5 reference-parity goldens

Generator: [`minicpm5_make_goldens.py`](minicpm5_make_goldens.py). Reference:
`transformers` **5.6.2** (the version the pinned `openbmb/MiniCPM5-2B` checkpoint
declares), torch 2.14.0 CPU, float32 forward, eager attention, single-threaded,
deterministic algorithms, seed 20260910. venv:

```bash
uv venv --python 3.12 scratch/minicpm5-parity-venv
VIRTUAL_ENV=scratch/minicpm5-parity-venv uv pip install torch --index-url https://download.pytorch.org/whl/cpu
VIRTUAL_ENV=scratch/minicpm5-parity-venv uv pip install "transformers==5.6.2" numpy safetensors jinja2 tokenizers
./scratch/minicpm5-parity-venv/bin/python Scripts/parity/minicpm5_make_goldens.py
```

Outputs (all committed, ~1.1 MB): `Tests/Mference/Fixtures/minicpm5/` (the gate
set — bf16-rounded weights with every projection replaced by its INT4 group-64
reconstruction) and `Tests/Mference/Fixtures/minicpm5-bf16/` (bf16-rounded
weights only, the record of the reference's own arithmetic), plus the bf16 toy
checkpoint in `Tests/Mference/Fixtures/minicpm5/toy-ckpt/` (HF llama layout,
committed because it is 333 KB and it lets the Swift gates run unconditionally).
Per file: `prefill_{short,long}.safetensors` (`embed_out`, per-layer `attn_out`
/ `mlp_out` / `hidden_out`, `final_norm_out`, `logits`), `decode_{short,long}.safetensors`
(the same per-layer keys for the 15 hooked decode steps, plus `step_logits`
with 16 rows), `integers_*.json` (argmax per position, greedy and uncached
rollouts, per-step top-2 margins) and `goldens-manifest.json`. Byte-reproducible:
regenerate and diff the sha256 list.

Chat-template fixtures for the same family are produced by
[`minicpm5_make_template_fixtures.py`](minicpm5_make_template_fixtures.py) from
the real `tokenizer.json` + `chat_template.jinja` (16 renders through
`apply_chat_template`, keys of tool-call arguments given sorted).

## Inkling 2 Mini reference goldens

Generators, the cmake-free build of the upstream llama.cpp reference and the
guarded run scripts are in [`inkling2/`](inkling2/README.md); the committed
fixtures (GGUF header and tensor table, tokenizer ids, chat-template renders)
are in `Tests/Mference/Fixtures/Inkling2/`. The reference is the shipped
weights file itself run through stock llama.cpp at tag b12641.
