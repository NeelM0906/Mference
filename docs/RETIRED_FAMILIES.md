# Retired model families

By maintainer decision, these families were removed on 2026-11-18, with
their runtime, installer, CLI and server support, tests, docs and scripts:

- Gemma 4 26B-A4B, and its QAT install
- DeepSeek-V4-Flash 284B-A13B
- Inkling-Small 276B-A12B
- GLM-5.3-Flash
- Flint
- Qwen3.8-Flash-Next 180B-A3.5B, and Unsloth's UD-IQ3_XXS GGUF install of it
- Swift-Qwen3.8, a separate Qwen 3.8 checkpoint with its own chat template

The last tree that has them is `mference-app` commit `8ff588c`; check it out
from git history to read their dossiers, design specs and code.

Install directories made for these families are not deleted, but this runtime
no longer loads them.

Mference supports Qwen 3.6 35B-A3B (`qwen36`), Qwen 3.8 27B (`qwen38`), Maple
Preview (`maple`) and MiniCPM5-2B (`minicpm5`). Older dated records under
`docs/` (release validation, experiments, plans) still name the retired
families as history.
