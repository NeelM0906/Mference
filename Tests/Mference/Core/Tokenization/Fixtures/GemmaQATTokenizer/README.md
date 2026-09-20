# QAT source-template fixture

`chat_template.jinja` is copied unchanged from the verified installation of
`mlx-community/gemma-4-26B-A4B-it-qat-q4_0-mlx-aligned`, revision
`745a97a754ed4b7713163c7d0e9c11da41809e0c`.

- [Pinned source](https://huggingface.co/mlx-community/gemma-4-26B-A4B-it-qat-q4_0-mlx-aligned/blob/745a97a754ed4b7713163c7d0e9c11da41809e0c/chat_template.jinja)
- Size: 16,934 bytes.
- SHA-256: `94899c0f917d93f6fe81c95744d1e8ddab2d21d39228d2e4aec1fb2a25bff413`.
- `generation_config.json` is the unchanged 203-byte source asset, SHA-256
  `b69207f9be617e982d13cc273cce6fd88c98dda99a4bdc5e2d52ffe0a0d9f0a9`.

The small tokenizer/config are copied from the existing Gemma thinking test
fixture. They are a test vocabulary, not the real checkpoint vocabulary.
`Scripts/gemma_qat_template_oracle.py` independently renders the exact source
with Python Jinja 3.1.6 and tokenizers 0.23.2. Its optional `--tokenizer` argument
also verifies native installed token IDs without copying the full vocabulary.
Swift tests never regenerate expected output. Original Gemma fixtures remain
unchanged.
