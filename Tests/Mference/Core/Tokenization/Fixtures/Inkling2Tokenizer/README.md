# Inkling 2 tokenizer fixture

`chat_template.jinja` and `generation_config.json` are the vendor's files,
unchanged, from `thinkingmachines/Inkling-2-Mini` at revision
`b85c3cfa615fc413cc9af3379f491b963f3b586a` (Apache 2.0). The template is
byte-identical to the one embedded in the shipped Unsloth UD-Q4_K_XL GGUF
(SHA-256 `53a9035f3255945704b5857657cfb00c1165090c0641c4f782a0cbf268cc16dc`).

`tokenizer.json` is synthetic: the MiniCPM5 fixture's byte-level BPE
vocabulary (ids 300 to 559, four merges) with Inkling 2's 60 added tokens at
their real ids 199,998 to 200,057, taken from the vendor
`tokenizer_config.json`. `tokenizer_config.json` keeps the vendor's class,
EOS, pad and added-token entries. No model weights or real vocabulary.
