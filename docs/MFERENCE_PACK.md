# Mference Pack (`.mfpack` v1)

The native engine's install format (decision D3 of the engine program): a
GGUF's quantized blocks, rearranged into each tensor's kernel tile layout with
**no requantization**, hash-checked against the source. The normative types
are the frozen contracts in
[`Sources/Mference/Native/Contracts/`](../Sources/Mference/Native/Contracts/)
(`NativePackManifest.swift`, `NativeTileLayout.swift`,
`NativePackLayout.swift`, `NativeQuantFormat.swift`); the architecture is in
[NATIVE_ENGINE.md](NATIVE_ENGINE.md). This page describes the files, the
production writer and the runtime store.

## Directory

A pack is a directory `<label>.mfpack` (for example
`~/llm-models/quick.mfpack`):

| File | Contents |
|---|---|
| `manifest.json` | `PackManifest`: format `mfpack` v1, architecture (`qwen35`), label, the source GGUF (file name, size, SHA-256, GGUF version), `Qwen35Config`, the `PackLayoutTable` version, every tensor in GGUF tensor-info order, scalar GGUF metadata as strings |
| `weights.bin` | every tensor's packed bytes; each tensor starts at a 4096-byte-aligned offset, zero padding between tensors, no trailing bytes |
| `tokenizer.json` | `TokenizerData`: the GGUF `tokenizer.ggml.*` arrays and ids verbatim, including every other `tokenizer.ggml.<name>_token_id` key in `specialTokenIDs` |
| `chat_template.jinja` | the GGUF `tokenizer.chat_template` bytes, verbatim |
| `verified-install.json` | the receipt, written last, after every check passed; a pack without it is incomplete |

Every JSON file is encoded with sorted keys and pretty printing, so the bytes
are deterministic.

Each manifest tensor records its name, ggml type, shape (ggml `ne` order),
tile layout, offset and byte count in `weights.bin` (the GGUF byte count: a
layout never pads), the SHA-256 of its bytes in GGUF order (`sourceSHA256`)
and of its bytes as stored (`packedSHA256`). In the identity layout the two
are equal.

`layoutTableVersion` is `PackLayoutTable.version` when every tensor is in the
table's layout for its type (falling back to `.gguf` only where the table's
layout does not fit), and 0 when a writer chose other layouts (tests and
experiments). Version 1 of the table is the identity for every type; a tiled
entry is a coordinator contract change naming W1 and W2, and existing packs
stay readable because each tensor records its own layout.

### The receipt

`verified-install.json` holds `format`, `formatVersion`, `label`, `writer`,
`sourceSHA256` (the whole GGUF), `manifestSHA256`, `tensorCount`,
`weightsByteCount` and `verifiedAt`. The production writer adds
`tokenizerSHA256` and `chatTemplateSHA256`; readers check them when present.
The reference writer's receipts carry only the first nine fields, and both
stores read both writers' packs.

## Writing a pack

The production writer is `MFPackWriter` in
[`Sources/MferenceRepack/Core/Pack/`](../Sources/MferenceRepack/Core/Pack/).
`MferenceRepackCore` does not depend on the runtime, so it carries mirror
types (`MFPack.*`) that `PackMirrorPinTests` pins to the contracts: identical
JSON encodings, block formats, layout table and `packedOffset` permutation.

What it guarantees:

- **Install lock.** `<output>.install.lock` (flock) excludes a second writer
  for the same output. The pack is built in `<output>.partial` and renamed
  into place only after verification; an existing pack is never overwritten.
- **Bounded memory.** The source is read with `pread` in 8 MiB chunks
  (`F_NOCACHE`). An identity-layout tensor streams straight through; a tiled
  tensor goes through two row-group buffers. Every buffer is a
  `BoundedScratch` capped at 64 MiB by default; no tensor is held whole.
- **Hashes.** One sequential pass hashes the whole source file and every
  tensor's source and packed bytes (SHA-256).
- **Resume.** `<output>.resume.json` lists the tensors whose bytes were
  fsynced (the checkpoint advances every 256 MiB written). It is tied to the
  source file's size, modification time and inode, the label and a hash of
  the write plan; a resume with anything else is refused. On resume every
  listed tensor's packed bytes are re-hashed, and one that no longer matches
  is rewritten. A partial pack is refused without `resume`;
  `MFPackWriter.discardPartial` removes it.
- **Verification.** Before the receipt is written, `weights.bin` is re-read:
  its size, zero padding, every packed hash, and every tensor unpacked back to
  GGUF order against its source hash.
- **Reference parity.** For the same GGUF, label and layouts, `weights.bin`,
  `manifest.json`, `tokenizer.json` and `chat_template.jinja` are
  byte-identical to the frozen reference writer's
  (`Native.ReferencePackWriter`); `PackWeightStoreTests` checks this in the
  table layout and in a tiled layout.

`MFPackVerifier.roundTrip(pack:gguf:)` checks an installed pack against its
source byte for byte: every tensor unpacked to GGUF block order and compared
with `memcmp`, plus the tensor list, types, shapes and source size.

Until the `MferenceRepack` command line has a GGUF mode, the writer runs
through env-gated tests (each a long I/O run; take the machine lock):

```bash
MFERENCE_PACK_WRITE="$HOME/llm-models/gguf/Qwen3.5-4B-UD-Q4_K_XL.gguf|$HOME/llm-models/quick.mfpack|quick" \
  Scripts/test.sh --filter 'PackQuickTests/writePack'
MFERENCE_PACK_ROUNDTRIP="$HOME/llm-models/quick.mfpack|$HOME/llm-models/gguf/Qwen3.5-4B-UD-Q4_K_XL.gguf" \
  Scripts/test.sh --filter 'PackQuickTests/roundTripAgainstTheGGUF'
```

Set `MFERENCE_PACK_RESUME=1` to continue a partial write, and
`MFERENCE_PACK_REFERENCE=<reference pack>` to compare `weights.bin`,
`tokenizer.json` and the manifest with the reference writer's pack.

## The runtime store

`Native.PackWeightStore` (`Sources/Mference/Native/Weights/`) implements the
`WeightStore` contract. It maps `weights.bin` read-only and shared and wraps
it once with `makeBuffer(bytesNoCopy:)`; every `QuantTensor` is an offset into
that buffer, in the layout its manifest entry records. The heap holds only the
manifest, the tensor index and the tokenizer tables.

Opening a pack checks, before any tensor is handed out:

- the receipt exists and matches `manifest.json`, the size of `weights.bin`
  and, when recorded, `tokenizer.json` and `chat_template.jinja`;
- format `mfpack` v1, architecture `qwen35`, a known layout table version;
- every tensor: a known block format, whole blocks, its byte count, a layout
  that fits (and is the table's under `PackLayoutTable.version`), a
  4096-aligned offset inside the file, no overlap with another tensor, and
  equal hashes in the identity layout;
- the tensor set against `Qwen35TensorCatalog`: every tensor the config
  implies is present with the implied shape (`token_embd`, `output_norm`,
  `output` unless tied; per block the norms and FFN; `attn_q` (query and
  gate), `attn_k`, `attn_v`, `attn_output` and the q/k norms on full-attention
  blocks; `attn_qkv`, `attn_gate`, `ssm_alpha`, `ssm_beta`, `ssm_out`,
  `ssm_conv1d`, `ssm_dt.bias`, `ssm_a` and `ssm_norm` on Gated DeltaNet
  blocks; the `nextn` blocks' attention, FFN, `eh_proj`, `enorm` and `hnorm`).

`Verification.packedSHA256` also hashes every tensor's packed bytes, in
parallel (one read of the file).

### Tokenizer and chat template

`Native.PackTokenizer` is llama.cpp's BPE tokenizer at the pin for
`tokenizer.ggml.model == "gpt2"` with the `qwen35` pre-tokenizer:
llama.cpp's special-token attribute rules and partition, the qwen35 split,
the bigram merge queue and `token_to_piece`. Every token text and merge
operand is interned once, so a merge is an integer-pair lookup. Its output is
byte-identical to the frozen reference tokenizer and to llama.cpp's prompt
ids on every golden prompt (`PackTokenizerTests`, `PackQuickTests`).

`Native.PackChatTemplateSource` serves the template bytes verbatim, with the
BOS and EOS token texts resolved as llama.cpp resolves the ids.
`PackQuickTests` renders every golden prompt's messages with the repository's
Jinja and compares the result with llama.cpp's rendering byte for byte.

## Acceptance (W1)

| Check | Where |
|---|---|
| Mirror types pinned to the contracts | `PackMirrorPinTests` |
| Writer byte-identical to the reference writer; reference path on the W1 store bit-identical | `PackWeightStoreTests` |
| Resume, lock, refusals, bounded scratch, hostile headers | `PackWriterTests` (`MferenceRepackTests`) |
| Every Quick and Deep tensor byte-exact to its GGUF blocks | `PackQuickTests/roundTripAgainstTheGGUF` (env-gated) |
| Tokenizer and rendered template byte-identical to llama.cpp on the fixed prompt set; bounded heap | `PackQuickTests/tokenizerAndTemplateMatchLlamaCpp` (env-gated) |
| D1 of the reference `qwen35` path on the W1 store | `PackQuickTests/teacherForcedD1OnTheW1Store` (env-gated model run) |
