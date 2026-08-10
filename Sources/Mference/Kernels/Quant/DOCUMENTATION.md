# Quant kernels

`MapleTernaryGEMV` owns Maple's native-BF16 projection, embedding, and exact
INT4 vocabulary-head kernels. `MapleFlashHead` is an optional singleton-decode
front end for the same exact original `lm_head` rows.

When enabled, FlashHead first scores the retained quantized centroids, selects
the configured number of clusters on the CPU, expands their `token_map`, and
computes only those original head rows on the GPU. The remaining vocabulary
logits are `-infinity`; sampling is therefore deliberately restricted to the
candidate set. It never retains or rebuilds the source's redundant reordered
head copy.

FlashHead is approximate and opt-in. The standard decode path and all prefill
paths always use the full exact `lm_head`; this keeps Maple parity export and
batched-prefill parity independent of the sparse candidate head.
