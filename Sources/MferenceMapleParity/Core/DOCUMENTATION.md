# Maple parity core

`MferenceMapleParityCore` exports and compares strict JSONL teacher-forcing
traces for the pinned Maple checkpoint. The pins take checkpoint identity from
`SupportedModelSource.maple`; corpus, token, and trace-shape pins make the
acceptance protocol explicit.

`MapleParityPreflight` separates host inspection from pure validation. Export
requires Apple Silicon, macOS 15, Swift 6.1, at least 10% free memory, a clean
resolved repository, no other model process, a bound Maple receipt, and an
exact installed file set before model loading. It verifies raw UTF-8/LF/NFC
corpus bytes and the no-BOS token sequence, then executes the Maple runner
once per token with its full vocabulary head.

Each JSONL metadata record includes engine, binary, command, runtime, model,
source-snapshot, config, tokenizer, corpus, and runtime-policy provenance.
Position records use canonical top ten ordering (descending logit, ascending
token ID on ties); summaries record load and execution timing plus exit status.
Both exporters hash every model payload before acceptance and record the
`full-sha256` integrity policy.

A positive diagnostic position limit writes an intentionally
`acceptance_eligible=false` prefix. The comparator rejects prefixes and any
metadata, input-token, ordered-top-k, or exact numeric-logit divergence.
