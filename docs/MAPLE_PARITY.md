# Maple parity harness

This is the maintained acceptance protocol for the clean Maple port. It is a
teacher-forcing comparison between a pinned local MLX oracle snapshot and a
pinned local Mference `.gturbo` install. The frozen model, runtime, and corpus
pins are defined in [MAPLE.md](MAPLE.md); this document describes how to use
the supported harness without changing those inputs.

The harness never downloads, creates, deletes, or overwrites a model, corpus,
trace, or Python environment. Supply only complete local assets. Do not run
two model owners at once, terminate an existing process, or remove an asset to
make a preflight pass.

## Required local assets

Before starting, obtain these assets by the approved installation process. This
protocol does not bootstrap them.

| Asset | Required state |
| --- | --- |
| Mference checkout | Clean Git checkout at the revision being evaluated. |
| Mference model | Complete pinned `maple.gturbo` install with its bound receipt. |
| MLX oracle | Complete local checkpoint snapshot at the exact revision in [MAPLE.md](MAPLE.md). |
| Oracle source and venv | Clean `scratch/mlx-lm-deepgrove` at the pinned fork revision, with a non-editable local install in the selected venv. The exporter verifies CPython 3.12.13 and the exact MLX, mlx-lm, Transformers, Tokenizers, and NumPy versions in [MAPLE.md](MAPLE.md). |
| Corpus | Existing normalized Raven file with the pinned bytes and token sequence. It must be local; do not invoke a fetcher as part of this protocol. |

The oracle exporter performs one full SHA-256 audit of the local snapshot,
including its large shards. The Mference exporter uses `full-sha256` when it
loads the `.gturbo` model. `trusted-receipt` is not an acceptance integrity
policy.

## Preflight and serial ownership

Run the preflight immediately before each engine. It requires Apple Silicon,
macOS 15+, Swift 6.1+, at least 10% free memory, at least the model reserve
free on disk, a complete pinned local model/snapshot, and no other Mference,
MLX, or package-test model process. The Mference preflight also requires a
clean checkout and validates the complete installed file set and receipt.

Use one engine at a time: finish the oracle process before starting the
Mference process. A failed preflight is a stop condition; report it rather
than killing a process or modifying local assets.

From the checkout root, substitute only the absolute paths for the already
available assets:

```bash
Scripts/maple_oracle_preflight.sh /absolute/path/to/maple-preview-2bit-mlx

Scripts/run_maple_mference_teacher_forcing.sh preflight \
  --model /absolute/path/to/maple.gturbo \
  --repository "$PWD"
```

The Mference wrapper first rejects a dirty checkout, then produces a clean
release `MferenceMapleParity` product before dispatching the command. Do not
replace it with manual object-file linking or an ad hoc executable.

## Export and compare

Use the selected, already pinned virtual environment explicitly. Its exporter
repeats the oracle preflight, verifies the local fork/package and pinned source
snapshot, disables FlashHead, and writes a new trace only when the destination
does not exist.

```bash
Scripts/run_maple_mlx_teacher_forcing.sh \
  --python /absolute/path/to/pinned-maple-venv/bin/python \
  --model /absolute/path/to/maple-preview-2bit-mlx \
  --corpus /absolute/path/to/the-raven.txt \
  --output scratch/maple-parity/mlx-trace.jsonl
```

After that process exits, build and run the Mference exporter through its
wrapper. It repeats Mference preflight, hashes every model payload under the
`full-sha256` policy, selects the Maple runner, forces the complete logits
head, and uses 16 expert-cache slots.

```bash
Scripts/run_maple_mference_teacher_forcing.sh export \
  --model /absolute/path/to/maple.gturbo \
  --corpus /absolute/path/to/the-raven.txt \
  --output scratch/maple-parity/mference-trace.jsonl \
  --repository "$PWD"

Scripts/run_maple_mference_teacher_forcing.sh compare \
  scratch/maple-parity/mlx-trace.jsonl \
  scratch/maple-parity/mference-trace.jsonl
```

An acceptance export has no `--max-positions` option and processes all 1,639
positions serially. The comparator accepts only matching full traces: pinned
metadata, clean source provenance, full position count, ordered vocabulary
top-10 IDs, and every retained top-10 logit must match exactly (zero
tolerance). It reports metadata failures and the first position divergence,
and exits nonzero when the comparison is ineligible or divergent.

`--max-positions 32` is useful only to diagnose a prefix:

```bash
Scripts/run_maple_mference_teacher_forcing.sh export \
  --model /absolute/path/to/maple.gturbo \
  --corpus /absolute/path/to/the-raven.txt \
  --output scratch/maple-parity/mference-prefix-32.jsonl \
  --repository "$PWD" \
  --max-positions 32
```

Every positive limit is diagnostic-only and writes
`acceptance_eligible=false`; the comparator rejects it. The MLX wrapper uses
the analogous positive `--max-positions 32` diagnostic prefix (its omitted or
zero value exports the complete trace). Never present either prefix as
acceptance evidence.

## JSONL contract and artifacts

Each trace is UTF-8 JSONL with exactly one `metadata` record first, contiguous
`position` records, and one `summary` record last. Unknown, missing, or
out-of-order fields/records and duplicate top-k token IDs are rejected.

| Record | Required payload |
| --- | --- |
| `metadata` | Schema and engine; engine source revision/dirty flag/binary SHA-256/command/runtime versions; model and source pins; config/tokenizer/corpus/token IDs hashes; corpus and position counts; top-K and tie policy; vocabulary and logit policy; exact-head/FlashHead/cache/integrity policies; acceptance eligibility. |
| `position` | Zero-based `position`, `input_id`, `target_id` (or `null` at the final token), and exactly ten `{ "id", "logit" }` entries ordered by descending logit then ascending ID. |
| `summary` | Position count, load seconds, elapsed seconds, positions per second, and exporter exit code. |

Store local corpus and traces below `scratch/maple-parity/`. `scratch/` is
ignored, so source text and full trace/logit artifacts are not committed.
Commit only compact, non-copyrighted metadata when a separately approved
result needs recording. Record the checkout commit and dirty state, hardware
and RAM, macOS, Swift, exact commands, exit codes, complete timing footers or
errors, and every protocol deviation with the run report.

## Acceptance boundary

This harness is a parity gate, not a benchmark. It does not transfer historical
research traces, binary hashes, generated text, measurements, or acceptance
claims. A clean full comparison and the separately required product smoke
workflows remain necessary before Maple support can be accepted.

The clean implementation run at commit `84d7b62` passed all 1,639 positions:
1,639/1,639 top-1 and ordered-top-10 matches with zero retained-logit
difference. The full traces remain ignored; the compact, non-copyrighted run
record is [maple-parity-2026-08-09.json](evidence/maple-parity-2026-08-09.json).
