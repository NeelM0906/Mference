# Qwen 3.6 resident-rung A/B — M5/24 GB, 2026-08-07

Community benchmark protocol (three frozen `real-generation-v1` prompts, app
sampling defaults, 4K context, discarded warmups, fresh-process runs) on the
24 GB M5 (`Mac17,2`), alternating blocks: slot32 ×2 reps, resident ×2 reps,
slot32 ×1, resident ×1. Branch `claude/apple-inference-engine-optimization-c1507f`
at the per-expert-buffer resident implementation.

## Exactness gate

`--expert-cache-slots resident` output is byte-identical to the 32-slot
control at 128 generated tokens (seed 42), and the toy-model greedy rollout
parity test (`QwenResidentParityTests`) guards the path.

## Decode medians (tok/s)

| Case | slot32 | resident | Δ | slot128 probe |
| --- | ---: | ---: | ---: | ---: |
| short-explanation | 29.12 | 28.65 | −1.6% | 27.54 |
| medium-review | 27.79 | 26.25 | −5.5% | — |
| long-synthesis | 24.10 | 10.44 | −56.7% | 23.27 |

## Prefill (representative, seconds)

| Case | slot32 | resident |
| --- | ---: | ---: |
| short-explanation (62 tok) | 7.3 | 4.6 |
| medium-review (426 tok) | 8.1 | 9.3 |
| long-synthesis (2,940 tok) | 28.3 | 38.4 |

## Verdict

Resident **fails** the acceptance rule on this host: the 18.1 GB mmap'd pool
plus KV, scratch, and the OS oversubscribes 24 GB, so long-context decode
faults experts from SSD (10.4 tok/s). The 128-slot near-resident fallback
(~9 GB wired LFU) repairs the collapse but beats 32 slots nowhere.

The explanatory observation: at 32 slots the process footprint is ~1.5 GB,
leaving the page cache free to hold essentially the entire expert pool — the
pool is already de-facto resident through the file cache. Qwen decode on this
host is therefore **not I/O-bound**; the remaining time is GPU compute and
CPU orchestration, which is where the kernel program aims next.

Defaults unchanged (`auto` keeps the slot rule). `resident`, `96`, and `128`
remain explicit `--expert-cache-slots` values: the resident rung's 37% faster
short-prompt prefill is real, and hosts with more headroom (48 GB+) may land
differently — both are future measured candidates, not defaults.

Raw run logs: task outputs from `run-benchmark.sh` labels `qwen36-slot32-a/b`,
`qwen36-resident-a/b`, `qwen36-slot128-probe`.
