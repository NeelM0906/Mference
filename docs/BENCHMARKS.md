# Benchmarks

This page records Mference measurements of Qwen 3.6 35B-A3B on a 24 GB M5
and a 32 GB M1 Max. Each number belongs to the workload shown. Prompt length,
generated length, cache state, and hardware all change throughput, so ranges
across workloads are not run-to-run variation.

Each table states its workload and decoding settings. Mference uses the
model installed by the [command-line instructions](../README.md#command-line).
Decode rate excludes model installation, model loading, and prompt prefill.

## Results at a glance

| Host and runtime | Decode rate | Reported memory |
| --- | ---: | ---: |
| M5, Mference, Qwen 3.6 35B-A3B | 23.5-29.3 tok/s | 32-slot Qwen profile (auto as of 2026-08-06) |
| 32 GB M1 Max, Mference, Qwen 3.6 35B-A3B (`cb99287`; ledger M23; `--verify full-sha256`) | 27.52-34.46 tok/s | 6.98-7.54 GB footprint |

A separate report covers Qwen 3.6, Qwen 3.8 27B and Maple on a 256 GB M3 Ultra:
[M3 Ultra benchmarks](BENCHMARKS_M3_ULTRA.md). Measurements of the retired
families were removed with them ([Retired model families](RETIRED_FAMILIES.md)).

## Qwen 3.6 35B-A3B measured decode

This optimized production run was recorded on 2026-08-06 on an M5
MacBook Pro (`Mac17,2`) with 24 GB, macOS 26.5, and Swift 6.3.3. The CLI's
automatic profile selected 32 Qwen expert-cache slots; no experimental flag or
profiler was active. One discarded warmup preceded one fresh-process measured
run per frozen community case, and every output ended normally.

| Case | Prompt / generated tokens | Prefill | Decode |
| --- | --- | ---: | ---: |
| short-explanation | 62 / 538 | 7.57 s | 29.293 tok/s |
| medium-review | 426 / 704 | 8.48 s | 27.460 tok/s |
| long-synthesis | 2,940 / 695 | 26.54 s | 23.470 tok/s |

The corresponding byte-identical 16-slot controls decoded at 21.487, 24.828,
and 21.490 tok/s. That is an 18.1% geometric-mean decode gain from the
model-aware default, on top of the custom Qwen decode kernels. Long-prompt
prefill is 2.20x faster than the original 58.45 s baseline. The original and
current generated token counts differ, so only the same-output controls are
used for the decode speedup claim.

Since this run, the 2026-08 performance campaign (PR #16, merged 2026-08-13)
changed the Qwen production defaults: the automatic profile now selects 96
expert-cache slots on hosts with at least 24 GiB (32 with at least 16 GiB,
else 16), with the GPU-resident slot map and eager routed commit on by
default. The rows above therefore describe the 2026-08-06 32-slot automatic
profile, not a current checkout.

### Historical 16-slot run

These rows ran on 2026-07-31 on an M5 with 24 GB of memory, macOS 26.5, and
Swift 6.2, against the experimental
[Qwen 3.6 35B-A3B](../docs/QWEN36_PERFORMANCE.md) path. They follow the
[community benchmark protocol](COMMUNITY_BENCHMARKS.md): the three frozen
`real-generation-v1` prompts with their fixed seeds, app sampling defaults
(temperature `0.2`, Top-K `64`, Top-P `0.95`), 4K context, 16 expert-cache
slots, one discarded warmup, then one measured run per case in a fresh
process. Every measured footer reported `stop=endOfTurn`.

| Case | Prompt / generated tokens | Prefill | Decode | Peak RSS / footprint |
| --- | --- | ---: | ---: | ---: |
| short-explanation | 62 / 493 | 7.74 s | 23.05 tok/s | 1,139 / 1,447 MiB |
| medium-review | 426 / 697 | 12.71 s | 21.20 tok/s | 1,142 / 1,448 MiB |
| long-synthesis | 2,940 / 700 | 59.16 s | 18.84 tok/s | 1,093 / 1,464 MiB |

Qwen 3.6's decode here is bound by expert-streaming I/O, not compute: its
18.1 GB expert pool does not fit the page cache, and its 16 slots cover 6.2%
of a layer's 256 experts. The
[performance notes](QWEN36_PERFORMANCE.md) break the token down phase by
phase.

### Under an 8 GB working set

**Correction (2026-08-08):** the table originally published here reported
throughput unchanged under an emulated 8 GB working set. That run's memory
pin did not hold — the ballast was reclaimable, so the page cache kept the
expert pool and the run measured the unconstrained machine twice. (The
original rationale, "the 18.1 GB pool does not fit the page cache on
either configuration," was also wrong for the 24 GB host, where it does.)
With a verified ballast (15.2 GiB `mlock`ed, free memory near zero), the
community cases at the 8 GB auto profile (16 slots) measure:

| Case | Decode, 24 GB | Decode, ~8 GB (verified) | Footprint |
| --- | ---: | ---: | ---: |
| short-explanation | 26.45 tok/s | 14.35 tok/s | ~1.1 GB |
| medium-review | 24.99 tok/s | 12.75 tok/s | ~1.1 GB |
| long-synthesis | 21.96 tok/s | 10.81 tok/s | ~1.1 GB |

Every run reached `stop=endOfTurn`. With the pool unable to cache, decode
is SSD-latency-bound; rerunning with the 2026-08-08 optimizations disabled
lands in the same regime (12.3/12.4 tok/s short/long), confirming the
correction concerns the pin, not the runtime changes. This remains
emulated pressure on M5 hardware: a physical 8 GB Mac has a slower SSD and
GPU and should be expected to decode more slowly still.

## M1 Max 32 GB measured (2026-11 performance program)

These rows ran between 2026-11-14 and 2026-11-17 on one MacBook Pro
(`MacBookPro18,4`): Apple M1 Max with a 24-core GPU, 32 GB, macOS 27.2.1
(26B131), and Swift 6.4.2. They cover the Qwen 3.6 35B-A3B install
(`qwen36.gturbo`, pinned `mlx-community/Qwen3.6-35B-A3B-4bit@38740b8`); the
program's rows for a retired install were removed with it. Every run used a
release build of the named commit, on AC power, with no other model process running. Each figure
names the ledger entry that records it (M1, M7, ...), the commit measured,
and the `--verify` mode. The ledger is the program's measurement log, with
raw output per entry; it is kept outside the repository. Figures measured by a
workstream lead in its own gated hold (machine lock, quiet machine, AC power,
swap guard) name that hold's record instead, for example `ws1-b-mma`; those
records are kept with the ledger. None of these figures was re-run on the
final `main`, `48461a1`.

Three workloads appear:

- **P** is the [community protocol](COMMUNITY_BENCHMARKS.md): the three frozen
  `real-generation-v1` cases, a fresh CLI process per run, `--max-new 1024
  --max-context 4096 --temperature 0.2 --top-k 64 --top-p 0.95`, the case
  seeds, and `--verify full-sha256`. Several A/B comparisons below ran the P
  cases with a different verify mode; each row says which.
- **S** is a fresh `MferenceServer` on loopback answering a cold request (0
  cached tokens) with 32 generated tokens. "S 3K" is the 2,940-token
  long-synthesis prompt. TTFT is measured by a streaming client.
- **L** is a 31,480-token prompt with 32 generated tokens, through the CLI
  (`--max-context 32768`) or through a server (`--max-context 36864`) that
  then answers a follow-up turn reusing 31,511 cached tokens.

The S and L streaming client is
[`Scripts/server_stream_bench.py`](../Scripts/server_stream_bench.py), and the
32K prompt and its follow-up are the frozen
[long-context-v1](benchmark-prompts/long-context-v1/README.md) set.
Decode over 32 tokens is indicative only: after a 32K prefill it takes
1.2-1.7 s (ledger M31; `8177a1a` and `b9e7a48`; `--verify auto`).

Under `--verify full-sha256`, P prefill includes hashing the whole expert pool
on first touch. For Qwen 3.6 that is about 8.5-8.7 s of the short and medium
prefill times; the same prompts prefill in about 2-3 s with `--verify auto`
(ledger note after M7; `06c13ee`-era binaries).

### Qwen 3.6 35B-A3B: community protocol baselines

Mode P, `--verify full-sha256`, five measured runs per case after a discarded
warmup. All runs ended at `stop=endOfTurn`.

| Case | Commit (ledger) | Prompt / generated | Prefill median | Decode median (min-max) | Peak footprint |
| --- | --- | --- | ---: | ---: | ---: |
| short-explanation | `06c13ee` (M1) | 62 / 520 | 10.44 s | 2.956 tok/s, slow mode in 5 of 5 runs | ~6.7 GB |
| medium-review | `06c13ee` (M1) | 426 / 703 | 11.71 s | bimodal: 3.006-3.072 (3 runs), 7.656-7.897 (2 runs) | ~6.75 GB |
| long-synthesis | `06c13ee` (M1) | 2,940 / 626 | 24.51 s | bimodal: 2.404-2.550 (4 runs), 7.415 (1 run) | ~7.2 GB |
| short-explanation | `477122c` (M7) | 62 / 520 | 10.67 s | 33.51 (32.74-34.24) | 6,658-6,680 MB |
| medium-review | `477122c` (M7) | 426 / 703 | 11.90 s | 34.80 (34.64-35.22) | 6,729-6,741 MB |
| long-synthesis | `477122c` (M7) | 2,940 / 626 | 24.49 s | 27.67 (27.03-30.76) | 7,171-7,195 MB |
| short-explanation | `cb99287` (M23) | 62 / 513 | 10.45 s | 34.46 (33.81-34.56) | 6,979-6,992 MB |
| medium-review | `cb99287` (M23) | 426 / 695 | 11.85 s | 34.16 (33.39-34.44) | 7,044-7,049 MB |
| long-synthesis | `cb99287` (M23) | 2,940 / 565 | 21.71 s | 27.52 (26.73-27.61) | 7,522-7,538 MB |

At `06c13ee` decode was held down by the GPU keep-warm filler, whose command
buffers took about 2.9 ms of GPU time each on this GPU; `477122c` (#54) is the
first commit without it (ledger M1 and M7; the filler timing is from the M2
diagnostic on a `06c13ee` build). The `cb99287` outputs
differ from the earlier ones because #59 changed them within its tolerance
gate, so its decode column covers different token sequences (ledger M23).
Peak footprint at `cb99287` is 4.6-4.8% above `477122c` on every case; the
cause was not tested (ledger M23).

### Qwen 3.6 35B-A3B: server and 32K baselines

S and L ran with generation capped at 32 tokens, except the `06c13ee` S short
and medium rows, which decoded to the end of the turn (TTFT does not depend on
the cap). `cb99287` used 5 measured S runs and 3 measured L runs per row; the
older rows used 5.

| Workload | Earlier baseline | `cb99287` (M23) |
| --- | ---: | ---: |
| S short TTFT, 62 tokens | 0.78 s (0.71-1.14), `06c13ee` (M3) | 0.958 s (0.949-0.963) |
| S medium TTFT, 426 tokens | 2.66 s (2.51-2.73), `06c13ee` (M3) | 2.085 s (2.057-2.096) |
| S long TTFT, 2,940 tokens | 17.71 s (17.66-17.82), `06c13ee` (M3) | 12.89 s (12.45-13.00) |
| S decode, 32 tokens (short / medium / long) | filler-bound, not a baseline (M3) | 37.04 / 35.42 / 27.30 tok/s |
| L server cold TTFT, 31,480 tokens | 581.8 s (577.8-609.1), `06c13ee` (M3) | 229.60 s (229.42-231.68) |
| L server follow-up TTFT, 31,511 cached | 48.2 s (47.7-55.4), `06c13ee` (M3) | 16.19 s (16.07-16.51) |
| L server decode, 32 tokens (cold / follow-up) | filler-bound, not a baseline (M3) | 18.01 / 17.95 tok/s |
| L CLI cold prefill, 31,480 tokens | 573.0 s (555.7-579.4), `2361831` (M12) | 227.70 s (227.68-228.27) |
| L CLI decode, 32 tokens | 7.36 tok/s (6.93-10.43), `2361831` (M12) | 18.68 tok/s (16.97-19.08) |
| Peak footprint | server 7.0-7.1 GB (M3); L CLI 7.53 GB (M12) | server 7.35 / 7.33 GB (S / L); L CLI 7.50-7.52 GB |

Every S and L run above used `--verify auto`, the server and CLI default
(ledger M3, M12, M23). The S short TTFT at `cb99287` is 22.8% above
`06c13ee`'s median, with overlapping ranges and a server-side cause that was
not tested. The L CLI decode rise from 7.36 to 18.68 tok/s happened between the
M19 builds (`ff0fc32`, `68b0aeb`) and `f831b21` and was not bisected
(ledger M23).

### Per-change A/B results

Each merged change was measured against its own base `main` in one of three
ways:

- **Tier-1 pair:** one control run, then one candidate run, in one session.
  The claimed metric is fixed before measurement. A change is accepted when
  the claim improves by 10% or more and no tracked metric is more than
  7% worse; a smaller result goes to an A/B/B/A.
- **A/B/B/A:** control, candidate, candidate, control segments in one session,
  each a fresh process or server. An effect counts only when both comparisons
  agree in direction and the ranges do not overlap.
- **Tier 0:** no runtime change. A static diff check plus an output-identity
  run.

Class E changes must keep output byte-identical to the control; every class E
row below did. Class T changes may change output within a tolerance gate
recorded with the change. Percentages are those recorded with each
measurement.

**Qwen 3.6 35B-A3B**

| PR (merge) | Change, class | Ledger; candidate vs control; method; verify | Claimed metric: control → candidate (Δ) | Verdict |
| --- | --- | --- | --- | --- |
| #57 (`b6f60cc`) | Live-memory budget type, not yet called by the runtime | M10; `25ddcc1` vs `477122c`; A/B/B/A, P long; `full-sha256` | No claim. Decode +0.3% / -0.2%, prefill -0.8% / +0.2%; output identical | PASS |
| | | M15; `a831aa5` (merged head) vs `2361831`; tier 0; `full-sha256` | Long output identical | PASS |
| #58 (`ff0fc32`) | Prefill stage breakdown, a diagnostic toggle off by default | M11; `3cc33a8` vs `477122c`; A/B/B/A, P; `full-sha256` | No claim. All ranges overlap; medium prefill +0.9% / +0.3% and decode -0.7% / -1.2% are not a measured effect | PASS |
| | | M13; `2f9b8cd` vs `2361831`; A/B/B/A, P; `full-sha256` | Short decode -7.0% / -5.5%, ranges overlap | PASS, flagged |
| | | M13a; same arms; A/B/B/A, P short, 4 runs per segment; `full-sha256` | Short decode +6.9% / +3.1%, prefill -3.2% / -1.6%: the M13 direction reversed | PASS |
| #59 (`b834cf7`) | GQA prefill attention (P3a), class T | M18; `68b0aeb` vs `ff0fc32`; tier-1 pair; `auto` | S 3K TTFT 18.03 → 14.82 s (-17.8%); L CLI cold prefill 561.8 → 237.8 s (-57.7%) | Tier-1 ACCEPT |
| | | M19; same arms; A/B/B/A, L CLI; `auto` | L cold prefill -57.5% / -58.2%, measured effect. L decode -1.6% / -3.0%, ranges overlap | PASS; L decode recorded as a limit |
| #60 (`f831b21`) | Prefill expert sweeps bypass the page cache, class E | M16; `bd88b1d` vs `ff0fc32`; tier-1 pair; `trusted-receipt` | S 3K TTFT 18.21 → 16.94 s (-7.0%) | Ambiguous, below the 10% bar |
| | | M17; same arms; A/B/B/A, S 3K and P medium/long; `trusted-receipt` | S 3K TTFT +20.2% / -4.8%: no measured effect. S decode after the 3K prefill +156% / +176%, measured effect | TTFT claim unsupported; the maintainer chose to merge on the S decode claim |
| | | M20; `eef6ba1` (merged head) vs `b834cf7`; tier-1 pair; `trusted-receipt` | S decode after 3K, 32 tokens, 10.50 → 29.47 tok/s (+180.7%) | Tier-1 ACCEPT |
| #61 (`cb99287`) | Installer stall fix, installer only | M21; `6b9fd50` vs `f831b21`; tier 0; `full-sha256` | Long output identical | Tier-0 PASS |
| #62 (`4b62f16`) | Multi-prefix server prompt cache, class E | M22; `3a296fb` vs `f831b21` runtime; tier-1 pair, server sequence; `auto` | Follow-up TTFT after a 32K turn and an unrelated short turn: 247.71 → 16.48 s (-93.3%). Server peak footprint 7.33 → 8.82 GB (+20.4%), within the cache's 4 GiB budget | Claim confirmed, class E holds; ACCEPT subject to a ruling on whether footprint counts for a cache meant to hold memory |
| #63 (`9a67018`) | Grouped routed-expert prefill (P3b), class T | M24; `9829772` vs `4b62f16`; tier-1 pair; `auto` | S 3K TTFT 12.349 → 12.267 s (-0.7%) | Ambiguous |
| | | M25; same arms; A/B/B/A, S 3K and P medium/long; `auto` | S 3K TTFT -2.0% / -1.8%, measured effect (-1.9%). P long prefill -5.7% / -5.5%, measured effect | PASS: claim confirmed below the bar, no regression |
| #64 (`c12f01d`) | Live-memory governor for the prefix cache | M26 (`d26509a`), M27 (`076cd6b`, merged head) vs `9a67018`; tier 0, server sequence; `auto` | Both turns identical in content, cached tokens and cache outcome; no shed fired | Tier-0 PASS |
| #65 (`0a138de`) | Batched grouped decode attention, class T | M28; `a94a86d` vs `c12f01d`; tier-1 pair; `auto` | L CLI decode, 32 tokens, median of 2: 17.86 → 27.56 tok/s (+54.3%) | Tier-1 PASS |
| #66 (`8177a1a`) | INT4 grouped GEMM for routed prefill off Apple10, class T | M29; `9e1482d` vs `0a138de`; tier-1 pair; `auto` | S 3K TTFT 12.281 → 9.328 s (-24.0%). L CLI cold prefill 216.46 → 179.44 s (-17.1%), reported, not claimed | Tier-1 PASS |
| #67 (`f8143c9`) | GQA prefill attention with shared K/V staging, class E | M30; `b9e7a48` vs `8177a1a`; tier-1 pair; `auto` | L CLI cold prefill, median of 2: 183.58 → 90.77 s (-50.6%). Also: 32K follow-up TTFT 15.16 → 6.42 s (-57.6%); S 3K TTFT 9.618 → 8.443 s (-12.2%) | Claim passes; one L decode run (12.31 tok/s) tripped the 7% guard |
| | | M31; same arms; A/B/B/A, L CLI; `auto` | L decode +20.6% then -10.1%: opposite directions, no effect. L prefill 88.98-89.27 s against 187.09-215.23 s | Guard flag cleared |
| #72 (`80106b8`) | Shadow speculative prefetch at budget 8 on 32 GiB hosts, class E | M37; `5e1b4ef` vs `67951bd`; A/B/B/A, P long, 3 runs per segment; `auto` | P long decode +7.8% / +6.9%, ranges do not overlap; pooled 31.345 → 33.394 tok/s (+6.5%). Short +4.5%, medium +2.8% and L 32K (128 tokens) +2.0%, one run each; no footprint increase | Confirmed |

Notes on reading these rows:

- P prefill in M18, M24, M25, M28, M29 and M30 ran with `--verify auto`, and in
  M16, M17 and M20 with `--verify trusted-receipt`, so those P times are not
  comparable with the `full-sha256` baselines above. #60 used
  `trusted-receipt` on purpose: a first-touch hash would load the expert pool
  into the page cache and hide the effect under test.
- M17's TTFT disagreement comes from one slow candidate segment (19.9-22.0 s)
  with no recorded cause.
- M31 voided one control run whose swap-outs grew (about 514 MB). It was
  replaced, and no candidate run was voided.
- After #67, the last Qwen 3.6 numbers in this series are that change's
  candidate arm (ledger M30, M31; `b9e7a48`; `--verify auto`): L CLI cold
  prefill 89.70 / 91.83 s (M30) and 88.98-89.27 s (M31), 32K follow-up TTFT
  6.42 s, S 3K TTFT 8.443 s. #72's candidate arm decoded P long at
  32.81-33.59 tok/s (ledger M37; `5e1b4ef`; `--verify auto`). No full
  baseline was re-run on a later `main`.

## Reproduce and contribute a result

The [community benchmark guide](COMMUNITY_BENCHMARKS.md) uses short, medium,
and long chat-framed prompts with fixed seeds. It requires coherent output and
a normal end of turn, so a repetition loop cannot become a published speed
result. The public CLI's timing footer reports decode-only throughput without a
separate research harness.

Community runs generate their own output. Compare community submissions only
when their prompt and generated token counts match.

A current checkout may not reproduce a historical number after the runtime,
compiler, or operating system changes. Report the commit and all three rows
rather than presenting one run as a general hardware result.

Read [System design](SYSTEM_DESIGN.md) for the runtime and resource split,
[Experiments](OPTIMIZATION_JOURNEY.md) for the main wins and failures, and the
[measurement lessons](experiments/summaries/09-validation-and-measurement-lessons.md)
for the rules used to evaluate performance changes.
