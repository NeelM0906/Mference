# Qwen 3.8 long context: paged KV with an SSD tier

Qwen 3.8-27B advertises a 262,144-token context, but its full-attention KV
cache costs 64 KiB/token — 16 GiB at full context, which cannot sit beside
14 GiB of weights on a 24 GB machine. And even if it could, dense attention
would read all 16 GiB per decoded token (~181 ms at the M5's ~95 GB/s):
long context is a *sparsity* problem before it is a *capacity* problem.

The paged KV mode solves both at full FP16 precision (no KV quantization):

- **Capacity**: full-attention KV lives in fixed 64-token pages. A bounded
  RAM pool (auto-sized from physical memory, ~4 GiB on a 24 GB host) holds
  the working set; sealed pages write behind to a sparse, layer-major spill
  file on SSD and evict under LRU pressure. The other 48 of 64 layers are
  Gated-DeltaNet linear attention whose 144 MiB recurrent state is constant
  and always exact — the architecture already carries most long-range signal
  outside the cache that grows.
- **Decode sparsity**: each token attends sink pages + the recent window +
  the top-k pages ranked by Quest criticality (arXiv 2406.10774): per page,
  element-wise min/max of its K rows summarize the page; `Σ_d max(q·min,
  q·max)` bounds the page's attention mass for the current query. Scores
  compute on-GPU each token and select for the *next* token (lag-one),
  which hides page-fetch latency in the inter-token gap. Selected-but-
  spilled pages return via one 256 KiB `pread` each (~2.3 GiB/s measured).
- **Exact prefill at any depth**: a chat turn appended beyond the pool runs
  the blocked streamed path — the sealed past flows through two staging
  buffers (one sequential `pread` per 8k-token window, overlapped with the
  previous window's GPU pass) and folds into FP32 running-softmax state;
  the chunk's own pages fold causally from the pool. Same math as the
  resident path, so prefill stays exact; only decode is sparse.

## Controls

```
--kv-paged <on|off|auto>   default auto: on above 32k --max-context
--kv-topk <pages>          decode budget beyond sinks+recent
                           (default 60 pages ≈ 3.8k attended tokens/layer)
--kv-pool-pages <n|auto>   resident pool per full-attn layer
                           (default auto: sized from RAM; 64 KiB tokens
                            resident per layer on a 24 GB host)
```

`RuntimeConfiguration`: `kvPagedPolicy`, `kvTopKPages`, `kvSinkPages`,
`kvRecentPages`, `kvPoolPagesPerLayer`. MTP speculative decode composes
with paged mode: verify rounds write draft rows through the page store and
run per-position paged attention over the round's pinned selection, with
cursor rewinds un-sealing pages on rejected drafts — byte-identical to
plain paged decode (`Qwen38PagedMTPTests`).

Speculative rounds run only while the page selection is **exhaustive**
(every context page fits the sinks + recent + top-k budget, ≈ 4.2k tokens
at the defaults). A round reuses one page table across its verify rows
where plain decode reselects per token, so under a sparse selection the
speculative stream could drift from the plain paged stream; the gate
(`Qwen38MTPSpeculator.canRunRound`) hands decode off to plain paged tokens
just before the selection turns sparse, preserving byte-identity across
the crossover. Raise `--kv-topk` to extend the exact-MTP window.

## Correctness

- The paged decode kernel is **bit-identical** to the contiguous kernel
  under a full selection, for any page→slot scattering
  (`PagedAttentionParityTests`).
- E2E: paged mode with a covering budget reproduces the dense runner's
  greedy stream exactly through decode, prefill + continuation, and reset
  (`Qwen38PagedKVParityTests`), on the toy and on the real checkpoint.
- Blocked streamed prefill agrees with the dense head (argmax) with sealed
  pages spilled and streamed back, including mid-page chunk boundaries;
  growing-chat flows replay deterministically under eviction, fetch, and
  selection pinning (`Qwen38BlockedPrefillTests`).
- Live selections are pinned for their token so their own fetches cannot
  evict them; page summaries are computed in the same command buffer that
  seals a page, so long prefills never trigger a metadata refetch storm.

## Measured (M5 MacBook Pro, 24 GB, real 27B checkpoint, 2026-08-15)

Plain decode (no MTP attach):

| run | context | pool | prefill | decode | notes |
|---|---|---|---|---|---|
| dense baseline | 4k | — | — | 7.8 tok/s | `--kv-paged off` |
| paged, all resident | 4k | auto | — | 8.0 tok/s | output identical to dense |
| paged + SSD, needle @30% | 5.4k prompt | 72 pages (4.6k tok) | 39 tok/s | 5.9 tok/s | passkey retrieved exactly |
| paged + SSD, needle @45% | 12.2k prompt | 128 pages (8k tok) | 27 tok/s | 6.3 tok/s | passkey retrieved exactly through spill |
| capacity smoke | 262,144 max-context | auto (2 GiB) | — | 7.9 tok/s | full-context settings, no decode regression |

With MTP speculative decode attached (byte-identical greedy):

| run | context | decode | notes |
|---|---|---|---|
| dense + MTP | 4k | 16.7 tok/s | reference |
| paged + MTP | 4k | **16.8 tok/s** | identical output to dense+MTP, zero paging overhead |
| paged + SSD needle + MTP | 5.4k prompt, 4.6k-token pool | 10.5 tok/s † | passkey retrieved exactly |
| 262k settings + MTP | 262,144 max-context | **14.0 tok/s** | |

† Measured before the exactness gate. At the default budget the needle's
5.4k context exceeds the exhaustive-selection window, so MTP now hands
those decodes to plain paged tokens (≈ the 5.9 tok/s plain rate); raise
`--kv-topk` past the context length to keep speculative rounds running
exactly.

Notes:

- The 64 KiB/token page geometry puts a K+V page pair at 256 KiB — the
  measured sweet spot of this NVMe's random-read curve (2.3 GiB/s; 4.96
  GiB/s sequential for the blocked-prefill streams).
- Reaching 262k *by prefill* costs ≈ an hour of compute (quadratic
  attention term at ~60 tok/s prefill); the paged mode's target workload is
  the growing chat, which pays that incrementally per turn.

## Files

- `Sources/Mference/Runtime/KVCache/KVPageStore.swift` — pools, spill file,
  LRU, pinning, metadata layout, streamed span reads.
- `Sources/Mference/Runtime/KVCache/KVPageSelector.swift` — sinks + recent +
  top-k policy.
- `Sources/Mference/Kernels/Attention/KVPageKernels.swift` +
  `Metal/Attention/attention.metal` — paged decode partial, page scores,
  page min/max, blocked-prefill flash init/update/finalize.
- `Qwen38ForwardRunner` — paged decode/prefill integration.
- Design spec: `docs/superpowers/specs/2026-08-15-qwen38-longctx-paged-kv-ssd-design.md`.
