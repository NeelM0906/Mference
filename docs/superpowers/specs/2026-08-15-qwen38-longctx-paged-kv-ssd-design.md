# Qwen 3.8 long context: paged KV cache with SSD tier and query-aware sparse decode

**Date:** 2026-08-15
**Target:** 128k–262k context for the Qwen 3.8-27B-4bit growing-chat workload on the 24 GB M5, full-fp16 KV (no KV quantization), decode ≥ ~5 tok/s at 262k.
**Branch:** `claude/qwen38-longctx-ssd-kv` (off main 579382f).

## Why this shape

Measured constraints (this machine, this model):

- Qwen 3.8 is hybrid: only **16 of 64 layers** are full attention (4 KV heads × 256 head_dim → **64 KiB/token** fp16 K+V). The other 48 are GDN linear-attention layers whose recurrent state is **144 MiB constant** — exact long-range signal that never grows. Full-attn KV at 262k = **16 GiB**; does not fit beside 14 GiB of weights in 24 GB.
- Dense attention at 262k is dead regardless of storage tier: 16 GiB/token read ≈ 181 ms/token from RAM (95 GB/s), 3.1 s/token from SSD.
- Measured NVMe (O_NOCACHE): 4.96 GiB/s sequential; **2.3 GiB/s at 256 KiB random** (0.11 ms/read); 0.76 GiB/s at 64 KiB.
- Therefore: **sparsity, not bandwidth**. Quest-style (arXiv 2406.10774) query-aware top-k page selection reads ~2–5% of pages per token at full precision. KVSwap (arXiv 2511.11907) validates the disk-resident + in-RAM-metadata shape for unified-memory devices.

Napkin at 262k: ~66 selected pages/layer × 64 tok/page ≈ 4k attended tokens/layer → 16 layers × 16 MiB = 256 MiB/token read from RAM pool ≈ 2.8 ms; SSD traffic only on selection misses (temporal locality keeps steady-state misses at a few pages/token ≈ ≤1 ms amortized).

## Architecture

Feature-flagged paged mode for the Qwen 3.8 runner only. Dense path untouched when off.

### 1. KVPageStore (`Sources/Mference/Runtime/KVCache/KVPageStore.swift`)

Owns full-attn-layer KV in **pages of 64 tokens** (per layer: 128 KiB K + 128 KiB V).

- **RAM pool:** per full-attn layer, one K pool buffer + one V pool buffer, `poolSlots` slots each. Auto-sized from free memory (default cap ~4 GiB total → ~65k tokens fully resident; beyond that LRU). Unsealed (tail) pages and pinned pages never evict.
- **SSD spill file:** sparse file in the model dir (`kvspill-<pid>` v1), layer-major layout: `offset(layer, page) = layerOrdinal · 1 GiB + page · 256 KiB` (K then V halves). Write-behind on seal via dedicated IO queue (pread/pwrite, F_NOCACHE like PreadExpertStreamer). Fetch = 256 KiB pread into a free/evicted slot.
- **Page metadata (Quest):** at seal, per (layer, page): element-wise min and max of the 64 post-RoPE K rows, per kv-head → 2 × 4 × 256 fp16 = 4 KiB. RAM-resident always (256 MiB at 262k). Computed on GPU (`kv_page_minmax` appended to the sealing CB).
- **States:** unsealed → sealed-resident(+clean-on-disk) → spilled. LRU over unpinned sealed pages.

### 2. Kernels (extend `attention.metal`)

- `attention_decode_paged_partial` (+ combine reuse): identical online-softmax to `attention_decode_partial`, but iterates logical selected slots and resolves K/V rows through a **page table** (`uint32` pool slot per selected page; last entry may be the partial tail page with `tail_valid` tokens). Grid/threadgroup geometry unchanged; existing `attention_decode_combine` merges partials as-is.
- `attention_page_scores`: per page, Quest criticality `score = max over q-heads of Σ_d max(q_d·minK_d, q_d·maxK_d)` against that head's kv-head metadata. One TG per page; output `float` per page per layer into a shared-storage scores buffer (read back after the token's CB completes).
- `kv_page_minmax`: reduce a sealed page's K rows to min/max vectors.
- (M5) `attention_prefill_blocked`: prefill attention with carry-in/out running state (m, d, o per query row) so past KV can stream through a bounded window of pool slots, block by block — exact, sequential-read-friendly.

### 3. Selection policy (`KVPageSelector`, CPU)

Per token, per layer: **sinks** (first 2 pages, StreamingLLM-style) ∪ **recent** (last 4 pages incl. unsealed tail) ∪ **top-k by score** (default ~60 pages), sorted ascending. Selection uses the *previous* token's scores (lag-one; standard, hides fetch latency behind the inter-CB gap): after CB t−1 completes → read scores → top-k → issue async fetches for misses in layer order → build page tables → encode CB t (per-layer wait on that layer's fetch just before encoding its attention). First decode token after a prefill uses sinks+recent+trailing-k (one-token warmup), corrected from the second token.

### 4. Runner integration (paged mode)

- Decode: QKV GEMV writes K/V into the unsealed page slot (same `kSlot`/`vSlot` shape via KVPageStore); RoPE in place; paged attention with the token's page table; `attention_page_scores` appended per layer (for the next token). Page seal on 64-token boundary appends `kv_page_minmax` and enqueues write-behind.
- Prefill (M4, in-RAM): pool slots allocated sequentially so each layer's pool region is contiguous → existing chunked-prefill kernels write/read it unchanged (guarded). Beyond-RAM growing-chat appends (M5) switch to `attention_prefill_blocked` streaming past pages from SSD (~2.5 s per 2k-token chunk at 200k context — acceptable turn latency; reads are sequential per layer).
- MTP speculative decode: auto-disabled in paged mode v1 (logged); re-enable later.
- GDN layers: untouched.

### 5. Config & CLI

`RuntimeConfiguration`: `kvPagedMode` (off/on/auto — auto = on when `maxContext > 32768`), `kvTopKPages`, `kvPoolBytes`, fixed `pageTokens = 64`. CLI `--kv-paged`, `--kv-topk`, `--max-context` up to 262144 in paged mode. Env mirrors (`MFERENCE_KV_PAGED`, …) per existing conventions.

## Correctness & quality gates

1. **Unit:** KVPageStore seal/spill/fetch/LRU/slot-map/file-offset tests; metadata vs CPU reference; selector determinism (Swift Testing, no GPU needed for store logic).
2. **Kernel parity:** paged attention with all pages selected + identity slot map ≡ contiguous kernel (same split geometry) within existing parity tolerances; score kernel vs CPU reference.
3. **E2E exact parity:** paged mode with k = ∞ (everything selected, all resident) must produce the identical greedy token stream as dense mode at 4k on the real model.
4. **Sparse quality:** needle-in-haystack + long-doc QA harness at 32k/64k (RAM-only) and 128k+ (SSD tier); acceptance = needle retrieval unimpaired at default k.
5. **Perf:** decode tok/s at 4k (no regression when off; bounded regression when on), 64k, 128k, 262k; page-fetch stall histogram; prefill throughput for streamed appends.

## Milestones

- **M1** KVPageStore + file format + LRU + metadata layout (CPU, TDD).
- **M2** Paged decode kernel + parity tests.
- **M3** Score kernel + selector + lag-one plumbing.
- **M4** Runner paged mode, in-RAM (≤ ~64k), E2E parity + needle at 32k.
- **M5** SSD tier live: spill/fetch/LRU + blocked streamed prefill → 128k–262k.
- **M6** Bench sweep, docs (`docs/QWEN38_LONG_CONTEXT.md`), memory entry.

## Risks

- Lag-one selection quality: mitigated by sinks+recent pinning and per-token correction; measured by needle harness before SSD work starts.
- Pool fragmentation vs prefill contiguity: M4 guards on contiguity, M5 removes the assumption via blocked prefill.
- 256 MiB metadata at 262k: acceptable v1; fp8 metadata is a known follow-up.
- One-CB-per-token invariant is preserved; all new GPU work rides the existing token CB.
