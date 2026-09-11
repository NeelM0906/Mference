#include <metal_stdlib>
using namespace metal;

// ============================================================================
// flashnext_indexer — the QSA indexer (`Qwen4ExpTextQSAIndexer`) and the gather
// that feeds gated full attention its selected KV subset.
//
// Semantics, transcribed from `FlashNextReferenceRunner.indexerSelect` (the
// float32 CPU oracle that passes all six golden parity gates) and the design
// doc's "QSA indexer" section:
//
//   proj      = index_qk_proj . x                        // (H_q + 1) * 128
//   q_h       = rope_at(t, rmsnorm(proj[h]) * q_ln)      // per query head
//   raw_key   = proj[H_q]                                // UN-normed, UN-roped
//   k_blk(b)  = rope_at(b * 4, rmsnorm(mean_fp32(raw keys of the 4 tokens)) * k_ln)
//   score(b)  = sum_h relu(q_h . k_blk(b)) / sqrt(128)
//   selected  = topk(score, budget/4) blocks, each contributing its 4 tokens,
//               plus the incomplete tail (always), sorted ascending
//
// # Precision: this path is FP32 end to end
//
// The rest of the runtime keeps activations FP16. The indexer does not, and the
// divergence is deliberate. Its output is a *selection*, not a tensor: a score
// perturbed below the FP16 floor can reorder the top-k boundary and change which
// KV a layer is allowed to read, which is a semantic change rather than a
// rounding one. Relu-zero ties at the boundary are common at short context (the
// reference records them in `Capture.indexerBoundaryTies`), so the margin there
// is genuinely zero and there is nothing for a tolerance to absorb.
//
// So the projection lands in FP32, the query heads are normed and roped in FP32,
// and the raw-key and pooled-block-key caches are FP32 — 6 KiB/token and
// 1.5 KiB/token amortized across the 12 attention layers, twice the design doc's
// FP16 estimate. Against this family's 24 KiB/token KV that is a small price for
// removing a whole class of divergence.
//
// # Accumulation order
//
// Every dot product here is a serial FP32 loop on ONE thread, not a `simd_sum`
// tree, because the oracle accumulates serially and a tree reassociates. One
// thread per (query, block) also keeps the top-k boundary reproducible run to
// run. The cost is real but bounded: at the production budget a query scores at
// most `ceil(context / 4)` blocks x 4 heads x 128 multiplies, which is small next
// to the attention it then gates.
// ============================================================================

// `1 / theta^(2i / rotary_dim)`, written in the reference's form (a reciprocal
// of a positive power, not a negative exponent) so the two agree bit for bit at
// the frequencies that matter.
static inline float flashnext_indexer_inv_freq(uint pair, uint rotaryDim,
                                               float theta) {
    return 1.0f / pow(theta, float(2u * pair) / float(rotaryDim));
}

// Partial NeoX RoPE over the leading `rotaryDim` channels of an FP32 head,
// pairing `(i, i + rotaryDim/2)`. Channels at or above `rotaryDim` pass through.
static inline void flashnext_indexer_rope(thread float* head,
                                          uint rotaryDim,
                                          float position,
                                          float theta) {
    const uint half_rotary = rotaryDim / 2u;
    for (uint i = 0; i < half_rotary; ++i) {
        const float angle = position * flashnext_indexer_inv_freq(i, rotaryDim, theta);
        const float c = cos(angle);
        const float s = sin(angle);
        const float a = head[i];
        const float b = head[i + half_rotary];
        head[i] = a * c - b * s;
        head[i + half_rotary] = b * c + a * s;
    }
}

// RMSNorm over `n` FP32 channels with a BF16 `(1 + w)`-baked weight, serial in
// one thread — the oracle's `rmsNorm`.
static inline void flashnext_indexer_rmsnorm(thread float* v,
                                             device const bfloat* weight,
                                             uint n,
                                             float eps) {
    float ms = 0.0f;
    for (uint d = 0; d < n; ++d) ms += v[d] * v[d];
    const float inv = 1.0f / sqrt(ms / float(n) + eps);
    for (uint d = 0; d < n; ++d) v[d] = v[d] * inv * float(weight[d]);
}

// Largest indexer head dim this file's thread-local scratch supports. The
// production value is 128 and the toy's is 8, so 128 is the bound — and it is a
// real one: the array is statically sized, so every thread pays for it in
// register/stack pressure whether or not it uses the width. The Swift wrapper
// preconditions on this rather than relying on the guards below, which exist so
// an out-of-contract dispatch writes nothing instead of scribbling.
constant constexpr uint kFlashNextIndexerMaxHeadDim = 128;

// ---------------------------------------------------------------------------
// 1. Query heads: rmsnorm + RoPE at the query's own absolute position.
//
// `proj` is the whole `index_qk_proj` output, `[rows, projRows]` FP32; the query
// heads are its leading `heads * headDim` channels and the single raw key head
// follows them. One thread per (row, head).
// ---------------------------------------------------------------------------
kernel void flashnext_indexer_prepare_queries(
    device const float*  proj          [[buffer(0)]],   // [rows, projRows] FP32
    device const bfloat* qNorm         [[buffer(1)]],   // [headDim] BF16
    device       float*  queries       [[buffer(2)]],   // [rows, heads, headDim] FP32
    constant     uint&   heads         [[buffer(3)]],
    constant     uint&   headDim       [[buffer(4)]],
    constant     uint&   projRows      [[buffer(5)]],
    constant     uint&   startPosition [[buffer(6)]],
    constant     uint&   rotaryDim     [[buffer(7)]],
    constant     float&  theta         [[buffer(8)]],
    constant     float&  eps           [[buffer(9)]],
    constant     uint&   rows          [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= rows * heads) return;
    if (headDim > kFlashNextIndexerMaxHeadDim) return;
    const uint row = gid / heads;
    const uint head = gid - row * heads;

    float scratch[kFlashNextIndexerMaxHeadDim];
    device const float* src = proj + ulong(row) * ulong(projRows)
        + ulong(head) * ulong(headDim);
    for (uint d = 0; d < headDim; ++d) scratch[d] = src[d];
    flashnext_indexer_rmsnorm(scratch, qNorm, headDim, eps);
    flashnext_indexer_rope(scratch, rotaryDim, float(startPosition + row), theta);

    device float* out = queries + ulong(gid) * ulong(headDim);
    for (uint d = 0; d < headDim; ++d) out[d] = scratch[d];
}

// ---------------------------------------------------------------------------
// 2. Raw keys: the key head of `proj`, copied verbatim into the append-only
//    cache. Un-normed and un-roped — the pooling below is what normalizes, and
//    it pools the raw values.
// ---------------------------------------------------------------------------
kernel void flashnext_indexer_append_raw_keys(
    device const float* proj      [[buffer(0)]],   // [rows, projRows] FP32
    device       float* rawKeys   [[buffer(1)]],   // [.., headDim] FP32, append-only
    constant     uint&  headDim   [[buffer(2)]],
    constant     uint&  projRows  [[buffer(3)]],
    constant     uint&  keyOffset [[buffer(4)]],   // channels into `proj` row
    constant     uint&  seen      [[buffer(5)]],   // tokens already cached
    constant     uint&  rows      [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]          // x = channel, y = row
) {
    if (gid.x >= headDim || gid.y >= rows) return;
    rawKeys[ulong(seen + gid.y) * ulong(headDim) + gid.x] =
        proj[ulong(gid.y) * ulong(projRows) + keyOffset + gid.x];
}

// ---------------------------------------------------------------------------
// 3. Pooled block keys, computed once when a block's last token lands.
//
//   k_blk = rope_at(block * ratio, rmsnorm(mean_fp32(raw keys of the block)))
//
// Immutable thereafter, which is the whole reason for the cache: a block that
// completed at token 40 is scored identically by every later query.
// One thread per new block.
// ---------------------------------------------------------------------------
kernel void flashnext_indexer_pool_block_keys(
    device const float*  rawKeys       [[buffer(0)]],   // [.., headDim] FP32
    device const bfloat* kNorm         [[buffer(1)]],   // [headDim] BF16
    device       float*  blockKeys     [[buffer(2)]],   // [.., headDim] FP32
    constant     uint&   headDim       [[buffer(3)]],
    constant     uint&   compressRatio [[buffer(4)]],
    constant     uint&   firstBlock    [[buffer(5)]],
    constant     uint&   blockCount    [[buffer(6)]],
    constant     uint&   rotaryDim     [[buffer(7)]],
    constant     float&  theta         [[buffer(8)]],
    constant     float&  eps           [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= blockCount) return;
    if (headDim > kFlashNextIndexerMaxHeadDim) return;
    const uint block = firstBlock + gid;
    const uint start = block * compressRatio;

    float scratch[kFlashNextIndexerMaxHeadDim];
    for (uint d = 0; d < headDim; ++d) {
        float acc = 0.0f;
        for (uint j = 0; j < compressRatio; ++j) {
            acc += rawKeys[ulong(start + j) * ulong(headDim) + d];
        }
        scratch[d] = acc / float(compressRatio);
    }
    flashnext_indexer_rmsnorm(scratch, kNorm, headDim, eps);
    // RoPE at the block's FIRST position, not the query's.
    flashnext_indexer_rope(scratch, rotaryDim, float(start), theta);

    device float* out = blockKeys + ulong(block) * ulong(headDim);
    for (uint d = 0; d < headDim; ++d) out[d] = scratch[d];
}

// ---------------------------------------------------------------------------
// 4. Scores: `sum over heads of relu(q_h . k_blk) / sqrt(headDim)`, FP32.
//
// One thread per (query row, block). Blocks a row cannot see are written
// `-INFINITY` so a fixed-stride readback can ignore them without a second
// length buffer; the CPU selection only ever looks at the row's own visible
// count.
//
// The relu is applied per head BEFORE the sum, and the `/sqrt(headDim)` after —
// the oracle's exact order, which is not the same as scaling the dots first.
// ---------------------------------------------------------------------------
kernel void flashnext_indexer_scores(
    device const float* queries       [[buffer(0)]],   // [rows, heads, headDim] FP32
    device const float* blockKeys     [[buffer(1)]],   // [.., headDim] FP32
    device       float* scores        [[buffer(2)]],   // [rows, scoreStride] FP32
    constant     uint&  heads         [[buffer(3)]],
    constant     uint&  headDim       [[buffer(4)]],
    constant     uint&  scoreStride   [[buffer(5)]],
    constant     uint&  startPosition [[buffer(6)]],
    constant     uint&  compressRatio [[buffer(7)]],
    constant     uint&  rows          [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]              // x = block, y = row
) {
    if (gid.y >= rows || gid.x >= scoreStride) return;
    device float* out = scores + ulong(gid.y) * ulong(scoreStride) + gid.x;

    // Blocks are complete and visible when all `compressRatio` of their tokens
    // are at or before the query: `visible / ratio` of them, exactly the
    // oracle's `completeBlocks`.
    const uint visible = startPosition + gid.y + 1u;
    const uint completeBlocks = visible / compressRatio;
    if (gid.x >= completeBlocks) {
        *out = -INFINITY;
        return;
    }

    device const float* k = blockKeys + ulong(gid.x) * ulong(headDim);
    device const float* q = queries
        + ulong(gid.y) * ulong(heads) * ulong(headDim);
    float score = 0.0f;
    for (uint h = 0; h < heads; ++h) {
        device const float* qh = q + ulong(h) * ulong(headDim);
        float dot = 0.0f;
        for (uint d = 0; d < headDim; ++d) dot += qh[d] * k[d];
        if (dot > 0.0f) score += dot;
    }
    *out = score / sqrt(float(headDim));
}

// ---------------------------------------------------------------------------
// 5. Exact device-side block selection.
//
// A worst-first heap retains the best `blockBudget` scores. Equal scores keep
// the lower block index, matching the CPU fallback used above libc++'s small-N
// branches. The heap itself is unordered; byte flags let the final scan emit
// blocks (and therefore token positions) in ascending order without a second
// sort. The production heap is 512 entries and each row owns one single-thread
// threadgroup, keeping the result deterministic and avoiding a CPU readback.
// ---------------------------------------------------------------------------
constant constexpr uint kFlashNextIndexerMaxBlockBudget = 512;

static inline bool flashnext_indexer_worse(float aValue, uint aIndex,
                                           float bValue, uint bIndex) {
    if (aValue < bValue) return true;
    if (aValue > bValue) return false;
    return aIndex > bIndex;
}

static inline void flashnext_indexer_heap_sift_down(
    thread float* values, thread uint* indices, uint count, uint root
) {
    while (true) {
        const uint left = root * 2u + 1u;
        if (left >= count) return;
        const uint right = left + 1u;
        uint worst = left;
        if (right < count && flashnext_indexer_worse(
                values[right], indices[right], values[left], indices[left])) {
            worst = right;
        }
        if (!flashnext_indexer_worse(values[worst], indices[worst],
                                     values[root], indices[root])) return;
        const float oldValue = values[root];
        const uint oldIndex = indices[root];
        values[root] = values[worst];
        indices[root] = indices[worst];
        values[worst] = oldValue;
        indices[worst] = oldIndex;
        root = worst;
    }
}

kernel void flashnext_indexer_select_blocks(
    device const float* scores       [[buffer(0)]],
    device uint* selected            [[buffer(1)]],
    device uchar* flags              [[buffer(2)]],
    constant uint& scoreStride       [[buffer(3)]],
    constant uint& selectionStride   [[buffer(4)]],
    constant uint& startPosition     [[buffer(5)]],
    constant uint& compressRatio     [[buffer(6)]],
    constant uint& blockBudget       [[buffer(7)]],
    constant uint& rows              [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]]
) {
    if (row >= rows || blockBudget > kFlashNextIndexerMaxBlockBudget) return;
    const uint visible = startPosition + row + 1u;
    const uint complete = visible / compressRatio;
    const uint keep = min(complete, blockBudget);
    device uint* output = selected + ulong(row) * ulong(selectionStride);

    // Below the budget every complete block survives, and the incomplete tail
    // follows it. This is the overwhelmingly common short-context path.
    if (complete <= blockBudget) {
        for (uint position = 0; position < visible; ++position) {
            output[position] = position;
        }
        return;
    }

    thread float heapValues[kFlashNextIndexerMaxBlockBudget];
    thread uint heapIndices[kFlashNextIndexerMaxBlockBudget];
    device const float* rowScores = scores + ulong(row) * ulong(scoreStride);
    for (uint i = 0; i < keep; ++i) {
        heapValues[i] = rowScores[i];
        heapIndices[i] = i;
    }
    for (int root = int(keep / 2u) - 1; root >= 0; --root) {
        flashnext_indexer_heap_sift_down(
            heapValues, heapIndices, keep, uint(root));
    }
    for (uint block = keep; block < complete; ++block) {
        const float value = rowScores[block];
        if (flashnext_indexer_worse(heapValues[0], heapIndices[0],
                                    value, block)) {
            heapValues[0] = value;
            heapIndices[0] = block;
            flashnext_indexer_heap_sift_down(heapValues, heapIndices, keep, 0u);
        }
    }

    device uchar* rowFlags = flags + ulong(row) * ulong(scoreStride);
    for (uint block = 0; block < complete; ++block) rowFlags[block] = 0;
    for (uint i = 0; i < keep; ++i) rowFlags[heapIndices[i]] = 1;

    uint out = 0;
    for (uint block = 0; block < complete; ++block) {
        if (rowFlags[block] == 0) continue;
        for (uint j = 0; j < compressRatio; ++j) {
            output[out++] = block * compressRatio + j;
        }
    }
    for (uint position = complete * compressRatio; position < visible;
         ++position) {
        output[out++] = position;
    }
}

// ---------------------------------------------------------------------------
// 6. Gather the selected KV into contiguous scratch.
//
// The selection is at most `budget + tail` positions (2048 + 3 in production),
// so materializing them beats threading a per-query index list through the
// attention kernel: the gathered subset is then just a short dense KV run and
// the shipped `attention_full` decode kernel applies unchanged, mask and all —
// every gathered position is at or before the query, so no causal mask is
// needed inside the subset.
//
// One thread per (slot, channel). K and V share the layout
// `[token, kvHeads * headDim]`, which is what `KVCacheManager` hands out.
// ---------------------------------------------------------------------------
kernel void flashnext_indexer_gather_kv(
    device const half* kCache   [[buffer(0)]],   // [.., kvDim] FP16
    device const half* vCache   [[buffer(1)]],   // [.., kvDim] FP16
    device const uint* selected [[buffer(2)]],   // [count] absolute positions
    device       half* kOut     [[buffer(3)]],   // [count, kvDim] FP16
    device       half* vOut     [[buffer(4)]],   // [count, kvDim] FP16
    constant     uint& kvDim    [[buffer(5)]],
    constant     uint& count    [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]        // x = channel, y = slot
) {
    if (gid.x >= kvDim || gid.y >= count) return;
    const ulong src = ulong(selected[gid.y]) * ulong(kvDim) + gid.x;
    const ulong dst = ulong(gid.y) * ulong(kvDim) + gid.x;
    kOut[dst] = kCache[src];
    vOut[dst] = vCache[src];
}
