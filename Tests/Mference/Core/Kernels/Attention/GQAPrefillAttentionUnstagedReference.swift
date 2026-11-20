import Foundation
import Metal
import Testing
@testable import Mference

/// The GQA-shared prefill attention kernel as P3a shipped it (4 tokens per
/// threadgroup, K/V read by each simdgroup from device memory), verbatim
/// apart from its name. The staged kernel that replaced it keeps each 8-key
/// step's arithmetic, so the two must agree bit for bit; tests compile this
/// one at run time as the reference.
enum GQAPrefillAttentionUnstagedReference {
    static let tokensPerGroup = 4

    static func pipeline(device: MTLDevice) throws -> MTLComputePipelineState {
        // The main library's language version, so the comparison is like for like.
        let options = MTLCompileOptions()
        options.languageVersion = MetalContext.shaderLanguageVersion
        let library = try device.makeLibrary(source: source, options: options)
        let function = try #require(library.makeFunction(name: "attention_prefill_gqa8_d256_simd_unstaged"))
        return try device.makeComputePipelineState(function: function)
    }

    static let source = #"""
    #include <metal_stdlib>
    #include <metal_simdgroup_matrix>
    using namespace metal;

    struct PrefillAttentionParams {
        uint startPosition;
        uint queryCount;
        uint headDim;
        uint numQHeads;
        uint numKVHeads;
        uint kvValidCount;
        uint slidingWindow;
        uint kvTokenStrideElements;
        uint qTokenStrideElements;
        uint oTokenStrideElements;
        float scale;
    };

    // GQA-shared causal prefill attention for headDim 256 and 8 query heads per KV
    // head (Qwen 3.6 full-attention layers), for GPUs without the TensorOps path.
    //
    // One simdgroup owns one query token and the 8 query heads that share a KV
    // head: those 8 rows form one 8x8 simdgroup-matrix row block, so every K/V
    // element it reads serves 8 heads, and the threadgroup's 4 simdgroups take 4
    // consecutive tokens of the same KV head so their K/V reads hit the same
    // cache lines. Scores, the online softmax and the output accumulator are
    // FP32 throughout; only Q/K/V/O are FP16, as in the tiled kernel. The result
    // differs from `attention_prefill_causal_tiled` by summation order only.
    //
    // Fragment layout is the 8x8 simdgroup-matrix register layout: each lane
    // holds row `fm`, columns `fn` and `fn + 1`; lanes sharing a row differ in
    // lane bits 0 and 3. No sliding-window clip, KV ring or sinks: the Swift
    // selector routes those shapes to the tiled kernel.
    constant constexpr uint kPrefillGQAHeadDim = 256u;
    constant constexpr uint kPrefillGQADimTiles = kPrefillGQAHeadDim / 8u;
    constant constexpr uint kPrefillGQAHeadsPerKV = 8u;
    constant constexpr uint kPrefillGQATokensPerGroup = 4u;
    constant constexpr uint kPrefillGQAKeysPerStep = 8u;

    [[kernel, max_total_threads_per_threadgroup(128)]]
    kernel void attention_prefill_gqa8_d256_simd_unstaged(
        device const half* Q [[buffer(0)]],
        device const half* K [[buffer(1)]],
        device const half* V [[buffer(2)]],
        device half* O [[buffer(3)]],
        constant PrefillAttentionParams& p [[buffer(4)]],
        uint2 tg [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]
    ) {
        threadgroup half q_tile[kPrefillGQATokensPerGroup][kPrefillGQAHeadsPerKV * kPrefillGQAHeadDim];

        const uint t = tg.x * kPrefillGQATokensPerGroup + sg;
        const uint kvh = tg.y;
        if (t >= p.queryCount || kvh >= p.numKVHeads) return;

        const uint qid = lane / 4u;
        const uint fm = (qid & 4u) + ((lane / 2u) % 4u);
        const uint fn = (qid & 2u) * 2u + (lane % 2u) * 2u;

        // The 8 heads of this KV group are contiguous in the token's Q row.
        device const half* q_src = Q + t * p.qTokenStrideElements
            + kvh * kPrefillGQAHeadsPerKV * kPrefillGQAHeadDim;
        threadgroup half* q_own = q_tile[sg];
        for (uint i = lane; i < kPrefillGQAHeadsPerKV * kPrefillGQAHeadDim; i += 32u) {
            q_own[i] = q_src[i];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 acc[kPrefillGQADimTiles];
        for (uint d = 0u; d < kPrefillGQADimTiles; ++d) {
            acc[d] = simdgroup_float8x8(0.0f);
        }
        float row_max = -INFINITY;
        float row_sum = 0.0f;

        const uint abs_q = p.startPosition + t;
        const uint last_exclusive = min(p.kvValidCount, abs_q + 1u);
        const uint kv_stride = p.kvTokenStrideElements;
        device const half* k_head = K + kvh * kPrefillGQAHeadDim;
        device const half* v_head = V + kvh * kPrefillGQAHeadDim;
        threadgroup const half* q_row = q_own + fm * kPrefillGQAHeadDim + fn;

        for (uint key0 = 0u; key0 < last_exclusive; key0 += kPrefillGQAKeysPerStep) {
            // S = Q (8 heads x 256) * K^T (256 x 8 keys). This lane's two score
            // columns are keys key0+fn and key0+fn+1; masked keys read key0,
            // which is always visible, and are dropped below.
            const uint ka = key0 + fn;
            const uint kb = ka + 1u;
            const bool va = ka < last_exclusive;
            const bool vb = kb < last_exclusive;
            device const half* k_a = k_head + (va ? ka : key0) * kv_stride + fm;
            device const half* k_b = k_head + (vb ? kb : key0) * kv_stride + fm;
            simdgroup_float8x8 scores = simdgroup_float8x8(0.0f);
            for (uint d = 0u; d < kPrefillGQADimTiles; ++d) {
                simdgroup_float8x8 a;
                simdgroup_float8x8 b;
                a.thread_elements()[0] = float(q_row[d * 8u]);
                a.thread_elements()[1] = float(q_row[d * 8u + 1u]);
                b.thread_elements()[0] = float(k_a[d * 8u]);
                b.thread_elements()[1] = float(k_b[d * 8u]);
                simdgroup_multiply_accumulate(scores, a, b, scores);
            }

            const float s_a = va ? scores.thread_elements()[0] * p.scale : -INFINITY;
            const float s_b = vb ? scores.thread_elements()[1] * p.scale : -INFINITY;
            // Key key0 is visible to every row, so the step maximum is finite.
            float step_max = max(s_a, s_b);
            step_max = max(step_max, simd_shuffle_xor(step_max, 1u));
            step_max = max(step_max, simd_shuffle_xor(step_max, 8u));
            const float new_max = max(row_max, step_max);
            const float rescale = row_sum > 0.0f ? fast::exp(row_max - new_max) : 0.0f;
            const float p_a = va ? fast::exp(s_a - new_max) : 0.0f;
            const float p_b = vb ? fast::exp(s_b - new_max) : 0.0f;
            float step_sum = p_a + p_b;
            step_sum += simd_shuffle_xor(step_sum, 1u);
            step_sum += simd_shuffle_xor(step_sum, 8u);
            row_sum = row_sum * rescale + step_sum;
            row_max = new_max;

            // O += P (8 heads x 8 keys) * V (8 keys x 256). P has the score
            // layout. Masked keys have P = 0 and read a visible row, never past
            // kvValidCount.
            simdgroup_float8x8 probs;
            probs.thread_elements()[0] = p_a;
            probs.thread_elements()[1] = p_b;
            const uint v_key = min(key0 + fm, last_exclusive - 1u);
            device const half* v_row = v_head + v_key * kv_stride + fn;
            for (uint d = 0u; d < kPrefillGQADimTiles; ++d) {
                acc[d].thread_elements()[0] *= rescale;
                acc[d].thread_elements()[1] *= rescale;
                simdgroup_float8x8 b;
                b.thread_elements()[0] = float(v_row[d * 8u]);
                b.thread_elements()[1] = float(v_row[d * 8u + 1u]);
                simdgroup_multiply_accumulate(acc[d], probs, b, acc[d]);
            }
        }

        // As in the tiled kernel, a row that saw no visible key writes zeros.
        const float inv_sum = row_sum > 0.0f ? 1.0f / row_sum : 0.0f;
        device half* out_row = O + t * p.oTokenStrideElements
            + (kvh * kPrefillGQAHeadsPerKV + fm) * kPrefillGQAHeadDim + fn;
        for (uint d = 0u; d < kPrefillGQADimTiles; ++d) {
            out_row[d * 8u] = half(acc[d].thread_elements()[0] * inv_sum);
            out_row[d * 8u + 1u] = half(acc[d].thread_elements()[1] * inv_sum);
        }
    }
    """#
}
