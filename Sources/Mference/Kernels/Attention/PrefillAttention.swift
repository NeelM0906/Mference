import Foundation
import Metal

struct PrefillAttentionParams: Sendable, Equatable {
    var startPosition: UInt32
    var queryCount: UInt32
    var headDim: UInt32
    var numQHeads: UInt32
    var numKVHeads: UInt32
    var kvValidCount: UInt32
    var slidingWindow: UInt32
    var kvTokenStrideElements: UInt32
    var qTokenStrideElements: UInt32
    var oTokenStrideElements: UInt32
    var scale: Float

    init(startPosition: UInt32,
                queryCount: UInt32,
                headDim: UInt32,
                numQHeads: UInt32,
                numKVHeads: UInt32,
                kvValidCount: UInt32,
                slidingWindow: UInt32,
                kvTokenStrideElements: UInt32,
                qTokenStrideElements: UInt32,
                oTokenStrideElements: UInt32,
                scale: Float) {
        self.startPosition = startPosition
        self.queryCount = queryCount
        self.headDim = headDim
        self.numQHeads = numQHeads
        self.numKVHeads = numKVHeads
        self.kvValidCount = kvValidCount
        self.slidingWindow = slidingWindow
        self.kvTokenStrideElements = kvTokenStrideElements
        self.qTokenStrideElements = qTokenStrideElements
        self.oTokenStrideElements = oTokenStrideElements
        self.scale = scale
    }
}


final class PrefillAttention {
    private let context: MetalContext
    private let psoCausalTiled: MTLComputePipelineState
    private let psoFullTensorOps2DValidityV2: MTLComputePipelineState?
    private var psoGQASimd: MTLComputePipelineState?
    private let gemmaQAT: GemmaQATPrefillAttention?

    /// Whether the MPP tensor-ops full-attention kernel is usable here. False
    /// with shader libraries below MSL 4.0, on GPUs where the pipeline does not
    /// build, and for the exact QAT profile, which keeps its source-arithmetic
    /// batched kernels.
    var tensorOpsPipelineAvailable: Bool { psoFullTensorOps2DValidityV2 != nil }

    /// Whether `attention_prefill_gqa8_d256_simd` was requested and built. The
    /// caller requests it only where it is the chosen default (Qwen 3.6 on GPUs
    /// without the Apple10 TensorOps path); everyone else keeps the tiled kernel.
    var gqaSimdPipelineAvailable: Bool { psoGQASimd != nil }

    /// Dispatches of the GQA kernel so far, so the real-model tolerance gate
    /// can prove the candidate ran it.
    private(set) var gqaSimdDispatches = 0

    /// Tests only: send the shapes the GQA kernel covers back to the tiled
    /// kernel, so the tolerance gate compares both in one process.
    func useTiledKernelForGQAShapes() { psoGQASimd = nil }

    /// Query tokens per threadgroup of the GQA kernel in use.
    private var gqaTokensPerGroup = PrefillAttention.gqaSimdTokensPerGroup

    /// Tests only: run another build of the GQA kernel (same buffers and
    /// params, one simdgroup per query token, `tokensPerGroup` tokens per
    /// threadgroup) for the shapes it covers, so the byte-identity gate can
    /// compare it with the default kernel in one process.
    func useGQAPipelineForTesting(_ pipeline: MTLComputePipelineState, tokensPerGroup: Int) {
        precondition(tokensPerGroup > 0
                     && tokensPerGroup * 32 <= pipeline.maxTotalThreadsPerThreadgroup)
        psoGQASimd = pipeline
        gqaTokensPerGroup = tokensPerGroup
    }

    /// `gemmaQATFullAttentionTensorOps` lets the QAT profile's full-attention
    /// layers use the tensor-ops kernel; its sliding-window layers and every
    /// fallback keep the source arithmetic.
    /// `gqaSimdgroup` selects the GQA-shared simdgroup-matrix kernel for the
    /// shapes it covers (headDim 256, 8 query heads per KV head, no window clip,
    /// ring or sinks) in place of the tiled kernel. It changes output within
    /// FP32-reassociation tolerance, so only the caller's default may set it.
    init(context: MetalContext, gemmaQATMaxContext: Int? = nil,
         gemmaQATFullAttentionTensorOps: Bool = false,
         gqaSimdgroup: Bool = false) throws {
        self.context = context
        self.gemmaQAT = try gemmaQATMaxContext.map { try GemmaQATPrefillAttention(context: context, maxContext: $0) }
        self.psoCausalTiled = try context.pipeline("attention_prefill_causal_tiled")
        // Selected by pipeline capability, not GPU family: the kernel also
        // builds on Apple8 (M2), where it measured 9x faster than tiled
        // attention. Below MSL 4.0 it is not in the library.
        self.psoFullTensorOps2DValidityV2 = gemmaQATMaxContext == nil || gemmaQATFullAttentionTensorOps
            ? try? context.pipeline("attention_prefill_full_tensorops_2d_validity_v2")
            : nil
        // The kernel is written for 8 simdgroups per threadgroup; a GPU that
        // allows fewer threads keeps the tiled kernel.
        let gqa = gqaSimdgroup ? try context.pipeline("attention_prefill_gqa8_d256_simd") : nil
        self.psoGQASimd = gqa.flatMap {
            $0.maxTotalThreadsPerThreadgroup >= Self.gqaSimdTokensPerGroup * 32 ? $0 : nil
        }
    }

    /// Returns whether the tensor-ops kernel ran.
    @discardableResult
    func encodeCausal(commandBuffer: MTLCommandBuffer,
                             q: MTLBuffer, qOffset: Int = 0,
                             k: MTLBuffer, kOffset: Int = 0,
                             v: MTLBuffer, vOffset: Int = 0,
                             out: MTLBuffer, outOffset: Int = 0,
                             params: PrefillAttentionParams,
                             kvRingCapacity: UInt32 = 0,
                             sinks: MTLBuffer? = nil,
                             sinksOffset: Int = 0,
                             numSinks: UInt32 = 0,
                             path: RuntimePrefillAttentionPath = .causalTiled) -> Bool {
        validate(params)
        precondition(sinks != nil || sinksOffset == 0,
                     "a sink offset requires a sink buffer")
        precondition(numSinks == 0 || sinks != nil,
                     "a sink count requires a sink buffer")

        let requestsTensorOps = path == .fullTensorOps2DPreferred
            || path == .fullTensorOps2DValidityV2
        // TensorOps starts its key loop at zero and ignores slidingWindow, so
        // these are visibility guards rather than shape optimizations.
        let windowNeverClips = params.slidingWindow == 0
            || params.slidingWindow >= params.kvValidCount
        let tensorOpsShape = requestsTensorOps
            && kvRingCapacity == 0
            && numSinks == 0
            && windowNeverClips
            && params.headDim == 512
            && params.numQHeads == 16
            && params.numKVHeads == 2
            && params.scale == 1.0
        let tensorOpsPipeline = tensorOpsShape ? psoFullTensorOps2DValidityV2 : nil

        if let gemmaQAT, tensorOpsPipeline == nil {
            gemmaQAT.encode(commandBuffer: commandBuffer,
                q: q, qOffset: qOffset, k: k, kOffset: kOffset,
                v: v, vOffset: vOffset, out: out, outOffset: outOffset,
                params: params, ringCapacity: kvRingCapacity)
            return false
        }

        let useTensorOps = tensorOpsPipeline != nil
        let gqaSimdPipeline = useTensorOps ? nil : gqaSimdPipeline(for: params,
                                                                  kvRingCapacity: kvRingCapacity,
                                                                  numSinks: numSinks,
                                                                  windowNeverClips: windowNeverClips,
                                                                  kOffset: kOffset, vOffset: vOffset)
        let pipeline: MTLComputePipelineState
        if let gqaSimdPipeline {
            pipeline = gqaSimdPipeline
            gqaSimdDispatches += 1
        } else if let tensorOpsPipeline {
            pipeline = tensorOpsPipeline
        } else if tensorOpsShape
                    && path == .fullTensorOps2DValidityV2
                    && context.device.supportsApple10TensorOps {
            // The device advertises MPP tensor support, so a missing pipeline
            // means the kernel failed to build — a bug, not a platform limit.
            preconditionFailure(
                "TensorOps 2D prefill attention pipeline is missing on an Apple10 device")
        } else {
            // Explicit mode also falls back for incompatible shapes, and on
            // hosts where the pipeline does not build or without MSL 4.0.
            // Benchmark fixtures must use 512/16/2 to prove that TensorOps ran.
            pipeline = causalTiledPipeline(kvRingCapacity: kvRingCapacity)
        }
        let headDim = Int(params.headDim)
        let threadWidth = max(1, pipeline.threadExecutionWidth)
        let threadCount = gqaSimdPipeline != nil
            ? gqaTokensPerGroup * 32
            : useTensorOps
            ? 128
            : roundUp(max(threadWidth, headDim), toMultipleOf: threadWidth)
        precondition(threadCount <= pipeline.maxTotalThreadsPerThreadgroup,
                     "tiled prefill attention requires headDim <= maxTotalThreadsPerThreadgroup")

        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return false }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(q, offset: qOffset, index: 0)
        enc.setBuffer(k, offset: kOffset, index: 1)
        enc.setBuffer(v, offset: vOffset, index: 2)
        enc.setBuffer(out, offset: outOffset, index: 3)
        var p = params
        enc.setBytes(&p, length: MemoryLayout<PrefillAttentionParams>.stride, index: 4)
        enc.setBuffer(sinks ?? out, offset: sinksOffset, index: 5)
        var sinkCount = numSinks
        enc.setBytes(&sinkCount, length: MemoryLayout<UInt32>.size, index: 6)
        let groups = gqaSimdPipeline != nil
            ? MTLSize(width: (Int(params.queryCount) + gqaTokensPerGroup - 1)
                          / gqaTokensPerGroup,
                      height: Int(params.numKVHeads),
                      depth: 1)
            : useTensorOps
            ? MTLSize(width: Int(params.queryCount),
                      height: Int(params.numQHeads) / 8,
                      depth: 1)
            : MTLSize(width: Int(params.queryCount),
                      height: Int(params.numQHeads),
                      depth: 1)
        enc.dispatchThreadgroups(
            groups,
            threadsPerThreadgroup: MTLSize(width: threadCount, height: 1, depth: 1))
        enc.endEncoding()
        return useTensorOps
    }


    private func validate(_ params: PrefillAttentionParams) {
        precondition(params.headDim > 0, "headDim must be positive")
        precondition(params.queryCount > 0, "queryCount must be positive")
        precondition(params.numQHeads > 0, "numQHeads must be positive")
        precondition(params.numKVHeads > 0, "numKVHeads must be positive")
        precondition(params.numQHeads % params.numKVHeads == 0,
                     "numQHeads must be divisible by numKVHeads")
        precondition(params.qTokenStrideElements >= params.numQHeads * params.headDim,
                     "q token stride is too small")
        precondition(params.oTokenStrideElements >= params.numQHeads * params.headDim,
                     "output token stride is too small")
        precondition(params.kvTokenStrideElements >= params.numKVHeads * params.headDim,
                     "KV token stride is too small")
        precondition(params.startPosition + params.queryCount <= params.kvValidCount,
                     "kvValidCount must include all in-flight query rows")
    }


    /// Query tokens per threadgroup of the GQA kernel (one per simdgroup):
    /// they share each staged 8-key K/V block (8,448 bytes of threadgroup
    /// memory).
    static let gqaSimdTokensPerGroup = 8

    /// The GQA kernel covers exactly Qwen 3.6's full-attention shape.
    private func gqaSimdPipeline(for params: PrefillAttentionParams,
                                 kvRingCapacity: UInt32,
                                 numSinks: UInt32,
                                 windowNeverClips: Bool,
                                 kOffset: Int,
                                 vOffset: Int) -> MTLComputePipelineState? {
        // K/V rows are loaded as half4, so they must start 8-byte aligned.
        guard let psoGQASimd,
              kvRingCapacity == 0,
              numSinks == 0,
              windowNeverClips,
              params.headDim == 256,
              params.numQHeads == params.numKVHeads * 8,
              params.kvTokenStrideElements % 4 == 0,
              kOffset % 8 == 0,
              vOffset % 8 == 0 else { return nil }
        return psoGQASimd
    }

    private func roundUp(_ value: Int, toMultipleOf multiple: Int) -> Int {
        ((value + multiple - 1) / multiple) * multiple
    }

    private func causalTiledPipeline(kvRingCapacity: UInt32) -> MTLComputePipelineState {
        guard kvRingCapacity > 0 else { return psoCausalTiled }
        do {
            return try context.pipeline(
                "attention_prefill_causal_tiled",
                constants: [MetalFunctionConstant(index: 76, value: .uint32(kvRingCapacity))])
        } catch {
            preconditionFailure("failed to build FP16 KV ring prefill attention pipeline: \(error)")
        }
    }
}
