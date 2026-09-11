import Foundation
import Metal

/// A resident Flash-Next projection matrix, in whichever of the three dtypes an
/// install may carry it.
///
/// The production `qwen38flashnext` install quantizes every rank-2 BF16
/// `.weight` whose row length the group size divides to affine group-64 —
/// INT4 for everything except the two MoE gating tensors (`mlp.gate`, the
/// top-10-of-512 router, and `mlp.shared_expert_gate`), which the repacker's
/// `QuantBitPolicy` keeps at INT8. Norms, conv kernels and any row width 64
/// does not divide ride through as BF16. The parity install is BF16 for
/// *everything*, so the same runner has to drive all three — hence one type
/// that carries the dtype with the buffer rather than a runner-wide assumption.
///
/// # Why the width is derived rather than read
///
/// The resident index has one dtype byte, and it says `0` — "packed integer
/// weights with BF16 scale and bias companions" — at both INT4 and INT8. The
/// writer has always emitted it that way, so a mixed-width install cannot be
/// told apart by dtype and there is no spare field to widen without breaking
/// every install on disk.
///
/// The entry already carries the answer implicitly: `sizeBytes` is exactly
/// `rows * columns * bits / 8`, so the width falls out of bytes and shape.
/// Deriving it that way is not a workaround, it is the same rule
/// `Scripts/quantizer-mixture-compare.py` uses to audit an install's mixture
/// against a control conversion's, and it deliberately trusts neither the
/// manifest's coarse per-slot summary nor the family. It is also what lets the
/// uniform-INT4 install predating the 2026-09-10 router measurement keep
/// loading unchanged: nothing about it says "INT4 family", each tensor simply
/// answers for itself.
enum FlashNextWeightMatrix {
    case int4(weights: MTLBuffer, weightsOffset: Int,
              scales: MTLBuffer, scalesOffset: Int,
              biases: MTLBuffer, biasesOffset: Int)
    case int8(weights: MTLBuffer, weightsOffset: Int,
              scales: MTLBuffer, scalesOffset: Int,
              biases: MTLBuffer, biasesOffset: Int)
    case bf16(buffer: MTLBuffer, offset: Int)

    /// Build from a loaded tensor view. Dtype 0 is affine-packed with companion
    /// scale/bias slices, at the width `sizeBytes` implies; dtype 1 is dense
    /// BF16.
    static func from(_ view: TensorView) -> FlashNextWeightMatrix {
        switch view.dtype {
        case 0:
            switch packedWeightBits(view) {
            case 4:
                return .int4(weights: view.buffer, weightsOffset: Int(view.offset),
                             scales: view.buffer, scalesOffset: Int(view.scaleOffset),
                             biases: view.buffer, biasesOffset: Int(view.biasOffset))
            case 8:
                return .int8(weights: view.buffer, weightsOffset: Int(view.offset),
                             scales: view.buffer, scalesOffset: Int(view.scaleOffset),
                             biases: view.buffer, biasesOffset: Int(view.biasOffset))
            case let bits:
                preconditionFailure(
                    "Flash-Next packed projections are INT4 or INT8 affine group-64; "
                        + "this entry's \(view.length) bytes over shape "
                        + "[\(view.shape.0), \(view.shape.1)] imply \(bits) bits")
            }
        case 1:
            return .bf16(buffer: view.buffer, offset: Int(view.offset))
        default:
            preconditionFailure(
                "Flash-Next projections are affine-packed or BF16, got dtype \(view.dtype)")
        }
    }

    /// `sizeBytes * 8 / (rows * columns)`, or 0 when the shape cannot carry a
    /// width (which `from` turns into the same loud failure as a bad one).
    static func packedWeightBits(_ view: TensorView) -> Int {
        let rows = Int(view.shape.0)
        let columns = Int(view.shape.1)
        guard rows > 0, columns > 0 else { return 0 }
        let weights = rows * columns
        let bits = Int(view.length) * 8
        guard bits % weights == 0 else { return 0 }
        return bits / weights
    }
}

/// `y = W . x` for a Flash-Next projection, dispatching on the stored dtype.
///
/// INT4 goes through the shipped `DequantInt4GEMV`; BF16 through this family's
/// own `flashnext_gemv_bf16`. Both have an FP32-output form, which the
/// hyper-connection path uses wherever a value is about to be pushed through a
/// sigmoid — rounding a pre-activation to FP16 costs more than the buffer saves.
///
/// INT8 goes through `router_gemv_gemma4_r4` — the shipped Gemma/Qwen router
/// GEMV, reused verbatim. It is the right kernel rather than a near-enough one:
/// it decodes one `uint8` per weight against one BF16 scale and one BF16 bias
/// per group of 64, which is byte-for-byte the layout `Int8AffineEncoder`
/// writes, and it already accumulates and stores in FP32, which is what both
/// INT8 call sites in this family want. Its one extra input is a per-element
/// `effective_scale` on the activation, which Gemma uses and this family does
/// not; binding a vector of ones makes it inert, exactly as DeepSeek V4 already
/// does with the BF16 sibling `router_gemv_bf16_r4`.
///
/// **No new Metal was written for the INT8 path, and none was needed.** The
/// dispatch below is the same unspecialized pipeline, threadgroup shape and
/// buffer binding that `MoE.encodeRouterGemma4` uses, which
/// `RouterWideTopK10Tests.decodeRouterAt512TopK10MatchesTheReference` already
/// gates against a CPU reference at this family's exact production geometry —
/// 512 x 2560, INT8 affine group-64, top-10. So the router logits are produced
/// by an already-parity-tested path rather than by a new one needing its own
/// gate. `FlashNextMatVecInt8Tests` adds the one thing that test cannot cover:
/// that *this* wrapper binds it correctly.
final class FlashNextMatVec {

    private let int4: DequantInt4GEMV
    private let int4MultiX: DequantInt4GEMVMultiX
    private let prefillQMM: PrefillInt4QMM
    private let prefillMPP: MPPPrefillInt4QMM
    private let bf16PSO: MTLComputePipelineState
    private let bf16F32PSO: MTLComputePipelineState
    private let int8PSO: MTLComputePipelineState
    /// A BF16 vector of ones, `int8Columns` long: the identity value for
    /// `router_gemv_gemma4_r4`'s activation scale. Sized once at init rather
    /// than grown on demand so the INT8 path allocates nothing per token and a
    /// row wider than the runner declared fails loudly instead of reading past
    /// the end.
    private let onesActivationScale: MTLBuffer?
    private let int8Columns: Int

    /// Rows per threadgroup; mirrors `kFlashNextGemvRowsPerThreadgroup`.
    private static let rowsPerThreadgroup = 8
    /// Rows per threadgroup in `router_gemv_gemma4_r4` — the `_r4` in its name.
    private static let int8RowsPerThreadgroup = 4

    /// `int8Columns` is the widest row the caller will ever hand to the INT8
    /// path (the hidden size, for this family: both INT8 tensors are
    /// `[*, hidden]`). Zero — the default — declares that this instance sees no
    /// INT8 tensors, which is what every kernel-level test and the BF16 parity
    /// install want.
    init(context: MetalContext, int4: DequantInt4GEMV, int8Columns: Int = 0) throws {
        self.int4 = int4
        self.int4MultiX = try DequantInt4GEMVMultiX(context: context)
        self.prefillQMM = try PrefillInt4QMM(context: context)
        self.prefillMPP = MPPPrefillInt4QMM(context: context)
        self.int8Columns = int8Columns
        self.bf16PSO = try context.pipeline("flashnext_gemv_bf16",
                                            constants: [],
                                            maxTotalThreadsPerThreadgroup: 256)
        self.bf16F32PSO = try context.pipeline("flashnext_gemv_bf16_f32out",
                                               constants: [],
                                               maxTotalThreadsPerThreadgroup: 256)
        // Unspecialized and 512-thread-capable: the same pipeline
        // `MoE.encodeRouterGemma4` falls back to for any shape that is not
        // Gemma's, which is the one `RouterWideTopK10Tests` drives at 512x2560.
        self.int8PSO = try context.pipeline("router_gemv_gemma4_r4",
                                            constants: [],
                                            maxTotalThreadsPerThreadgroup: 512)
        if int8Columns > 0 {
            let ones = [UInt16](repeating: Self.bf16One, count: int8Columns)
            guard let buffer = context.device.makeBuffer(
                    bytes: ones,
                    length: ones.count * MemoryLayout<UInt16>.stride,
                    options: .storageModeShared) else {
                throw MetalError.noDevice
            }
            self.onesActivationScale = buffer
        } else {
            self.onesActivationScale = nil
        }
    }

    /// BF16 `1.0` — the top 16 bits of FP32 `1.0` (`0x3F80_0000`).
    private static let bf16One: UInt16 = 0x3F80

    func encode(commandBuffer: MTLCommandBuffer,
                matrix: FlashNextWeightMatrix,
                x: MTLBuffer, xOffset: Int = 0,
                y: MTLBuffer, yOffset: Int = 0,
                rows: Int, cols: Int,
                outputFloat32: Bool = false) {
        precondition(rows > 0 && cols > 0)
        switch matrix {
        case let .int4(weights, weightsOffset, scales, scalesOffset,
                       biases, biasesOffset):
            int4.encode(commandBuffer: commandBuffer,
                        weights: weights, weightsOffset: weightsOffset,
                        scales: scales, scalesOffset: scalesOffset,
                        biases: biases, biasesOffset: biasesOffset,
                        x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                        m: UInt32(rows), n: UInt32(cols),
                        outputFloat32: outputFloat32)
        case let .int8(weights, weightsOffset, scales, scalesOffset,
                       biases, biasesOffset):
            encodeInt8(commandBuffer: commandBuffer,
                       weights: weights, weightsOffset: weightsOffset,
                       scales: scales, scalesOffset: scalesOffset,
                       biases: biases, biasesOffset: biasesOffset,
                       x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                       rows: rows, cols: cols, outputFloat32: outputFloat32)
        case let .bf16(buffer, offset):
            guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
            let pso = outputFloat32 ? bf16F32PSO : bf16PSO
            enc.setComputePipelineState(pso)
            enc.setBuffer(buffer, offset: offset, index: 0)
            enc.setBuffer(x, offset: xOffset, index: 1)
            enc.setBuffer(y, offset: yOffset, index: 2)
            var rowsVar = UInt32(rows)
            var colsVar = UInt32(cols)
            enc.setBytes(&rowsVar, length: MemoryLayout<UInt32>.size, index: 3)
            enc.setBytes(&colsVar, length: MemoryLayout<UInt32>.size, index: 4)
            let groups = (rows + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup
            enc.dispatchThreadgroups(
                MTLSize(width: groups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32 * Self.rowsPerThreadgroup,
                                               height: 1, depth: 1))
            enc.endEncoding()
        }
    }

    /// `Y[T, M] = X[T, N] * W[M, N]^T` for a contiguous prefill block.
    ///
    /// Production INT4 projections use TensorOps for a tile-tall block and the
    /// decode-identical multi-X kernel otherwise. FP32 outputs deliberately use
    /// multi-X in groups of eight: the QSA indexer and hyper-connection gates
    /// consume pre-activations whose exact FP32 reduction order is part of the
    /// model's discrete decisions. Dense BF16 is retained for the parity toy,
    /// where repeated rows are a reference-only compatibility path rather than
    /// a production checkpoint path.
    func encodeBatched(commandBuffer: MTLCommandBuffer,
                       matrix: FlashNextWeightMatrix,
                       x: MTLBuffer, xOffset: Int = 0,
                       y: MTLBuffer, yOffset: Int = 0,
                       matrixRows: Int, matrixColumns: Int,
                       tokens: Int,
                       outputFloat32: Bool = false) {
        precondition(tokens > 0)
        let half = MemoryLayout<Float16>.stride
        let outStride = outputFloat32
            ? MemoryLayout<Float>.stride : MemoryLayout<Float16>.stride
        switch matrix {
        case let .int4(weights, weightsOffset, scales, scalesOffset,
                       biases, biasesOffset):
            if !outputFloat32, tokens >= MPPPrefillInt4QMM.tileN,
               prefillMPP.isAvailable {
                let path = prefillMPP.encode(
                    commandBuffer: commandBuffer,
                    weights: weights, weightsOffset: weightsOffset,
                    scales: scales, scalesOffset: scalesOffset,
                    biases: biases, biasesOffset: biasesOffset,
                    x: x, xOffset: xOffset,
                    y: y, yOffset: yOffset,
                    m: tokens, n: matrixRows, k: matrixColumns)
                if path == .affineThreadgroupF16 { return }
            }
            if !outputFloat32, tokens >= 32 {
                prefillQMM.encode(commandBuffer: commandBuffer,
                                  weights: weights, weightsOffset: weightsOffset,
                                  scales: scales, scalesOffset: scalesOffset,
                                  biases: biases, biasesOffset: biasesOffset,
                                  x: x, xOffset: xOffset,
                                  y: y, yOffset: yOffset,
                                  t: tokens, n: matrixRows, k: matrixColumns)
                return
            }
            var row = 0
            while row < tokens {
                let count = min(DequantInt4GEMVMultiX.maxTokens, tokens - row)
                int4MultiX.encode(
                    commandBuffer: commandBuffer,
                    weights: weights, weightsOffset: weightsOffset,
                    scales: scales, scalesOffset: scalesOffset,
                    biases: biases, biasesOffset: biasesOffset,
                    x: x, xOffset: xOffset + row * matrixColumns * half,
                    y: y, yOffset: yOffset + row * matrixRows * outStride,
                    m: matrixRows, n: matrixColumns, tokens: count,
                    outputFloat32: outputFloat32)
                row += count
            }
        case .int8, .bf16:
            for row in 0..<tokens {
                encode(commandBuffer: commandBuffer, matrix: matrix,
                       x: x, xOffset: xOffset + row * matrixColumns * half,
                       y: y, yOffset: yOffset + row * matrixRows * outStride,
                       rows: matrixRows, cols: matrixColumns,
                       outputFloat32: outputFloat32)
            }
        }
    }

    /// Batched INT4 projection that retains the cooperative accumulator in FP32.
    /// Continuous hyper-connection gates use it directly. The Flash-Next
    /// indexer also opts in before its deterministic selector; routed-MoE
    /// top-k keeps the exact multi-X implementation above.
    func encodeBatchedContinuousF32(commandBuffer: MTLCommandBuffer,
                                    matrix: FlashNextWeightMatrix,
                                    x: MTLBuffer, xOffset: Int = 0,
                                    y: MTLBuffer, yOffset: Int = 0,
                                    matrixRows: Int, matrixColumns: Int,
                                    tokens: Int) {
        if case let .int4(weights, weightsOffset, scales, scalesOffset,
                          biases, biasesOffset) = matrix,
           tokens >= MPPPrefillInt4QMM.tileN,
           prefillMPP.isFloat32Available {
            let path = prefillMPP.encodeFloat32(
                commandBuffer: commandBuffer,
                weights: weights, weightsOffset: weightsOffset,
                scales: scales, scalesOffset: scalesOffset,
                biases: biases, biasesOffset: biasesOffset,
                x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                m: tokens, n: matrixRows, k: matrixColumns)
            if path == .affineThreadgroupF32 { return }
        }
        encodeBatched(commandBuffer: commandBuffer, matrix: matrix,
                      x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                      matrixRows: matrixRows, matrixColumns: matrixColumns,
                      tokens: tokens, outputFloat32: true)
    }

    private func encodeInt8(commandBuffer: MTLCommandBuffer,
                            weights: MTLBuffer, weightsOffset: Int,
                            scales: MTLBuffer, scalesOffset: Int,
                            biases: MTLBuffer, biasesOffset: Int,
                            x: MTLBuffer, xOffset: Int,
                            y: MTLBuffer, yOffset: Int,
                            rows: Int, cols: Int,
                            outputFloat32: Bool) {
        // The kernel stores FP32 unconditionally. Every INT8 tensor this family
        // has is a gating tensor whose consumer wants FP32 — the router logits
        // feed `router_topk_select_k10_par`, the shared-expert gate feeds a
        // sigmoid — so an FP16 request here means a tensor got an INT8 width it
        // was never meant to have, and silently writing FP32 into an FP16
        // buffer would corrupt twice the bytes asked for.
        precondition(outputFloat32,
                     "the INT8 router GEMV stores FP32; a caller asking for FP16 "
                         + "output has bound a tensor this path does not serve")
        precondition(cols.isMultiple(of: Quantization.groupSize),
                     "INT8 affine group-64 needs a row length \(Quantization.groupSize) "
                         + "divides, got \(cols)")
        guard let effectiveScale = onesActivationScale else {
            preconditionFailure(
                "this FlashNextMatVec was built with int8Columns: 0, so it holds no "
                    + "activation-scale vector — an INT8 tensor reached a runner that "
                    + "did not declare one")
        }
        precondition(cols <= int8Columns,
                     "INT8 row length \(cols) exceeds the declared int8Columns "
                         + "\(int8Columns)")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        var rowsVar = UInt32(rows)
        var colsVar = UInt32(cols)
        enc.setComputePipelineState(int8PSO)
        enc.setBuffer(weights, offset: weightsOffset, index: 0)
        enc.setBuffer(scales, offset: scalesOffset, index: 1)
        enc.setBuffer(biases, offset: biasesOffset, index: 2)
        enc.setBuffer(x, offset: xOffset, index: 3)
        enc.setBuffer(effectiveScale, offset: 0, index: 4)
        enc.setBuffer(y, offset: yOffset, index: 5)
        enc.setBytes(&rowsVar, length: MemoryLayout<UInt32>.stride, index: 6)
        enc.setBytes(&colsVar, length: MemoryLayout<UInt32>.stride, index: 7)
        let groups = (rows + Self.int8RowsPerThreadgroup - 1) / Self.int8RowsPerThreadgroup
        enc.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * Self.int8RowsPerThreadgroup,
                                           height: 1, depth: 1))
        enc.endEncoding()
    }
}
