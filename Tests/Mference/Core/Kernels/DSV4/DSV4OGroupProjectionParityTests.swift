import Foundation
import Metal
import Testing

@testable import Mference
import MferenceValidationSupport

/// The DeepSeek-V4 grouped output projection used to be `oGroups` separate
/// INT4 GEMV dispatches per layer, each with its own byte-offset arithmetic in
/// Swift (8 dispatches x 43 layers x every token). `dsv4_o_group_gemv_int4`
/// covers all groups in one dispatch by deriving the group from the output row
/// index, and reuses `dequant_int4_gemv_simd`'s row body verbatim — so the
/// result must be bit-identical, not merely close.
@Suite struct DSV4OGroupProjectionParityTests {

    @Test func fusedDispatchMatchesPerGroupGEMVsExactly() throws {
        let cfg = ArchConfig.deepseekV4Flash_284B_A13B
        let ca = cfg.compressedAttention
        let rank = ca.oLoraRank
        let groups = ca.oGroups
        let groupIn = cfg.numHeads * cfg.fullHeadDim / groups
        let rows = rank * groups
        let groupsPerRow = groupIn / Quantization.groupSize

        let ctx = try MetalContext()
        let kernels = try DSV4Kernels(context: ctx, config: cfg)
        let int4 = try DequantInt4GEMV(context: ctx,
                                       additionalShapes: cfg.decodeInt4GEMVShapes)
        let device = ctx.device
        var rng = SplitMix64(seed: 0x0A_5017_2026_0802)

        // Packed INT4 weights: one nibble per input channel, row-major.
        let weightBytes = rows * groupIn / 2
        let weights = try #require(device.makeBuffer(length: weightBytes,
                                                    options: .storageModeShared))
        let words = weights.contents().assumingMemoryBound(to: UInt64.self)
        for i in 0..<(weightBytes / 8) { words[i] = rng.next() }

        // BF16 affine scale / bias per 64-channel group.
        func bf16Buffer(_ count: Int, lo: Float, hi: Float) throws -> MTLBuffer {
            let buf = try #require(device.makeBuffer(
                length: count * MemoryLayout<UInt16>.stride,
                options: .storageModeShared))
            let ptr = buf.contents().assumingMemoryBound(to: UInt16.self)
            for i in 0..<count {
                ptr[i] = UInt16(truncatingIfNeeded:
                                    rng.uniform(lo, hi).bitPattern >> 16)
            }
            return buf
        }
        let scales = try bf16Buffer(rows * groupsPerRow, lo: 0.002, hi: 0.02)
        let biases = try bf16Buffer(rows * groupsPerRow, lo: -0.05, hi: 0.05)

        // Activation: the whole roped attention output, all groups end to end.
        let x = try #require(device.makeBuffer(
            length: groups * groupIn * MemoryLayout<Float16>.stride,
            options: .storageModeShared))
        let xPtr = x.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<(groups * groupIn) { xPtr[i] = Float16(rng.uniform(-1, 1)) }

        func outBuffer() throws -> MTLBuffer {
            let buf = try #require(device.makeBuffer(
                length: rows * MemoryLayout<Float16>.stride,
                options: .storageModeShared))
            memset(buf.contents(), 0, buf.length)
            return buf
        }
        let yFused = try outBuffer()
        let yPerGroup = try outBuffer()

        let cb = try #require(ctx.queue.makeCommandBuffer())
        kernels.encodeOGroupProjection(
            commandBuffer: cb,
            weights: weights, weightsOffset: 0,
            scales: scales, scalesOffset: 0,
            biases: biases, biasesOffset: 0,
            x: x, y: yFused,
            rank: rank, groupIn: groupIn, groups: groups)

        // The retired call site: one GEMV per group, offsets computed Swift-side.
        let fp16 = MemoryLayout<Float16>.stride
        let rowBytes = groupIn / 2
        for g in 0..<groups {
            let rowBase = g * rank
            int4.encode(commandBuffer: cb,
                        weights: weights, weightsOffset: rowBase * rowBytes,
                        scales: scales, scalesOffset: rowBase * groupsPerRow * 2,
                        biases: biases, biasesOffset: rowBase * groupsPerRow * 2,
                        x: x, xOffset: g * groupIn * fp16,
                        y: yPerGroup, yOffset: g * rank * fp16,
                        m: UInt32(rank), n: UInt32(groupIn))
        }
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)

        let fusedPtr = yFused.contents().assumingMemoryBound(to: Float16.self)
        let refPtr = yPerGroup.contents().assumingMemoryBound(to: Float16.self)
        var mismatches = 0
        var firstMismatch = -1
        var nonZero = 0
        for i in 0..<rows {
            if fusedPtr[i] != refPtr[i] {
                mismatches += 1
                if firstMismatch < 0 { firstMismatch = i }
            }
            if fusedPtr[i] != 0 { nonZero += 1 }
        }
        #expect(mismatches == 0,
                "\(mismatches)/\(rows) rows differ; first at row \(firstMismatch), group \(firstMismatch / max(rank, 1))")
        #expect(nonZero > rows / 2, "fused output is mostly zero")
    }
}
