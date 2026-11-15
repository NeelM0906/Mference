import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

/// `gguf_lut4_expand` against `GGUFLUT4Reference.decompose`, byte for byte.
/// Each type runs several jobs of different lengths into one poisoned
/// destination, so an index, D or M byte written out of place shows up as a
/// mismatch, and so do bytes the kernel must not touch.
@Suite struct GGUFLUT4ExpandTests {
    static let poison: UInt8 = 0xA5

    /// Planar LUT4 bytes of `parts`: indices (IQ4_NL nibble order per 32),
    /// then D (fp16 LE), then M.
    static func planar(_ parts: GGUFLUT4Reference.Parts) -> (indices: [UInt8], d: [UInt8], m: [UInt8]) {
        var indices = [UInt8]()
        for g in stride(from: 0, to: parts.indices.count, by: 32) {
            for r in 0..<16 { indices.append(parts.indices[g + r] | parts.indices[g + r + 16] << 4) }
        }
        let d = parts.d.flatMap { [UInt8($0 & 0xFF), UInt8($0 >> 8)] }
        return (indices, d, parts.m.map { UInt8(bitPattern: $0) })
    }

    @Test(arguments: GGUFLUT4.SourceType.allCases)
    func gpuExpansionMatchesCPUReference(type: GGUFLUT4.SourceType) throws {
        let ctx = try MetalContext()
        let kernel = try GGUFLUT4Expand(context: ctx)
        var rng = SeedTree(0x4C55_5445).key("lut4-expand-\(type.rawValue)")
        let jobBlocks = [1, 7, 40, 129]
        let p = GGUFLUT4.parameters(type)
        let mPerBlock = p.bm == 0 ? 0 : type.blockWeights / p.bm

        var src = [UInt8]()
        var expected = [UInt8]()
        var jobs = [GGUFLUT4Expand.Job]()
        for blocks in jobBlocks {
            let bytes = randomGGUFBlocks(type, count: blocks, rng: &rng)
            let parts = bytes.withUnsafeBytes { GGUFLUT4Reference.decompose(type, $0) }
            let (indices, d, m) = Self.planar(parts)
            #expect(indices.count == blocks * type.blockWeights / 2)
            #expect(m.count == blocks * mPerBlock)
            // Sections start 16-aligned, with a poisoned gap after each.
            func align(_ n: Int) -> Int { (n + 15) / 16 * 16 }
            let indexAt = align(expected.count + 16)
            expected += [UInt8](repeating: Self.poison, count: indexAt - expected.count) + indices
            let dAt = align(expected.count + 16)
            expected += [UInt8](repeating: Self.poison, count: dAt - expected.count) + d
            let mAt = align(expected.count + 16)
            expected += [UInt8](repeating: Self.poison, count: mAt - expected.count) + m
            jobs.append(GGUFLUT4Expand.Job(
                srcOffset: UInt64(src.count), dstIndexOffset: UInt64(indexAt),
                dstDOffset: UInt64(dAt), dstMOffset: UInt64(mAt), blocks: UInt32(blocks)))
            src += bytes
            src += [UInt8](repeating: 0, count: 3)  // unaligned next source block
        }
        expected += [UInt8](repeating: Self.poison, count: 64)

        let srcBuf = try #require(src.withUnsafeBytes {
            ctx.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
        })
        let dstBuf = try #require(ctx.device.makeBuffer(length: expected.count, options: .storageModeShared))
        memset(dstBuf.contents(), Int32(Self.poison), expected.count)

        let cmd = try #require(ctx.queue.makeCommandBuffer())
        kernel.encode(commandBuffer: cmd, type: type, src: srcBuf, dst: dstBuf, jobs: jobs)
        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(cmd.status == .completed)

        let got = Array(UnsafeRawBufferPointer(start: dstBuf.contents(), count: expected.count))
        let mismatches = zip(got, expected).enumerated().filter { $0.element.0 != $0.element.1 }
        #expect(mismatches.isEmpty, "first mismatch at byte \(mismatches.first?.offset ?? -1)")
    }
}
