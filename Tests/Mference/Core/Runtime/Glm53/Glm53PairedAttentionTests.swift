import Metal
import Testing
@testable import Mference

@Suite struct Glm53PairedAttentionTests {
    @Test func pairedCausalQueriesPreserveSingleQueryArithmetic() throws {
        let context = try MetalContext()
        let kernels = try Glm53PrefillKernels(context: context, swigluLimit: 0)
        for dim in [64, 512] {
            for (base, rows) in [(0, 1), (0, 3), (7, 33), (2040, 7)] {
                let heads = dim == 512 ? 64 : 2
                let count = rows * heads * dim
                let query = (0..<count).map { Float16(Float(($0 * 17 + 5) % 101 - 50) / 97) }
                let latent = (0..<((base + rows) * dim)).map { Float16(Float(($0 * 13 + 7) % 89 - 44) / 71) }
                let q = try #require(context.device.makeBuffer(bytes: query, length: query.count * 2,
                                                               options: .storageModeShared))
                let kv = try #require(context.device.makeBuffer(bytes: latent, length: latent.count * 2,
                                                                options: .storageModeShared))
                let reference = try #require(context.device.makeBuffer(length: (count + 1) * 2,
                                                                       options: .storageModeShared))
                let actual = try #require(context.device.makeBuffer(length: (count + 1) * 2,
                                                                    options: .storageModeShared))
                let ref = reference.contents().assumingMemoryBound(to: UInt16.self)
                let got = actual.contents().assumingMemoryBound(to: UInt16.self)
                got[count] = 0x1234
                let cb = try #require(context.queue.makeCommandBuffer())
                kernels.encodeLatentAttentionCausal(commandBuffer: cb, qLatent: q, latents: kv,
                    out: reference, heads: heads, latentDim: dim, base: base, tokens: rows,
                    scale: 0.0625, pairedQueries: false)
                kernels.encodeLatentAttentionCausal(commandBuffer: cb, qLatent: q, latents: kv,
                    out: actual, heads: heads, latentDim: dim, base: base, tokens: rows, scale: 0.0625)
                cb.commit()
                cb.waitUntilCompleted()
                try #require(cb.error == nil)
                #expect(got[count] == 0x1234)
                let expected = Array(UnsafeBufferPointer(start: ref, count: count))
                let result = Array(UnsafeBufferPointer(start: got, count: count))
                #expect(result == expected, "dim=\(dim) base=\(base) rows=\(rows)")
                #expect(result.allSatisfy { Float16(bitPattern: $0).isFinite })
            }
        }
    }
}
