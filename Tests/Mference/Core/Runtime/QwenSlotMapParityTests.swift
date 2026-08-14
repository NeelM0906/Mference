import Testing
import Foundation
import Metal
@testable import Mference

/// The S3 slot-map fast path must be an exact transformation: identical
/// greedy tokens whether all-hit layers run GPU-side in cb1 or through the
/// CPU-planned fallback. The tiny toy cache guarantees both hit and miss
/// layers occur during the rollout.
@Suite struct QwenSlotMapParityTests {

    private func greedyRollout(slotMap: Bool) async throws -> [UInt32] {
        let dir = try QwenToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: dir) }
        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir,
                                   device: ctx.device,
                                   expecting: .qwen36Toy())
        let runner = try RealForwardRunner(model: model,
                                           context: ctx,
                                           maxContext: 64)
        runner.slotMapEnabled = slotMap
        guard let logits = ctx.device.makeBuffer(
            length: 1024 * MemoryLayout<Float16>.stride,
            options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        var tokens: [UInt32] = []
        var current: Int32 = 1
        for position in 0..<12 {
            try await runner.produce(token: current,
                                     position: position,
                                     into: logits)
            let next = runner.lastGreedyToken
            tokens.append(next)
            current = Int32(next)
        }
        return tokens
    }

    @Test func slotMapMatchesFallbackTokens() async throws {
        let fallback = try await greedyRollout(slotMap: false)
        let mapped = try await greedyRollout(slotMap: true)
        #expect(fallback.count == mapped.count)
        #expect(fallback == mapped,
                "slot-map tokens \(mapped) != fallback \(fallback)")
    }
}
