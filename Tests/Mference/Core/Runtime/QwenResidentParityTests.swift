import Testing
import Foundation
import Metal
@testable import Mference

/// Resident mode must be an exact transformation: the same greedy token
/// sequence as the pread slot cache on the Qwen toy fixture, decode and
/// chunked prefill alike.
@Suite struct QwenResidentParityTests {

    private func makeRunner(dir: URL,
                            mode: ExpertStreamingMode) throws
        -> (MetalContext, RealForwardRunner) {
        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir,
                                   device: ctx.device,
                                   expecting: .qwen36Toy(),
                                   streamingMode: mode)
        let runner = try RealForwardRunner(model: model,
                                           context: ctx,
                                           maxContext: 64)
        return (ctx, runner)
    }

    private func makeLogits(_ ctx: MetalContext, vocab: Int) throws -> MTLBuffer {
        guard let buf = ctx.device.makeBuffer(
            length: vocab * MemoryLayout<Float16>.stride,
            options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        return buf
    }

    @Test func residentDecodeMatchesPreadTokens() async throws {
        let dir = try QwenToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: dir) }

        func greedyRollout(_ mode: ExpertStreamingMode) async throws -> [UInt32] {
            let (ctx, runner) = try makeRunner(dir: dir, mode: mode)
            let logits = try makeLogits(ctx, vocab: 1024)
            var tokens: [UInt32] = []
            var current: Int32 = 1
            for position in 0..<8 {
                try await runner.produce(token: current,
                                         position: position,
                                         into: logits)
                let next = runner.lastGreedyToken
                tokens.append(next)
                current = Int32(next)
            }
            return tokens
        }

        let pread = try await greedyRollout(.pread(slotCount: 8))
        let resident = try await greedyRollout(.resident)
        #expect(pread.count == resident.count)
        #expect(pread == resident,
                "resident greedy tokens \(resident) != pread \(pread)")
    }
}
