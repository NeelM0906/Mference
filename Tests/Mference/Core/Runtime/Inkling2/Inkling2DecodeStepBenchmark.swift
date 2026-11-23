import Foundation
import Metal
import Testing
@testable import Mference

/// Wall time of one decode step of the runner on the Mini-width toy
/// (`Inkling2RunnerToy`: Mini's hidden width, attention shape and
/// 128-expert router, experts in fixed resident pools, so no staging): a
/// 4-token prefill, then 40 decode steps; the median of the last 32 and
/// the GPU time of each step's command buffers summed. Compares decode-path
/// kernels in place, without the install. Model-free, but about 1.7 GB of
/// buffers and a GPU benchmark, so it runs only with
/// `MFERENCE_INKLING2_DECODE_BENCH` set, under `lock.sh build`.
@Suite("Inkling2 decode step benchmark", .serialized)
struct Inkling2DecodeStepBenchmark {
    static var enabled: Bool { ProcessInfo.processInfo.environment["MFERENCE_INKLING2_DECODE_BENCH"] != nil }

    @Test(.enabled(if: enabled, "needs MFERENCE_INKLING2_DECODE_BENCH (1.7 GB of toy buffers)"))
    func decodeStepTime() async throws {
        let context = try MetalContext()
        let toy = try Inkling2RunnerToy(device: context.device)
        let c = Inkling2RunnerToy.config
        let runner = try Inkling2ForwardRunner(weights: toy.runnerWeights, context: context,
                                               maxContext: 64, maxChunkTokens: 8)
        let out = try #require(context.device.makeBuffer(length: c.vocabPadded * 2, options: .storageModeShared))
        _ = try await runner.prefillChunked(tokens: [17, 401, 3, 250][...], startPosition: 0, outputMode: .logits,
                                            config: .production(chunkTokens: 64), into: out, onProgress: { _ in })
        var walls: [Double] = []
        var position = 4
        for step in 0..<40 {
            let token = Int32((step * 131 + 7) % c.vocabUnpadded)
            let started = Date()
            try await runner.produce(token: token, position: position, into: out)
            walls.append(Date().timeIntervalSince(started))
            position += 1
        }
        let tail = walls.suffix(32).sorted()
        print(String(format: "inkling2 decode bench (toy, resident experts): median %.3f ms, min %.3f ms, max %.3f ms per step",
                     tail[tail.count / 2] * 1e3, tail.first! * 1e3, tail.last! * 1e3))
    }
}
