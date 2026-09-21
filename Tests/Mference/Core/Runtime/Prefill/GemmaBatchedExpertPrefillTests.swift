import Foundation
import Metal
import Testing
@testable import Mference

private let groupedGEMMAvailable: Bool = {
    guard let context = try? MetalContext() else { return false }
    return MPPGroupedRoutedMoE(context: context).isAvailable
}()

/// Runner-level wiring of grouped-GEMM routed experts: tile parameters, the
/// borrowed activation scratch and argument-buffer lifetime. Kernel arithmetic
/// is covered by the kernel suites; here both schedules must agree on a toy
/// model up to the matmul reordering.
@Suite(.serialized) struct GemmaBatchedExpertPrefillTests {
    struct Harness {
        let directory: URL
        let runner: RealForwardRunner
        let logits: MTLBuffer
        let config = ArchConfig.gemma4Toy(topKExperts: 8)

        init(environment: [String: String]) throws {
            directory = try ModelLoaderTests.writeToySynthetic(config: config, finiteNorms: true)
            let context = try MetalContext()
            let model = try Model.load(directoryURL: directory, device: context.device,
                expecting: config, streamingMode: .pread(slotCount: 8))
            runner = try RealForwardRunner(model: model, context: context, maxContext: 512,
                runtimeConfiguration: RuntimeConfiguration(expertCacheSlots: 8,
                    prefillChunkTokens: 64, forceLogitsHead: true),
                gemmaPrefillPolicy: GemmaPrefillPolicy(modelID: model.modelID, environment: environment))
            logits = try #require(context.device.makeBuffer(length: config.vocabSize * 2,
                options: .storageModeShared))
        }

        func prefillRow(_ tokens: ArraySlice<Int32>) async throws -> [Float] {
            _ = try await runner.prefillChunked(tokens: tokens, startPosition: 0,
                outputMode: .logits, config: .production(chunkTokens: 64), into: logits, onProgress: { _ in })
            return UnsafeBufferPointer(start: logits.contents().assumingMemoryBound(to: Float16.self),
                                       count: config.vocabSize).map(Float.init)
        }
    }

    static let tokens: [Int32] = (0..<192).map { Int32(4 + ($0 * 37 + 11) % 1000) }

    @Test(.enabled(if: groupedGEMMAvailable, "Requires runtime MPP TensorOps support"))
    func groupedExpertsReproducePerRowExpertLogits() async throws {
        let legacy = try Harness(environment: ["MFERENCE_GEMMA_PREFILL_LEGACY": "1"])
        let batched = try Harness(environment: [:])
        defer {
            try? FileManager.default.removeItem(at: legacy.directory)
            try? FileManager.default.removeItem(at: batched.directory)
        }
        let expected = try await legacy.prefillRow(Self.tokens[...])
        let actual = try await batched.prefillRow(Self.tokens[...])

        // Every token routes to all eight experts: 64 rows per expert per
        // chunk, which fills the matrix tiles exactly.
        #expect(legacy.runner.prefillGroupedExpertTiles == 0)
        #expect(batched.runner.prefillGroupedExpertTiles > 0)
        // The toy's shared expert is 8-bit, which has no batched INT4 form;
        // the runner must report the path it actually took.
        #expect(batched.runner.lastPrefillSharedExpertPath == .repeatedRows)
        let finite = actual.allSatisfy { $0.isFinite }
        #expect(finite)
        let worst = zip(expected, actual).reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
        let scale = expected.reduce(Float(0)) { max($0, abs($1)) }
        print("grouped-expert toy prefill: worst |dlogit|=\(worst), max |logit|=\(scale)")
        #expect(worst <= 0.01 * max(scale, 1), "worst=\(worst), scale=\(scale)")
        #expect(expected.indices.max { expected[$0] < expected[$1] }
            == actual.indices.max { actual[$0] < actual[$1] })
    }
}
