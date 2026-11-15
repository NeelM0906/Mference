import Foundation
import Metal
import Testing
@testable import Mference

/// `MFERENCE_PREFILL_BREAKDOWN=1` splits the shared prefill path's attention
/// command buffer at stage boundaries. It is a diagnostic, so it must leave the
/// logits and the following decode step bit-identical.
@Suite(.serialized) struct PrefillStageBreakdownTests {
    @Test(arguments: ["gemma", "qwen"])
    func breakdownLeavesPrefillAndDecodeUnchanged(family: String) async throws {
        let off = try await Self.run(family: family, breakdown: nil)
        let breakdown = PrefillStageBreakdown()
        let on = try await Self.run(family: family, breakdown: breakdown)
        #expect(on.prefill == off.prefill, "prefill logits are bit-identical")
        #expect(on.decode == off.decode, "the decode step after prefill is bit-identical")

        #expect(!breakdown.isEmpty)
        let recorded = breakdown.recordedStages
        // PrefillChunkPlanner.spans: 65 tokens at chunk 32 are [32, 32, 1].
        #expect(breakdown.chunkCount == 3)
        #expect(breakdown.tokenCount == 65)
        for stage: PrefillStage in [.attentionInputProjections, .attentionCore,
                                    .attentionOutputProjection, .postAttentionAndRouter,
                                    .sharedExpert, .routedTiles, .moeTail, .head] {
            #expect(recorded.contains(stage), "\(stage) recorded")
        }
        if family == "qwen" {
            for stage: PrefillStage in [.gdnInputProjections, .gdnCore, .gdnOutputProjection] {
                #expect(recorded.contains(stage), "\(stage) recorded")
            }
        } else {
            #expect(!recorded.contains(.gdnCore))
        }
        #expect(breakdown.report().contains("tokens=65"))
    }

    private static func run(family: String,
                            breakdown: PrefillStageBreakdown?) async throws
        -> (prefill: [UInt16], decode: [UInt16]) {
        let gemma = family == "gemma"
        let config: ArchConfig = gemma ? .gemma4Toy(topKExperts: 8) : .qwen36Toy()
        let directory = try (gemma
            ? ModelLoaderTests.writeToySynthetic(config: config, finiteNorms: true)
            : QwenToySynthetic.write())
        defer { try? FileManager.default.removeItem(at: directory) }
        let context = try MetalContext()
        let model = try Model.load(directoryURL: directory, device: context.device,
            expecting: config, streamingMode: .pread(slotCount: 8))
        let runner = try RealForwardRunner(model: model, context: context, maxContext: 128,
            runtimeConfiguration: RuntimeConfiguration(expertCacheSlots: 8,
                prefillChunkTokens: 32, forceLogitsHead: true))
        // Off unless MFERENCE_PREFILL_BREAKDOWN=1 is exported.
        if !RealForwardRunner.prefillBreakdownEnabled {
            #expect(runner.prefillStageBreakdown == nil)
        }
        runner.prefillStageBreakdown = breakdown
        let logits = try #require(context.device.makeBuffer(length: config.vocabSize * 2,
                                                            options: .storageModeShared))
        func row() -> [UInt16] {
            Array(UnsafeBufferPointer(start: logits.contents().assumingMemoryBound(to: UInt16.self),
                                      count: config.vocabSize))
        }
        let tokens = (0..<65).map { Int32(4 + $0 % 16) }
        _ = try await runner.prefillChunked(tokens: tokens[...], startPosition: 0,
            outputMode: .logits, config: .production(chunkTokens: 32),
            into: logits, onProgress: { _ in })
        let prefill = row()
        try runner.prepareForContinuation(expectedPosition: 65)
        try await runner.produce(token: 7, position: 65, into: logits)
        return (prefill, row())
    }
}
