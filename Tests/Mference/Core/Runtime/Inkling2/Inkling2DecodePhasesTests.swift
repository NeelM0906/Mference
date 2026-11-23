import Foundation
import Metal
import Testing
@testable import Mference

/// The runner's `MFERENCE_PHASES=1` counters on the toy with K2's staged
/// store (32 GB host: 12 staged layers), split and deferred readbacks: the
/// same logits with the counters on and off, nothing recorded when off, and
/// counts that agree with the decode-cache statistics when on.
@Suite("Inkling 2 decode phase counters", .serialized)
struct Inkling2DecodePhasesTests {
    @Test(.fullTier)
    func countersChangeNothingAndAgreeWithTheDecodeCache() async throws {
        let context = try MetalContext()
        let toy = try Inkling2RunnerToy(device: context.device)
        let c = Inkling2RunnerToy.config
        let shape = Inkling2MoEShape(d: c.hiddenSize, f: c.expertFF, experts: c.numExperts, topK: c.topK)
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("inkling2-phases-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        var specs: [Inkling2ExpertLayerSpec] = []
        for (layer, info) in toy.expertLayers.enumerated() {
            let bytes = c.numExperts * info.stride
            let url = directory.appendingPathComponent("layer_\(layer).bin")
            try Data(bytesNoCopy: toy.experts[layer].contents(), count: bytes, deallocator: .none).write(to: url)
            let nvfp4 = info.encoding == .nvfp4
            specs.append(Inkling2ExpertLayerSpec(
                file: Inkling2ExpertLayerFile(path: url.path, offset: 0, bytes: bytes),
                encoding: info.encoding, expertStride: info.stride,
                gateUpScales: nvfp4 ? (toy.expertScales, info.scaleOffset) : nil,
                downScales: nvfp4 ? (toy.expertScales, info.scaleOffset + c.numExperts * 4) : nil))
        }
        let host = UInt64(32) << 30
        let plan = Inkling2ExpertResidencyPlan.make(layerBytes: specs.map(\.file.bytes), loopedLayers: 6..<18,
                                                    physicalMemory: host)
        var weights = toy.runnerWeights
        // 24 slots a layer keep the caches evicting, as K2's store test does.
        weights.experts = try Inkling2ExpertStore(device: context.device, shape: shape, layers: specs, plan: plan,
                                                  readMode: .forHost(physicalMemory: host),
                                                  decodeSlots: .perLayer(24))
        let prompt: [Int32] = [88, 5, 61, 140, 7, 300]
        let decoded: [Int32] = [41, 9, 222, 17, 63, 405]

        func run(phases: Bool, deferred: Bool) async throws
            -> (logits: [[UInt16]], phases: Inkling2DecodePhaseStats, cache: Inkling2ForwardRunner.DecodeCacheStats) {
            let runner = try Inkling2ForwardRunner(weights: weights, context: context, maxContext: 64, maxChunkTokens: 4)
            // Read once at load from the environment; the suite runs without it.
            #expect(runner.tracesPhases == (ProcessInfo.processInfo.environment["MFERENCE_PHASES"] == "1"))
            runner.tracesPhases = phases
            runner.defersDecodeReadback = deferred
            let out = try #require(context.device.makeBuffer(length: c.vocabPadded * 2, options: .storageModeShared))
            func logits() -> [UInt16] {
                Array(UnsafeBufferPointer(start: out.contents().bindMemory(to: UInt16.self, capacity: c.vocabPadded),
                                          count: c.vocabPadded))
            }
            var all: [[UInt16]] = []
            _ = try await runner.prefillChunked(tokens: prompt[...], startPosition: 0, outputMode: .logits,
                                                config: .production(chunkTokens: 32), into: out, onProgress: { _ in })
            all.append(logits())
            for (i, token) in decoded.enumerated() {
                try await runner.produce(token: token, position: prompt.count + i, into: out)
                all.append(logits())
            }
            return (all, runner.decodePhaseStats, runner.decodeCacheStats)
        }

        for deferred in [false, true] {
            let off = try await run(phases: false, deferred: deferred)
            let on = try await run(phases: true, deferred: deferred)
            #expect(on.logits == off.logits, "the counters changed the output (deferred \(deferred))")
            #expect(off.phases == Inkling2DecodePhaseStats(), "nothing is recorded without the variable")

            let p = on.phases
            #expect(p.decodeSteps == decoded.count)
            #expect(on.cache.visits > 0, "the toy should decode through staged layers")
            #expect(p.stagedVisits == on.cache.visits && p.deferredVisits == on.cache.deferred)
            #expect(p.hits == on.cache.hits && p.misses == on.cache.misses)
            #expect(abs(p.slotLoadSeconds - on.cache.seconds) < 1e-9)
            #expect(abs(p.readbackSeconds - on.cache.readbackSeconds) < 1e-9)
            #expect(deferred ? p.deferredVisits == p.stagedVisits && p.readbackSeconds == 0 : p.deferredVisits == 0)
            #expect(p.wallSeconds > 0 && p.blockedSeconds <= p.wallSeconds && p.gpuBusySeconds > 0)
        }
    }
}
