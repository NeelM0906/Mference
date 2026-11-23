import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

/// `Inkling2ForwardRunner` end to end on the Mini-width synthetic model
/// (`Inkling2RunnerToy`) against `Inkling2ReferenceModel` reading the same
/// bytes: a prefill split into chunks, a second prefill, two decode steps,
/// all 48 visits, every kernel of K2, K3 and K4 in the path. Production
/// shape (about 1.7 GB of buffers, a slow CPU reference), so full tier.
///
/// Each visit is checked alone. The reference rounds to FP16 where the
/// runner stores half (`.metalRunner`) and continues after every visit from
/// the runner's output (`forcedVisitOutputs`): the random toy amplifies the
/// small differences left (fast-math `exp` and `tanh`, accumulation order)
/// about twofold every six visits, which a free-running comparison over 48
/// visits cannot tell from a bug, while each visit's own error stays small.
/// The reference's block lists, message banks, KV caches and convolution
/// tails derive from the runner's outputs, so a wrong input, bank, slot or
/// tail on the GPU shows at the visit that uses it. With 128 experts a
/// router still meets near-ties it breaks the other way, so the reference
/// also follows the runner's routes (`forcedRoutes`) and every disagreement
/// must be a near-tie: a forced expert at most `tieMargin` below the
/// reference's own top-8 cutoff.
@Suite("Inkling2ForwardRunner on the Mini-width toy", .serialized)
struct Inkling2ForwardRunnerToyTests {

    /// Prefill, a second prefill, then one decode step per token.
    static let shortSteps: [(tokens: [Int32], decode: Bool)] = [
        ([17, 401, 3], false), ([250, 99], false), ([7], true), ([333], true),
    ]
    /// A 37-row prefill (the tiled projection path past 8 rows, the grouped
    /// experts with up to 296 routed pairs), a second prefill, a decode step.
    static let longSteps: [(tokens: [Int32], decode: Bool)] = [
        ((0..<37).map { Int32(($0 * 131 + 17) % 500) }, false), ([250, 99], false), ([7], true),
    ]
    static let traceRows = 64
    /// Largest difference of one visit's output, and of its own increment
    /// (the output minus the running sum it continued, the whole output at a
    /// block start), relative to the reference's largest magnitude; and of
    /// the logits. Measured 2026-11-23 (M1 Max, simdgroup projections): at
    /// most 2.0e-3 for a visit and for its increment (the 37-row chunk; 8.2e-4
    /// in chunks of up to 4), 3.6e-4 for the logits.
    static let tolerance: Float = 0.005
    static let incrementTolerance: Float = 0.005
    static let logitTolerance: Float = 0.002
    /// Selection-score shortfall of a forced expert (scores are a sigmoid
    /// plus a bias of at most 0.1). Measured: two near-ties in 2,592 routed
    /// token-visits, both at most 3e-5 below the cutoff.
    static let tieMargin: Float = 0.005

    private func logits(_ buffer: MTLBuffer, count: Int) -> [Float] {
        let p = buffer.contents().bindMemory(to: Float16.self, capacity: count)
        return (0..<count).map { Float(p[$0]) }
    }

    private func argmax(_ v: [Float]) -> Int { v.indices.max { v[$0] < v[$1] }! }

    /// Largest difference relative to the reference's largest magnitude.
    /// Infinite when either side holds a non-finite value: `max` skips NaN,
    /// so a NaN must not reach it.
    private func relative(_ got: [Float], _ want: [Float]) -> Float {
        guard got.allSatisfy(\.isFinite), want.allSatisfy(\.isFinite) else { return .infinity }
        return zip(got, want).map { abs($0 - $1) }.max()! / max(want.map(abs).max()!, 1e-6)
    }

    /// Chunks of 2 split the 3-token prefill into 2 + 1 (the 1-row span runs
    /// the decode kernels); chunks of 4 take it whole; chunks of 16 split the
    /// 37-row prefill into balanced chunks of 13, 12 and 12
    /// (`Inkling2ForwardRunner.prefillSpans`); chunks of 40 take it whole. Positions stay below 45: the window, the ring
    /// wrap, the log-scaled global queries and the relative-bias extent
    /// (all from 1,024 or 8,192 tokens) are K3's own tests' and G6's.
    @Test(.fullTier, arguments: [2, 4, 16, 40])
    func runnerMatchesTheReference(maxChunkTokens: Int) async throws {
        let context = try MetalContext()
        let toy = try Inkling2RunnerToy(device: context.device)
        let c = Inkling2RunnerToy.config
        let d = c.hiddenSize, topK = c.topK, rows = Self.traceRows
        let runner = try Inkling2ForwardRunner(weights: toy.runnerWeights, context: context,
                                               maxContext: 64, maxChunkTokens: maxChunkTokens)
        let reference = try Inkling2ReferenceModel(config: c, weights: toy.referenceWeights)
        reference.captureTrace = true
        reference.precision = .metalRunner
        let device = context.device
        let out = try #require(device.makeBuffer(length: c.vocabPadded * 2, options: .storageModeShared))
        let outputs = try #require(device.makeBuffer(length: 48 * rows * d * 4, options: .storageModeShared))
        let routes = try #require(device.makeBuffer(length: 48 * rows * topK * 4, options: .storageModeShared))
        runner.visitTrace = .init(outputs: outputs, routes: routes, rows: rows)
        let ids = routes.contents().bindMemory(to: UInt32.self, capacity: 48 * rows * topK)
        let schedule = c.schedule
        let steps = maxChunkTokens >= 16 ? Self.longSteps : Self.shortSteps

        var position = 0
        for (step, (chunk, decode)) in steps.enumerated() {
            let n = chunk.count
            if decode {
                try await runner.produce(token: chunk[0], position: position, into: out)
            } else {
                let result = try await runner.prefillChunked(tokens: chunk[...], startPosition: position,
                                                             outputMode: .logits,
                                                             config: .production(chunkTokens: 64),
                                                             into: out, onProgress: { _ in })
                #expect(result.newPosition == position + n)
            }
            reference.forcedRoutes = (0..<48).map { visit in
                (0..<n).map { t in (0..<topK).map { Int(ids[((visit * rows) + position + t) * topK + $0]) } }
            }
            let gpuOutputs = (0..<48).map { visit in
                (0..<n).map { t in
                    Inkling2BufferMatrix.floats(outputs, offset: (visit * rows + position + t) * d * 4, count: d)
                }
            }
            reference.forcedVisitOutputs = gpuOutputs
            let want = reference.forward(tokens: chunk, logits: .last)[0]
            let trace = try #require(reference.lastTrace)

            var visitErrors: [Float] = [], incrementErrors: [Float] = []
            var flips = 0
            var shortfall: Float = 0
            for visit in 0..<48 {
                // A block-start visit's output is its own branches alone; any
                // other adds them to the running sum it continued.
                let blockStart = schedule[visit].layer % c.attnResBlockSize == 0
                var err: Float = 0, incrementErr: Float = 0
                for t in 0..<n {
                    let got = gpuOutputs[visit][t], want = trace.visitOutputs[visit][t]
                    err = max(err, relative(got, want))
                    let input = blockStart ? [Float](repeating: 0, count: d) : gpuOutputs[visit - 1][t]
                    incrementErr = max(incrementErr, relative(zip(got, input).map { $0 - $1 },
                                                              zip(want, input).map { $0 - $1 }))
                    let s = trace.routeShortfalls[visit][t]
                    if s > 0 { flips += 1 }
                    shortfall = max(shortfall, s)
                }
                visitErrors.append(err)
                incrementErrors.append(incrementErr)
            }
            let all = logits(out, count: c.vocabPadded)
            let real = Array(all[..<c.vocabUnpadded])
            let logitError = relative(real, want)
            let worst = visitErrors.max()!, worstIncrement = incrementErrors.max()!
            print("inkling2 toy chunk \(maxChunkTokens) step \(step) (\(n) at \(position)): visit error \(worst), "
                  + "increment error \(worstIncrement), logit error \(logitError), argmax \(argmax(real)) vs "
                  + "\(argmax(want)), \(flips) near-tie route flips, largest shortfall \(shortfall)")
            if worst >= Self.tolerance || worstIncrement >= Self.incrementTolerance {
                print("inkling2 toy per-visit errors: \(visitErrors.map { String(format: "%.4f", $0) })")
                print("inkling2 toy per-visit increment errors: "
                      + "\(incrementErrors.map { String(format: "%.4f", $0) })")
            }
            #expect(worst < Self.tolerance, "step \(step)")
            #expect(worstIncrement < Self.incrementTolerance, "step \(step)")
            #expect(logitError < Self.logitTolerance, "step \(step)")
            #expect(shortfall <= Self.tieMargin, "step \(step)")
            // The runner's top logit is the reference's or within the error of it.
            let maxAbs = zip(real, want).map { abs($0 - $1) }.max()!
            #expect(want[argmax(real)] >= want.max()! - 2 * maxAbs, "step \(step)")
            // Padded rows past the vocabulary are never sampled.
            #expect(all[c.vocabUnpadded...].allSatisfy { $0 < -60_000 }, "step \(step)")
            position += n
        }
        #expect(runner.continuationPosition == position)
    }

    /// Weights that break the schedule's assumptions are refused at init,
    /// and a prefill that cannot finish is refused before its first span
    /// commits: the cursor stays put and the next prefill runs.
    @Test(.fullTier)
    func weightsAndRangesAreCheckedUpFront() async throws {
        let context = try MetalContext()
        let toy = try Inkling2RunnerToy(device: context.device)
        let c = Inkling2RunnerToy.config
        func refused(_ change: (inout Inkling2RunnerWeights) -> Void) -> Bool {
            var w = toy.runnerWeights
            change(&w)
            return (try? Inkling2ForwardRunner(weights: w, context: context, maxContext: 16, maxChunkTokens: 2)) == nil
        }
        #expect(refused { $0.layers[7].loop = nil })
        #expect(refused { $0.layers[5].isGlobal = false })
        #expect(refused { $0.layers[0].attnResQuery = $0.layers[1].attnResQuery })
        #expect(refused { $0.gateRank = Inkling2RunnerKernels.maxGateRank + 1 })
        #expect(refused { $0.vocab = $0.vocabRows + 1 })
        #expect(refused { $0.layers.removeLast() })
        // The toy's experts (about 0.9 GB) exceed what a 1 GiB host allows.
        #expect(throws: Inkling2ForwardRunnerError.self) {
            _ = try Inkling2ForwardRunner(weights: toy.runnerWeights, context: context, maxContext: 16,
                                          maxChunkTokens: 2, physicalMemory: 1 << 30)
        }

        let runner = try Inkling2ForwardRunner(weights: toy.runnerWeights, context: context,
                                               maxContext: 8, maxChunkTokens: 2)
        let out = try #require(context.device.makeBuffer(length: c.vocabPadded * 2, options: .storageModeShared))
        func prefill(_ tokens: [Int32]) async throws {
            _ = try await runner.prefillChunked(tokens: tokens[...], startPosition: 0, outputMode: .logits,
                                                config: .production(chunkTokens: 32), into: out,
                                                onProgress: { _ in })
        }
        // Past the context, and a bad token in the second span.
        await #expect(throws: Inkling2ForwardRunnerError.self) { try await prefill([1, 2, 3, 4, 5, 6, 7, 8, 9]) }
        await #expect(throws: Inkling2ForwardRunnerError.self) { try await prefill([1, 2, Int32(c.vocabUnpadded)]) }
        #expect(runner.continuationPosition == 0)
        try await prefill([1, 2, 3])
        #expect(runner.continuationPosition == 3)
    }

    /// The runner on K2's expert store against the same runner on fixed
    /// pools: the same kernels read the same bytes in the same order, so
    /// every visit output and every logit must match bit for bit. Planned as
    /// for a 32 GB host (layers 6-17 resident, the other 12 staged through two
    /// slabs) and for a 24 GB host (every layer staged, the looped ones
    /// included). Prefill visits of a staged layer read the slab ring; decode
    /// visits read the routes back and decode from the layer's slot cache,
    /// and on the looped staged layers the passes route to experts the pass
    /// before did not use (llama.cpp issue 32710 computed later passes with
    /// the first pass's experts). The toy's expert layers are written to
    /// temporary files for the store to read.
    @Test(.fullTier, arguments: [32, 24])
    func expertStoreMatchesFixedPoolsBitForBit(hostGigabytes: Int) async throws {
        let context = try MetalContext()
        let toy = try Inkling2RunnerToy(device: context.device)
        let c = Inkling2RunnerToy.config
        let d = c.hiddenSize, rows = 24
        // The short steps, then a prefill after the decodes (it restages
        // layers whose slots live decode visits used) and one more decode,
        // then a 10-token prefill in balanced chunks of 4, 3 and 3 (each chunk but the
        // last prefetches the next one's first staged read) and a decode.
        let steps = Self.shortSteps + [([88, 5, 61], false), ([41], true),
                                       ((0..<10).map { Int32(($0 * 37 + 11) % 500) }, false), ([9], true)]
        let shape = Inkling2MoEShape(d: d, f: c.expertFF, experts: c.numExperts, topK: c.topK)
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("inkling2-store-\(UUID().uuidString)")
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
        let plan = Inkling2ExpertResidencyPlan.make(layerBytes: specs.map(\.file.bytes), loopedLayers: 6..<18,
                                                    physicalMemory: UInt64(hostGigabytes) << 30)
        let stagedLayers = plan.placements.filter { $0 == .staged }.count
        #expect(stagedLayers == (hostGigabytes == 32 ? 12 : 24))
        var stored = toy.runnerWeights
        // The toy's layers are small, so the plan's `.fill` would hold every
        // expert; 24 slots a layer keep the caches evicting.
        stored.experts = try Inkling2ExpertStore(device: context.device, shape: shape, layers: specs, plan: plan,
                                                 readMode: .forHost(physicalMemory: UInt64(hostGigabytes) << 30),
                                                 decodeSlots: .perLayer(24))

        func run(_ weights: Inkling2RunnerWeights, shortPrefill: Int? = nil, splitReadbacks: Bool = false) async throws
            -> (logits: [[UInt16]], outputs: [Float], routes: [UInt32], routerLogits: [Float],
                cache: Inkling2ForwardRunner.DecodeCacheStats, short: Inkling2ForwardRunner.ShortPrefillStats) {
            let runner = try Inkling2ForwardRunner(weights: weights, context: context,
                                                   maxContext: 64, maxChunkTokens: 4)
            if let shortPrefill { runner.shortPrefillMaxTokens = shortPrefill }
            if splitReadbacks { runner.defersDecodeReadback = false }
            let device = context.device
            let out = try #require(device.makeBuffer(length: c.vocabPadded * 2, options: .storageModeShared))
            let outputs = try #require(device.makeBuffer(length: 48 * rows * d * 4, options: .storageModeShared))
            let routes = try #require(device.makeBuffer(length: 48 * rows * c.topK * 4, options: .storageModeShared))
            // The trace's router logits too: a staged decode visit records them
            // in its own buffer, after the visit (deferred, the default), or in
            // the buffer that follows the readback `split()` (split readbacks).
            let routerLogits = try #require(device.makeBuffer(length: 48 * rows * c.numExperts * 4,
                                                              options: .storageModeShared))
            runner.visitTrace = .init(outputs: outputs, routes: routes, rows: rows, routerLogits: routerLogits)
            var logits: [[UInt16]] = []
            var position = 0
            for (chunk, decode) in steps {
                if decode {
                    try await runner.produce(token: chunk[0], position: position, into: out)
                } else {
                    _ = try await runner.prefillChunked(tokens: chunk[...], startPosition: position,
                                                        outputMode: .logits, config: .production(chunkTokens: 32),
                                                        into: out, onProgress: { _ in })
                }
                let p = out.contents().bindMemory(to: UInt16.self, capacity: c.vocabPadded)
                logits.append(Array(UnsafeBufferPointer(start: p, count: c.vocabPadded)))
                position += chunk.count
            }
            let traced = routes.contents().bindMemory(to: UInt32.self, capacity: 48 * rows * c.topK)
            return (logits, Inkling2BufferMatrix.floats(outputs, offset: 0, count: 48 * rows * d),
                    Array(UnsafeBufferPointer(start: traced, count: 48 * rows * c.topK)),
                    Inkling2BufferMatrix.floats(routerLogits, offset: 0, count: 48 * rows * c.numExperts),
                    runner.decodeCacheStats, runner.shortPrefillStats)
        }
        let fixed = try await run(toy.runnerWeights)
        let staged = try await run(stored)
        // Short prefill chunks through the slots (the default, up to 64
        // tokens) against the whole-layer reads forced: the same bits.
        let swept = try await run(stored, shortPrefill: 0)
        #expect(swept.logits == fixed.logits && swept.outputs == fixed.outputs)
        #expect(swept.short.visits == 0 && staged.short.visits > 0)
        // Decode visits deferred (the GPU waits on a CPU-signalled event; the
        // default) against split readbacks: the same bits, the same caching.
        let split = try await run(stored, splitReadbacks: true)
        #expect(split.logits == fixed.logits && split.outputs == fixed.outputs)
        // K4's traced router logits: recorded after each visit in whichever
        // buffer is current, the same bits on every path.
        #expect(split.routerLogits == fixed.routerLogits && swept.routerLogits == fixed.routerLogits)
        #expect(split.cache.visits == staged.cache.visits)
        #expect(staged.logits == fixed.logits)
        #expect(staged.outputs == fixed.outputs)
        #expect(staged.routes == fixed.routes)
        #expect(staged.routerLogits == fixed.routerLogits)
        #expect(fixed.routerLogits.allSatisfy(\.isFinite) && fixed.routerLogits.contains { $0 != 0 })
        // Every decode step's staged visits went through the slot caches.
        let decodeSteps = steps.filter(\.decode).count
        /// The stored layer of visit `v`: prelude 0-5, three passes of 6-17, coda 18-23.
        func layer(visit v: Int) -> Int { v < 6 ? v : v < 42 ? 6 + (v - 6) % 12 : v - 24 }
        #expect(fixed.cache.visits == 0)
        #expect(staged.cache.visits == decodeSteps * (0..<48).filter { plan.placements[layer(visit: $0)] == .staged }.count)
        #expect(staged.cache.misses > 0)
        // On the looped layers, a later pass of a decode step routes to
        // experts the pass before it did not use (so a cache keyed by stored
        // layer has to load them).
        let decodeRows = steps.reduce(into: (position: 0, rows: [Int]())) { acc, step in
            if step.decode { acc.rows.append(acc.position) }
            acc.position += step.tokens.count
        }.rows
        func routed(_ visit: Int, _ row: Int) -> Set<UInt32> {
            Set(staged.routes[((visit * rows) + row) * c.topK ..< ((visit * rows) + row + 1) * c.topK])
        }
        var newExperts = 0
        for row in decodeRows {
            for layer in 6..<18 {
                for pass in 1..<3 {
                    let now = 6 + pass * 12 + (layer - 6), before = now - 12
                    newExperts += routed(now, row).subtracting(routed(before, row)).count
                }
            }
        }
        #expect(newExperts > 0)
        print("inkling2 toy expert store at \(hostGigabytes) GB: \(plan.placements.filter { $0 == .resident }.count) "
              + "resident, \(stagedLayers) staged layers; decode slot cache \(staged.cache.hits) hits, "
              + "\(staged.cache.misses) misses over \(staged.cache.visits) visits; experts new to a later pass on "
              + "the looped layers: \(newExperts); short prefill \(staged.short.visits) staged visits, "
              + "\(staged.short.fallbacks) read whole; bit-identical: "
              + "\(staged.logits == fixed.logits && staged.outputs == fixed.outputs)")
    }
}
