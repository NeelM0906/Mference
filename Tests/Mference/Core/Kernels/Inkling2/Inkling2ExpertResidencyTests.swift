import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

/// The residency ladder and the staged expert layers: staged visits through
/// a two-slab ring, in one command buffer with more visits than slabs, give
/// the same bytes as the resident layers; a failed read still lets the GPU
/// finish and reports the failure.
@Suite(.serialized) struct Inkling2ExpertResidencyTests {
    /// Layer files of the Mini install at K1's proposed 16 KiB stride.
    static let miniLayerBytes = (0..<24).map { $0 < 20 ? 128 * 4_489_216 : 128 * 8_470_528 }

    @Test func thirtyTwoGigabytesHoldsTheLoopedLayers() {
        let plan = Inkling2ExpertResidencyPlan.make(layerBytes: Self.miniLayerBytes, loopedLayers: 6..<18,
                                                    physicalMemory: 32 << 30)
        #expect(plan.placements.enumerated().allSatisfy { ($0.offset >= 6 && $0.offset < 18) == ($0.element == .resident) })
        #expect(plan.residentBytes == 12 * 128 * 4_489_216)          // 6.90 GB
        #expect(plan.stagingSlabs == 2)
        #expect(plan.stagingBytes == 2 * 128 * 8_470_528)            // 2.17 GB, sized for a Q8_0 layer
    }

    @Test func largeHostsHoldEverythingAndSmallHostsStageEverything() {
        let large = Inkling2ExpertResidencyPlan.make(layerBytes: Self.miniLayerBytes, loopedLayers: 6..<18,
                                                     physicalMemory: 64 << 30)
        #expect(large.placements.allSatisfy { $0 == .resident })
        #expect(large.stagingSlabs == 0 && large.stagingBytes == 0)
        let small = Inkling2ExpertResidencyPlan.make(layerBytes: Self.miniLayerBytes, loopedLayers: 6..<18,
                                                     physicalMemory: 24 << 30)
        #expect(small.placements.allSatisfy { $0 == .staged })
        #expect(small.residentBytes == 0)
    }

    /// Coordinator ruling (2026-11-23): staged reads bypass the file cache by
    /// default on hosts below 48 GB (the hosts that stage); cached stays an
    /// option, and the store hands its mode to the staging ring.
    @Test func stagedReadsDefaultToUncachedBelowFortyEightGigabytes() {
        #expect(Inkling2StagedReadMode.forHost(physicalMemory: 32 << 30) == .uncached)
        #expect(Inkling2StagedReadMode.forHost(physicalMemory: 36 << 30) == .uncached)
        #expect(Inkling2StagedReadMode.forHost(physicalMemory: 47 << 30) == .uncached)
        #expect(Inkling2StagedReadMode.forHost(physicalMemory: (48 << 30) - 1) == .uncached)
        #expect(Inkling2StagedReadMode.forHost(physicalMemory: 48 << 30) == .cached)
        #expect(Inkling2StagedReadMode.forHost(physicalMemory: 64 << 30) == .cached)
        #expect(Inkling2StagedReadMode.forHost()
                == (ProcessInfo.processInfo.physicalMemory < 48 << 30 ? .uncached : .cached))
    }

    private static func writeLayers(_ pools: [Inkling2SyntheticPool], header: Int) throws -> (URL, [Int: Inkling2ExpertLayerFile]) {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("inkling2-staging-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var files: [Int: Inkling2ExpertLayerFile] = [:]
        for (layer, pool) in pools.enumerated() {
            let bytes = pool.pool.expertStride * pool.shape.experts
            var data = Data(count: header)
            data.append(Data(bytes: pool.pool.buffer.contents(), count: bytes))
            let url = directory.appendingPathComponent(String(format: "layer_%02d.bin", layer))
            try data.write(to: url)
            files[layer] = Inkling2ExpertLayerFile(path: url.path, offset: UInt64(header), bytes: bytes)
        }
        return (directory, files)
    }

    @Test(arguments: [Inkling2StagedReadMode.uncached, .cached])
    func stagedVisitsMatchResidentLayers(mode: Inkling2StagedReadMode) throws {
        let context = try MetalContext()
        let kernels = try Inkling2MoEKernels(context: context)
        let shape = Inkling2MoEShape(d: 256, f: 128, experts: 8, topK: 4)
        let layers = try (0..<3).map {
            try Inkling2SyntheticPool.make(device: context.device, encoding: .nvfp4, shape: shape,
                                           seed: 0x2B00 + UInt64($0), quantized: true)
        }
        let (directory, files) = try Self.writeLayers(layers, header: 4096)
        defer { try? FileManager.default.removeItem(at: directory) }
        let staged = try Inkling2StagedExpertLayers(device: context.device, files: files, slabs: 2, readMode: mode)
        #expect(staged.readMode == mode)

        let tokens = 48
        let (hidden, _) = Inkling2MoETestSupport.halfBuffer(context.device, rows: tokens, stride: shape.d, d: shape.d, seed: 0x2B10)
        let routes = Inkling2MoETestSupport.randomRoutes(tokens: tokens, experts: shape.experts, topK: shape.topK, seed: 0x2B11)
        let scratch = try Inkling2MoEPrefillScratch(device: context.device, shape: shape, maxTokens: tokens)
        Inkling2MoETestSupport.write(routes.ids, to: scratch.ids)
        Inkling2MoETestSupport.write(routes.weights, to: scratch.routeWeights)
        let bytes = tokens * shape.d * 4
        func output() -> MTLBuffer { context.device.makeBuffer(length: bytes, options: .storageModeShared)! }

        // Resident references, one per layer.
        let expected: [[Float]] = layers.map { layer in
            let out = output()
            _ = Inkling2MoETestSupport.run(context) {
                kernels.encodePrefill($0, hidden: hidden, hiddenStride: shape.d, pool: layer.pool, scratch: scratch,
                                      tokens: tokens, out: out, outStride: shape.d)
            }
            return Inkling2MoETestSupport.read(out, as: Float.self, count: tokens * shape.d)
        }

        // Seven staged visits over two slabs, one command buffer.
        let visits = [0, 1, 2, 0, 2, 1, 0]
        let outs = visits.map { _ in output() }
        let cb = context.queue.makeCommandBuffer()!
        for (i, layer) in visits.enumerated() {
            let ticket = try staged.stage(layer: layer)
            staged.encodeWait(ticket, commandBuffer: cb)
            let pool = try layers[layer].pool.rebound(to: ticket.slab)
            let encoder = cb.makeComputeCommandEncoder()!
            kernels.encodePrefill(encoder, hidden: hidden, hiddenStride: shape.d, pool: pool, scratch: scratch,
                                  tokens: tokens, out: outs[i], outStride: shape.d)
            encoder.endEncoding()
            staged.encodeRelease(ticket, commandBuffer: cb)
        }
        cb.commit()
        cb.waitUntilCompleted()
        staged.drain()
        #expect(cb.status == .completed)
        #expect(staged.error == nil)
        for (i, layer) in visits.enumerated() {
            #expect(Inkling2MoETestSupport.read(outs[i], as: Float.self, count: tokens * shape.d) == expected[layer],
                    "visit \(i) (layer \(layer))")
        }
        #expect(staged.bytesRead == visits.reduce(0) { $0 + files[$1]!.bytes })
    }

    @Test func failedReadSignalsAndReports() throws {
        let context = try MetalContext()
        let missing = Inkling2ExpertLayerFile(path: "/nonexistent/inkling2/layer_00.bin", offset: 0, bytes: 1 << 16)
        let staged = try Inkling2StagedExpertLayers(device: context.device, files: [0: missing], slabs: 2)
        let cb = context.queue.makeCommandBuffer()!
        let ticket = try staged.stage(layer: 0)
        staged.encodeWait(ticket, commandBuffer: cb)
        staged.encodeRelease(ticket, commandBuffer: cb)
        cb.commit()
        cb.waitUntilCompleted()
        staged.drain()
        #expect(cb.status == .completed)
        #expect(staged.error == .openFailed(path: missing.path, errno: ENOENT))
    }

    @Test(arguments: [Inkling2StagedReadMode.uncached, .cached])
    func storeServesResidentAndStagedLayers(mode: Inkling2StagedReadMode) throws {
        let context = try MetalContext()
        let kernels = try Inkling2MoEKernels(context: context)
        let shape = Inkling2MoEShape(d: 256, f: 128, experts: 8, topK: 4)
        let layers = try (0..<3).map {
            try Inkling2SyntheticPool.make(device: context.device, encoding: .nvfp4, shape: shape,
                                           seed: 0x2B20 + UInt64($0), quantized: true)
        }
        // A header off the page boundary: staged reads start mid-page.
        let (directory, files) = try Self.writeLayers(layers, header: 4100)
        defer { try? FileManager.default.removeItem(at: directory) }
        let specs = layers.enumerated().map { layer, pool in
            Inkling2ExpertLayerSpec(file: files[layer]!, encoding: .nvfp4, expertStride: pool.pool.expertStride,
                                    gateUpScales: (pool.pool.gateUpScales, pool.pool.gateUpScalesOffset),
                                    downScales: (pool.pool.downScales, pool.pool.downScalesOffset))
        }
        let plan = Inkling2ExpertResidencyPlan(placements: [.resident, .staged, .staged], stagingSlabs: 2,
                                               residentBytes: files[0]!.bytes, stagingBytes: 0)
        let store = try Inkling2ExpertStore(device: context.device, shape: shape, layers: specs, plan: plan, readMode: mode)
        #expect(store.readMode == mode && store.staging?.readMode == mode)
        let byDefault = try Inkling2ExpertStore(device: context.device, shape: shape, layers: specs, plan: plan)
        #expect(byDefault.readMode == .forHost() && byDefault.staging?.readMode == .forHost())

        let tokens = 40
        let (hidden, _) = Inkling2MoETestSupport.halfBuffer(context.device, rows: tokens, stride: shape.d, d: shape.d, seed: 0x2B30)
        let routes = Inkling2MoETestSupport.randomRoutes(tokens: tokens, experts: shape.experts, topK: shape.topK, seed: 0x2B31)
        let scratch = try Inkling2MoEPrefillScratch(device: context.device, shape: shape, maxTokens: tokens)
        Inkling2MoETestSupport.write(routes.ids, to: scratch.ids)
        Inkling2MoETestSupport.write(routes.weights, to: scratch.routeWeights)
        let count = tokens * shape.d
        func output() -> MTLBuffer { context.device.makeBuffer(length: count * 4, options: .storageModeShared)! }
        let expected: [[Float]] = layers.map { layer in
            let out = output()
            _ = Inkling2MoETestSupport.run(context) {
                kernels.encodePrefill($0, hidden: hidden, hiddenStride: shape.d, pool: layer.pool, scratch: scratch,
                                      tokens: tokens, out: out, outStride: shape.d)
            }
            return Inkling2MoETestSupport.read(out, as: Float.self, count: count)
        }

        let visits = [0, 1, 2, 1, 0, 2]
        let outs = visits.map { _ in output() }
        let cb = context.queue.makeCommandBuffer()!
        for (i, layer) in visits.enumerated() {
            let pool = try store.beginPrefillVisit(layer: layer, commandBuffer: cb)
            let encoder = cb.makeComputeCommandEncoder()!
            kernels.encodePrefill(encoder, hidden: hidden, hiddenStride: shape.d, pool: pool, scratch: scratch,
                                  tokens: tokens, out: outs[i], outStride: shape.d)
            encoder.endEncoding()
            try store.endPrefillVisit(layer: layer, commandBuffer: cb)
        }
        cb.commit()
        cb.waitUntilCompleted()
        store.drainStaging()
        #expect(cb.status == .completed && store.stagingError == nil)
        for (i, layer) in visits.enumerated() {
            #expect(Inkling2MoETestSupport.read(outs[i], as: Float.self, count: count) == expected[layer], "visit \(i)")
        }
        #expect(store.stagingBytesRead == 4 * files[1]!.bytes)

        // Decode: the resident pool, and staged layers through their caches,
        // give the same bytes as the originals.
        let activation = context.device.makeBuffer(length: shape.topK * shape.f * 2, options: .storageModeShared)!
        let routed = Inkling2MoETestSupport.read(scratch.ids, as: UInt32.self, count: shape.topK).map(Int.init)
        for layer in 0..<3 {
            let a = output(), b = output()
            _ = Inkling2MoETestSupport.run(context) {
                kernels.encodeDecode($0, shape: shape, x: hidden, ids: scratch.ids, routeWeights: scratch.routeWeights,
                                     pool: layers[layer].pool, activation: activation, out: a)
            }
            if let pool = store.residentPool(layer: layer) {
                #expect(!store.isStaged(layer: layer))
                _ = Inkling2MoETestSupport.run(context) {
                    kernels.encodeDecode($0, shape: shape, x: hidden, ids: scratch.ids, routeWeights: scratch.routeWeights,
                                         pool: pool, activation: activation, out: b)
                }
            } else {
                let binding = try store.prepareDecodeVisit(layer: layer, expertIDs: routed)
                #expect(binding.misses == shape.topK && binding.hits == 0)
                let cb = context.queue.makeCommandBuffer()!
                let encoder = cb.makeComputeCommandEncoder()!
                kernels.encodeDecode(encoder, shape: shape, x: hidden, binding: binding, routeWeights: scratch.routeWeights,
                                     activation: activation, out: b)
                encoder.endEncoding()
                store.endDecodeVisit(binding, commandBuffer: cb)
                cb.commit()
                cb.waitUntilCompleted()
                let again = try store.prepareDecodeVisit(layer: layer, expertIDs: routed)
                #expect(again.hits == shape.topK && again.misses == 0)
            }
            #expect(Inkling2MoETestSupport.read(a, as: Float.self, count: shape.d)
                    == Inkling2MoETestSupport.read(b, as: Float.self, count: shape.d), "layer \(layer)")
        }
        #expect(throws: Inkling2ExpertStore.StoreError.visitNotStarted(layer: 1)) {
            try store.endPrefillVisit(layer: 1, commandBuffer: context.queue.makeCommandBuffer()!)
        }

        // A chunk that fails while encoding: its command buffer is dropped,
        // the open visits are abandoned, and the next chunk runs normally.
        let dropped = context.queue.makeCommandBuffer()!
        _ = try store.beginPrefillVisit(layer: 1, commandBuffer: dropped)
        _ = try store.beginPrefillVisit(layer: 2, commandBuffer: dropped)
        try store.abandonOpenVisits()
        let again = context.queue.makeCommandBuffer()!
        let againOut = output()
        let pool = try store.beginPrefillVisit(layer: 2, commandBuffer: again)
        let encoder = again.makeComputeCommandEncoder()!
        kernels.encodePrefill(encoder, hidden: hidden, hiddenStride: shape.d, pool: pool, scratch: scratch,
                              tokens: tokens, out: againOut, outStride: shape.d)
        encoder.endEncoding()
        try store.endPrefillVisit(layer: 2, commandBuffer: again)
        again.commit()
        again.waitUntilCompleted()
        store.drainStaging()
        #expect(again.status == .completed && store.stagingError == nil)
        #expect(Inkling2MoETestSupport.read(againOut, as: Float.self, count: count) == expected[2])
    }

    /// Three staged NVFP4 layers (with per-expert scales) share two slabs:
    /// layers 1 and 2 split slab 0, layer 3 has slab 1; each cache holds half
    /// a layer. Per token every staged layer is visited in turn, so slots
    /// evict and scales refresh; every visit equals the resident pool's
    /// output bit for bit.
    private static func cacheStore(_ context: MetalContext, shape: Inkling2MoEShape, encoding: Inkling2MoEEncoding = .nvfp4,
                                   readMode: Inkling2StagedReadMode = .forHost())
        throws -> (store: Inkling2ExpertStore, layers: [Inkling2SyntheticPool], directory: URL, files: [Int: Inkling2ExpertLayerFile]) {
        let layers = try (0..<4).map {
            try Inkling2SyntheticPool.make(device: context.device, encoding: encoding, shape: shape,
                                           seed: 0x2B40 + UInt64($0), quantized: true)
        }
        let (directory, files) = try writeLayers(layers, header: 4096)
        let specs = layers.enumerated().map { layer, pool in
            Inkling2ExpertLayerSpec(file: files[layer]!, encoding: encoding, expertStride: pool.pool.expertStride,
                                    gateUpScales: encoding == .nvfp4 ? (pool.pool.gateUpScales, pool.pool.gateUpScalesOffset) : nil,
                                    downScales: encoding == .nvfp4 ? (pool.pool.downScales, pool.pool.downScalesOffset) : nil)
        }
        let plan = Inkling2ExpertResidencyPlan(placements: [.resident, .staged, .staged, .staged], stagingSlabs: 2,
                                               residentBytes: files[0]!.bytes, stagingBytes: 0)
        let store = try Inkling2ExpertStore(device: context.device, shape: shape, layers: specs, plan: plan,
                                            readMode: readMode)
        #expect(store.readMode == readMode && store.staging?.readMode == readMode)
        return (store, layers, directory, files)
    }

    private static func decode(_ context: MetalContext, _ kernels: Inkling2MoEKernels, shape: Inkling2MoEShape,
                               x: MTLBuffer, xOffset: Int, weights: MTLBuffer, weightsOffset: Int,
                               pool: Inkling2ExpertPool? = nil, ids: MTLBuffer? = nil, idsOffset: Int = 0,
                               binding: Inkling2DecodeBinding? = nil, store: Inkling2ExpertStore? = nil) -> [Float] {
        let activation = context.device.makeBuffer(length: shape.topK * shape.f * 2, options: .storageModeShared)!
        let out = context.device.makeBuffer(length: shape.d * 4, options: .storageModeShared)!
        let cb = context.queue.makeCommandBuffer()!
        let encoder = cb.makeComputeCommandEncoder()!
        if let binding {
            kernels.encodeDecode(encoder, shape: shape, x: x, xOffset: xOffset, binding: binding, routeWeights: weights,
                                 routeWeightsOffset: weightsOffset, activation: activation, out: out)
        } else {
            kernels.encodeDecode(encoder, shape: shape, x: x, xOffset: xOffset, ids: ids!, idsOffset: idsOffset,
                                 routeWeights: weights, routeWeightsOffset: weightsOffset, pool: pool!,
                                 activation: activation, out: out)
        }
        encoder.endEncoding()
        if let binding, let store { store.endDecodeVisit(binding, commandBuffer: cb) }
        cb.commit()
        cb.waitUntilCompleted()
        return Inkling2MoETestSupport.read(out, as: Float.self, count: shape.d)
    }

    @Test(arguments: [Inkling2StagedReadMode.uncached, .cached])
    func decodeCachesMatchResidentPoolsAcrossInterleavedLayers(mode: Inkling2StagedReadMode) throws {
        let context = try MetalContext()
        let kernels = try Inkling2MoEKernels(context: context)
        let shape = Inkling2MoEShape(d: 256, f: 128, experts: 16, topK: 4)
        let (store, layers, directory, _) = try Self.cacheStore(context, shape: shape, readMode: mode)
        defer { try? FileManager.default.removeItem(at: directory) }
        for layer in 1...3 { #expect(store.decodeSlots(layer: layer) == 8 && store.isStaged(layer: layer)) }
        #expect(!store.isStaged(layer: 0) && store.residentPool(layer: 0) != nil && store.residentPool(layer: 1) == nil)

        let tokens = 10
        let (x, _) = Inkling2MoETestSupport.halfBuffer(context.device, rows: tokens, stride: shape.d, d: shape.d, seed: 0x2B41)
        let routes = Inkling2MoETestSupport.randomRoutes(tokens: tokens * 3, experts: shape.experts, topK: shape.topK, seed: 0x2B42)
        let weights = context.device.makeBuffer(length: routes.weights.count * 4, options: .storageModeShared)!
        let ids = context.device.makeBuffer(length: routes.ids.count * 4, options: .storageModeShared)!
        Inkling2MoETestSupport.write(routes.weights, to: weights)
        Inkling2MoETestSupport.write(routes.ids, to: ids)
        var hits = 0, misses = 0
        for t in 0..<tokens {
            for layer in 1...3 {
                let r = t * 3 + (layer - 1)
                let routed = (0..<shape.topK).map { Int(routes.ids[r * shape.topK + $0]) }
                let binding = try store.prepareDecodeVisit(layer: layer, expertIDs: routed)
                #expect(binding.hits + binding.misses == shape.topK)
                hits += binding.hits; misses += binding.misses
                let want = Self.decode(context, kernels, shape: shape, x: x, xOffset: t * shape.d * 2, weights: weights,
                                       weightsOffset: r * shape.topK * 4, pool: layers[layer].pool, ids: ids,
                                       idsOffset: r * shape.topK * 4)
                let got = Self.decode(context, kernels, shape: shape, x: x, xOffset: t * shape.d * 2, weights: weights,
                                      weightsOffset: r * shape.topK * 4, binding: binding, store: store)
                #expect(got == want, "token \(t) layer \(layer)")
            }
        }
        #expect(misses > 3 * 8, "caches of 8 slots must evict over \(tokens) tokens")
        #expect(hits > 0)
    }

    /// Least recently used, not first in: after A and B fill the 8 slots, C
    /// reuses two of A's experts, so the next misses evict A's other two.
    @Test func decodeCacheEvictsLeastRecentlyUsed() throws {
        let context = try MetalContext()
        let shape = Inkling2MoEShape(d: 256, f: 128, experts: 16, topK: 4)
        let (store, _, directory, _) = try Self.cacheStore(context, shape: shape, encoding: .q8_0)
        defer { try? FileManager.default.removeItem(at: directory) }
        func visit(_ experts: [Int]) throws -> (Int, Int) {
            let b = try store.prepareDecodeVisit(layer: 1, expertIDs: experts)
            return (b.hits, b.misses)
        }
        #expect(try visit([0, 1, 2, 3]) == (0, 4))
        #expect(try visit([4, 5, 6, 7]) == (0, 4))
        #expect(try visit([0, 1, 8, 9]) == (2, 2))    // evicts 2 and 3 (least recent), not 0 and 1 (first in)
        #expect(try visit([2, 3, 4, 5]) == (2, 2))    // 4, 5 hit; 2, 3 were evicted
    }

    /// A staged prefill visit overwrites the slabs: experts cached before it
    /// miss after it.
    @Test func prefillEmptiesTheDecodeCaches() throws {
        let context = try MetalContext()
        let shape = Inkling2MoEShape(d: 256, f: 128, experts: 16, topK: 4)
        let (store, _, directory, _) = try Self.cacheStore(context, shape: shape, encoding: .q8_0)
        defer { try? FileManager.default.removeItem(at: directory) }
        let experts = [3, 7, 11, 15]
        _ = try store.prepareDecodeVisit(layer: 2, expertIDs: experts)
        #expect(try store.prepareDecodeVisit(layer: 2, expertIDs: experts).hits == 4)
        let cb = context.queue.makeCommandBuffer()!
        _ = try store.beginPrefillVisit(layer: 2, commandBuffer: cb)
        try store.endPrefillVisit(layer: 2, commandBuffer: cb)
        cb.commit()
        cb.waitUntilCompleted()
        let after = try store.prepareDecodeVisit(layer: 2, expertIDs: experts)
        #expect(after.hits == 0 && after.misses == 4)
        #expect(throws: Inkling2ExpertStore.StoreError.notStaged(layer: 0)) {
            try store.prepareDecodeVisit(layer: 0, expertIDs: experts)
        }
    }

    /// Looped layers on the staged path (llama.cpp issue 32710: with a looped
    /// layer's experts not resident, passes 2 and 3 computed with the experts
    /// loaded for pass 1). Layers 1 and 2 stand for a looped block visited
    /// three times between layer 0 (resident) and layer 3, as in the all-staged
    /// plan of a host under 30 GB; each pass of a looped layer routes only to
    /// experts its earlier passes did not use (pass 1 to 0-4, pass 2 to 5-9,
    /// pass 3 to 10-15). Prefill through the staging ring and decode through
    /// the slot caches (keyed by stored layer) both equal the fixed pools bit
    /// for bit, and on the first decode token every pass-2 and pass-3 visit
    /// misses all four experts, so nothing pass 1 loaded stands in for them.
    @Test func loopedPassesOnStagedLayersUseTheirOwnExperts() throws {
        let context = try MetalContext()
        let kernels = try Inkling2MoEKernels(context: context)
        let shape = Inkling2MoEShape(d: 256, f: 128, experts: 16, topK: 4)
        let (store, layers, directory, _) = try Self.cacheStore(context, shape: shape)
        defer { try? FileManager.default.removeItem(at: directory) }
        // (stored layer, pass); layers 1 and 2 are the looped block.
        let schedule = [(0, 1), (1, 1), (2, 1), (1, 2), (2, 2), (1, 3), (2, 3), (3, 1)]
        let passExperts = [Array(0..<5), Array(5..<10), Array(10..<16)]
        func experts(layer: Int, pass: Int, token: Int) -> [Int] {
            let set = (1...2).contains(layer) ? passExperts[pass - 1] : Array(0..<shape.experts)
            return (0..<shape.topK).map { set[(token + layer + $0) % set.count] }
        }

        // Prefill: one command buffer for the whole schedule, a route scratch
        // per visit.
        let tokens = 24
        let (hidden, _) = Inkling2MoETestSupport.halfBuffer(context.device, rows: tokens, stride: shape.d, d: shape.d, seed: 0x2B60)
        let count = tokens * shape.d
        func output() -> MTLBuffer { context.device.makeBuffer(length: count * 4, options: .storageModeShared)! }
        let scratches = try schedule.enumerated().map { visit, step in
            let scratch = try Inkling2MoEPrefillScratch(device: context.device, shape: shape, maxTokens: tokens)
            let ids = (0..<tokens).flatMap { t in experts(layer: step.0, pass: step.1, token: t).map(UInt32.init) }
            let weights = Inkling2MoETestSupport.randomRoutes(tokens: tokens, experts: shape.experts, topK: shape.topK,
                                                              seed: 0x2B61 + UInt64(visit)).weights
            Inkling2MoETestSupport.write(ids, to: scratch.ids)
            Inkling2MoETestSupport.write(weights, to: scratch.routeWeights)
            return scratch
        }
        let expected: [[Float]] = schedule.enumerated().map { visit, step in
            let out = output()
            _ = Inkling2MoETestSupport.run(context) {
                kernels.encodePrefill($0, hidden: hidden, hiddenStride: shape.d, pool: layers[step.0].pool,
                                      scratch: scratches[visit], tokens: tokens, out: out, outStride: shape.d)
            }
            return Inkling2MoETestSupport.read(out, as: Float.self, count: count)
        }
        #expect(expected[1] != expected[3] && expected[3] != expected[5], "the passes must differ")
        let outs = schedule.map { _ in output() }
        let cb = context.queue.makeCommandBuffer()!
        for (visit, step) in schedule.enumerated() {
            let pool = try store.beginPrefillVisit(layer: step.0, commandBuffer: cb)
            let encoder = cb.makeComputeCommandEncoder()!
            kernels.encodePrefill(encoder, hidden: hidden, hiddenStride: shape.d, pool: pool, scratch: scratches[visit],
                                  tokens: tokens, out: outs[visit], outStride: shape.d)
            encoder.endEncoding()
            try store.endPrefillVisit(layer: step.0, commandBuffer: cb)
        }
        cb.commit()
        cb.waitUntilCompleted()
        store.drainStaging()
        #expect(cb.status == .completed && store.stagingError == nil)
        for (visit, step) in schedule.enumerated() {
            let got = Inkling2MoETestSupport.read(outs[visit], as: Float.self, count: count)
            #expect(got.allSatisfy { $0.isFinite })
            #expect(got == expected[visit], "prefill visit \(visit) (layer \(step.0), pass \(step.1))")
        }

        // Decode: four tokens over the same schedule through the slot caches
        // (8 slots per staged layer, so later passes also evict).
        let decodeTokens = 4
        let (x, _) = Inkling2MoETestSupport.halfBuffer(context.device, rows: decodeTokens, stride: shape.d, d: shape.d, seed: 0x2B62)
        let visits = decodeTokens * schedule.count
        let routeWeights = Inkling2MoETestSupport.randomRoutes(tokens: visits, experts: shape.experts, topK: shape.topK,
                                                               seed: 0x2B63).weights
        var routeIDs: [UInt32] = []
        for t in 0..<decodeTokens {
            for step in schedule { routeIDs += experts(layer: step.0, pass: step.1, token: t).map(UInt32.init) }
        }
        let ids = context.device.makeBuffer(length: visits * shape.topK * 4, options: .storageModeShared)!
        let weights = context.device.makeBuffer(length: visits * shape.topK * 4, options: .storageModeShared)!
        Inkling2MoETestSupport.write(routeIDs, to: ids)
        Inkling2MoETestSupport.write(routeWeights, to: weights)
        for t in 0..<decodeTokens {
            for (visit, step) in schedule.enumerated() {
                let r = t * schedule.count + visit, offset = r * shape.topK * 4
                let want = Self.decode(context, kernels, shape: shape, x: x, xOffset: t * shape.d * 2, weights: weights,
                                       weightsOffset: offset, pool: layers[step.0].pool, ids: ids, idsOffset: offset)
                let got: [Float]
                if let pool = store.residentPool(layer: step.0) {
                    got = Self.decode(context, kernels, shape: shape, x: x, xOffset: t * shape.d * 2, weights: weights,
                                      weightsOffset: offset, pool: pool, ids: ids, idsOffset: offset)
                } else {
                    let routed = routeIDs[(r * shape.topK)..<((r + 1) * shape.topK)].map(Int.init)
                    let binding = try store.prepareDecodeVisit(layer: step.0, expertIDs: routed)
                    #expect(binding.hits + binding.misses == shape.topK)
                    if t == 0 && step.1 > 1 {
                        #expect(binding.misses == shape.topK, "pass \(step.1) of layer \(step.0) loads its own experts")
                    }
                    got = Self.decode(context, kernels, shape: shape, x: x, xOffset: t * shape.d * 2, weights: weights,
                                      weightsOffset: offset, binding: binding, store: store)
                }
                #expect(got.allSatisfy { $0.isFinite })
                #expect(got == want, "decode token \(t) visit \(visit) (layer \(step.0), pass \(step.1))")
            }
        }
    }

    /// A failed read leaves no slot claiming an expert it does not hold.
    @Test func failedDecodeReadRollsBack() throws {
        let context = try MetalContext()
        let shape = Inkling2MoEShape(d: 256, f: 128, experts: 16, topK: 4)
        let (store, _, directory, files) = try Self.cacheStore(context, shape: shape, encoding: .q8_0)
        defer { try? FileManager.default.removeItem(at: directory) }
        let path = files[3]!.path
        let saved = try Data(contentsOf: URL(fileURLWithPath: path))
        try FileManager.default.removeItem(atPath: path)
        #expect(throws: Inkling2ExpertStore.StoreError.openFailed(path: path, errno: ENOENT)) {
            try store.prepareDecodeVisit(layer: 3, expertIDs: [1, 2, 3, 4])
        }
        try saved.write(to: URL(fileURLWithPath: path))
        let retry = try store.prepareDecodeVisit(layer: 3, expertIDs: [1, 2, 3, 4])
        #expect(retry.hits == 0 && retry.misses == 4)
    }

    /// Slot ownership across the two users of the slabs: a prefill fill does
    /// not start while a queued decode command still reads cache slots in
    /// the same slab, and the decode reads the bytes it was bound to.
    @Test func prefillFillWaitsForQueuedDecodeReads() throws {
        let context = try MetalContext()
        let kernels = try Inkling2MoEKernels(context: context)
        let shape = Inkling2MoEShape(d: 256, f: 128, experts: 16, topK: 4)
        let (store, layers, directory, _) = try Self.cacheStore(context, shape: shape)
        defer { try? FileManager.default.removeItem(at: directory) }
        let (x, _) = Inkling2MoETestSupport.halfBuffer(context.device, rows: 1, stride: shape.d, d: shape.d, seed: 0x2B50)
        let routes = Inkling2MoETestSupport.randomRoutes(tokens: 1, experts: shape.experts, topK: shape.topK, seed: 0x2B51)
        let ids = context.device.makeBuffer(length: 16, options: .storageModeShared)!
        let weights = context.device.makeBuffer(length: 16, options: .storageModeShared)!
        Inkling2MoETestSupport.write(routes.ids, to: ids)
        Inkling2MoETestSupport.write(routes.weights, to: weights)
        let want = Self.decode(context, kernels, shape: shape, x: x, xOffset: 0, weights: weights, weightsOffset: 0,
                               pool: layers[1].pool, ids: ids)

        // A decode visit of layer 1 (slab 0), held behind a gate.
        let gate = context.device.makeSharedEvent()!
        let binding = try store.prepareDecodeVisit(layer: 1, expertIDs: routes.ids.map(Int.init))
        let activation = context.device.makeBuffer(length: shape.topK * shape.f * 2, options: .storageModeShared)!
        let out = context.device.makeBuffer(length: shape.d * 4, options: .storageModeShared)!
        let decodeCB = context.queue.makeCommandBuffer()!
        decodeCB.encodeWaitForEvent(gate, value: 1)
        let encoder = decodeCB.makeComputeCommandEncoder()!
        kernels.encodeDecode(encoder, shape: shape, x: x, binding: binding, routeWeights: weights,
                             activation: activation, out: out)
        encoder.endEncoding()
        store.endDecodeVisit(binding, commandBuffer: decodeCB)
        decodeCB.commit()

        // A prefill visit of layer 2, whose ticket also uses slab 0.
        let prefillCB = context.queue.makeCommandBuffer()!
        _ = try store.beginPrefillVisit(layer: 2, commandBuffer: prefillCB)
        try store.endPrefillVisit(layer: 2, commandBuffer: prefillCB)
        prefillCB.commit()
        Thread.sleep(forTimeInterval: 0.3)
        #expect(store.stagingBytesRead == 0, "the fill waited for the queued decode")

        gate.signaledValue = 1
        decodeCB.waitUntilCompleted()
        prefillCB.waitUntilCompleted()
        store.drainStaging()
        #expect(Inkling2MoETestSupport.read(out, as: Float.self, count: shape.d) == want)
        #expect(store.stagingBytesRead > 0 && store.stagingError == nil)
    }
}
