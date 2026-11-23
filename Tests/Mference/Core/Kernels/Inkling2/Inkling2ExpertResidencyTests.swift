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

    @Test func stagedVisitsMatchResidentLayers() throws {
        let context = try MetalContext()
        let kernels = try Inkling2MoEKernels(context: context)
        let shape = Inkling2MoEShape(d: 256, f: 128, experts: 8, topK: 4)
        let layers = try (0..<3).map {
            try Inkling2SyntheticPool.make(device: context.device, encoding: .nvfp4, shape: shape,
                                           seed: 0x2B00 + UInt64($0), quantized: true)
        }
        let (directory, files) = try Self.writeLayers(layers, header: 4096)
        defer { try? FileManager.default.removeItem(at: directory) }
        let staged = try Inkling2StagedExpertLayers(device: context.device, files: files, slabs: 2)

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

    @Test func storeServesResidentStagedAndMappedLayers() throws {
        let context = try MetalContext()
        let kernels = try Inkling2MoEKernels(context: context)
        let shape = Inkling2MoEShape(d: 256, f: 128, experts: 8, topK: 4)
        let layers = try (0..<3).map {
            try Inkling2SyntheticPool.make(device: context.device, encoding: .nvfp4, shape: shape,
                                           seed: 0x2B20 + UInt64($0), quantized: true)
        }
        let (directory, files) = try Self.writeLayers(layers, header: 16384)
        defer { try? FileManager.default.removeItem(at: directory) }
        let specs = layers.enumerated().map { layer, pool in
            Inkling2ExpertLayerSpec(file: files[layer]!, encoding: .nvfp4, expertStride: pool.pool.expertStride,
                                    gateUpScales: (pool.pool.gateUpScales, pool.pool.gateUpScalesOffset),
                                    downScales: (pool.pool.downScales, pool.pool.downScalesOffset))
        }
        let plan = Inkling2ExpertResidencyPlan(placements: [.resident, .staged, .staged], stagingSlabs: 2,
                                               residentBytes: files[0]!.bytes, stagingBytes: 0)
        let store = try Inkling2ExpertStore(device: context.device, shape: shape, layers: specs, plan: plan)

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
        store.staging?.drain()
        #expect(cb.status == .completed && store.staging?.error == nil)
        for (i, layer) in visits.enumerated() {
            #expect(Inkling2MoETestSupport.read(outs[i], as: Float.self, count: count) == expected[layer], "visit \(i)")
        }
        #expect(store.staging?.bytesRead == 4 * files[1]!.bytes)

        // Decode: resident and mapped pools give the same bytes as the originals.
        let activation = context.device.makeBuffer(length: shape.topK * shape.f * 2, options: .storageModeShared)!
        for layer in 0..<3 {
            let a = output(), b = output()
            _ = Inkling2MoETestSupport.run(context) {
                kernels.encodeDecode($0, shape: shape, x: hidden, ids: scratch.ids, routeWeights: scratch.routeWeights,
                                     pool: layers[layer].pool, activation: activation, out: a)
            }
            _ = Inkling2MoETestSupport.run(context) {
                kernels.encodeDecode($0, shape: shape, x: hidden, ids: scratch.ids, routeWeights: scratch.routeWeights,
                                     pool: store.decodePool(layer: layer), activation: activation, out: b)
            }
            #expect(Inkling2MoETestSupport.read(a, as: Float.self, count: shape.d)
                    == Inkling2MoETestSupport.read(b, as: Float.self, count: shape.d), "layer \(layer)")
        }
        #expect(throws: Inkling2ExpertStore.StoreError.visitNotStarted(layer: 1)) {
            try store.endPrefillVisit(layer: 1, commandBuffer: context.queue.makeCommandBuffer()!)
        }
    }
}
