import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

@Suite struct MoEFusedFFNTests {
    private static let dimension = 128
    private static let intermediate = 64

    private struct RoutedBlob {
        let bytes: [UInt8]
        let offsets: MoEExpertOffsets
    }

    @Test func routedArgumentBufferReusesPreallocatedStorage() throws {
        let context = try MetalContext()
        let kernel = try MoE(context: context, specializedTopK: 6)
        let routedBuffers = (0..<6).compactMap { _ in
            context.device.makeBuffer(length: 1, options: .storageModeShared)
        }
        #expect(routedBuffers.count == 6)

        let first = try #require(kernel.makeRoutedArgumentBuffer(
            routedBlobs: routedBuffers.map { (buffer: $0, offset: 0) }, topK: 6))
        let second = try #require(kernel.makeRoutedArgumentBuffer(
            routedBlobs: routedBuffers.map { (buffer: $0, offset: 0) }, topK: 6))

        #expect(first === second)
    }

    @Test func productionTop8PipelineAndHitSplitMatchReference() throws {
        try Self.checkPipeline(topK: 8,
                               splitSlots: [[0, 1, 2, 3], [4, 5, 6, 7]],
                               seedKey: "production-routed-moe-top8")
    }

    @Test func inklingTop6PipelineAndHitSplitMatchReference() throws {
        try Self.checkPipeline(topK: 6,
                               splitSlots: [[0, 2, 4], [1, 3, 5]],
                               seedKey: "inkling-routed-moe-top6")
    }

    private static func checkPipeline(topK: Int,
                                      splitSlots: [[UInt32]],
                                      seedKey: String) throws {
        var rng = SeedTree(0x2D3).key(seedKey)
        func matrix(rows: Int, columns: Int) -> [[Float]] {
            (0..<rows).map { _ in
                (0..<columns).map { _ in rng.uniform(-0.4, 0.4) }
            }
        }

        var gates = [[[Float]]]()
        var ups = [[[Float]]]()
        var downs = [[[Float]]]()
        for _ in 0..<topK {
            gates.append(matrix(rows: Self.intermediate, columns: Self.dimension))
            ups.append(matrix(rows: Self.intermediate, columns: Self.dimension))
            downs.append(matrix(rows: Self.dimension, columns: Self.intermediate))
        }
        let x = (0..<Self.dimension).map { _ in
            Float(Float16(rng.uniform(-0.5, 0.5)))
        }
        let residual = (0..<Self.dimension).map { _ in
            Float(Float16(rng.uniform(-0.5, 0.5)))
        }
        let routingWeights = (0..<topK).map {
            Float(Float16(0.04 + Float($0) * 0.015))
        }
        let expected = MoeRef.applyStreamedRouted(
            x: x,
            residual: residual,
            routedGate: gates.map { rows in
                rows.map { Quantization.quantizeInt4Affine($0) }
            },
            routedUp: ups.map { rows in
                rows.map { Quantization.quantizeInt4Affine($0) }
            },
            routedDown: downs.map { rows in
                rows.map { Quantization.quantizeInt4Affine($0) }
            },
            indices: Array(0..<topK),
            routingWeights: routingWeights,
            d: Self.dimension,
            f: Self.intermediate)
        let blobs = (0..<topK).map {
            Self.makeBlob(gate: gates[$0], up: ups[$0], down: downs[$0])
        }

        let context = try MetalContext()
        let kernel = try MoE(context: context,
                             specializedD: UInt32(Self.dimension),
                             specializedF: UInt32(Self.intermediate),
                             specializedNumExperts: 128,
                             specializedTopK: UInt32(topK))
        let routedBuffers = blobs.compactMap {
            context.device.makeBuffer(bytes: $0.bytes,
                                      length: $0.bytes.count,
                                      options: .storageModeShared)
        }
        let activeBuffers = splitSlots.compactMap { slots in
            context.device.makeBuffer(
                bytes: slots,
                length: slots.count * MemoryLayout<UInt32>.stride,
                options: .storageModeShared)
        }
        guard routedBuffers.count == topK,
              activeBuffers.count == splitSlots.count,
              let xBuffer = Fp16Buffer.make(context.device, values: x),
              let residualBuffer = Fp16Buffer.make(context.device, values: residual),
              let routingBuffer = Fp16Buffer.make(context.device, values: routingWeights),
              let fullActs = Fp16Buffer.make(
                context.device, count: topK * Self.intermediate),
              let splitActs = Fp16Buffer.make(
                context.device, count: topK * Self.intermediate),
              let fullOutput = Fp16Buffer.make(context.device, count: Self.dimension),
              let splitOutput = Fp16Buffer.make(context.device, count: Self.dimension),
              let argumentBuffer = kernel.makeRoutedArgumentBuffer(
                routedBlobs: routedBuffers.map { (buffer: $0, offset: 0) },
                topK: UInt32(topK)) else {
            Issue.record("buffer allocation failed")
            return
        }

        let fullCommand = context.queue.makeCommandBuffer()!
        kernel.encodeRoutedPersistentPhase1U16Load(
            commandBuffer: fullCommand,
            routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers.map { (buffer: $0, offset: 0) },
            routedOffsets: blobs[0].offsets,
            x: xBuffer,
            acts: fullActs,
            d: UInt32(Self.dimension),
            f: UInt32(Self.intermediate),
            topK: UInt32(topK))
        kernel.encodeRoutedPersistentPhase2Reduce(
            commandBuffer: fullCommand,
            routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers.map { (buffer: $0, offset: 0) },
            routedOffsets: blobs[0].offsets,
            acts: fullActs,
            routingWeights: routingBuffer,
            residual: residualBuffer,
            y: fullOutput,
            d: UInt32(Self.dimension),
            f: UInt32(Self.intermediate),
            topK: UInt32(topK))
        fullCommand.commit()
        fullCommand.waitUntilCompleted()
        #expect(fullCommand.error == nil)

        let splitCommand = context.queue.makeCommandBuffer()!
        for (slots, activeSlots) in zip(splitSlots, activeBuffers) {
            kernel.encodeRoutedPersistentPhase1SubsetU16Load(
                commandBuffer: splitCommand,
                routedArgBuffer: argumentBuffer,
                routedBlobs: routedBuffers.map { (buffer: $0, offset: 0) },
                routedOffsets: blobs[0].offsets,
                x: xBuffer,
                acts: splitActs,
                activeSlots: activeSlots,
                activeSlotIndices: slots,
                activeCount: UInt32(slots.count),
                d: UInt32(Self.dimension),
                f: UInt32(Self.intermediate),
                topK: UInt32(topK))
        }
        kernel.encodeRoutedPersistentPhase2Reduce(
            commandBuffer: splitCommand,
            routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers.map { (buffer: $0, offset: 0) },
            routedOffsets: blobs[0].offsets,
            acts: splitActs,
            routingWeights: routingBuffer,
            residual: residualBuffer,
            y: splitOutput,
            d: UInt32(Self.dimension),
            f: UInt32(Self.intermediate),
            topK: UInt32(topK))
        splitCommand.commit()
        splitCommand.waitUntilCompleted()
        #expect(splitCommand.error == nil)

        let full = Fp16Buffer.read(fullOutput, count: Self.dimension)
        let split = Fp16Buffer.read(splitOutput, count: Self.dimension)
        #expect(full == split)
        #expect(RelError.compute(actual: full, reference: expected)
            < Tolerance.fp16ChainedReduction)
    }

    /// S3 parity: the slot-map kernels (slab + GPU-resolved offsets +
    /// all-hit guard) must produce bit-identical output to the
    /// argument-buffer production path on identical expert bytes.
    @Test func slotMapPipelineMatchesArgumentBufferPath() throws {
        var rng = SeedTree(0x51A9).key("slotmap-parity-top8")
        let topK = 8
        func matrix(rows: Int, columns: Int) -> [[Float]] {
            (0..<rows).map { _ in
                (0..<columns).map { _ in rng.uniform(-0.4, 0.4) }
            }
        }
        var blobs = [RoutedBlob]()
        for _ in 0..<topK {
            blobs.append(Self.makeBlob(
                gate: matrix(rows: Self.intermediate, columns: Self.dimension),
                up: matrix(rows: Self.intermediate, columns: Self.dimension),
                down: matrix(rows: Self.dimension, columns: Self.intermediate)))
        }
        let x = (0..<Self.dimension).map { _ in Float(Float16(rng.uniform(-0.5, 0.5))) }
        let residual = (0..<Self.dimension).map { _ in Float(Float16(rng.uniform(-0.5, 0.5))) }
        let routingWeights = (0..<topK).map { Float(Float16(0.04 + Float($0) * 0.015)) }

        let context = try MetalContext()
        let kernel = try MoE(context: context,
                             specializedD: UInt32(Self.dimension),
                             specializedF: UInt32(Self.intermediate),
                             specializedNumExperts: 128,
                             specializedTopK: UInt32(topK))

        // One contiguous slab holding the blobs at page-ish strides, with the
        // experts deliberately permuted across slots so offsets matter.
        let stride = ((blobs.map { $0.bytes.count }.max()! + 4095) / 4096) * 4096
        var slabBytes = [UInt8](repeating: 0, count: stride * topK)
        let slotOfExpert: [Int] = [5, 2, 7, 0, 3, 6, 1, 4]
        for (expert, slot) in slotOfExpert.enumerated() {
            slabBytes.replaceSubrange(
                slot * stride ..< slot * stride + blobs[expert].bytes.count,
                with: blobs[expert].bytes)
        }
        // 256-entry expert->slot table; experts 0..7 live in permuted slots.
        var table = [Int16](repeating: -1, count: 256)
        for (expert, slot) in slotOfExpert.enumerated() { table[expert] = Int16(slot) }
        // The router picked experts [0..7] in order.
        let indices: [UInt32] = (0..<topK).map { UInt32($0) }

        guard let slab = context.device.makeBuffer(
                bytes: slabBytes, length: slabBytes.count, options: .storageModeShared),
              let tableBuf = context.device.makeBuffer(
                bytes: table, length: 256 * MemoryLayout<Int16>.stride,
                options: .storageModeShared),
              let indexBuf = context.device.makeBuffer(
                bytes: indices, length: topK * MemoryLayout<UInt32>.stride,
                options: .storageModeShared),
              let offsetsBuf = context.device.makeBuffer(
                length: topK * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let allHitBuf = context.device.makeBuffer(
                length: MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let xBuffer = Fp16Buffer.make(context.device, values: x),
              let residualBuffer = Fp16Buffer.make(context.device, values: residual),
              let routingBuffer = Fp16Buffer.make(context.device, values: routingWeights),
              let refActs = Fp16Buffer.make(context.device, count: topK * Self.intermediate),
              let mapActs = Fp16Buffer.make(context.device, count: topK * Self.intermediate),
              let refOut = Fp16Buffer.make(context.device, count: Self.dimension),
              let mapOut = Fp16Buffer.make(context.device, count: Self.dimension),
              let hidden = Fp16Buffer.make(context.device, values: residual) else {
            Issue.record("buffer allocation failed")
            return
        }

        // Reference: argument-buffer path over per-expert slab subranges.
        let refBlobs = (0..<topK).map {
            (buffer: slab, offset: slotOfExpert[$0] * stride)
        }
        guard let argumentBuffer = kernel.makeRoutedArgumentBuffer(
            routedBlobs: refBlobs, topK: UInt32(topK)) else {
            Issue.record("argument buffer failed")
            return
        }
        let refCommand = context.queue.makeCommandBuffer()!
        kernel.encodeRoutedPersistentPhase1U16Load(
            commandBuffer: refCommand, routedArgBuffer: argumentBuffer,
            routedBlobs: refBlobs, routedOffsets: blobs[0].offsets,
            x: xBuffer, acts: refActs,
            d: UInt32(Self.dimension), f: UInt32(Self.intermediate), topK: UInt32(topK))
        kernel.encodeRoutedPersistentPhase2Reduce(
            commandBuffer: refCommand, routedArgBuffer: argumentBuffer,
            routedBlobs: refBlobs, routedOffsets: blobs[0].offsets,
            acts: refActs, routingWeights: routingBuffer,
            residual: residualBuffer, y: refOut,
            d: UInt32(Self.dimension), f: UInt32(Self.intermediate), topK: UInt32(topK))
        refCommand.commit()
        refCommand.waitUntilCompleted()
        #expect(refCommand.error == nil)

        // Slot-map path: GPU lookup resolves the same slots, then the guarded
        // chain runs (and must find all_hit == 1).
        let mapCommand = context.queue.makeCommandBuffer()!
        kernel.encodeSlotLookup(
            commandBuffer: mapCommand, indices: indexBuf, table: tableBuf,
            slotStride: stride, slotOffsets: offsetsBuf, allHit: allHitBuf,
            numExperts: 256, topK: UInt32(topK))
        kernel.encodeSlotMapGuardedFFN(
            commandBuffer: mapCommand, slab: slab, slotOffsets: offsetsBuf,
            allHit: allHitBuf, routedOffsets: blobs[0].offsets,
            x: xBuffer, acts: mapActs, routingWeights: routingBuffer,
            residual: residualBuffer, y: mapOut, hidden: hidden,
            d: UInt32(Self.dimension), f: UInt32(Self.intermediate), topK: UInt32(topK))
        mapCommand.commit()
        mapCommand.waitUntilCompleted()
        #expect(mapCommand.error == nil)

        #expect(allHitBuf.contents().load(as: UInt32.self) == 1)
        let refA = Fp16Buffer.read(refActs, count: topK * Self.intermediate)
        let mapA = Fp16Buffer.read(mapActs, count: topK * Self.intermediate)
        #expect(refA == mapA, "phase-1 activations diverge")
        let ref = Fp16Buffer.read(refOut, count: Self.dimension)
        let map = Fp16Buffer.read(mapOut, count: Self.dimension)
        #expect(ref == map, "phase-2 outputs diverge")
        // hidden must equal residual + y (the guarded residual add ran).
        let hiddenOut = Fp16Buffer.read(hidden, count: Self.dimension)
        let expectedHidden = zip(residual, map).map {
            Float(Float16(Float(Float16($0)) + Float(Float16($1))))
        }
        #expect(hiddenOut == expectedHidden, "guarded residual add diverges")
    }

    private static func makeBlob(gate: [[Float]],
                                 up: [[Float]],
                                 down: [[Float]]) -> RoutedBlob {
        func packed(_ rows: [[Float]])
            -> (weights: [UInt8], scales: [UInt16], biases: [UInt16]) {
            let quantized = rows.map { Quantization.quantizeInt4Affine($0) }
            return (quantized.flatMap(\.packed),
                    quantized.flatMap(\.scales),
                    quantized.flatMap(\.biases))
        }
        var bytes = [UInt8]()
        func append(_ values: [UInt8]) { bytes.append(contentsOf: values) }
        func append(_ values: [UInt16]) {
            for value in values {
                bytes.append(UInt8(truncatingIfNeeded: value))
                bytes.append(UInt8(truncatingIfNeeded: value >> 8))
            }
        }
        let gateValues = packed(gate)
        let upValues = packed(up)
        let downValues = packed(down)
        let gateW = UInt32(bytes.count); append(gateValues.weights)
        let gateS = UInt32(bytes.count); append(gateValues.scales)
        let gateB = UInt32(bytes.count); append(gateValues.biases)
        let upW = UInt32(bytes.count); append(upValues.weights)
        let upS = UInt32(bytes.count); append(upValues.scales)
        let upB = UInt32(bytes.count); append(upValues.biases)
        let downW = UInt32(bytes.count); append(downValues.weights)
        let downS = UInt32(bytes.count); append(downValues.scales)
        let downB = UInt32(bytes.count); append(downValues.biases)
        return RoutedBlob(
            bytes: bytes,
            offsets: MoEExpertOffsets(
                gateWOff: gateW, gateSOff: gateS, gateBOff: gateB,
                upWOff: upW, upSOff: upS, upBOff: upB,
                downWOff: downW, downSOff: downS, downBOff: downB))
    }
}
