import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

private let sharedExpertTensorOpsAvailable: Bool = {
    guard let context = try? MetalContext() else { return false }
    return MPPPrefillInt4QMM(context: context).isAvailable
}()

@Suite struct PrefillSharedExpertTests {
    private static let rows = 4
    private static let d = 128
    private static let f = 64
    private static let xStride = 144
    private static let yStride = 137
    private static let sentinel = Float16(-7.25)

    @Test func blockSharedExpertMatchesRepeatedScalarRows() throws {
        try Self.runBlockSharedExpertMatchesRepeatedScalarRows()
    }

    /// Qwen's INT4 shared expert must execute the whole token block as three
    /// matrix operations, while retaining the FP16 boundaries between the
    /// gate/up projections, SiLU product, and down projection.
    @Test(.enabled(if: sharedExpertTensorOpsAvailable,
                   "Requires runtime MPP TensorOps support"))
    func qwenInt4BlockUsesTensorOpsAndMatchesRepeatedRows() throws {
        let tokenCount = 32
        var rng = SeedTree(0xC0FFEE).key("qwen-prefill-shared-int4")
        let ctx = try MetalContext()
        let scalar = try SharedExpertInt4(context: ctx, siluActivation: true)
        let prefill = try PrefillSharedExpert(context: ctx,
                                              weightBits: 4,
                                              siluActivation: true)
        let input = (0..<(tokenCount * Self.d)).map { _ in
            Float16(rng.uniform(-0.2, 0.2))
        }
        let gate = Self.makeInt4Weights(rows: Self.f, cols: Self.d, rng: &rng)
        let up = Self.makeInt4Weights(rows: Self.f, cols: Self.d, rng: &rng)
        let down = Self.makeInt4Weights(rows: Self.d, cols: Self.f, rng: &rng)
        let gateProj = Self.makeInt4Projection(ctx: ctx, packed: gate,
                                               rows: Self.f, cols: Self.d)
        let upProj = Self.makeInt4Projection(ctx: ctx, packed: up,
                                             rows: Self.f, cols: Self.d)
        let downProj = Self.makeInt4Projection(ctx: ctx, packed: down,
                                               rows: Self.d, cols: Self.f)

        guard let x = Fp16Buffer.make(ctx.device, halves: input),
              let expected = Fp16Buffer.make(ctx.device, count: tokenCount * Self.d),
              let actual = Fp16Buffer.make(ctx.device, count: tokenCount * Self.d),
              let scalarGate = Fp16Buffer.make(ctx.device, count: Self.f),
              let scalarUp = Fp16Buffer.make(ctx.device, count: Self.f),
              let scalarAct = Fp16Buffer.make(ctx.device, count: Self.f),
              let batchGate = Fp16Buffer.make(ctx.device, count: tokenCount * Self.f),
              let batchUp = Fp16Buffer.make(ctx.device, count: tokenCount * Self.f),
              let batchAct = Fp16Buffer.make(ctx.device, count: tokenCount * Self.f) else {
            Issue.record("buffer allocation failed")
            return
        }
        let halfBytes = MemoryLayout<Float16>.stride
        let referenceCB = ctx.queue.makeCommandBuffer()!
        for token in 0..<tokenCount {
            try scalar.encode(commandBuffer: referenceCB,
                              x: x,
                              xOffset: token * Self.d * halfBytes,
                              gate: gateProj,
                              up: upProj,
                              down: downProj,
                              y: expected,
                              yOffset: token * Self.d * halfBytes,
                              scratchGate: scalarGate,
                              scratchUp: scalarUp,
                              scratchAct: scalarAct)
        }
        referenceCB.commit()
        referenceCB.waitUntilCompleted()
        #expect(referenceCB.error == nil)

        let batchCB = ctx.queue.makeCommandBuffer()!
        let path = try prefill.encodeBlock(
            commandBuffer: batchCB,
            x: x,
            y: actual,
            gate: gateProj,
            up: upProj,
            down: downProj,
            scratchGate: batchGate,
            scratchUp: batchUp,
            scratchAct: batchAct,
            queryCount: tokenCount,
            d: Self.d,
            intermediate: Self.f,
            xStrideElements: Self.d,
            yStrideElements: Self.d)
        batchCB.commit()
        batchCB.waitUntilCompleted()
        #expect(batchCB.error == nil)
        #expect(path == .tensorOpsInt4)

        let reference = Fp16Buffer.read(expected, count: tokenCount * Self.d)
        let got = Fp16Buffer.read(actual, count: tokenCount * Self.d)
        let maxAbs = RelError.maxAbsDiff(got, reference)
        let relative = RelError.compute(actual: got, reference: reference)
        #expect(maxAbs <= 0.05, "maxAbs=\(maxAbs), rel=\(relative)")
        #expect(relative <= 0.003, "maxAbs=\(maxAbs), rel=\(relative)")
    }

    /// Catches either restoring the per-row dispatch loop or changing the
    /// scalar-gate FP16 rounding boundary. The fused block dispatch must match
    /// the exact GEMV-then-sigmoid reference and leave row padding untouched.
    @Test func qwenScalarGateBlockMatchesRepeatedRows() throws {
        var rng = SeedTree(0xC0FFEE).key("qwen-prefill-scalar-gate")
        let ctx = try MetalContext()
        let scalar = try DequantInt8GEMV(context: ctx,
                                         additionalShapes: [(m: 1, n: Self.d)])
        let elementwise = try Elementwise(context: ctx)
        let prefill = try PrefillSharedExpert(context: ctx)
        let x = Self.makeInputBlock(rng: &rng)
        let gate = Self.makeWeights(rows: 1, cols: Self.d, rng: &rng)
        var initialY = Array(repeating: Self.sentinel, count: Self.rows * Self.yStride)
        for row in 0..<Self.rows {
            for col in 0..<Self.d {
                initialY[row * Self.yStride + col] = Float16(rng.uniform(-0.5, 0.5))
            }
        }

        guard let xBuf = Fp16Buffer.make(ctx.device, halves: x),
              let yRef = Fp16Buffer.make(ctx.device, halves: initialY),
              let yGot = Fp16Buffer.make(ctx.device, halves: initialY),
              let gateScratch = Fp16Buffer.make(ctx.device, count: Self.rows) else {
            Issue.record("buffer allocation failed")
            return
        }
        let projection = Self.makeProjection(ctx: ctx, packed: gate,
                                             rows: 1, cols: Self.d)
        let halfBytes = MemoryLayout<Float16>.stride

        let refCB = ctx.queue.makeCommandBuffer()!
        for row in 0..<Self.rows {
            scalar.encode(commandBuffer: refCB,
                          weights: projection.weights,
                          scales: projection.scales,
                          biases: projection.biases,
                          x: xBuf,
                          xOffset: row * Self.xStride * halfBytes,
                          y: gateScratch,
                          yOffset: row * halfBytes,
                          m: 1, n: UInt32(Self.d))
            elementwise.encodeSigmoidScalarMul(
                commandBuffer: refCB,
                y: yRef,
                yOffset: row * Self.yStride * halfBytes,
                gate: gateScratch,
                gateOffset: row * halfBytes,
                count: Self.d)
        }
        refCB.commit()
        refCB.waitUntilCompleted()
        #expect(refCB.error == nil)

        let gotCB = ctx.queue.makeCommandBuffer()!
        try prefill.encodeQwenScalarGate(
            commandBuffer: gotCB,
            x: xBuf,
            gate: projection,
            y: yGot,
            queryCount: Self.rows,
            d: Self.d,
            xStrideElements: Self.xStride,
            yStrideElements: Self.yStride)
        gotCB.commit()
        gotCB.waitUntilCompleted()
        #expect(gotCB.error == nil)

        let expected = Fp16Buffer.readHalf(yRef, count: Self.rows * Self.yStride)
        let actual = Fp16Buffer.readHalf(yGot, count: Self.rows * Self.yStride)
        #expect(actual == expected)
        Self.assertPaddingUnchanged(actual)
    }

    @Test func sharedExpertPhaseSplitMatchesCurrentPrefillBlock() throws {
        var rng = SeedTree(0xC0FFEE).key("prefill-shared-expert-phase-split")
        let ctx = try MetalContext()
        let shared = try SharedExpertInt8(context: ctx)
        let prefill = try PrefillSharedExpert(context: ctx)

        let x = Self.makeInputBlock(rng: &rng)
        let gate = Self.makeWeights(rows: Self.f, cols: Self.d, rng: &rng)
        let up = Self.makeWeights(rows: Self.f, cols: Self.d, rng: &rng)
        let down = Self.makeWeights(rows: Self.d, cols: Self.f, rng: &rng)
        let yElements = Self.rows * Self.d

        guard let xBuf = Fp16Buffer.make(ctx.device, halves: x),
              let yRef = Fp16Buffer.make(ctx.device, halves: Array(repeating: Self.sentinel, count: yElements)),
              let ySplit = Fp16Buffer.make(ctx.device, halves: Array(repeating: Self.sentinel, count: yElements)),
              let scratchGate = Fp16Buffer.make(ctx.device, count: Self.f),
              let scratchUp = Fp16Buffer.make(ctx.device, count: Self.f),
              let scratchAct = Fp16Buffer.make(ctx.device, count: Self.rows * Self.f) else {
            Issue.record("buffer allocation failed")
            return
        }

        let gateProj = Self.makeProjection(ctx: ctx, packed: gate, rows: Self.f, cols: Self.d)
        let upProj = Self.makeProjection(ctx: ctx, packed: up, rows: Self.f, cols: Self.d)
        let downProj = Self.makeProjection(ctx: ctx, packed: down, rows: Self.d, cols: Self.f)
        let halfBytes = MemoryLayout<Float16>.stride

        let refCB = ctx.queue.makeCommandBuffer()!
        try prefill.encodeBlock(commandBuffer: refCB,
                                x: xBuf,
                                y: yRef,
                                gate: gateProj,
                                up: upProj,
                                down: downProj,
                                scratchGate: scratchGate,
                                scratchUp: scratchUp,
                                scratchAct: scratchAct,
                                queryCount: Self.rows,
                                d: Self.d,
                                intermediate: Self.f,
                                xStrideElements: Self.xStride,
                                yStrideElements: Self.d)
        refCB.commit()
        refCB.waitUntilCompleted()
        #expect(refCB.error == nil)

        let splitCB = ctx.queue.makeCommandBuffer()!
        for row in 0..<Self.rows {
            try shared.encodePhase1(commandBuffer: splitCB,
                                    x: xBuf,
                                    xOffset: row * Self.xStride * halfBytes,
                                    gate: gateProj,
                                    up: upProj,
                                    scratchAct: scratchAct,
                                    scratchActOffset: row * Self.f * halfBytes)
        }
        for row in 0..<Self.rows {
            try shared.encodeDown(commandBuffer: splitCB,
                                  down: downProj,
                                  y: ySplit,
                                  yOffset: row * Self.d * halfBytes,
                                  scratchAct: scratchAct,
                                  scratchActOffset: row * Self.f * halfBytes)
        }
        splitCB.commit()
        splitCB.waitUntilCompleted()
        #expect(splitCB.error == nil)

        let ref = Fp16Buffer.readHalf(yRef, count: yElements)
        let got = Fp16Buffer.readHalf(ySplit, count: yElements)
        #expect(got == ref)
    }

    @Test func blockSharedExpertThenPostF1NormMatchesRepeatedScalarRows() throws {
        var rng = SeedTree(0xC0FFEE).key("prefill-shared-expert-postf1")
        let ctx = try MetalContext()
        let scalar = try SharedExpertInt8(context: ctx)
        let scalarRMS = try RMSNorm(context: ctx)
        let prefill = try PrefillSharedExpert(context: ctx)
        let prefillRMS = try PrefillRMSNorm(context: ctx)

        let x = Self.makeInputBlock(rng: &rng)
        let gate = Self.makeWeights(rows: Self.f, cols: Self.d, rng: &rng)
        let up = Self.makeWeights(rows: Self.f, cols: Self.d, rng: &rng)
        let down = Self.makeWeights(rows: Self.d, cols: Self.f, rng: &rng)
        let postF1 = Self.makeBF16Weights(rng: &rng)
        let yElements = Self.rows * Self.d

        guard let xBuf = Fp16Buffer.make(ctx.device, halves: x),
              let yRef = Fp16Buffer.make(ctx.device, halves: Array(repeating: Self.sentinel, count: yElements)),
              let yGot = Fp16Buffer.make(ctx.device, halves: Array(repeating: Self.sentinel, count: yElements)),
              let scratchGate = Fp16Buffer.make(ctx.device, count: Self.f),
              let scratchUp = Fp16Buffer.make(ctx.device, count: Self.f),
              let scratchAct = Fp16Buffer.make(ctx.device, count: Self.f),
              let postF1Buf = ctx.device.makeBuffer(bytes: postF1,
                                                     length: postF1.count * MemoryLayout<UInt16>.stride,
                                                     options: .storageModeShared) else {
            Issue.record("buffer allocation failed")
            return
        }

        let gateProj = Self.makeProjection(ctx: ctx, packed: gate, rows: Self.f, cols: Self.d)
        let upProj = Self.makeProjection(ctx: ctx, packed: up, rows: Self.f, cols: Self.d)
        let downProj = Self.makeProjection(ctx: ctx, packed: down, rows: Self.d, cols: Self.f)

        let halfBytes = MemoryLayout<Float16>.stride
        let refCB = ctx.queue.makeCommandBuffer()!
        for row in 0..<Self.rows {
            try scalar.encode(commandBuffer: refCB,
                              x: xBuf,
                              xOffset: row * Self.xStride * halfBytes,
                              gate: gateProj,
                              up: upProj,
                              down: downProj,
                              y: yRef,
                              yOffset: row * Self.d * halfBytes,
                              scratchAct: scratchAct)
            scalarRMS.encodeBF16W(commandBuffer: refCB,
                                  x: yRef,
                                  xOffset: row * Self.d * halfBytes,
                                  weight: postF1Buf,
                                  out: yRef,
                                  outOffset: row * Self.d * halfBytes,
                                  d: UInt32(Self.d),
                                  eps: 1e-6)
        }
        refCB.commit()
        refCB.waitUntilCompleted()
        #expect(refCB.error == nil)

        let gotCB = ctx.queue.makeCommandBuffer()!
        try prefill.encodeBlock(commandBuffer: gotCB,
                                x: xBuf,
                                y: yGot,
                                gate: gateProj,
                                up: upProj,
                                down: downProj,
                                scratchGate: scratchGate,
                                scratchUp: scratchUp,
                                scratchAct: scratchAct,
                                queryCount: Self.rows,
                                d: Self.d,
                                intermediate: Self.f,
                                xStrideElements: Self.xStride,
                                yStrideElements: Self.d)
        prefillRMS.encodeBF16W(commandBuffer: gotCB,
                               x: yGot,
                               weight: postF1Buf,
                               out: yGot,
                               t: UInt32(Self.rows),
                               d: UInt32(Self.d),
                               eps: 1e-6)
        gotCB.commit()
        gotCB.waitUntilCompleted()
        #expect(gotCB.error == nil)

        let ref = Fp16Buffer.readHalf(yRef, count: yElements)
        let got = Fp16Buffer.readHalf(yGot, count: yElements)
        #expect(got == ref)
    }

    private static func runBlockSharedExpertMatchesRepeatedScalarRows() throws {
        var rng = SeedTree(0xC0FFEE).key("prefill-shared-expert")
        let ctx = try MetalContext()
        let scalar = try SharedExpertInt8(context: ctx)
        let prefill = try PrefillSharedExpert(context: ctx)

        let x = makeInputBlock(rng: &rng)
        let gate = makeWeights(rows: f, cols: d, rng: &rng)
        let up = makeWeights(rows: f, cols: d, rng: &rng)
        let down = makeWeights(rows: d, cols: f, rng: &rng)

        guard let xBuf = Fp16Buffer.make(ctx.device, halves: x),
              let yRef = Fp16Buffer.make(ctx.device, halves: Array(repeating: sentinel, count: rows * yStride)),
              let yGot = Fp16Buffer.make(ctx.device, halves: Array(repeating: sentinel, count: rows * yStride)),
              let scratchGate = Fp16Buffer.make(ctx.device, count: f),
              let scratchUp = Fp16Buffer.make(ctx.device, count: f),
              let scratchAct = Fp16Buffer.make(ctx.device, count: f) else {
            Issue.record("buffer allocation failed")
            return
        }

        let gateProj = makeProjection(ctx: ctx, packed: gate, rows: f, cols: d)
        let upProj = makeProjection(ctx: ctx, packed: up, rows: f, cols: d)
        let downProj = makeProjection(ctx: ctx, packed: down, rows: d, cols: f)

        let halfBytes = MemoryLayout<Float16>.stride
        let refCB = ctx.queue.makeCommandBuffer()!
        for row in 0..<rows {
            try scalar.encode(commandBuffer: refCB,
                              x: xBuf,
                              xOffset: row * xStride * halfBytes,
                              gate: gateProj,
                              up: upProj,
                              down: downProj,
                              y: yRef,
                              yOffset: row * yStride * halfBytes,
                              scratchAct: scratchAct)
        }
        refCB.commit()
        refCB.waitUntilCompleted()
        #expect(refCB.error == nil)

        let gotCB = ctx.queue.makeCommandBuffer()!
        try prefill.encodeBlock(commandBuffer: gotCB,
                                x: xBuf,
                                y: yGot,
                                gate: gateProj,
                                up: upProj,
                                down: downProj,
                                scratchGate: scratchGate,
                                scratchUp: scratchUp,
                                scratchAct: scratchAct,
                                queryCount: rows,
                                d: d,
                                intermediate: f,
                                xStrideElements: xStride,
                                yStrideElements: yStride)
        gotCB.commit()
        gotCB.waitUntilCompleted()
        #expect(gotCB.error == nil)

        let ref = Fp16Buffer.readHalf(yRef, count: rows * yStride)
        let got = Fp16Buffer.readHalf(yGot, count: rows * yStride)
        #expect(got == ref)
        assertPaddingUnchanged(got)
    }

    private static func makeInputBlock(rng: inout SplitMix64) -> [Float16] {
        var block = Array(repeating: sentinel, count: rows * xStride)
        for row in 0..<rows {
            for col in 0..<d {
                block[row * xStride + col] = Float16(rng.uniform(-0.35, 0.35))
            }
        }
        return block
    }

    private static func makeWeights(rows: Int,
                                    cols: Int,
                                    rng: inout SplitMix64)
        -> (packed: [UInt8], scales: [UInt16], biases: [UInt16])
    {
        let groupsPerRow = cols / Quantization.groupSize
        var packed = [UInt8](repeating: 0, count: rows * cols)
        var scales = [UInt16](repeating: 0, count: rows * groupsPerRow)
        var biases = [UInt16](repeating: 0, count: rows * groupsPerRow)
        for row in 0..<rows {
            let values = (0..<cols).map { _ in rng.uniform(-0.4, 0.4) }
            let q = Quantization.quantizeInt8Affine(values)
            for col in 0..<cols {
                packed[row * cols + col] = q.packed[col]
            }
            for group in 0..<groupsPerRow {
                scales[row * groupsPerRow + group] = q.scales[group]
                biases[row * groupsPerRow + group] = q.biases[group]
            }
        }
        return (packed, scales, biases)
    }

    private static func makeInt4Weights(rows: Int,
                                        cols: Int,
                                        rng: inout SplitMix64)
        -> (packed: [UInt8], scales: [UInt16], biases: [UInt16])
    {
        let quantized = (0..<rows).map { _ in
            Quantization.quantizeInt4Affine(
                (0..<cols).map { _ in rng.uniform(-0.25, 0.25) })
        }
        return (quantized.flatMap(\.packed),
                quantized.flatMap(\.scales),
                quantized.flatMap(\.biases))
    }

    private static func makeBF16Weights(rng: inout SplitMix64) -> [UInt16] {
        (0..<d).map { _ in Quantization.bf16Bits(rng.uniform(0.75, 1.25)) }
    }

    private static func makeProjection(
        ctx: MetalContext,
        packed: (packed: [UInt8], scales: [UInt16], biases: [UInt16]),
        rows: Int,
        cols: Int
    ) -> SharedExpertInt8Proj {
        let w = ctx.device.makeBuffer(bytes: packed.packed,
                                      length: packed.packed.count,
                                      options: .storageModeShared)!
        let s = ctx.device.makeBuffer(bytes: packed.scales,
                                      length: packed.scales.count * MemoryLayout<UInt16>.stride,
                                      options: .storageModeShared)!
        let b = ctx.device.makeBuffer(bytes: packed.biases,
                                      length: packed.biases.count * MemoryLayout<UInt16>.stride,
                                      options: .storageModeShared)!
        return SharedExpertInt8Proj(weights: w,
                                    scales: s,
                                    biases: b,
                                    rows: UInt32(rows),
                                    cols: UInt32(cols))
    }

    private static func makeInt4Projection(
        ctx: MetalContext,
        packed: (packed: [UInt8], scales: [UInt16], biases: [UInt16]),
        rows: Int,
        cols: Int
    ) -> SharedExpertProjection {
        let weights = ctx.device.makeBuffer(bytes: packed.packed,
                                            length: packed.packed.count,
                                            options: .storageModeShared)!
        let scales = ctx.device.makeBuffer(bytes: packed.scales,
                                           length: packed.scales.count * 2,
                                           options: .storageModeShared)!
        let biases = ctx.device.makeBuffer(bytes: packed.biases,
                                           length: packed.biases.count * 2,
                                           options: .storageModeShared)!
        return SharedExpertProjection(weights: weights,
                                      scales: scales,
                                      biases: biases,
                                      rows: UInt32(rows),
                                      cols: UInt32(cols))
    }

    private static func assertPaddingUnchanged(_ values: [Float16]) {
        for row in 0..<rows {
            for col in d..<yStride {
                #expect(values[row * yStride + col] == sentinel)
            }
        }
    }
}
