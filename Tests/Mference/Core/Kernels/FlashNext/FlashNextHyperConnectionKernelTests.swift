import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

/// Stage 2a of the Flash-Next Metal port: group RMSNorm, the BF16 mat-vec the
/// BF16-passthrough install needs, and the low-rank hyper-connection mix /
/// inject — each against `FlashNextHyperConnectionReference`, which is the CPU
/// reference forward's own arithmetic (see its tie-back suite).
///
/// # Tolerance policy
///
/// Every test feeds the CPU reference the values the GPU actually sees: inputs
/// rounded through FP16, weights through BF16 or INT4-affine-group-64 first. So
/// the residual difference is the *kernel's* error — summation order, `rsqrt`
/// versus `1/sqrt`, and the FP16 stores the runtime's convention imposes — not
/// the storage format's. Each gate states its own bound and prints the observed
/// margin, so the numbers can be compared against the CPU reference's own
/// margins against the goldens (3e-6..2.9e-5 max-abs, printed by
/// `reportsObservedParityMargins`).
@Suite struct FlashNextHyperConnectionKernelTests {

    /// Production geometry: 2560 hidden, 4 streams, rank 320.
    private static let hidden = 2560
    private static let hcCount = 4
    private static let lowRank = 320
    private static let eps: Float = 1e-6
    private static var bundle: Int { hidden * hcCount }

    private static var geometry: FlashNextHyperConnectionReference.Geometry {
        .init(hidden: hidden, hcCount: hcCount, lowRank: lowRank, eps: eps)
    }

    // MARK: - Group RMSNorm

    @Test(arguments: [1, 3])
    func groupRMSNormMatchesTheReference(rows: Int) throws {
        let context = try MetalContext()
        let rms = try RMSNorm(context: context)
        var rng = SplitMix64(seed: 0xF1A5_0001 &+ UInt64(rows))

        // The GPU reads FP16 activations and a BF16 weight; the reference is
        // handed exactly those values, so only kernel error is left.
        let x = Self.roundedFP16((0..<(rows * Self.bundle)).map { _ in rng.uniform(-3.0, 3.0) })
        let weight = Self.roundedBF16((0..<Self.bundle).map { _ in rng.uniform(0.2, 1.8) })

        var expected = [Float](repeating: 0, count: rows * Self.bundle)
        for t in 0..<rows {
            let row = FlashNextHyperConnectionReference.groupRMSNorm(
                x, offset: t * Self.bundle, weight: weight, g: Self.geometry)
            for i in 0..<Self.bundle { expected[t * Self.bundle + i] = row[i] }
        }

        guard let xBuffer = Fp16Buffer.make(context.device, values: x),
              let weightBuffer = context.device.makeBuffer(
                  bytes: weight.map(Quantization.bf16Bits),
                  length: weight.count * MemoryLayout<UInt16>.stride,
                  options: .storageModeShared),
              let outBuffer = Fp16Buffer.make(context.device, count: rows * Self.bundle),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            throw CocoaError(.fileReadUnknown)
        }
        rms.encodeBF16WGrouped(commandBuffer: commandBuffer,
                               x: xBuffer, weight: weightBuffer, out: outBuffer,
                               groupSize: UInt32(Self.hidden),
                               groups: UInt32(Self.hcCount),
                               rows: rows, eps: Self.eps)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        let actual = Fp16Buffer.read(outBuffer, count: rows * Self.bundle)
        // Gate: the FP16 store is the floor. Normed values sit around 1, where
        // FP16 resolution is ~5e-4, so anything under 2e-3 is store noise.
        Self.report("group rmsnorm rows=\(rows)", actual, expected, bound: 2e-3)
    }

    /// The grouped kernel must not silently become the per-head one: the weight
    /// is indexed by (stream, channel) over the whole bundle, so giving stream 1
    /// a different gain than stream 0 has to change stream 1's output only.
    @Test func groupRMSNormUsesAPerStreamWeightSlice() throws {
        let context = try MetalContext()
        let rms = try RMSNorm(context: context)
        var rng = SplitMix64(seed: 0xF1A5_0002)
        let x = Self.roundedFP16((0..<Self.bundle).map { _ in rng.uniform(-3.0, 3.0) })
        var weight = [Float](repeating: 1, count: Self.bundle)
        for d in 0..<Self.hidden { weight[Self.hidden + d] = 2 }   // stream 1 only

        guard let xBuffer = Fp16Buffer.make(context.device, values: x),
              let weightBuffer = context.device.makeBuffer(
                  bytes: weight.map(Quantization.bf16Bits),
                  length: weight.count * MemoryLayout<UInt16>.stride,
                  options: .storageModeShared),
              let outBuffer = Fp16Buffer.make(context.device, count: Self.bundle),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            throw CocoaError(.fileReadUnknown)
        }
        rms.encodeBF16WGrouped(commandBuffer: commandBuffer,
                               x: xBuffer, weight: weightBuffer, out: outBuffer,
                               groupSize: UInt32(Self.hidden),
                               groups: UInt32(Self.hcCount),
                               rows: 1, eps: Self.eps)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let out = Fp16Buffer.read(outBuffer, count: Self.bundle)

        // Streams 0 and 1 hold different data, so compare each against its own
        // unweighted normalization instead of against each other.
        let expected = FlashNextHyperConnectionReference.groupRMSNorm(
            x, offset: 0, weight: weight, g: Self.geometry)
        Self.report("per-stream weight slice", out, expected, bound: 2e-3)
        let ones = [Float](repeating: 1, count: Self.bundle)
        let unweighted = FlashNextHyperConnectionReference.groupRMSNorm(
            x, offset: 0, weight: ones, g: Self.geometry)
        for d in 0..<Self.hidden {
            #expect(abs(out[Self.hidden + d] - 2 * unweighted[Self.hidden + d]) < 4e-3,
                    "stream 1 must carry the doubled gain at channel \(d)")
        }
    }

    // MARK: - BF16 mat-vec

    @Test("BF16 mat-vec matches the reference",
          arguments: [(320, 10240), (10240, 320), (4, 10240), (2560, 2560)])
    func bf16MatVecMatchesTheReference(shape: (rows: Int, cols: Int)) throws {
        let context = try MetalContext()
        let matVec = try FlashNextMatVec(context: context,
                                         int4: try DequantInt4GEMV(context: context))
        var rng = SplitMix64(seed: 0xF1A5_0003 &+ UInt64(shape.rows))
        let w = Self.roundedBF16((0..<(shape.rows * shape.cols)).map { _ in
            rng.uniform(-0.08, 0.08)
        })
        let x = Self.roundedFP16((0..<shape.cols).map { _ in rng.uniform(-1.5, 1.5) })
        let expected = FlashNextRouterReference.matVec(
            w, rows: shape.rows, cols: shape.cols, x: x)

        guard let wBuffer = context.device.makeBuffer(
                  bytes: w.map(Quantization.bf16Bits),
                  length: w.count * MemoryLayout<UInt16>.stride,
                  options: .storageModeShared),
              let xBuffer = Fp16Buffer.make(context.device, values: x),
              let yBuffer = context.device.makeBuffer(
                  length: shape.rows * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            throw CocoaError(.fileReadUnknown)
        }
        matVec.encode(commandBuffer: commandBuffer,
                      matrix: .bf16(buffer: wBuffer, offset: 0),
                      x: xBuffer, y: yBuffer,
                      rows: shape.rows, cols: shape.cols, outputFloat32: true)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        let pointer = yBuffer.contents().bindMemory(to: Float.self,
                                                    capacity: shape.rows)
        let actual = (0..<shape.rows).map { pointer[$0] }
        // FP32 output: only the SIMD-tree summation order differs from the
        // reference's sequential accumulation.
        Self.report("bf16 matvec \(shape.rows)x\(shape.cols)", actual, expected,
               bound: 1e-3)
    }

    // MARK: - Hyper connections, end to end

    @Test("Hyper-connection mix and inject match the reference",
          arguments: [false, true])
    func hyperConnectionMatchesTheReference(int4Weights: Bool) throws {
        let rows = 3
        let context = try MetalContext()
        var rng = SplitMix64(seed: int4Weights ? 0xF1A5_0004 : 0xF1A5_0005)

        let hyper = Self.roundedFP16((0..<(rows * Self.bundle)).map { _ in rng.uniform(-3.0, 3.0) })
        let norm = Self.roundedBF16((0..<Self.bundle).map { _ in rng.uniform(0.4, 1.6) })
        // Both dtype paths: the production install quantizes these to INT4
        // affine group-64, the parity install leaves them BF16. Whichever the
        // GPU reads, the reference is handed the same decoded values.
        let (mixDown, mixDownMatrix) = try Self.projection(
            context, rows: Self.lowRank, cols: Self.bundle, int4: int4Weights, rng: &rng)
        let (mixUp, mixUpMatrix) = try Self.projection(
            context, rows: Self.bundle, cols: Self.lowRank, int4: int4Weights, rng: &rng)
        let (inject, injectMatrix) = try Self.projection(
            context, rows: Self.hcCount, cols: Self.bundle, int4: int4Weights, rng: &rng)

        let expected = FlashNextHyperConnectionReference.gatedResidual(
            hyper,
            .init(norm: norm, mixDown: mixDown, mixUp: mixUp, inject: inject),
            rows: rows, g: Self.geometry)

        let rms = try RMSNorm(context: context)
        let matVec = try FlashNextMatVec(context: context,
                                         int4: try DequantInt4GEMV(context: context))
        let hc = try FlashNextHyperConnections(
            context: context, rms: rms, matVec: matVec,
            hidden: Self.hidden, hcCount: Self.hcCount, lowRank: Self.lowRank,
            eps: Self.eps)
        let scratch = try hc.makeScratch(device: context.device, rows: rows)

        guard let normBuffer = context.device.makeBuffer(
                  bytes: norm.map(Quantization.bf16Bits),
                  length: norm.count * MemoryLayout<UInt16>.stride,
                  options: .storageModeShared),
              let hyperBuffer = Fp16Buffer.make(context.device, values: hyper),
              let mixedBuffer = Fp16Buffer.make(context.device,
                                                count: rows * Self.hidden),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            throw CocoaError(.fileReadUnknown)
        }
        let weights = FlashNextHyperConnections.Weights(
            norm: normBuffer, normOffset: 0,
            mixDown: mixDownMatrix, mixUp: mixUpMatrix, inject: injectMatrix)
        hc.encodeMix(commandBuffer: commandBuffer, weights: weights,
                     scratch: scratch, hyper: hyperBuffer, mixed: mixedBuffer,
                     rows: rows)
        hc.encodeInjectGate(commandBuffer: commandBuffer, weights: weights,
                            scratch: scratch, rows: rows)
        // The inject gate lives in a private buffer; stage it back for reading.
        guard let injectReadback = context.device.makeBuffer(
                  length: rows * Self.hcCount * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw CocoaError(.fileReadUnknown)
        }
        blit.copy(from: scratch.injectGate, sourceOffset: 0,
                  to: injectReadback, destinationOffset: 0,
                  size: rows * Self.hcCount * MemoryLayout<Float>.stride)
        blit.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        let label = int4Weights ? "int4" : "bf16"
        // `mixed` is the block input the whole layer stack rides on. It lands in
        // FP16 after two GEMVs and a sigmoid gate, so a few 1e-3 is the store
        // floor at these magnitudes; the assertion is that nothing structural
        // (a wrong division, a swapped stream) is hiding under it.
        Self.report("hc mixed (\(label))",
               Fp16Buffer.read(mixedBuffer, count: rows * Self.hidden),
               expected.mixed, bound: 5e-3)
        let injectPointer = injectReadback.contents().bindMemory(
            to: Float.self, capacity: rows * Self.hcCount)
        Self.report("hc inject (\(label))",
               (0..<(rows * Self.hcCount)).map { injectPointer[$0] },
               try #require(expected.inject), bound: 2e-3)
    }

    @Test func injectAccumulateMatchesTheReference() throws {
        let rows = 3
        let context = try MetalContext()
        var rng = SplitMix64(seed: 0xF1A5_0006)
        let hyper = Self.roundedFP16((0..<(rows * Self.bundle)).map { _ in rng.uniform(-4.0, 4.0) })
        let block = Self.roundedFP16((0..<(rows * Self.hidden)).map { _ in rng.uniform(-2.0, 2.0) })
        let gate = (0..<(rows * Self.hcCount)).map { _ in rng.uniform(0.05, 1.95) }
        let expected = FlashNextHyperConnectionReference.injectBlock(
            hyper, block: block, inject: gate, rows: rows, g: Self.geometry)

        let rms = try RMSNorm(context: context)
        let matVec = try FlashNextMatVec(context: context,
                                         int4: try DequantInt4GEMV(context: context))
        let hc = try FlashNextHyperConnections(
            context: context, rms: rms, matVec: matVec,
            hidden: Self.hidden, hcCount: Self.hcCount, lowRank: Self.lowRank,
            eps: Self.eps)
        let scratch = try hc.makeScratch(device: context.device, rows: rows)

        guard let hyperBuffer = Fp16Buffer.make(context.device, values: hyper),
              let blockBuffer = Fp16Buffer.make(context.device, values: block),
              let gateStaging = context.device.makeBuffer(
                  bytes: gate, length: gate.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let commandBuffer = context.queue.makeCommandBuffer(),
              let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw CocoaError(.fileReadUnknown)
        }
        blit.copy(from: gateStaging, sourceOffset: 0,
                  to: scratch.injectGate, destinationOffset: 0,
                  size: gate.count * MemoryLayout<Float>.stride)
        blit.endEncoding()
        hc.encodeInjectAccumulate(commandBuffer: commandBuffer, scratch: scratch,
                                  hyper: hyperBuffer, block: blockBuffer, rows: rows)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        Self.report("hc inject accumulate",
               Fp16Buffer.read(hyperBuffer, count: rows * Self.bundle),
               expected, bound: 8e-3)
    }

    @Test func tileEmbeddingCopiesEachStreamExactly() throws {
        let rows = 3
        let context = try MetalContext()
        var rng = SplitMix64(seed: 0xF1A5_0007)
        let embedding = Self.roundedFP16((0..<(rows * Self.hidden)).map { _ in
            rng.uniform(-2.0, 2.0)
        })

        let rms = try RMSNorm(context: context)
        let matVec = try FlashNextMatVec(context: context,
                                         int4: try DequantInt4GEMV(context: context))
        let hc = try FlashNextHyperConnections(
            context: context, rms: rms, matVec: matVec,
            hidden: Self.hidden, hcCount: Self.hcCount, lowRank: Self.lowRank,
            eps: Self.eps)

        guard let embedBuffer = Fp16Buffer.make(context.device, values: embedding),
              let hyperBuffer = Fp16Buffer.make(context.device, count: rows * Self.bundle),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            throw CocoaError(.fileReadUnknown)
        }
        hc.encodeTileEmbedding(commandBuffer: commandBuffer,
                               embedding: embedBuffer, hyper: hyperBuffer, rows: rows)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        // A tile, not an interleave: stream j is a bit-exact copy of the row.
        let out = Fp16Buffer.read(hyperBuffer, count: rows * Self.bundle)
        for t in 0..<rows {
            let row = Array(embedding[(t * Self.hidden)..<((t + 1) * Self.hidden)])
            for j in 0..<Self.hcCount {
                let start = t * Self.bundle + j * Self.hidden
                let slice = Array(out[start..<(start + Self.hidden)])
                #expect(slice == row, "row \(t) stream \(j) is not an exact copy")
            }
        }
    }

    // MARK: - Helpers

    /// Build a projection in the requested dtype, returning both the values the
    /// GPU will decode and the matrix handle that points at them.
    private static func projection(_ context: MetalContext,
                                   rows: Int, cols: Int, int4: Bool,
                                   rng: inout SplitMix64) throws
        -> (values: [Float], matrix: FlashNextWeightMatrix) {
        let raw = (0..<(rows * cols)).map { _ in rng.uniform(-0.08, 0.08) }
        if !int4 {
            let values = Self.roundedBF16(raw)
            guard let buffer = context.device.makeBuffer(
                      bytes: values.map(Quantization.bf16Bits),
                      length: values.count * MemoryLayout<UInt16>.stride,
                      options: .storageModeShared) else {
                throw CocoaError(.fileReadUnknown)
            }
            return (values, .bf16(buffer: buffer, offset: 0))
        }
        precondition(cols % Quantization.groupSize == 0,
                     "INT4 group-64 needs a row length the group size divides")
        var packed: [UInt8] = []
        var scales: [UInt16] = []
        var biases: [UInt16] = []
        var decoded: [Float] = []
        for r in 0..<rows {
            let row = Array(raw[(r * cols)..<((r + 1) * cols)])
            let encoded = Quantization.quantizeInt4Affine(row)
            packed.append(contentsOf: encoded.packed)
            scales.append(contentsOf: encoded.scales)
            biases.append(contentsOf: encoded.biases)
            decoded.append(contentsOf:
                Quantization.dequantizeInt4Affine(encoded, n: cols))
        }
        guard let weightBuffer = context.device.makeBuffer(
                  bytes: packed, length: packed.count,
                  options: .storageModeShared),
              let scaleBuffer = context.device.makeBuffer(
                  bytes: scales, length: scales.count * MemoryLayout<UInt16>.stride,
                  options: .storageModeShared),
              let biasBuffer = context.device.makeBuffer(
                  bytes: biases, length: biases.count * MemoryLayout<UInt16>.stride,
                  options: .storageModeShared) else {
            throw CocoaError(.fileReadUnknown)
        }
        return (decoded, .int4(weights: weightBuffer, weightsOffset: 0,
                               scales: scaleBuffer, scalesOffset: 0,
                               biases: biasBuffer, biasesOffset: 0))
    }

    private static func roundedFP16(_ values: [Float]) -> [Float] {
        values.map { Float(Float16($0)) }
    }

    private static func roundedBF16(_ values: [Float]) -> [Float] {
        values.map { Quantization.bf16ToFloat(Quantization.bf16Bits($0)) }
    }

    /// Assert within `bound` and print the observed margin, so the gate's
    /// headroom is a number in the log rather than an inference.
    private static func report(_ label: String, _ actual: [Float],
                               _ expected: [Float], bound: Float) {
        #expect(actual.count == expected.count, "\(label): length mismatch")
        var maxAbs: Float = 0
        var worstAt = 0
        for i in 0..<min(actual.count, expected.count) {
            let d = abs(actual[i] - expected[i])
            if d > maxAbs { maxAbs = d; worstAt = i }
        }
        print("flashnext \(label): maxAbs=\(maxAbs) at \(worstAt) "
                + "(gate \(bound), \(actual.count) elements)")
        #expect(maxAbs < bound,
                Comment(rawValue: "\(label): maxAbs \(maxAbs) exceeds \(bound) "
                            + "at element \(worstAt)"))
    }
}
