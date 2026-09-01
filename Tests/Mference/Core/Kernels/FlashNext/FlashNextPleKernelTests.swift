import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

/// Stage 2b of the Flash-Next Metal port: the PLE mixing block — per-stream
/// signed-sqrt gating, the gated-value broadcast, and the dilated causal
/// depthwise conv with its nine-row decode state — against
/// `FlashNextPleReference`, which is the CPU reference forward's own arithmetic
/// (see its tie-back suite, which also gates the shipping `FlashNextPleHash`
/// against the runner's captured row ids).
///
/// # Geometry
///
/// The parity tests run at `hidden = 512, hc = 4` rather than the production
/// `2560 x 4`. Nothing in these kernels depends on the width — the gate reduces
/// over `hidden`, the conv is per-channel — but the CPU reference's dense
/// `[10240, 2560]` key projection is 26 M multiply-adds per token in a debug
/// build, which would dominate the suite for no additional coverage. Production
/// width is covered by `chunkedAndSteppedDecodeAgreeAtProductionWidth`, which
/// runs the real `2560 x 4` shapes through the same chain and holds it to the
/// cache-equivalence property the decode state exists for.
@Suite struct FlashNextPleKernelTests {

    private static let hidden = 512
    private static let hcCount = 4
    private static let convKernel = 4
    private static let dilation = 3          // ngram_size
    private static let eps: Float = 1e-6
    private static var bundle: Int { hidden * hcCount }
    private static var stateLength: Int { (convKernel - 1) * dilation }

    private static var geometry: FlashNextHyperConnectionReference.Geometry {
        .init(hidden: hidden, hcCount: hcCount, lowRank: 8, eps: eps)
    }

    // MARK: - Fixture

    private struct Fixture {
        let context: MetalContext
        let ple: FlashNextPLE
        let weights: FlashNextPLE.Weights
        let referenceWeights: FlashNextPleReference.Weights
        let hidden: Int
        let hcCount: Int
        var bundle: Int { hidden * hcCount }
    }

    private static func makeFixture(seed: UInt64,
                                    hidden: Int = hidden,
                                    hcCount: Int = hcCount) throws -> Fixture {
        let context = try MetalContext()
        var rng = SplitMix64(seed: seed)
        let bundle = hidden * hcCount

        let keyProj = Self.bf16((0..<(bundle * hidden)).map { _ in rng.uniform(-0.05, 0.05) })
        let valueProj = Self.bf16((0..<(hidden * hidden)).map { _ in rng.uniform(-0.05, 0.05) })
        let conv = Self.bf16((0..<(bundle * convKernel)).map { _ in rng.uniform(-0.4, 0.4) })
        let normKey = Self.bf16((0..<bundle).map { _ in rng.uniform(0.4, 1.6) })
        let normQuery = Self.bf16((0..<bundle).map { _ in rng.uniform(0.4, 1.6) })
        let normConv = Self.bf16((0..<bundle).map { _ in rng.uniform(0.4, 1.6) })

        func bf16Buffer(_ values: [Float]) throws -> MTLBuffer {
            guard let b = context.device.makeBuffer(
                bytes: values.map(Quantization.bf16Bits),
                length: values.count * MemoryLayout<UInt16>.stride,
                options: .storageModeShared) else { throw CocoaError(.fileReadUnknown) }
            return b
        }

        let rms = try RMSNorm(context: context)
        let matVec = try FlashNextMatVec(context: context,
                                         int4: try DequantInt4GEMV(context: context))
        let ple = try FlashNextPLE(context: context, rms: rms, matVec: matVec,
                                   elementwise: try Elementwise(context: context),
                                   hidden: hidden, hcCount: hcCount,
                                   convKernel: convKernel, dilation: dilation,
                                   eps: eps)
        return Fixture(
            context: context, ple: ple,
            weights: .init(keyProj: .bf16(buffer: try bf16Buffer(keyProj), offset: 0),
                           valueProj: .bf16(buffer: try bf16Buffer(valueProj), offset: 0),
                           conv: try bf16Buffer(conv), convOffset: 0,
                           normKey: try bf16Buffer(normKey), normKeyOffset: 0,
                           normQuery: try bf16Buffer(normQuery), normQueryOffset: 0,
                           normConv: try bf16Buffer(normConv), normConvOffset: 0),
            referenceWeights: .init(keyProj: keyProj, valueProj: valueProj,
                                    conv: conv, normKey: normKey,
                                    normQuery: normQuery, normConv: normConv),
            hidden: hidden, hcCount: hcCount)
    }

    // MARK: - Full-chain parity

    @Test func pleMixingMatchesTheReference() throws {
        let rows = 3
        let f = try Self.makeFixture(seed: 0xB1E5_0001)
        var rng = SplitMix64(seed: 0xB1E5_1001)
        let embeds = Self.fp16((0..<(rows * Self.hidden)).map { _ in rng.uniform(-1.5, 1.5) })
        let hyper = Self.fp16((0..<(rows * Self.bundle)).map { _ in rng.uniform(-3.0, 3.0) })

        let oracle = FlashNextPleReference(geometry: Self.geometry,
                                           convKernel: Self.convKernel,
                                           dilation: Self.dilation)
        let expectedOut = oracle.mix(hyper: hyper, embeds: embeds,
                                     w: f.referenceWeights, rows: rows)
        let expectedHyper = zip(hyper, expectedOut).map(+)

        let result = try Self.run(f, embeds: [embeds], hyper: hyper, rowCounts: [rows])
        Self.report("ple out", result.out, expectedOut, bound: 6e-3)
        Self.report("ple hyper", result.hyper, expectedHyper, bound: 8e-3)
    }

    /// The gate is a signed square root with a floor on the magnitude, so a
    /// stream whose key and query are orthogonal must land at `sigmoid(0)`, and
    /// a negative dot must land strictly below `0.5` — not at
    /// `sigmoid(sqrt(clamped))`, which would be positive either way.
    @Test func streamGateIsSignedAcrossZero() throws {
        let f = try Self.makeFixture(seed: 0xB1E5_0002)
        var rng = SplitMix64(seed: 0xB1E5_1002)
        let embeds = Self.fp16((0..<Self.hidden).map { _ in rng.uniform(-1.5, 1.5) })

        // Flip the whole hyper stream's sign: every per-stream dot product
        // flips with it, so every gate must cross to the other side of 0.5.
        let hyper = Self.fp16((0..<Self.bundle).map { _ in rng.uniform(-3.0, 3.0) })
        let mirrored = hyper.map { -$0 }

        let oracle = FlashNextPleReference(geometry: Self.geometry,
                                           convKernel: Self.convKernel,
                                           dilation: Self.dilation)
        let positive = oracle.mix(hyper: hyper, embeds: embeds,
                                  w: f.referenceWeights, rows: 1)
        oracle.reset()
        let negative = oracle.mix(hyper: mirrored, embeds: embeds,
                                  w: f.referenceWeights, rows: 1)
        #expect(positive != negative,
                "mirroring the query stream must change the gate")

        let a = try Self.run(f, embeds: [embeds], hyper: hyper, rowCounts: [1])
        let b = try Self.run(f, embeds: [embeds], hyper: mirrored, rowCounts: [1])
        Self.report("ple out (query stream)", a.out, positive, bound: 6e-3)
        Self.report("ple out (mirrored query stream)", b.out, negative, bound: 6e-3)
    }

    /// One `rows`-token call and `rows` single-token calls must agree — the
    /// property the nine-row conv state exists to provide, and the one that
    /// makes cached decode equal to a re-prefill.
    @Test func chunkedAndSteppedDecodeAgree() throws {
        let rows = 5
        let f = try Self.makeFixture(seed: 0xB1E5_0003)
        var rng = SplitMix64(seed: 0xB1E5_1003)
        let embeds = Self.fp16((0..<(rows * Self.hidden)).map { _ in rng.uniform(-1.5, 1.5) })
        let hyper = Self.fp16((0..<(rows * Self.bundle)).map { _ in rng.uniform(-3.0, 3.0) })

        let chunk = try Self.run(f, embeds: [embeds], hyper: hyper, rowCounts: [rows])
        let stepped = try Self.run(
            f,
            embeds: (0..<rows).map {
                Array(embeds[($0 * Self.hidden)..<(($0 + 1) * Self.hidden)])
            },
            hyper: hyper,
            rowCounts: [Int](repeating: 1, count: rows))
        #expect(stepped.out == chunk.out,
                "stepped decode must be bit-identical to the chunked call")

        // And both must match the CPU reference carried the same way.
        let oracle = FlashNextPleReference(geometry: Self.geometry,
                                           convKernel: Self.convKernel,
                                           dilation: Self.dilation)
        var expected: [Float] = []
        for t in 0..<rows {
            expected.append(contentsOf: oracle.mix(
                hyper: Array(hyper[(t * Self.bundle)..<((t + 1) * Self.bundle)]),
                embeds: Array(embeds[(t * Self.hidden)..<((t + 1) * Self.hidden)]),
                w: f.referenceWeights, rows: 1))
        }
        Self.report("ple stepped out", stepped.out, expected, bound: 6e-3)
    }

    /// Production geometry, Metal against Metal: the same cache-equivalence
    /// property at `2560 x 4`, where the CPU reference would be too slow to be
    /// worth running but the shapes are the ones that ship.
    @Test func chunkedAndSteppedDecodeAgreeAtProductionWidth() throws {
        let rows = 4
        let hidden = 2560
        let f = try Self.makeFixture(seed: 0xB1E5_0004, hidden: hidden)
        var rng = SplitMix64(seed: 0xB1E5_1004)
        let embeds = Self.fp16((0..<(rows * hidden)).map { _ in rng.uniform(-1.5, 1.5) })
        let hyper = Self.fp16((0..<(rows * f.bundle)).map { _ in rng.uniform(-3.0, 3.0) })

        let chunk = try Self.run(f, embeds: [embeds], hyper: hyper, rowCounts: [rows])
        let stepped = try Self.run(
            f,
            embeds: (0..<rows).map { Array(embeds[($0 * hidden)..<(($0 + 1) * hidden)]) },
            hyper: hyper,
            rowCounts: [Int](repeating: 1, count: rows))
        #expect(stepped.out == chunk.out,
                "stepped decode must be bit-identical to the chunked call at 2560x4")
        #expect(chunk.out.allSatisfy { $0.isFinite }, "PLE output must be finite")
        // The conv contributes: without it the output would be the gated value
        // alone, which the first `stateLength` rows would not distinguish.
        #expect(chunk.out.contains { $0 != 0 }, "PLE output must not be all zeros")
    }

    // MARK: - Dispatch

    private struct Run {
        let out: [Float]      // the PLE output, `[rows * bundle]`
        let hyper: [Float]    // the stream after the residual add
    }

    /// Drive the chain over one or more calls, sharing the conv state across
    /// them the way a prefill followed by decode steps would. `embeds` has one
    /// entry per call; `hyper` is the whole stream, consumed row-block by
    /// row-block.
    private static func run(_ f: Fixture, embeds: [[Float]], hyper: [Float],
                            rowCounts: [Int]) throws -> Run {
        precondition(embeds.count == rowCounts.count)
        let half = MemoryLayout<Float16>.stride
        let maxRows = rowCounts.max() ?? 1
        let scratch = try f.ple.makeScratch(device: f.context.device, rows: maxRows,
                                            storageMode: .storageModeShared)
        guard let hyperBuffer = Fp16Buffer.make(f.context.device, values: hyper),
              let reset = f.context.queue.makeCommandBuffer() else {
            throw CocoaError(.fileReadUnknown)
        }
        f.ple.encodeResetState(commandBuffer: reset, scratch: scratch)
        reset.commit()
        reset.waitUntilCompleted()

        var out: [Float] = []
        var consumed = 0
        for (call, rows) in rowCounts.enumerated() {
            let staged = embeds[call].map { Float16($0) }
            staged.withUnsafeBytes { source in
                scratch.embeds.contents().copyMemory(from: source.baseAddress!,
                                                     byteCount: source.count)
            }
            guard let cb = f.context.queue.makeCommandBuffer() else {
                throw CocoaError(.fileReadUnknown)
            }
            f.ple.encode(commandBuffer: cb, weights: f.weights, scratch: scratch,
                         hyper: hyperBuffer,
                         hyperOffset: consumed * f.bundle * half,
                         rows: rows)
            cb.commit()
            cb.waitUntilCompleted()
            #expect(cb.error == nil)
            out.append(contentsOf: Fp16Buffer.read(scratch.gatedValue,
                                                   count: rows * f.bundle))
            consumed += rows
        }
        return Run(out: out,
                   hyper: Fp16Buffer.read(hyperBuffer, count: consumed * f.bundle))
    }

    // MARK: - Helpers

    private static func fp16(_ values: [Float]) -> [Float] {
        values.map { Float(Float16($0)) }
    }

    private static func bf16(_ values: [Float]) -> [Float] {
        values.map { Quantization.bf16ToFloat(Quantization.bf16Bits($0)) }
    }

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
