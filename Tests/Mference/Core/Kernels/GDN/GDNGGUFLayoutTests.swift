import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

/// `GDN.CheckpointLayout.gguf` (WS4 U4): llama.cpp's converter stores the
/// linear-attention value heads in tiled order (`_reorder_v_heads`: grouped
/// head `k * r + j` becomes `j * Hk + k`) and the decay as `A = -exp(A_log)` in
/// FP32. Under the GGUF constants the recurrence kernels must compute exactly
/// the vendor result, permuted: same inputs reordered into tiled order with
/// `A` precomputed, and the outputs and state read back through the inverse
/// permutation. Checked for the fused Hv=48 decode, the unfused decode step and
/// the prefill recurrence, at the production geometry.
@Suite struct GDNGGUFLayoutTests {
    private static let config = LinearAttentionConfig(
        numKHeads: 16, numVHeads: 48, keyHeadDim: 128, valueHeadDim: 128,
        convKernelSize: 4)

    /// Tiled position of grouped value head `g`.
    static func tiled(_ g: Int) -> Int {
        let r = config.numVHeads / config.numKHeads
        return (g % r) * config.numKHeads + g / r
    }

    /// Reorders per-head blocks of `width` elements from grouped to tiled order.
    static func toTiled<T>(_ grouped: [T], width: Int) -> [T] {
        var out = grouped
        for g in 0..<config.numVHeads {
            let t = tiled(g)
            for i in 0..<width { out[t * width + i] = grouped[g * width + i] }
        }
        return out
    }

    struct Inputs {
        var convOut: [Float]   // [rows, qkvDim]
        var z: [Float]         // [valueDim]
        var a: [Float]         // [rows, Hv]
        var b: [Float]         // [rows, Hv]
        var aLog: [Float]      // [Hv], BF16 values
        var dtBias: [Float]    // [Hv], BF16 values
        var norm: [Float]      // [Dv]
        var state: [Float]     // [Hv, Dv, Dk]
    }

    static func makeInputs(rows: Int, seed: UInt64) -> Inputs {
        let c = config
        let qkvDim = 2 * c.numKHeads * c.keyHeadDim + c.numVHeads * c.valueHeadDim
        var rng = SplitMix64(seed: seed)
        func half(_ lo: Float, _ hi: Float) -> Float { Float(Float16(rng.uniform(lo, hi))) }
        func bf16(_ lo: Float, _ hi: Float) -> Float {
            Quantization.bf16ToFloat(Quantization.bf16Bits(rng.uniform(lo, hi)))
        }
        return Inputs(
            convOut: (0..<(rows * qkvDim)).map { _ in half(-1.5, 1.5) },
            z: (0..<(c.numVHeads * c.valueHeadDim)).map { _ in half(-3, 3) },
            a: (0..<(rows * c.numVHeads)).map { _ in half(-2, 2) },
            b: (0..<(rows * c.numVHeads)).map { _ in half(-2, 2) },
            aLog: (0..<c.numVHeads).map { _ in bf16(-2, 0.5) },
            dtBias: (0..<c.numVHeads).map { _ in bf16(-1, 1) },
            norm: (0..<c.valueHeadDim).map { _ in bf16(0.5, 1.5) },
            state: (0..<(c.numVHeads * c.valueHeadDim * c.keyHeadDim)).map { _ in rng.uniform(-0.05, 0.05) })
    }

    /// The same inputs as the GGUF install stores them.
    static func ggufInputs(_ v: Inputs, rows: Int) -> (inputs: Inputs, negA: [Float]) {
        let c = config
        let qk = 2 * c.numKHeads * c.keyHeadDim
        let qkvDim = qk + c.numVHeads * c.valueHeadDim
        var out = v
        for t in 0..<rows {
            let row = Array(v.convOut[(t * qkvDim)..<((t + 1) * qkvDim)])
            let tiledV = toTiled(Array(row[qk...]), width: c.valueHeadDim)
            out.convOut.replaceSubrange((t * qkvDim + qk)..<((t + 1) * qkvDim), with: tiledV)
            out.a.replaceSubrange((t * c.numVHeads)..<((t + 1) * c.numVHeads),
                                  with: toTiled(Array(v.a[(t * c.numVHeads)..<((t + 1) * c.numVHeads)]), width: 1))
            out.b.replaceSubrange((t * c.numVHeads)..<((t + 1) * c.numVHeads),
                                  with: toTiled(Array(v.b[(t * c.numVHeads)..<((t + 1) * c.numVHeads)]), width: 1))
        }
        out.z = toTiled(v.z, width: c.valueHeadDim)
        out.dtBias = toTiled(v.dtBias, width: 1)
        out.state = toTiled(v.state, width: c.valueHeadDim * c.keyHeadDim)
        return (out, toTiled(v.aLog.map { -expf($0) }, width: 1))
    }

    struct Buffers {
        let convOut, z, a, b, aLog, dtBias, norm, state, out, y: MTLBuffer
    }

    static func buffers(_ ctx: MetalContext, _ v: Inputs, negA: [Float]?, rows: Int) throws -> Buffers {
        let d = ctx.device
        func bf16(_ values: [Float]) -> MTLBuffer? {
            d.makeBuffer(bytes: values.map(Quantization.bf16Bits),
                         length: values.count * 2, options: .storageModeShared)
        }
        func f32(_ values: [Float]) -> MTLBuffer? {
            d.makeBuffer(bytes: values, length: values.count * 4, options: .storageModeShared)
        }
        let valueDim = config.numVHeads * config.valueHeadDim
        return Buffers(
            convOut: try #require(Fp16Buffer.make(d, values: v.convOut)),
            z: try #require(Fp16Buffer.make(d, values: v.z)),
            a: try #require(Fp16Buffer.make(d, values: v.a)),
            b: try #require(Fp16Buffer.make(d, values: v.b)),
            aLog: try #require(negA.map(f32) ?? bf16(v.aLog)),
            dtBias: try #require(bf16(v.dtBias)),
            norm: try #require(bf16(v.norm)),
            state: try #require(f32(v.state)),
            out: try #require(Fp16Buffer.make(d, count: valueDim)),
            y: try #require(Fp16Buffer.make(d, count: rows * valueDim)))
    }

    static func readFloats(_ buffer: MTLBuffer, count: Int) -> [Float] {
        Array(UnsafeBufferPointer(start: buffer.contents().bindMemory(to: Float.self, capacity: count),
                                  count: count))
    }

    enum Path: String, CaseIterable, CustomTestStringConvertible, Sendable {
        case fusedDecode, unfusedDecode, prefill
        var testDescription: String { rawValue }
    }

    @Test(arguments: Path.allCases)
    func ggufLayoutIsThePermutedVendorResult(path: Path) throws {
        let ctx = try MetalContext()
        let c = Self.config
        let rows = path == .prefill ? 3 : 1
        let vendor = Self.makeInputs(rows: rows, seed: 0x4747_5546 &+ UInt64(rows))
        let (gguf, negA) = Self.ggufInputs(vendor, rows: rows)
        let vendorGDN = try GDN(context: ctx, config: c, outputGate: .sigmoid)
        let ggufGDN = try GDN(context: ctx, config: c, outputGate: .sigmoid, checkpointLayout: .gguf)
        let vb = try Self.buffers(ctx, vendor, negA: nil, rows: rows)
        let gb = try Self.buffers(ctx, gguf, negA: negA, rows: rows)
        let cb = try #require(ctx.queue.makeCommandBuffer())
        for (gdn, buf) in [(vendorGDN, vb), (ggufGDN, gb)] {
            switch path {
            case .fusedDecode:
                let fused = gdn.encodeDeltaGatedDecode(
                    commandBuffer: cb, convOut: buf.convOut, aProj: buf.a, bProj: buf.b,
                    aLog: buf.aLog, aLogOffset: 0, dtBias: buf.dtBias, dtBiasOffset: 0,
                    state: buf.state, z: buf.z, weight: buf.norm, weightOffset: 0, out: buf.out)
                #expect(fused)
            case .unfusedDecode:
                gdn.encodeDeltaStepDecode(
                    commandBuffer: cb, convOut: buf.convOut, aProj: buf.a, bProj: buf.b,
                    aLog: buf.aLog, aLogOffset: 0, dtBias: buf.dtBias, dtBiasOffset: 0,
                    state: buf.state, y: buf.y)
            case .prefill:
                gdn.encodeDeltaStepPrefill(
                    commandBuffer: cb, convOut: buf.convOut, aProj: buf.a, bProj: buf.b,
                    aLog: buf.aLog, aLogOffset: 0, dtBias: buf.dtBias, dtBiasOffset: 0,
                    state: buf.state, y: buf.y, rows: rows)
            }
        }
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)

        let valueDim = c.numVHeads * c.valueHeadDim
        let vendorOut: [Float], ggufOut: [Float]
        if path == .fusedDecode {
            vendorOut = Fp16Buffer.read(vb.out, count: valueDim)
            ggufOut = Fp16Buffer.read(gb.out, count: valueDim)
        } else {
            vendorOut = Fp16Buffer.read(vb.y, count: rows * valueDim)
            ggufOut = Fp16Buffer.read(gb.y, count: rows * valueDim)
        }
        var expected = [Float]()
        for t in 0..<rows {
            expected += Self.toTiled(Array(vendorOut[(t * valueDim)..<((t + 1) * valueDim)]), width: c.valueHeadDim)
        }
        // A = -exp(A_log) is precomputed on the CPU and the vendor kernel takes
        // the GPU's exp, so the two agree to the exp rounding, not bit for bit.
        let outError = RelError.compute(actual: ggufOut, reference: expected)
        let stateCount = c.numVHeads * c.valueHeadDim * c.keyHeadDim
        let stateError = RelError.compute(
            actual: Self.readFloats(gb.state, count: stateCount),
            reference: Self.toTiled(Self.readFloats(vb.state, count: stateCount), width: c.valueHeadDim * c.keyHeadDim))
        print("GDN GGUF layout \(path): output rel error \(outError), state rel error \(stateError)")
        #expect(outError < 1e-3, "output off by \(outError)")
        #expect(stateError < 1e-5, "state off by \(stateError)")
        // The grouped read of tiled inputs must be wrong: the constants reached the compiler.
        let ungrouped = RelError.compute(actual: ggufOut, reference: Array(vendorOut))
        #expect(ungrouped > 1e-2, "tiled output equals the grouped one; the layout constant had no effect")
    }
}
