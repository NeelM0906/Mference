import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

/// The two halves of Flash-Next's GDN branch.
///
/// Flash-Next's GDN block is `Qwen3_5GatedDeltaNet` at Qwen 3.8's exact geometry
/// (Hk 16, Hv 48, Dk 128, Dv 128, conv 4), so the production runner takes the
/// shipped kernels — the fused Hv=48 decode included — with one change: the gated
/// norm's activation on `z` is **sigmoid**, from `output_gate_type`, where Qwen
/// 3.6 and Qwen 3.8 use silu. That switch is a function constant, so the shipped
/// pipelines (built with no constants) compile to exactly the code they did
/// before and only a caller asking for sigmoid gets different arithmetic.
///
/// This suite gates:
///
/// 1. `gdn_gated_norm` with the sigmoid constant against an FP32 CPU reference at
///    the production Hv=48 geometry, and the silu default against *its* reference
///    in the same run — so "the default is unchanged" is measured, not assumed.
/// 2. That the two actually differ, which is the only thing that proves the
///    constant reached the compiler.
/// 3. The fused `gdn_delta_gated_decode_qwen38` under sigmoid against the
///    unfused `gdn_delta_step_decode` + `gdn_gated_norm` pair on identical
///    inputs and identical initial state — the same A/B the fused kernel's silu
///    form is held to.
///
/// The dimension-generic fallback in `flashnext_gdn.metal` is gated where it is
/// actually used: `FlashNextForwardRunnerParityTests` measures its whole block
/// output at layer 0 against the reference forward from a bit-exact stream.
@Suite struct FlashNextGDNGateTests {

    /// Qwen 3.8 / Flash-Next geometry. The fused decode kernel only exists for
    /// this shape.
    private static let config = LinearAttentionConfig(
        numKHeads: 16, numVHeads: 48, keyHeadDim: 128, valueHeadDim: 128,
        convKernelSize: 4)
    private static let eps: Float = 1e-6

    private static func silu(_ x: Float) -> Float { x / (1 + expf(-x)) }
    private static func sigmoid(_ x: Float) -> Float { 1 / (1 + expf(-x)) }

    /// `out = rmsnorm(y; weight) * gate(z)`, per value head, in FP32.
    private static func gatedNormReference(y: [Float], z: [Float],
                                           weight: [Float],
                                           gate: (Float) -> Float) -> [Float] {
        let hv = config.numVHeads
        let dv = config.valueHeadDim
        var out = [Float](repeating: 0, count: hv * dv)
        for h in 0..<hv {
            let base = h * dv
            var sumsq: Float = 0
            for i in 0..<dv { sumsq += y[base + i] * y[base + i] }
            let inv = 1 / sqrtf(sumsq / Float(dv) + eps)
            for i in 0..<dv {
                out[base + i] = y[base + i] * inv * weight[i] * gate(z[base + i])
            }
        }
        return out
    }

    // MARK: - 1 & 2. The gated norm's activation

    @Test func gatedNormHonoursTheOutputGateConstant() throws {
        let context = try MetalContext()
        let hv = Self.config.numVHeads
        let dv = Self.config.valueHeadDim
        var rng = SplitMix64(seed: 0x6D_0001)
        let y = (0..<(hv * dv)).map { _ in Float(Float16(rng.uniform(-2.0, 2.0))) }
        let z = (0..<(hv * dv)).map { _ in Float(Float16(rng.uniform(-3.0, 3.0))) }
        let weight = (0..<dv).map { _ in
            Quantization.bf16ToFloat(Quantization.bf16Bits(rng.uniform(0.5, 1.5)))
        }

        func run(_ gate: GDN.OutputGate) throws -> [Float] {
            let gdn = try GDN(context: context, config: Self.config,
                              outputGate: gate)
            guard let yBuf = Fp16Buffer.make(context.device, values: y),
                  let zBuf = Fp16Buffer.make(context.device, values: z),
                  let wBuf = context.device.makeBuffer(
                      bytes: weight.map(Quantization.bf16Bits),
                      length: dv * MemoryLayout<UInt16>.stride,
                      options: .storageModeShared),
                  let out = Fp16Buffer.make(context.device, count: hv * dv),
                  let cb = context.queue.makeCommandBuffer() else {
                throw CocoaError(.fileReadUnknown)
            }
            gdn.encodeGatedNorm(commandBuffer: cb, y: yBuf, z: zBuf,
                                weight: wBuf, weightOffset: 0, out: out)
            cb.commit()
            cb.waitUntilCompleted()
            #expect(cb.error == nil)
            return Fp16Buffer.read(out, count: hv * dv)
        }

        let sigmoidOut = try run(.sigmoid)
        let siluOut = try run(.silu)
        let sigmoidRef = Self.gatedNormReference(y: y, z: z, weight: weight,
                                                 gate: Self.sigmoid)
        let siluRef = Self.gatedNormReference(y: y, z: z, weight: weight,
                                              gate: Self.silu)

        let sigmoidError = RelError.compute(actual: sigmoidOut, reference: sigmoidRef)
        let siluError = RelError.compute(actual: siluOut, reference: siluRef)
        print("flashnext GDN gated norm at Hv=48: sigmoid relative error "
                + "\(sigmoidError), silu relative error \(siluError)")
        #expect(sigmoidError < Tolerance.fp16ChainedReduction,
                "sigmoid gated norm off by \(sigmoidError)")
        #expect(siluError < Tolerance.fp16ChainedReduction,
                "the silu default moved: off by \(siluError)")
        // Without this the whole suite would pass on a constant that never
        // reached the compiler.
        #expect(sigmoidOut != siluOut,
                "the output-gate constant had no effect on the kernel")
    }

    // MARK: - 3. The fused Hv=48 decode under sigmoid

    /// The fused kernel folds the recurrence and the gated norm into one
    /// dispatch. Under the sigmoid constant it must still agree with the unfused
    /// pair — same inputs, same initial state, same activation.
    @Test func fusedHv48DecodeUnderSigmoidMatchesTheUnfusedPair() throws {
        let context = try MetalContext()
        let c = Self.config
        let qkvDim = 2 * c.numKHeads * c.keyHeadDim + c.numVHeads * c.valueHeadDim
        let valueDim = c.numVHeads * c.valueHeadDim
        var rng = SplitMix64(seed: 0x6D_0002)

        // `convOut` is post-conv, post-SiLU, so its values are already the
        // kernel's own input domain; q/k are l2-normed inside.
        let convOut = (0..<qkvDim).map { _ in Float(Float16(rng.uniform(-1.5, 1.5))) }
        let z = (0..<valueDim).map { _ in Float(Float16(rng.uniform(-3.0, 3.0))) }
        let a = (0..<c.numVHeads).map { _ in Float(Float16(rng.uniform(-2.0, 2.0))) }
        let b = (0..<c.numVHeads).map { _ in Float(Float16(rng.uniform(-2.0, 2.0))) }
        let aLog = (0..<c.numVHeads).map { _ in rng.uniform(-2.0, 0.5) }
        let dtBias = (0..<c.numVHeads).map { _ in rng.uniform(-1.0, 1.0) }
        let normW = (0..<c.valueHeadDim).map { _ in rng.uniform(0.5, 1.5) }
        // A non-zero starting state, so the decay and the rank-one update both
        // matter rather than cancelling out of a zeroed buffer.
        let stateCount = c.numVHeads * c.valueHeadDim * c.keyHeadDim
        let state = (0..<stateCount).map { _ in rng.uniform(-0.05, 0.05) }

        let gdn = try GDN(context: context, config: c, outputGate: .sigmoid)
        func bf16Buffer(_ values: [Float]) -> MTLBuffer? {
            context.device.makeBuffer(bytes: values.map(Quantization.bf16Bits),
                                      length: values.count * MemoryLayout<UInt16>.stride,
                                      options: .storageModeShared)
        }
        func stateBuffer() -> MTLBuffer? {
            context.device.makeBuffer(bytes: state,
                                      length: stateCount * MemoryLayout<Float>.stride,
                                      options: .storageModeShared)
        }
        guard let convBuf = Fp16Buffer.make(context.device, values: convOut),
              let zBuf = Fp16Buffer.make(context.device, values: z),
              let aBuf = Fp16Buffer.make(context.device, values: a),
              let bBuf = Fp16Buffer.make(context.device, values: b),
              let aLogBuf = bf16Buffer(aLog),
              let dtBuf = bf16Buffer(dtBias),
              let normBuf = bf16Buffer(normW),
              let fusedState = stateBuffer(),
              let pairState = stateBuffer(),
              let fusedOut = Fp16Buffer.make(context.device, count: valueDim),
              let pairY = Fp16Buffer.make(context.device, count: valueDim),
              let pairOut = Fp16Buffer.make(context.device, count: valueDim),
              let cb = context.queue.makeCommandBuffer() else {
            Issue.record("buffer allocation failed")
            return
        }

        let fused = gdn.encodeDeltaGatedDecode(
            commandBuffer: cb, convOut: convBuf, aProj: aBuf, bProj: bBuf,
            aLog: aLogBuf, aLogOffset: 0, dtBias: dtBuf, dtBiasOffset: 0,
            state: fusedState, z: zBuf, weight: normBuf, weightOffset: 0,
            out: fusedOut)
        #expect(fused, Comment(rawValue: "the fused Hv=48 decode kernel did not "
                                   + "apply at its own geometry"))
        gdn.encodeDeltaStepDecode(
            commandBuffer: cb, convOut: convBuf, aProj: aBuf, bProj: bBuf,
            aLog: aLogBuf, aLogOffset: 0, dtBias: dtBuf, dtBiasOffset: 0,
            state: pairState, y: pairY)
        gdn.encodeGatedNorm(commandBuffer: cb, y: pairY, z: zBuf,
                            weight: normBuf, weightOffset: 0, out: pairOut)
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)

        let fusedValues = Fp16Buffer.read(fusedOut, count: valueDim)
        let pairValues = Fp16Buffer.read(pairOut, count: valueDim)
        let error = RelError.compute(actual: fusedValues, reference: pairValues)
        print("flashnext fused Hv=48 decode under sigmoid vs the unfused pair: "
                + "relative error \(error) (max-abs "
                + "\(RelError.maxAbsDiff(fusedValues, pairValues)))")
        // The fused kernel keeps `y` in threadgroup memory where the pair round
        // trips it through an FP16 buffer, so they agree to that rounding rather
        // than bit for bit.
        #expect(error < Tolerance.fp16ChainedReduction,
                "fused and unfused Hv=48 decode diverge by \(error)")

        // Both must have advanced the state identically, or the next token would
        // diverge even if this one did not.
        let fusedStateOut = fusedState.contents()
            .bindMemory(to: Float.self, capacity: stateCount)
        let pairStateOut = pairState.contents()
            .bindMemory(to: Float.self, capacity: stateCount)
        var worstState: Float = 0
        for i in 0..<stateCount {
            worstState = max(worstState, abs(fusedStateOut[i] - pairStateOut[i]))
        }
        print("flashnext fused Hv=48 decode: recurrent state max-abs difference "
                + "\(worstState)")
        #expect(worstState == 0,
                "the fused and unfused decode left different recurrent states")
    }
}
