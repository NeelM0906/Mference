import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

/// LUT4 routed-expert decode against `GGUFLUT4MoEReference` at the Unsloth
/// install's shape (D 2560, F 640, top-10), for every gate/up × down mix the
/// install uses. Slots are built on the CPU from `GGUFLUT4Reference`, so this
/// is independent of the expand kernel. Weights are exact, so the bound is the
/// fp16 activation / single-reduction tolerance.
@Suite struct GGUFLUT4MoETests {
    static let d = 2560, f = 640, topK = 10, distinctExperts = 3

    struct Mix: CustomTestStringConvertible, Sendable {
        let gateUp: GGUFLUT4.SourceType, down: GGUFLUT4.SourceType
        var testDescription: String { "\(gateUp) + \(down)" }
    }
    static let mixes = [Mix(gateUp: .iq2_s, down: .iq4_nl), Mix(gateUp: .iq3_xxs, down: .iq4_nl),
                        Mix(gateUp: .iq2_s, down: .q2_0), Mix(gateUp: .iq4_xs, down: .iq4_nl)]

    /// Random blocks with a positive fp16 scale in [5e-4, 3e-3].
    static func blocks(_ t: GGUFLUT4.SourceType, weights: Int, rng: inout SplitMix64) -> [UInt8] {
        var b = randomGGUFBlocks(t, count: weights / t.blockWeights, rng: &rng)
        for i in 0..<(weights / t.blockWeights) {
            let bits = Float16(rng.uniform(5e-4, 3e-3)).bitPattern
            b[i * t.blockBytes] = UInt8(bits & 0xFF)
            b[i * t.blockBytes + 1] = UInt8(bits >> 8)
        }
        return b
    }

    /// Appends one projection's planar sections at a 16-aligned offset;
    /// returns (index, d, m) offsets.
    static func append(_ parts: GGUFLUT4Reference.Parts, to slot: inout [UInt8]) -> (UInt32, UInt32, UInt32) {
        let p = GGUFLUT4ExpandTests.planar(parts)
        slot += [UInt8](repeating: 0, count: (16 - slot.count % 16) % 16)
        let i = slot.count; slot += p.indices
        let d = slot.count; slot += p.d
        let m = slot.count; slot += p.m
        return (UInt32(i), UInt32(d), UInt32(m))
    }

    @Test(arguments: mixes)
    func decodeMatchesReference(mix: Mix) throws {
        let ctx = try MetalContext()
        let kernel = try GGUFLUT4MoE(context: ctx)
        var rng = SeedTree(0x4C55_544D).key("lut4-moe-\(mix.gateUp.rawValue)-\(mix.down.rawValue)")
        let n = Self.d * Self.f

        var slots = [[UInt8]](), experts = [GGUFLUT4MoEReference.Expert]()
        var offsets: GGUFLUT4MoE.ExpertOffsets?
        for _ in 0..<Self.distinctExperts {
            var slot = [UInt8]()
            func projection(_ t: GGUFLUT4.SourceType) -> ((UInt32, UInt32, UInt32), [Float]) {
                let raw = Self.blocks(t, weights: n, rng: &rng)
                let parts = raw.withUnsafeBytes { GGUFLUT4Reference.decompose(t, $0) }
                return (Self.append(parts, to: &slot), GGUFLUT4Reference.dequantize(t, parts))
            }
            let (g, gw) = projection(mix.gateUp)
            let (u, uw) = projection(mix.gateUp)
            let (dn, dw) = projection(mix.down)
            offsets = GGUFLUT4MoE.ExpertOffsets(gateIndex: g.0, gateD: g.1, gateM: g.2,
                                                upIndex: u.0, upD: u.1, upM: u.2,
                                                downIndex: dn.0, downD: dn.1, downM: dn.2)
            slots.append(slot)
            experts.append(GGUFLUT4MoEReference.Expert(gate: gw, up: uw, down: dw))
        }
        let stride = (slots[0].count + 16_383) / 16_384 * 16_384
        let slab = try #require(ctx.device.makeBuffer(length: stride * slots.count, options: .storageModeShared))
        for (i, s) in slots.enumerated() {
            s.withUnsafeBytes { slab.contents().advanced(by: i * stride).copyMemory(from: $0.baseAddress!, byteCount: s.count) }
        }
        let routedSlots = (0..<Self.topK).map { $0 % Self.distinctExperts }
        let blobs = routedSlots.map { (buffer: slab, offset: $0 * stride) }
        let routed = routedSlots.map { experts[$0] }

        func halfBuffer(_ v: [Float16]) throws -> MTLBuffer {
            try #require(v.withUnsafeBytes { ctx.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) })
        }
        let x = (0..<Self.d).map { _ in Float16(rng.uniform(-1, 1)) }
        let w = (0..<Self.topK).map { _ in Float16(rng.uniform(0.01, 0.2)) }
        let residual = (0..<Self.d).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let xBuf = try halfBuffer(x), wBuf = try halfBuffer(w), rBuf = try halfBuffer(residual)
        let actsBuf = try halfBuffer([Float16](repeating: .nan, count: Self.topK * Self.f))
        let yBuf = try halfBuffer([Float16](repeating: .nan, count: Self.d))

        let cb1 = try #require(ctx.queue.makeCommandBuffer())
        kernel.encodePhase1(commandBuffer: cb1, blobs: blobs, offsets: offsets!, gateUp: mix.gateUp,
                            x: xBuf, acts: actsBuf, d: UInt32(Self.d), f: UInt32(Self.f))
        cb1.commit(); cb1.waitUntilCompleted()
        #expect(cb1.status == .completed)
        let gpuActs = Array(UnsafeBufferPointer(start: actsBuf.contents().bindMemory(to: Float16.self, capacity: Self.topK * Self.f),
                                                count: Self.topK * Self.f))
        let refActs = GGUFLUT4MoEReference.phase1(experts: routed, x: x, d: Self.d, f: Self.f)
        let e1 = RelError.compute(actual: gpuActs.map(Float.init), reference: refActs.map(Float.init))
        #expect(e1 < Tolerance.fp16Reduction, "phase 1 rel error \(e1)")

        // Phase 2 consumes the GPU's own acts, so it is checked in isolation.
        let cb2 = try #require(ctx.queue.makeCommandBuffer())
        kernel.encodePhase2(commandBuffer: cb2, blobs: blobs, offsets: offsets!, down: mix.down,
                            acts: actsBuf, routingWeights: wBuf, residual: rBuf, y: yBuf,
                            d: UInt32(Self.d), f: UInt32(Self.f))
        cb2.commit(); cb2.waitUntilCompleted()
        #expect(cb2.status == .completed)
        let gpuY = Array(UnsafeBufferPointer(start: yBuf.contents().bindMemory(to: Float16.self, capacity: Self.d), count: Self.d))
        let refY = GGUFLUT4MoEReference.phase2(experts: routed, acts: gpuActs, routingWeights: w,
                                               residual: residual, d: Self.d, f: Self.f)
        let e2 = RelError.compute(actual: gpuY.map(Float.init), reference: refY.map(Float.init))
        #expect(e2 < Tolerance.fp16Reduction, "phase 2 rel error \(e2)")
    }
}
