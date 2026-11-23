import Testing
import Foundation
import Metal
@testable import Mference
import MferenceValidationSupport

/// Attention at the model's own geometry: the 1,024-token window over a
/// wrapped ring sized `window + chunk`, and a global visit past the
/// 1,024-distance bias extent with tau active, prefill and decode, against
/// the CPU reference. Full tier (`MFERENCE_TEST_TIER=full`): the reference
/// is slow in debug builds.
@Suite(.serialized) struct Inkling2ProductionShapeTests {
    typealias S = Inkling2TestSupport

    @Test(.fullTier, arguments: [false, true])
    func attentionAtModelShape(global: Bool) throws {
        let ctx = try MetalContext()
        let library = try Inkling2KernelLibrary(context: ctx)
        let prefill = try Inkling2PrefillAttention(library: library)
        let decode = try Inkling2DecodeAttention(library: library)
        var rng = SeedTree(global ? 0x7b02 : 0x7b01).key("ink2-production")
        let rows = global ? 16 : 40
        let start = global ? 8_300 : 2_500             // global: tau > 1 past 8,192
        let mask: Inkling2AttentionMask = global ? .globalDefault : .slidingDefault
        let capacity = global ? 0 : 1024 + 64
        let positions = start + rows
        let first = global ? 0 : start - 1023
        let hkvd = 1024
        let q = S.halfRounded(S.random(rows * 3072, &rng, -0.4, 0.4))
        let k = S.halfRounded(S.random((positions - first) * hkvd, &rng, -0.4, 0.4))
        let v = S.halfRounded(S.random((positions - first) * hkvd, &rng, -1, 1))
        let rel = S.random(rows * 384, &rng, -0.5, 0.5)
        let table = S.random(1024 * 16, &rng, -0.5, 0.5)
        let sink = S.random(rows * 24, &rng, -1, 2)
        let gate = S.random(rows * 3072, &rng, -3, 3)
        let slots = global ? positions : capacity
        var kc = [Float16](repeating: 0, count: slots * hkvd), vc = kc
        for p in first..<positions {
            let slot = global ? p : p % capacity
            for i in 0..<hkvd {
                kc[slot * hkvd + i] = Float16(k[(p - first) * hkvd + i])
                vc[slot * hkvd + i] = Float16(v[(p - first) * hkvd + i])
            }
        }
        let cache = Inkling2KVView(keys: .init(S.buffer(ctx.device, halves: kc)),
                                   values: .init(S.buffer(ctx.device, halves: vc)), ringCapacity: capacity)
        let qBuffer = S.buffer(ctx.device, halves: q.map { Float16($0) })
        let relBuffer = S.buffer(ctx.device, floats: rel)
        let tableBuffer = S.buffer(ctx.device, floats: table)
        let sinkBuffer = S.buffer(ctx.device, floats: sink)
        let gateBuffer = S.buffer(ctx.device, floats: gate)
        let out = ctx.device.makeBuffer(length: rows * 3072 * 2, options: .storageModeShared)!
        let last = ctx.device.makeBuffer(length: 3072 * 2, options: .storageModeShared)!
        let keys = Inkling2DecodeAttention.keyRange(position: positions - 1, mask: mask).count
        let partials = ctx.device.makeBuffer(length: Inkling2DecodeAttention.partialsLength(keys: keys),
                                             options: .storageModeShared)!
        var reference = Inkling2AttentionRef.attend(
            q: q, k: k, v: v, rel: rel, table: table, sink: sink, queryCount: rows, startPosition: start,
            keyBase: first, window: global ? nil : 1024, logScale: global ? .init() : nil)
        reference = Inkling2AttentionRef.sigmoidGate(reference, gateLogits: gate)
        for variant in Inkling2PrefillAttention.Variant.allCases where prefill.supports(variant) {
            // NaN everywhere first, so a variant that skips rows or heads cannot pass on leftovers.
            let halves = out.contents().bindMemory(to: UInt16.self, capacity: rows * 3072)
            for i in 0..<(rows * 3072) { halves[i] = 0x7E00 }
            S.run(ctx) { cb in
                prefill.encode(commandBuffer: cb, .init(
                    queries: .init(qBuffer), cache: cache, relative: .init(relBuffer), table: .init(tableBuffer),
                    sink: .init(sinkBuffer), gate: .init(gateBuffer), output: .init(out), rows: rows,
                    startPosition: start, mask: mask), variant: variant)
            }
            #expect(RelError.computeStrict(actual: S.readHalves(out, count: rows * 3072), reference: reference)
                    < Tolerance.fp16Reduction, "variant \(variant)")
        }
        S.run(ctx) { cb in
            decode.encode(commandBuffer: cb, .init(
                queries: .init(qBuffer, offset: (rows - 1) * 3072 * 2), cache: cache, relative: .init(relBuffer, offset: (rows - 1) * 384 * 4),
                table: .init(tableBuffer), sink: .init(sinkBuffer, offset: (rows - 1) * 24 * 4),
                gate: .init(gateBuffer, offset: (rows - 1) * 3072 * 4), output: .init(last), partials: .init(partials),
                position: positions - 1, mask: mask))
        }
        #expect(RelError.computeStrict(actual: S.readHalves(last, count: 3072),
                                 reference: Array(reference[((rows - 1) * 3072)...])) < Tolerance.fp16Reduction)
    }
}
