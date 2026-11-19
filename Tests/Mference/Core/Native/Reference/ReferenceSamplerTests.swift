import Foundation
import Metal
import Testing
@testable import Mference

/// Reference sampler semantics (llama.cpp `common_sampler` order).
@Suite("ReferenceSamplerTests", .serialized)
struct ReferenceSamplerTests {
    @Test func mt19937MatchesTheStandard() {
        var rng = Native.ReferenceMT19937(seed: 5489)
        #expect(rng.next() == 3_499_211_612)
        for _ in 0..<9998 { _ = rng.next() }
        #expect(rng.next() == 4_123_659_995)
        var a = Native.ReferenceMT19937(seed: 42), b = Native.ReferenceMT19937(seed: 42)
        let lo = Double(b.next()), hi = Double(b.next())
        #expect(a.nextCanonical() == (lo + hi * 4_294_967_296) / 18_446_744_073_709_551_616)
    }

    static func params(temperature: Float = 0, topK: Int? = nil, topP: Float? = nil, minP: Float = 0,
                       repetition: Float = 1, presence: Float = 0, frequency: Float = 0, lastN: Int = 64,
                       seed: UInt64? = 1, bias: [Int32: Float] = [:]) -> Native.SamplingParams {
        Native.SamplingParams(temperature: temperature, topK: topK, topP: topP, minP: minP,
                              repetitionPenalty: repetition, presencePenalty: presence, frequencyPenalty: frequency,
                              repeatLastN: lastN, seed: seed, logitBias: bias)
    }

    static func sample(_ s: Native.ReferenceSampler, _ logits: [Float]) -> Int32 {
        logits.withUnsafeBufferPointer { s.sample(logits: $0) }
    }

    @Test func greedyPenaltiesAndBias() throws {
        let s = try Native.ReferenceSampler(context: nil)
        s.reset(params: Self.params())
        #expect(Self.sample(s, [1, 3, 3, 2]) == 1)
        #expect(s.canChooseOnGPU)
        s.reset(params: Self.params(repetition: 2))
        #expect(!s.canChooseOnGPU)
        s.accept(1)
        // 3 / 2 = 1.5 < 2.9
        #expect(Self.sample(s, [1, 3, 2.9, -1]) == 2)
        s.reset(params: Self.params(presence: 0.5, frequency: 0.25))
        s.accept(2); s.accept(2)
        // 3 - (2 * 0.25 + 0.5) = 2 < 2.5
        #expect(Self.sample(s, [1, 0, 3, 2.5]) == 3)
        s.reset(params: Self.params(bias: [3: -.infinity]))
        #expect(Self.sample(s, [1, 0, 2, 5]) == 2)
        // A window of 1 forgets older tokens.
        s.reset(params: Self.params(repetition: 10, lastN: 1))
        s.accept(0); s.accept(1)
        #expect(Self.sample(s, [5, 4.9, 0]) == 0)
    }

    @Test func truncationAndSeededDraws() throws {
        let s = try Native.ReferenceSampler(context: nil)
        let logits: [Float] = [0.1, 2, 1.9, -3, 0.5]
        s.reset(params: Self.params(temperature: 1, topK: 1))
        for _ in 0..<5 { #expect(Self.sample(s, logits) == 1) }
        s.reset(params: Self.params(temperature: 1, topP: 0.01))
        #expect(Self.sample(s, logits) == 1)
        s.reset(params: Self.params(temperature: 1, minP: 0.99))
        #expect(Self.sample(s, logits) == 1)
        // Seeded draws are reproducible and follow the cumulative rule.
        s.reset(params: Self.params(temperature: 0.7, topK: 3, seed: 1234))
        let first = (0..<20).map { _ in Self.sample(s, logits) }
        s.reset(params: Self.params(temperature: 0.7, topK: 3, seed: 1234))
        #expect((0..<20).map { _ in Self.sample(s, logits) } == first)
        #expect(Set(first).isSubset(of: [1, 2, 4]))
        // Two candidates, by hand: p = softmax([2, 1.9] / 1).
        s.reset(params: Self.params(temperature: 1, topK: 2, seed: 77))
        var rng = Native.ReferenceMT19937(seed: 77)
        for _ in 0..<10 {
            let e0 = Double(expf(0)), e1 = Double(expf(1.9 - 2))
            let pick: Int32 = e0 >= (e0 + e1) * rng.nextCanonical() ? 1 : 2
            #expect(Self.sample(s, logits) == pick)
        }
    }

    @Test func encodeChoiceEqualsSample() throws {
        let ctx = try MetalContext()
        let s = try Native.ReferenceSampler(context: ctx)
        s.reset(params: Self.params())
        var rng = RefTest.SplitMix64(3)
        var logits = (0..<5000).map { _ in rng.unit() }
        logits[4321] = 2
        logits[17] = 2
        let t = RefTest.tensor(ctx.device, floats: logits, rows: 1, cols: logits.count)
        let out = ctx.device.makeBuffer(length: 16, options: .storageModeShared)!
        try RefTest.run(ctx) { try s.encodeChoice($0, logits: t, output: Native.TokenSlot(buffer: out, offset: 4, count: 1)) }
        #expect(out.contents().advanced(by: 4).load(as: UInt32.self) == 17)
        #expect(Self.sample(s, logits) == 17)
    }
}
