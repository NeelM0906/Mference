import Foundation
import Metal
import Testing
@testable import Mference

/// The fast sampler against llama.cpp's own chain (the pinned fixture R
/// generated with `Scripts/native/reference/run_llama_sampler_cases.sh`) and
/// against the frozen `ReferenceSampler` on generated cases covering every
/// stage and their combinations, ties, -inf and NaN rows. Model-free.
@Suite("SamplingChainSamplerTests", .serialized)
struct SamplingChainSamplerTests {
    static func params(temperature: Float = 0, topK: Int? = nil, topP: Float? = nil, minP: Float = 0,
                       repetition: Float = 1, presence: Float = 0, frequency: Float = 0, lastN: Int = 64,
                       seed: UInt64? = 1, bias: [Int32: Float] = [:]) -> Native.SamplingParams {
        Native.SamplingParams(temperature: temperature, topK: topK, topP: topP, minP: minP,
                              repetitionPenalty: repetition, presencePenalty: presence, frequencyPenalty: frequency,
                              repeatLastN: lastN, seed: seed, logitBias: bias)
    }

    static func sample(_ s: any Native.Sampler, _ logits: [Float]) -> Int32 {
        logits.withUnsafeBufferPointer { s.sample(logits: $0) }
    }

    @Test func mt19937MatchesTheReferenceGenerator() {
        for seed: UInt32 in [0, 1, 5489, 42, 0xFFFF_FFFE] {
            var a = Native.SamplingMT19937(seed: seed), b = Native.ReferenceMT19937(seed: seed)
            for _ in 0..<5000 { #expect(a.next() == b.next()) }
            for _ in 0..<1000 { #expect(a.nextCanonical() == b.nextCanonical()) }
        }
        var std = Native.SamplingMT19937(seed: 5489)
        #expect(std.next() == 3_499_211_612)
        for _ in 0..<9998 { _ = std.next() }
        #expect(std.next() == 4_123_659_995)
    }

    @Test func choicesEqualLlamaCppOnTheReferenceFixture() throws {
        let s = try Native.ChainSampler(context: nil)
        var checked = 0
        for c in ReferenceSamplerTests.llamaCppCases {
            var rng = RefTest.SplitMix64(c.inputSeed)
            let logits = (0..<c.vocab).map { _ in rng.unit() * 6 }
            let history = (0..<24).map { _ in Int32(truncatingIfNeeded: rng.next() % UInt64(c.vocab)) }
            // As in ReferenceSamplerTests: one bias entry per token.
            guard history[0] != 7 else { continue }
            var bias: [Int32: Float] = [:]
            if c.bias == 1 { bias = [history[0]: -.infinity, 7: 1.5] }
            if c.bias == 2 { bias = [history[0]: 2, 7: -3] }
            s.reset(params: Native.SamplingParams(
                temperature: c.temperature, topK: c.topK, topP: c.topP, minP: c.minP, repetitionPenalty: c.repeat,
                presencePenalty: c.presence, frequencyPenalty: c.frequency, repeatLastN: c.lastN,
                seed: UInt64(c.drawSeed), logitBias: bias))
            for t in history { s.accept(t) }
            var chosen: [Int32] = []
            for _ in 0..<c.tokens.count {
                let t = Self.sample(s, logits)
                s.accept(t)
                chosen.append(t)
            }
            #expect(chosen == c.tokens, "case \(String(c.inputSeed, radix: 16)): \(chosen) vs llama.cpp \(c.tokens)")
            checked += 1
        }
        #expect(checked >= 230)
    }

    /// One generated case: logits (with ties, -inf and outliers by `shape`),
    /// a history and params; both samplers draw `draws` tokens, accepting each.
    static func compare(seed: UInt64, vocab: Int, shape: Int, params p: Native.SamplingParams, draws: Int,
                        fast: Native.ChainSampler, reference: Native.ReferenceSampler) -> (fast: [Int32], ref: [Int32]) {
        var rng = RefTest.SplitMix64(seed)
        var logits = (0..<vocab).map { _ in rng.unit() * 8 }
        switch shape {
        case 1: // heavy ties: values on a coarse grid
            for i in logits.indices { logits[i] = (logits[i] * 2).rounded() / 2 }
        case 2: // a peaked distribution with -inf holes
            for i in logits.indices where rng.next() % 7 == 0 { logits[i] = -.infinity }
            logits[Int(rng.next() % UInt64(vocab))] = 30
        case 3: // every value equal
            for i in logits.indices { logits[i] = 0.25 }
        default: break
        }
        let history = (0..<40).map { _ in Int32(truncatingIfNeeded: rng.next() % UInt64(vocab)) }
        fast.reset(params: p)
        reference.reset(params: p)
        for t in history {
            fast.accept(t)
            reference.accept(t)
        }
        var a: [Int32] = [], b: [Int32] = []
        for _ in 0..<draws {
            let x = sample(fast, logits), y = sample(reference, logits)
            a.append(x)
            b.append(y)
            fast.accept(x)
            reference.accept(y)
        }
        return (a, b)
    }

    @Test func choicesEqualTheReferenceOnGeneratedCases() throws {
        let fast = try Native.ChainSampler(context: nil)
        let reference = try Native.ReferenceSampler(context: nil)
        let temperatures: [Float] = [0, 0.3, 0.7, 1, 1.5]
        let topKs: [Int?] = [nil, 0, 1, 5, 40, 300, 5000]
        let topPs: [Float?] = [nil, 1, 0.5, 0.9, 0.95, 0.999]
        let minPs: [Float] = [0, 0.05, 0.2]
        let lastNs = [0, -1, 1, 16, 64]
        var rng = RefTest.SplitMix64(0x5A3D_1E55)
        var cases = 0
        for i in 0..<600 {
            let vocab = [1, 2, 17, 31, 300, 2000, 4097][i % 7]
            let shape = Int(rng.next() % 4)
            let biasKind = rng.next() % 3
            var bias: [Int32: Float] = [:]
            if biasKind == 1 { bias = [Int32(rng.next() % UInt64(vocab)): -.infinity] }
            if biasKind == 2 { bias = [Int32(rng.next() % UInt64(vocab)): 2.5, Int32(vocab + 3): 9, -1: 1] }
            let p = Self.params(
                temperature: temperatures[Int(rng.next() % 5)], topK: topKs[Int(rng.next() % 7)],
                topP: topPs[Int(rng.next() % 6)], minP: minPs[Int(rng.next() % 3)],
                repetition: [1, 1.1, 1.3, 0.8][Int(rng.next() % 4)], presence: [0, 0.5][Int(rng.next() % 2)],
                frequency: [0, 0.2][Int(rng.next() % 2)], lastN: lastNs[Int(rng.next() % 5)],
                seed: UInt64(rng.next() % 1_000_000), bias: bias)
            let r = Self.compare(seed: rng.next(), vocab: vocab, shape: shape, params: p, draws: 6,
                                 fast: fast, reference: reference)
            #expect(r.fast == r.ref, "case \(i) vocab \(vocab) shape \(shape) \(p): \(r.fast) vs \(r.ref)")
            cases += 1
        }
        #expect(cases == 600)
    }

    /// Quick's vocabulary through every truncation path, including top-p
    /// without top-k (the partial-sort nucleus) and the plain draw.
    @Test func choicesEqualTheReferenceAtQuickVocabulary() throws {
        let fast = try Native.ChainSampler(context: nil)
        let reference = try Native.ReferenceSampler(context: nil)
        let settings: [Native.SamplingParams] = [
            Self.params(),
            Self.params(repetition: 1.1, presence: 0.5, frequency: 0.2, bias: [100: 4]),
            Self.params(temperature: 0.7, topK: 20, topP: 0.95, seed: 7),
            Self.params(temperature: 1, topP: 0.9, seed: 8),
            Self.params(temperature: 1, topP: 0.999, seed: 9),
            Self.params(temperature: 0.6, minP: 0.05, seed: 10),
            Self.params(temperature: 1.2, seed: 11),
            Self.params(temperature: 0.8, topK: 40, minP: 0.1, repetition: 1.3, lastN: -1, seed: 12),
        ]
        for (i, p) in settings.enumerated() {
            for shape in [0, 2] {
                let r = Self.compare(seed: UInt64(1000 + i), vocab: 248_320, shape: shape, params: p, draws: 2,
                                     fast: fast, reference: reference)
                #expect(r.fast == r.ref, "setting \(i) shape \(shape): \(r.fast) vs \(r.ref)")
            }
        }
    }

    @Test func greedyEdgeCases() throws {
        let s = try Native.ChainSampler(context: nil)
        let reference = try Native.ReferenceSampler(context: nil)
        s.reset(params: Self.params())
        reference.reset(params: Self.params())
        var rows: [[Float]] = [
            [1, 3, 3, 2],
            [.nan, 5, 6],
            [1, .nan, 0.5],
            [-.infinity, -.infinity],
            [0, -0.0, 0],
            [Float](repeating: 2, count: 100),
        ]
        // Maxima at and around SIMD block edges, ties across blocks.
        for at in [0, 1, 15, 16, 17, 31, 32, 63, 64, 99] {
            var r = [Float](repeating: -1, count: 100)
            r[at] = 4
            r[min(99, at + 16)] = 4
            rows.append(r)
        }
        var none = [Float](repeating: .nan, count: 40)
        none[0] = 0
        rows.append(none)
        for row in rows {
            #expect(Self.sample(s, row) == Self.sample(reference, row), "\(row.prefix(20))")
        }
    }

    @Test func gpuChoiceRules() throws {
        let s = try Native.ChainSampler(context: nil)
        for p in [Self.params(), Self.params(repetition: 1.1), Self.params(repetition: 1.1, lastN: 0),
                  Self.params(temperature: 0.5), Self.params(bias: [1: 1])] {
            let reference = try Native.ReferenceSampler(context: nil)
            s.reset(params: p)
            reference.reset(params: p)
            #expect(s.canChooseOnGPU == reference.canChooseOnGPU)
        }
    }

    @Test func encodeChoiceEqualsSample() throws {
        let ctx = try MetalContext()
        let s = try Native.ChainSampler(context: ctx)
        s.reset(params: Self.params())
        var rng = RefTest.SplitMix64(11)
        var rows: [[Float]] = []
        for n in [1, 2, 15, 16, 17, 255, 256, 257, 4096, 4097, 70_000, 248_320] {
            rows.append((0..<n).map { _ in rng.unit() })
        }
        var tie = (0..<248_320).map { _ in rng.unit() }
        tie[200_001] = 3
        tie[17] = 3
        tie[248_319] = 3
        rows.append(tie)
        var nanFirst = (0..<5000).map { _ in rng.unit() }
        nanFirst[0] = .nan
        rows.append(nanFirst)
        var nans = (0..<5000).map { _ in rng.unit() }
        for i in stride(from: 1, to: 5000, by: 3) { nans[i] = .nan }
        rows.append(nans)
        rows.append([Float](repeating: -.infinity, count: 9000))
        rows.append([Float](repeating: 1, count: 9000))
        let out = ctx.device.makeBuffer(length: 64, options: .storageModeShared)!
        for (i, row) in rows.enumerated() {
            let t = RefTest.tensor(ctx.device, floats: row, rows: 1, cols: row.count, offset: 256)
            out.contents().storeBytes(of: 0xDEAD_BEEF, toByteOffset: 8, as: UInt32.self)
            try RefTest.run(ctx) {
                try s.encodeChoice($0, logits: t, output: Native.TokenSlot(buffer: out, offset: 8, count: 1))
            }
            let gpu = Int32(bitPattern: out.contents().load(fromByteOffset: 8, as: UInt32.self))
            #expect(gpu == Self.sample(s, row), "row \(i) of \(row.count)")
        }
        // Not greedy: refused.
        s.reset(params: Self.params(temperature: 1))
        let t = RefTest.tensor(ctx.device, floats: [1, 2], rows: 1, cols: 2)
        try RefTest.run(ctx) { enc in
            #expect(throws: Native.NativeError.self) {
                try s.encodeChoice(enc, logits: t, output: Native.TokenSlot(buffer: out, offset: 0, count: 1))
            }
        }
    }

    @Test func penaltyWindowMatchesTheReference() throws {
        // A long accept stream through windows of several sizes.
        for lastN in [1, 2, 7, 64, -1] {
            let s = try Native.ChainSampler(context: nil)
            let reference = try Native.ReferenceSampler(context: nil)
            let p = Self.params(repetition: 1.5, presence: 0.3, frequency: 0.1, lastN: lastN)
            s.reset(params: p)
            reference.reset(params: p)
            var rng = RefTest.SplitMix64(UInt64(lastN + 100))
            let logits = (0..<12).map { _ in rng.unit() }
            for _ in 0..<300 {
                let t = Int32(rng.next() % 12)
                s.accept(t)
                reference.accept(t)
                #expect(Self.sample(s, logits) == Self.sample(reference, logits))
            }
        }
    }
}
