import Testing
import Foundation
import Metal
@testable import Mference
import MferenceValidationSupport

/// `Sampler` exercises: greedy=argmax, seeded determinism + per-position
/// reproducibility, top-k / top-p truncation, repetition penalty, temperature
/// spread. Inputs are raw pre-softcap logits
/// (the sampler runs the softcap+softmax front-end itself).
@Suite struct SamplerTests {

    @Test(arguments: [4, 64])
    func llamaMinPExcludesTokensBelowPeakRatio(topK: Int) throws {
        let rig = try Rig(vocab: 128, logitSoftcap: 0)
        var logits = [Float](repeating: -8, count: 128)
        logits[3] = 2; logits[7] = 1.5; logits[9] = 1
        // exp(1.5 - 2) < 0.8: only token 3 survives before temperature.
        for seed in UInt64(1)...32 {
            let config = GenerationConfig(temperature: 2, topK: topK, topP: 1,
                                          minP: 0.8, seed: seed)
            #expect(rig.draw(logits, config: config).id == 3)
        }
    }

    @Test func llamaDefaultPenaltyWindowExpiresOldTokens() throws {
        let rig = try Rig(vocab: 128, logitSoftcap: 0)
        var logits = [Float](repeating: -8, count: 128)
        logits[3] = 2; logits[7] = 1.5
        let config = GenerationConfig(temperature: 0, presencePenalty: 1)
        #expect(rig.draw(logits, config: config, history: [3]).id == 7)
        #expect(rig.draw(logits, config: config,
                         history: [3] + Array(repeating: 9, count: 64)).id == 3)
    }

    /// Reusable rig — one MetalContext + Sampler + buffers, shared across the
    /// many draws a single test makes (avoids recompiling the shader library
    /// per draw).
    private final class Rig {
        let ctx: MetalContext
        let sampler: Sampler
        let vocab: Int
        let logits: MTLBuffer
        let probs: MTLBuffer
        let outToken: MTLBuffer

        init(vocab: Int, logitSoftcap: Float = 30) throws {
            self.ctx = try MetalContext()
            self.sampler = try Sampler(context: ctx, vocab: vocab, logitSoftcap: logitSoftcap)
            self.vocab = vocab
            guard let l = ctx.device.makeBuffer(length: vocab * MemoryLayout<Float16>.size,
                                                options: .storageModeShared),
                  let p = ctx.device.makeBuffer(length: vocab * MemoryLayout<Float16>.size,
                                                options: .storageModeShared),
                  let o = ctx.device.makeBuffer(length: MemoryLayout<UInt32>.size,
                                                options: .storageModeShared) else {
                throw MetalError.noDevice
            }
            self.logits = l; self.probs = p; self.outToken = o
        }

        func writeLogits(_ values: [Float]) {
            let ptr = logits.contents().bindMemory(to: Float16.self, capacity: vocab)
            for i in 0..<vocab { ptr[i] = Float16(values[i]) }
        }

        @discardableResult
        func draw(_ values: [Float], config: GenerationConfig,
                  position: Int = 0, history: [Int32] = []) -> (id: UInt32, path: SamplePath) {
            writeLogits(values)
            let cmd = ctx.queue.makeCommandBuffer()!
            let path = sampler.sample(commandBuffer: cmd, logits: logits, probs: probs,
                                      history: history, config: config,
                                      position: position, outToken: outToken)
            cmd.commit(); cmd.waitUntilCompleted()
            return (outToken.contents().load(as: UInt32.self), path)
        }
    }

    @Test func greedy_picksArgmax() throws {
        let v = 2048
        let rig = try Rig(vocab: v)
        var logits = [Float](repeating: 0.1, count: v)
        logits[1337] = 9.0
        let (id, path) = rig.draw(logits, config: GenerationConfig(temperature: 0))
        #expect(id == 1337, "got \(id)")
        #expect(path == .greedyGPU)
    }

    @Test(arguments: [Float(0), 30], [Float(-1.5), 1.5])
    func presenceActsOnceInPostSoftcapSpace(softcap: Float, presence: Float) throws {
        let rig = try Rig(vocab: 64, logitSoftcap: softcap)
        var logits = [Float](repeating: 0.25, count: 64)
        logits[2] = 4; logits[3] = -4
        logits[4] = 400; logits[5] = -400
        for repetition: Float in [1, 1.25] {
            let config = GenerationConfig(temperature: 0, repetitionPenalty: repetition,
                                          presencePenalty: presence)
            let once = rig.draw(logits, config: config, history: [2, 3, 4, 5])
            #expect(once.path == .hostPenalty)
            let pointer = rig.probs.contents().bindMemory(to: Float16.self, capacity: 64)
            let probabilities = (0..<64).map { pointer[$0] }
            let referenceLogits = logits.enumerated().map { i, raw -> Double in
                let z = Double(Float(Float16(raw)))
                let capped = softcap > 0 ? Double(softcap) * tanh(z / Double(softcap)) : z
                return (2...5).contains(i)
                    ? (capped > 0 ? capped / Double(repetition) : capped * Double(repetition)) - Double(presence)
                    : capped
            }
            let peak = referenceLogits.max()!
            let weights = referenceLogits.map { exp($0 - peak) }
            let sum = weights.reduce(0, +)
            for i in 0..<64 {
                let expected = Float16(weights[i] / sum)
                #expect(abs(Float(pointer[i]) - Float(expected)) <= Float(expected.ulp) * 2)
            }
            // Presence is once per seen token, irrespective of duplicate count.
            let repeated = rig.draw(logits, config: config,
                                    history: [-1, 2, 2, 3, 3, 4, 5, 5, 64, Int32.max])
            #expect(repeated.id == once.id)
            #expect((0..<64).map { pointer[$0] } == probabilities)
        }
    }

    @Test(arguments: [Float(0), 30])
    func presenceChangesSeenTokenSelection(softcap: Float) throws {
        let rig = try Rig(vocab: 64, logitSoftcap: softcap)
        var logits = [Float](repeating: -10, count: 64)
        logits[5] = 5; logits[7] = 4.5
        #expect(rig.draw(logits, config: GenerationConfig(temperature: 0), history: [5]).id == 5)
        let positive = GenerationConfig(temperature: 0, presencePenalty: 1.5)
        #expect(rig.draw(logits, config: positive, history: [5, 5]).id == 7)
        let negative = GenerationConfig(temperature: 0, presencePenalty: -1.5)
        #expect(rig.draw(logits, config: negative, history: [7, 7]).id == 7)
        let empty = rig.draw(logits, config: positive)
        #expect(empty.id == 5)
        #expect(empty.path == .greedyGPU)
    }

    @Test(arguments: [20, 64], [Float(1), 1.3])
    func omittedAndZeroPresencePreserveSeededSequence(topK: Int, repetition: Float) throws {
        let rig = try Rig(vocab: 128)
        let logits = (0..<128).map { Float($0 % 13) / 4 - 1 }
        let omitted = GenerationConfig(temperature: 1, topK: topK, topP: 0.95,
                                       repetitionPenalty: repetition, minP: 0, seed: 777)
        let zero = GenerationConfig(temperature: 1, topK: topK, topP: 0.95,
                                    repetitionPenalty: repetition, presencePenalty: 0,
                                    minP: 0, seed: 777)
        for position in 0..<8 {
            let a = rig.draw(logits, config: omitted, position: position, history: [2, 2, 5])
            let b = rig.draw(logits, config: zero, position: position, history: [2, 2, 5])
            #expect(a.id == b.id)
            #expect(a.path == b.path)
        }
    }

    @Test func samplingConfigValidatesPenaltiesAndMinP() throws {
        for value: Float in [-2, -1.5, 0, 1.5, 2] {
            let config = GenerationConfig(temperature: 0, presencePenalty: value)
            try config.validate()
            #expect(config.isPureGreedy == (value == 0))
        }
        for value: Float in [-2.01, 2.01, .nan, .infinity, -.infinity] {
            #expect(throws: GeneratorError.self) {
                try GenerationConfig(presencePenalty: value).validate()
            }
        }
        for value: Float in [-0.1, 1.01, .nan, .infinity, -.infinity] {
            #expect(throws: GeneratorError.self) {
                try GenerationConfig(minP: value).validate()
            }
        }
        for value: Float in [0, 0.05, 0.5, 1] { try GenerationConfig(minP: value).validate() }
        #expect(GenerationConfig(temperature: 0).isPureGreedy)
        #expect(!GenerationConfig(temperature: 1).isPureGreedy)
        #expect(!GenerationConfig(temperature: 0, repetitionPenalty: 1.1).isPureGreedy)
    }

    @Test func seeded_isDeterministicAtPosition() throws {
        let v = 1024
        let rig = try Rig(vocab: v)
        var rng = SeedTree(0x51A7_1005).key("sampler-position-determinism")
        let logits = (0..<v).map { _ in rng.uniform(-2, 2) }
        let cfg = GenerationConfig(temperature: 1.0, seed: 42)
        let a = rig.draw(logits, config: cfg, position: 3).id
        let b = rig.draw(logits, config: cfg, position: 3).id
        #expect(a == b, "same seed+position gave \(a) vs \(b)")
        #expect(rig.draw(logits, config: cfg, position: 0).path == .gpuSampled)
    }

    @Test func seeded_reproducibleAcrossPositions() throws {
        let v = 1024
        let rig = try Rig(vocab: v)
        var rng = SeedTree(0x51A7_1006).key("sampler-position-replay")
        let logits = (0..<v).map { _ in rng.uniform(-2, 2) }
        let cfg = GenerationConfig(temperature: 1.0, seed: 42)
        let run1 = (0..<5).map { rig.draw(logits, config: cfg, position: $0).id }
        let run2 = (0..<5).map { rig.draw(logits, config: cfg, position: $0).id }
        #expect(run1 == run2, "seed=42 not reproducible: \(run1) vs \(run2)")

        // A different seed should diverge on at least one position.
        let cfg43 = GenerationConfig(temperature: 1.0, seed: 43)
        let run3 = (0..<5).map { rig.draw(logits, config: cfg43, position: $0).id }
        #expect(run3 != run1, "seed 42 and 43 produced identical sequences")
    }

    @Test func topK_restrictsToTop() throws {
        let v = 1024
        let rig = try Rig(vocab: v)
        var logits = [Float](repeating: -8.0, count: v)
        let top: [Int] = [10, 200, 500, 900]
        logits[top[0]] = 4.0; logits[top[1]] = 3.0; logits[top[2]] = 2.0; logits[top[3]] = 1.0
        let topSet = Set(top.map { UInt32($0) })
        for t in 0..<32 {
            let cfg = GenerationConfig(temperature: 1.0, topK: 4, seed: UInt64(t) &+ 1)
            let id = rig.draw(logits, config: cfg, position: t).id
            #expect(topSet.contains(id), "trial \(t): id=\(id) outside top-4")
        }
    }

    @Test func topP_restrictsToNucleus() throws {
        let v = 512
        let rig = try Rig(vocab: v)
        // Two tokens hold ~99% of the mass after softmax.
        var logits = [Float](repeating: -10.0, count: v)
        logits[7] = 6.0
        logits[42] = 5.6
        let nucleus: Set<UInt32> = [7, 42]
        for t in 0..<32 {
            let cfg = GenerationConfig(temperature: 1.0, topP: 0.9, seed: UInt64(t) &+ 1)
            let id = rig.draw(logits, config: cfg, position: t).id
            #expect(nucleus.contains(id), "trial \(t): id=\(id) outside nucleus")
        }
    }

    @Test func repetitionPenalty_suppressesHistory() throws {
        let v = 64
        let rig = try Rig(vocab: v)
        // Large positive logits so penalty 2.0 (logit 8 -> 4) suppresses id 5
        // decisively post-softmax (~0.03x the others); flat logits of 1.0 only
        // scale its mass by ~0.6x, which the statistical bound below cannot
        // separate from noise.
        let logits = [Float](repeating: 8.0, count: v)
        let history: [Int32] = [5, 5, 5]
        var count5 = 0
        let trials = 200
        for t in 0..<trials {
            let cfg = GenerationConfig(temperature: 1.0, repetitionPenalty: 2.0, seed: UInt64(t) &+ 1)
            let (id, path) = rig.draw(logits, config: cfg, position: t, history: history)
            #expect(path == .hostPenalty)
            if id == 5 { count5 += 1 }
        }
        // Uniform would pick 5 about trials/v times; suppression should push it
        // well below that. Generous bound to avoid flakiness.
        let uniformExpect = Double(trials) / Double(v)
        #expect(Double(count5) < 0.5 * uniformExpect, "id 5 chosen \(count5) times (uniform≈\(uniformExpect))")
    }

    /// Saturated-logit suppression: real Gemma 4 raw logits reach the
    /// hundreds, deep in softcap-tanh saturation. The penalty must act on the
    /// post-softcap value — applied to the raw logit it moves the capped
    /// result by ~nothing and the penalty silently no-ops (the repetition-loop
    /// regression this pins).
    @Test func repetitionPenalty_bitesOnSaturatedLogits() throws {
        let v = 64
        let rig = try Rig(vocab: v)
        // All raw logits deep in tanh saturation; id 5 is the model's strong
        // favorite and also the repeated-history token.
        var logits = [Float](repeating: 300.0, count: v)
        logits[5] = 400.0
        let history: [Int32] = [5]
        var count5 = 0
        let trials = 64
        for t in 0..<trials {
            let cfg = GenerationConfig(temperature: 1.0, repetitionPenalty: 1.3, seed: UInt64(t) &+ 1)
            let (id, path) = rig.draw(logits, config: cfg, position: t, history: history)
            #expect(path == .hostPenalty)
            if id == 5 { count5 += 1 }
        }
        // Post-softcap both land near the 30 cap; penalty 1.3 drops id 5 to
        // ~23, ~e^-7 of the others — it should essentially never win. The raw
        // pre-fix math left id 5 the argmax favorite at >half the draws.
        #expect(count5 < trials / 8, "saturated id 5 drawn \(count5)/\(trials) despite penalty")
    }

    /// Temperature spread: a logit sharp enough that greedy would always pick
    /// index 0 should, under raised temperature, distribute mass across many
    /// tokens. Exercised through the enumerated top-k path (`topK == v`).
    /// The `topK == 0` branch is the Gumbel-max fast path (argmax over noised
    /// log-probs), which cannot return an out-of-range id by construction.
    @Test func raisedTemperature_spreadsMass() throws {
        let v = 32
        let rig = try Rig(vocab: v)
        var logits = [Float](repeating: 0.0, count: v)
        logits[0] = 8.0
        var counts = [Int](repeating: 0, count: v)
        let trials = 1600
        for t in 0..<trials {
            // This test isolates temperature: the llama.cpp default filters
            // run first and would intentionally remove this fixture's tail.
            let cfg = GenerationConfig(temperature: 2.0, topK: v, topP: 1,
                                       minP: 0, seed: UInt64(t) &+ 1)
            let raw = rig.draw(logits, config: cfg, position: t).id
            let id = Int(raw)
            guard id >= 0 && id < v else { Issue.record("id \(raw) (0x\(String(raw, radix: 16))) out of range v=\(v)"); continue }
            counts[id] += 1
        }
        let maxShare = Double(counts.max() ?? trials) / Double(trials)
        let distinct = counts.filter { $0 > 0 }.count
        // Greedy would give a top-token share of 1.0; raised temperature must
        // pull it well below that. (Bound kept loose — this asserts spread, not
        // a precise distribution.)
        #expect(maxShare < 0.7, "top token share \(maxShare) — temperature did not spread mass")
        #expect(distinct > v / 4, "only \(distinct)/\(v) tokens drawn — too concentrated")
    }

}
