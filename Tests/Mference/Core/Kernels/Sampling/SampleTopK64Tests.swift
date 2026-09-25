import Metal
import Testing
@testable import Mference

@Suite struct SampleTopK64Tests {
    @Test func truncationDefaultsDoNotDisableGreedyEligibility() {
        let config = GenerationConfig(temperature: 0, topK: 64, topP: 0.95)
        #expect(config.isPureGreedy)
    }

    @Test func generationConfigRejectsSamplerStatesTheKernelCannotHonor() throws {
        #expect(throws: GeneratorError.self) {
            try GenerationConfig(temperature: 1, topK: 257, topP: 0.95).validate()
        }
        #expect(throws: GeneratorError.self) {
            try GenerationConfig(temperature: 1, topK: nil, topP: 0.95).validate()
        }
        try GenerationConfig(temperature: 0, topK: nil, topP: 0.95).validate()
    }

    private final class Rig {
        let context: MetalContext
        let current: Sample
        let candidate: SampleTopK64
        let probs: MTLBuffer
        let currentOutput: MTLBuffer
        let candidateOutput: MTLBuffer
        let vocab: Int

        init(vocab: Int) throws {
            self.context = try MetalContext()
            self.current = try Sample(context: context)
            self.candidate = try SampleTopK64(context: context, vocab: vocab)
            self.vocab = vocab
            guard let probs = context.device.makeBuffer(
                      length: vocab * MemoryLayout<Float16>.stride,
                      options: .storageModeShared),
                  let currentOutput = context.device.makeBuffer(
                      length: MemoryLayout<UInt32>.stride,
                      options: .storageModeShared),
                  let candidateOutput = context.device.makeBuffer(
                      length: MemoryLayout<UInt32>.stride,
                      options: .storageModeShared)
            else {
                throw MetalError.noDevice
            }
            self.probs = probs
            self.currentOutput = currentOutput
            self.candidateOutput = candidateOutput
        }

        func write(_ values: (Int) -> Float) {
            let ptr = probs.contents().bindMemory(to: Float16.self, capacity: vocab)
            for i in 0..<vocab {
                ptr[i] = Float16(values(i))
            }
        }

        func draw(seed: UInt64,
                  temperature: Float = 1.0,
                  topP: Float,
                  minP: Float = 0,
                  topK: UInt32 = 64) -> (current: UInt32, candidate: UInt32) {
            let cb = context.queue.makeCommandBuffer()!
            current.encode(commandBuffer: cb,
                           probs: probs,
                           outToken: currentOutput,
                           v: UInt32(vocab),
                           temperature: temperature,
                           topK: topK,
                           topP: topP,
                           minP: minP,
                           seed: seed)
            candidate.encode(commandBuffer: cb,
                             probs: probs,
                             outToken: candidateOutput,
                             temperature: temperature,
                             topP: topP,
                             minP: minP,
                             topK: topK,
                             seed: seed)
            cb.commit()
            cb.waitUntilCompleted()
            #expect(cb.status == .completed)
            return (currentOutput.contents().load(as: UInt32.self),
                    candidateOutput.contents().load(as: UInt32.self))
        }
    }

    @Test func productionVocabularyMatchesCurrentSampler() throws {
        let rig = try Rig(vocab: 262_144)
        #expect(rig.candidate.scratchBytes == 139_264)
        rig.write { i in
            let mixed = UInt64(i) &* 6364136223846793005 &+ 1442695040888963407
            return Float(UInt32(mixed >> 40) + 1) * (1.0 / 16_777_217.0)
        }

        for temperature: Float in [0.7, 0.85, 1.0] {
            for seed: UInt64 in [1, 2, 0x1234_5678_9ABC_DEF0, UInt64.max] {
                let result = rig.draw(seed: seed, temperature: temperature, topP: 0.95)
                #expect(result.candidate == result.current,
                        "temperature \(temperature), seed \(seed): candidate \(result.candidate), current \(result.current)")
            }
        }
    }

    @Test func tiesAndPartialTailMatchCurrentSampler() throws {
        let rig = try Rig(vocab: 1_003)
        rig.write { _ in 1.0 }

        for seed in UInt64(1)...UInt64(8) {
            let result = rig.draw(seed: seed, topP: 0.95)
            #expect(result.candidate == result.current,
                    "seed \(seed): candidate \(result.candidate), current \(result.current)")
            #expect(result.candidate < 64)
        }
    }

    @Test func topPNormalizesTheTopKSetLikeLlama() throws {
        let rig = try Rig(vocab: 1_003)
        rig.write { _ in 1.0 / 1_003.0 }
        // llama.cpp applies Top-K before Top-P: ceil(64 * .95) == 61.
        var sawBoundary = false
        for seed in UInt64(1)...UInt64(256) {
            let result = rig.draw(seed: seed, topP: 0.95)
            #expect(result.candidate == result.current)
            #expect(result.candidate < 61)
            if result.candidate == 60 { sawBoundary = true }
        }
        #expect(sawBoundary, "Top-P must include the token crossing its threshold")
    }

    @Test(arguments: [UInt32(1), 4, 40, 64], [Float(0), 0.05, 0.5, 1])
    func requestedKAndMinPMatchGeneralSampler(topK: UInt32, minP: Float) throws {
        let rig = try Rig(vocab: 1_003)
        // Exact FP16 values put tokens on both sides of Min-P's inclusive edge.
        rig.write { i in i < 2 ? 1 : (i < 4 ? 0.5 : 0.125) }
        var seen = Set<UInt32>()
        for seed in UInt64(1)...128 {
            let result = rig.draw(seed: seed, temperature: 0.8, topP: 1,
                                  minP: minP, topK: topK)
            #expect(result.candidate == result.current)
            #expect(result.candidate < topK)
            if minP == 1 { #expect(result.candidate < 2) }
            if minP == 0.5 { #expect(result.candidate < 4) }
            seen.insert(result.candidate)
        }
        if topK >= 4 && minP == 0.5 {
            #expect(seen.contains(2) && seen.contains(3), "Min-P must retain equality at its threshold")
        }
        if topK >= 4 && minP == 1 {
            #expect(seen == [0, 1], "all tied maxima must remain eligible")
        }
    }
}
