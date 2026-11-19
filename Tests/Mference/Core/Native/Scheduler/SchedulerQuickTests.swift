import Foundation
import Metal
import Testing
@testable import Mference

/// W5's end-to-end acceptance on Quick (a model run, AGENTS.md and spec
/// section 10): with all-reference model components, the pipelined scheduler
/// with the chain sampler produces output byte-identical to the reference
/// scheduler with the reference sampler, greedy (GPU choice, pipelined) and
/// sampled (CPU path). Skipped unless `MFERENCE_NATIVE_QUICK_PACK` is set;
/// `MFERENCE_NATIVE_W5_OUT=<dir>` also writes `w5-quick-identity.json`.
@Suite("SchedulerQuickTests", .serialized)
struct SchedulerQuickTests {
    static let env = ProcessInfo.processInfo.environment

    struct Result: Codable {
        var prompt: String
        var params: String
        var referenceTokens: [Int32]
        var pipelinedTokens: [Int32]
        var referenceStop: String
        var pipelinedStop: String
        var identical: Bool
        var referenceDecodeSeconds: Double
        var pipelinedDecodeSeconds: Double
        var pipelinedMedianGapMicroseconds: Double
        var pipelinedMedianSchedulerCPUMicroseconds: Double
    }

    @Test(.enabled(if: env["MFERENCE_NATIVE_QUICK_PACK"] != nil))
    func pipelinedSchedulerIsByteIdenticalOnQuick() async throws {
        let ctx = try MetalContext()
        let packURL = URL(fileURLWithPath: try #require(Self.env["MFERENCE_NATIVE_QUICK_PACK"]))
        let store = try Native.ReferenceWeightStore(directory: packURL, device: ctx.device)
        let tokenizer = try store.tokenizer.makeTokenizer()
        let stops = tokenizer.endOfGenerationTokenIDs
        let reference = try Native.ReferenceStepSchedulerFactory().make(
            model: try ReferenceQuickTests.makeModel(ctx: ctx, packURL: packURL),
            sampler: try Native.ReferenceSampler(context: ctx), strategy: Native.ReferencePlainDecoding(),
            queue: ctx.queue)
        let pipelined = try Native.PipelinedStepSchedulerFactory().make(
            model: try ReferenceQuickTests.makeModel(ctx: ctx, packURL: packURL),
            sampler: try Native.ChainSampler(context: ctx), strategy: Native.PlainDecoding(), queue: ctx.queue)
        let greedy = SchedulerPipelinedTests.greedy
        let sampled = Native.SamplingParams(temperature: 0.7, topK: 20, topP: 0.95, minP: 0, repetitionPenalty: 1.05,
                                            presencePenalty: 0, frequencyPenalty: 0, repeatLastN: 64, seed: 42)
        let cases: [(String, String, Native.SamplingParams, Int)] = ReferenceQuickTests.prompts.map {
            ($0, "greedy", greedy, 32)
        } + [(ReferenceQuickTests.prompts[1], "t0.7-k20-p0.95-r1.05-seed42", sampled, 24)]
        var results: [Result] = []
        for (prompt, name, params, maxNew) in cases {
            let tokens = try tokenizer.encode(prompt, addSpecial: true, parseSpecial: true)
            let job = Native.GenerationJob(promptTokens: tokens, params: params, maxNewTokens: maxNew,
                                           stopTokens: stops, reusePrefix: false, prefillChunk: 64)
            let a = try await reference.run(job) { _ in }
            let b = try await pipelined.run(job) { _ in }
            let text = String(decoding: b.tokens.flatMap { tokenizer.piece($0, special: false) }, as: UTF8.self)
            print("w5 quick [\(name)] \(prompt.debugDescription) -> \(text.debugDescription) "
                  + "identical \(a.tokens == b.tokens && a.stopReason == b.stopReason); "
                  + "decode reference \(a.timings.decodeSeconds) s, pipelined \(b.timings.decodeSeconds) s")
            #expect(a.tokens == b.tokens && a.stopReason == b.stopReason, "\(name) \(prompt.debugDescription)")
            #expect(!a.tokens.isEmpty)
            results.append(Result(prompt: prompt, params: name, referenceTokens: a.tokens, pipelinedTokens: b.tokens,
                                  referenceStop: a.stopReason.rawValue, pipelinedStop: b.stopReason.rawValue,
                                  identical: a.tokens == b.tokens && a.stopReason == b.stopReason,
                                  referenceDecodeSeconds: a.timings.decodeSeconds,
                                  pipelinedDecodeSeconds: b.timings.decodeSeconds,
                                  pipelinedMedianGapMicroseconds: b.timings.medianGPUGapMicroseconds,
                                  pipelinedMedianSchedulerCPUMicroseconds: b.timings.medianSchedulerCPUMicroseconds))
        }
        if let out = Self.env["MFERENCE_NATIVE_W5_OUT"] {
            let url = URL(fileURLWithPath: out)
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(results).write(to: url.appendingPathComponent("w5-quick-identity.json"))
        }
    }
}
