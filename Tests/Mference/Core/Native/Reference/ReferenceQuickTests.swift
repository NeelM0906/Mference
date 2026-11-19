import Foundation
import Metal
import Testing
@testable import Mference

/// Real-model checks of the all-reference `qwen35` path on Quick. Every test
/// here is a model run (AGENTS.md, spec section 10) and is skipped unless its
/// environment variable is set (`ReferenceQuickPackTests` writes the pack):
///
/// - `MFERENCE_NATIVE_QUICK_PACK=<quick.mfpack>`: open the pack, check its
///   config and hashes, and run greedy generation over the smoke prompts;
///   `MFERENCE_NATIVE_REFERENCE_OUT=<dir>` also writes the results as JSON.
@Suite("ReferenceQuickTests", .serialized)
struct ReferenceQuickTests {
    static let env = ProcessInfo.processInfo.environment

    struct SmokeResult: Codable {
        var prompt: String
        var promptTokens: [Int32]
        var generatedTokens: [Int32]
        var text: String
        var stopReason: String
        var prefillSeconds: Double
        var decodeSeconds: Double
        var firstStepTop: [[Float]]
    }

    static let prompts = [
        "The capital of France is",
        "<|im_start|>user\nWrite a haiku about the sea.<|im_end|>\n<|im_start|>assistant\n",
        "def fibonacci(n):",
    ]

    @Test(.enabled(if: env["MFERENCE_NATIVE_QUICK_PACK"] != nil))
    func quickReferenceGreedySmoke() async throws {
        let ctx = try MetalContext()
        let packURL = URL(fileURLWithPath: try #require(Self.env["MFERENCE_NATIVE_QUICK_PACK"]))
        let store = try Native.ReferenceWeightStore(directory: packURL, device: ctx.device, verifyHashes: true)
        #expect(store.config == ReferenceQuickPackTests.quickConfig)
        let tokenizer = try store.tokenizer.makeTokenizer()
        let stops = tokenizer.endOfGenerationTokenIDs
        let model = try Self.makeModel(ctx: ctx, packURL: packURL)
        let scheduler = try Native.ReferenceStepSchedulerFactory().make(
            model: model, sampler: try Native.ReferenceSampler(context: ctx),
            strategy: Native.ReferencePlainDecoding(), queue: ctx.queue)
        var results: [SmokeResult] = []
        for prompt in Self.prompts {
            let tokens = try tokenizer.encode(prompt, addSpecial: true, parseSpecial: true)
            let params = Native.SamplingParams(temperature: 0, topK: nil, topP: nil, minP: 0, repetitionPenalty: 1,
                                               presencePenalty: 0, frequencyPenalty: 0, repeatLastN: 0, seed: nil)
            let outcome = try await scheduler.run(Native.GenerationJob(
                promptTokens: tokens, params: params, maxNewTokens: 32, stopTokens: stops, reusePrefix: false,
                prefillChunk: 64)) { _ in }
            let text = String(decoding: outcome.tokens.flatMap { tokenizer.piece($0, special: false) }, as: UTF8.self)
            print("reference quick: \(prompt.debugDescription) -> \(text.debugDescription) "
                  + "[\(outcome.stopReason)] prefill \(outcome.timings.prefillSeconds) s, "
                  + "decode \(outcome.timings.decodeSeconds) s / \(outcome.timings.decodeSteps) steps")
            #expect(!outcome.tokens.isEmpty)
            results.append(SmokeResult(prompt: prompt, promptTokens: tokens, generatedTokens: outcome.tokens,
                                       text: text, stopReason: outcome.stopReason.rawValue,
                                       prefillSeconds: outcome.timings.prefillSeconds,
                                       decodeSeconds: outcome.timings.decodeSeconds, firstStepTop: []))
        }
        if let out = Self.env["MFERENCE_NATIVE_REFERENCE_OUT"] {
            let url = URL(fileURLWithPath: out)
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(results).write(to: url.appendingPathComponent("quick-reference-smoke.json"))
        }
    }

    static func makeModel(ctx: MetalContext, packURL: URL, context: Int = 4096,
                          chunk: Int = 64) throws -> Native.ReferenceForwardModel {
        let store = try Native.ReferenceWeightStore(directory: packURL, device: ctx.device)
        let factory = Native.ReferenceForwardModelFactory(context: ctx)
        let c = store.config
        let plan = try Native.ReferenceMemoryPlanner().resolve(
            config: c, weightsMapped: store.mappedByteCount, contextLength: context, precision: .f16,
            maxVerifyRows: 1, maxPrefillChunk: chunk,
            activationBytes: factory.activationBytes(config: c, maxPrefillChunk: chunk, maxVerifyRows: 1),
            prefixCacheRAMBudget: 1 << 28, budget: 16 << 30)
        let kv = try Native.ReferenceKVStore(config: c, plan: plan, device: ctx.device)
        let components = Native.ModelComponents(
            device: ctx.device, weights: store, gemm: Native.ReferenceQuantGEMM(context: ctx),
            attention: Native.ReferenceAttentionOp(context: ctx), gdn: Native.ReferenceGDNOp(context: ctx),
            glue: Native.ReferenceGlueOps(context: ctx), kv: kv)
        return try factory.make(components, plan: plan) as! Native.ReferenceForwardModel
    }
}
