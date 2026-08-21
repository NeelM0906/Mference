import Testing
import Foundation
import Metal
@testable import Mference

/// End-to-end generation regression for the Inkling family, run against the
/// real install. Env-gated on `MFERENCE_INKLING_GTURBO`; skipped otherwise.
///
/// The bug this guards: layer 41's shared-expert down projection has a channel
/// (3895 on the released 4-bit checkpoint) that runs at 1.5e4-6e4 on every
/// token. It was stored as FP16, whose ceiling is 65 504, and the ÷32 FFN
/// prescale is applied *after* that store, so it gave the raw row no headroom.
/// A token that pushed the channel to 69 307 clipped to `-inf`; the FP32
/// residual inherited it; the head's RMS norm turned `inf * 0` into NaN across
/// the whole row; and the greedy argmax — seeded at `(index 0, -infinity)`,
/// where every `>` comparison against NaN is false — returned token 0, which
/// this vocabulary decodes as `!`. Because layer 41's `mlp_sconv` state holds
/// the last K-1 = 3 inputs, one clipped token produced a four-token burst:
///
///     "Coastal wetlands—salt marshes, mang!!!! Wait, I need to be careful…"
///
/// Deterministic, and invisible to every install and manifest check.
@Suite struct InklingGenerationRegressionTests {

    /// docs/benchmark-prompts/real-generation-v1/<name>.json, located from this
    /// file rather than an env var so the fixture cannot silently drift.
    private static func benchmarkMessages(_ name: String) throws
        -> [MFTokenizer.Message]
    {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let url = root
            .appendingPathComponent("docs/benchmark-prompts/real-generation-v1")
            .appendingPathComponent("\(name).json")
        struct Turn: Decodable { let role: String; let content: String }
        let turns = try JSONDecoder().decode(
            [Turn].self, from: try Data(contentsOf: url))
        return turns.map {
            MFTokenizer.Message(role: $0.role == "system" ? .system : .user,
                                content: $0.content)
        }
    }

    /// The reported reproduction: greedy decode of the `short-explanation`
    /// benchmark prompt must not contain a run of exclamation marks.
    ///
    /// `!!` is the assertion rather than `!` because a single exclamation mark
    /// is ordinary prose; a run of two or more is the signature of consecutive
    /// argmax fallbacks. `MFERENCE_INKLING_GTURBO` must point at the install
    /// (~148 GB); this prefills 59 tokens and decodes 120, so expect a couple
    /// of minutes.
    @Test func shortExplanationHasNoExclamationBurst() async throws {
        guard let path = ProcessInfo.processInfo
            .environment["MFERENCE_INKLING_GTURBO"] else { return }
        let ctx = try MetalContext()
        let modelURL = URL(fileURLWithPath: path)
        let tokenizer = try await MFTokenizer.load(forModelDirectory: modelURL)
        let config = GenerationConfig(maxNewTokens: 120, temperature: 0)
        let runtime = RuntimeConfiguration(prefillChunkTokens: 128,
                                           forceLogitsHead: !config.isPureGreedy)
        let model = try Model.load(
            directoryURL: modelURL,
            device: ctx.device,
            streamingMode: .pread(slotCount: runtime.expertCacheSlots),
            expertCachePolicy: runtime.modelExpertCachePolicy)
        let runner = try RealForwardRunner(model: model, context: ctx,
                                          maxContext: 4096,
                                          runtimeConfiguration: runtime)
        let scratch = try RawCompletionScratch(
            context: ctx, vocab: model.config.vocabSize,
            logitSoftcap: Float(model.config.finalLogitSoftcap))
        let prompt = try tokenizer.applyChatTemplate(
            try Self.benchmarkMessages("short-explanation"))
        var text = ""
        // A throw here is itself the regression firing: the head guard reports
        // `InklingHeadError.nonFiniteLogits` for exactly the row that used to
        // be silently reported as token 0.
        _ = try await runRawCompletion(
            producer: runner,
            tokenizer: tokenizer,
            promptIds: tokenizer.encode(prompt, addBOS: false),
            config: config,
            context: ctx,
            scratch: scratch,
            prefillConfig: runtime.prefillConfig) { progress in
                switch progress {
                case .prefill: break
                case .token(_, _, let delta): text += delta
                case .tail(let tail): text += tail
                }
            }
        #expect(!text.contains("!!"),
                "exclamation-mark burst in Inkling output — a logit row went non-finite and the argmax fell back to token 0:\n\(text)")
        // A burst replaced real text mid-word, so the completion also has to
        // look like prose: the failing run reached 15 words before the burst,
        // and every healthy run runs well past that.
        #expect(text.split(separator: " ").count > 40,
                "completion is implausibly short:\n\(text)")
    }

    /// Shadow speculative prefetch must be a pure transport change: greedy
    /// tokens with the pilot+shadow path on are byte-identical to `off`.
    /// Env-gated on the real install like the suite's other tests.
    @Test func shadowPrefetchKeepsGreedyTokensIdentical() async throws {
        guard let path = ProcessInfo.processInfo
            .environment["MFERENCE_INKLING_GTURBO"] else { return }
        let ctx = try MetalContext()
        let cfg = try #require(ArchConfig.knownArchitectures[.inklingSmall])
        // Eight slots force eviction churn, so speculation actually issues.
        let model = try Model.load(directoryURL: URL(fileURLWithPath: path),
                                   device: ctx.device,
                                   expecting: cfg,
                                   streamingMode: .pread(slotCount: 8))
        let runner = try RealForwardRunner(model: model, context: ctx,
                                           maxContext: 64)
        let logits = ctx.device.makeBuffer(
            length: cfg.vocabSize * MemoryLayout<Float16>.stride,
            options: .storageModeShared)!

        func decode(steps: Int) async throws -> [UInt32] {
            var out: [UInt32] = []
            var token: Int32 = 200_028
            for position in 0..<steps {
                try await runner.produce(token: token, position: position,
                                         into: logits)
                out.append(runner.lastGreedyToken)
                token = Int32(runner.lastGreedyToken)
            }
            return out
        }

        runner.speculativePrefetchMode = .off
        let reference = try await decode(steps: 8)
        runner.reset()
        runner.speculativePrefetchMode = .shadow
        let shadowed = try await decode(steps: 8)
        #expect(reference == shadowed)
    }

    /// The head guard itself: with a finite residual stream the epilogue must
    /// report zero non-finite logits, so `produce()` cannot throw
    /// `InklingHeadError`. Cheap — one token, no generation loop.
    @Test func headReportsNoNonFiniteLogitsOnAHealthyStep() async throws {
        guard let path = ProcessInfo.processInfo
            .environment["MFERENCE_INKLING_GTURBO"] else { return }
        let ctx = try MetalContext()
        let cfg = try #require(ArchConfig.knownArchitectures[.inklingSmall])
        let model = try Model.load(directoryURL: URL(fileURLWithPath: path),
                                   device: ctx.device,
                                   expecting: cfg)
        let runner = try RealForwardRunner(model: model, context: ctx,
                                          maxContext: 64)
        let logits = ctx.device.makeBuffer(
            length: cfg.vocabSize * MemoryLayout<Float16>.stride,
            options: .storageModeShared)!
        try await runner.produce(token: 200_028, position: 0, into: logits)
        #expect(runner.inklingNonFiniteLogitCount == 0)
    }
}
