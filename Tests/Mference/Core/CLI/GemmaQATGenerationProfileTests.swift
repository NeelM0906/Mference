import Foundation
import Testing
@testable import Mference
@testable import MferenceCLICore

@Suite struct GemmaQATGenerationProfileTests {
    // Verbatim generation_config.json from revision
    // 745a97a754ed4b7713163c7d0e9c11da41809e0c, not native-generated expectations.
    static func sourceData() throws -> Data {
        try Data(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Tokenization/Fixtures/GemmaQATProfile/generation_config.json"))
    }
    static let base = ["--model", "gemma4qat.gturbo", "--prompt", "The capital of France is"]

    @Test func omittedControlsUseCheckpointSettingsWithoutAdditionalProcessors() throws {
        let defaults = try GemmaQATCheckpoint.generationDefaults(from: Self.sourceData())
        let args = try Args.parse(Self.base + ["--seed", "42", "--stop", "done"])
        let config = try args.generationConfig(defaults: defaults, maxNewTokens: 17)
        #expect(config.temperature == 1 && config.topK == 64 && config.topP == 0.95)
        #expect(config.minP == 0 && config.repetitionPenalty == 1)
        #expect(config.presencePenalty == 0 && config.frequencyPenalty == 0)
        #expect(config.maxNewTokens == 17 && config.seed == 42 && config.stopStrings == ["done"])
        #expect(!config.isPureGreedy)
    }

    @Test func explicitValuesIncludingSharedDefaultsAndZeroRemainOverrides() throws {
        let defaults = try GemmaQATCheckpoint.generationDefaults(from: Self.sourceData())
        let explicit = try Args.parse(Self.base + ["--temperature", "0.8", "--top-k", "40",
            "--top-p", "0.7", "--min-p", "0.05", "--repeat-penalty", "1.2",
            "--presence-penalty", "0.3", "--frequency-penalty", "-0.2", "--repeat-last-n", "-1"])
        let c = try explicit.generationConfig(defaults: defaults, maxNewTokens: 5)
        #expect(c.temperature == 0.8 && c.topK == 40 && c.topP == 0.7 && c.minP == 0.05)
        #expect(c.repetitionPenalty == 1.2 && c.presencePenalty == 0.3 && c.frequencyPenalty == -0.2)
        #expect(c.repeatLastN == -1)
        let zeros = try Args.parse(Self.base + ["--temperature", "0", "--top-k", "0",
            "--top-p", "1", "--min-p", "0", "--repetition-penalty", "1",
            "--presence-penalty", "0", "--frequency-penalty", "0", "--repeat-last-n", "0"])
        let z = try zeros.generationConfig(defaults: defaults, maxNewTokens: 5)
        #expect(z.temperature == 0 && z.topK == nil && z.topP == 1 && z.minP == 0)
        #expect(z.repetitionPenalty == 1 && z.presencePenalty == 0 && z.frequencyPenalty == 0)
        #expect(z.repeatLastN == 0 && z.isPureGreedy)
        // Off remains off for stochastic decoding too.
        let off = try Args.parse(Self.base + ["--top-k", "0", "--top-p", "1"])
        let o = try off.generationConfig(defaults: defaults, maxNewTokens: 5)
        #expect(o.temperature == 1 && o.topK == nil && o.topP == 1 && o.minP == 0)
    }

    @Test func originalDefaultsAndProgrammaticOverridesArePreserved() throws {
        var args = try Args.parse(Self.base)
        let original = try args.generationConfig(defaults: .defaults, maxNewTokens: 1)
        #expect(original.temperature == 0.8 && original.topK == 40 && original.topP == 0.95)
        #expect(original.minP == 0.05 && original.repeatLastN == 64)
        // Public property mutation is explicit, even when equal to the old default.
        args.temperature = 0.8
        args.topK = 40
        args.minP = 0.05
        let qat = try args.generationConfig(
            defaults: GemmaQATCheckpoint.generationDefaults(from: Self.sourceData()), maxNewTokens: 1)
        #expect(qat.temperature == 0.8 && qat.topK == 40 && qat.minP == 0.05)
        let direct = Args(model: "gemma4qat.gturbo", prompt: "hi", temperature: 0, topK: nil, topP: nil)
        let greedy = try direct.generationConfig(
            defaults: GemmaQATCheckpoint.generationDefaults(from: Self.sourceData()), maxNewTokens: 1)
        #expect(greedy.temperature == 0 && greedy.topK == nil && greedy.topP == nil)
    }

    @Test func sourceTokenIDsAndGenerationFieldsAreRequired() throws {
        let source = try #require(JSONSerialization.jsonObject(with: Self.sourceData()) as? [String: Any])
        for (key, invalid) in [("bos_token_id", 3 as Any), ("pad_token_id", 1 as Any),
            ("eos_token_id", [1, 106] as Any), ("do_sample", false as Any),
            ("temperature", -1 as Any), ("top_k", 257 as Any), ("top_p", 1.1 as Any)] {
            var data = source
            data[key] = invalid
            #expect(throws: (any Error).self) {
                _ = try GemmaQATCheckpoint.generationDefaults(from: JSONSerialization.data(withJSONObject: data))
            }
            data.removeValue(forKey: key)
            #expect(throws: (any Error).self) {
                _ = try GemmaQATCheckpoint.generationDefaults(from: JSONSerialization.data(withJSONObject: data))
            }
        }
    }

    @Test func tokenizerWithOtherTokenIDsCannotExecuteQAT() async throws {
        let tokenizer = try await MFTokenizer.load(from: GemmaThinkingTests.fixtureFolder(), family: .gemma4)
        #expect(throws: ModelError.self) { try GemmaQATCheckpoint.validateTokenizer(tokenizer) }
        #expect(tokenizer.generationDefaults.temperature == 0.8 && tokenizer.generationDefaults.minP == 0.05)
    }

    @Test func missingAssetsCannotFallBackToAnEnvironmentTokenizer() async throws {
        let (directory, _) = try GemmaQATManifestTests.makeInstall()
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.removeItem(at: directory.appendingPathComponent("tokenizer/tokenizer.json"))
        await #expect(throws: (any Error).self) {
            _ = try await MFTokenizer.load(forModelDirectory: directory,
                environment: ["MFERENCE_TOKENIZER_DIR": GemmaThinkingTests.fixtureFolder().path])
        }
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["MFERENCE_GEMMA_QAT_GTURBO"] != nil))
    func installedTokenizerUsesVerifiedLocalProfileForRawAndChat() async throws {
        let directory = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["MFERENCE_GEMMA_QAT_GTURBO"]))
        let source = try Data(contentsOf: directory.appendingPathComponent("tokenizer/generation_config.json"))
        #expect(source == (try Self.sourceData()))
        let tokenizer = try await MFTokenizer.load(forModelDirectory: directory,
            environment: ["MFERENCE_TOKENIZER_DIR": GemmaThinkingTests.fixtureFolder().path])
        #expect(tokenizer.bosID == 2 && tokenizer.padID == 0 && tokenizer.stopTokenIDs == [1, 106, 50])
        #expect(tokenizer.encode("The capital of France is", addBOS: true).first == 2)
        let args = try Args.parse(["--model", directory.path, "--prompt", "hi"])
        let config = try args.generationConfig(defaults: tokenizer.generationDefaults, maxNewTokens: 1)
        #expect(config.temperature == 1 && config.topK == 64 && config.topP == 0.95 && config.minP == 0)
        let chat = try await MFTokenizer.load(forModelDirectory: directory)
        #expect(chat.isGemmaQAT)
        #expect(chat.generationDefaults.temperature == 1 && chat.generationDefaults.topK == 64)
        #expect(try chat.encodeChat(messages: [.init(role: .user, content: "Hi")])
            == tokenizer.encodeChat(messages: [.init(role: .user, content: "Hi")]))
    }
}
