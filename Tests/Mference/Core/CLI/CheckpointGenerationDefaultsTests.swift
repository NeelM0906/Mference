import Foundation
import Testing
@testable import Mference
@testable import MferenceCLICore

@Suite struct CheckpointGenerationDefaultsTests {
    // Verbatim generation_config.json from the pinned qwen36 revision 38740b84
    // in SupportedModelSource.
    static func fixture(_ name: String) throws -> Data {
        try Data(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Tokenization/Fixtures/CheckpointGeneration/\(name).json"))
    }
    static let base = ["--model", "qwen36.gturbo", "--prompt", "The capital of France is"]

    @Test func presentKeysBecomeDefaultsWithoutAdditionalProcessors() throws {
        let c = try CheckpointGeneration.defaults(from: Self.fixture("qwen36"))
        #expect(c.temperature == 1 && c.topK == 20 && c.topP == 0.95)
        #expect(c.minP == 0 && c.repetitionPenalty == 1)
        #expect(c.presencePenalty == 0 && c.frequencyPenalty == 0)
    }

    @Test func absentKeysAreOffNotLibraryDefaults() throws {
        // Shapes of real checkpoints' files (formerly the DeepSeek-V4-Flash and
        // GLM-5.3-Flash fixtures): do_sample without top_k, and temperature
        // without do_sample.
        let sampled = try CheckpointGeneration.defaults(from: Data(#"""
            {"_from_model_config": true, "bos_token_id": 0, "eos_token_id": 1, "do_sample": true,
             "temperature": 1.0, "top_p": 1.0, "transformers_version": "4.46.3"}
            """#.utf8))
        #expect(sampled.temperature == 1 && sampled.topK == nil && sampled.topP == 1 && sampled.minP == 0)
        // Temperature without do_sample: the author's values still sample.
        let implicit = try CheckpointGeneration.defaults(from: Data(#"""
            {"_from_model_config": true, "eos_token_id": [154820, 154827, 154829],
             "pad_token_id": 154820, "temperature": 1.0, "top_p": 0.95,
             "transformers_version": "5.16.0"}
            """#.utf8))
        #expect(implicit.temperature == 1 && implicit.topK == 256 && implicit.topP == 0.95)
        #expect(!implicit.isPureGreedy)
    }

    @Test func zeroTopKMeansDisabled() throws {
        let nucleus = try CheckpointGeneration.defaults(from: Data(#"{"top_k": 0, "top_p": 0.9}"#.utf8))
        #expect(nucleus.topK == 256 && nucleus.topP == 0.9)
        let plain = try CheckpointGeneration.defaults(from: Data(#"{"top_k": 0}"#.utf8))
        #expect(plain.topK == nil)
    }

    @Test func doSampleFalseIsGreedy() throws {
        let data = Data(#"{"do_sample": false, "temperature": 0.6, "top_k": 20}"#.utf8)
        let c = try CheckpointGeneration.defaults(from: data)
        #expect(c.temperature == 0 && c.isPureGreedy)
    }

    @Test func malformedConfigThrows() {
        #expect(throws: (any Error).self) {
            try CheckpointGeneration.defaults(from: Data("[1, 2]".utf8))
        }
        #expect(throws: (any Error).self) {
            try CheckpointGeneration.defaults(from: Data(#"{"temperature": "hot"}"#.utf8))
        }
        #expect(throws: (any Error).self) {
            try CheckpointGeneration.defaults(from: Data(#"{"temperature": -1}"#.utf8))
        }
    }

    @Test func missingFileKeepsSharedDefaults() throws {
        let empty = FileManager.default.temporaryDirectory
            .appendingPathComponent("ckpt-gen-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: empty) }
        for folder in [empty, nil] {
            let c = try CheckpointGeneration.defaults(inFolder: folder)
            #expect(c.temperature == 0.8 && c.topK == 40 && c.topP == 0.95)
            #expect(c.minP == 0.05 && c.repeatLastN == 64)
        }
        try Self.fixture("qwen36").write(to: empty.appendingPathComponent("generation_config.json"))
        #expect(try CheckpointGeneration.defaults(inFolder: empty).topK == 20)
    }

    @Test func explicitFlagsOverrideAndTemperatureZeroIsGreedy() throws {
        let defaults = try CheckpointGeneration.defaults(from: Self.fixture("qwen36"))
        let greedy = try Args.parse(Self.base + ["--temperature", "0"])
            .generationConfig(defaults: defaults, maxNewTokens: 64)
        #expect(greedy.isPureGreedy)
        let explicit = try Args.parse(Self.base + ["--temperature", "0.8", "--top-k", "40"])
            .generationConfig(defaults: defaults, maxNewTokens: 5)
        #expect(explicit.temperature == 0.8 && explicit.topK == 40 && explicit.topP == 0.95)
    }
}
