import Foundation
import Testing
@testable import Mference

@Suite struct ContextLimitTests {
    /// Each pinned checkpoint's own context: `max_position_embeddings`, and
    /// `model_max_length` for Inkling, which has none.
    @Test func everyFamilyCarriesItsCheckpointsNativeContext() {
        let expected: [ModelFamily: Int] = [
            .gemma4: 262_144,
            .qwen36: 262_144,
            .qwen38: 262_144,
            .qwen38flashnext: 262_144,
            .minicpm5: 131_072,
            .maple: 128_000,
            .deepseekV4Flash: 1_048_576,
            .glm53Flash: 1_048_576,
            .inklingSmall: 1_048_576,
        ]
        #expect(Set(expected.keys) == Set(ModelFamily.allCases))
        for (family, tokens) in expected {
            #expect(family.maximumContext == tokens, "\(family)")
        }
        #expect(ModelFamily.largestMaximumContext == 1_048_576)
    }

    @Test func refusalNamesTheModelAndItsLimit() {
        let error = ContextLimitError(family: .maple, requested: 262_144, maximum: 128_000)
        #expect(error.description
                == "--max-context 262144 is above maple's native context of 128000 tokens")
    }

    @Test func runnerFactoryRefusesAContextAboveTheModelLimit() throws {
        let dir = try QwenToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: dir) }
        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir, device: ctx.device,
                                   expecting: .qwen36Toy(), streamingMode: .pread(slotCount: 8))
        #expect(throws: ContextLimitError(family: .qwen36, requested: 262_145, maximum: 262_144)) {
            _ = try ForwardRunnerFactory.make(model: model, context: ctx, maxContext: 262_145)
        }
        _ = try ForwardRunnerFactory.make(model: model, context: ctx, maxContext: 262_144)
    }
}
