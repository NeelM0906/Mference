import Foundation
@testable import Mference

extension MFTokenizer {
    /// The committed ChatML (Qwen 3.6) tokenizer fixture, for tests that need
    /// a real tokenizer as a token source. Replaces the retired Gemma 4 Hub
    /// tokenizer these tests used to load; it needs no network or Hub cache.
    static func chatMLFixture() async throws -> MFTokenizer {
        try await MFTokenizer.load(from: try ChatMLTemplateTests.fixtureFolder(), family: .qwen36)
    }
}
