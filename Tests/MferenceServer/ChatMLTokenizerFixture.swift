import Foundation
@testable import Mference

/// The committed ChatML tokenizer fixture (`Tests/Mference/Core/Tokenization/
/// Fixtures/ChatMLTokenizer`): the Qwen 3.6 chat template over a byte-level
/// toy vocabulary. It renders and round-trips text exactly, but its token
/// counts are one per byte, so it says nothing about a real model's budget.
enum ChatMLTokenizerFixture {
    static var folder: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Mference/Core/Tokenization/Fixtures/ChatMLTokenizer")
    }

    static func load(family: ModelFamily = .qwen36) async throws -> MFTokenizer {
        try await MFTokenizer.load(from: folder, family: family)
    }
}
