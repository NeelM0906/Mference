import Foundation

/// A source of streamed assistant tokens for a chat transcript.
public protocol ChatBackend: Sendable {
    func stream(
        messages: [ChatMessage],
        model: String
    ) -> AsyncThrowingStream<String, Error>
}
