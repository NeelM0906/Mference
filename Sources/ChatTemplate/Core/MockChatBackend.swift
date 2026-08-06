import Foundation
import Synchronization

/// Word-by-word canned streaming, ported from the template's
/// `mockStreamResponse` + `MOCK_RESPONSES` in `src/app/index.tsx`.
public final class MockChatBackend: ChatBackend {
    public static let responses: [String] = [
        "That's a great question! Here's what I think:\n\nThe key insight is that **simplicity** often beats complexity. When you break down the problem into smaller pieces, the solution becomes much clearer.\n\n```javascript\nconst answer = problems\n  .map(simplify)\n  .reduce(combine, []);\n```\n\nHope that helps!",
        "I'd be happy to help with that. Let me walk you through it step by step:\n\n1. **First**, identify the core requirements\n2. **Then**, design the interface\n3. **Finally**, implement and test\n\nThe most important thing is to start simple and iterate. You can always add more features later.",
        "Interesting! Here's a quick overview:\n\n> The best code is the code you don't have to write.\n\nThat said, when you *do* need to write code, keep these principles in mind:\n\n- **Readability** over cleverness\n- **Composition** over inheritance\n- **Explicit** over implicit\n\nLet me know if you want me to dive deeper into any of these!",
        "Sure thing! Here's a concise answer:\n\nThe approach I'd recommend is to use a **streaming architecture** where data flows through the system in real-time. This gives you:\n\n- Lower latency\n- Better resource utilization\n- Simpler error handling\n\n```python\nasync for chunk in stream:\n    process(chunk)\n```\n\nWant me to elaborate on any part?",
    ]

    private let nextIndex = Mutex(0)
    private let tokenDelay: ClosedRange<Double>?

    /// - Parameter tokenDelay: per-word delay range in seconds; nil disables
    ///   delays (for tests). Default mirrors the template's 30-70 ms.
    public init(tokenDelay: ClosedRange<Double>? = 0.03...0.07) {
        self.tokenDelay = tokenDelay
    }

    public func stream(
        messages: [ChatMessage],
        model: String
    ) -> AsyncThrowingStream<String, Error> {
        let response = nextResponse()
        let delay = tokenDelay
        return AsyncThrowingStream { continuation in
            let task = Task {
                for word in Self.splitKeepingWhitespace(response) {
                    if Task.isCancelled { break }
                    if let delay {
                        try? await Task.sleep(for: .seconds(Double.random(in: delay)))
                    }
                    continuation.yield(word)
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func nextResponse() -> String {
        nextIndex.withLock { index in
            let response = Self.responses[index % Self.responses.count]
            index += 1
            return response
        }
    }

    /// Split after each run of whitespace, like the template's `split(/(?<=\s)/)`.
    static func splitKeepingWhitespace(_ text: String) -> [String] {
        var words: [String] = []
        var current = ""
        var inWhitespace = false
        for character in text {
            let isWhitespace = character.isWhitespace
            if inWhitespace, !isWhitespace {
                words.append(current)
                current = ""
            }
            current.append(character)
            inWhitespace = isWhitespace
        }
        if !current.isEmpty { words.append(current) }
        return words
    }
}
