import Mference

/// Outcome of fitting a chat history into the context window.
public struct TrimmedChatHistory: Equatable, Sendable {
    /// The kept messages, oldest first.
    public let messages: [MFTokenizer.Message]
    /// Rendered prompt length of `messages`.
    public let tokens: Int
    /// How many messages were dropped from the front of the history.
    public let dropped: Int
    /// Whether `tokens` leaves room for at least one generated token.
    public let fits: Bool
}

/// Drop the oldest turns until the rendered history fits in `limit` tokens.
///
/// A leading system message and the newest message are never dropped, so the
/// result can still exceed the limit; callers report that as an error rather
/// than sending an oversized prompt. `measure` re-renders after every drop
/// because both chat dialects add per-render markup that no per-message
/// estimate can predict.
public func trimChatHistory(
    _ messages: [MFTokenizer.Message],
    limit: Int,
    measure: ([MFTokenizer.Message]) throws -> Int
) rethrows -> TrimmedChatHistory {
    let firstDroppable = messages.first?.role == .system ? 1 : 0
    var kept = messages
    var tokens = try measure(kept)
    while tokens >= limit, kept.count > firstDroppable + 1 {
        kept.remove(at: firstDroppable)
        tokens = try measure(kept)
    }
    return TrimmedChatHistory(messages: kept,
                              tokens: tokens,
                              dropped: messages.count - kept.count,
                              fits: tokens < limit)
}
