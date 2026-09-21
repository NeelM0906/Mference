import Mference

/// What `--chat --reuse-prefix` runs for a turn, given the verdict of the
/// shared `PromptPrefixCache` (the same rules the server applies).
enum ChatPrefixReuse {
    struct Plan: Equatable {
        let promptIDs: [Int32]
        let start: RawCompletionStart
    }

    /// A hit continues after the cached tokens with the cache's effective
    /// prompt, which can differ from the fresh render: ChatML keeps the turn as
    /// it was generated and appends a bridge. History is trimmed against the
    /// rendered prompt, so an effective prompt that no longer fits the context
    /// falls back to the rendered one from a reset cache.
    static func plan(match: PromptPrefixCacheMatch,
                     renderedPromptIDs: [Int32],
                     maxContext: Int) -> Plan {
        guard case .hit(let effectivePromptIDs, let cachedPromptTokens) = match,
              effectivePromptIDs.count < maxContext else {
            return Plan(promptIDs: renderedPromptIDs, start: .reset)
        }
        return Plan(promptIDs: effectivePromptIDs, start: .resume(cachedPromptTokens: cachedPromptTokens))
    }
}
