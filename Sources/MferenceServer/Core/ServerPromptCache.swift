import Foundation
import Mference

public enum ServerPromptCacheMode: String, Sendable, Equatable {
    case off
    case singlePrefix = "single-prefix"
}

// The cache itself lives in the core library (`PromptPrefixCache`) so the CLI
// can reuse prefixes by the same rules. The server keeps its names and call
// shapes through these aliases and the request adapter below.
typealias ServerPromptCacheDomain = PromptPrefixCacheDomain
typealias ServerPromptCacheEntry = PromptPrefixCacheEntry
typealias ServerPromptCacheMatch = PromptPrefixCacheMatch
typealias ServerPromptCache = PromptPrefixCache

extension ValidatedChatRequest {
    var promptCacheTurn: PromptPrefixCacheTurn {
        PromptPrefixCacheTurn(messages: messages,
                              tools: tools,
                              reasoningEffort: reasoningEffort,
                              preserveThinking: preserveThinking)
    }
}

extension PromptPrefixCache {
    mutating func publish(
        domain: ServerPromptCacheDomain,
        request: ValidatedChatRequest,
        content: String,
        calls: [ParsedToolCall],
        result: RawDecodeResult,
        reasoningContent: String? = nil,
        stopStringFiltered: Bool = false
    ) {
        publish(domain: domain,
                turn: request.promptCacheTurn,
                content: content,
                calls: calls,
                result: result,
                reasoningContent: reasoningContent,
                stopStringFiltered: stopStringFiltered)
    }

    func match(
        domain: ServerPromptCacheDomain,
        request: ValidatedChatRequest,
        renderedPromptIDs: [Int32],
        tokenizer: MFTokenizer,
        gemmaRecoverablePrefix: ((Int) -> Int)? = nil
    ) -> ServerPromptCacheMatch {
        match(domain: domain,
              turn: request.promptCacheTurn,
              renderedPromptIDs: renderedPromptIDs,
              tokenizer: tokenizer,
              gemmaRecoverablePrefix: gemmaRecoverablePrefix)
    }
}
