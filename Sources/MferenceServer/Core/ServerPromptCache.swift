import Foundation
import Mference

public enum ServerPromptCacheMode: String, Sendable, Equatable {
    case off
    case singlePrefix = "single-prefix"
    /// Several retained turn-boundary prefixes under a byte budget, for
    /// runners that can snapshot their prefix state (Qwen 3.6 and the
    /// Flash-Next families). Every other
    /// runner serves it exactly as `singlePrefix`.
    case multiPrefix = "multi-prefix"
}

/// `--prompt-cache-bytes`: the ceiling on what the multi-prefix cache retains
/// besides the runner's own live buffers.
public enum ServerPromptCacheBytes: Equatable, Sendable {
    case auto
    case bytes(Int)

    static let gibibyte = 1 << 30

    /// `auto` is min(4 GiB, physical / 8) on hosts with at least 24 GiB,
    /// 1 GiB on 16 to under 24 GiB, and 0 (single-prefix behaviour) below.
    /// The Flash-Next families cap that at 1.5 GiB: their routed experts
    /// (50–175 GB) stream through the file cache, which every retained byte
    /// shrinks. 1.5 GiB holds one 32K row image (about 1.06 GB on the
    /// Unsloth install) plus a few recurrent checkpoints (about 116 MB each).
    func resolved(family: ModelFamily? = nil,
                  physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory) -> Int {
        switch self {
        case .bytes(let value):
            return max(0, value)
        case .auto:
            let physical = Int(clamping: physicalMemoryBytes)
            let general: Int
            if physical >= 24 * Self.gibibyte {
                general = min(4 * Self.gibibyte, physical / 8)
            } else if physical >= 16 * Self.gibibyte {
                general = Self.gibibyte
            } else {
                general = 0
            }
            switch family {
            case .qwen38flashnext?, .qwen38flashnextudiq3xxs?:
                return min(general, 3 * Self.gibibyte / 2)
            default:
                return general
            }
        }
    }
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

extension PromptPrefixCacheMatch {
    /// `prepare` admitted the rendered prompt, but a hit can be longer: the KV
    /// cache keeps generated tokens, such as a thought, that the re-render
    /// drops. A hit that no longer fits is treated as a miss, as the CLI's
    /// `--reuse-prefix` does, instead of failing after the response started.
    func fitting(maxContext: Int) -> PromptPrefixCacheMatch {
        if case .hit(let effective, _) = self, effective.count >= maxContext {
            return .miss
        }
        return self
    }
}
