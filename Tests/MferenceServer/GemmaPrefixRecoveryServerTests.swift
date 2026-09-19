import Foundation
import Testing
@testable import Mference
@testable import MferenceServerCore

@Suite struct GemmaPrefixRecoveryServerTests {
    @Test(arguments: [false, true])
    func newUserCanFollowThePendingToolResults(preserve: Bool) async throws {
        let folder = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Mference/Core/Tokenization/Fixtures/GemmaThinkingTokenizer")
        let tok = try await MFTokenizer.load(from: folder, family: .gemma4)
        let domain = ServerPromptCacheDomain(modelID: "alias", sourceSnapshotHash: "source",
            runtimeProfileHash: "runtime", maximumContext: 4096, kvStorage: "fp16",
            fp16RingEnabled: true, templateSHA256: "canonical")
        let tools: [MFTokenizer.FunctionDefinition] = [.init(name: "lookup", description: "Lookup",
            parameters: .object(["type": .string("object"), "properties": .object([:])]))]
        func request(_ messages: [MFTokenizer.Message]) -> ValidatedChatRequest {
            .init(reasoningEffort: .medium, preserveThinking: preserve, messages: messages,
                tools: tools, stream: false, includeUsage: false,
                generationConfig: .init(maxNewTokens: 32, temperature: 0), maximumCompletionTokens: 32)
        }
        let first = request([.init(role: .user, content: "Lookup")])
        let prompt = try tok.encodeChat(messages: first.messages, tools: tools,
            reasoningEffort: .medium, preserveThinking: preserve)
        let call = ParsedToolCall(id: "call_1", name: "lookup", arguments: .object([:]), argumentsJSON: "{}")
        let kv = prompt + tok.encode("<|channel>thought\nCheck<channel|><|tool_call>call:lookup{}<tool_call|>", addBOS: false)
        var cache = ServerPromptCache()
        cache.publish(domain: domain, request: first, content: "", calls: [call],
            result: RawDecodeResult(prefillTokens: prompt.count, cachedPromptTokens: 0,
                computedPrefillTokens: prompt.count, prefillSeconds: 0, newTokens: 1,
                decodeSeconds: 0, reason: .toolCalls, kvPosition: kv.count,
                kvBackedTokenIDs: kv, uncommittedBoundaryTokenIDs: [tok.toolResponseID], prefillExecution: nil),
            reasoningContent: "Check")
        let next = request(first.messages + [
            .init(role: .assistant, content: nil, toolCalls: [.init(id: call.id, name: call.name, arguments: call.arguments)], reasoningContent: "Check"),
            .init(role: .tool, content: "Found", toolCallID: call.id),
            .init(role: .user, content: "New question")])
        let canonical = try tok.encodeChat(messages: next.messages, tools: tools,
            reasoningEffort: .medium, preserveThinking: preserve)
        let common = zip(kv, canonical).prefix { $0 == $1 }.count
        #expect(common > 0 && common < kv.count)
        #expect(cache.match(domain: domain, request: next, renderedPromptIDs: canonical,
            tokenizer: tok, gemmaRecoverablePrefix: { $0 })
            == .hit(effectivePromptIDs: canonical, cachedPromptTokens: common))
    }

    @Test(arguments: [false, true], [false, true])
    func newUserReusesLongestAvailableCanonicalPrefix(thinking: Bool, preserve: Bool) async throws {
        let folder = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Mference/Core/Tokenization/Fixtures/GemmaThinkingTokenizer")
        let tok = try await MFTokenizer.load(from: folder, family: .gemma4)
        let domain = ServerPromptCacheDomain(modelID: "alias", sourceSnapshotHash: "source",
            runtimeProfileHash: "runtime", maximumContext: 4096, kvStorage: "fp16",
            fp16RingEnabled: true, templateSHA256: "canonical")
        func request(_ messages: [MFTokenizer.Message]) -> ValidatedChatRequest {
            .init(reasoningEffort: thinking ? .medium : .off, preserveThinking: preserve,
                messages: messages, tools: [], stream: false, includeUsage: false,
                generationConfig: .init(maxNewTokens: 32, temperature: 0), maximumCompletionTokens: 32)
        }
        func render(_ r: ValidatedChatRequest) throws -> [Int32] {
            try tok.encodeChat(messages: r.messages, reasoningEffort: r.reasoningEffort,
                preserveThinking: r.preserveThinking)
        }
        let first = request([.init(role: .user, content: "First question")])
        let prompt = try render(first)
        let reasoning: String? = thinking ? "Check carefully" : nil
        let generated = (thinking ? "<|channel>thought\nCheck carefully\n<channel|>" : "") + "Answer"
        let kv = prompt + tok.encode(generated, addBOS: false)
        var cache = ServerPromptCache()
        cache.publish(domain: domain, request: first, content: "Answer", calls: [],
            result: RawDecodeResult(prefillTokens: prompt.count, cachedPromptTokens: 0,
                computedPrefillTokens: prompt.count, prefillSeconds: 0, newTokens: 1,
                decodeSeconds: 0, reason: .endOfTurn, kvPosition: kv.count,
                kvBackedTokenIDs: kv, uncommittedBoundaryTokenIDs: [tok.endOfTurnID], prefillExecution: nil),
            reasoningContent: reasoning)
        let next = request(first.messages + [.init(role: .assistant, content: "Answer", reasoningContent: reasoning),
                                            .init(role: .user, content: "Second question")])
        let canonical = try render(next)
        let common = zip(kv, canonical).prefix { $0 == $1 }.count
        #expect(common > 0 && common < kv.count)
        for available in [common, common / 2, 0] {
            let match = cache.match(domain: domain, request: next, renderedPromptIDs: canonical,
                tokenizer: tok, gemmaRecoverablePrefix: { limit in
                    #expect(limit == common)
                    return min(limit, available)
                })
            if available == 0 { #expect(match == .miss) }
            else { #expect(match == .hit(effectivePromptIDs: canonical, cachedPromptTokens: available)) }
        }
        var editedMessages = next.messages
        editedMessages[0] = .init(role: .user, content: "Different question")
        let edited = request(editedMessages)
        #expect(cache.match(domain: domain, request: edited, renderedPromptIDs: try render(edited),
            tokenizer: tok, gemmaRecoverablePrefix: { $0 }) == .miss)
    }
}
