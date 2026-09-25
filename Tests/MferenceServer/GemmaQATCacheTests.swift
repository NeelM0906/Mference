import Foundation
import Testing
@testable import Mference
@testable import MferenceServerCore

@Suite struct GemmaQATCacheTests {
    static func domain(_ tokenizer: MFTokenizer, model: String = CheckpointIdentity.gemma4QAT) throws -> ServerPromptCacheDomain {
        .init(modelID: model, sourceSnapshotHash: "source", runtimeProfileHash: "runtime",
            maximumContext: 4096, kvStorage: "fp16", fp16RingEnabled: true,
            templateSHA256: Sha256Verifier.hashData(try tokenizer.effectiveGemmaChatTemplateData()))
    }

    @Test(arguments: [false, true], [false, true])
    func toolResultsReuseOnlyTheActualSourcePrefix(thinking: Bool, newlineBeforeClose: Bool) async throws {
        let tok = try await GemmaQATServerTests.tokenizer()
        let domain = try Self.domain(tok)
        #expect(domain.templateSHA256 == GemmaQATCheckpoint.chatTemplateSHA256)
        let tools: [MFTokenizer.FunctionDefinition] = [.init(name: "lookup", description: "Lookup",
            parameters: .object(["type": .string("object"), "properties": .object([:])]))]
        func request(_ messages: [MFTokenizer.Message]) -> ValidatedChatRequest {
            .init(reasoningEffort: thinking ? .medium : .off, messages: messages, tools: tools,
                stream: false, includeUsage: false, generationConfig: .init(maxNewTokens: 32, temperature: 0),
                maximumCompletionTokens: 32)
        }
        func render(_ value: ValidatedChatRequest) throws -> [Int32] {
            try tok.encodeChat(messages: value.messages, tools: value.tools, reasoningEffort: value.reasoningEffort)
        }
        let first = request([.init(role: .user, content: "Lookup")])
        let prompt = try render(first)
        let call = ParsedToolCall(id: "call_1", name: "lookup", arguments: .object([:]), argumentsJSON: "{}")
        let thought = thinking ? "<|channel>thought\nCheck" + (newlineBeforeClose ? "\n" : "") + "<channel|>" : ""
        let kv = prompt + tok.encode(thought + "<|tool_call>call:lookup{}<tool_call|>", addBOS: false)
        var cache = ServerPromptCache()
        cache.publish(domain: domain, request: first, content: "", calls: [call],
            result: RawDecodeResult(prefillTokens: prompt.count, cachedPromptTokens: 0,
                computedPrefillTokens: prompt.count, prefillSeconds: 0, newTokens: 1, decodeSeconds: 0,
                reason: .toolCalls, kvPosition: kv.count, kvBackedTokenIDs: kv,
                uncommittedBoundaryTokenIDs: [tok.toolResponseID], prefillExecution: nil),
            reasoningContent: thinking ? "Check" : nil)
        let next = request(first.messages + [
            .init(role: .assistant, content: nil,
                toolCalls: [.init(id: call.id, name: call.name, arguments: call.arguments)],
                reasoningContent: thinking ? "Check" : nil),
            .init(role: .tool, content: "Found", toolCallID: call.id)])
        let fresh = try render(next)
        #expect(fresh.last == tok.toolResponseEndID)
        let common = zip(kv, fresh).prefix { $0 == $1 }.count
        #expect(common > 0)
        if common < kv.count {
            #expect(cache.match(domain: domain, request: next, renderedPromptIDs: fresh, tokenizer: tok,
                gemmaRecoverablePrefix: { _ in 0 }) == .miss)
        }
        #expect(cache.match(domain: domain, request: next, renderedPromptIDs: fresh, tokenizer: tok,
            gemmaRecoverablePrefix: { $0 }) == .hit(effectivePromptIDs: fresh, cachedPromptTokens: common))
        let original = try await GemmaQATServerTests.tokenizer(qat: false)
        #expect(cache.match(domain: try Self.domain(original), request: next, renderedPromptIDs: fresh,
            tokenizer: tok, gemmaRecoverablePrefix: { $0 }) == .miss)
        #expect(cache.match(domain: try Self.domain(tok, model: "original"), request: next,
            renderedPromptIDs: fresh, tokenizer: tok, gemmaRecoverablePrefix: { $0 }) == .miss)
    }

    @Test func laterUserReplaysSourceHistoryWithoutOrdinaryReasoning() async throws {
        let tok = try await GemmaQATServerTests.tokenizer()
        let domain = try Self.domain(tok)
        func request(_ messages: [MFTokenizer.Message]) -> ValidatedChatRequest {
            .init(reasoningEffort: .medium, messages: messages, tools: [], stream: false,
                includeUsage: false, generationConfig: .init(maxNewTokens: 32, temperature: 0),
                maximumCompletionTokens: 32)
        }
        let first = request([.init(role: .user, content: "Hi")])
        let prompt = try tok.encodeChat(messages: first.messages, reasoningEffort: .medium)
        let kv = prompt + tok.encode("<|channel>thought\nCheck.<channel|>Hello.", addBOS: false)
        var cache = ServerPromptCache()
        cache.publish(domain: domain, request: first, content: "Hello.", calls: [],
            result: RawDecodeResult(prefillTokens: prompt.count, cachedPromptTokens: 0,
                computedPrefillTokens: prompt.count, prefillSeconds: 0, newTokens: 1, decodeSeconds: 0,
                reason: .endOfTurn, kvPosition: kv.count, kvBackedTokenIDs: kv,
                uncommittedBoundaryTokenIDs: [tok.endOfTurnID], prefillExecution: nil), reasoningContent: "Check.")
        let next = request(first.messages + [.init(role: .assistant, content: "Hello.", reasoningContent: "Check."),
            .init(role: .user, content: "Next")])
        let fresh = try tok.encodeChat(messages: next.messages, reasoningEffort: .medium)
        #expect(!tok.decode(fresh, skipSpecialTokens: false).contains("Check."))
        let common = zip(kv, fresh).prefix { $0 == $1 }.count
        #expect(common > 0 && common < kv.count)
        #expect(cache.match(domain: domain, request: next, renderedPromptIDs: fresh, tokenizer: tok,
            gemmaRecoverablePrefix: { $0 }) == .hit(effectivePromptIDs: fresh, cachedPromptTokens: common))
        cache.invalidate()
        #expect(cache.match(domain: domain, request: next, renderedPromptIDs: fresh, tokenizer: tok,
            gemmaRecoverablePrefix: { $0 }) == .miss)
    }
}
