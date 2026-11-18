import Foundation
import Testing

@testable import Mference
@testable import MferenceServerCore

@Suite("Server prompt cache")
struct ServerPromptCacheTests {
    @Test(arguments: [false, true], [false, true])
    func explicitBaseEffortNeverUsesLegacyBridge(before: Bool, after: Bool) async throws {
        let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Mference/Core/Tokenization/Fixtures/ChatMLTokenizer")
        let tok = try await MFTokenizer.load(from: fixture, family: .qwen38)
            .forCheckpoint(CheckpointIdentity.baseQwen38)
        var initial = request(messages: [.init(role: .user, content: "A")])
        initial.reasoningEffort = before ? .low : nil
        var cache = ServerPromptCache()
        cache.publish(domain: domain, request: initial, content: "B", calls: [],
            result: rawResult(prompt: [1], kvBacked: [1, 2], boundary: tok.endOfTurnID, reason: .endOfTurn))
        var continuation = request(messages: initial.messages + [.init(role: .assistant, content: "B"),
            .init(role: .user, content: "C")])
        continuation.reasoningEffort = after ? .medium : nil
        let match = cache.match(domain: domain, request: continuation,
            renderedPromptIDs: [1, 9, 3, 4], tokenizer: tok)
        if before || after { #expect(match == .miss) }
        else if case .hit = match {} else { Issue.record("Legacy bridge must remain available") }
        if case .hit = cache.match(domain: domain, request: continuation,
            renderedPromptIDs: [1, 2, 3, 4], tokenizer: tok) {} else {
            Issue.record("Exact rendered prefixes remain reusable")
        }
    }
    /// A source-template (explicit-effort) Qwen 3.8 turn keeps its reasoning
    /// history, reuses only an exact rendered prefix, and is never reused
    /// across cache domains (here another checkpoint's model ID).
    @Test func baseQwen38HistoryIsPreservedAndOtherDomainCannotReuseIt() async throws {
        let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Mference/Core/Tokenization/Fixtures/ChatMLTokenizer")
        let tok = try await MFTokenizer.load(from: fixture, family: .qwen38)
            .forCheckpoint(CheckpointIdentity.baseQwen38)
        var initial = request(messages: [.init(role: .user, content: "A")])
        initial.reasoningEffort = .low
        func checkpointDomain(_ id: String) -> ServerPromptCacheDomain {
            .init(modelID: id, sourceSnapshotHash: "source", runtimeProfileHash: "profile",
                  maximumContext: 16384, kvStorage: "fp16", fp16RingEnabled: true, templateSHA256: "template")
        }
        let base = checkpointDomain(CheckpointIdentity.baseQwen38)
        var cache = ServerPromptCache()
        cache.publish(domain: base, request: initial, content: "B", calls: [],
                      result: rawResult(prompt: [1], kvBacked: [1, 2], boundary: 3, reason: .endOfTurn),
                      reasoningContent: "Check A")
        #expect(cache.entry?.assistantTurn.message.reasoningContent == "Check A")
        #expect(cache.match(domain: checkpointDomain("qwen3.6-35b-a3b"), request: initial,
                            renderedPromptIDs: [1, 2, 3, 4], tokenizer: tok) == .miss)
        var continuation = request(messages: initial.messages + [
            .init(role: .assistant, content: "B", reasoningContent: "Check A"),
            .init(role: .user, content: "C")])
        continuation.reasoningEffort = .low
        #expect(cache.match(domain: base, request: continuation,
                            renderedPromptIDs: [1, 9, 3, 4], tokenizer: tok) == .miss)
        if case .hit = cache.match(domain: base, request: continuation,
                                    renderedPromptIDs: [1, 2, 3, 4], tokenizer: tok) {} else {
            Issue.record("an exact base Qwen 3.8 prefix should remain reusable")
        }
        cache.invalidate()
        #expect(cache.entry == nil)
    }
    private let domain = ServerPromptCacheDomain(
        modelID: "model",
        sourceSnapshotHash: "snapshot",
        runtimeProfileHash: "profile",
        maximumContext: 16_384,
        kvStorage: "fp16",
        fp16RingEnabled: true,
        templateSHA256: "template")

    /// The captured OpenCode tool-result turn under ChatML: no dialect can
    /// bridge a cached tool-call turn by token boundary, so a KV prefix that
    /// the re-render does not reproduce misses, while one it does reproduce is
    /// reused as an exact rendered prefix.
    @Test func capturedOpenCodeToolResultReusesOnlyAnExactRenderedPrefix() async throws {
        let tokenizer = try await ChatMLTokenizerFixture.load()
        let initial = try validatedFixture("opencode-1.15.11-initial.json")
        let continuation = try validatedFixture("opencode-1.15.11-tool-result.json")
        let initialPrompt = try tokenizer.encodeToolChat(
            messages: initial.messages,
            tools: initial.tools)
        let assistant = continuation.messages[initial.messages.count]
        let historicalCall = try #require(assistant.toolCalls.first)
        let parsedCall = ParsedToolCall(
            id: historicalCall.id,
            name: historicalCall.name,
            arguments: historicalCall.arguments,
            argumentsJSON: try historicalCall.arguments.encoded())
        let rendered = try tokenizer.encodeToolChat(
            messages: continuation.messages,
            tools: continuation.tools)

        // What the model generated is not what the re-render produces.
        let generated = initialPrompt + [tokenizer.toolCallStartID, tokenizer.toolCallEndID]
        #expect(!rendered.prefix(generated.count).elementsEqual(generated))
        var cache = ServerPromptCache()
        cache.publish(
            domain: domain,
            request: initial,
            content: "",
            calls: [parsedCall],
            result: rawResult(
                prompt: initialPrompt,
                kvBacked: generated,
                boundary: tokenizer.toolResponseID,
                reason: .toolCalls))
        #expect(cache.match(
            domain: domain,
            request: continuation,
            renderedPromptIDs: rendered,
            tokenizer: tokenizer) == .miss)

        // A KV prefix the re-render reproduces token for token is reused.
        let exact = Array(rendered.dropLast())
        cache.publish(
            domain: domain,
            request: initial,
            content: "",
            calls: [parsedCall],
            result: rawResult(
                prompt: initialPrompt,
                kvBacked: exact,
                boundary: tokenizer.toolResponseID,
                reason: .toolCalls))
        guard case .hit(let effective, let cached) = cache.match(
            domain: domain,
            request: continuation,
            renderedPromptIDs: rendered,
            tokenizer: tokenizer) else {
            Issue.record("an exact rendered prefix must be reused")
            return
        }
        #expect(effective == rendered)
        #expect(cached == exact.count)
    }

    @Test func mismatchedLineageDomainAndUnsafeStopsMiss() async throws {
        let tokenizer = try await ChatMLTokenizerFixture.load()
        let initial = request(messages: [
            MFTokenizer.Message(role: .user, content: "first"),
        ])
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        var cache = ServerPromptCache()

        for reason in [StopReason.stopString, .eos] {
            cache.publish(
                domain: domain,
                request: initial,
                content: "answer",
                calls: [],
                result: rawResult(
                    prompt: prompt,
                    kvBacked: prompt,
                    boundary: tokenizer.eosID,
                    reason: reason))
            #expect(cache.entry == nil)
        }

        cache.publish(
            domain: domain,
            request: initial,
            content: "answer",
            calls: [],
            result: rawResult(
                prompt: prompt,
                kvBacked: prompt + tokenizer.encode("answer", addBOS: false),
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))
        let changed = request(messages: [
            MFTokenizer.Message(role: .user, content: "changed"),
            MFTokenizer.Message(role: .assistant, content: "answer"),
            MFTokenizer.Message(role: .user, content: "second"),
        ])
        let rendered = tokenizer.encode(
            try tokenizer.applyChatTemplate(changed.messages),
            addBOS: false)
        #expect(cache.match(
            domain: domain,
            request: changed,
            renderedPromptIDs: rendered,
            tokenizer: tokenizer) == .miss)
    }

    @Test func tailCompletedStopStringDoesNotPublishPrefix() async throws {
        let tokenizer = try await ChatMLTokenizerFixture.load()
        let initial = request(messages: [
            MFTokenizer.Message(role: .user, content: "first"),
        ])
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        var matcher = StreamingStopMatcher(stops: ["🌳stop"])
        #expect(matcher.push("answer 🌳") == "answer ")
        #expect(matcher.push("stop") == "")
        #expect(matcher.isStopped)

        var cache = ServerPromptCache()
        cache.publish(
            domain: domain,
            request: initial,
            content: "answer ",
            calls: [],
            result: rawResult(
                prompt: prompt,
                kvBacked: prompt,
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn),
            stopStringFiltered: matcher.isStopped)
        #expect(cache.entry == nil)
    }

    @Test func aHitWithARetainedThoughtThatNoLongerFitsFallsBackToTheRenderedPrompt() async throws {
        let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Mference/Core/Tokenization/Fixtures/ChatMLTokenizer")
        let tok = try await MFTokenizer.load(from: fixture, family: .qwen38)
            .forCheckpoint(CheckpointIdentity.baseQwen38)
        let initial = request(messages: [.init(role: .user, content: "A")])
        let prompt = try tok.encodeChat(messages: initial.messages)
        // The KV cache keeps the generated thought; the next render drops it.
        let generated = tok.encode(String(repeating: "step ", count: 32) + "\n</think>\n\nB", addBOS: false)
        var cache = ServerPromptCache()
        cache.publish(domain: domain, request: initial, content: "B", calls: [],
            result: rawResult(prompt: prompt, kvBacked: prompt + generated,
                              boundary: tok.endOfTurnID, reason: .endOfTurn))
        let next = request(messages: initial.messages + [
            .init(role: .assistant, content: "B"), .init(role: .user, content: "C")])
        let rendered = try tok.encodeChat(messages: next.messages)
        let match = cache.match(domain: domain, request: next, renderedPromptIDs: rendered, tokenizer: tok)
        guard case .hit(let effective, _) = match else {
            Issue.record("expected the legacy continuation to reuse the prefix")
            return
        }
        #expect(effective.count > rendered.count)
        // `prepare` admits the rendered prompt against this context.
        #expect(match.fitting(maxContext: rendered.count + 1) == .miss)
        #expect(match.fitting(maxContext: effective.count + 1) == match)
    }

    private func request(
        messages: [MFTokenizer.Message],
        tools: [MFTokenizer.FunctionDefinition] = []
    ) -> ValidatedChatRequest {
        ValidatedChatRequest(
            messages: messages,
            tools: tools,
            stream: false,
            includeUsage: false,
            generationConfig: GenerationConfig(maxNewTokens: 16, temperature: 0),
            maximumCompletionTokens: 16)
    }

    private func rawResult(
        prompt: [Int32],
        kvBacked: [Int32],
        boundary: Int32,
        reason: StopReason
    ) -> RawDecodeResult {
        RawDecodeResult(
            prefillTokens: prompt.count,
            cachedPromptTokens: 0,
            computedPrefillTokens: prompt.count,
            prefillSeconds: 0,
            newTokens: 1,
            decodeSeconds: 0,
            reason: reason,
            kvPosition: kvBacked.count,
            kvBackedTokenIDs: kvBacked,
            uncommittedBoundaryTokenIDs: [boundary],
            prefillExecution: nil)
    }

    private func validatedFixture(_ name: String) throws -> ValidatedChatRequest {
        let url = try #require(Bundle.module.url(
            forResource: name,
            withExtension: nil,
            subdirectory: "Fixtures"))
        let request = try JSONDecoder().decode(
            OpenAIChatRequest.self,
            from: Data(contentsOf: url))
        // The validator only requires that the request names the served
        // model; the fixtures keep the identifier they were captured with.
        return try OpenAIRequestValidator.validate(
            request,
            modelID: request.model)
    }
}
