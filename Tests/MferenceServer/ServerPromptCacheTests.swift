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
    @Test func swiftQwenHistoryIsPreservedAndBaseDomainCannotReuseIt() async throws {
        let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Mference/Core/Tokenization/Fixtures/ChatMLTokenizer")
        let tok = try await MFTokenizer.load(from: fixture, family: .qwen38)
            .forCheckpoint(CheckpointIdentity.swiftQwen38)
        let initial = request(messages: [.init(role: .user, content: "A")])
        func checkpointDomain(_ id: String) -> ServerPromptCacheDomain {
            .init(modelID: id, sourceSnapshotHash: "source", runtimeProfileHash: "profile",
                  maximumContext: 16384, kvStorage: "fp16", fp16RingEnabled: true, templateSHA256: "template")
        }
        let swift = checkpointDomain(CheckpointIdentity.swiftQwen38)
        var cache = ServerPromptCache()
        cache.publish(domain: swift, request: initial, content: "B", calls: [],
                      result: rawResult(prompt: [1], kvBacked: [1, 2], boundary: 3, reason: .endOfTurn),
                      reasoningContent: "Check A")
        #expect(cache.entry?.assistantTurn.message.reasoningContent == "Check A")
        #expect(cache.match(domain: checkpointDomain("qwen3.8-27b-4bit"), request: initial,
                            renderedPromptIDs: [1, 2, 3, 4], tokenizer: tok) == .miss)
        let continuation = request(messages: initial.messages + [
            .init(role: .assistant, content: "B", reasoningContent: "Check A"),
            .init(role: .user, content: "C")])
        #expect(cache.match(domain: swift, request: continuation,
                            renderedPromptIDs: [1, 9, 3, 4], tokenizer: tok) == .miss)
        if case .hit = cache.match(domain: swift, request: continuation,
                                    renderedPromptIDs: [1, 2, 3, 4], tokenizer: tok) {} else {
            Issue.record("an exact Swift prefix should remain reusable")
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

    @Test func textContinuationUsesActualGeneratedHistoryAndOnlyPrefillsSuffix() async throws {
        let tokenizer = try await MFTokenizer.load()
        let initial = request(messages: [
            MFTokenizer.Message(role: .user, content: "first"),
        ])
        let initialPrompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        let generated = tokenizer.encode("answer", addBOS: false)
        let kvBacked = initialPrompt + generated
        var cache = ServerPromptCache()
        cache.publish(
            domain: domain,
            request: initial,
            content: "answer",
            calls: [],
            result: rawResult(
                prompt: initialPrompt,
                kvBacked: kvBacked,
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))

        let continuation = request(messages: initial.messages + [
            MFTokenizer.Message(role: .assistant, content: "answer"),
            MFTokenizer.Message(role: .user, content: "second"),
        ])
        let rendered = tokenizer.encode(
            try tokenizer.applyChatTemplate(continuation.messages),
            addBOS: false)
        let match = cache.match(
            domain: domain,
            request: continuation,
            renderedPromptIDs: rendered,
            tokenizer: tokenizer)

        guard case .hit(let effective, let cached) = match else {
            Issue.record("expected text continuation hit")
            return
        }
        let bridge = tokenizer.encodeTextContinuation(userContent: "second")
        #expect(cached == kvBacked.count)
        #expect(effective == kvBacked + bridge)
        #expect(!rendered.prefix(kvBacked.count).elementsEqual(kvBacked))
        #expect(effective[cached] == tokenizer.endOfTurnID)
    }

    @Test func capturedOpenCodeToolResultUsesFrozenToolBoundary() async throws {
        let tokenizer = try await MFTokenizer.load()
        let initial = try validatedFixture("opencode-1.15.11-initial.json")
        let continuation = try validatedFixture("opencode-1.15.11-tool-result.json")
        let initialPrompt = try tokenizer.encodeToolChat(
            messages: initial.messages,
            tools: initial.tools)
        let assistant = continuation.messages[initial.messages.count]
        let prefix = try tokenizer.encodeToolChat(
            messages: initial.messages + [assistant],
            tools: initial.tools)
        let callStart = try #require(prefix.lastIndex(of: tokenizer.toolCallStartID))
        let callEnd = try #require(prefix.lastIndex(of: tokenizer.toolCallEndID))
        let generatedCall = Array(prefix[callStart...callEnd])
        let kvBacked = initialPrompt + generatedCall
        let historicalCall = try #require(assistant.toolCalls.first)
        let parsedCall = ParsedToolCall(
            id: historicalCall.id,
            name: historicalCall.name,
            arguments: historicalCall.arguments,
            argumentsJSON: try historicalCall.arguments.encoded())
        var cache = ServerPromptCache()
        cache.publish(
            domain: domain,
            request: initial,
            content: "",
            calls: [parsedCall],
            result: rawResult(
                prompt: initialPrompt,
                kvBacked: kvBacked,
                boundary: tokenizer.toolResponseID,
                reason: .toolCalls))
        let rendered = try tokenizer.encodeToolChat(
            messages: continuation.messages,
            tools: continuation.tools)

        let match = cache.match(
            domain: domain,
            request: continuation,
            renderedPromptIDs: rendered,
            tokenizer: tokenizer)

        guard case .hit(let effective, let cached) = match else {
            Issue.record("expected captured OpenCode tool-result hit")
            return
        }
        let bridge = try tokenizer.encodeToolResultContinuation(
            cachedMessages: initial.messages,
            assistant: assistant,
            incomingMessages: continuation.messages,
            tools: continuation.tools)
        #expect(cached == kvBacked.count)
        #expect(effective == kvBacked + bridge)
        #expect(bridge.first == tokenizer.toolResponseID)
        #expect(!rendered.prefix(kvBacked.count).elementsEqual(kvBacked))
    }

    @Test func mismatchedLineageDomainAndUnsafeStopsMiss() async throws {
        let tokenizer = try await MFTokenizer.load()
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
        let tokenizer = try await MFTokenizer.load()
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
        return try OpenAIRequestValidator.validate(
            request,
            modelID: "gemma-4-26b-a4b-it")
    }
}
