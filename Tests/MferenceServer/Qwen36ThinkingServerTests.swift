import Foundation
import Testing
@testable import Mference
@testable import MferenceServerCore

@Suite("Qwen 3.6 opt-in thinking (server)")
struct Qwen36ThinkingServerTests {
    private func decoded(effort: String?) throws -> OpenAIChatRequest {
        let field = effort.map { "\"reasoning_effort\":\"\($0)\"," } ?? ""
        let data = Data("""
        {"model":"qwen3.6-35b-a3b",\(field)"messages":[
          {"role":"user","content":"A"},
          {"role":"assistant","content":"B","reasoning_content":"Check A"},
          {"role":"user","content":"C"}]}
        """.utf8)
        return try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
    }

    @Test func effortIsAcceptedWhenTheBackendSupportsIt() throws {
        for effort in ["xhigh", "medium", "low", "none"] {
            let request = try decoded(effort: effort)
            let validated = try OpenAIRequestValidator.validate(
                request, modelID: request.model, dialect: .chatml,
                acceptsReasoningEffort: true)
            #expect(validated.reasoningEffort?.rawValue == effort)
            #expect(validated.messages[1].reasoningContent == "Check A")
        }
    }

    @Test func effortIsStillRejectedForBackendsWithoutIt() throws {
        let request = try decoded(effort: "medium")
        #expect(throws: ServerRequestError.self) {
            try OpenAIRequestValidator.validate(
                request, modelID: request.model, dialect: .chatml,
                acceptsReasoningEffort: false)
        }
        #expect(throws: ServerRequestError.self) {
            try OpenAIRequestValidator.validate(
                request, modelID: request.model, dialect: .chatml)
        }
    }

    private func decoded(kwargs: String, effort: String? = nil) throws -> OpenAIChatRequest {
        let field = effort.map { "\"reasoning_effort\":\"\($0)\"," } ?? ""
        let data = Data("""
        {"model":"qwen3.6-35b-a3b",\(field)"chat_template_kwargs":\(kwargs),
         "messages":[{"role":"user","content":"A"}]}
        """.utf8)
        return try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
    }

    /// The model card's own switch, as vLLM and SGLang clients send it.
    @Test func chatTemplateKwargsSelectThinking() throws {
        for (kwargs, expected) in [
            ("{\"enable_thinking\":true,\"preserve_thinking\":true}", QwenReasoningEffort.xhigh),
            ("{\"enable_thinking\":false}", .off),
        ] {
            let request = try decoded(kwargs: kwargs)
            let validated = try OpenAIRequestValidator.validate(
                request, modelID: request.model, dialect: .chatml,
                acceptsReasoningEffort: true)
            #expect(validated.reasoningEffort == expected, "\(kwargs)")
        }
        let untouched = try decoded(kwargs: "{\"preserve_thinking\":true}")
        #expect(try OpenAIRequestValidator.validate(
            untouched, modelID: untouched.model, dialect: .chatml,
            acceptsReasoningEffort: true).reasoningEffort == nil)
    }

    /// The model card asks for a 32,768-token output budget in thinking mode;
    /// the 4,096 default would cut a long thought off before any answer.
    @Test func thinkingRaisesTheDefaultCompletionBudget() throws {
        func budget(_ kwargs: String, extra: String = "") throws -> Int {
            let data = Data("""
            {"model":"qwen3.6-35b-a3b",\(extra)"chat_template_kwargs":\(kwargs),
             "messages":[{"role":"user","content":"A"}]}
            """.utf8)
            let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
            return try OpenAIRequestValidator.validate(
                request, modelID: request.model, dialect: .chatml,
                acceptsReasoningEffort: true).maximumCompletionTokens
        }
        #expect(try budget("{\"enable_thinking\":true}") == 32_768)
        #expect(try budget("{\"enable_thinking\":false}") == 4_096)
        #expect(try budget("{}") == 4_096)
        #expect(try budget("{\"enable_thinking\":true}", extra: "\"max_tokens\":900,") == 900)
    }

    @Test func explicitEffortWinsOverChatTemplateKwargs() throws {
        let request = try decoded(kwargs: "{\"enable_thinking\":false}", effort: "low")
        let validated = try OpenAIRequestValidator.validate(
            request, modelID: request.model, dialect: .chatml,
            acceptsReasoningEffort: true)
        #expect(validated.reasoningEffort == .low)
    }

    @Test func chatTemplateKwargsAreRejectedForBackendsWithoutThinkingControl() throws {
        let request = try decoded(kwargs: "{\"enable_thinking\":true}")
        #expect(throws: ServerRequestError.self) {
            try OpenAIRequestValidator.validate(
                request, modelID: request.model, dialect: .chatml,
                acceptsReasoningEffort: false)
        }
    }

    /// The legacy hand-written continuation bridge ends in the non-thinking
    /// generation prompt. A thinking request must never take it: the decoder
    /// would start in thought and swallow the whole visible answer.
    @Test func thinkingRequestNeverTakesTheLegacyBridge() async throws {
        let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Mference/Core/Tokenization/Fixtures/ChatMLTokenizer")
        let tok = try await MFTokenizer.load(from: fixture, family: .qwen36)
        let domain = ServerPromptCacheDomain(
            modelID: "qwen3.6-35b-a3b", sourceSnapshotHash: "source",
            runtimeProfileHash: "profile", maximumContext: 16_384,
            kvStorage: "fp16", fp16RingEnabled: true, templateSHA256: "template")
        let initial = request(messages: [.init(role: .user, content: "A")], effort: .medium)
        var cache = ServerPromptCache()
        cache.publish(domain: domain, request: initial, content: "B", calls: [],
                      result: rawResult(kvBacked: [1, 2], boundary: tok.endOfTurnID),
                      reasoningContent: "Check A")
        let continuation = request(messages: initial.messages + [
            .init(role: .assistant, content: "B", reasoningContent: "Check A"),
            .init(role: .user, content: "C"),
        ], effort: .medium)
        guard case .hit(let effective, let cached) = cache.match(
            domain: domain, request: continuation,
            renderedPromptIDs: [1, 9, 3, 4], tokenizer: tok) else {
            Issue.record("the generated turn should be continued from the KV cache")
            return
        }
        #expect(cached == 2)
        #expect(Array(effective.prefix(2)) == [1, 2])
        #expect(tok.decode(Array(effective.dropFirst(2)), skipSpecialTokens: false)
                == "<|im_end|>\n<|im_start|>user\nC<|im_end|>\n<|im_start|>assistant\n<think>\n")
    }

    /// The split-stage shape: system + user, then rounds of
    /// assistant tool call -> tool result. Every round after the first must
    /// continue from the KV cache and prefill only the new tool result.
    @Test func toolLoopRoundsContinueFromTheCache() async throws {
        let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Mference/Core/Tokenization/Fixtures/ChatMLTokenizer")
        let tok = try await MFTokenizer.load(from: fixture, family: .qwen36)
        let domain = ServerPromptCacheDomain(
            modelID: "qwen3.6-35b-a3b", sourceSnapshotHash: "source",
            runtimeProfileHash: "profile", maximumContext: 16_384,
            kvStorage: "fp16", fp16RingEnabled: true, templateSHA256: "template")
        let tools: [MFTokenizer.FunctionDefinition] = [
            .init(name: "mark", description: "Mark",
                  parameters: .object(["type": .string("object")])),
        ]
        func call(_ n: Int) -> ParsedToolCall {
            ParsedToolCall(id: "call_\(n)", name: "mark",
                           arguments: .object(["part": .integer(Int64(n))]),
                           argumentsJSON: "{\"part\":\(n)}")
        }
        func turn(_ n: Int) -> [MFTokenizer.Message] {
            [.init(role: .assistant, content: nil,
                   toolCalls: [.init(id: "call_\(n)", name: "mark",
                                     arguments: .object(["part": .integer(Int64(n))]))],
                   reasoningContent: "Plan \(n)."),
             .init(role: .tool, content: "ok \(n)", toolCallID: "call_\(n)")]
        }
        let opening: [MFTokenizer.Message] = [
            .init(role: .system, content: "Split the note."),
            .init(role: .user, content: "Note"),
        ]

        // Round 1 generated tokens the template would never render byte-for-byte.
        var cache = ServerPromptCache()
        let round1 = request(messages: opening, tools: tools, effort: .medium)
        let kv1: [Int32] = [11, 12, 13, 14]
        cache.publish(domain: domain, request: round1, content: "", calls: [call(1)],
                      result: rawResult(kvBacked: kv1, boundary: tok.endOfTurnID),
                      reasoningContent: "Plan 1.")

        let round2 = request(messages: opening + turn(1), tools: tools, effort: .medium)
        let rendered2 = try tok.encodeChat(messages: round2.messages, tools: tools,
                                           reasoningEffort: .medium)
        guard case .hit(let effective2, let cached2) = cache.match(
            domain: domain, request: round2, renderedPromptIDs: rendered2, tokenizer: tok) else {
            Issue.record("round 2 should continue from the cache")
            return
        }
        #expect(cached2 == kv1.count)
        #expect(Array(effective2.prefix(kv1.count)) == kv1)
        let tail2 = tok.decode(Array(effective2.dropFirst(kv1.count)), skipSpecialTokens: false)
        #expect(tail2.hasPrefix("<|im_end|>\n<|im_start|>user\n<tool_response>\nok 1\n</tool_response>"))
        #expect(tail2.hasSuffix("<|im_start|>assistant\n<think>\n"))

        // Round 3 continues from round 2's cache the same way.
        let kv2 = effective2 + [21, 22]
        cache.publish(domain: domain, request: round2, content: "", calls: [call(2)],
                      result: rawResult(kvBacked: kv2, boundary: tok.endOfTurnID),
                      reasoningContent: "Plan 2.")
        let round3 = request(messages: opening + turn(1) + turn(2), tools: tools, effort: .medium)
        let rendered3 = try tok.encodeChat(messages: round3.messages, tools: tools,
                                           reasoningEffort: .medium)
        guard case .hit(let effective3, let cached3) = cache.match(
            domain: domain, request: round3, renderedPromptIDs: rendered3, tokenizer: tok) else {
            Issue.record("round 3 should continue from the cache")
            return
        }
        #expect(cached3 == kv2.count)
        #expect(tok.decode(Array(effective3.dropFirst(kv2.count)), skipSpecialTokens: false)
                    .contains("ok 2"))

        // A client that drops or edits the turn gets a clean miss, never a
        // cache that disagrees with its history.
        var edited = turn(2)
        edited[0] = .init(role: .assistant, content: nil, toolCalls: edited[0].toolCalls,
                          reasoningContent: nil)
        let tampered = request(messages: opening + turn(1) + edited, tools: tools, effort: .medium)
        #expect(cache.match(domain: domain, request: tampered,
                            renderedPromptIDs: [1, 2, 3], tokenizer: tok) == .miss)
    }

    private func request(messages: [MFTokenizer.Message],
                         tools: [MFTokenizer.FunctionDefinition] = [],
                         effort: QwenReasoningEffort?) -> ValidatedChatRequest {
        ValidatedChatRequest(
            reasoningEffort: effort,
            messages: messages,
            tools: tools,
            stream: false,
            includeUsage: false,
            generationConfig: GenerationConfig(maxNewTokens: 16, temperature: 0),
            maximumCompletionTokens: 16)
    }

    private func rawResult(kvBacked: [Int32], boundary: Int32) -> RawDecodeResult {
        RawDecodeResult(
            prefillTokens: 1,
            cachedPromptTokens: 0,
            computedPrefillTokens: 1,
            prefillSeconds: 0,
            newTokens: 1,
            decodeSeconds: 0,
            reason: .endOfTurn,
            kvPosition: kvBacked.count,
            kvBackedTokenIDs: kvBacked,
            uncommittedBoundaryTokenIDs: [boundary],
            prefillExecution: nil)
    }
}
