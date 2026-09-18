import CryptoKit
import Foundation
import Testing
@testable import Mference
@testable import MferenceCLICore

@Suite("Swift-Qwen source template")
struct SwiftQwenChatTests {
    typealias Message = MFTokenizer.Message

    @Test func requestControlsAndSpeculationDefaultsAreExplicit() throws {
        for effort in QwenReasoningEffort.allCases {
            let args = try Args.parse(["--model", "candidate.gturbo", "--chat", "--reasoning-effort", effort.rawValue])
            #expect(args.reasoningEffort == effort)
        }
        #expect(throws: (any Error).self) {
            try Args.parse(["--model", "candidate.gturbo", "--prompt", "Hi", "--reasoning-effort", "low"])
        }
        #expect(throws: (any Error).self) {
            try Args.parse(["--model", "candidate.gturbo", "--chat", "--reasoning-effort", "high"])
        }
        #expect(!CheckpointIdentity.qwenMTPEnabled(modelID: CheckpointIdentity.swiftQwen38, setting: nil))
        #expect(!CheckpointIdentity.qwenMTPEnabled(modelID: CheckpointIdentity.swiftQwen38, setting: "0"))
        #expect(CheckpointIdentity.qwenMTPEnabled(modelID: CheckpointIdentity.swiftQwen38, setting: "1"))
        #expect(CheckpointIdentity.qwenMTPEnabled(modelID: "qwen3.8-27b-4bit", setting: nil))
    }

    private func tokenizer() async throws -> MFTokenizer {
        let fixture = try #require(Bundle.module.url(forResource: "SwiftQwenTemplate",
                                                     withExtension: nil, subdirectory: "Fixtures"))
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.copyItem(at: ChatMLTemplateTests.fixtureFolder(), to: folder)
        defer { try? FileManager.default.removeItem(at: folder) }
        let source = try Data(contentsOf: fixture.appendingPathComponent("chat_template.jinja"))
        #expect(SHA256.hash(data: source.dropLast()).map { String(format: "%02x", $0) }.joined()
                == "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041")
        try source.write(to: folder.appendingPathComponent("chat_template.jinja"))
        return try await MFTokenizer.load(from: folder, family: .qwen38)
            .forCheckpoint(CheckpointIdentity.swiftQwen38)
    }

    private func messages(_ name: String) -> [Message] {
        switch name {
        case "history":
            return [.init(role: .system, content: " Be terse. "),
                    .init(role: .user, content: "A"),
                    .init(role: .assistant, content: "B", reasoningContent: " Check A. "),
                    .init(role: .user, content: "C")]
        case "tool_result":
            return [.init(role: .user, content: "Lookup A"),
                    .init(role: .assistant, content: nil, toolCalls: [
                        .init(id: "call_1", name: "lookup", arguments: .object(["query": .string("A")]))
                    ], reasoningContent: "Need lookup."),
                    .init(role: .tool, content: "Found A", toolCallID: "call_1")]
        default: return [.init(role: .user, content: " Hi ")]
        }
    }

    @Test func rendersMatchIndependentJinjaOracle() async throws {
        struct Oracle: Decodable { let name: String; let effort: String; let render: String }
        let tok = try await tokenizer()
        let base = try tok.forCheckpoint(CheckpointIdentity.baseQwen38)
        let url = try #require(Bundle.module.url(forResource: "oracle", withExtension: "json",
                                               subdirectory: "Fixtures/SwiftQwenTemplate"))
        let cases = try JSONDecoder().decode([Oracle].self, from: Data(contentsOf: url))
        for item in cases {
            let effort = try #require(QwenReasoningEffort(rawValue: item.effort))
            let ids = try tok.encodeChat(messages: messages(item.name), reasoningEffort: effort)
            #expect(tok.decode(ids, skipSpecialTokens: false) == item.render,
                    "\(item.name)/\(item.effort)")
            #expect(ids == tok.encode(item.render, addBOS: false))
            #expect(try base.encodeChat(messages: messages(item.name), reasoningEffort: effort) == ids)
            #expect(tok.startsInThinking(reasoningEffort: effort) == (effort != .off))
        }
        #expect(try tok.encodeChat(messages: messages("single")) ==
                tok.encodeChat(messages: messages("single"), reasoningEffort: .xhigh))
        #expect(tok.stopTokenIDs == [248046, 248044])
    }

    @Test func toolsDoNotForceThinkingOff() async throws {
        let tok = try await tokenizer()
        let tools: [MFTokenizer.FunctionDefinition] = [.init(name: "lookup", description: "Lookup",
                                                 parameters: .object(["type": .string("object")]))]
        for effort in QwenReasoningEffort.allCases {
            let ids = try tok.encodeChat(messages: messages("single"), tools: tools, reasoningEffort: effort)
            let text = tok.decode(ids, skipSpecialTokens: false)
            #expect(text.contains("<tools>"))
            #expect(text.hasSuffix(effort == .off ? "<think>\n\n</think>\n\n" : "<think>\n"))
        }
    }

    @Test func rejectsUnsupportedRolesAndDoesNotMutateBaseProfile() async throws {
        let tok = try await tokenizer()
        #expect(throws: MFTokenizerError.self) {
            try tok.encodeChat(messages: [.init(role: .developer, content: "Guide"),
                                         .init(role: .user, content: "Hi")])
        }
        let base = try tok.forCheckpoint("qwen3.8-27b-4bit")
        #expect(!base.isSwiftQwen)
        #expect(tok.isSwiftQwen)
        #expect(base.supportsQwenReasoningEffort)
        #expect(try base.encodeChat(messages: messages("single"), reasoningEffort: .low) ==
            tok.encodeChat(messages: messages("single"), reasoningEffort: .low))
        let other = try tok.forCheckpoint("qwen3.6-35b-a3b")
        #expect(!other.supportsQwenReasoningEffort)
        #expect(throws: MFTokenizerError.self) {
            try other.encodeChat(messages: messages("single"), reasoningEffort: .low)
        }
        #expect(throws: (any Error).self) { try tok.encodeChat(messages: []) }
    }

    @Test func matchedToolsAndBaseLegacyDefaults() async throws {
        let swift = try await tokenizer()
        let base = try swift.forCheckpoint(CheckpointIdentity.baseQwen38)
        let input = messages("single")
        let legacy = try base.encodeChat(messages: input)
        #expect(base.decode(legacy, skipSpecialTokens: false) == "<|im_start|>user\n Hi <|im_end|>\n<|im_start|>assistant\n<think>\n")
        let tools: [MFTokenizer.FunctionDefinition] = [.init(name: "lookup", description: "Lookup",
            parameters: .object(["type": .string("object"), "properties": .object([
                "query": .object(["type": .string("string")])])]))]
        for effort in QwenReasoningEffort.allCases {
            for name in ["single", "history", "tool_result"] {
                let actual = try base.encodeChat(messages: messages(name), tools: tools, reasoningEffort: effort)
                let expected = try swift.encodeChat(messages: messages(name), tools: tools, reasoningEffort: effort)
                #expect(actual == expected)
                #expect(base.startsInThinking(reasoningEffort: effort, promptIDs: actual) == (effort != .off))
            }
            #expect(throws: MFTokenizerError.self) {
                try base.encodeChat(messages: [.init(role: .developer, content: "Guide"),
                    .init(role: .user, content: "Hi")], reasoningEffort: effort)
            }
        }
        #expect(try base.encodeChat(messages: input) == legacy)
        #expect(base.decode(try base.encodeChat(messages: input, tools: tools), skipSpecialTokens: false)
            .hasSuffix("<think>\n\n</think>\n\n"))
    }

    @Test func toolHistorySuffixDoesNotHideBaseQwenVisibleAnswer() async throws {
        let tok = try await tokenizer().forCheckpoint("qwen3.8-27b-4bit")
        let history = messages("tool_result") + [.init(role: .assistant, content: "Found A"),
                                                 .init(role: .user, content: "What is 7 times 3?")]
        let ids = try tok.encodeChat(messages: history)
        #expect(tok.generationPromptStartsInThinking)
        #expect(tok.decode(ids, skipSpecialTokens: false).hasSuffix("<think>\n\n</think>\n\n"))
        let starts = tok.startsInThinking(reasoningEffort: nil, promptIDs: ids)
        #expect(!starts)
        let decoder = StructuredAssistantDecoder(tokenizer: tok, allowedTools: [], startsInThought: starts)
        #expect(try decoder.consume(tokenID: 12, delta: "21") == [.content("21")])
        let plain = try tok.encodeChat(messages: [.init(role: .user, content: "Hi")])
        #expect(tok.startsInThinking(reasoningEffort: nil, promptIDs: plain))
        // An earlier marker must not override an unmarked generation suffix.
        let historical = tok.encode("</think>\n<|im_start|>assistant\n", addBOS: false)
        #expect(tok.startsInThinking(reasoningEffort: nil, promptIDs: historical))
    }

    @Test(arguments: [false, true], [false, true])
    func thinkingDoesNotSelectJSONToolSyntax(startsInThought: Bool, swift: Bool) async throws {
        let candidate = try await tokenizer()
        let tok = try candidate.forCheckpoint(swift ? CheckpointIdentity.swiftQwen38 : "qwen3.8-27b-4bit")
        #expect(tok.generationPromptStartsInThinking)
        let decoder = StructuredAssistantDecoder(tokenizer: tok, allowedTools: ["add"],
                                                  startsInThought: startsInThought,
                                                  idGenerator: { "call_test" })
        if startsInThought {
            _ = try decoder.consume(tokenID: #require(tok.thinkEndID), delta: "</think>")
        }
        _ = try decoder.consume(tokenID: tok.toolCallStartID, delta: "<tool_call>")
        let payload = "\n<function=add>\n<parameter=a>\n2\n</parameter>\n<parameter=b>\n3\n</parameter>\n</function>\n"
        for id in tok.encode(payload, addBOS: false) {
            #expect(try decoder.consume(tokenID: id, delta: "") == [])
        }
        let events = try decoder.consume(tokenID: tok.toolCallEndID, delta: "</tool_call>")
        #expect(events.count == 1)
        guard case .toolCall(let call) = try #require(events.first) else {
            Issue.record("Expected a parsed tool call")
            return
        }
        #expect(call.name == "add")
        #expect(call.arguments == .object(["a": .integer(2), "b": .integer(3)]))
        _ = try decoder.finish()
    }

    @Test(arguments: [false, true])
    func reasoningToolExamplesCannotEmitCallsOrFailParsing(swift: Bool) async throws {
        let tok = try await tokenizer().forCheckpoint(swift ? CheckpointIdentity.swiftQwen38 : "qwen3.8-27b-4bit")
        let decoder = StructuredAssistantDecoder(tokenizer: tok, allowedTools: ["echo"],
            startsInThought: true, idGenerator: { "call_test" })
        var reasoning = ""
        decoder.onReasoning = { reasoning += $0 }
        // Even unknown/malformed or unclosed examples belong to reasoning.
        let example = "Consider <tool_call>not a call</tool_call> or <tool_call>another example"
        for id in tok.encode(example, addBOS: false) {
            #expect(try decoder.consume(tokenID: id, delta: tok.decode([id], skipSpecialTokens: false)).isEmpty)
        }
        #expect(reasoning == example)
        #expect(!decoder.hasToolCalls)
        _ = try decoder.consume(tokenID: #require(tok.thinkEndID), delta: "</think>")
        #expect(try decoder.consumeFlushedText("Answer") == [.content("Answer")])
        _ = try decoder.consume(tokenID: tok.toolCallStartID, delta: "<tool_call>")
        let callBody = "\n<function=echo>\n<parameter=text>\n<think>literal</think>\n</parameter>\n</function>\n"
        for id in tok.encode(callBody, addBOS: false) {
            _ = try decoder.consume(tokenID: id, delta: "")
        }
        let events = try decoder.consume(tokenID: tok.toolCallEndID, delta: "</tool_call>")
        guard case .toolCall(let call) = try #require(events.first) else {
            Issue.record("Expected visible-channel call"); return
        }
        #expect(call.arguments == .object(["text": .string("<think>literal</think>")]))
        #expect(decoder.hasToolCalls)
        #expect(try decoder.finish().isEmpty)
    }

    @Test(arguments: [false, true])
    func toolSchemaPreservesStringThroughDecoderAndHistory(swift: Bool) async throws {
        let tok = try await tokenizer().forCheckpoint(swift ? CheckpointIdentity.swiftQwen38 : "qwen3.8-27b-4bit")
        let tools: [MFTokenizer.FunctionDefinition] = [.init(name: "echo", description: "Echo text",
            parameters: .object(["type": .string("object"), "properties": .object([
                "text": .object(["type": .string("string")])])]))]
        let decoder = StructuredAssistantDecoder(tokenizer: tok, allowedTools: ["echo"],
            startsInThought: true, toolDefinitions: tools, idGenerator: { "call_echo" })
        _ = try decoder.consume(tokenID: #require(tok.thinkEndID), delta: "</think>")
        _ = try decoder.consume(tokenID: tok.toolCallStartID, delta: "<tool_call>")
        let payload = "\n<function=echo>\n<parameter=text>\n123\n</parameter>\n</function>\n"
        for id in tok.encode(payload, addBOS: false) {
            _ = try decoder.consume(tokenID: id, delta: "")
        }
        let events = try decoder.consume(tokenID: tok.toolCallEndID, delta: "</tool_call>")
        guard case .toolCall(let call) = try #require(events.first) else {
            Issue.record("Expected tool call"); return
        }
        #expect(call.arguments == .object(["text": .string("123")]))
        let history: [Message] = [.init(role: .user, content: "Echo 123"),
            .init(role: .assistant, content: nil, toolCalls: [
                .init(id: call.id, name: call.name, arguments: call.arguments)
            ], reasoningContent: "Use echo."),
            .init(role: .tool, content: "123", toolCallID: call.id)]
        let rendered = tok.decode(try tok.encodeChat(messages: history, tools: tools), skipSpecialTokens: false)
        #expect(rendered.contains("<parameter=text>\n123\n</parameter>"))
        #expect(try decoder.finish().isEmpty)
    }

    @Test(arguments: [false, true])
    func reasoningIsSeparateAndIncludesDetokenizerFlush(startsInThought: Bool) async throws {
        let tok = try await tokenizer()
        let decoder = StructuredAssistantDecoder(tokenizer: tok, allowedTools: [], startsInThought: startsInThought)
        var reasoning = ""
        decoder.onReasoning = { reasoning += $0 }
        if !startsInThought {
            #expect(try decoder.consume(tokenID: #require(tok.thinkStartID), delta: "<think>") == [])
        }
        #expect(try decoder.consume(tokenID: 12, delta: "Check") == [])
        #expect(try decoder.consumeFlushedText(" A") == [])
        #expect(try decoder.consume(tokenID: #require(tok.thinkEndID), delta: "</think>") == [])
        #expect(try decoder.consume(tokenID: 13, delta: "Answer") == [.content("Answer")])
        #expect(reasoning == "Check A")
    }
}
