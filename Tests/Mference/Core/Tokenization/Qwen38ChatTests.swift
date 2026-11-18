import Foundation
import Testing
@testable import Mference

/// Base Qwen 3.8 27B (`qwen3.8-27b-4bit`) chat and structured-decoder cases:
/// the visible answer after a tool-history suffix, XML tool calls after the
/// thought, tool examples inside reasoning, and a string-typed tool argument
/// through the decoder and the rendered history. Moved from the removed
/// Swift-Qwen suite, which ran each case for both checkpoints; these are the
/// base-checkpoint runs, on the synthetic ChatML fixture.
@Suite("Qwen 3.8 chat and structured decoding")
struct Qwen38ChatTests {
    typealias Message = MFTokenizer.Message

    private func tokenizer() async throws -> MFTokenizer {
        try await MFTokenizer.load(from: try ChatMLTemplateTests.fixtureFolder(), family: .qwen38)
            .forCheckpoint(CheckpointIdentity.baseQwen38)
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

    @Test func toolHistorySuffixDoesNotHideBaseQwenVisibleAnswer() async throws {
        let tok = try await tokenizer()
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

    @Test(arguments: [false, true])
    func thinkingDoesNotSelectJSONToolSyntax(startsInThought: Bool) async throws {
        let tok = try await tokenizer()
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

    @Test func reasoningToolExamplesCannotEmitCallsOrFailParsing() async throws {
        let tok = try await tokenizer()
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

    @Test func toolSchemaPreservesStringThroughDecoderAndHistory() async throws {
        let tok = try await tokenizer()
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
}
