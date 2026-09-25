import Foundation
import Testing
@testable import Mference

@Suite("Qwen 3.6 opt-in thinking")
struct Qwen36ThinkingTests {
    typealias Message = MFTokenizer.Message

    private func tokenizer(family: ModelFamily? = .qwen36) async throws -> MFTokenizer {
        try await MFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder(), family: family)
    }

    private let single: [Message] = [.init(role: .user, content: "Hi")]

    @Test func effortOpensOrClosesTheThinkBlock() async throws {
        let tok = try await tokenizer()
        for effort in QwenReasoningEffort.allCases {
            let ids = try tok.encodeChat(messages: single, reasoningEffort: effort)
            let text = tok.decode(ids, skipSpecialTokens: false)
            #expect(text.hasSuffix(effort == .off ? "<think>\n\n</think>\n\n" : "<think>\n"),
                    "\(effort.rawValue)")
            #expect(tok.startsInThinking(reasoningEffort: effort) == (effort != .off))
        }
    }

    @Test func omittedEffortKeepsTheNativeNonThinkingRender() async throws {
        let tok = try await tokenizer()
        #expect(try tok.encodeChat(messages: single)
                == tok.encode(try tok.applyChatTemplate(single), addBOS: false))
        #expect(!tok.startsInThinking(reasoningEffort: nil))
    }

    @Test func earlierReasoningStaysInTheRenderedHistory() async throws {
        let tok = try await tokenizer()
        let history: [Message] = [
            .init(role: .user, content: "A"),
            .init(role: .assistant, content: "B", reasoningContent: "Check A."),
            .init(role: .user, content: "C"),
        ]
        let text = tok.decode(try tok.encodeChat(messages: history, reasoningEffort: .medium),
                              skipSpecialTokens: false)
        #expect(text.contains("<think>\nCheck A.\n</think>\n\nB"))
        #expect(text.hasSuffix("<think>\n"))
    }

    @Test func toolsDoNotForceThinkingOff() async throws {
        let tok = try await tokenizer()
        let tools: [MFTokenizer.FunctionDefinition] = [
            .init(name: "lookup", description: "Lookup",
                  parameters: .object(["type": .string("object")])),
        ]
        let text = tok.decode(
            try tok.encodeChat(messages: single, tools: tools, reasoningEffort: .medium),
            skipSpecialTokens: false)
        #expect(text.contains("<tools>"))
        #expect(text.hasSuffix("<think>\n"))
    }

    // MARK: - KV continuation after a cached assistant turn

    private let lookup: [MFTokenizer.FunctionDefinition] = [
        .init(name: "lookup", description: "Lookup",
              parameters: .object(["type": .string("object")])),
    ]

    private func toolLoop(results: Int) -> [Message] {
        let calls = (1...results).map {
            MFTokenizer.HistoricalToolCall(id: "call_\($0)", name: "lookup",
                                           arguments: .object(["query": .string("A\($0)")]))
        }
        return [.init(role: .system, content: "Be terse."),
                .init(role: .user, content: "Lookup A"),
                .init(role: .assistant, content: nil, toolCalls: calls,
                      reasoningContent: "Need lookup.")]
            + (1...results).map {
                Message(role: .tool, content: "Found A\($0)", toolCallID: "call_\($0)")
            }
    }

    /// The continuation is what follows the cached turn in the full render, so
    /// appending it to a KV cache that holds the generated turn reproduces the
    /// rest of the prompt without re-rendering that turn.
    @Test(arguments: [1, 2])
    func toolResultContinuationIsTheTailOfTheFullRender(results: Int) async throws {
        let tok = try await tokenizer()
        let messages = toolLoop(results: results)
        let full = try tok.encodeChat(messages: messages, tools: lookup, reasoningEffort: .medium)
        let tail = try tok.encodeSourceTemplateContinuation(
            messages: messages, cachedTurnIndex: 2, tools: lookup, reasoningEffort: .medium)
        #expect(tail.first == tok.endOfTurnID)
        #expect(Array(full.suffix(tail.count)) == tail)
        let text = tok.decode(tail, skipSpecialTokens: false)
        #expect(text.hasPrefix("<|im_end|>\n<|im_start|>user\n<tool_response>\nFound A1\n</tool_response>"))
        #expect(text.hasSuffix("<|im_start|>assistant\n<think>\n"))
        #expect(!text.contains("Need lookup."))
        #expect(text.components(separatedBy: "<tool_response>").count == results + 1)
    }

    @Test func textContinuationIsTheTailOfTheFullRender() async throws {
        let tok = try await tokenizer()
        let messages: [Message] = [
            .init(role: .user, content: "A"),
            .init(role: .assistant, content: "B", reasoningContent: "Check A."),
            .init(role: .user, content: "C"),
        ]
        for effort in [QwenReasoningEffort.medium, .off] {
            let tail = try tok.encodeSourceTemplateContinuation(
                messages: messages, cachedTurnIndex: 1, tools: [], reasoningEffort: effort)
            #expect(tok.decode(tail, skipSpecialTokens: false)
                    == "<|im_end|>\n<|im_start|>user\nC<|im_end|>\n<|im_start|>assistant\n"
                        + (effort == .off ? "<think>\n\n</think>\n\n" : "<think>\n"))
        }
    }

    @Test func continuationRejectsATurnThatIsNotAnAssistantTurn() async throws {
        let tok = try await tokenizer()
        #expect(throws: MFTokenizerError.self) {
            try tok.encodeSourceTemplateContinuation(
                messages: toolLoop(results: 1), cachedTurnIndex: 1,
                tools: lookup, reasoningEffort: .medium)
        }
    }

    // MARK: - Visible content after the thought block

    /// `</think>\n\n` is the template's framing, not part of the answer: a
    /// client that echoes it back would double the separator on re-render,
    /// and a tool-call turn would carry a non-empty `"\n\n"` content.
    @Test func framingNewlinesAfterTheThoughtAreNotContent() async throws {
        let tok = try await tokenizer()
        let decoder = StructuredAssistantDecoder(tokenizer: tok, allowedTools: [],
                                                  startsInThought: true)
        var reasoning = ""
        decoder.onReasoning = { reasoning += $0 }
        #expect(try decoder.consume(tokenID: 12, delta: "Check") == [])
        #expect(try decoder.consume(tokenID: #require(tok.thinkEndID), delta: "</think>") == [])
        #expect(try decoder.consume(tokenID: 13, delta: "\n\n") == [])
        #expect(try decoder.consume(tokenID: 14, delta: "\nNine") == [.content("Nine")])
        #expect(try decoder.consume(tokenID: 15, delta: "\n\nleft.") == [.content("\n\nleft.")])
        #expect(reasoning == "Check")
    }

    @Test func otherChatMLFamiliesStillRejectEffort() async throws {
        for family in [ModelFamily.qwen38, .qwen38flashnext] {
            let tok = try await tokenizer(family: family)
            #expect(throws: MFTokenizerError.self) {
                try tok.encodeChat(messages: single, reasoningEffort: .low)
            }
        }
    }
}
