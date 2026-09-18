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
        let url = try #require(Bundle.module.url(forResource: "oracle", withExtension: "json",
                                               subdirectory: "Fixtures/SwiftQwenTemplate"))
        let cases = try JSONDecoder().decode([Oracle].self, from: Data(contentsOf: url))
        for item in cases {
            let effort = try #require(QwenReasoningEffort(rawValue: item.effort))
            let ids = try tok.encodeChat(messages: messages(item.name), reasoningEffort: effort)
            #expect(tok.decode(ids, skipSpecialTokens: false) == item.render,
                    "\(item.name)/\(item.effort)")
            #expect(ids == tok.encode(item.render, addBOS: false))
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
        #expect(throws: MFTokenizerError.self) {
            try base.encodeChat(messages: messages("single"), reasoningEffort: .low)
        }
        #expect(throws: (any Error).self) { try tok.encodeChat(messages: []) }
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
