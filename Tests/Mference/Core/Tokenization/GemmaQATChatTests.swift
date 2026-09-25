import Foundation
import Testing
@testable import Mference

@Suite struct GemmaQATChatTests {
    static let templateSHA = "94899c0f917d93f6fe81c95744d1e8ddab2d21d39228d2e4aec1fb2a25bff413"

    static func fixtureFolder() -> URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/GemmaQATTokenizer")
    }

    struct Oracle: Decodable {
        let template_sha256: String
        let tokenizer_sha256: String
        let jinja: String
        let tokenizers: String
        let cases: [Case]
        struct Case: Decodable {
            let name: String
            let thinking: Bool
            let generate: Bool
            let render: String?
            let ids: [Int32]?
            let error: SourceError?
            let messages: [Message]
            let tools: [Tool]
        }
        struct SourceError: Decodable { let type: String; let message: String }
        struct Message: Decodable {
            let role: String
            let content: String?
            let reasoning_content: String?
            let tool_call_id: String?
            let tool_calls: [Call]?
        }
        struct Call: Decodable { let id: String; let function: Function }
        struct Function: Decodable {
            let name: String
            let description: String?
            let parameters: JSONValue?
            let arguments: JSONValue?
        }
        struct Tool: Decodable { let function: Function }
    }

    static func oracle(at url: URL? = nil) throws -> Oracle {
        try JSONDecoder().decode(Oracle.self,
            from: Data(contentsOf: url ?? fixtureFolder().appendingPathComponent("oracle.json")))
    }

    static func nativeMessages(_ item: Oracle.Case) throws -> [MFTokenizer.Message] {
        try item.messages.map { message in
            let role = try #require(MFTokenizer.Role(rawValue: message.role))
            return MFTokenizer.Message(role: role, content: message.content,
                toolCalls: (message.tool_calls ?? []).map {
                    .init(id: $0.id, name: $0.function.name, arguments: $0.function.arguments ?? .object([:]))
                }, toolCallID: message.tool_call_id, reasoningContent: message.reasoning_content)
        }
    }

    static func nativeTools(_ item: Oracle.Case) -> [MFTokenizer.FunctionDefinition] {
        item.tools.map { .init(name: $0.function.name, description: $0.function.description ?? "",
                              parameters: $0.function.parameters ?? .object([:])) }
    }

    static func compare(_ tokenizer: MFTokenizer, oracle: Oracle) throws {
        #expect(oracle.template_sha256 == templateSHA)
        #expect(oracle.jinja == "3.1.6" && oracle.tokenizers == "0.23.2")
        #expect(oracle.cases.count == 72)
        for item in oracle.cases {
            let messages = try nativeMessages(item)
            let tools = nativeTools(item)
            let effort: QwenReasoningEffort = item.thinking ? .medium : .off
            if item.error != nil {
                #expect(throws: (any Error).self) {
                    _ = try tokenizer.encodeToolChat(messages: messages, tools: tools,
                        reasoningEffort: effort, addGenerationPrompt: item.generate)
                }
                continue
            }
            let ids = try tokenizer.encodeToolChat(messages: messages, tools: tools,
                reasoningEffort: effort, addGenerationPrompt: item.generate)
            #expect(ids == item.ids, "\(item.name), thinking=\(item.thinking), generate=\(item.generate)")
            #expect(tokenizer.decode(ids, skipSpecialTokens: false) == item.render,
                    "\(item.name), thinking=\(item.thinking), generate=\(item.generate)")
            if item.generate {
                #expect(tokenizer.startsInThinking(reasoningEffort: effort, promptIDs: ids)
                    == (item.render?.hasSuffix("<|channel>thought\n") == true), "\(item.name)")
                #expect(try tokenizer.encodeChat(messages: messages, tools: tools, reasoningEffort: effort) == item.ids)
            }
        }
    }

    @Test func sourceTemplateRendersAndTokensMatchFrozenOracle() async throws {
        let oracle = try Self.oracle()
        #expect(Sha256Verifier.hashData(try Data(contentsOf: Self.fixtureFolder().appendingPathComponent("tokenizer.json")))
            == oracle.tokenizer_sha256)
        let tokenizer = try await MFTokenizer.load(from: Self.fixtureFolder(), family: .gemma4)
            .forCheckpoint(CheckpointIdentity.gemma4QAT)
        try Self.compare(tokenizer, oracle: oracle)
    }

    @Test func preserveThinkingTrueIsAnExplicitUnsupportedQATControl() async throws {
        let tokenizer = try await MFTokenizer.load(from: Self.fixtureFolder(), family: .gemma4)
            .forCheckpoint(CheckpointIdentity.gemma4QAT)
        let messages: [MFTokenizer.Message] = [.init(role: .user, content: "Hi")]
        #expect(throws: MFTokenizerError.self) {
            _ = try tokenizer.encodeChat(messages: messages, preserveThinking: true)
        }
        #expect(try tokenizer.encodeChat(messages: messages, preserveThinking: false)
            == tokenizer.encodeChat(messages: messages))
    }

    @Test func checkpointCopiesKeepOriginalTemplateDefaultsAndThinkingAliases() async throws {
        let base = try await MFTokenizer.load(from: Self.fixtureFolder(), family: .gemma4)
        let qat = try base.forCheckpoint(CheckpointIdentity.gemma4QAT)
        let original = try qat.forCheckpoint("gemma-4-26b-a4b-it")
        let canonical = try await MFTokenizer.load(from: GemmaThinkingTests.fixtureFolder(), family: .gemma4)
        let messages: [MFTokenizer.Message] = [.init(role: .user, content: "A"),
            .init(role: .assistant, content: "B", reasoningContent: "Retained only by the original template.")]
        #expect(try original.encodeChat(messages: messages) == canonical.encodeChat(messages: messages))
        #expect(try base.encodeChat(messages: messages) == canonical.encodeChat(messages: messages))
        #expect(try qat.encodeChat(messages: messages) != original.encodeChat(messages: messages))
        #expect(qat.generationDefaults.temperature == 1 && qat.generationDefaults.topK == 64 && qat.generationDefaults.minP == 0)
        #expect(original.generationDefaults.temperature == 0.8 && original.generationDefaults.topK == 40 && original.generationDefaults.minP == 0.05)
        #expect(base.generationDefaults.temperature == 0.8 && base.generationDefaults.minP == 0.05)
        let enabled = try qat.encodeChat(messages: messages, reasoningEffort: .medium)
        #expect(try qat.encodeChat(messages: messages, reasoningEffort: .low) == enabled)
        #expect(try qat.encodeChat(messages: messages, reasoningEffort: .xhigh) == enabled)
        #expect(try qat.encodeChat(messages: messages) == qat.encodeChat(messages: messages, reasoningEffort: .off))
    }

    @Test(arguments: [false, true])
    func missingOrAlteredQATTemplateCannotSelectCanonicalFallback(missing: Bool) async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.copyItem(at: Self.fixtureFolder(), to: directory)
        defer { try? FileManager.default.removeItem(at: directory) }
        let template = directory.appendingPathComponent("chat_template.jinja")
        if missing { try FileManager.default.removeItem(at: template) }
        else { try Data("{{ bos_token }}different".utf8).write(to: template) }
        await #expect(throws: (any Error).self) {
            _ = try await MFTokenizer.load(from: directory, family: .gemma4)
                .forCheckpoint(CheckpointIdentity.gemma4QAT)
        }
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["MFERENCE_GEMMA_QAT_TEMPLATE_ORACLE"] != nil))
    func installedSourceTemplateAndNativeTokenIDsMatchFrozenOracle() async throws {
        let environment = ProcessInfo.processInfo.environment
        let directory = URL(fileURLWithPath: try #require(environment["MFERENCE_GEMMA_QAT_GTURBO"]))
        let oracle = try Self.oracle(at: URL(fileURLWithPath: try #require(environment["MFERENCE_GEMMA_QAT_TEMPLATE_ORACLE"])))
        #expect(Sha256Verifier.hashData(try Data(contentsOf: directory.appendingPathComponent("tokenizer/tokenizer.json")))
            == oracle.tokenizer_sha256)
        // This loads validated local assets without creating a model runner.
        let tokenizer = try await MFTokenizer.load(forModelDirectory: directory)
        try Self.compare(tokenizer, oracle: oracle)
    }

    @Test func sourceToolArgumentsRoundTripThroughTheStrictDecoder() async throws {
        let tokenizer = try await MFTokenizer.load(from: Self.fixtureFolder(), family: .gemma4)
            .forCheckpoint(CheckpointIdentity.gemma4QAT)
        let item = try #require(Self.oracle().cases.first {
            $0.name == "nested_null_arguments" && !$0.thinking && !$0.generate
        })
        let assistant = try #require(item.messages.first { !($0.tool_calls ?? []).isEmpty })
        let expected = try #require(assistant.tool_calls?.first)
        let render = try #require(item.render)
        let payload = try #require(render.components(separatedBy: "<|tool_call>").last?
            .components(separatedBy: "<tool_call|>").first)
        let decoder = StructuredAssistantDecoder(tokenizer: tokenizer, allowedTools: [expected.function.name],
            idGenerator: { "call_test" })
        var reasoning = ""
        decoder.onReasoning = { reasoning += $0 }
        let text = "<|channel>thought\nCheck.<channel|><|tool_call>" + payload + "<tool_call|>"
        var events: [StructuredAssistantEvent] = []
        for id in tokenizer.encode(text, addBOS: false) {
            events += try decoder.consume(tokenID: id, delta: tokenizer.decode([id], skipSpecialTokens: false))
        }
        events += try decoder.finish()
        #expect(reasoning == "Check.")
        #expect(events.count == 1)
        guard case .toolCall(let call) = try #require(events.first) else {
            Issue.record("source tool call was not decoded")
            return
        }
        #expect(call.arguments == expected.function.arguments)
        #expect(try JSONDecoder().decode(JSONValue.self, from: Data(call.argumentsJSON.utf8)) == expected.function.arguments)
        #expect(throws: ToolCallParserError.self) {
            _ = try GemmaToolCallParser().parse(payload, allowedTools: [expected.function.name], id: "original")
        }
    }

    @Test(arguments: [false, true])
    func toolResultContinuationStartsVisibleAndRejectsBrokenCalls(thinking: Bool) async throws {
        let tokenizer = try await MFTokenizer.load(from: Self.fixtureFolder(), family: .gemma4)
            .forCheckpoint(CheckpointIdentity.gemma4QAT)
        let item = try #require(Self.oracle().cases.first {
            $0.name == "tool_result" && $0.thinking == thinking && $0.generate
        })
        let ids = try tokenizer.encodeChat(messages: Self.nativeMessages(item), tools: Self.nativeTools(item),
            reasoningEffort: thinking ? .medium : .off)
        let decoder = StructuredAssistantDecoder(tokenizer: tokenizer, allowedTools: ["lookup"],
            startsInThought: tokenizer.startsInThinking(reasoningEffort: thinking ? .medium : .off, promptIDs: ids))
        #expect(ids.last == tokenizer.toolResponseEndID)
        #expect(try decoder.consume(tokenID: -1, delta: "Found.") == [.content("Found.")])
        for payload in ["call:lookup{x:NoneX}", "call:lookup{x:[None}", "call:lookup{"] {
            let malformed = StructuredAssistantDecoder(tokenizer: tokenizer, allowedTools: ["lookup"])
            #expect(throws: ToolCallParserError.self) {
                let text = "<|tool_call>" + payload + (payload.hasSuffix("{") ? "" : "<tool_call|>")
                for id in tokenizer.encode(text, addBOS: false) {
                    _ = try malformed.consume(tokenID: id, delta: tokenizer.decode([id], skipSpecialTokens: false))
                }
                _ = try malformed.finish()
            }
            #expect(!malformed.hasToolCalls)
        }
    }
}
