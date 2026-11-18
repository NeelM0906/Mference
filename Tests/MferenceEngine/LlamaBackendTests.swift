import CryptoKit
import Foundation
import Mference
import Testing
@testable import MferenceEngine

/// Model-free tests of the llama.cpp backend: template rendering against
/// llama.cpp's own renders, the thinking switch, UTF-8 streaming, stop-string
/// assembly, and load errors. Real-model behaviour (token identity with the
/// upstream tools) is checked by the engine program's G6 runs.
@Suite("Llama backend")
struct LlamaBackendTests {
    // MARK: Chat template

    @Test(arguments: LlamaChatTemplateFixtures.cases.map(\.name))
    func rendersLikeLlamaCpp(_ name: String) throws {
        let fixture = try #require(LlamaChatTemplateFixtures.cases.first { $0.name == name })
        let template = try LlamaChatTemplate(source: LlamaChatTemplateFixtures.quickTemplate,
                                             bosToken: nil, eosToken: "<|im_end|>")
        // The fixtures' upstream renders pass enable_thinking explicitly;
        // the engine leaves it unset unless the request chose an effort, which
        // this template renders the same as `false`.
        let effort: QwenReasoningEffort? = fixture.enableThinking ? .xhigh : nil
        let rendered = try template.render(
            messages: try Self.messages(fixture.messagesJSON),
            tools: try fixture.toolsJSON.map(Self.tools) ?? [],
            keywordArguments: LlamaChatTemplate.keywordArguments(
                reasoningEffort: effort, preserveThinking: false,
                readsEnableThinking: template.readsEnableThinking))
        #expect(rendered == fixture.expected, "rendered:\n\(rendered)")
    }

    @Test func fixtureTemplateIsTheQuickGGUFTemplate() {
        let digest = SHA256.hash(data: Data(LlamaChatTemplateFixtures.quickTemplate.utf8))
            .map { String(format: "%02x", $0) }.joined()
        #expect(digest == LlamaChatTemplateFixtures.quickTemplateSHA256)
    }

    @Test func thinkingSwitchFollowsReasoningEffort() {
        func kwargs(_ effort: QwenReasoningEffort?, reads: Bool = true) -> Bool? {
            LlamaChatTemplate.keywordArguments(reasoningEffort: effort, preserveThinking: false,
                                               readsEnableThinking: reads)["enable_thinking"] as? Bool
        }
        #expect(kwargs(nil) == nil)
        #expect(kwargs(.off) == false)
        #expect(kwargs(.low) == true)
        #expect(kwargs(.medium) == true)
        #expect(kwargs(.xhigh) == true)
        #expect(kwargs(.xhigh, reads: false) == nil)
    }

    @Test func defaultRenderOfQuickClosesThinking() throws {
        let template = try LlamaChatTemplate(source: LlamaChatTemplateFixtures.quickTemplate,
                                             bosToken: nil, eosToken: "<|im_end|>")
        let messages = [MFTokenizer.Message(role: .user, content: "hi")]
        let plain = try template.render(messages: messages, tools: [])
        #expect(plain.hasSuffix("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
        let thinking = try template.render(
            messages: messages, tools: [],
            keywordArguments: LlamaChatTemplate.keywordArguments(
                reasoningEffort: .medium, preserveThinking: false, readsEnableThinking: true))
        #expect(thinking.hasSuffix("<|im_start|>assistant\n<think>\n"))
    }

    /// Like llama.cpp's renderer, a BOS/EOS the template prints is dropped when
    /// tokenization adds that token itself, so it is never doubled.
    @Test func implicitSpecialTokensAreNotDoubled() throws {
        let template = try LlamaChatTemplate(
            source: "{{ bos_token }}{% for m in messages %}{{ m['content'] }}{% endfor %}{{ eos_token }}",
            bosToken: "<s>", eosToken: "</s>")
        let rendered = try template.render(messages: [MFTokenizer.Message(role: .user, content: "hi")],
                                           tools: [])
        #expect(rendered == "<s>hi</s>")
        #expect(template.strippingImplicitSpecialTokens(rendered, addsBOS: true, addsEOS: false) == "hi</s>")
        #expect(template.strippingImplicitSpecialTokens(rendered, addsBOS: false, addsEOS: true) == "<s>hi")
        #expect(template.strippingImplicitSpecialTokens(rendered, addsBOS: false, addsEOS: false) == rendered)
        let quick = try LlamaChatTemplate(source: LlamaChatTemplateFixtures.quickTemplate,
                                          bosToken: nil, eosToken: "<|im_end|>")
        let prompt = try quick.render(messages: [MFTokenizer.Message(role: .user, content: "hi")], tools: [])
        #expect(quick.strippingImplicitSpecialTokens(prompt, addsBOS: false, addsEOS: false) == prompt)
    }

    @Test func malformedTemplateIsALoadError() {
        #expect(throws: LlamaBackendError.self) {
            _ = try LlamaChatTemplate(source: "{% if %}", bosToken: nil, eosToken: nil)
        }
    }

    // MARK: Streaming text

    @Test func utf8StreamHoldsSplitCharacters() {
        var stream = LlamaUTF8Stream()
        let bytes = Array("a→é🙂".utf8)
        var text = ""
        for byte in bytes { text += stream.push([byte]) }
        text += stream.finish()
        #expect(text == "a→é🙂")

        var partial = LlamaUTF8Stream()
        #expect(partial.push([0xF0, 0x9F]) == "")
        #expect(partial.push([0x99, 0x82, 0x41]) == "🙂A")
    }

    @Test func utf8StreamReplacesInvalidBytes() {
        var stream = LlamaUTF8Stream()
        #expect(stream.push([0x41, 0xFF, 0x42]) == "A\u{FFFD}B")
        #expect(stream.push([0xE2, 0x82]) == "")
        #expect(stream.finish() == "\u{FFFD}")
    }

    @Test func assemblerAppliesStopStringsToContentOnly() {
        let events = Collector<EngineEvent>()
        var assembler = LlamaEventAssembler(stops: ["END"]) { events.append($0) }
        assembler.take([.content("Hello E")])
        #expect(!assembler.isStopped)
        assembler.take([.content("ND world")])
        #expect(assembler.isStopped)
        assembler.finish()
        #expect(assembler.content == "Hello ")
        #expect(events.all == [.content("Hello ")])
    }

    @Test func assemblerReleasesHeldPrefixAtFinish() {
        let events = Collector<EngineEvent>()
        var assembler = LlamaEventAssembler(stops: ["STOP"]) { events.append($0) }
        assembler.take([.content("ends with ST")])
        assembler.finish()
        #expect(assembler.content == "ends with ST")
        #expect(!assembler.isStopped)
        #expect(events.all.map { if case .content(let text) = $0 { text } else { "" } }.joined()
                == "ends with ST")
    }

    @Test func assemblerForwardsToolCalls() {
        let events = Collector<EngineEvent>()
        var assembler = LlamaEventAssembler(stops: []) { events.append($0) }
        let call = ParsedToolCall(id: "call_1", name: "get_weather",
                                  arguments: .object(["city": .string("Paris")]),
                                  argumentsJSON: #"{"city":"Paris"}"#)
        assembler.take([.toolCall(call)])
        #expect(assembler.toolCalls == [call])
        #expect(events.all == [.toolCall(call)])
    }

    // MARK: Loading

    @Test func refusesANativeDescriptor() async {
        let descriptor = EngineModelDescriptor(
            id: "native", displayName: "native", engine: .mference,
            format: .gturbo(family: .qwen36, checkpointID: nil),
            location: URL(fileURLWithPath: "/nonexistent.gturbo"),
            sizeBytes: 0, contextLength: 4096, product: nil)
        await #expect(throws: LlamaBackendError.notGGUF("native")) {
            _ = try await LlamaBackend().load(descriptor, options: EngineLoadOptions())
        }
    }

    @Test func missingGGUFFailsToLoad() async {
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("llama-backend-missing-\(UUID().uuidString).gguf")
        let descriptor = EngineModelDescriptor(
            id: "missing", displayName: "missing", engine: .llamaCpp,
            format: .gguf(EngineGGUFMetadata(architecture: "qwen35", contextLength: 4096,
                                             tensorCount: 0, nextNTensorCount: 0)),
            location: path, sizeBytes: 0, contextLength: 4096, product: nil)
        await #expect(throws: LlamaBackendError.modelLoadFailed(path.path)) {
            _ = try await LlamaBackend().load(descriptor, options: EngineLoadOptions())
        }
    }

    // MARK: Fixture decoding (OpenAI JSON → runtime chat types)

    static func messages(_ json: String) throws -> [MFTokenizer.Message] {
        let items = try JSONDecoder().decode([OpenAIMessage].self, from: Data(json.utf8))
        return try items.map { item in
            let role = try #require(MFTokenizer.Role(rawValue: item.role))
            let calls = try (item.tool_calls ?? []).map { call in
                MFTokenizer.HistoricalToolCall(
                    id: call.id, name: call.function.name,
                    arguments: try JSONDecoder().decode(JSONValue.self,
                                                        from: Data(call.function.arguments.utf8)))
            }
            return MFTokenizer.Message(role: role, content: item.content, toolCalls: calls,
                                       toolCallID: item.tool_call_id,
                                       reasoningContent: item.reasoning_content)
        }
    }

    static func tools(_ json: String) throws -> [MFTokenizer.FunctionDefinition] {
        try JSONDecoder().decode([OpenAITool].self, from: Data(json.utf8)).map {
            MFTokenizer.FunctionDefinition(name: $0.function.name,
                                           description: $0.function.description,
                                           parameters: $0.function.parameters)
        }
    }

    private struct OpenAIMessage: Decodable {
        struct Call: Decodable {
            struct Function: Decodable { let name: String; let arguments: String }
            let id: String
            let function: Function
        }
        let role: String
        let content: String?
        let tool_calls: [Call]?
        let tool_call_id: String?
        let reasoning_content: String?
    }

    private struct OpenAITool: Decodable {
        struct Function: Decodable {
            let name: String
            let description: String
            let parameters: JSONValue
        }
        let function: Function
    }
}
