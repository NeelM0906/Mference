import Foundation
import Testing
@testable import Mference

/// The `minicpm` dialect against a synthetic tokenizer fixture (a byte-level
/// BPE vocab plus the family's real special ids) and the committed HF renders
/// (`renders.json`, produced by `transformers` 5.6.2 from the real
/// `tokenizer.json` + `chat_template.jinja` at revision `cd199ce3`).
@Suite("MiniCPM5 template")
struct MiniCPM5TemplateTests {
    let tok: MFTokenizer

    init() async throws {
        self.tok = try await MFTokenizer.load(from: Self.fixtureFolder())
    }

    static func fixtureFolder() throws -> URL {
        try #require(Bundle.module.url(
            forResource: "MiniCPM5Tokenizer",
            withExtension: nil,
            subdirectory: "Fixtures"))
    }

    private typealias Message = MFTokenizer.Message

    // MARK: - Fixture decoding

    struct Render {
        let name: String
        let messages: [MFTokenizer.Message]
        let tools: [MFTokenizer.FunctionDefinition]
        let thinking: MFTokenizer.MiniCPMThinking
        let render: String
    }

    static func renders() throws -> [Render] {
        let url = try fixtureFolder().appendingPathComponent("renders.json")
        guard let root = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
                as? [String: Any],
              let items = root["renders"] as? [[String: Any]] else {
            throw CocoaError(.fileReadCorruptFile)
        }
        return try items.map { item in
            let messages = try (item["messages"] as! [[String: Any]]).map { m -> Message in
                let role = MFTokenizer.Role(rawValue: m["role"] as! String)!
                let calls = try ((m["tool_calls"] as? [[String: Any]]) ?? []).map { call in
                    let function = call["function"] as! [String: Any]
                    return MFTokenizer.HistoricalToolCall(
                        id: call["id"] as! String,
                        name: function["name"] as! String,
                        arguments: try jsonValue(function["arguments"]!))
                }
                return Message(role: role,
                               content: m["content"] as? String,
                               toolCalls: calls,
                               toolCallID: m["tool_call_id"] as? String)
            }
            let tools = try ((item["tools"] as? [[String: Any]]) ?? []).map { tool in
                let function = tool["function"] as! [String: Any]
                return MFTokenizer.FunctionDefinition(
                    name: function["name"] as! String,
                    description: function["description"] as! String,
                    parameters: try jsonValue(function["parameters"]!))
            }
            let thinking: MFTokenizer.MiniCPMThinking
            switch item["enable_thinking"] {
            case let flag as Bool: thinking = flag ? .enabled : .disabled
            default: thinking = .unspecified
            }
            return Render(name: item["name"] as! String, messages: messages, tools: tools,
                          thinking: thinking, render: item["render"] as! String)
        }
    }

    private static func jsonValue(_ any: Any) throws -> JSONValue {
        let data = try JSONSerialization.data(withJSONObject: any, options: [.fragmentsAllowed])
        return try JSONDecoder().decode(JSONValue.self, from: data)
    }

    // MARK: - Dialect contract

    @Test("Fixture resolves to the minicpm dialect, with or without the family hint")
    func dialectDetection() async throws {
        #expect(tok.dialect == .minicpm)
        let hinted = try await MFTokenizer.load(from: Self.fixtureFolder(), family: .minicpm5)
        #expect(hinted.dialect == .minicpm)
        // A ChatML fixture that also carries `<|im_end|>` stays ChatML: the
        // discriminator is the `<function` special token.
        let chatml = try await MFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())
        #expect(chatml.dialect == .chatml)
    }

    @Test("Special-token ids match the openbmb/MiniCPM5-2B tokenizer")
    func specialTokenIDs() {
        #expect(tok.bosID == 0)
        #expect(tok.padID == 1)
        #expect(tok.endOfTurnID == 130_073)
        #expect(tok.eosID == 130_073)
        #expect(tok.toolCallStartID == 18)      // <function
        #expect(tok.toolCallEndID == 19)        // </function>
        #expect(tok.toolResponseID == 10)
        #expect(tok.toolResponseEndID == 11)
        #expect(tok.thinkStartID == 8)
        #expect(tok.thinkEndID == 9)
        #expect(tok.vocabSize == 130_560)
    }

    @Test("Both EOS ids stop generation: </s> and <|im_end|>")
    func stopTokens() {
        #expect(tok.stopTokenIDs == [1, 130_073])
    }

    @Test("Thinking is the family default, so the decoder starts in thought")
    func thinkingDefault() {
        #expect(tok.generationPromptStartsInThinking)
    }

    @Test("Raw prompts get <s> once; the template supplies its own")
    func bosHandling() throws {
        #expect(tok.encode("hi", addBOS: true).first == 0)
        #expect(tok.encode("hi", addBOS: false).first != 0)
        let rendered = try tok.applyChatTemplate([Message(role: .user, content: "Hi")])
        #expect(rendered.hasPrefix("<s><|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n"))
        let ids = tok.encode(rendered, addBOS: false)
        #expect(ids.first == 0)
        #expect(ids.filter { $0 == 0 }.count == 1)
    }

    @Test("Added tokens flagged non-special still encode as single ids")
    func thinkTokensAreSingleIDs() {
        #expect(tok.encode("<think>\n", addBOS: false).first == 8)
        #expect(tok.encode("</think>\n\n", addBOS: false).first == 9)
        #expect(tok.encode("<function name=\"x\">", addBOS: false).first == 18)
        #expect(tok.encode("<param name=\"x\">", addBOS: false).first == 20)
        #expect(tok.encode("</function>", addBOS: false) == [19])
    }

    // MARK: - Byte-exact renders

    @Test("Every committed HF render is reproduced byte for byte",
          arguments: try MiniCPM5TemplateTests.renders().map(\.name))
    func rendersMatchHF(name: String) throws {
        let fixture = try #require(try Self.renders().first { $0.name == name })
        let actual = try tok.miniCPMRender(messages: fixture.messages,
                                           tools: fixture.tools,
                                           thinking: fixture.thinking)
        if actual != fixture.render {
            let a = Array(actual.utf8), e = Array(fixture.render.utf8)
            let firstDiff = zip(a, e).enumerated().first { $0.element.0 != $0.element.1 }?.offset
                ?? min(a.count, e.count)
            let context = String(decoding: e[max(0, firstDiff - 40)..<min(e.count, firstDiff + 60)],
                                 as: UTF8.self)
            Issue.record("\(name) differs at byte \(firstDiff) near: \(context.debugDescription)")
        }
        #expect(actual == fixture.render)
    }

    @Test("The public text-only render is the thinking-enabled render")
    func publicRenderIsThinkingEnabled() throws {
        let fixture = try #require(try Self.renders().first { $0.name == "user_only_thinking_true" })
        #expect(try tok.applyChatTemplate(fixture.messages) == fixture.render)
    }

    @Test("encodeToolChat encodes the byte-exact tool render")
    func encodeToolChatMatchesRender() throws {
        let fixture = try #require(try Self.renders().first { $0.name == "tool_call_and_response" })
        let ids = try tok.encodeToolChat(messages: fixture.messages, tools: fixture.tools)
        #expect(ids == tok.encode(fixture.render, addBOS: false))
        #expect(ids.first == 0)
    }

    @Test("Text continuation closes the cached turn and reopens with the thinking prompt")
    func textContinuation() {
        let ids = tok.encodeTextContinuation(userContent: "Again")
        #expect(ids.first == tok.endOfTurnID)
        #expect(Array(ids.dropFirst()) == tok.encode(
            "\n<|im_start|>user\nAgain<|im_end|>\n<|im_start|>assistant\n<think>\n",
            addBOS: false))
    }

    @Test("Tool-result KV continuation is declined so the server falls back to prefix matching")
    func toolResultContinuationIsUnsupported() {
        #expect(throws: MFTokenizerError.self) {
            _ = try tok.encodeToolResultContinuation(
                cachedMessages: [Message(role: .user, content: "x")],
                assistant: Message(role: .assistant, content: nil),
                incomingMessages: [], tools: [])
        }
    }

    // MARK: - The real tokenizer (env-gated)

    /// `MFERENCE_MINICPM5_TOKENIZER_DIR` pointing at an install's `tokenizer/`
    /// folder replays the fixture's real-tokenizer probes: the ids
    /// swift-transformers produces for the special sequences and for a full
    /// render must equal what `transformers` 5.6.2 produced.
    @Test("Real tokenizer: encodings match transformers on the fixture probes")
    func realTokenizerProbes() async throws {
        guard let dir = ProcessInfo.processInfo.environment["MFERENCE_MINICPM5_TOKENIZER_DIR"],
              !dir.isEmpty else { return }
        let real = try await MFTokenizer.load(from: URL(fileURLWithPath: dir), family: .minicpm5)
        #expect(real.dialect == .minicpm)
        let url = try Self.fixtureFolder().appendingPathComponent("renders.json")
        let root = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
        let probes = root["encoding_probes_real_tokenizer"] as! [String: [Int]]
        for (text, expected) in probes {
            #expect(real.encode(text, addBOS: false).map(Int.init) == expected,
                    Comment(rawValue: text.debugDescription))
        }
        for item in root["renders"] as! [[String: Any]] {
            let render = item["render"] as! String
            let expected = item["token_ids_real_tokenizer"] as! [Int]
            #expect(real.encode(render, addBOS: false).map(Int.init) == expected,
                    Comment(rawValue: item["name"] as! String))
        }
    }
}
