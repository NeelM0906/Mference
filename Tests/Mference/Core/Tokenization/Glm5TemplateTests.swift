import Foundation
import Testing
@testable import Mference

/// The `glm5` dialect against a synthetic tokenizer fixture (a byte-level BPE
/// vocab plus GLM-5.3-Flash's real added tokens at their real ids) and the
/// committed HF renders (`renders.json`, produced by `transformers` 5.17 from
/// the real `tokenizer.json` + `chat_template.jinja` at revision `d43ea8b4`;
/// `Scripts/parity/glm5_make_template_fixtures.py`).
@Suite("GLM-5 template")
struct Glm5TemplateTests {
    let tok: MFTokenizer

    init() async throws {
        self.tok = try await MFTokenizer.load(from: Self.fixtureFolder())
    }

    static func fixtureFolder() throws -> URL {
        try #require(Bundle.module.url(
            forResource: "Glm5Tokenizer",
            withExtension: nil,
            subdirectory: "Fixtures"))
    }

    private typealias Message = MFTokenizer.Message

    // MARK: - Fixture decoding

    struct Render {
        let name: String
        let messages: [MFTokenizer.Message]
        let tools: [MFTokenizer.FunctionDefinition]
        let effort: MFTokenizer.Glm5ReasoningEffort
        let clearThinking: Bool
        let render: String
    }

    static func fixtureRoot() throws -> [String: Any] {
        let url = try fixtureFolder().appendingPathComponent("renders.json")
        guard let root = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
                as? [String: Any] else {
            throw CocoaError(.fileReadCorruptFile)
        }
        return root
    }

    static func renders() throws -> [Render] {
        guard let items = try fixtureRoot()["renders"] as? [[String: Any]] else {
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
            let effort: MFTokenizer.Glm5ReasoningEffort
            switch item["reasoning_effort"] as? String {
            case "low": effort = .low
            case "high": effort = .high
            default: effort = .max
            }
            return Render(name: item["name"] as! String, messages: messages, tools: tools,
                          effort: effort, clearThinking: item["clear_thinking"] as? Bool ?? false,
                          render: item["render"] as! String)
        }
    }

    private static func jsonValue(_ any: Any) throws -> JSONValue {
        let data = try JSONSerialization.data(withJSONObject: any, options: [.fragmentsAllowed])
        return try JSONDecoder().decode(JSONValue.self, from: data)
    }

    // MARK: - Dialect contract

    @Test("Fixture resolves to the glm5 dialect, with or without the family hint")
    func dialectDetection() async throws {
        #expect(tok.dialect == .glm5)
        let hinted = try await MFTokenizer.load(from: Self.fixtureFolder(), family: .glm53Flash)
        #expect(hinted.dialect == .glm5)
    }

    @Test("Special-token ids match the GLM-5.3-Flash tokenizer")
    func specialTokenIDs() throws {
        let real = try #require(try Self.fixtureRoot()["special_token_ids"] as? [String: Int])
        #expect(tok.bosID == Int32(real["[gMASK]"]!))
        #expect(tok.eosID == 154_820)
        #expect(tok.padID == 154_820)
        #expect(tok.endOfTurnID == Int32(real["<|user|>"]!))
        #expect(tok.endOfTurnID == 154_827)
        #expect(tok.toolCallStartID == 154_843)
        #expect(tok.toolCallEndID == 154_844)
        #expect(tok.toolResponseID == 154_845)
        #expect(tok.toolResponseEndID == 154_846)
        #expect(tok.thinkStartID == 154_841)
        #expect(tok.thinkEndID == 154_842)
        #expect(tok.vocabSize == 154_880)
    }

    @Test("The three generation_config EOS ids stop generation")
    func stopTokens() {
        #expect(tok.stopTokenIDs == [154_820, 154_827, 154_829])
    }

    @Test("The generation prompt opens the think block, so the decoder starts in thought")
    func thinkingDefault() {
        #expect(tok.generationPromptStartsInThinking)
    }

    @Test("No BOS is prepended to raw prompts; the render carries [gMASK]<sop> itself")
    func bosHandling() throws {
        #expect(tok.encode("hi", addBOS: true) == tok.encode("hi", addBOS: false))
        let rendered = try tok.applyChatTemplate([Message(role: .user, content: "Hi")])
        #expect(rendered == "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Hi<|assistant|><think>")
        let ids = tok.encode(rendered, addBOS: false)
        #expect(Array(ids.prefix(2)) == [154_822, 154_824])
        #expect(ids.suffix(2) == [154_828, 154_841])
    }

    @Test("Added tokens flagged non-special still encode as single ids, like the real tokenizer")
    func addedTokensAreSingleIDs() throws {
        let probes = try #require(try Self.fixtureRoot()["encoding_probes_real_tokenizer"] as? [String: [Int]])
        // Probes made only of added tokens match the real tokenizer exactly;
        // the synthetic vocab spells plain text differently, so only those.
        for probe in ["[gMASK]<sop>", "<|assistant|><think>", "<think></think>", "</think>"] {
            #expect(tok.encode(probe, addBOS: false) == probes[probe]!.map(Int32.init), Comment(rawValue: probe))
        }
        let call = tok.encode("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
                              addBOS: false)
        #expect(call.first == 154_843 && call.last == 154_844)
        #expect(call.contains(154_847) && call.contains(154_848) && call.contains(154_849) && call.contains(154_850))
    }

    // MARK: - Byte-exact renders

    @Test("Every committed HF render is reproduced byte for byte",
          arguments: try Glm5TemplateTests.renders().map(\.name))
    func rendersMatchHF(name: String) throws {
        let fixture = try #require(try Self.renders().first { $0.name == name })
        let actual = try tok.glm5Render(messages: fixture.messages,
                                        tools: fixture.tools,
                                        reasoningEffort: fixture.effort,
                                        clearThinking: fixture.clearThinking)
        if actual != fixture.render {
            let a = Array(actual.utf8), e = Array(fixture.render.utf8)
            let firstDiff = zip(a, e).enumerated().first { $0.element.0 != $0.element.1 }?.offset
                ?? min(a.count, e.count)
            let lo = max(0, firstDiff - 60), hi = min(firstDiff + 60, min(a.count, e.count))
            Issue.record(Comment(rawValue: "\(name): first byte difference at \(firstDiff)\n  ours:   \(String(decoding: a[lo..<min(hi, a.count)], as: UTF8.self).debugDescription)\n  theirs: \(String(decoding: e[lo..<min(hi, e.count)], as: UTF8.self).debugDescription)"))
        }
        #expect(actual == fixture.render)
    }

    @Test("Text-only and tool encoders are the render, encoded without BOS")
    func encodersAreTheRender() throws {
        let fixtures = try Self.renders()
        let text = try #require(fixtures.first { $0.name == "multi_turn_plain_assistant" })
        #expect(try tok.applyChatTemplate(text.messages) == text.render)
        let tools = try #require(fixtures.first { $0.name == "two_calls_two_results" })
        #expect(try tok.encodeToolChat(messages: tools.messages, tools: tools.tools)
                == tok.encode(tools.render, addBOS: false))
    }

    @Test("Historical tool calls need object arguments")
    func historicalCallArgumentsMustBeObjects() {
        let call = MFTokenizer.HistoricalToolCall(id: "c", name: "run_code", arguments: .string("x"))
        #expect(throws: MFTokenizerError.self) {
            try tok.glm5Render(messages: [Message(role: .user, content: "A"),
                                          Message(role: .assistant, content: "", toolCalls: [call])],
                               tools: [])
        }
    }

    @Test("Text continuation supplies <|user|>, the untrimmed content and the generation prompt")
    func textContinuation() {
        let ids = tok.encodeTextContinuation(userContent: " next \n")
        #expect(ids.first == tok.endOfTurnID)
        #expect(Array(ids.dropFirst()) == tok.encode(" next \n<|assistant|><think>", addBOS: false))
        // A fresh render of the same history ends in the same bytes.
        let full = tok.encode(
            "<|assistant|><think>r</think>answer<|user|> next \n<|assistant|><think>", addBOS: false)
        #expect(Array(full.suffix(ids.count)) == ids)
    }
}
