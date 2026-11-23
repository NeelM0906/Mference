import Foundation
import Testing
@testable import Mference

/// The `inkling2` dialect against a synthetic tokenizer fixture (the MiniCPM5
/// fixture's byte-level BPE vocabulary plus Inkling 2's 60 added tokens at
/// their real ids) and the vendor's `chat_template.jinja` (`b85c3cf`,
/// SHA-256 `53a9035f…16dc`, identical to the one embedded in the shipped
/// GGUF). Expected renders are written out from the template by hand; K0's
/// goldens replace them when they land.
@Suite("Inkling 2 chat format")
struct Inkling2ChatTests {
    let tok: MFTokenizer

    init() async throws {
        self.tok = try await MFTokenizer.load(from: Self.fixtureFolder())
    }

    static func fixtureFolder() throws -> URL {
        try #require(Bundle.module.url(forResource: "Inkling2Tokenizer", withExtension: nil,
                                       subdirectory: "Fixtures"))
    }

    private typealias Message = MFTokenizer.Message
    private typealias Effort = MFTokenizer.Inkling2Effort

    // Template pieces.
    private let sys = "<|message_system|>", usr = "<|message_user|>", mdl = "<|message_model|>"
    private let tool = "<|message_tool|>", txt = "<|content_text|>", thk = "<|content_thinking|>"
    private let end = "<|end_message|>", eos = "<|content_model_end_sampling|>"
    private let invoke = "<|content_invoke_tool_json|>"
    private func effortLine(_ level: String) -> String {
        sys + txt + "Thinking effort level: " + level + end
    }

    private func render(_ messages: [Message], tools: [MFTokenizer.FunctionDefinition] = [],
                        effort: Effort? = nil) throws -> String {
        try tok.renderInkling2Chat(messages: messages, tools: tools, effort: effort)
    }

    private func json(_ text: String) throws -> JSONValue {
        try JSONDecoder().decode(JSONValue.self, from: Data(text.utf8))
    }

    // MARK: Dialect

    @Test func tokenizerResolvesTheInkling2Dialect() throws {
        #expect(tok.dialect == .inkling2)
        #expect(tok.eosID == 200_006 && tok.endOfTurnID == 200_006)
        #expect(tok.padID == 199_999)
        #expect(tok.stopTokenIDs == [200_006])
        #expect(tok.vocabSize == 200_058)
        let t = try #require(tok.inkling2Tokens)
        #expect(t.messageUser == 200_000 && t.messageModel == 200_001)
        #expect(t.messageSystem == 200_002 && t.messageTool == 200_003)
        #expect(t.contentText == 200_004 && t.contentThinking == 200_008)
        #expect(t.endMessage == 200_010 && t.contentInvokeToolJSON == 200_049)
        #expect(tok.acceptsReasoningEffort)
        #expect(!tok.generationPromptStartsInThinking)
    }

    @Test func samplingDefaultsAreTheVendors() throws {
        // generation_config.json: do_sample, temperature 1.0, nothing else.
        let profiled = try tok.forCheckpoint("inkling2-mini")
        let g = profiled.generationDefaults
        #expect(g.temperature == 1.0)
        #expect(g.topK == nil && g.topP == nil)
        #expect(g.minP == 0 && g.repetitionPenalty == 1)
    }

    @Test func samplingDefaultsHoldWithoutTheGenerationConfig() async throws {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("inkling2-tokenizer-\(UUID().uuidString)")
        try FileManager.default.copyItem(at: try Self.fixtureFolder(), to: folder)
        defer { try? FileManager.default.removeItem(at: folder) }
        try FileManager.default.removeItem(at: folder.appendingPathComponent("generation_config.json"))
        let g = try await MFTokenizer.loadUncached(from: folder).forCheckpoint("inkling2-mini").generationDefaults
        #expect(g.temperature == 1.0 && g.topK == nil && g.topP == nil && g.minP == 0)
    }

    // MARK: Rendering

    @Test func singleUserMessageGetsTheDefaultEffort() throws {
        let out = try render([Message(role: .user, content: "Hello")])
        #expect(out == effortLine("0.9") + usr + txt + "Hello" + end)
    }

    @Test func effortFollowsTheSystemMessages() throws {
        let out = try render([Message(role: .system, content: "Be brief."),
                              Message(role: .user, content: "Hi")],
                             effort: .named("low"))
        #expect(out == sys + txt + "Be brief." + end + effortLine("0.2") + usr + txt + "Hi" + end)
    }

    @Test func effortLevelsPrintAsTheTemplatePrintsThem() throws {
        let cases: [(Effort, String)] = [
            (.named("none"), "0"), (.level(0), "0"), (.named("minimal"), "0.1"),
            (.named("medium"), "0.7"), (.named("high"), "0.9"), (.named("xhigh"), "0.99"),
            (.named("max"), "0.99"), (.level(0.5), "0.5"), (.level(0.25), "0.25"),
        ]
        for (effort, printed) in cases {
            let out = try render([Message(role: .user, content: "x")], effort: effort)
            #expect(out.hasPrefix(effortLine(printed)), "\(effort) printed as \(out)")
        }
    }

    @Test func outOfRangeOrUnknownEffortIsRejected() {
        #expect(throws: MFTokenizerError.self) {
            try render([Message(role: .user, content: "x")], effort: .level(1.0))
        }
        #expect(throws: MFTokenizerError.self) {
            try render([Message(role: .user, content: "x")], effort: .named("extreme"))
        }
    }

    @Test func onlySystemMessagesPutTheEffortLast() throws {
        let out = try render([Message(role: .system, content: "S")])
        #expect(out == sys + txt + "S" + end + effortLine("0.9"))
    }

    @Test func assistantTurnsKeepTheirThinking() throws {
        let out = try render([
            Message(role: .user, content: "Q1"),
            Message(role: .assistant, content: "A1", reasoningContent: "think 1"),
            Message(role: .user, content: "Q2"),
        ])
        #expect(out == effortLine("0.9") + usr + txt + "Q1" + end
                + mdl + thk + "think 1" + end + mdl + txt + "A1" + end + eos
                + usr + txt + "Q2" + end)
    }

    @Test func toolsCallsAndResults() throws {
        let weather = MFTokenizer.FunctionDefinition(
            name: "get_weather", description: "Get the weather",
            parameters: try json(#"{"type":"object","properties":{"unit":{"type":"string","enum":["c","f"]},"city":{"type":"string"}},"required":["city"]}"#))
        let out = try render([
            Message(role: .user, content: "Weather in Paris?"),
            Message(role: .assistant, content: nil,
                    toolCalls: [.init(id: "call_1", name: "get_weather",
                                      arguments: try json(#"{"unit":"c","city":"Paris"}"#))],
                    reasoningContent: "Need the weather."),
            Message(role: .tool, content: "18C", toolCallID: "call_1"),
            Message(role: .tool, content: "sunny", name: "get_weather"),
        ], tools: [weather])
        let declared = #"[{"description":"Get the weather","name":"get_weather","parameters":{"properties":{"city":{"type":"string"},"unit":{"enum":["c","f"],"type":"string"}},"required":["city"],"type":"object"},"type":"function"}]"#
        #expect(out == sys + "tool_declare<|content_xml|>" + declared + end
                + effortLine("0.9") + usr + txt + "Weather in Paris?" + end
                + mdl + thk + "Need the weather." + end
                + mdl + "get_weather" + invoke + #"{"name":"get_weather","args":{"city":"Paris","unit":"c"}}"# + end + eos
                + tool + "get_weather" + txt + "18C" + end
                + tool + "get_weather" + txt + "sunny" + end)
    }

    /// Known deviation, pinned on purpose: arguments arrive as `JSONValue`,
    /// whose objects have no key order, and reach the template with sorted
    /// keys. The template sorts the top level itself; transformers and
    /// llama.cpp keep the order nested objects arrived in, so a nested
    /// object written by the model in another order re-renders differently.
    @Test func nestedArgumentsAreCompactWithRawUnicode() throws {
        let out = try render([
            Message(role: .user, content: "Grüße"),
            Message(role: .assistant, content: "",
                    toolCalls: [.init(id: "c", name: "f",
                                      arguments: try json(#"{"b":1,"a":{"y":[1,2.5],"x":"é"}}"#))]),
        ])
        #expect(out == effortLine("0.9") + usr + txt + "Grüße" + end
                + mdl + "f" + invoke + #"{"name":"f","args":{"a":{"x":"é","y":[1,2.5]},"b":1}}"# + end + eos)
    }

    @Test func developerRoleIsRefused() {
        #expect(throws: MFTokenizerError.self) {
            try render([Message(role: .developer, content: "x"), Message(role: .user, content: "y")])
        }
    }

    @Test func tokensAreTheEncodedRender() throws {
        let messages = [Message(role: .user, content: "Hello there")]
        let ids = try tok.encodeInkling2Chat(messages: messages)
        #expect(ids == tok.encode(try render(messages), addBOS: false))
        #expect(ids.first == 200_002 && ids.last == 200_010)
        // The front door routes this dialect to the same render; the Qwen
        // effort names map onto the template's own names.
        #expect(try tok.encodeChat(messages: messages) == ids)
        #expect(try tok.encodeChat(messages: messages, reasoningEffort: .low)
                == tok.encodeInkling2Chat(messages: messages, effort: .named("low")))
    }

    @Test func userContinuationClosesTheCachedTurn() throws {
        let ids = tok.encodeTextContinuation(userContent: "Next?")
        #expect(ids == [200_006] + tok.encode(usr + txt + "Next?" + end, addBOS: false))
    }

    @Test func promptPlusGeneratedTurnPlusContinuationIsTheFullRender() throws {
        // The cache holds the prompt and the generated turn without its EOS
        // (sampled, never fed back); the continuation must make the whole
        // equal to rendering the conversation from scratch.
        let prompt = try tok.encodeInkling2Chat(messages: [Message(role: .user, content: "Q1")])
        let generated = tok.encode(mdl + thk + "hmm" + end + mdl + txt + "A1" + end, addBOS: false)
        let continued = prompt + generated + tok.encodeTextContinuation(userContent: "Q2")
        let full = try tok.encodeInkling2Chat(messages: [
            Message(role: .user, content: "Q1"),
            Message(role: .assistant, content: "A1", reasoningContent: "hmm"),
            Message(role: .user, content: "Q2"),
        ])
        #expect(continued == full)
    }

    @Test func effortNamesAreTrimmedLikeTheTemplateDoes() throws {
        let out = try render([Message(role: .user, content: "x")], effort: .named(" high\n"))
        #expect(out.hasPrefix(effortLine("0.9")))
    }

    @Test func aVocabularyWithMovedFramingIdsIsRefused() async throws {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("inkling2-tokenizer-\(UUID().uuidString)")
        try FileManager.default.copyItem(at: try Self.fixtureFolder(), to: folder)
        defer { try? FileManager.default.removeItem(at: folder) }
        let url = folder.appendingPathComponent("tokenizer.json")
        var root = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
        root["added_tokens"] = (root["added_tokens"] as! [[String: Any]]).map { token in
            var t = token
            switch t["content"] as! String {
            case "<|content_model_end_sampling|>": t["id"] = 200_007
            case "<|unused_200007|>": t["id"] = 200_006
            default: break
            }
            return t
        }
        try JSONSerialization.data(withJSONObject: root).write(to: url)
        await #expect(throws: MFTokenizerError.self) { _ = try await MFTokenizer.loadUncached(from: folder) }
    }

    // MARK: Tokenization

    @Test func specialTokensEncodeAsSingleIDs() {
        #expect(tok.encode(mdl + thk + end, addBOS: false) == [200_001, 200_008, 200_010])
        #expect(tok.encode("a<|content_text|>b", addBOS: false).contains(200_004))
    }

    @Test func wholeWordsInTheVocabularyIgnoreMerges() async throws {
        // The fixture's BPE has no merge for "ab", so BPE alone gives two
        // byte tokens; with ignore_merges a word that is a vocabulary entry
        // is one token.
        #expect(tok.encode("ab", addBOS: false).count == 2)
        let folder = try Self.copyFixture(addingToken: Inkling2TextEncoder.byteLevel("ab"), id: 1_000)
        defer { try? FileManager.default.removeItem(at: folder) }
        let t = try await MFTokenizer.loadUncached(from: folder)
        #expect(t.encode("ab", addBOS: false) == [1_000])
        #expect(t.encode("ab ab", addBOS: false).first == 1_000)   // the second word is " ab"
        #expect(t.encode(" ab", addBOS: false) != [1_000])
        #expect(t.encode("abc", addBOS: false).count == 3)
        #expect(t.encode(sys + "ab" + end, addBOS: false) == [200_002, 1_000, 200_010])
    }

    @Test func byteLevelMapsLikeGPT2() {
        #expect(Inkling2TextEncoder.byteLevel(" a") == "\u{0120}a")
        #expect(Inkling2TextEncoder.byteLevel("\n") == "\u{010A}")
        #expect(Inkling2TextEncoder.byteLevel("é") == "\u{00C3}\u{00A9}")
    }

    @Test func wordsSplitLikeGPT4o() throws {
        let regex = try NSRegularExpression(pattern: Inkling2TextEncoder.pattern)
        func split(_ text: String) -> [String] {
            let ns = text as NSString
            return regex.matches(in: text, range: NSRange(location: 0, length: ns.length))
                .map { ns.substring(with: $0.range) }
        }
        #expect(split("Hello world") == ["Hello", " world"])
        #expect(split("I'm 12345 ok!!\n\nnext") == ["I'm", " ", "123", "45", " ok", "!!\n\n", "next"])
        #expect(split("a   b") == ["a", "  ", " b"])
        #expect(split("CamelCase") == ["Camel", "Case"])
        // The vendor's split, which llama.cpp's collapsed rewrite does not
        // reproduce (it gives one word and `É`, `TAT`); see Inkling2TextEncoder.
        #expect(split("getÜbersicht") == ["get", "Übersicht"])
        #expect(split("ÉTAT") == ["ÉTAT"])
    }

    private static func copyFixture(addingToken token: String, id: Int) throws -> URL {
        let source = try fixtureFolder()
        let target = FileManager.default.temporaryDirectory
            .appendingPathComponent("inkling2-tokenizer-\(UUID().uuidString)")
        try FileManager.default.copyItem(at: source, to: target)
        let url = target.appendingPathComponent("tokenizer.json")
        var root = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
        var model = root["model"] as! [String: Any]
        var vocab = model["vocab"] as! [String: Any]
        vocab[token] = id
        model["vocab"] = vocab
        root["model"] = model
        try JSONSerialization.data(withJSONObject: root).write(to: url)
        return target
    }
}
