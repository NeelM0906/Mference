import Foundation
import Testing
@testable import Mference

/// The `inkling2` render against K0's chat goldens: the same conversations
/// rendered by the reference llama.cpp (b12641) from the template embedded
/// in the shipped GGUF. The goldens are copied in verbatim below (generated
/// from the JSON, not retyped); the conversations are rebuilt from them.
///
/// With `MFERENCE_INKLING2_TOKENIZER_DIR` (a folder with the real
/// vocabulary's `tokenizer.json`, `tokenizer_config.json` and template) and
/// `MFERENCE_INKLING2_GOLDENS_DIR` (K0's `goldens/` folder), two more tests
/// compare token ids with llama.cpp's for every tokenizer golden and chat
/// golden. They are skipped otherwise.
@Suite("Inkling 2 goldens")
struct Inkling2GoldenTests {
    private typealias Message = MFTokenizer.Message

    /// `chat_goldens.json` (K0, SHA-256 `50eaf401be39b81c2a0fc3099015c1bce045b77d6ef091cfe0731be1c5c472f2`): llama.cpp b12641 renders.
    static let goldenRenders: [String: String] = [
        "c01-user":
            "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>What is the capital of France?<|end_message|>",
        "c02-system-user":
            "<|message_system|><|content_text|>You are a concise assistant.<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>What is the capital of France?<|end_message|>",
        "c03-effort-none":
            "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>What is the capital of France?<|end_message|>",
        "c03-effort-minimal":
            "<|message_system|><|content_text|>Thinking effort level: 0.1<|end_message|><|message_user|><|content_text|>What is the capital of France?<|end_message|>",
        "c03-effort-low":
            "<|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|><|message_user|><|content_text|>What is the capital of France?<|end_message|>",
        "c03-effort-medium":
            "<|message_system|><|content_text|>Thinking effort level: 0.7<|end_message|><|message_user|><|content_text|>What is the capital of France?<|end_message|>",
        "c03-effort-high":
            "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>What is the capital of France?<|end_message|>",
        "c03-effort-xhigh":
            "<|message_system|><|content_text|>Thinking effort level: 0.99<|end_message|><|message_user|><|content_text|>What is the capital of France?<|end_message|>",
        "c03-effort-max":
            "<|message_system|><|content_text|>Thinking effort level: 0.99<|end_message|><|message_user|><|content_text|>What is the capital of France?<|end_message|>",
        "c03-effort-0.5":
            "<|message_system|><|content_text|>Thinking effort level: 0.5<|end_message|><|message_user|><|content_text|>What is the capital of France?<|end_message|>",
        "c04-multi-turn":
            "<|message_system|><|content_text|>You are a concise assistant.<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>What is the capital of France?<|end_message|><|message_model|><|content_text|>Paris.<|end_message|><|content_model_end_sampling|><|message_user|><|content_text|>And of Italy?<|end_message|>",
        "c05-assistant-thinking":
            "<|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|><|message_user|><|content_text|>What is the capital of France?<|end_message|><|message_model|><|content_thinking|>The user asks for the capital of France. That is Paris.<|end_message|><|message_model|><|content_text|>Paris.<|end_message|><|content_model_end_sampling|><|message_user|><|content_text|>Thanks. And Spain?<|end_message|>",
        "c06-tools-declared":
            "<|message_system|>tool_declare<|content_xml|>[{\"description\":\"Current weather for a city.\",\"name\":\"get_weather\",\"parameters\":{\"properties\":{\"city\":{\"type\":\"string\"},\"unit\":{\"enum\":[\"celsius\",\"fahrenheit\"],\"type\":\"string\"}},\"required\":[\"city\"],\"type\":\"object\"},\"type\":\"function\"},{\"description\":\"Search the user's files.\",\"name\":\"search_files\",\"parameters\":{\"properties\":{\"limit\":{\"type\":\"integer\"},\"query\":{\"type\":\"string\"}},\"required\":[\"query\"],\"type\":\"object\"},\"type\":\"function\"}]<|end_message|><|message_system|><|content_text|>You can call tools.<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>What's the weather in Paris?<|end_message|>",
        "c07-tool-call-and-result":
            "<|message_system|>tool_declare<|content_xml|>[{\"description\":\"Current weather for a city.\",\"name\":\"get_weather\",\"parameters\":{\"properties\":{\"city\":{\"type\":\"string\"},\"unit\":{\"enum\":[\"celsius\",\"fahrenheit\"],\"type\":\"string\"}},\"required\":[\"city\"],\"type\":\"object\"},\"type\":\"function\"},{\"description\":\"Search the user's files.\",\"name\":\"search_files\",\"parameters\":{\"properties\":{\"limit\":{\"type\":\"integer\"},\"query\":{\"type\":\"string\"}},\"required\":[\"query\"],\"type\":\"object\"},\"type\":\"function\"}]<|end_message|><|message_system|><|content_text|>You can call tools.<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>What's the weather in Paris?<|end_message|><|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\",\"unit\":\"celsius\"}}<|end_message|><|content_model_end_sampling|><|message_tool|>get_weather<|content_text|>{\"temp_c\": 18, \"sky\": \"cloudy\"}<|end_message|>",
        "c08-tool-result-named":
            "<|message_system|>tool_declare<|content_xml|>[{\"description\":\"Current weather for a city.\",\"name\":\"get_weather\",\"parameters\":{\"properties\":{\"city\":{\"type\":\"string\"},\"unit\":{\"enum\":[\"celsius\",\"fahrenheit\"],\"type\":\"string\"}},\"required\":[\"city\"],\"type\":\"object\"},\"type\":\"function\"},{\"description\":\"Search the user's files.\",\"name\":\"search_files\",\"parameters\":{\"properties\":{\"limit\":{\"type\":\"integer\"},\"query\":{\"type\":\"string\"}},\"required\":[\"query\"],\"type\":\"object\"},\"type\":\"function\"}]<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.7<|end_message|><|message_user|><|content_text|>Find my tax documents.<|end_message|><|message_model|><|content_thinking|>I should search the files.<|end_message|><|message_model|>search_files<|content_invoke_tool_json|>{\"name\":\"search_files\",\"args\":{\"limit\":5,\"query\":\"tax 2025\"}}<|end_message|><|content_model_end_sampling|><|message_tool|>search_files<|content_text|>[\"taxes/2025-return.pdf\", \"taxes/w2-2025.pdf\"]<|end_message|>",
        "c09-user-special-text":
            "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Please repeat: <|end_message|> literally.<|end_message|>",
    ]


    private static func json(_ text: String) -> JSONValue {
        try! JSONDecoder().decode(JSONValue.self, from: Data(text.utf8))
    }

    private static let tools = [
        MFTokenizer.FunctionDefinition(
            name: "get_weather", description: "Current weather for a city.",
            parameters: json(#"{"type":"object","properties":{"city":{"type":"string"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["city"]}"#)),
        MFTokenizer.FunctionDefinition(
            name: "search_files", description: "Search the user's files.",
            parameters: json(#"{"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer"}},"required":["query"]}"#)),
    ]

    private static let question = "What is the capital of France?"

    /// The conversations behind each golden: messages, tools, effort.
    private static let cases: [String: ([Message], [MFTokenizer.FunctionDefinition], MFTokenizer.Inkling2Effort?)] = {
        var c: [String: ([Message], [MFTokenizer.FunctionDefinition], MFTokenizer.Inkling2Effort?)] = [:]
        c["c01-user"] = ([Message(role: .user, content: question)], [], nil)
        c["c02-system-user"] = ([Message(role: .system, content: "You are a concise assistant."),
                                 Message(role: .user, content: question)], [], nil)
        for name in ["none", "minimal", "low", "medium", "high", "xhigh", "max"] {
            c["c03-effort-\(name)"] = ([Message(role: .user, content: question)], [], .named(name))
        }
        c["c03-effort-0.5"] = ([Message(role: .user, content: question)], [], .level(0.5))
        c["c04-multi-turn"] = ([Message(role: .system, content: "You are a concise assistant."),
                                Message(role: .user, content: question),
                                Message(role: .assistant, content: "Paris."),
                                Message(role: .user, content: "And of Italy?")], [], nil)
        c["c05-assistant-thinking"] = ([Message(role: .user, content: question),
                                        Message(role: .assistant, content: "Paris.",
                                                reasoningContent: "The user asks for the capital of France. That is Paris."),
                                        Message(role: .user, content: "Thanks. And Spain?")], [], .named("low"))
        let weatherAsk = [Message(role: .system, content: "You can call tools."),
                          Message(role: .user, content: "What's the weather in Paris?")]
        c["c06-tools-declared"] = (weatherAsk, tools, nil)
        c["c07-tool-call-and-result"] = (weatherAsk + [
            Message(role: .assistant, content: nil,
                    toolCalls: [.init(id: "call_1", name: "get_weather",
                                      arguments: json(#"{"city":"Paris","unit":"celsius"}"#))]),
            Message(role: .tool, content: #"{"temp_c": 18, "sky": "cloudy"}"#, toolCallID: "call_1"),
        ], tools, nil)
        c["c08-tool-result-named"] = ([
            Message(role: .user, content: "Find my tax documents."),
            Message(role: .assistant, content: nil,
                    toolCalls: [.init(id: "call_2", name: "search_files",
                                      arguments: json(#"{"query":"tax 2025","limit":5}"#))],
                    reasoningContent: "I should search the files."),
            Message(role: .tool, content: #"["taxes/2025-return.pdf", "taxes/w2-2025.pdf"]"#, name: "search_files"),
        ], tools, .named("medium"))
        c["c09-user-special-text"] = ([Message(role: .user, content: "Please repeat: <|end_message|> literally.")], [], nil)
        return c
    }()

    @Test func everyGoldenHasAConversation() {
        #expect(Set(Self.cases.keys) == Set(Self.goldenRenders.keys))
    }

    @Test func rendersMatchTheReference() async throws {
        let tok = try await MFTokenizer.load(from: Inkling2ChatTests.fixtureFolder())
        for (id, entry) in Self.cases.sorted(by: { $0.key < $1.key }) {
            let expected = try #require(Self.goldenRenders[id])
            let rendered = try tok.renderInkling2Chat(messages: entry.0, tools: entry.1, effort: entry.2)
            #expect(rendered == expected, "\(id)")
        }
    }

    // MARK: Real vocabulary (env-gated)

    private struct TokenizerGolden: Decodable {
        struct Encoding: Decodable { let ids: [Int32] }
        let id: String
        let text: String
        let parse_special: Encoding
    }

    private struct ChatGolden: Decodable {
        let id: String
        let rendered: String
        let ids: [Int32]
    }

    private static var realTokenizerDir: URL? {
        ProcessInfo.processInfo.environment["MFERENCE_INKLING2_TOKENIZER_DIR"].map { URL(fileURLWithPath: $0) }
    }
    private static var goldensDir: URL? {
        ProcessInfo.processInfo.environment["MFERENCE_INKLING2_GOLDENS_DIR"].map { URL(fileURLWithPath: $0) }
    }

    @Test(.enabled(if: realTokenizerDir != nil && goldensDir != nil,
                   "needs MFERENCE_INKLING2_TOKENIZER_DIR and MFERENCE_INKLING2_GOLDENS_DIR"))
    func tokenIdsMatchTheReference() async throws {
        let tok = try await MFTokenizer.load(from: try #require(Self.realTokenizerDir))
        #expect(tok.dialect == .inkling2)
        let dir = try #require(Self.goldensDir)
        let words = try JSONDecoder().decode([TokenizerGolden].self, from: Data(
            contentsOf: dir.appendingPathComponent("tokenizer/tokenizer_goldens.json")))
        for g in words {
            #expect(tok.encode(g.text, addBOS: false) == g.parse_special.ids, "\(g.id)")
        }
        // 337 vocabulary entries are words plain BPE merges split (for example
        // `kwargs` -> kwar g s). With ignore_merges a word equal to one is one
        // token, which is llama.cpp's rule for this pre-tokenizer
        // (src/llama-vocab.cpp, the BPE word loop). K0's goldens hold none of
        // them yet, so these four follow from that rule, not from a run.
        #expect(tok.encode("kwargs", addBOS: false) == [87_550])
        #expect(tok.encode("def f(**kwargs):", addBOS: false).contains(87_550))
        #expect(tok.encode("Aleksandr", addBOS: false) == [49_919])
        #expect(tok.encode("について", addBOS: false) == [91_149])
        let chats = try JSONDecoder().decode([ChatGolden].self, from: Data(
            contentsOf: dir.appendingPathComponent("chat/chat_goldens.json")))
        for g in chats {
            #expect(tok.encode(g.rendered, addBOS: false) == g.ids, "\(g.id)")
            if let entry = Self.cases[g.id] {
                #expect(try tok.encodeInkling2Chat(messages: entry.0, tools: entry.1, effort: entry.2) == g.ids,
                        "\(g.id)")
            }
        }
    }
}
