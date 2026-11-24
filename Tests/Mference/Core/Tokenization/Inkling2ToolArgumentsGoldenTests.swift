import Foundation
import Testing
@testable import Mference

/// The Inkling 2 render of tool-call histories against K0's tool-argument
/// goldens: llama.cpp b12641 rendering conversations whose call arguments are
/// JSON strings (`chat_cases_tool_args.json`, SHA-256 `b7a333382583d36942741488b81afaa38d01f32a361c27cf1356aea4c4b4db98`;
/// renders `chat_goldens_tool_args.json`, SHA-256 `50bbc49260b755f0cd5ca35be82cdf134d78cad851f9e6e92a58dde7317639b6`, both in
/// `Tests/Mference/Fixtures/Inkling2/`). The cases and renders below are
/// generated from those files, not retyped. Each case: a user message, the
/// assistant's call(s) to `query_records`, an "ok" result per call.
///
/// `t07` holds an integer above Int64, where the references disagree:
/// llama.cpp converts every JSON integer with `get<int64_t>()` and wraps it,
/// Python keeps the digits. Mference refuses it, as before (the server answers
/// 400, "historical tool arguments cannot be represented exactly").
///
/// With `MFERENCE_INKLING2_TOKENIZER_DIR` (the real vocabulary) the token ids
/// are compared with llama.cpp's too; skipped otherwise.
@Suite("Inkling 2 tool-call argument goldens")
struct Inkling2ToolArgumentsGoldenTests {
    private typealias Message = MFTokenizer.Message

    struct Case {
        let id: String
        let calls: [(id: String, arguments: String)]
        let rendered: String
        let ids: [Int32]
    }

    static let tool = MFTokenizer.FunctionDefinition(name: "query_records", description: "Query the user's records.",
                                                     parameters: .object(["type": .string("object")]))

    static let cases: [Case] = [
        .init(id: "t01-nested-order",
              calls: [("call_1", "{\"query\":\"tax\",\"filter\":{\"year\":2025,\"kind\":\"pdf\",\"tags\":[\"b\",\"a\"],\"range\":{\"to\":10,\"from\":1}},\"limit\":5}")],
              rendered: "<|message_system|>tool_declare<|content_xml|>[{\"description\":\"Query the user's records.\",\"name\":\"query_records\",\"parameters\":{\"type\":\"object\"},\"type\":\"function\"}]<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Look this up in my records.<|end_message|><|message_model|>query_records<|content_invoke_tool_json|>{\"name\":\"query_records\",\"args\":{\"filter\":{\"year\":2025,\"kind\":\"pdf\",\"tags\":[\"b\",\"a\"],\"range\":{\"to\":10,\"from\":1}},\"limit\":5,\"query\":\"tax\"}}<|end_message|><|content_model_end_sampling|><|message_tool|>query_records<|content_text|>ok<|end_message|>",
              ids: [200002, 2251, 62, 76529, 200024, 58, 90, 1, 70469, 1, 25, 1, 40729, 149876, 150409, 6, 82, 155911, 13, 1, 11, 1, 930, 1, 25, 1, 35790, 62, 79816, 1, 11, 1, 71968, 1, 25, 90, 1, 1008, 1, 25, 1, 3156, 1, 92, 11, 1, 1008, 1, 25, 1, 36404, 1, 92, 60, 200010, 200002, 200004, 34182, 279, 154554, 150628, 25, 220, 15, 13, 24, 200010, 200000, 200004, 31410, 149928, 150069, 149881, 150084, 155911, 13, 200010, 200001, 35790, 62, 79816, 200049, 90, 1, 930, 1, 25, 1, 35790, 62, 79816, 1, 11, 1, 70747, 1, 25, 90, 1, 30733, 1, 25, 90, 1, 1258, 1, 25, 68111, 20, 11, 1, 2678, 1, 25, 1, 3909, 1, 11, 1, 73841, 1, 25, 58, 1, 65, 1, 11, 1, 64, 1, 60, 11, 1, 1499, 1, 25, 90, 1, 278, 1, 25, 68026, 11, 1, 778, 1, 25, 16, 92, 92, 11, 1, 1982, 1, 25, 20, 11, 1, 35790, 1, 25, 1, 2523, 1, 92, 92, 200010, 200006, 200003, 35790, 62, 79816, 200004, 873, 200010]),
        .init(id: "t02-numbers",
              calls: [("call_1", "{\"f1\":1.0,\"f2\":5.0,\"f3\":1e3,\"f4\":-0.0,\"f5\":0.1,\"f6\":2.5e-7,\"f7\":1e16,\"f8\":9.9e15,\"f9\":0.00001,\"f10\":123.456,\"i1\":-3,\"i2\":0,\"i3\":9223372036854775807}")],
              rendered: "<|message_system|>tool_declare<|content_xml|>[{\"description\":\"Query the user's records.\",\"name\":\"query_records\",\"parameters\":{\"type\":\"object\"},\"type\":\"function\"}]<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Look this up in my records.<|end_message|><|message_model|>query_records<|content_invoke_tool_json|>{\"name\":\"query_records\",\"args\":{\"f1\":1.0,\"f10\":123.456,\"f2\":5.0,\"f3\":1000.0,\"f4\":-0.0,\"f5\":0.1,\"f6\":2.5e-07,\"f7\":1e+16,\"f8\":9900000000000000.0,\"f9\":1e-05,\"i1\":-3,\"i2\":0,\"i3\":9223372036854775807}}<|end_message|><|content_model_end_sampling|><|message_tool|>query_records<|content_text|>ok<|end_message|>",
              ids: [200002, 2251, 62, 76529, 200024, 58, 90, 1, 70469, 1, 25, 1, 40729, 149876, 150409, 6, 82, 155911, 13, 1, 11, 1, 930, 1, 25, 1, 35790, 62, 79816, 1, 11, 1, 71968, 1, 25, 90, 1, 1008, 1, 25, 1, 3156, 1, 92, 11, 1, 1008, 1, 25, 1, 36404, 1, 92, 60, 200010, 200002, 200004, 34182, 279, 154554, 150628, 25, 220, 15, 13, 24, 200010, 200000, 200004, 31410, 149928, 150069, 149881, 150084, 155911, 13, 200010, 200001, 35790, 62, 79816, 200049, 90, 1, 930, 1, 25, 1, 35790, 62, 79816, 1, 11, 1, 70747, 1, 25, 90, 1, 69, 16, 1, 25, 16, 13, 15, 11, 1, 69, 68026, 1, 25, 70904, 13, 81554, 11, 1, 69, 17, 1, 25, 20, 13, 15, 11, 1, 69, 18, 1, 25, 68189, 15, 13, 15, 11, 1, 69, 19, 1, 25, 12, 15, 13, 15, 11, 1, 69, 20, 1, 25, 15, 13, 16, 11, 1, 69, 21, 1, 25, 17, 13, 20, 68, 12, 68620, 11, 1, 69, 22, 1, 25, 16, 68, 10, 68080, 11, 1, 69, 23, 1, 25, 92914, 68128, 68128, 68128, 68128, 15, 13, 15, 11, 1, 69, 24, 1, 25, 16, 68, 12, 68435, 11, 1, 72, 16, 1, 25, 12, 18, 11, 1, 72, 17, 1, 25, 15, 11, 1, 72, 18, 1, 25, 104259, 85883, 72500, 101327, 94654, 89230, 22, 92, 92, 200010, 200006, 200003, 35790, 62, 79816, 200004, 873, 200010]),
        .init(id: "t03-strings",
              calls: [("call_1", "{\"text\":\"café ☕ 東京 😀\",\"esc\":\"a/b \\\"q\\\" back\\\\slash\\nnl\\ttab\",\"ctl\":\"\\u0001x\",\"uesc\":\"\\u00e9\\ud83d\\ude00\"}")],
              rendered: "<|message_system|>tool_declare<|content_xml|>[{\"description\":\"Query the user's records.\",\"name\":\"query_records\",\"parameters\":{\"type\":\"object\"},\"type\":\"function\"}]<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Look this up in my records.<|end_message|><|message_model|>query_records<|content_invoke_tool_json|>{\"name\":\"query_records\",\"args\":{\"ctl\":\"\\u0001x\",\"esc\":\"a/b \\\"q\\\" back\\\\slash\\nnl\\ttab\",\"text\":\"café ☕ 東京 😀\",\"uesc\":\"é😀\"}}<|end_message|><|content_model_end_sampling|><|message_tool|>query_records<|content_text|>ok<|end_message|>",
              ids: [200002, 2251, 62, 76529, 200024, 58, 90, 1, 70469, 1, 25, 1, 40729, 149876, 150409, 6, 82, 155911, 13, 1, 11, 1, 930, 1, 25, 1, 35790, 62, 79816, 1, 11, 1, 71968, 1, 25, 90, 1, 1008, 1, 25, 1, 3156, 1, 92, 11, 1, 1008, 1, 25, 1, 36404, 1, 92, 60, 200010, 200002, 200004, 34182, 279, 154554, 150628, 25, 220, 15, 13, 24, 200010, 200000, 200004, 31410, 149928, 150069, 149881, 150084, 155911, 13, 200010, 200001, 35790, 62, 79816, 200049, 90, 1, 930, 1, 25, 1, 35790, 62, 79816, 1, 11, 1, 70747, 1, 25, 90, 1, 44018, 1, 25, 1, 59, 84, 68128, 16, 87, 1, 11, 1, 32953, 1, 25, 1, 64, 14, 65, 196978, 1, 80, 59, 1, 150387, 68566, 34332, 59, 77, 32168, 59, 1270, 332, 1, 11, 1, 911, 1, 25, 1, 85593, 220, 26460, 220, 118229, 220, 110689, 1, 11, 1, 339, 796, 1, 25, 1, 5164, 110689, 1, 92, 92, 200010, 200006, 200003, 35790, 62, 79816, 200004, 873, 200010]),
        .init(id: "t04-structure",
              calls: [("call_1", "{\"items\":[{\"b\":1,\"a\":2},{\"z\":null,\"y\":true,\"x\":false}],\"empty_obj\":{},\"empty_arr\":[],\"nested_arr\":[[1,2],[]]}")],
              rendered: "<|message_system|>tool_declare<|content_xml|>[{\"description\":\"Query the user's records.\",\"name\":\"query_records\",\"parameters\":{\"type\":\"object\"},\"type\":\"function\"}]<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Look this up in my records.<|end_message|><|message_model|>query_records<|content_invoke_tool_json|>{\"name\":\"query_records\",\"args\":{\"empty_arr\":[],\"empty_obj\":{},\"items\":[{\"b\":1,\"a\":2},{\"z\":null,\"y\":true,\"x\":false}],\"nested_arr\":[[1,2],[]]}}<|end_message|><|content_model_end_sampling|><|message_tool|>query_records<|content_text|>ok<|end_message|>",
              ids: [200002, 2251, 62, 76529, 200024, 58, 90, 1, 70469, 1, 25, 1, 40729, 149876, 150409, 6, 82, 155911, 13, 1, 11, 1, 930, 1, 25, 1, 35790, 62, 79816, 1, 11, 1, 71968, 1, 25, 90, 1, 1008, 1, 25, 1, 3156, 1, 92, 11, 1, 1008, 1, 25, 1, 36404, 1, 92, 60, 200010, 200002, 200004, 34182, 279, 154554, 150628, 25, 220, 15, 13, 24, 200010, 200000, 200004, 31410, 149928, 150069, 149881, 150084, 155911, 13, 200010, 200001, 35790, 62, 79816, 200049, 90, 1, 930, 1, 25, 1, 35790, 62, 79816, 1, 11, 1, 70747, 1, 25, 90, 1, 3741, 62, 4754, 1, 25, 69231, 11, 1, 3741, 62, 4698, 1, 25, 70044, 11, 1, 70681, 1, 25, 58, 90, 1, 65, 1, 25, 16, 11, 1, 64, 1, 25, 17, 92, 11, 90, 1, 89, 1, 25, 2451, 11, 1, 88, 1, 25, 1750, 11, 1, 87, 1, 25, 2121, 92, 60, 11, 1, 112324, 62, 4754, 1, 25, 58, 58, 16, 11, 17, 60, 11, 69231, 60, 92, 92, 200010, 200006, 200003, 35790, 62, 79816, 200004, 873, 200010]),
        .init(id: "t05-whitespace",
              calls: [("call_1", "{ \"b\" : 1 ,\n  \"a\" : { \"y\": 2, \"x\": 1 } }")],
              rendered: "<|message_system|>tool_declare<|content_xml|>[{\"description\":\"Query the user's records.\",\"name\":\"query_records\",\"parameters\":{\"type\":\"object\"},\"type\":\"function\"}]<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Look this up in my records.<|end_message|><|message_model|>query_records<|content_invoke_tool_json|>{\"name\":\"query_records\",\"args\":{\"a\":{\"y\":2,\"x\":1},\"b\":1}}<|end_message|><|content_model_end_sampling|><|message_tool|>query_records<|content_text|>ok<|end_message|>",
              ids: [200002, 2251, 62, 76529, 200024, 58, 90, 1, 70469, 1, 25, 1, 40729, 149876, 150409, 6, 82, 155911, 13, 1, 11, 1, 930, 1, 25, 1, 35790, 62, 79816, 1, 11, 1, 71968, 1, 25, 90, 1, 1008, 1, 25, 1, 3156, 1, 92, 11, 1, 1008, 1, 25, 1, 36404, 1, 92, 60, 200010, 200002, 200004, 34182, 279, 154554, 150628, 25, 220, 15, 13, 24, 200010, 200000, 200004, 31410, 149928, 150069, 149881, 150084, 155911, 13, 200010, 200001, 35790, 62, 79816, 200049, 90, 1, 930, 1, 25, 1, 35790, 62, 79816, 1, 11, 1, 70747, 1, 25, 90, 1, 64, 1, 25, 90, 1, 88, 1, 25, 17, 11, 1, 87, 1, 25, 16, 92, 11, 1, 65, 1, 25, 16, 92, 92, 200010, 200006, 200003, 35790, 62, 79816, 200004, 873, 200010]),
        .init(id: "t06-two-calls",
              calls: [("call_1", "{\"query\":\"tax\",\"filter\":{\"year\":2025,\"kind\":\"pdf\",\"tags\":[\"b\",\"a\"],\"range\":{\"to\":10,\"from\":1}},\"limit\":5}"), ("call_2", "{\"z\":{\"b\":2,\"a\":1}}")],
              rendered: "<|message_system|>tool_declare<|content_xml|>[{\"description\":\"Query the user's records.\",\"name\":\"query_records\",\"parameters\":{\"type\":\"object\"},\"type\":\"function\"}]<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Look this up in my records.<|end_message|><|message_model|>query_records<|content_invoke_tool_json|>{\"name\":\"query_records\",\"args\":{\"filter\":{\"year\":2025,\"kind\":\"pdf\",\"tags\":[\"b\",\"a\"],\"range\":{\"to\":10,\"from\":1}},\"limit\":5,\"query\":\"tax\"}}<|end_message|><|message_model|>query_records<|content_invoke_tool_json|>{\"name\":\"query_records\",\"args\":{\"z\":{\"b\":2,\"a\":1}}}<|end_message|><|content_model_end_sampling|><|message_tool|>query_records<|content_text|>ok<|end_message|><|message_tool|>query_records<|content_text|>ok<|end_message|>",
              ids: [200002, 2251, 62, 76529, 200024, 58, 90, 1, 70469, 1, 25, 1, 40729, 149876, 150409, 6, 82, 155911, 13, 1, 11, 1, 930, 1, 25, 1, 35790, 62, 79816, 1, 11, 1, 71968, 1, 25, 90, 1, 1008, 1, 25, 1, 3156, 1, 92, 11, 1, 1008, 1, 25, 1, 36404, 1, 92, 60, 200010, 200002, 200004, 34182, 279, 154554, 150628, 25, 220, 15, 13, 24, 200010, 200000, 200004, 31410, 149928, 150069, 149881, 150084, 155911, 13, 200010, 200001, 35790, 62, 79816, 200049, 90, 1, 930, 1, 25, 1, 35790, 62, 79816, 1, 11, 1, 70747, 1, 25, 90, 1, 30733, 1, 25, 90, 1, 1258, 1, 25, 68111, 20, 11, 1, 2678, 1, 25, 1, 3909, 1, 11, 1, 73841, 1, 25, 58, 1, 65, 1, 11, 1, 64, 1, 60, 11, 1, 1499, 1, 25, 90, 1, 278, 1, 25, 68026, 11, 1, 778, 1, 25, 16, 92, 92, 11, 1, 1982, 1, 25, 20, 11, 1, 35790, 1, 25, 1, 2523, 1, 92, 92, 200010, 200001, 35790, 62, 79816, 200049, 90, 1, 930, 1, 25, 1, 35790, 62, 79816, 1, 11, 1, 70747, 1, 25, 90, 1, 89, 1, 25, 90, 1, 65, 1, 25, 17, 11, 1, 64, 1, 25, 16, 92, 92, 92, 200010, 200006, 200003, 35790, 62, 79816, 200004, 873, 200010, 200003, 35790, 62, 79816, 200004, 873, 200010]),
        .init(id: "t07-int-beyond-int64",
              calls: [("call_1", "{\"big\":12345678901234567890,\"small\":1}")],
              rendered: "<|message_system|>tool_declare<|content_xml|>[{\"description\":\"Query the user's records.\",\"name\":\"query_records\",\"parameters\":{\"type\":\"object\"},\"type\":\"function\"}]<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Look this up in my records.<|end_message|><|message_model|>query_records<|content_invoke_tool_json|>{\"name\":\"query_records\",\"args\":{\"big\":-6101065172474983726,\"small\":1}}<|end_message|><|content_model_end_sampling|><|message_tool|>query_records<|content_text|>ok<|end_message|>",
              ids: [200002, 2251, 62, 76529, 200024, 58, 90, 1, 70469, 1, 25, 1, 40729, 149876, 150409, 6, 82, 155911, 13, 1, 11, 1, 930, 1, 25, 1, 35790, 62, 79816, 1, 11, 1, 71968, 1, 25, 90, 1, 1008, 1, 25, 1, 3156, 1, 92, 11, 1, 1008, 1, 25, 1, 36404, 1, 92, 60, 200010, 200002, 200004, 34182, 279, 154554, 150628, 25, 220, 15, 13, 24, 200010, 200000, 200004, 31410, 149928, 150069, 149881, 150084, 155911, 13, 200010, 200001, 35790, 62, 79816, 200049, 90, 1, 930, 1, 25, 1, 35790, 62, 79816, 1, 11, 1, 70747, 1, 25, 90, 1, 3197, 1, 25, 12, 85265, 71619, 95480, 80506, 96367, 88347, 21, 11, 1, 1916, 1, 25, 16, 92, 92, 200010, 200006, 200003, 35790, 62, 79816, 200004, 873, 200010]),
    ]

    /// Cases whose arguments a template value carries exactly.
    static var exact: [Case] { cases.filter { $0.id != "t07-int-beyond-int64" } }

    /// The case's conversation, each call's arguments given as `texts` (by
    /// default the case's own raw strings).
    static func messages(_ c: Case, texts: [String]? = nil) throws -> [MFTokenizer.Message] {
        let texts = texts ?? c.calls.map(\.arguments)
        let calls = try zip(c.calls, texts).map { call, text in
            MFTokenizer.HistoricalToolCall(
                id: call.id, name: "query_records",
                arguments: try JSONDecoder().decode(JSONValue.self, from: Data(text.utf8)),
                argumentsText: text)
        }
        return [Message(role: .user, content: "Look this up in my records."),
                Message(role: .assistant, content: "", toolCalls: calls)]
            + c.calls.map { Message(role: .tool, content: "ok", toolCallID: $0.id) }
    }

    @Test func rendersMatchTheReference() async throws {
        let tok = try await MFTokenizer.load(from: Inkling2ChatTests.fixtureFolder())
        for c in Self.exact {
            let rendered = try tok.renderInkling2Chat(messages: try Self.messages(c), tools: [Self.tool])
            #expect(rendered == c.rendered, "\(c.id)")
        }
    }

    /// The round trip prefix reuse needs: a call the model writes as the
    /// reference renders it goes through the tool-call parser, back out as the
    /// arguments the server returns, and into the next request's history,
    /// which renders the same text again.
    @Test func aReferenceCallReRendersToItself() async throws {
        let tok = try await MFTokenizer.load(from: Inkling2ChatTests.fixtureFolder())
        let invoke = "<|content_invoke_tool_json|>", end = "<|end_message|>"
        for c in Self.exact {
            let bodies = c.rendered.components(separatedBy: invoke).dropFirst()
                .map { String($0.components(separatedBy: end)[0]) }
            #expect(bodies.count == c.calls.count, "\(c.id)")
            let texts = try bodies.map { body in
                try Inkling2ToolCallParser().parse(body, headerName: "query_records",
                                                   allowedTools: ["query_records"], id: "call").argumentsJSON
            }
            let again = try tok.renderInkling2Chat(messages: try Self.messages(c, texts: texts), tools: [Self.tool])
            #expect(again == c.rendered, "\(c.id)")
        }
    }

    /// `t07`: the reference wraps the integer; Mference refuses to render it.
    @Test func anIntegerBeyondInt64IsRefused() async throws {
        let tok = try await MFTokenizer.load(from: Inkling2ChatTests.fixtureFolder())
        let c = try #require(Self.cases.first { $0.id == "t07-int-beyond-int64" })
        #expect(c.rendered.contains("-6101065172474983726"), "llama.cpp's wrapped value")
        #expect(throws: (any Error).self) {
            _ = try tok.renderInkling2Chat(messages: try Self.messages(c), tools: [Self.tool])
        }
    }

    private static var realTokenizerDir: URL? {
        ProcessInfo.processInfo.environment["MFERENCE_INKLING2_TOKENIZER_DIR"].map { URL(fileURLWithPath: $0) }
    }

    @Test(.enabled(if: realTokenizerDir != nil, "needs MFERENCE_INKLING2_TOKENIZER_DIR"))
    func tokenIdsMatchTheReference() async throws {
        let tok = try await MFTokenizer.load(from: try #require(Self.realTokenizerDir))
        #expect(tok.dialect == .inkling2)
        for c in Self.exact {
            #expect(tok.encode(c.rendered, addBOS: false) == c.ids, "\(c.id)")
            #expect(try tok.encodeInkling2Chat(messages: try Self.messages(c), tools: [Self.tool]) == c.ids,
                    "\(c.id)")
        }
    }
}
