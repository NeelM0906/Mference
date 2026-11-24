import Foundation
import Testing
@testable import Mference

/// Inkling 2 renders a tool call's arguments from their JSON text
/// (`Inkling2ToolArguments`), as transformers and llama.cpp do: nested objects
/// in the text's order, numbers of the text's kind. Expected strings are
/// Python's: `json.dumps(json.loads(raw), ensure_ascii=False,
/// separators=(",", ":"))` for the compact text, and the template's
/// `dictsort` plus `tojson` for the rendered `args`. Token-for-token
/// comparisons with the reference render are in `Inkling2ToolArgumentsGoldenTests`.
@Suite("Inkling 2 tool-call arguments from their text")
struct Inkling2ToolArgumentsTests {
    let tok: MFTokenizer

    init() async throws {
        self.tok = try await MFTokenizer.load(from: Inkling2ChatTests.fixtureFolder())
    }

    private typealias Message = MFTokenizer.Message

    /// (raw arguments text, Python's compact dump, the template's rendered `args`).
    static let cases: [(name: String, raw: String, compact: String, args: String)] = [
        ("nested order",
         #"{"query":"tax","filter":{"year":2025,"kind":"pdf","tags":["b","a"],"range":{"to":10,"from":1}},"limit":5}"#,
         #"{"query":"tax","filter":{"year":2025,"kind":"pdf","tags":["b","a"],"range":{"to":10,"from":1}},"limit":5}"#,
         #"{"filter":{"year":2025,"kind":"pdf","tags":["b","a"],"range":{"to":10,"from":1}},"limit":5,"query":"tax"}"#),
        ("numbers",
         #"{"f1":1.0,"f2":5.0,"f3":1e3,"f4":-0.0,"f5":0.1,"f6":2.5e-7,"f7":1e16,"f8":9.9e15,"f9":0.00001,"f10":123.456,"i1":-3,"i2":0,"i3":9223372036854775807}"#,
         #"{"f1":1.0,"f2":5.0,"f3":1000.0,"f4":-0.0,"f5":0.1,"f6":2.5e-07,"f7":1e+16,"f8":9900000000000000.0,"f9":1e-05,"f10":123.456,"i1":-3,"i2":0,"i3":9223372036854775807}"#,
         #"{"f1":1.0,"f10":123.456,"f2":5.0,"f3":1000.0,"f4":-0.0,"f5":0.1,"f6":2.5e-07,"f7":1e+16,"f8":9900000000000000.0,"f9":1e-05,"i1":-3,"i2":0,"i3":9223372036854775807}"#),
        ("strings",
         #"{"text":"café ☕ 東京 😀","esc":"a/b \"q\" back\\slash\nnl\ttab","ctl":"\u0001x","uesc":"é😀"}"#,
         #"{"text":"café ☕ 東京 😀","esc":"a/b \"q\" back\\slash\nnl\ttab","ctl":"\u0001x","uesc":"é😀"}"#,
         #"{"ctl":"\u0001x","esc":"a/b \"q\" back\\slash\nnl\ttab","text":"café ☕ 東京 😀","uesc":"é😀"}"#),
        ("structure",
         #"{"items":[{"b":1,"a":2},{"z":null,"y":true,"x":false}],"empty_obj":{},"empty_arr":[],"nested_arr":[[1,2],[]]}"#,
         #"{"items":[{"b":1,"a":2},{"z":null,"y":true,"x":false}],"empty_obj":{},"empty_arr":[],"nested_arr":[[1,2],[]]}"#,
         #"{"empty_arr":[],"empty_obj":{},"items":[{"b":1,"a":2},{"z":null,"y":true,"x":false}],"nested_arr":[[1,2],[]]}"#),
        ("whitespace",
         "{ \"b\" : 1 ,\n  \"a\" : { \"y\": 2, \"x\": 1 } }",
         #"{"b":1,"a":{"y":2,"x":1}}"#,
         #"{"a":{"y":2,"x":1},"b":1}"#),
        ("repeated key: first place, last value, as a Python dict",
         #"{"a":1,"b":2,"a":3}"#,
         #"{"a":3,"b":2}"#,
         #"{"a":3,"b":2}"#),
    ]

    @Test func theCompactTextIsPythons() throws {
        for c in Self.cases {
            let value = try #require(Inkling2ToolArguments.templateValue(c.raw), "\(c.name)")
            #expect(try Inkling2ToolArguments.compactText(value) == c.compact, "\(c.name)")
            // The compact text parses back to itself: the round trip a client's echo takes.
            let again = try #require(Inkling2ToolArguments.templateValue(c.compact), "\(c.name)")
            #expect(try Inkling2ToolArguments.compactText(again) == c.compact, "\(c.name)")
        }
    }

    @Test func aCallRendersFromItsText() throws {
        for c in Self.cases {
            let parsed = try JSONDecoder().decode(JSONValue.self, from: Data(c.raw.utf8))
            let out = try tok.renderInkling2Chat(messages: [
                Message(role: .user, content: "Look this up."),
                Message(role: .assistant, content: "",
                        toolCalls: [.init(id: "call_1", name: "query_records", arguments: parsed,
                                          argumentsText: c.raw)]),
            ])
            let call = "<|message_model|>query_records<|content_invoke_tool_json|>"
                + #"{"name":"query_records","args":"# + c.args + "}<|end_message|><|content_model_end_sampling|>"
            #expect(out.hasSuffix(call), "\(c.name): \(out)")
        }
    }

    /// Without a text, or with one that does not parse as an object, the call
    /// renders from `arguments` exactly as before: nested keys sorted
    /// (`Inkling2ChatTests.nestedArgumentsAreCompactWithRawUnicode`), and
    /// `1.0` printed `1`, because `JSONDecoder` reads an integral `1.0`, `1e3`
    /// or `-0.0` into `JSONValue.integer`.
    @Test func withoutAUsableTextTheParsedValueRendersAsBefore() throws {
        let raw = #"{"b":1,"a":{"y":2,"x":1.0}}"#
        let parsed = try JSONDecoder().decode(JSONValue.self, from: Data(raw.utf8))
        func render(_ text: String?) throws -> String {
            try tok.renderInkling2Chat(messages: [
                Message(role: .user, content: "x"),
                Message(role: .assistant, content: "",
                        toolCalls: [.init(id: "c", name: "f", arguments: parsed, argumentsText: text)]),
            ])
        }
        let before = try render(nil)
        #expect(before.contains(#"{"name":"f","args":{"a":{"x":1,"y":2},"b":1}}"#), "\(before)")
        #expect(try render("[1]") == before)
        #expect(try render(#"{"a":1} trailing"#) == before)
        #expect(try render(raw).contains(#"{"name":"f","args":{"a":{"y":2,"x":1.0},"b":1}}"#))
    }

    /// Text no template value carries exactly, or that is not one JSON
    /// object, is refused; the caller falls back to `arguments`.
    @Test func whatCannotBeCarriedExactlyIsRefused() {
        let deep = String(repeating: #"{"a":"#, count: JSONValue.maximumDepth + 1) + "1"
            + String(repeating: "}", count: JSONValue.maximumDepth + 1)
        let refused = [
            "", "[1]", "1", #""s""#, "{", #"{"a":1} x"#, #"{"a":1,}"#, #"{a:1}"#, "{'a':1}",
            #"{"a":12345678901234567890}"#, #"{"a":-9223372036854775809}"#, #"{"a":1e400}"#,
            #"{"a":"\ud83d"}"#, #"{"a":"\ude00"}"#, #"{"a":"\ud83dA"}"#,
            #"{"a":01}"#, #"{"a":.5}"#, #"{"a":1.}"#, #"{"a":+1}"#, #"{"a":-}"#, #"{"a":1e}"#,
            "{\"a\":\"x\u{01}\"}", #"{"a":"\x"}"#, #"{"a":"\u00G1"}"#, #"{"a":"\u+0e9"}"#, #"{"a":"\u00e"}"#,
            #"{"a":tru}"#, #"{"a":nul}"#, deep,
        ]
        for text in refused {
            #expect(Inkling2ToolArguments.templateValue(text) == nil, "\(text)")
        }
        let deepest = String(repeating: #"{"a":"#, count: JSONValue.maximumDepth) + "1"
            + String(repeating: "}", count: JSONValue.maximumDepth)
        #expect(Inkling2ToolArguments.templateValue(deepest) != nil)
        #expect(Inkling2ToolArguments.templateValue(#"{"a":-9223372036854775808,"b":-0}"#) != nil)
        #expect(Inkling2ToolArguments.templateValue("{}") != nil)
    }

    // MARK: The tool-call parser

    /// A call the model wrote keeps its arguments' order and number kinds in
    /// the text returned to the client and kept for the history.
    @Test func aParsedCallKeepsTheModelsText() throws {
        let body = #"{"name":"query_records","args":{"query":"tax","filter":{"year":2025,"kind":"pdf"},"weight":1.0}}"#
        let call = try Inkling2ToolCallParser().parse(body, headerName: "query_records",
                                                       allowedTools: ["query_records"], id: "call_1")
        let ordered = #"{"query":"tax","filter":{"year":2025,"kind":"pdf"},"weight":1.0}"#
        #expect(call.argumentsJSON == ordered)
        #expect(call.argumentsText == ordered)
        let parsed = try JSONDecoder().decode(JSONValue.self, from: Data(ordered.utf8))
        #expect(call.arguments == parsed)
    }

    /// A flat call with sorted string and integer arguments returns the same
    /// text as before this change.
    @Test func aFlatSortedCallReturnsTheSameTextAsBefore() throws {
        let body = #"{"name":"get_weather","args":{"city":"Paris","days":3}}"#
        let call = try Inkling2ToolCallParser().parse(body, headerName: "", allowedTools: ["get_weather"], id: "c")
        let before = try call.arguments.encoded()
        #expect(call.argumentsJSON == #"{"city":"Paris","days":3}"#)
        #expect(call.argumentsJSON == before)
    }

    /// Arguments a template value cannot carry exactly keep the sorted
    /// `JSONValue` text and no history text, as before.
    @Test func anIntegerBeyondIntKeepsTheOldText() throws {
        let body = #"{"name":"f","args":{"b":12345678901234567890,"a":1}}"#
        let call = try Inkling2ToolCallParser().parse(body, headerName: "", allowedTools: ["f"], id: "c")
        #expect(call.argumentsJSON == #"{"a":1,"b":12345678901234567890}"#)
        #expect(call.argumentsText == nil)
    }

    /// Decision 3's malformed-call path is unchanged: text that is not a valid
    /// call still throws `malformed` before anything here runs.
    @Test func malformedCallsStillThrow() {
        for body in [#"{"name":"f","args":{"city":}"#, #"{"name":"f","args":"{}"}"#,
                     #"{"name":"f","args":{"a":1,"a":2}}"#, #"{"name":"f"}"#] {
            #expect(throws: ToolCallParserError.malformed) {
                _ = try Inkling2ToolCallParser().parse(body, headerName: "", allowedTools: ["f"], id: "c")
            }
        }
    }
}
