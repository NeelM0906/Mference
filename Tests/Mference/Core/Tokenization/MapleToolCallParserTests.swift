import Foundation
import Testing
@testable import Mference

@Suite("Maple tool call parser")
struct MapleToolCallParserTests {
    private let parser = MapleToolCallParser()

    private func parse(_ body: String,
                       allowedTools: Set<String> = ["lookup"]) throws -> ParsedToolCall {
        try parser.parse(body, allowedTools: allowedTools, id: "call_fixed")
    }

    @Test("Valid JSON envelope preserves typed arguments")
    func validEnvelope() throws {
        let call = try parse(
            #"{"name":"lookup","arguments":{"city":"Paris","count":2,"nested":{"ok":true}}}"#)
        #expect(call == ParsedToolCall(
            id: "call_fixed",
            name: "lookup",
            arguments: .object([
                "city": .string("Paris"),
                "count": .integer(2),
                "nested": .object(["ok": .bool(true)]),
            ]),
            argumentsJSON: #"{"city":"Paris","count":2,"nested":{"ok":true}}"#))
    }

    @Test("Unknown tools fail closed")
    func unknownTool() {
        #expect(throws: ToolCallParserError.unknownTool("other")) {
            _ = try parse(#"{"name":"other","arguments":{}}"#)
        }
    }

    @Test("Malformed envelopes fail closed", arguments: [
        "",
        "not json",
        "[]",
        #"{"name":"lookup"}"#,
        #"{"arguments":{}}"#,
        #"{"name":7,"arguments":{}}"#,
    ])
    func malformedEnvelope(_ body: String) {
        #expect(throws: ToolCallParserError.malformed) {
            _ = try parse(body)
        }
    }

    @Test("Extra envelope fields are rejected")
    func extraEnvelopeField() {
        #expect(throws: ToolCallParserError.malformed) {
            _ = try parse(#"{"name":"lookup","arguments":{},"extra":true}"#)
        }
    }

    @Test("Arguments must be an object", arguments: [
        #"[]"#,
        #""text""#,
        #"null"#,
        #"1"#,
    ])
    func nonObjectArguments(_ arguments: String) {
        #expect(throws: ToolCallParserError.malformed) {
            _ = try parse(#"{"name":"lookup","arguments":\#(arguments)}"#)
        }
    }

    @Test("JSON nesting past the shared limit is malformed")
    func excessiveDepth() {
        let levels = JSONValue.maximumDepth + 8
        let nested = String(repeating: "[", count: levels)
            + "0"
            + String(repeating: "]", count: levels)
        #expect(throws: ToolCallParserError.malformed) {
            _ = try parse(#"{"name":"lookup","arguments":{"value":\#(nested)}}"#)
        }
    }

    @Test("Oversized bodies are rejected before parsing")
    func oversized() {
        #expect(throws: ToolCallParserError.oversized) {
            _ = try parse(String(
                repeating: "x",
                count: MapleToolCallParser.maximumBytes + 1))
        }
    }

    @Test("Duplicate keys are rejected at every object level", arguments: [
        #"{"name":"lookup","name":"lookup","arguments":{}}"#,
        #"{"name":"lookup","arguments":{"items":[{"x":1,"x":2}]}}"#,
        #"{"name":"lookup","arguments":{"a":1,"\u0061":2}}"#,
    ])
    func duplicateKeys(_ body: String) {
        #expect(throws: ToolCallParserError.malformed) {
            _ = try parse(body)
        }
    }
}
