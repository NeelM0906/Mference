import Foundation
import Testing
@testable import Mference

@Suite("Qwen tool call parser depth")
struct QwenToolCallParserDepthTests {
    private let parser = QwenToolCallParser()
    private let tools: Set<String> = ["f"]

    private func parse(value: String) throws -> ParsedToolCall {
        try parser.parse("""

        <function=f>
        <parameter=a>
        \(value)
        </parameter>
        </function>

        """, allowedTools: tools, id: "call_test")
    }

    /// `[[[…]]]` with `levels` nested arrays.
    private func nestedArrays(levels: Int) -> String {
        String(repeating: "[", count: levels)
            + String(repeating: "]", count: levels)
    }

    /// `{"a":{"a":{…{}}}}` with `levels` nested objects.
    private func nestedObjects(levels: Int) -> String {
        String(repeating: #"{"a":"#, count: levels - 1) + "{}"
            + String(repeating: "}", count: levels - 1)
    }

    @Test("Nested arrays past the limit are malformed")
    func nestedArraysBeyondTheLimitAreMalformed() {
        // Deliberately only just past the limit: output deep enough to actually
        // overflow would take the test runner down with it, uncatchably.
        #expect(throws: ToolCallParserError.malformed) {
            _ = try parse(value: nestedArrays(levels: JSONValue.maximumDepth + 8))
        }
    }

    @Test("Nested objects past the limit are malformed")
    func nestedObjectsBeyondTheLimitAreMalformed() {
        #expect(throws: ToolCallParserError.malformed) {
            _ = try parse(value: nestedObjects(levels: JSONValue.maximumDepth + 8))
        }
    }

    @Test("Nesting exactly at the limit still parses typed")
    func nestingAtTheLimitStillParses() throws {
        let call = try parse(value: nestedArrays(levels: JSONValue.maximumDepth))
        var cursor = try #require(call.arguments.objectValue?["a"])
        for _ in 1..<JSONValue.maximumDepth {
            guard case .array(let members) = cursor else {
                Issue.record("expected an array at every level")
                return
            }
            cursor = try #require(members.first)
        }
        #expect(cursor == .array([]))
    }

    @Test("Deeply bracketed non-JSON values keep the raw-string fallback")
    func deeplyBracketedNonJSONStaysString() throws {
        // Unquoted string arguments may legitimately start with a bracket and
        // nest deeply — e.g. code or minified JSON passed as string content.
        // Only values that really are structural JSON are depth-rejected.
        let soup = String(repeating: "[", count: JSONValue.maximumDepth * 2)
            + "not json"
        let call = try parse(value: soup)
        #expect(call.arguments == .object(["a": .string(soup)]))
    }

    @Test("Brackets inside string literals do not count toward depth")
    func bracketsInsideStringsDoNotNest() throws {
        let brackets = String(repeating: "[", count: JSONValue.maximumDepth * 2)
        let call = try parse(value: #"{"a": "\#(brackets)", "b": "\""}"#)
        #expect(call.arguments == .object([
            "a": .object([
                "a": .string(brackets),
                "b": .string("\""),
            ]),
        ]))
    }

    @Test("Realistic nesting is unaffected")
    func realisticNestingParsesUnchanged() throws {
        let call = try parse(value: #"{"tags": ["a", "b"], "page": {"n": 2}}"#)
        #expect(call.arguments == .object([
            "a": .object([
                "tags": .array([.string("a"), .string("b")]),
                "page": .object(["n": .integer(2)]),
            ]),
        ]))
    }
}
