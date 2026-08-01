import Foundation
import Testing
@testable import Mference

@Suite("Gemma tool call parser depth")
struct GemmaToolCallParserDepthTests {
    private let parser = GemmaToolCallParser()
    private let tools: Set<String> = ["f"]

    private func parse(_ body: String) throws -> ParsedToolCall {
        try parser.parse(body, allowedTools: tools, id: "call_test")
    }

    /// `call:f{a:[[[…]]]}` with `levels` nested arrays, so the innermost array
    /// sits `levels` deep relative to the argument object.
    private func nestedArrays(levels: Int) -> String {
        "call:f{a:" + String(repeating: "[", count: levels)
            + String(repeating: "]", count: levels) + "}"
    }

    /// `call:f{a:{a:{…{}}}}` with `levels` nested objects.
    private func nestedObjects(levels: Int) -> String {
        "call:f{a:" + String(repeating: "{a:", count: levels - 1) + "{}"
            + String(repeating: "}", count: levels - 1) + "}"
    }

    @Test("Nested arrays past the limit are malformed")
    func nestedArraysBeyondTheLimitAreMalformed() {
        // Deliberately only just past the limit: output deep enough to actually
        // overflow would take the test runner down with it, uncatchably.
        #expect(throws: ToolCallParserError.malformed) {
            _ = try parse(nestedArrays(levels: JSONValue.maximumDepth + 8))
        }
    }

    @Test("Nested objects past the limit are malformed")
    func nestedObjectsBeyondTheLimitAreMalformed() {
        #expect(throws: ToolCallParserError.malformed) {
            _ = try parse(nestedObjects(levels: JSONValue.maximumDepth + 8))
        }
    }

    @Test("Nesting exactly at the limit still parses")
    func nestingAtTheLimitStillParses() throws {
        let call = try parse(nestedArrays(levels: JSONValue.maximumDepth))
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

    @Test("The limit matches the one the Codable path enforces")
    func theLimitMatchesTheDecodeBoundary() throws {
        // Both entry points build the same recursive tree, so a value the
        // decoder accepts must not be one the Gemma parser rejects.
        let fragment = String(repeating: "[", count: JSONValue.maximumDepth)
            + String(repeating: "]", count: JSONValue.maximumDepth)
        _ = try JSONDecoder().decode(JSONValue.self, from: Data(fragment.utf8))
        _ = try parse("call:f{a:\(fragment)}")
    }

    @Test("Realistic nesting is unaffected")
    func realisticNestingParsesUnchanged() throws {
        let call = try parse(
            #"call:f{query:<|"|>rain<|"|>,filters:{tags:["a","b"],page:{n:2}}}"#)
        #expect(call.arguments == .object([
            "query": .string("rain"),
            "filters": .object([
                "tags": .array([.string("a"), .string("b")]),
                "page": .object(["n": .integer(2)]),
            ]),
        ]))
        #expect(call.argumentsJSON
            == #"{"filters":{"page":{"n":2},"tags":["a","b"]},"query":"rain"}"#)
    }
}
