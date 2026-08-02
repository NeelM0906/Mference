import Foundation
import Testing
@testable import Mference

@Suite("DeepSeek tool call parser")
struct DeepseekToolCallParserTests {
    private let parser = DeepseekToolCallParser()
    private let tools: Set<String> = ["get_weather", "run_query", "no_args"]

    private func parse(_ body: String,
                       allowedTools: Set<String>? = nil) throws -> [ParsedToolCall] {
        var counter = 0
        return try parser.parse(body, allowedTools: allowedTools ?? tools) {
            counter += 1
            return "call_\(counter)"
        }
    }

    @Test("Happy path with a single string parameter")
    func singleStringParameter() throws {
        let calls = try parse("""

        <｜DSML｜invoke name="get_weather">
        <｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>
        </｜DSML｜invoke>

        """)
        #expect(calls.count == 1)
        #expect(calls[0].name == "get_weather")
        #expect(calls[0].arguments == .object(["city": .string("Paris")]))
        #expect(calls[0].argumentsJSON == #"{"city":"Paris"}"#)
        #expect(calls[0].id == "call_1")
    }

    @Test("string=\"false\" values decode as JSON, string=\"true\" stays raw")
    func jsonAndStringValues() throws {
        let calls = try parse("""

        <｜DSML｜invoke name="run_query">
        <｜DSML｜parameter name="limit" string="false">25</｜DSML｜parameter>
        <｜DSML｜parameter name="filters" string="false">{"active": true, "tags": ["a", "b"]}</｜DSML｜parameter>
        <｜DSML｜parameter name="ratio" string="false">0.5</｜DSML｜parameter>
        <｜DSML｜parameter name="verbose" string="false">true</｜DSML｜parameter>
        <｜DSML｜parameter name="cursor" string="false">null</｜DSML｜parameter>
        <｜DSML｜parameter name="query" string="true">SELECT * FROM t</｜DSML｜parameter>
        <｜DSML｜parameter name="count" string="true">25</｜DSML｜parameter>
        </｜DSML｜invoke>

        """)
        #expect(calls.count == 1)
        #expect(calls[0].arguments == .object([
            "limit": .integer(25),
            "filters": .object([
                "active": .bool(true),
                "tags": .array([.string("a"), .string("b")]),
            ]),
            "ratio": .decimal(Decimal(string: "0.5", locale: Locale(identifier: "en_US_POSIX"))!),
            "verbose": .bool(true),
            "cursor": .null,
            "query": .string("SELECT * FROM t"),
            "count": .string("25"),
        ]))
    }

    @Test("Multi-line string values are preserved verbatim")
    func multiLineValue() throws {
        let calls = try parse("""

        <｜DSML｜invoke name="run_query">
        <｜DSML｜parameter name="script" string="true">line one
          line two
        line three</｜DSML｜parameter>
        </｜DSML｜invoke>

        """)
        #expect(calls[0].arguments == .object([
            "script": .string("line one\n  line two\nline three"),
        ]))
    }

    @Test("Quoted string values keep their literal quotes when string=\"true\"")
    func quotedStringStaysRaw() throws {
        let calls = try parse("""

        <｜DSML｜invoke name="run_query">
        <｜DSML｜parameter name="q" string="true">"hello"</｜DSML｜parameter>
        </｜DSML｜invoke>

        """)
        #expect(calls[0].arguments == .object(["q": .string("\"hello\"")]))
    }

    @Test("Multiple invokes each produce a call, in block order")
    func multipleInvokes() throws {
        let calls = try parse("""

        <｜DSML｜invoke name="get_weather">
        <｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>
        </｜DSML｜invoke>
        <｜DSML｜invoke name="run_query">
        <｜DSML｜parameter name="limit" string="false">5</｜DSML｜parameter>
        </｜DSML｜invoke>

        """)
        #expect(calls.map(\.name) == ["get_weather", "run_query"])
        #expect(calls.map(\.id) == ["call_1", "call_2"])
        #expect(calls[1].arguments == .object(["limit": .integer(5)]))
    }

    @Test("Zero-parameter invoke parses to an empty object")
    func noParameters() throws {
        let calls = try parse("\n<｜DSML｜invoke name=\"no_args\">\n\n</｜DSML｜invoke>\n")
        #expect(calls.count == 1)
        #expect(calls[0].name == "no_args")
        #expect(calls[0].arguments == .object([:]))
        #expect(calls[0].argumentsJSON == "{}")
    }

    @Test("Empty string parameter value parses to an empty string")
    func emptyValue() throws {
        let calls = try parse("""

        <｜DSML｜invoke name="run_query">
        <｜DSML｜parameter name="q" string="true"></｜DSML｜parameter>
        </｜DSML｜invoke>

        """)
        #expect(calls[0].arguments == .object(["q": .string("")]))
    }

    @Test("Unknown tools fail closed")
    func unknownTool() {
        #expect(throws: ToolCallParserError.unknownTool("secret_tool")) {
            _ = try parse("\n<｜DSML｜invoke name=\"secret_tool\">\n\n</｜DSML｜invoke>\n")
        }
    }

    @Test("Invalid function names are malformed even if allowed", arguments: [
        "a b", "café", "x/y", "", String(repeating: "a", count: 65),
    ])
    func invalidFunctionName(_ name: String) {
        #expect(throws: ToolCallParserError.malformed) {
            _ = try parse("\n<｜DSML｜invoke name=\"\(name)\">\n\n</｜DSML｜invoke>\n",
                          allowedTools: [name])
        }
    }

    @Test("Malformed bodies are rejected", arguments: [
        "just text",
        "<｜DSML｜invoke name=\"get_weather\">",
        "\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"city\" string=\"true\">Paris\n</｜DSML｜invoke>\n",
        "\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"\" string=\"true\">x</｜DSML｜parameter>\n</｜DSML｜invoke>\n",
        "\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"city\">Paris</｜DSML｜parameter>\n</｜DSML｜invoke>\n",
        "\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"city\" string=\"yes\">Paris</｜DSML｜parameter>\n</｜DSML｜invoke>\n",
        "\n<｜DSML｜invoke name=\"get_weather\">\n\n</｜DSML｜invoke>\ntrailing junk",
        "",
    ])
    func malformedBodies(_ body: String) {
        #expect(throws: ToolCallParserError.malformed) {
            _ = try parse(body)
        }
    }

    @Test("string=\"false\" with non-JSON value is malformed")
    func nonJSONTypedValue() {
        #expect(throws: ToolCallParserError.malformed) {
            _ = try parse("""

            <｜DSML｜invoke name="run_query">
            <｜DSML｜parameter name="limit" string="false">not json</｜DSML｜parameter>
            </｜DSML｜invoke>

            """)
        }
    }

    @Test("Oversized bodies are rejected before parsing")
    func oversized() {
        let body = "\n<｜DSML｜invoke name=\"run_query\">\n<｜DSML｜parameter name=\"q\" string=\"true\">"
            + String(repeating: "x", count: DeepseekToolCallParser.maximumBytes)
            + "</｜DSML｜parameter>\n</｜DSML｜invoke>\n"
        #expect(throws: ToolCallParserError.oversized) {
            _ = try parse(body)
        }
    }

    @Test("Duplicate parameter keys keep the last value")
    func duplicateKeysLastWins() throws {
        let calls = try parse("""

        <｜DSML｜invoke name="run_query">
        <｜DSML｜parameter name="q" string="true">first</｜DSML｜parameter>
        <｜DSML｜parameter name="q" string="true">second</｜DSML｜parameter>
        </｜DSML｜invoke>

        """)
        #expect(calls[0].arguments == .object(["q": .string("second")]))
    }
}
