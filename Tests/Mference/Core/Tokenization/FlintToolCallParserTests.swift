import Testing
@testable import Mference

@Suite
struct FlintToolCallParserTests {
    @Test func parsesGlmXmlArguments() throws {
        let call = try FlintToolCallParser().parse(
            "search\n<arg_key>query</arg_key><arg_value>swift metal</arg_value>"
                + "<arg_key>limit</arg_key><arg_value>5</arg_value>",
            allowedTools: ["search"],
            id: "call_1")
        #expect(call.name == "search")
        #expect(call.id == "call_1")
        #expect(call.arguments == .object([
            "query": .string("swift metal"),
            "limit": .integer(5),
        ]))
        #expect(call.argumentsJSON == #"{"limit":5,"query":"swift metal"}"#)
    }

    @Test func rejectsUnknownAndDuplicateArguments() {
        #expect(throws: ToolCallParserError.self) {
            _ = try FlintToolCallParser().parse(
                "missing", allowedTools: ["search"], id: "call_1")
        }
        #expect(throws: ToolCallParserError.self) {
            _ = try FlintToolCallParser().parse(
                "search<arg_key>x</arg_key><arg_value>1</arg_value>"
                    + "<arg_key>x</arg_key><arg_value>2</arg_value>",
                allowedTools: ["search"], id: "call_1")
        }
    }
}
