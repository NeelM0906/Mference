import Foundation
import Testing
@testable import Mference

/// StructuredAssistantDecoder in glm5 mode: `<think>` suppression keyed on the
/// non-special think ids, and `<tool_call>`…`</tool_call>` buffering into
/// `Glm5ToolCallParser`, driven by the fixture tokenizer's real ids.
@Suite("GLM-5 decoder")
struct Glm5DecoderTests {
    let tok: MFTokenizer

    init() async throws {
        self.tok = try await MFTokenizer.load(from: Glm5TemplateTests.fixtureFolder())
    }

    private func decoder(allowedTools: Set<String> = ["get_weather", "run_code"],
                         startsInThought: Bool = true) -> StructuredAssistantDecoder {
        StructuredAssistantDecoder(tokenizer: tok,
                                   allowedTools: allowedTools,
                                   startsInThought: startsInThought,
                                   idGenerator: { "call_fixed" })
    }

    private func feed(_ text: String,
                      into decoder: StructuredAssistantDecoder) throws -> [StructuredAssistantEvent] {
        var events: [StructuredAssistantEvent] = []
        var detok = MFDetokenizer(tokenizer: tok)
        for id in tok.encode(text, addBOS: false) {
            events += try decoder.consume(tokenID: id, delta: detok.push(id))
        }
        events += try decoder.consumeFlushedText(detok.flush())
        return events
    }

    private func visibleText(_ events: [StructuredAssistantEvent]) -> String {
        events.reduce(into: "") { result, event in
            if case .content(let delta) = event { result += delta }
        }
    }

    private func calls(_ events: [StructuredAssistantEvent]) -> [ParsedToolCall] {
        events.compactMap { if case .toolCall(let call) = $0 { call } else { nil } }
    }

    @Test("A reply that starts inside the think block shows only what follows </think>")
    func startsInThought() throws {
        let d = decoder()
        let events = try feed("weighing options</think>The answer.", into: d)
        #expect(visibleText(events) == "The answer.")
        _ = try d.finish()
        #expect(!d.hasToolCalls)
    }

    @Test("Visible text streams through unchanged once the think block is closed")
    func plainText() throws {
        let d = decoder(startsInThought: false)
        let events = try feed("Hello there!", into: d)
        #expect(visibleText(events) == "Hello there!")
    }

    @Test("A later think span is suppressed too")
    func thinkSuppression() throws {
        let d = decoder(startsInThought: false)
        let events = try feed("a<think>hidden</think>b", into: d)
        #expect(visibleText(events) == "ab")
    }

    @Test("A tool call buffers and emits a parsed call with typed and raw values")
    func toolCall() throws {
        let d = decoder()
        let events = try feed(
            "</think>Checking.<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>"
            + "<arg_key>unit</arg_key><arg_value>c</arg_value></tool_call>", into: d)
        #expect(visibleText(events) == "Checking.")
        let parsed = calls(events)
        #expect(parsed.count == 1)
        #expect(parsed.first?.name == "get_weather")
        #expect(parsed.first?.arguments == .object(["city": .string("Paris"), "unit": .string("c")]))
        #expect(parsed.first?.id == "call_fixed")
        _ = try d.finish()
        #expect(d.hasToolCalls)
    }

    @Test("Two calls back to back, JSON values typed")
    func twoCalls() throws {
        let d = decoder()
        let events = try feed(
            "</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>"
            + "<tool_call>run_code<arg_key>code</arg_key><arg_value>1+1</arg_value>"
            + "<arg_key>timeout</arg_key><arg_value>5</arg_value>"
            + "<arg_key>opts</arg_key><arg_value>{\"f\": true, \"k\": [1, 2.5, null]}</arg_value></tool_call>",
            into: d)
        let parsed = calls(events)
        #expect(parsed.map(\.name) == ["get_weather", "run_code"])
        #expect(parsed[1].arguments == .object([
            "code": .string("1+1"), "timeout": .integer(5),
            "opts": .object(["f": .bool(true), "k": .array([.integer(1), .decimal(2.5), .null])]),
        ]))
        #expect(visibleText(events).isEmpty)
    }

    @Test("Unknown tools and malformed bodies fail the stream")
    func failures() throws {
        let unknown = decoder(allowedTools: ["get_weather"])
        #expect(throws: ToolCallParserError.self) {
            _ = try feed("</think><tool_call>run_code<arg_key>code</arg_key><arg_value>x</arg_value></tool_call>",
                         into: unknown)
        }
        let malformed = decoder()
        #expect(throws: ToolCallParserError.self) {
            _ = try feed("</think><tool_call>get_weather<arg_key>city</arg_key>Paris</tool_call>", into: malformed)
        }
        let duplicate = decoder()
        #expect(throws: ToolCallParserError.self) {
            _ = try feed("</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>a</arg_value>"
                         + "<arg_key>city</arg_key><arg_value>b</arg_value></tool_call>", into: duplicate)
        }
        let unclosed = decoder()
        _ = try feed("</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>a</arg_value>", into: unclosed)
        #expect(throws: ToolCallParserError.self) { _ = try unclosed.finish() }
    }

    @Test("A stray <tool_response> in a reply is malformed")
    func toolResponseInReply() throws {
        let d = decoder()
        #expect(throws: ToolCallParserError.self) {
            _ = try feed("</think><tool_response>x</tool_response>", into: d)
        }
    }

    @Test("Parser: name is trimmed, values keep inner whitespace, non-JSON stays a string")
    func parserDetails() throws {
        let call = try Glm5ToolCallParser().parse(
            " run_code <arg_key>code</arg_key><arg_value> print(1)\n</arg_value><arg_key>n</arg_key><arg_value>-3</arg_value>",
            allowedTools: ["run_code"], id: "x")
        #expect(call.name == "run_code")
        #expect(call.arguments == .object(["code": .string(" print(1)\n"), "n": .integer(-3)]))
        let bare = try Glm5ToolCallParser().parse("get_weather", allowedTools: ["get_weather"], id: "y")
        #expect(bare.arguments == .object([:]))
    }
}
