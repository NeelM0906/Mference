import Foundation
import Testing
@testable import Mference

/// StructuredAssistantDecoder in minicpm mode: `<think>` suppression keyed on
/// the non-special think ids, and `<function`…`</function>` buffering into
/// `MiniCPMToolCallParser`, driven by the fixture tokenizer's special ids.
@Suite("MiniCPM5 decoder")
struct MiniCPM5DecoderTests {
    let tok: MFTokenizer

    init() async throws {
        self.tok = try await MFTokenizer.load(from: MiniCPM5TemplateTests.fixtureFolder())
    }

    private func decoder(allowedTools: Set<String> = ["get_weather", "run_code"],
                         startsInThought: Bool = false) -> StructuredAssistantDecoder {
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

    @Test("Visible text streams through unchanged")
    func plainText() throws {
        let d = decoder()
        let events = try feed("Hello there!", into: d)
        #expect(visibleText(events) == "Hello there!")
        _ = try d.finish()
        #expect(!d.hasToolCalls)
    }

    @Test("Think spans are suppressed even though the ids are not flagged special")
    func thinkSuppression() throws {
        let d = decoder()
        let events = try feed("<think>\nhidden reasoning\n</think>\n\nvisible answer", into: d)
        let text = visibleText(events)
        #expect(!text.contains("hidden reasoning"))
        #expect(text.contains("visible answer"))
        _ = try d.finish()
    }

    @Test("A generation that starts inside the think block only shows text after </think>")
    func startsInThought() throws {
        let d = decoder(startsInThought: true)
        let events = try feed("still thinking\n</think>\n\nnow visible", into: d)
        let text = visibleText(events)
        #expect(!text.contains("still thinking"))
        #expect(text.contains("now visible"))
    }

    @Test("An XML function call buffers and emits a parsed call")
    func toolCall() throws {
        let d = decoder()
        let events = try feed(
            "Checking.\n<function name=\"get_weather\"><param name=\"city\">Paris</param><param name=\"unit\">c</param></function>",
            into: d)
        #expect(visibleText(events) == "Checking.\n")
        let parsed = calls(events)
        #expect(parsed.count == 1)
        #expect(parsed.first?.name == "get_weather")
        #expect(parsed.first?.id == "call_fixed")
        #expect(parsed.first?.arguments == .object(["city": .string("Paris"), "unit": .string("c")]))
        _ = try d.finish()
        #expect(d.hasToolCalls)
    }

    @Test("CDATA values keep <, & and newlines; bare values are typed like the template prints them")
    func cdataAndTypes() throws {
        let d = decoder()
        let events = try feed(
            "<function name=\"run_code\"><param name=\"code\"><![CDATA[print(1 < 2 & 3)\nprint('x')]]></param><param name=\"timeout\">5</param><param name=\"verbose\">True</param><param name=\"label\">plain text</param></function>",
            into: d)
        let parsed = try #require(calls(events).first)
        #expect(parsed.name == "run_code")
        #expect(parsed.arguments == .object([
            "code": .string("print(1 < 2 & 3)\nprint('x')"),
            "timeout": .integer(5),
            "verbose": .bool(true),
            "label": .string("plain text"),
        ]))
    }

    @Test("Two calls in one turn emit two events in order")
    func twoCalls() throws {
        let d = decoder()
        let events = try feed(
            "<function name=\"get_weather\"><param name=\"city\">Rome</param></function>\n<function name=\"run_code\"><param name=\"code\">2*2</param></function>",
            into: d)
        #expect(calls(events).map(\.name) == ["get_weather", "run_code"])
        #expect(visibleText(events) == "\n")
    }

    @Test("A call to a tool that was not offered is rejected")
    func unknownTool() throws {
        let d = decoder(allowedTools: ["get_weather"])
        #expect(throws: ToolCallParserError.self) {
            _ = try feed("<function name=\"run_code\"><param name=\"code\">1</param></function>", into: d)
        }
    }

    @Test("Malformed bodies fail loudly")
    func malformed() throws {
        for body in ["<function name=\"get_weather\"><param name=\"city\">Paris</function>",
                     "<function ><param name=\"city\">Paris</param></function>",
                     "<function name=\"get_weather\"><param name=\"city\">A</param><param name=\"city\">B</param></function>"] {
            let d = decoder()
            #expect(throws: ToolCallParserError.self, Comment(rawValue: body)) {
                _ = try feed(body, into: d)
            }
        }
        let unclosed = decoder()
        _ = try feed("<function name=\"get_weather\"><param name=\"city\">Paris</param>", into: unclosed)
        #expect(throws: ToolCallParserError.self) { _ = try unclosed.finish() }
    }

    @Test("The parser reads the body the decoder hands it, without the <function token")
    func parserBody() throws {
        let call = try MiniCPMToolCallParser().parse(
            " name=\"get_weather\"><param name=\"city\">Oslo</param>",
            allowedTools: ["get_weather"], id: "c1")
        #expect(call.name == "get_weather")
        #expect(call.arguments == .object(["city": .string("Oslo")]))
        #expect(MiniCPMToolCallParser.typedValue("None") == .null)
        #expect(MiniCPMToolCallParser.typedValue("[1, 2]") == .array([.integer(1), .integer(2)]))
        #expect(MiniCPMToolCallParser.typedValue("1.5") == .decimal(1.5))
        #expect(MiniCPMToolCallParser.typedValue("hello 1") == .string("hello 1"))
    }
}
