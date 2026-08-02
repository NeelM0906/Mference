import Foundation
import Testing
@testable import Mference

/// StructuredAssistantDecoder in DeepSeek mode: `<think>` suppression via the
/// fixture tokenizer's special-token IDs, and DSML tool-call buffering on the
/// plain-text delta stream.
@Suite("DeepSeek decoder")
struct DeepseekDecoderTests {
    let tok: MFTokenizer

    init() async throws {
        self.tok = try await MFTokenizer.load(from: DeepseekTemplateTests.fixtureFolder())
    }

    private func decoder(allowedTools: Set<String> = ["get_weather"]) -> StructuredAssistantDecoder {
        StructuredAssistantDecoder(tokenizer: tok,
                                   allowedTools: allowedTools,
                                   idGenerator: { "call_fixed" })
    }

    /// Feeds text through the streaming detokenizer so each token carries the
    /// same delta the generation loop would produce; DSML markers arrive
    /// split across chunks exactly as they would in production.
    private func feed(_ text: String,
                      into decoder: StructuredAssistantDecoder) throws -> [StructuredAssistantEvent] {
        var events: [StructuredAssistantEvent] = []
        var detok = MFDetokenizer(tokenizer: tok)
        for id in tok.encode(text, addBOS: false) {
            events += try decoder.consume(tokenID: id, delta: detok.push(id))
        }
        return events
    }

    private func visibleText(_ events: [StructuredAssistantEvent]) -> String {
        events.reduce(into: "") { result, event in
            if case .content(let delta) = event { result += delta }
        }
    }

    @Test("Visible text streams through unchanged")
    func plainText() throws {
        let d = decoder()
        let events = try feed("Hello there!", into: d)
        #expect(visibleText(events) == "Hello there!")
        _ = try d.finish()
        #expect(!d.hasToolCalls)
    }

    @Test("Think spans are suppressed, text after them is visible")
    func thinkSuppression() throws {
        let d = decoder()
        let events = try feed("<think>\nhidden reasoning\n</think>\n\nvisible answer", into: d)
        let text = visibleText(events)
        #expect(!text.contains("hidden reasoning"))
        #expect(text.contains("visible answer"))
        _ = try d.finish()
    }

    @Test("DSML tool-call blocks buffer and emit a parsed call")
    func toolCallBuffering() throws {
        let d = decoder()
        let events = try feed(
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n"
                + "<｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter>\n"
                + "</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
            into: d)
        #expect(events == [.toolCall(ParsedToolCall(
            id: "call_fixed",
            name: "get_weather",
            arguments: .object(["city": .string("Paris")]),
            argumentsJSON: #"{"city":"Paris"}"#))])
        #expect(d.hasToolCalls)
        _ = try d.finish()
    }

    @Test("One block with several invokes emits several calls")
    func multipleInvokesInOneBlock() throws {
        let d = decoder()
        let events = try feed(
            "<｜DSML｜tool_calls>\n"
                + "<｜DSML｜invoke name=\"get_weather\">\n"
                + "<｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter>\n"
                + "</｜DSML｜invoke>\n"
                + "<｜DSML｜invoke name=\"get_weather\">\n"
                + "<｜DSML｜parameter name=\"city\" string=\"true\">Tokyo</｜DSML｜parameter>\n"
                + "</｜DSML｜invoke>\n"
                + "</｜DSML｜tool_calls>",
            into: d)
        let calls = events.compactMap { event -> ParsedToolCall? in
            if case .toolCall(let call) = event { return call }
            return nil
        }
        #expect(calls.count == 2)
        #expect(calls.map(\.name) == ["get_weather", "get_weather"])
        #expect(calls.last?.arguments == .object(["city": .string("Tokyo")]))
        #expect(d.hasToolCalls)
        _ = try d.finish()
    }

    @Test("Preamble text before the block stays visible")
    func preambleThenToolCall() throws {
        let d = decoder()
        let events = try feed(
            "Checking the weather now.\n\n<｜DSML｜tool_calls>\n"
                + "<｜DSML｜invoke name=\"get_weather\">\n\n</｜DSML｜invoke>\n"
                + "</｜DSML｜tool_calls>",
            into: d)
        #expect(visibleText(events) == "Checking the weather now.\n\n")
        #expect(d.hasToolCalls)
        _ = try d.finish()
    }

    @Test("Thinking stream still yields the call after </think>")
    func thinkThenToolCall() throws {
        let d = decoder()
        let events = try feed(
            "<think>plan the lookup</think>\n"
                + "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n\n"
                + "</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
            into: d)
        #expect(!visibleText(events).contains("plan the lookup"))
        #expect(d.hasToolCalls)
        _ = try d.finish()
    }

    @Test("Angle brackets that never form the marker stream through")
    func lessThanIsNotSwallowed() throws {
        let d = decoder()
        let text = "a < b and tags like <｜DSML｜toolbox> are plain text."
        let events = try feed(text, into: d)
        #expect(visibleText(events) == text)
        _ = try d.finish()
        #expect(!d.hasToolCalls)
    }

    @Test("Unknown tool inside a block fails closed")
    func unknownToolFails() {
        let d = decoder(allowedTools: [])
        #expect(throws: ToolCallParserError.unknownTool("get_weather")) {
            _ = try feed(
                "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n\n"
                    + "</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
                into: d)
        }
    }

    @Test("Malformed block bodies fail closed at the close marker")
    func malformedBlockFails() {
        let d = decoder()
        #expect(throws: ToolCallParserError.malformed) {
            _ = try feed("<｜DSML｜tool_calls>\njust text\n</｜DSML｜tool_calls>", into: d)
        }
    }

    @Test("Finish with an unterminated block is malformed")
    func unterminatedBlock() throws {
        let d = decoder()
        _ = try feed("<｜DSML｜tool_calls>\n<｜DSML｜invoke", into: d)
        #expect(throws: ToolCallParserError.malformed) {
            _ = try d.finish()
        }
    }
}

extension DeepseekDecoderTests {
    @Test("Flushed tail text routes through DSML scanning in order")
    func flushedTailKeepsOrderAndScanning() throws {
        let d = decoder()
        // A delta ending in a bare `<` is withheld as a potential DSML
        // prefix; a detokenizer flush arriving afterwards must come out
        // AFTER the held text, not before it.
        var events = try feed("count: 1 <", into: d)
        events += try d.consumeFlushedText("2")
        events += d.drain()
        #expect(visibleText(events) == "count: 1 <2")
        _ = try d.finish()
    }

    @Test("drain releases a reply that legitimately ends in a DSML prefix")
    func drainReleasesHeldSuffix() throws {
        let d = decoder()
        var events = try feed("a < b, and x <｜", into: d)
        // Before drain, the trailing run that could open a DSML block is
        // withheld; drain at end of stream must release it verbatim.
        events += d.drain()
        #expect(visibleText(events) == "a < b, and x <｜")
        _ = try d.finish()
    }
}

extension DeepseekDecoderTests {
    @Test("finish releases a withheld DSML-open prefix instead of dropping it")
    func finishReleasesHeldPrefix() throws {
        let d = decoder()
        // The classic truncation case: a completion that legitimately ends
        // in a bare `<` (any proper prefix of the DSML open marker). The
        // scanner withholds it mid-stream; finish() must hand it back.
        var events = try feed("The answer is 2 <", into: d)
        events += try d.finish()
        #expect(visibleText(events) == "The answer is 2 <")
        #expect(!d.hasToolCalls)
    }

    @Test("finish still rejects an unclosed DSML block without leaking it")
    func finishRejectsUnclosedBlock() throws {
        let d = decoder()
        let events = try feed("ok <｜DSML｜tool_calls>\npartial", into: d)
        #expect(visibleText(events) == "ok ")
        #expect(throws: ToolCallParserError.self) {
            try d.finish()
        }
    }
}
