import Foundation
import Testing
@testable import Mference

/// StructuredAssistantDecoder in `inkling2` mode, driven by the fixture
/// tokenizer's real special ids: thinking blocks go to the reasoning channel,
/// text blocks are visible, `<|message_model|>NAME<|content_invoke_tool_json|>`
/// blocks parse into tool calls, header text never leaks.
@Suite("Inkling 2 decoder")
struct Inkling2DecoderTests {
    let tok: MFTokenizer

    init() async throws {
        self.tok = try await MFTokenizer.load(from: Inkling2ChatTests.fixtureFolder())
    }

    private let mdl = "<|message_model|>", txt = "<|content_text|>", thk = "<|content_thinking|>"
    private let end = "<|end_message|>", eos = "<|content_model_end_sampling|>"
    private let invoke = "<|content_invoke_tool_json|>"

    private func decoder(allowedTools: Set<String> = ["get_weather"]) -> StructuredAssistantDecoder {
        StructuredAssistantDecoder(tokenizer: tok, allowedTools: allowedTools,
                                   idGenerator: { "call_fixed" })
    }

    private func feed(_ text: String, into decoder: StructuredAssistantDecoder,
                      reasoning: inout String) throws -> [StructuredAssistantEvent] {
        var collected = ""
        decoder.onReasoning = { collected += $0 }
        var events: [StructuredAssistantEvent] = []
        var detok = MFDetokenizer(tokenizer: tok)
        for id in tok.encode(text, addBOS: false) {
            events += try decoder.consume(tokenID: id, delta: detok.push(id))
        }
        events += try decoder.consumeFlushedText(detok.flush())
        reasoning = collected
        return events
    }

    private func visible(_ events: [StructuredAssistantEvent]) -> String {
        events.reduce(into: "") { if case .content(let d) = $1 { $0 += d } }
    }

    private func calls(_ events: [StructuredAssistantEvent]) -> [ParsedToolCall] {
        events.compactMap { if case .toolCall(let c) = $0 { c } else { nil } }
    }

    @Test func thinkingThenAnswer() throws {
        let d = decoder()
        var reasoning = ""
        let events = try feed(mdl + thk + "Let me think." + end + mdl + txt + "Hello there!" + end + eos,
                              into: d, reasoning: &reasoning)
        #expect(visible(events) == "Hello there!")
        #expect(reasoning == "Let me think.")
        _ = try d.finish()
        #expect(!d.hasToolCalls)
        let counts = try #require(d.payloadTokenCounts)
        #expect(counts.reasoning > 0 && counts.visible > 0)
    }

    @Test func toolCallParsesAndTheHeaderNameStaysHidden() throws {
        let d = decoder()
        var reasoning = ""
        let events = try feed(mdl + thk + "Need weather." + end
                              + mdl + "get_weather" + invoke + #"{"name":"get_weather","args":{"city":"Paris"}}"# + end + eos,
                              into: d, reasoning: &reasoning)
        #expect(visible(events).isEmpty)
        let call = try #require(calls(events).first)
        #expect(call.name == "get_weather" && call.id == "call_fixed")
        #expect(call.arguments == .object(["city": .string("Paris")]))
        #expect(call.argumentsJSON == #"{"city":"Paris"}"#)
        _ = try d.finish()
        #expect(d.hasToolCalls)
    }

    @Test func textThenTwoCalls() throws {
        let d = decoder(allowedTools: ["get_weather", "get_time"])
        var reasoning = ""
        let events = try feed(mdl + txt + "Checking." + end
                              + mdl + "get_weather" + invoke + #"{"name":"get_weather","args":{}}"# + end
                              + mdl + "get_time" + invoke + #"{"name":"get_time","args":{"tz":"UTC"}}"# + end + eos,
                              into: d, reasoning: &reasoning)
        #expect(visible(events) == "Checking.")
        #expect(calls(events).map(\.name) == ["get_weather", "get_time"])
    }

    @Test func unknownToolIsRejected() {
        let d = decoder()
        var reasoning = ""
        #expect(throws: ToolCallParserError.unknownTool("rm_rf")) {
            _ = try feed(mdl + "rm_rf" + invoke + #"{"name":"rm_rf","args":{}}"# + end,
                         into: d, reasoning: &reasoning)
        }
    }

    @Test func headerAndBodyMustNameTheSameTool() {
        let d = decoder(allowedTools: ["get_weather", "get_time"])
        var reasoning = ""
        #expect(throws: ToolCallParserError.malformed) {
            _ = try feed(mdl + "get_time" + invoke + #"{"name":"get_weather","args":{}}"# + end,
                         into: d, reasoning: &reasoning)
        }
    }

    @Test func unclosedToolCallFailsAtFinish() throws {
        let d = decoder()
        var reasoning = ""
        _ = try feed(mdl + "get_weather" + invoke + #"{"name":"get_weather""#, into: d, reasoning: &reasoning)
        #expect(throws: ToolCallParserError.malformed) { _ = try d.finish() }
    }

    @Test func textOutsideABlockIsShown() throws {
        let d = decoder()
        var reasoning = ""
        let events = try feed("plain", into: d, reasoning: &reasoning)
        #expect(visible(events) == "plain")
    }

    @Test func toolErrorAndTextInvokeBlocksAreVisible() throws {
        let d = decoder()
        var reasoning = ""
        let events = try feed(mdl + "<|content_tool_error|>" + "bad input" + end
                              + mdl + "<|content_invoke_tool_text|>" + " retry" + end + eos,
                              into: d, reasoning: &reasoning)
        #expect(visible(events) == "bad input retry")
    }

    @Test func headerTextBeforeATextBlockStaysHidden() throws {
        let d = decoder()
        var reasoning = ""
        let events = try feed(mdl + "assistant" + txt + "Hi" + end, into: d, reasoning: &reasoning)
        #expect(visible(events) == "Hi")
    }

    @Test func eosInsideABlockEndsTheVisibleText() throws {
        let d = decoder()
        var reasoning = ""
        let events = try feed(mdl + txt + "partial" + eos, into: d, reasoning: &reasoning)
        #expect(visible(events) == "partial")
        _ = try d.finish()
    }

    @Test func aHeaderNamesOnlyTheBlockItOpens() throws {
        // The second call has no header of its own; the first call's name
        // must not carry over and reject it.
        let d = decoder(allowedTools: ["get_weather", "get_time"])
        var reasoning = ""
        let events = try feed(mdl + "get_weather" + invoke + #"{"name":"get_weather","args":{}}"# + end
                              + invoke + #"{"name":"get_time","args":{}}"# + end,
                              into: d, reasoning: &reasoning)
        #expect(calls(events).map(\.name) == ["get_weather", "get_time"])
    }

    @Test func anOversizedCallFails() throws {
        // Token ids straight into the decoder: past the cap, the buffered call fails.
        let d = decoder()
        let t = try #require(tok.inkling2Tokens)
        _ = try d.consume(tokenID: t.messageModel, delta: "")
        _ = try d.consume(tokenID: t.contentInvokeToolJSON, delta: "")
        let a = try #require(tok.encode("a", addBOS: false).first)
        let cap = Inkling2ToolCallParser.maximumBytes / MemoryLayout<Int32>.size
        for _ in 0..<cap { _ = try d.consume(tokenID: a, delta: "a") }
        #expect(throws: ToolCallParserError.oversized) { _ = try d.consume(tokenID: a, delta: "a") }
    }

    @Test func parserRejectsMalformedBodies() {
        let p = Inkling2ToolCallParser()
        for body in [#"{"name":"get_weather"}"#, #"{"name":"get_weather","args":[]}"#,
                     #"{"name":"get_weather","args":{},"x":1}"#,
                     #"{"name":"get_weather","args":{"a":1,"a":2}}"#, "not json"] {
            #expect(throws: ToolCallParserError.malformed, "\(body)") {
                _ = try p.parse(body, headerName: "", allowedTools: ["get_weather"], id: "c")
            }
        }
        #expect(throws: ToolCallParserError.oversized) {
            _ = try p.parse(String(repeating: " ", count: Inkling2ToolCallParser.maximumBytes + 1),
                            headerName: "", allowedTools: ["get_weather"], id: "c")
        }
    }
}
