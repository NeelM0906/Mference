import Foundation
import Testing
@testable import Mference

/// StructuredAssistantDecoder in flint mode: ` thinking` suppression and
/// `<tool_call>` buffering driven by the fixture tokenizer's special ids.
@Suite("Flint decoder")
struct FlintDecoderTests {
    let tok: MFTokenizer

    init() async throws {
        self.tok = try await MFTokenizer.load(from: Self.fixtureFolder())
    }

    static func fixtureFolder() throws -> URL {
        try #require(Bundle.module.url(
            forResource: "FlintTokenizer",
            withExtension: nil,
            subdirectory: "Fixtures"))
    }

    private func decoder(allowedTools: Set<String> = ["search"]) -> StructuredAssistantDecoder {
        StructuredAssistantDecoder(tokenizer: tok,
                                   allowedTools: allowedTools,
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

    @Test func fixtureResolvesToTheFlintDialect() {
        #expect(tok.dialect == .flint)
        #expect(tok.thinkStartID != nil)
        #expect(tok.toolCallStartID != nil)
    }

    @Test func visibleTextStreamsThroughUnchanged() throws {
        let events = try feed("Hello there!", into: decoder())
        #expect(visibleText(events) == "Hello there!")
        #expect(calls(events).isEmpty)
    }

    /// Reasoning may quote tool syntax. Only the visible channel issues a
    /// call; markers inside `thinking` must not parse or fail the stream.
    @Test func toolSyntaxInsideThinkingIsNotACall() throws {
        let d = decoder()
        var reasoning = ""
        d.onReasoning = { reasoning += $0 }
        var events = try d.consume(tokenID: #require(tok.thinkStartID), delta: "")
        events += try feed("I might call ", into: d)
        events += try d.consume(tokenID: #require(tok.toolCallStartID), delta: "")
        events += try feed("search", into: d)
        events += try d.consume(tokenID: #require(tok.toolCallEndID), delta: "")
        events += try d.consume(tokenID: #require(tok.thinkEndID), delta: "")
        events += try feed("answer", into: d)
        #expect(calls(events).isEmpty, "a call quoted in reasoning must not fire")
        #expect(visibleText(events) == "answer")
        #expect(reasoning.contains("search"))
        #expect(try d.finish().isEmpty)
    }

    @Test func unknownToolInsideThinkingDoesNotFail() throws {
        let d = decoder(allowedTools: [])
        var events = try d.consume(tokenID: #require(tok.thinkStartID), delta: "")
        events += try d.consume(tokenID: #require(tok.toolCallStartID), delta: "")
        events += try feed("search", into: d)
        events += try d.consume(tokenID: #require(tok.toolCallEndID), delta: "")
        events += try d.consume(tokenID: #require(tok.thinkEndID), delta: "")
        events += try feed("ok", into: d)
        #expect(calls(events).isEmpty)
        #expect(visibleText(events) == "ok")
    }

    @Test func visibleToolCallStillParses() throws {
        let d = decoder()
        let events = try feed(
            "done<tool_call>search<arg_key>query</arg_key>"
                + "<arg_value>swift</arg_value></tool_call>",
            into: d)
        let parsed = calls(events)
        #expect(parsed.count == 1)
        #expect(parsed.first?.name == "search")
        #expect(parsed.first?.arguments == .object(["query": .string("swift")]))
    }

}
