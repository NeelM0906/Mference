import Foundation
import Testing
@testable import Mference

/// `StructuredAssistantDecoder.inkling2MalformedToolCallsAsText`, the
/// servers' choice for Inkling 2: a rejected or unclosed JSON tool-call block
/// becomes its body as visible text. Off (the default), the decoder throws as
/// before; on, other dialects are unaffected.
@Suite("Inkling 2 malformed tool calls as text")
struct Inkling2MalformedToolCallTextTests {
    let tok: MFTokenizer
    init() async throws { tok = try await MFTokenizer.load(from: Inkling2ChatTests.fixtureFolder()) }

    private let mdl = "<|message_model|>", end = "<|end_message|>", eos = "<|content_model_end_sampling|>"
    private let invoke = "<|content_invoke_tool_json|>", txt = "<|content_text|>"

    private func run(_ text: String, asText: Bool) throws -> (events: [StructuredAssistantEvent], finish: [StructuredAssistantEvent]) {
        let decoder = StructuredAssistantDecoder(tokenizer: tok, allowedTools: ["get_weather"],
                                                 idGenerator: { "call_fixed" })
        decoder.inkling2MalformedToolCallsAsText = asText
        var detok = MFDetokenizer(tokenizer: tok)
        var events: [StructuredAssistantEvent] = []
        for id in tok.encode(text, addBOS: false) {
            events += try decoder.consume(tokenID: id, delta: detok.push(id))
        }
        events += try decoder.consumeFlushedText(detok.flush())
        return (events, try decoder.finish())
    }

    private func visible(_ events: [StructuredAssistantEvent]) -> String {
        events.reduce(into: "") { if case .content(let d) = $1 { $0 += d } }
    }

    private func calls(_ events: [StructuredAssistantEvent]) -> Int {
        events.filter { if case .toolCall = $0 { true } else { false } }.count
    }

    static let malformed: [(String, String)] = [
        ("get_weather", #"{"name":"get_weather","args":{"city":}"#),
        ("send_email", #"{"name":"send_email","args":{"to":"a@b.c"}}"#),
        ("get_weather", #"{"name":"get_time","args":{"city":"Paris"}}"#),
    ]

    @Test func aRejectedBlockIsItsBodyAsText() throws {
        for (header, body) in Self.malformed {
            let text = mdl + header + invoke + body + end + eos
            let (events, finish) = try run(text, asText: true)
            #expect(visible(events + finish) == body)
            #expect(calls(events) == 0)
            // Off: the decoder fails, as before.
            #expect(throws: (any Error).self) { _ = try run(text, asText: false) }
        }
    }

    @Test func aBlockOpenAtTheEndIsItsBodyAsText() throws {
        let body = #"{"name":"get_weather","args":{"city":"Paris"}}"#
        // As production ends: EOS is never passed to the decoder.
        let text = mdl + "get_weather" + invoke + body
        let (events, finish) = try run(text, asText: true)
        #expect(visible(events + finish) == body)
        #expect(throws: (any Error).self) { _ = try run(text, asText: false) }
        // And with EOS fed (defensive): stripped from the text.
        let (withEOS, finishEOS) = try run(text + eos, asText: true)
        #expect(visible(withEOS + finishEOS) == body)
    }

    @Test func aConvertedBlockIsCountedAndItsPayloadIsVisible() throws {
        let decoder = StructuredAssistantDecoder(tokenizer: tok, allowedTools: ["get_weather"],
                                                 idGenerator: { "call_fixed" })
        decoder.inkling2MalformedToolCallsAsText = true
        var detok = MFDetokenizer(tokenizer: tok)
        let body = #"{"name":"send_email","args":{}}"#
        for id in tok.encode(mdl + "send_email" + invoke + body + end, addBOS: false) {
            _ = try decoder.consume(tokenID: id, delta: detok.push(id))
        }
        #expect(decoder.convertedInkling2MalformedToolCalls == 1)
        #expect(decoder.payloadTokenCounts?.visible == tok.encode(body, addBOS: false).count)
        #expect(!decoder.hasToolCalls)
    }

    @Test func goodBlocksAroundABadOneAreUnchanged() throws {
        let good = #"{"name":"get_weather","args":{"city":"Paris"}}"#
        let text = mdl + txt + "Checking." + end + mdl + "get_weather" + invoke + #"{"name":"get_weather","args":{"city":}"#
            + end + mdl + "get_weather" + invoke + good + end + eos
        let (events, finish) = try run(text, asText: true)
        #expect(visible(events + finish) == "Checking." + #"{"name":"get_weather","args":{"city":}"#)
        #expect(calls(events) == 1)
    }

    @Test func otherDialectsIgnoreTheSwitch() async throws {
        let chatml = try await MFTokenizer.load(from: try #require(Bundle.module.url(
            forResource: "ChatMLTokenizer", withExtension: nil, subdirectory: "Fixtures")))
        #expect(chatml.dialect == .chatml)
        let decoder = StructuredAssistantDecoder(tokenizer: chatml, allowedTools: ["get_weather"])
        decoder.inkling2MalformedToolCallsAsText = true
        var detok = MFDetokenizer(tokenizer: chatml)
        let text = "<tool_call>" + #"{"name":"get_weather","arguments":{"city":}"# + "</tool_call>"
        #expect(throws: (any Error).self) {
            for id in chatml.encode(text, addBOS: false) {
                _ = try decoder.consume(tokenID: id, delta: detok.push(id))
            }
            _ = try decoder.finish()
        }
    }
}
