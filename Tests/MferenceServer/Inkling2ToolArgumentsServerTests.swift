import Foundation
import Testing
@testable import Mference
@testable import MferenceServerCore

/// The round trip that prompt-prefix reuse needs after a tool call, model-free:
/// the model writes a call whose arguments nest objects out of key order and
/// hold a float written `1.0`; the server returns those arguments; the client
/// sends them back in the history with the tool's result; and the history's
/// render of the call is the tokens the model sampled, so the next turn
/// continues from the call's KV instead of prefilling everything again.
@Suite("Inkling 2 tool-call arguments through the server", .serialized)
struct Inkling2ToolArgumentsServerTests {
    typealias T = Inkling2ServerTests

    static var question: [String: Any] { ["role": "user", "content": "Look this up in my records."] }
    /// As the template writes a call: top level sorted, nested objects in the
    /// model's order, compact, numbers as written.
    static let arguments = #"{"filter":{"year":2025,"kind":"pdf","range":{"to":10,"from":1}},"query":"tax","weight":1.0}"#

    static func history(arguments: String) -> [[String: Any]] {
        [question,
         ["role": "assistant", "content": "",
          "tool_calls": [["id": "call_1", "type": "function",
                          "function": ["name": "query_records", "arguments": arguments]]]],
         ["role": "tool", "tool_call_id": "call_1", "content": "ok"]]
    }

    /// Runs the call turn, then the turn after it with `echo` as the history's
    /// arguments; returns the call's returned arguments, the KV the call left
    /// (its prompt and every sampled token but the EOS), the next turn's
    /// prompt and cached-token count, and the prompt tokens the runner was fed
    /// for it. The second turn is posted directly: `Harness.run` checks that
    /// the whole prompt was fed, which a reused prefix is not.
    static func callThenResult(_ h: T.Harness, echo: (String) -> String) async throws
        -> (returned: String, kv: [Int32], prompt: [Int32], cached: Int, fed: [Int32]) {
        let tools = [T.tool("query_records", "Look up records.")]
        let call = T.mdl + "query_records" + T.invoke + #"{"name":"query_records","args":"# + arguments + "}"
            + T.end + T.eos
        let first = try await h.run(T.base([question], tools: tools), generates: call)
        #expect(first.status == 200, "\(first.text)")
        let s = T.Stream(first.text)
        #expect(s.finishReasons == ["tool_calls"])
        let function = try #require(s.toolCalls.first?["function"] as? [String: Any])
        let returned = try #require(function["arguments"] as? String)
        let kv = h.runner.inputs

        h.runner.plan(script: h.tokenizer.encode(T.mdl + T.txt + "Found it." + T.end + T.eos, addBOS: false))
        let second = try await h.post(T.base(history(arguments: echo(returned)), tools: tools))
        #expect(second.status == 200, "\(second.text)")
        let stream = T.Stream(second.text)
        let usage = try #require(stream.usage)
        let cached = try #require((usage["prompt_tokens_details"] as? [String: Any])?["cached_tokens"] as? Int)
        let completion = try #require(usage["completion_tokens"] as? Int)
        // Every generated token but the last (EOS) is fed back after the prompt.
        let fed = Array(h.runner.inputs.dropLast(completion - 1))
        return (returned, kv, h.model.lastPrompt, cached, fed)
    }

    @Test func aToolTurnContinuesFromTheCallsKV() async throws {
        try await T.withHarness(promptCacheMode: .singlePrefix) { h in
            let r = try await Self.callThenResult(h, echo: { $0 })
            // The arguments come back as the model wrote them.
            #expect(r.returned == Self.arguments)
            // The history's render begins with exactly the KV the call left:
            // the call's sampled tokens re-render to themselves.
            #expect(r.prompt.starts(with: r.kv), "the re-rendered call differs from the sampled one")
            #expect(r.prompt.count > r.kv.count)
            #expect(r.cached == r.kv.count, "the tool turn should continue from the call's KV")
            // The runner was fed only what follows the call.
            #expect(r.fed == Array(r.prompt.dropFirst(r.kv.count)))
        }
    }

    /// The render follows the text the client sends: arguments re-serialized
    /// with sorted keys and `1` for `1.0` render as written, which is not what
    /// the model sampled, so that turn prefills its whole prompt.
    @Test func aReserializedEchoRendersAsSentAndIsNotReused() async throws {
        try await T.withHarness(promptCacheMode: .singlePrefix) { h in
            let sorted = #"{"filter":{"kind":"pdf","range":{"from":1,"to":10},"year":2025},"query":"tax","weight":1}"#
            let r = try await Self.callThenResult(h, echo: { _ in sorted })
            #expect(!r.prompt.starts(with: r.kv))
            #expect(r.cached == 0)
            #expect(r.fed == r.prompt)
            let rendered = h.tokenizer.decode(r.prompt)
            #expect(rendered.contains(#"{"name":"query_records","args":"# + sorted + "}"), "\(rendered)")
        }
    }

    /// Every other dialect's history keeps no arguments text, so it parses and
    /// renders exactly as before; Inkling 2's keeps the text the client sent,
    /// in compact form, so a re-spaced echo compares equal to the call.
    @Test func onlyInklingKeepsTheArgumentsText() throws {
        let compact = #"{"b":{"y":2,"x":1.0},"a":1}"#
        for raw in [compact, "{ \"b\": {\"y\": 2, \"x\": 1.0},\n \"a\": 1 }"] {
            let body: [String: Any] = ["model": "m", "messages": Self.history(arguments: raw)]
            let request = try JSONDecoder().decode(OpenAIChatRequest.self,
                                                   from: JSONSerialization.data(withJSONObject: body))
            for dialect in [ChatDialect.chatml, .minicpm, .inkling2] {
                let validated = try OpenAIRequestValidator.validate(request, modelID: "m", dialect: dialect)
                let call = try #require(validated.messages.first { !$0.toolCalls.isEmpty }?.toolCalls.first)
                #expect(call.argumentsText == (dialect == .inkling2 ? compact : nil), "\(dialect)")
                let parsed = try JSONDecoder().decode(JSONValue.self, from: Data(raw.utf8))
                #expect(call.arguments == parsed, "\(dialect)")
            }
        }
    }

    /// A text the render would not use keeps none: an unpaired surrogate
    /// escape the history validator accepts is rendered from the parsed
    /// value, as before.
    @Test func aTextTheRenderCannotUseIsNotKept() {
        #expect(Inkling2ToolArguments.normalizedText(#"{"a":"\ud83d"}"#) == nil)
        #expect(Inkling2ToolArguments.normalizedText("[1]") == nil)
        #expect(Inkling2ToolArguments.normalizedText("{ \"a\" : 1.0 }") == #"{"a":1.0}"#)
    }
}
