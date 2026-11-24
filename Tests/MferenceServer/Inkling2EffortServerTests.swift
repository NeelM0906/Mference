import Foundation
import Testing
@testable import Mference
@testable import MferenceEngine
@testable import MferenceServerCore

/// Ruling D (fixture F05): Inkling 2 takes every name its template's
/// `effort_map` has, the four shared ones (`none`, `low`, `medium`, `xhigh`)
/// and its own `minimal`, `high` and `max`, rendered as the template's
/// levels. Other dialects keep the four. Same harness as the F01-F06 tests.
@Suite("Inkling 2 reasoning effort through the server (K0 fixture F05)", .serialized)
struct Inkling2EffortServerTests {
    static let levels: [(name: String?, printed: String)] = [
        (nil, "0.9"), ("none", "0"), ("minimal", "0.1"), ("low", "0.2"), ("medium", "0.7"),
        ("high", "0.9"), ("xhigh", "0.99"), ("max", "0.99"),
    ]

    @Test func everyTemplateNameRendersItsLevel() async throws {
        try await Inkling2ServerTests.withHarness { h in
            for level in Self.levels {
                var body = Inkling2ServerTests.base([["role": "user", "content": "Hi"]])
                if let name = level.name { body["reasoning_effort"] = name }
                let (status, text, prompt) = try await h.run(
                    body, generates: Inkling2ServerTests.mdl + Inkling2ServerTests.txt + "Hello."
                        + Inkling2ServerTests.end + Inkling2ServerTests.eos)
                let label = level.name ?? "omitted"
                #expect(status == 200, "\(label): \(text)")
                let rendered = h.tokenizer.decode(prompt, skipSpecialTokens: false)
                #expect(rendered.contains("Thinking effort level: \(level.printed)<|end_message|>"),
                        "\(label): \(rendered)")
                // The shared names travel as the shared setting, Inkling's own
                // as its own; nothing for an omitted effort.
                let request = try #require(h.model.requests.last)
                switch level.name {
                case nil:
                    #expect(request.reasoningEffort == nil && request.inkling2Effort == nil)
                case let name? where QwenReasoningEffort(rawValue: name) != nil:
                    #expect(request.reasoningEffort?.rawValue == name && request.inkling2Effort == nil)
                case let name?:
                    #expect(request.reasoningEffort == nil && request.inkling2Effort == .named(name))
                }
            }
        }
    }

    @Test func otherValuesAreRefused() async throws {
        try await Inkling2ServerTests.withHarness { h in
            for value in ["ultra", "High", "0.5", ""] {
                var body = Inkling2ServerTests.base([["role": "user", "content": "Hi"]])
                body["reasoning_effort"] = value
                h.runner.plan(script: [])
                let (status, text) = try await h.post(body)
                #expect(status == 400, "\(value): \(text)")
                #expect(text.contains("unsupported_value") && text.contains("reasoning_effort"), "\(value)")
                #expect(text.contains("none, minimal, low, medium, high, xhigh, or max"), "\(value)")
            }
            #expect(h.model.requests.isEmpty)
        }
    }

    /// Other dialects keep the four shared names: Qwen 3.8's validator still
    /// refuses `high`, `minimal` and `max` with its own message, and sets no
    /// Inkling effort for the names it takes.
    @Test func otherDialectsKeepTheFourNames() throws {
        func validate(_ effort: String) throws -> ValidatedChatRequest {
            let data = Data(#"{"model":"m","reasoning_effort":"\#(effort)","messages":[{"role":"user","content":"Hi"}]}"#.utf8)
            let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
            return try OpenAIRequestValidator.validate(request, modelID: "m", dialect: .chatml, qwenReasoning: true)
        }
        for effort in ["high", "minimal", "max"] {
            do {
                _ = try validate(effort)
                Issue.record("\(effort) was accepted for ChatML")
            } catch let error as ServerRequestError {
                #expect(String(describing: error).contains("xhigh, medium, low, or none"), "\(error)")
            }
        }
        for effort in ["xhigh", "medium", "low", "none"] {
            let validated = try validate(effort)
            #expect(validated.reasoningEffort?.rawValue == effort && validated.inkling2Effort == nil)
        }
    }

    /// With single-prefix reuse on, a follow-up turn at another effort level
    /// is not continued from KV rendered at the first level (the effort line
    /// differs), for Inkling's own names and the shared ones alike; at the
    /// same level it is reused.
    @Test func aDifferentEffortIsNotReusedAsAPrefix() async throws {
        try await Inkling2ServerTests.withHarness(promptCacheMode: .singlePrefix) { h in
            func turn(_ messages: [[String: Any]], _ effort: String) async throws -> (prefilled: Int, prompt: Int) {
                var body = Inkling2ServerTests.base(messages)
                body["reasoning_effort"] = effort
                h.runner.plan(script: h.tokenizer.encode(Inkling2ServerTests.mdl + Inkling2ServerTests.txt + "Fine."
                    + Inkling2ServerTests.end + Inkling2ServerTests.eos, addBOS: false))
                let (status, text) = try await h.post(body)
                #expect(status == 200, "\(text)")
                let completion = try #require(Inkling2ServerTests.Stream(text).usage?["completion_tokens"] as? Int)
                return (h.runner.inputs.count - (completion - 1), h.model.lastPrompt.count)
            }
            let question: [String: Any] = ["role": "user", "content": "How are you?"]
            let reply: [String: Any] = ["role": "assistant", "content": "Fine."]
            let next: [String: Any] = ["role": "user", "content": "Good."]
            for (first, second, reused) in [("high", "high", true), ("high", "minimal", false),
                                            ("max", "xhigh", true), ("low", "xhigh", false),
                                            ("minimal", "low", false)] {
                _ = try await turn([question], first)
                let follow = try await turn([question, reply, next], second)
                if reused {
                    #expect(follow.prefilled < follow.prompt, "\(first) then \(second) should be reused")
                } else {
                    #expect(follow.prefilled == follow.prompt, "\(first) then \(second) must not be reused")
                }
            }
        }
    }

    /// Inkling 2's own names think like the shared names other than `none`:
    /// the same default completion budget, and an explicit name takes
    /// precedence over `chat_template_kwargs.enable_thinking`.
    @Test func ownNamesGetTheThinkingBudgetAndWinOverEnableThinking() throws {
        func validate(_ json: String) throws -> ValidatedChatRequest {
            let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: Data(json.utf8))
            return try OpenAIRequestValidator.validate(request, modelID: "m", dialect: .inkling2,
                                                       acceptsReasoningEffort: true)
        }
        let user = #""messages":[{"role":"user","content":"Hi"}]"#
        for name in ["minimal", "high", "max", "low", "medium", "xhigh"] {
            let validated = try validate(#"{"model":"m","reasoning_effort":"\#(name)",\#(user)}"#)
            #expect(validated.maximumCompletionTokens == 32_768, "\(name)")
        }
        #expect(try validate(#"{"model":"m","reasoning_effort":"none",\#(user)}"#).maximumCompletionTokens == 4_096)
        let both = try validate(
            #"{"model":"m","reasoning_effort":"high","chat_template_kwargs":{"enable_thinking":false},\#(user)}"#)
        #expect(both.inkling2Effort == .named("high") && both.reasoningEffort == nil)
        #expect(both.maximumCompletionTokens == 32_768)
    }
}
