import Foundation
import Testing
@testable import Mference
@testable import MferenceServerCore

/// Opt-in real-install coverage for Inkling 2 Mini on the single-model server
/// path (`MferenceServer --model`), which the model-free suites cannot reach:
/// `Inkling2ServerTests` serves a scripted runner through the library path,
/// while `--model` serves a `ServerModelSession` over the real runner. Its
/// Inkling-only branches (a malformed call sent as text, and the prefix
/// cache's choice after a turn) run here on one real turn; a malformed call
/// cannot be scripted on a real model, so this pins the normal turn.
///
/// Loads the 17.6 GB install lazily (`.pread`, the session's default), so it
/// runs only with `MFERENCE_INKLING2_GTURBO=<install>` set, and then only
/// inside `model-hold.sh`.
@Suite("Inkling 2 Mini on the --model server path (real install)", .serialized)
struct Inkling2InstalledServerTests {
    static let install = ProcessInfo.processInfo.environment["MFERENCE_INKLING2_GTURBO"]

    private static func log(_ message: String) {
        FileHandle.standardError.write(Data(("[inkling2 --model server] " + message + "\n").utf8))
    }

    /// One streamed chat turn through the server as `--model` builds it
    /// (`MferenceServer/Command/main.swift`), with single-prefix reuse so the
    /// turn's cache branch runs: thinking then text, `stop`, and usage that
    /// counts the thinking tokens.
    @Test(.enabled(if: install != nil, "needs MFERENCE_INKLING2_GTURBO"))
    func aRealTurnThinksThenAnswersWithUsage() async throws {
        let path = try #require(Self.install)
        let started = Date()
        let backend = try await ServerModelSession.load(modelDirectory: URL(fileURLWithPath: path),
                                                        maxContext: 4_096, promptCacheMode: .singlePrefix)
        Self.log("loaded in \(String(format: "%.1f", Date().timeIntervalSince(started))) s")
        #expect(backend.defaultModelID == Inkling2ServerTests.modelID)
        #expect(backend.chatDialect == .inkling2)
        #expect(backend.maxContext == 4_096)
        let server = MferenceHTTPServer(modelID: backend.defaultModelID, queueLimit: 1, backend: backend,
                                        chatDialect: backend.chatDialect, maxModelLen: backend.maxContext)
        let channel = try await server.start(port: 0)
        do {
            let port = try #require(channel.localAddress?.port)
            let body: [String: Any] = [
                "model": backend.defaultModelID, "stream": true, "stream_options": ["include_usage": true],
                "max_completion_tokens": 1_024, "temperature": 0, "seed": 1, "reasoning_effort": "low",
                "messages": [["role": "user", "content": "What is the capital of France? Answer in one word."]],
            ]
            var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "content-type")
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
            request.timeoutInterval = 1_200
            let turnStarted = Date()
            let (data, response) = try await URLSession.shared.data(for: request)
            let text = String(decoding: data, as: UTF8.self)
            let s = Inkling2ServerTests.Stream(text)
            Self.log("turn in \(String(format: "%.1f", Date().timeIntervalSince(turnStarted))) s; "
                + "finish=\(s.finishReasons) usage=\(s.usage.map { "\($0)" } ?? "-")")
            Self.log("reasoning: \(s.reasoning)")
            Self.log("content: \(s.content)")

            #expect((response as? HTTPURLResponse)?.statusCode == 200, "\(text)")
            #expect(s.errors == 0 && s.done && s.shapeOK, "\(text)")
            #expect(s.models == [backend.defaultModelID])
            #expect(s.roles == ["assistant"])
            #expect(s.deltaKeysInOrder == ["role", "reasoning", "content"], "\(s.deltaKeysInOrder)")
            #expect(s.toolCalls.isEmpty && s.finishReasons == ["stop"])
            #expect(s.content.localizedCaseInsensitiveContains("Paris"), "\(s.content)")
            for framing in Inkling2ServerTests.framing {
                #expect(!s.content.contains(framing) && !s.reasoning.contains(framing), "\(framing) leaked")
            }

            let usage = try #require(s.usage)
            let promptTokens = try #require(usage["prompt_tokens"] as? Int)
            let completionTokens = try #require(usage["completion_tokens"] as? Int)
            #expect(promptTokens > 0 && completionTokens > 0)
            #expect(usage["total_tokens"] as? Int == promptTokens + completionTokens)
            #expect((usage["prompt_tokens_details"] as? [String: Any])?["cached_tokens"] as? Int == 0)
            let details = try #require(usage["completion_tokens_details"] as? [String: Any])
            let reasoningTokens = try #require(details["reasoning_tokens"] as? Int)
            #expect(reasoningTokens > 0 && reasoningTokens < completionTokens, "\(details)")
        } catch {
            try? await server.shutdown()
            throw error
        }
        try await server.shutdown()
    }
}
