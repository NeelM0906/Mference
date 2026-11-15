import Foundation
import Testing
@testable import Mference
@testable import MferenceServerCore

/// Opt-in real-checkpoint coverage for the server's multi-prefix cache on
/// Qwen 3.6. Gate: `MFERENCE_QWEN36_GTURBO=<install>`; prompt size:
/// `MFERENCE_PREFIX_CACHE_TOKENS` (default 2048). Run only after AGENTS.md
/// preflight, under the machine lock. Each arm loads the install and releases
/// it before the next arm opens it.
///
/// The control is today's `single-prefix` server on the conversation alone.
/// Every multi-prefix arm must give the continued turn exactly the control's
/// greedy output and cached-token count, with other requests in between.
@Suite(.serialized) struct InstalledPrefixCacheTests {
    private static let path = ProcessInfo.processInfo.environment["MFERENCE_QWEN36_GTURBO"]
    private static let promptTokens = ProcessInfo.processInfo.environment["MFERENCE_PREFIX_CACHE_TOKENS"]
        .flatMap(Int.init) ?? 2048

    private static func log(_ message: String) {
        FileHandle.standardError.write(Data(("[installed prefix cache] " + message + "\n").utf8))
    }

    private struct Turn: Equatable {
        let content: String
        let reasoning: String?
        let completionTokens: Int
        let cachedTokens: Int
        let outcome: String?
    }

    private static func request(_ messages: [MFTokenizer.Message], maxTokens: Int = 48) -> ValidatedChatRequest {
        ValidatedChatRequest(messages: messages, tools: [], stream: false, includeUsage: true,
                             generationConfig: GenerationConfig(maxNewTokens: maxTokens, temperature: 0),
                             maximumCompletionTokens: maxTokens)
    }

    private static func send(_ session: ServerModelSession, _ messages: [MFTokenizer.Message]) async throws -> Turn {
        let prepared = try await session.prepare(request(messages))
        let started = Date()
        let completion = try await session.generate(prepared) { _ in }
        let turn = Turn(content: completion.content, reasoning: completion.reasoningContent,
                        completionTokens: completion.usage.completionTokens,
                        cachedTokens: completion.usage.promptTokensDetails.cachedTokens,
                        outcome: completion.diagnostics?.prefixCache?.outcome)
        log("prompt=\(prepared.promptIDs.count) cached=\(turn.cachedTokens) new=\(turn.completionTokens) "
            + "outcome=\(turn.outcome ?? "-") seconds=\(String(format: "%.1f", Date().timeIntervalSince(started))) "
            + "prefixCache=\(completion.diagnostics?.prefixCache.map { "\($0)" } ?? "-")")
        return turn
    }

    /// A long first user message of about `promptTokens` tokens.
    private static func firstMessages() -> [MFTokenizer.Message] {
        var text = "Read these field notes, then answer the questions that follow.\n"
        var day = 0
        while text.utf8.count < promptTokens * 4 {
            day += 1
            text += "Day \(day): the river gauge read \(day * 37 % 211) cm at dawn and \(day * 53 % 197) cm at dusk; "
                + "the crew logged \(day % 7 + 2) samples and noted \(["clear", "silty", "green", "brown"][day % 4]) water.\n"
        }
        return [.init(role: .user, content: text + "\nWhich day had the highest dawn reading?")]
    }

    private static func followUp(_ first: [MFTokenizer.Message], _ reply: Turn, _ question: String)
        -> [MFTokenizer.Message] {
        first + [.init(role: .assistant, content: reply.content, reasoningContent: reply.reasoning),
                 .init(role: .user, content: question)]
    }

    private static let otherConversation: [MFTokenizer.Message] = [
        .init(role: .user, content: "Name three prime numbers greater than fifty.")]
    private static let question = "List the days whose water was green, in order."
    private static let editedQuestion = "How many samples were logged on day 3?"

    private static func load(_ mode: ServerPromptCacheMode) async throws -> ServerModelSession {
        let path = try #require(Self.path)
        return try await ServerModelSession.load(
            modelDirectory: URL(fileURLWithPath: path), maxContext: promptTokens + 4096,
            promptCacheMode: mode, promptCacheBytes: .auto)
    }

    @Test(.enabled(if: path != nil)) func continuedTurnsMatchTheSinglePrefixControl() async throws {
        setenv("MFERENCE_DIAGNOSTICS", "1", 1)
        let first = Self.firstMessages()

        Self.log("arm: control (single-prefix) A1, A2; A1, A2'")
        var control: (a1: Turn, a2: Turn, edited: Turn)
        do {
            let session = try await Self.load(.singlePrefix)
            let a1 = try await Self.send(session, first)
            let a2 = try await Self.send(session, Self.followUp(first, a1, Self.question))
            _ = try await Self.send(session, Self.otherConversation)
            let againA1 = try await Self.send(session, first)
            #expect(againA1 == a1)
            let edited = try await Self.send(session, Self.followUp(first, a1, Self.editedQuestion))
            control = (a1, a2, edited)
        }
        #expect(control.a2.cachedTokens > 0)

        Self.log("arm: multi-prefix A1, B, A2 (restore), A2' (rewind), A2 again")
        do {
            let session = try await Self.load(.multiPrefix)
            let a1 = try await Self.send(session, first)
            #expect(a1 == Turn(content: control.a1.content, reasoning: control.a1.reasoning,
                               completionTokens: control.a1.completionTokens, cachedTokens: 0, outcome: "miss"))
            _ = try await Self.send(session, Self.otherConversation)
            let a2 = try await Self.send(session, Self.followUp(first, a1, Self.question))
            #expect(a2.outcome == "restored")
            #expect(a2.content == control.a2.content)
            #expect(a2.completionTokens == control.a2.completionTokens)
            #expect(a2.cachedTokens == control.a2.cachedTokens)
            let edited = try await Self.send(session, Self.followUp(first, a1, Self.editedQuestion))
            #expect(edited.outcome == "rewound")
            #expect(edited.content == control.edited.content)
            #expect(edited.cachedTokens == control.edited.cachedTokens)
            let again = try await Self.send(session, Self.followUp(first, a1, Self.question))
            #expect(again.outcome == "rewound")
            #expect(again.content == control.a2.content)
        }

        Self.log("arm: multi-prefix A1, B cancelled, A2")
        do {
            let session = try await Self.load(.multiPrefix)
            let a1 = try await Self.send(session, first)
            // B cancels itself at its first streamed event, so the cancel
            // always lands after B has started writing the runner.
            let prepared = try await session.prepare(Self.request(Self.otherConversation))
            await #expect(throws: (any Error).self) {
                _ = try await session.generate(prepared) { _ in withUnsafeCurrentTask { $0?.cancel() } }
            }
            let a2 = try await Self.send(session, Self.followUp(first, a1, Self.question))
            #expect(a2.outcome == "restored")
            #expect(a2.content == control.a2.content)
            #expect(a2.cachedTokens == control.a2.cachedTokens)
        }

        Self.log("arm: multi-prefix with a zero budget A1, B, A2 (cold, as single-prefix)")
        do {
            let path = try #require(Self.path)
            let session = try await ServerModelSession.load(
                modelDirectory: URL(fileURLWithPath: path), maxContext: Self.promptTokens + 4096,
                promptCacheMode: .multiPrefix, promptCacheBytes: .bytes(0))
            let a1 = try await Self.send(session, first)
            _ = try await Self.send(session, Self.otherConversation)
            let a2 = try await Self.send(session, Self.followUp(first, a1, Self.question))
            #expect(a2.outcome == "miss")
            #expect(a2.cachedTokens == 0)
        }
    }
}
