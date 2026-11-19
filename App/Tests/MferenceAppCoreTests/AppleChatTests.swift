import Foundation
@testable import MferenceAppCore
import Synchronization
import Testing

/// Settings > Models > Apple Intelligence model, flipped during a test.
final class AppleSwitch: Sendable {
    private let value: Mutex<Bool>
    init(_ on: Bool) { value = Mutex(on) }
    var isOn: Bool {
        get { value.withLock { $0 } }
        set { value.withLock { $0 = newValue } }
    }
}

@MainActor
@Suite struct AppleChatTests {
    static func routing(on: Bool = true, info: AppleModelInfo = AppleModelInfo(state: .ready),
                        answer: Result<GenerationSummary, AssistantModelError>? = nil)
        -> (AppleChatRouting, ScriptedEngineService, ScriptedAppleChatModel, AppleSwitch) {
        let engine = ScriptedEngineService(replies: [], fallback: .init(content: "From the engine."))
        let apple = answer.map { ScriptedAppleChatModel(info: info, answer: $0) } ?? ScriptedAppleChatModel(info: info)
        let toggle = AppleSwitch(on)
        let routing = AppleChatRouting(engine: engine, apple: apple, isEnabled: { toggle.isOn })
        return (routing, engine, apple, toggle)
    }

    static func summary(_ content: String, trimmed: Bool = false) -> GenerationSummary {
        GenerationSummary(content: content, reasoning: nil, toolCalls: [], finish: .stop,
                          promptTokens: 10, completionTokens: 5, historyTrimmed: trimmed)
    }

    // MARK: The model list

    @Test func appleIsNeverAnEngineModel() {
        #expect(!AppModel.engineModels.contains(.apple))
        #expect(AppModel.engineModels.count == AppModel.allCases.count - 1)
        #expect(AppModel.apple.group == .builtIntoMacOS)
        #expect(AppModel(modelID: "apple-foundation-model") == .apple)
        #expect(!AppModel.apple.supportsThinking)
    }

    @Test func appleRowFollowsTheSettingsSwitch() async {
        let (routing, _, _, toggle) = Self.routing(on: false)
        #expect(await routing.models().contains { $0.model == .apple } == false)
        toggle.isOn = true
        let apple = await routing.models().first { $0.model == .apple }
        #expect(apple?.availability == .ready)
        #expect(apple?.contextLength == 4_096)
    }

    @Test func appleRowReflectsAppleIntelligence() async {
        let (routing, _, apple, _) = Self.routing(info: AppleModelInfo(state: .turnedOff))
        var row = await routing.models().first { $0.model == .apple }
        #expect(row?.availability == .notInstalled(reason: AppleChatRouting.turnedOffReason))
        #expect(AppleChatRouting.status(forUnavailable: AppleChatRouting.turnedOffReason) == "Turned off")

        apple.setInfo(AppleModelInfo(state: .gettingReady))
        row = await routing.models().first { $0.model == .apple }
        #expect(row?.availability == .notInstalled(reason: AppleChatRouting.gettingReadyReason))
        #expect(AppleChatRouting.status(forUnavailable: AppleChatRouting.gettingReadyReason) == "Getting ready")

        // A Mac that cannot run Apple Intelligence never shows the row.
        apple.setInfo(AppleModelInfo(state: .notEligible))
        #expect(await routing.models().contains { $0.model == .apple } == false)
    }

    // MARK: Routing

    @Test func appleChatsGoToAppleAndOthersToTheEngine() async throws {
        let (routing, engine, apple, _) = Self.routing(answer: .success(Self.summary("From Apple.")))
        let turns = [ChatTurn(role: .user, content: "Hi")]
        var appleText = ""
        for try await update in routing.generate(model: .apple, turns: turns, options: GenerationOptions()) {
            if case .content(let text) = update { appleText += text }
        }
        #expect(appleText == "From Apple.")
        #expect(apple.requests.count == 1)
        #expect(engine.requests.isEmpty)

        var engineText = ""
        for try await update in routing.generate(model: .quick, turns: turns, options: GenerationOptions()) {
            if case .content(let text) = update { engineText += text }
        }
        #expect(engineText == "From the engine.")
        #expect(engine.requests.map(\.model) == [.quick])
    }

    @Test func appleRefusesWhileSwitchedOff() async throws {
        let (routing, _, apple, _) = Self.routing(on: false)
        await #expect(throws: AssistantModelError.noModelAvailable(reason: AppleChatRouting.switchedOffReason)) {
            for try await _ in routing.generate(model: .apple, turns: [ChatTurn(role: .user, content: "Hi")],
                                                options: GenerationOptions()) {}
        }
        #expect(apple.requests.isEmpty)
    }

    @Test func pickingAppleLoadsNothing() async throws {
        let (routing, engine, _, _) = Self.routing()
        try await routing.load(.apple)
        #expect(engine.status.phase == .empty)
    }

    // MARK: The chat

    @Test func chatAnswersWithAppleWithoutTheWeb() async throws {
        let (routing, engine, apple, _) = Self.routing(answer: .success(Self.summary("Sure thing.")))
        let (chat, _) = try ChatViewModelTests.viewModel(routing, tools: StubToolbox())
        chat.start()
        await waitUntil { chat.option(for: .apple) != nil }
        chat.setWebEnabled(true)
        chat.selectModel(.apple)
        #expect(chat.model == .apple)
        #expect(chat.webBlockedReason == ChatViewModel.appleWebReason)
        #expect(chat.webBlockedByModel)

        chat.draft = "Make this friendlier"
        chat.send()
        await chat.waitUntilIdle()
        #expect(engine.requests.isEmpty)
        let turns = try #require(apple.requests.first)
        // No tool prompt: Apple's chats run without the web.
        #expect(!turns.contains { $0.role == .system && ($0.content ?? "").contains("Use the tools.") })
        #expect(chat.messages.last?.content == "Sure thing.")
        #expect(chat.messages.last?.model == .apple)
        #expect(chat.historyTrimmed == false)
        chat.stopObserving()
    }

    @Test func trimmedHistoryShowsTheNoticeUntilTheModelChanges() async throws {
        let (routing, _, _, _) = Self.routing(answer: .success(Self.summary("Short answer.", trimmed: true)))
        let (chat, _) = try ChatViewModelTests.viewModel(routing)
        chat.start()
        await waitUntil { chat.option(for: .apple) != nil }
        chat.selectModel(.apple)
        chat.draft = "A long chat"
        chat.send()
        await chat.waitUntilIdle()
        #expect(chat.historyTrimmed)
        #expect(chat.longChatModel == .deep)
        chat.selectModel(.deep)
        #expect(chat.historyTrimmed == false)
        chat.stopObserving()
    }

    @Test func switchingAppleOffMovesItsChatToTheDefaultModel() async throws {
        let (routing, _, _, toggle) = Self.routing(answer: .success(Self.summary("Hi.")))
        let (chat, _) = try ChatViewModelTests.viewModel(routing)
        chat.start()
        await waitUntil { chat.option(for: .apple) != nil }
        chat.selectModel(.apple)
        chat.draft = "Hello"
        chat.send()
        await chat.waitUntilIdle()
        let id = try #require(chat.selectedID)

        toggle.isOn = false
        routing.appleModelChanged()
        await waitUntil { chat.option(for: .apple) == nil }
        #expect(chat.model == .quick)
        // The chat itself moved, so its next answer comes from Quick.
        chat.newChat()
        chat.select(id)
        #expect(chat.model == .quick)
        chat.stopObserving()
    }

    @Test func appleIsNotPickableWithoutTheRouting() async throws {
        let (chat, _) = try ChatViewModelTests.viewModel(ScriptedEngineService())
        chat.start()
        await waitUntil { chat.option(for: .maple)?.availability == .ready }
        chat.selectModel(.apple)
        #expect(chat.model == .quick)
        #expect(chat.option(for: .apple) == nil)
        chat.stopObserving()
    }

    // MARK: Fitting the context

    @Test func splitKeepsInstructionsHistoryAndPrompt() throws {
        let call = ToolCall(id: "c", name: "web_search", arguments: .object([:]), argumentsJSON: "{}")
        let plan = try #require(AppleChatContext.split([
            ChatTurn(role: .system, content: "Be brief."),
            ChatTurn(role: .user, content: "One"),
            ChatTurn(role: .assistant, content: "", toolCalls: [call]),
            ChatTurn(role: .tool, content: "result", toolCallID: "c", toolName: "web_search"),
            ChatTurn(role: .assistant, content: "Uno"),
            ChatTurn(role: .user, content: "Two"),
        ]))
        #expect(plan.instructions == "Be brief.")
        #expect(plan.history.map(\.content) == ["One", "Uno"])
        #expect(plan.prompt == "Two")
        #expect(AppleChatContext.split([ChatTurn(role: .assistant, content: "No question")]) == nil)
    }

    @Test func fitDropsTheOldestExchangesFirst() async throws {
        let turns = (1...4).flatMap { index in
            [ChatTurn(role: .user, content: "q\(index)"), ChatTurn(role: .assistant, content: "a\(index)")]
        } + [ChatTurn(role: .user, content: "last")]
        let plan = try #require(AppleChatContext.split(turns))
        // Cost: one token per entry.
        let fitted = await AppleChatContext.fit(plan, budget: 5) { $0.history.count + 1 }
        #expect(fitted?.history.map(\.content) == ["q3", "a3", "q4", "a4"])
        #expect(fitted?.trimmed == true)
        let untouched = await AppleChatContext.fit(plan, budget: 100) { $0.history.count + 1 }
        #expect(untouched?.trimmed == false)
        // The prompt alone does not fit.
        let tooSmall = await AppleChatContext.fit(plan, budget: 0) { $0.history.count + 1 }
        #expect(tooSmall == nil)
    }

    @Test func streamingDeltasAndBudgets() {
        #expect(AppleChatContext.delta(sent: "Hel", snapshot: "Hello") == "lo")
        #expect(AppleChatContext.delta(sent: "Hello", snapshot: "Hello") == nil)
        #expect(AppleChatContext.delta(sent: "Hi", snapshot: "Hello") == nil)
        #expect(AppleChatContext.answerReserve(contextSize: 4_096, requested: 4_096) == 1_024)
        #expect(AppleChatContext.answerReserve(contextSize: 4_096, requested: 200) == 200)
    }

    // MARK: Live (Apple Intelligence on this Mac; off unless asked)

    /// `MFERENCE_APPLE_CHAT_LIVE=1`: one short answer and one chat longer
    /// than the context, through Apple's real model. Loads no Mference model.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["MFERENCE_APPLE_CHAT_LIVE"] == "1"))
    func liveAppleAnswersAndTrimsLongChats() async throws {
        let model = FoundationChatModel()
        let info = await model.info()
        try #require(info.state == .ready, "Apple Intelligence is not ready on this Mac")
        #expect(AppleModelLabels.current == info.labels)

        func run(_ turns: [ChatTurn]) async throws -> (streamed: String, summary: GenerationSummary?) {
            var streamed = ""
            var summary: GenerationSummary?
            for try await update in model.generate(turns: turns, options: GenerationOptions(maximumCompletionTokens: 64)) {
                switch update {
                case .content(let text): streamed += text
                case .completed(let done): summary = done
                default: break
                }
            }
            return (streamed, summary)
        }

        let short = try await run([ChatTurn(role: .system, content: "Answer in one short sentence."),
                                   ChatTurn(role: .user, content: "What color is a clear daytime sky?")])
        #expect(!short.streamed.isEmpty)
        #expect(short.summary?.content == short.streamed)
        #expect(short.summary?.historyTrimmed == false)

        let filler = String(repeating: "The quick brown fox jumps over the lazy dog. ", count: 20)
        let long = (1...30).flatMap { index in
            [ChatTurn(role: .user, content: "Note \(index): \(filler)"),
             ChatTurn(role: .assistant, content: "Noted \(index).")]
        } + [ChatTurn(role: .user, content: "Reply with the word done.")]
        let trimmed = try await run(long)
        #expect(!trimmed.streamed.isEmpty)
        #expect(trimmed.summary?.historyTrimmed == true)
        print("live apple: \(info.labels.detail); short=\(short.streamed.debugDescription); "
              + "long prompt tokens=\(trimmed.summary?.promptTokens ?? -1)")
    }

    @Test func labelsNameTheVariantAndContext() {
        let core = AppleModelLabels(variantName: "AFM 3 Core", shortName: "Apple 3B", contextSize: 4_096)
        #expect(core.detail == "AFM 3 Core · 4K context")
        #expect(AppleModelLabels(variantName: "AFM 3 Core Advanced", shortName: "Apple 20B",
                                 contextSize: 8_192).contextLabel == "8K")
    }
}
