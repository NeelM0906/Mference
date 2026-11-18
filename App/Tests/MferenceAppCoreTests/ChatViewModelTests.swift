import Foundation
import Mference
import MferenceAppCore
import MferenceEngine
import SwiftData
import Testing

/// A toolbox that answers every call with one fixed source.
struct StubToolbox: ChatToolbox, ChatToolboxProvider {
    func systemPrompt(now: Date) -> String? { "Use the tools." }

    var definitions: [ToolDefinition] {
        [ToolDefinition(name: "web_search", description: "Search", parameters: .object(["type": .string("object")]))]
    }

    func run(_ call: ToolCall) async -> ToolOutcome {
        ToolOutcome(content: "[1] Oakline Duo sale, $499",
                    steps: [.search("oakline duo sale", results: 3)],
                    citations: [ChatCitation(number: 1, url: URL(string: "https://oakline.example/sale")!,
                                             title: "Oakline Duo sale")])
    }

    func makeToolbox(webEnabled: Bool) -> (any ChatToolbox)? { webEnabled ? self : nil }
    func webUsageChanged(enabled: Bool) {}
    var needsWebConsent: Bool { false }
    func grantWebConsent() {}
}

/// Needs consent until granted.
@MainActor final class ConsentTools: ChatToolboxProvider {
    var granted = false
    func makeToolbox(webEnabled: Bool) -> (any ChatToolbox)? { nil }
    func webUsageChanged(enabled: Bool) {}
    var needsWebConsent: Bool { !granted }
    func grantWebConsent() { granted = true }
}

/// A toolbox whose tool runs until it is cancelled.
struct HangingToolbox: ChatToolbox, ChatToolboxProvider {
    let started: FakeEngineLatch

    func systemPrompt(now: Date) -> String? { nil }
    var definitions: [ToolDefinition] { StubToolbox().definitions }

    func run(_ call: ToolCall) async -> ToolOutcome {
        started.open()
        try? await Task.sleep(for: .seconds(30))
        return ToolOutcome(content: "interrupted")
    }

    func makeToolbox(webEnabled: Bool) -> (any ChatToolbox)? { self }
    func webUsageChanged(enabled: Bool) {}
    var needsWebConsent: Bool { false }
    func grantWebConsent() {}
}

@MainActor
@Suite struct ChatViewModelTests {
    static func fakeService(_ script: FakeEngineScript) -> EngineService {
        let journal = FakeEngineJournal()
        let scripts = Dictionary(uniqueKeysWithValues: AppModel.allCases.map { ($0.modelID, script) })
        let native = FakeEngineBackend(engine: .mference, journal: journal, scripts: scripts)
        let llama = FakeEngineBackend(engine: .llamaCpp, journal: journal, scripts: scripts)
        let catalog = EngineCatalog(models: AppModel.allCases.map { model in
            FakeEngineBackend.descriptor(id: model.modelID,
                                         engine: model.engineLabel == "llama.cpp" ? .llamaCpp : .mference,
                                         product: EngineProductSlot(modelID: model.modelID))
        })
        return EngineService(discover: { catalog }) { catalog, observer in
            EngineHost(catalog: catalog, backends: [native, llama], observer: observer)
        }
    }

    static func viewModel(_ service: any EngineServicing,
                          tools: (any ChatToolboxProvider)? = nil) throws -> (ChatViewModel, ModelContainer) {
        let container = try AppModelContainer.inMemory()
        let model = ChatViewModel(service: service, store: ConversationStore(container: container), tools: tools)
        return (model, container)
    }

    static func storedMessages(_ container: ModelContainer) throws -> [StoredMessage] {
        try ModelContext(container).fetch(FetchDescriptor<StoredMessage>(sortBy: [SortDescriptor(\.index)]))
    }

    @Test func sendStreamsThroughTheEngineAndSavesBothMessages() async throws {
        let service = Self.fakeService(FakeEngineScript(events: [.reasoning("Let me think."), .content("Hello"),
                                                                 .content(" world")]))
        let (chat, container) = try Self.viewModel(service)
        chat.draft = "Say hello"
        chat.send()
        #expect(chat.isGenerating)
        #expect(chat.draft.isEmpty)
        await chat.waitUntilIdle()

        #expect(!chat.isGenerating)
        #expect(chat.streaming == nil)
        #expect(chat.errorMessage == nil)
        #expect(chat.messages.map(\.content) == ["Say hello", "Hello world"])
        #expect(chat.messages.last?.reasoning == "Let me think.")
        #expect(chat.messages.last?.model == .quick)
        #expect(chat.messages.last?.finish == .stop)
        let stored = try Self.storedMessages(container)
        #expect(stored.map(\.role) == [.user, .assistant])
        #expect(stored[1].reasoning == "Let me think.")
        #expect(stored[1].completionTokens == 3)
        #expect(chat.title == "Say hello")
        #expect(chat.groups.groups.first?.rows.first?.title == "Say hello")
    }

    @Test func stopKeepsThePartialAnswer() async throws {
        let reached = FakeEngineLatch()
        let service = Self.fakeService(FakeEngineScript(
            events: [.content("Partial"), .content(" answer"), .content(" never")],
            holdAfterEvents: 2, holdReached: reached, holdGate: FakeEngineLatch()))
        let (chat, container) = try Self.viewModel(service)
        chat.draft = "Long question"
        chat.send()
        try await reached.wait()
        let deadline = ContinuousClock.now + .seconds(5)
        while chat.streaming?.content != "Partial answer", ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(5))
        }
        chat.stop()
        await chat.waitUntilIdle()

        #expect(!chat.isGenerating)
        #expect(chat.errorMessage == nil)
        let stored = try Self.storedMessages(container)
        #expect(stored.count == 2)
        #expect(stored[1].content == "Partial answer")
        #expect(stored[1].finish == .cancelled)
        #expect(chat.messages.last?.finish == .cancelled)
        #expect(service.status.isGenerating == false)
    }

    @Test func regenerateReplacesTheLastAnswer() async throws {
        let service = ScriptedEngineService(replies: [.init(content: "First"), .init(content: "Second")])
        let (chat, container) = try Self.viewModel(service)
        chat.draft = "Question"
        chat.send()
        await chat.waitUntilIdle()
        chat.regenerate()
        await chat.waitUntilIdle()

        #expect(chat.messages.map(\.content) == ["Question", "Second"])
        #expect(try Self.storedMessages(container).map(\.content) == ["Question", "Second"])
        #expect(service.requests.count == 2)
        #expect(service.requests[1].turns.map(\.content) == ["Question"])
    }

    @Test func reasoningIsReplayedInHistory() async throws {
        let service = ScriptedEngineService(replies: [.init(content: "Paris.", reasoning: "Capital of France.")])
        let (chat, _) = try Self.viewModel(service)
        chat.draft = "Capital of France?"
        chat.send()
        await chat.waitUntilIdle()
        chat.draft = "And Spain?"
        chat.send()
        await chat.waitUntilIdle()

        let history = service.requests[1].turns
        #expect(history.map(\.role) == [.user, .assistant, .user])
        #expect(history[1].reasoning == "Capital of France.")
        #expect(history[1].content == "Paris.")
    }

    @Test func tokenUpdatesAreBatched() async throws {
        let text = String(repeating: "x", count: 400)
        let service = ScriptedEngineService(replies: [.init(content: text, chunk: 1)])
        let (chat, _) = try Self.viewModel(service)
        chat.draft = "Write a lot"
        chat.send()
        await chat.waitUntilIdle()

        #expect(chat.messages.last?.content == text)
        #expect(chat.flushCount >= 1)
        #expect(chat.flushCount < 100, "\(chat.flushCount) view updates for 400 tokens")
    }

    @Test func streamingTextReachesTheView() async throws {
        let service = ScriptedEngineService(replies: [.init(content: "abcdef", chunk: 1, delay: .milliseconds(20))])
        let (chat, _) = try Self.viewModel(service)
        chat.flushInterval = .milliseconds(10)
        chat.draft = "Go"
        chat.send()
        var seen: Set<String> = []
        while chat.isGenerating {
            if let text = chat.streaming?.content { seen.insert(text) }
            try await Task.sleep(for: .milliseconds(5))
        }
        #expect(seen.contains { !$0.isEmpty && $0 != "abcdef" }, "no intermediate text: \(seen)")
        #expect(chat.messages.last?.content == "abcdef")
    }

    @Test func errorsUseTheServiceMessage() async throws {
        let service = ScriptedEngineService(replies: [.init(content: "", error: .busy)])
        let (chat, container) = try Self.viewModel(service)
        chat.draft = "Hello"
        chat.send()
        await chat.waitUntilIdle()

        #expect(chat.errorMessage == EngineServiceError.busy.userMessage)
        #expect(chat.canRetryAfterError)
        chat.errorMessage = nil
        #expect(!chat.canRetryAfterError)
        #expect(try Self.storedMessages(container).map(\.role) == [.user])
    }

    @Test func toolLoopStoresHistoryAndShowsOneAnswerWithSources() async throws {
        let call = ToolCall(id: "call-1", name: "web_search", arguments: .object(["query": .string("oakline duo sale")]),
                            argumentsJSON: #"{"query":"oakline duo sale"}"#)
        let service = ScriptedEngineService(replies: [
            .init(content: "", reasoning: "I should search.", toolCalls: [call]),
            .init(content: "The Oakline Duo is on sale [1]."),
        ])
        let (chat, container) = try Self.viewModel(service, tools: StubToolbox())
        chat.setWebEnabled(true)
        chat.draft = "Is the Oakline Duo on sale?"
        chat.send()
        await chat.waitUntilIdle()

        let stored = try Self.storedMessages(container)
        #expect(stored.map(\.role) == [.user, .assistant, .tool, .assistant])
        #expect(stored[1].toolCalls == [call])
        #expect(stored[2].toolCallID == "call-1")
        #expect(stored[3].citations.map(\.number) == [1])
        #expect(stored[3].steps == [.search("oakline duo sale", results: 3)])

        #expect(chat.messages.count == 2)
        #expect(chat.messages[1].content == "The Oakline Duo is on sale [1].")
        #expect(chat.messages[1].stepsTitle == "Searched the web · 1 search")

        let second = service.requests[1]
        #expect(second.turns.first?.role == .system)
        #expect(second.turns.map(\.role) == [.system, .user, .assistant, .tool])
        #expect(second.turns[2].reasoning == "I should search.")
        #expect(second.options.tools.map(\.name) == ["web_search"])
    }

    @Test func deletingTheChatBeingAnsweredStopsItFirst() async throws {
        let reached = FakeEngineLatch()
        let service = Self.fakeService(FakeEngineScript(
            events: [.content("Partial"), .content(" never")],
            holdAfterEvents: 1, holdReached: reached, holdGate: FakeEngineLatch()))
        let (chat, container) = try Self.viewModel(service)
        chat.draft = "Question"
        chat.send()
        let id = try #require(chat.selectedID)
        try await reached.wait()
        await chat.delete(id)

        #expect(!chat.isGenerating)
        #expect(chat.selectedID == nil)
        #expect(chat.groups.groups.isEmpty)
        #expect(try Self.storedMessages(container).isEmpty)
        #expect(try ModelContext(container).fetch(FetchDescriptor<Conversation>()).isEmpty)
    }

    @Test func stoppingDuringToolsAnswersEveryCall() async throws {
        let calls = [
            ToolCall(id: "a", name: "web_search", arguments: .object([:]), argumentsJSON: "{}"),
            ToolCall(id: "b", name: "web_search", arguments: .object([:]), argumentsJSON: "{}"),
        ]
        let started = FakeEngineLatch()
        let service = ScriptedEngineService(replies: [.init(content: "", toolCalls: calls)])
        let (chat, container) = try Self.viewModel(service, tools: HangingToolbox(started: started))
        chat.draft = "Search twice"
        chat.send()
        try await started.wait()
        chat.stop()
        await chat.waitUntilIdle()

        let stored = try Self.storedMessages(container)
        #expect(stored.map(\.role) == [.user, .assistant, .tool, .tool])
        #expect(stored[2].toolCallID == "a")
        #expect(stored[3].toolCallID == "b")
        #expect(stored[3].content == "Stopped.")
        #expect(chat.errorMessage == nil)
    }

    @Test func webAsksForConsentOnce() throws {
        let tools = ConsentTools()
        let (chat, _) = try Self.viewModel(ScriptedEngineService(), tools: tools)
        chat.setWebEnabled(true)
        #expect(chat.webConsentRequested)
        #expect(!chat.webEnabled)
        chat.declineWebConsent()
        #expect(!chat.webConsentRequested)
        #expect(!chat.webEnabled)
        chat.setWebEnabled(true)
        chat.confirmWebConsent()
        #expect(tools.granted)
        #expect(chat.webEnabled)
        chat.setWebEnabled(false)
        chat.setWebEnabled(true)
        #expect(!chat.webConsentRequested)
        #expect(chat.webEnabled)
    }

    @Test func aNewChatStartsWithWebOff() async throws {
        let service = ScriptedEngineService(replies: [.init(content: "One"), .init(content: "Two")])
        let (chat, _) = try Self.viewModel(service, tools: StubToolbox())
        chat.setWebEnabled(true)
        chat.draft = "First"
        chat.send()
        await chat.waitUntilIdle()
        let first = try #require(chat.selectedID)

        for isPrivate in [false, true] {
            chat.newChat(isPrivate: isPrivate)
            #expect(!chat.webEnabled)
        }
        chat.draft = "Second, private"
        chat.send()
        await chat.waitUntilIdle()
        #expect(service.requests.last?.options.tools.isEmpty == true)
        #expect(chat.selectedConversation?.webEnabled == false)

        chat.select(first)
        #expect(chat.webEnabled)
    }

    @Test func answersKeepOnlyTheSourcesTheyCite() async throws {
        let call = ToolCall(id: "c", name: "web_search", arguments: .object([:]), argumentsJSON: "{}")
        let service = ScriptedEngineService(replies: [
            .init(content: "", toolCalls: [call]),
            .init(content: "No sources cited here."),
        ])
        let (chat, _) = try Self.viewModel(service, tools: StubToolbox())
        chat.setWebEnabled(true)
        chat.draft = "Q"
        chat.send()
        await chat.waitUntilIdle()
        #expect(chat.messages.last?.citations.isEmpty == true)
        #expect(chat.messages.last?.steps.count == 1)
    }

    @Test func toolsAreWithheldWhenWebIsOff() async throws {
        let service = ScriptedEngineService(replies: [.init(content: "Offline answer.")])
        let (chat, _) = try Self.viewModel(service, tools: StubToolbox())
        chat.draft = "Hi"
        chat.send()
        await chat.waitUntilIdle()
        #expect(service.requests[0].options.tools.isEmpty)
        #expect(service.requests[0].turns.map(\.role) == [.user])
    }

    @Test func toolRoundsAreBounded() async throws {
        let call = ToolCall(id: "c", name: "web_search", arguments: .object([:]), argumentsJSON: "{}")
        let service = ScriptedEngineService(fallback: .init(content: "", toolCalls: [call]))
        let (chat, _) = try Self.viewModel(service, tools: StubToolbox())
        chat.maxToolRounds = 2
        chat.setWebEnabled(true)
        chat.draft = "Loop"
        chat.send()
        await chat.waitUntilIdle()
        #expect(service.requests.count == 3)
        #expect(service.requests[2].options.tools.isEmpty)
    }

    @Test func modelChoiceIsPerChatAndNotInstalledIsRefused() async throws {
        let service = ScriptedEngineService(options: [
            ModelOption(model: .quick, availability: .ready),
            ModelOption(model: .deep, availability: .notInstalled(reason: "Not downloaded.")),
            ModelOption(model: .maple, availability: .ready),
        ])
        let (chat, _) = try Self.viewModel(service)
        chat.start()
        while chat.option(for: .maple)?.availability != .ready { await Task.yield() }
        chat.selectModel(.deep)
        #expect(chat.model == .quick)
        chat.selectModel(.maple)
        #expect(chat.model == .maple)
        chat.draft = "Hi"
        chat.send()
        await chat.waitUntilIdle()
        #expect(service.requests.last?.model == .maple)
        let first = try #require(chat.selectedID)

        chat.newChat()
        chat.selectModel(.quick)
        chat.draft = "Second"
        chat.send()
        await chat.waitUntilIdle()
        chat.select(first)
        #expect(chat.model == .maple)
        #expect(chat.messages.map(\.content) == ["Hi", "Hello from a scripted model."])
        chat.stopObserving()
    }
}
