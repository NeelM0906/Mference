import Foundation
import Synchronization
import Mference
@testable import MferenceAppCore
import MferenceEngine
import Testing

@Suite struct EngineServiceTests {
    static func service(scripts: [String: FakeEngineScript] = [:],
                        installed: [AppModel] = AppModel.allCases,
                        idleUnloadAfter: Duration? = nil) -> (EngineService, FakeEngineJournal) {
        let journal = FakeEngineJournal()
        let native = FakeEngineBackend(engine: .mference, journal: journal, scripts: scripts)
        let llama = FakeEngineBackend(engine: .llamaCpp, journal: journal, scripts: scripts)
        let catalog = EngineCatalog(models: installed.map { model in
            FakeEngineBackend.descriptor(id: model.modelID,
                                         engine: model.engineLabel == "llama.cpp" ? .llamaCpp : .mference,
                                         product: EngineProductSlot(modelID: model.modelID))
        })
        let service = EngineService(discover: { catalog }) { catalog, observer in
            EngineHost(catalog: catalog, backends: [native, llama], idleUnloadAfter: idleUnloadAfter,
                       observer: observer)
        }
        return (service, journal)
    }

    @Test func appModelsMatchEngineProductSlots() {
        // Apple's on-device model is chat-only and never an engine slot.
        #expect(EngineProductSlot(modelID: AppModel.apple.modelID) == nil)
        for model in AppModel.engineModels {
            let slot = EngineProductSlot(modelID: model.modelID)
            #expect(slot != nil, "\(model) has no engine product slot")
            let expected: EngineKind = model.engineLabel == "llama.cpp" ? .llamaCpp : .mference
            #expect(slot?.engine == expected)
        }
        #expect(AppModel.engineModels.count == EngineProductSlot.allCases.count)
    }

    @Test func pickerRowsShowInstalledAndMissingModels() async {
        let (service, _) = Self.service(installed: [.quick, .maple])
        let rows = await service.models()
        #expect(rows.map(\.model) == AppModel.engineModels)
        #expect(rows.first { $0.model == .quick }?.availability == .ready)
        if case .notInstalled = rows.first(where: { $0.model == .deep })?.availability {} else {
            Issue.record("Deep should be not installed")
        }
    }

    @Test func streamsContentReasoningAndCompletion() async throws {
        let script = FakeEngineScript(events: [.reasoning("think"), .content("Hello"), .content(" world")])
        let (service, _) = Self.service(scripts: [AppModel.quick.modelID: script])
        var updates: [GenerationUpdate] = []
        for try await update in service.generate(model: .quick, turns: [ChatTurn(role: .user, content: "hi")],
                                                 options: GenerationOptions()) {
            updates.append(update)
        }
        guard case .prepared = updates.first else { Issue.record("first update is not prepared"); return }
        #expect(updates.contains(.reasoning("think")))
        #expect(updates.contains(.content("Hello")))
        guard case .completed(let summary) = updates.last else { Issue.record("no completion"); return }
        #expect(summary.content == "Hello world")
        #expect(service.status.residentModel == .quick)
        #expect(service.status.phase == .ready)
    }

    @Test func loadReportsLoadedAndSwapReleasesFirst() async throws {
        let (service, journal) = Self.service()
        try await service.load(.quick)
        #expect(await service.models().first { $0.model == .quick }?.availability == .loaded)
        try await service.load(.maple)
        #expect(service.status.residentModel == .maple)
        #expect(journal.maxLiveModels == 1)
    }

    @Test func unknownModelMapsToUserMessage() async {
        let (service, _) = Self.service(installed: [.quick])
        await #expect(throws: EngineServiceError.self) {
            try await service.load(.deep)
        }
    }

    @Test func toolJSONRoundTripsThroughTheEngine() {
        let value: ToolJSON = .object(["q": .string("desk"), "n": .integer(3), "x": .number(1.5),
                                       "ok": .bool(true), "list": .array([.null])])
        #expect(EngineService.appJSON(EngineService.engineJSON(value)) == value)
        #expect(EngineService.appJSON(.unsignedInteger(7)) == .integer(7))
    }

    @Test func historyKeepsReasoningAndToolCalls() {
        let call = ToolCall(id: "c1", name: "web_search", arguments: .object(["query": .string("desk")]),
                            argumentsJSON: #"{"query":"desk"}"#)
        let message = EngineService.message(ChatTurn(role: .assistant, content: nil, reasoning: "plan",
                                                     toolCalls: [call]))
        #expect(message.role == .assistant)
        #expect(message.reasoningContent == "plan")
        #expect(message.toolCalls.first?.name == "web_search")
        let result = EngineService.message(ChatTurn(role: .tool, content: "[]", toolCallID: "c1", toolName: "web_search"))
        #expect(result.toolCallID == "c1")
        #expect(result.name == "web_search")
    }

    @Test func contextOverflowAndOtherInvalidRequestsMapApart() {
        let tooLong = EngineService.map(EngineError.invalidRequest(message: "m", param: nil, code: "context_length_exceeded"))
        let other = EngineService.map(EngineError.invalidRequest(message: "m", param: nil, code: "invalid_chat_template"))
        #expect(tooLong as? EngineServiceError == .requestTooLarge("m"))
        #expect(other as? EngineServiceError == .invalidRequest("m"))
    }

    @Test func refreshPicksUpANewInstallWhileAModelIsLoaded() async throws {
        let journal = FakeEngineJournal()
        let backend = FakeEngineBackend(engine: .llamaCpp, journal: journal)
        let installed = Mutex<[AppModel]>([.quick])
        let service = EngineService(discover: {
            EngineCatalog(models: installed.withLock { $0 }.map {
                FakeEngineBackend.descriptor(id: $0.modelID, engine: .llamaCpp)
            })
        }) { catalog, observer in
            EngineHost(catalog: catalog, backends: [backend], observer: observer)
        }
        try await service.load(.quick)
        installed.withLock { $0.append(.deep) }
        await service.refreshCatalog()
        #expect(await service.models().first { $0.model == .deep }?.availability == .ready)
        #expect(journal.maxLiveModels == 1)
        try await service.load(.deep)
        #expect(service.status.residentModel == .deep)
    }

    @Test func droppingTheIteratorCancelsAndFreesTheEngine() async throws {
        let reached = FakeEngineLatch()
        let gate = FakeEngineLatch()
        let held = FakeEngineScript(events: [.content("a"), .content("b")], holdAfterEvents: 1,
                                    holdReached: reached, holdGate: gate)
        let (service, journal) = Self.service(scripts: [AppModel.quick.modelID: held])
        let stream = service.generate(model: .quick, turns: [ChatTurn(role: .user, content: "hi")],
                                      options: GenerationOptions())
        // Iterate in a scope and break; `stream` itself stays alive.
        do {
            for try await update in stream {
                if case .content = update { break }
            }
        }
        try await reached.wait()
        // The request was cancelled, so a second one gets the engine.
        var finished = false
        for try await update in service.generate(model: .maple, turns: [ChatTurn(role: .user, content: "x")],
                                                 options: GenerationOptions()) {
            if case .completed = update { finished = true }
        }
        #expect(finished)
        #expect(journal.entries.contains(.generationCancelled(AppModel.quick.modelID)))
        withExtendedLifetime(stream) {}
    }

    /// Driven explicitly: the host's idle clock is a `TestSleeper` fired by
    /// the test, and only published statuses are asserted (the synchronous
    /// snapshot may lag a published `.empty` by one host step).
    @Test func idleUnloadPublishesEmpty() async throws {
        let sleeper = TestSleeper()
        let harness = Harness(idleUnload: IdleUnloadControl(delay: .seconds(600), sleep: sleeper.sleep))
        try await harness.service.load(.quick)
        #expect(await waitForStatus(harness.service) { $0.phase == .ready && $0.residentModel == .quick } != nil)
        // The idle timer is armed; nothing unloads until it fires.
        while sleeper.calls == 0 { await Task.yield() }
        #expect(!harness.journal.entries.contains(.unloaded(AppModel.quick.modelID)))
        sleeper.fire()
        let empty = await waitForStatus(harness.service) { $0.phase == .empty }
        #expect(empty?.residentModel == nil)
        #expect(empty?.targetModel == nil)
        #expect(harness.journal.entries.contains(.unloaded(AppModel.quick.modelID)))
    }
}
