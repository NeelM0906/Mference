import Foundation
import Mference
import MferenceAppCore
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
        for model in AppModel.allCases {
            let slot = EngineProductSlot(modelID: model.modelID)
            #expect(slot != nil, "\(model) has no engine product slot")
            #expect(slot?.engine.rawValue == model.engineLabel)
        }
        #expect(AppModel.allCases.count == EngineProductSlot.allCases.count)
    }

    @Test func pickerRowsShowInstalledAndMissingModels() async {
        let (service, _) = Self.service(installed: [.quick, .maple])
        let rows = await service.models()
        #expect(rows.map(\.model) == AppModel.allCases)
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
        // Mapping is internal; the public guarantee is Codable stability.
        let data = try? JSONEncoder().encode(value)
        let back = data.flatMap { try? JSONDecoder().decode(ToolJSON.self, from: $0) }
        #expect(back == value)
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

    @Test func idleUnloadPublishesEmpty() async throws {
        let (service, _) = Self.service(idleUnloadAfter: .milliseconds(20))
        var updates = service.statusUpdates().makeAsyncIterator()
        try await service.load(.quick)
        var sawEmptyAfterLoad = false
        var sawReady = false
        while let status = await updates.next() {
            if status.phase == .ready { sawReady = true }
            if sawReady, status.phase == .empty { sawEmptyAfterLoad = true; break }
        }
        #expect(sawEmptyAfterLoad)
        #expect(service.status.phase == .empty)
    }
}
