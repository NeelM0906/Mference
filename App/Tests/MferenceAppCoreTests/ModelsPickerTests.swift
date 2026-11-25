import Foundation
import MferenceAppCore
import MferenceEngine
import Testing

@Suite struct ModelsPickerTests {
    @Test @MainActor func rowsAreGroupedAndShowInstallState() async {
        let engine = AppServices.fakeEngine(installed: [.quick, .qwen36, .maple])
        let picker = ModelPickerModel(engine: engine, selected: .quick)
        await picker.refresh()
        #expect(picker.recommended.map(\.model) == [.quick, .deep])
        #expect(picker.alsoOnThisMac.map(\.model) == [.bonsai27b, .inkling2Mini, .qwen36, .minicpm5, .maple])
        #expect(picker.row(for: .bonsai27b)?.statusText == "Not installed")
        #expect(picker.row(for: .quick)?.statusText == "Ready")
        #expect(picker.row(for: .quick)?.isSelected == true)
        #expect(picker.row(for: .deep)?.statusText == "Not installed")
        #expect(picker.row(for: .deep)?.isSelectable == false)
        #expect(ModelPickerModel.autoDetail.hasPrefix("Coming later"))
    }

    @Test @MainActor func pickingWarmsTheModelAndRowsFollowTheEngine() async throws {
        let started = FakeEngineLatch()
        let gate = FakeEngineLatch()
        let engine = AppServices.fakeEngine(scripts: [AppModel.deep.modelID: FakeEngineScript(loadStarted: started,
                                                                                               loadGate: gate)])
        let picker = ModelPickerModel(engine: engine, selected: .quick)
        await picker.refresh()
        let follow = Task { await picker.run() }
        defer { follow.cancel() }
        #expect(picker.select(.deep))
        try await started.wait()
        await waitUntil { picker.row(for: .deep)?.state == .loading }
        #expect(picker.isSelectedModelLoading)
        gate.open()
        await waitUntil { picker.row(for: .deep)?.state == .loaded }
        #expect(picker.row(for: .deep)?.isSelected == true)
        #expect(picker.row(for: .quick)?.isSelected == false)
        #expect(picker.isSelectedModelLoading == false)
    }

    @Test @MainActor func notInstalledCannotBePicked() async {
        let picker = ModelPickerModel(engine: AppServices.fakeEngine(installed: [.quick]), selected: .quick)
        await picker.refresh()
        #expect(picker.select(.deep) == false)
        #expect(picker.selected == .quick)
    }

    @Test @MainActor func nothingIsPickedBeforeTheRowsAreKnown() {
        let picker = ModelPickerModel(engine: AppServices.fakeEngine(), selected: .quick)
        #expect(picker.select(.maple) == false)
        #expect(picker.selected == .quick)
    }

    @Test @MainActor func supersededPicksAreNotLoaded() async throws {
        let started = FakeEngineLatch()
        let gate = FakeEngineLatch()
        let journal = FakeEngineJournal()
        let engine = AppServices.fakeEngine(scripts: [AppModel.qwen36.modelID: FakeEngineScript(loadStarted: started,
                                                                                                loadGate: gate)],
                                            journal: journal)
        let picker = ModelPickerModel(engine: engine, selected: .quick)
        await picker.refresh()
        picker.select(.qwen36)
        try await started.wait()
        // Two more picks while Qwen 3.6 loads; only the last one matters.
        picker.select(.maple)
        picker.select(.minicpm5)
        gate.open()
        await waitUntil { engine.status.residentModel == .minicpm5 }
        #expect(!journal.entries.contains(.loadStarted(AppModel.maple.modelID)))
        #expect(journal.maxLiveModels == 1)
    }

    @Test @MainActor func deepNeedsMoreThanSixteenGigabytes() async {
        let engine = AppServices.fakeEngine()
        let picker = ModelPickerModel(engine: engine, selected: .quick, physicalMemory: 16 << 30)
        await picker.refresh()
        #expect(picker.row(for: .deep)?.statusText == "Needs more than 16 GB")
        #expect(picker.select(.deep) == false)
        #expect(picker.row(for: .quick)?.isSelectable == true)
    }

    @Test @MainActor func failedWarmUpShowsTheMessage() async {
        struct Boom: Error, CustomStringConvertible { var description: String { "no memory" } }
        let engine = AppServices.fakeEngine(scripts: [AppModel.maple.modelID: FakeEngineScript(loadError: Boom())])
        let picker = ModelPickerModel(engine: engine, selected: .quick)
        await picker.refresh()
        picker.select(.maple)
        await waitUntil { picker.loadError != nil }
        #expect(picker.loadError == EngineServiceError.loadFailed("no memory").userMessage)
    }
}

/// Polls `condition` on the main actor, yielding between checks, for up to
/// five seconds.
@MainActor
func waitUntil(_ condition: () -> Bool) async {
    let deadline = ContinuousClock.now + .seconds(5)
    while !condition(), ContinuousClock.now < deadline {
        try? await Task.sleep(for: .milliseconds(5))
    }
    #expect(condition())
}
