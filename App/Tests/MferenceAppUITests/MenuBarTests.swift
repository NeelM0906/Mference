import Foundation
import MferenceAppCore
@testable import MferenceAppUI
import Testing

@Suite struct MenuBarTests {
    static let memory32: UInt64 = 32 << 30
    static let memory16: UInt64 = 16 << 30

    static let options: [ModelOption] = [
        ModelOption(model: .quick, availability: .loaded, sizeBytes: 2_910_000_000),
        ModelOption(model: .deep, availability: .ready, sizeBytes: 12_120_000_000),
        ModelOption(model: .qwen36, availability: .ready, sizeBytes: 19_550_000_000),
        ModelOption(model: .minicpm5, availability: .ready, sizeBytes: 1_430_000_000),
        ModelOption(model: .maple, availability: .notInstalled(reason: "Not installed.")),
    ]

    static func summary(_ status: EngineServiceStatus, memory: UInt64 = memory32, idle: Int = 10,
                        entries: [ModelEntry] = []) -> MenuBarSummary {
        MenuBarSummary(status: status, options: options, entries: entries, idleUnloadMinutes: idle,
                       physicalMemory: memory)
    }

    @Test func loadedShowsModelMemoryAndIdleUnload() {
        let summary = Self.summary(EngineServiceStatus(phase: .ready, residentModel: .quick))
        #expect(summary.headline == "Quick 4B · Loaded")
        #expect(summary.details == ["Idle", "About 4.5 GB in memory · unloads after 10 min idle"])
        #expect(summary.unloadDisabledReason == nil)
        #expect(summary.switchDisabledReason == nil)
        #expect(summary.symbol == "cpu.fill")
        let never = Self.summary(EngineServiceStatus(phase: .ready, residentModel: .quick), idle: 0)
        #expect(never.details.last == "About 4.5 GB in memory · stays loaded")
    }

    @Test func answeringCountsWaitingRequests() {
        let summary = Self.summary(EngineServiceStatus(phase: .ready, residentModel: .quick,
                                                       isGenerating: true, queuedRequests: 2))
        #expect(summary.details.first == "Answering · 2 requests waiting")
        #expect(summary.symbol == "ellipsis.bubble.fill")
    }

    /// Loads cannot be interrupted, so Unload and switching wait, with the reason.
    @Test func loadingDisablesUnloadAndSwitchWithTheReason() {
        let summary = Self.summary(EngineServiceStatus(phase: .loading, targetModel: .deep, queuedRequests: 1))
        #expect(summary.headline == "Loading Deep 27B")
        #expect(summary.details == ["About 14.8 GB in memory when loaded", "1 request waiting, then answers"])
        #expect(summary.unloadDisabledReason == "Unload after this load")
        #expect(summary.switchDisabledReason == "Switch after this load")
        #expect(summary.modelRows.allSatisfy { !$0.isEnabled })
    }

    @Test func emptyAndUnloading() {
        let empty = Self.summary(.empty)
        #expect(empty.headline == "No model in memory")
        #expect(empty.unloadDisabledReason == "No model in memory")
        #expect(empty.switchDisabledReason == nil)
        let unloading = Self.summary(EngineServiceStatus(phase: .unloading, targetModel: .quick))
        #expect(unloading.headline == "Unloading Quick 4B")
        #expect(unloading.unloadDisabledReason == "Unloading now")
    }

    @Test func failedLoadOffersRetry() {
        let summary = Self.summary(EngineServiceStatus(phase: .empty, lastLoadFailure: "Out of memory.",
                                                       lastLoadFailureModel: .deep))
        #expect(summary.details.last == "Deep 27B didn't load: Out of memory.")
        #expect(summary.retryModel == .deep)
        #expect(summary.symbol == "exclamationmark.triangle")
    }

    @Test func modelRowsShowFitAndInstallState() {
        let rows = Self.summary(EngineServiceStatus(phase: .ready, residentModel: .quick)).modelRows
        func row(_ model: AppModel) -> MenuBarSummary.ModelRow? { rows.first { $0.model == model } }
        #expect(row(.quick)?.detail == "Loaded · 4.5 GB")
        #expect(row(.quick)?.isResident == true)
        #expect(row(.quick)?.isEnabled == false, "already resident")
        #expect(row(.deep)?.detail == "Fits · 14.8 GB")
        #expect(row(.qwen36)?.detail == "Fits · 8.0 GB")
        #expect(row(.maple)?.detail == "Not installed")
        #expect(row(.maple)?.isEnabled == false)

        let small = Self.summary(EngineServiceStatus(phase: .ready, residentModel: .quick), memory: Self.memory16)
            .modelRows.first { $0.model == .deep }
        #expect(small?.detail == "Too large for this Mac")
        #expect(small?.isEnabled == false, "no Deep on a 16 GB Mac (L4)")
    }

    @Test func downloadsListRunningPausedAndFailed() {
        let entries = FixtureModelLibrary(quick: .paused(TransferProgress(completedBytes: 1_000_000_000,
                                                                          totalBytes: 2_900_000_000)),
                                          deep: .downloading(TransferProgress(completedBytes: 4_970_000_000,
                                                                              totalBytes: 12_120_000_000))).entries
        let summary = Self.summary(.empty, entries: entries)
        #expect(summary.downloads == ["Quick 4B paused · 1.0 of 2.9 GB", "Deep 27B · 41% · 5.0 of 12.1 GB"])
        let failed = FixtureModelLibrary(quick: .downloadFailed("The network went away."), deep: .ready).entries
        #expect(Self.summary(.empty, entries: failed).downloads == ["Quick 4B didn't download: The network went away."])
    }

    /// The model follows the engine's status and asks the engine to unload;
    /// it never loads anything on its own.
    @Test func modelFollowsTheEngineAndUnloads() async throws {
        let engine = ScriptedEngineService(status: EngineServiceStatus(phase: .ready, residentModel: .quick))
        let model = MenuBarModel(engine: engine, physicalMemory: Self.memory32)
        let task = Task { await model.run() }
        defer { task.cancel() }
        try await waitUntil { model.options.count == AppModel.engineModels.count }
        #expect(model.summary(entries: [], idleUnloadMinutes: 10).headline == "Quick 4B · Loaded")
        #expect(engine.requests.isEmpty)

        engine.setStatus(EngineServiceStatus(phase: .loading, targetModel: .deep))
        try await waitUntil { model.status.phase == .loading }
        #expect(model.options.first { $0.model == .deep }?.availability == .loading)

        engine.setStatus(EngineServiceStatus(phase: .ready, residentModel: .deep))
        model.unload()
        try await waitUntil { model.status.phase == .empty }
        #expect(model.summary(entries: [], idleUnloadMinutes: 10).headline == "No model in memory")
    }

    private func waitUntil(_ condition: @escaping () -> Bool) async throws {
        for _ in 0..<200 {
            if condition() { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        Issue.record("condition not met")
    }
}
