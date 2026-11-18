import Foundation
import MferenceAppCore
@testable import MferenceAppUI
import Testing

/// Polls `condition` for up to five seconds.
func eventually(_ condition: () -> Bool) async -> Bool {
    let deadline = ContinuousClock.now + .seconds(5)
    while !condition(), ContinuousClock.now < deadline {
        try? await Task.sleep(for: .milliseconds(5))
    }
    return condition()
}

@Suite struct AdaptersTests {
    static func library(_ services: AppServices, memory: UInt64 = 32 << 30) -> ModelLibrary {
        ModelLibrary(engine: services.engine, downloads: services.downloads, settings: services.settings,
                     memoryFit: MemoryFit(physicalMemory: memory), chipName: "Apple M1 Max")
    }

    @Test func libraryEntriesCarryValuesOnly() async throws {
        let services = AppServices.preview(installed: [.quick, .qwen36])
        let library = Self.library(services)
        await library.refresh()
        let model = LiveModelLibraryModel(library: library, settings: services.settings,
                                          downloads: services.downloads, freeDisk: { 500_000_000_000 })
        let quick = try #require(model.entries.first { $0.id == AppModel.quick.modelID })
        #expect(quick.title == "Quick · Qwen3.5 4B")
        #expect(quick.state == .ready)
        #expect(quick.isRecommended)
        #expect(quick.engineName == "llama.cpp")
        #expect(quick.fit == .fits)
        let deep = try #require(model.entries.first { $0.id == AppModel.deep.modelID })
        #expect(deep.state == .notInstalled)
        #expect(deep.fit == .offerWithApproval)
        #expect(deep.sizeBytes == Int64(DownloadSource.deep.sizeBytes))
        #expect(deep.licence == "Apache 2.0")
        let maple = try #require(model.entries.first { $0.id == AppModel.maple.modelID })
        #expect(maple.title == "Maple")
        #expect(maple.isRecommended == false)
        #expect(model.machine.chipName == "Apple M1 Max")
        #expect(model.machine.canOfferDeep)
        #expect(model.machine.freeDiskBytes == 500_000_000_000)
    }

    @Test func loadedStateComesFromTheEngine() async throws {
        let services = AppServices.preview(installed: [.quick])
        let library = Self.library(services)
        try await services.engine.load(.quick)
        await library.refresh()
        let model = LiveModelLibraryModel(library: library, settings: services.settings, downloads: nil)
        #expect(model.entries.first { $0.id == AppModel.quick.modelID }?.state == .loaded)
    }

    @Test func deepIsTooLargeOnSixteenGigabytes() async throws {
        let services = AppServices.preview(installed: [.quick])
        let library = Self.library(services, memory: 16 << 30)
        await library.refresh()
        let model = LiveModelLibraryModel(library: library, settings: services.settings, downloads: nil)
        guard case .tooLarge? = model.entries.first(where: { $0.id == AppModel.deep.modelID })?.fit else {
            Issue.record("Deep must not be offered on 16 GB"); return
        }
        // Without an offer there is nothing to download.
        model.download(AppModel.deep.modelID)
        #expect(model.entries.first { $0.id == AppModel.deep.modelID }?.state == .notInstalled)
    }

    @Test func defaultModelAndIdleUnloadWriteSettings() async {
        let services = AppServices.preview()
        let model = LiveModelLibraryModel(library: Self.library(services), settings: services.settings, downloads: nil)
        model.defaultModelID = AppModel.deep.modelID
        #expect(services.settings.defaultModel == .deep)
        model.defaultModelID = "not-a-model"
        #expect(services.settings.defaultModel == .deep)
        model.idleUnloadMinutes = 30
        #expect(services.settings.idleUnloadMinutes == 30)
    }

    @Test func approvedDownloadGoesThroughTheManagerAndItsHostCheck() async throws {
        // The preview's download manager allows no host, so this proves the
        // approval path reaches DownloadManager without touching the network.
        let services = AppServices.preview(installed: [.quick])
        let library = Self.library(services)
        let follow = Task { await library.run() }
        defer { follow.cancel() }
        await library.refresh()
        let model = LiveModelLibraryModel(library: library, settings: services.settings,
                                          downloads: services.downloads)
        model.download(AppModel.deep.modelID)
        let failed = await eventually {
            if case .failed? = model.entries.first(where: { $0.id == AppModel.deep.modelID })?.state { true } else { false }
        }
        #expect(failed)
    }

    @Test func memoryListAddsEditsDeletes() async throws {
        let services = AppServices.preview()
        let memory = LiveMemoryListModel(store: MemoryStore(context: services.modelContainer.mainContext))
        #expect(memory.memories.isEmpty)
        memory.add("Prefers aisle seats")
        let entry = try #require(memory.memories.first)
        #expect(entry.text == "Prefers aisle seats")
        #expect(entry.source.hasPrefix("Added by you"))
        memory.update(entry.id, text: "Prefers window seats")
        #expect(memory.memories.map(\.text) == ["Prefers window seats"])
        memory.delete(entry.id)
        #expect(memory.memories.isEmpty)
        #expect(memory.lastError == nil)
    }

    @Test func memoryListFollowsSavesMadeElsewhere() async throws {
        let services = AppServices.preview()
        let context = services.modelContainer.mainContext
        let memory = LiveMemoryListModel(store: MemoryStore(context: context))
        _ = try MemoryStore(context: context).learn("Partner: Lena",
                                                   from: .chat(id: UUID(), title: "Lisbon", isPrivate: false))
        #expect(await eventually { memory.memories.map(\.text) == ["Partner: Lena"] })
    }

    @Test func settingsForwardAndConnectionsMapBrokerStates() async throws {
        let broker = FakePermissionBroker(states: [.calendar: .granted],
                                          answers: [.fullDiskAccess: .needsSystemSettings])
        let services = AppServices.preview(permissions: broker)
        let connections = BrokerConnections(broker: broker)
        await connections.refresh()
        let settings = LiveAppSettingsModel(settings: services.settings, connections: connections)

        #expect(settings.webSearchEnabled == false)
        settings.webSearchEnabled = true
        settings.offlineMode = true
        settings.useAppleModel = false
        #expect(services.settings.webSearchEnabled && services.settings.offlineMode)
        #expect(services.settings.useAppleModel == false)

        #expect(settings.connectionState(.appleCalendar) == .connected)
        #expect(settings.connectionState(.appleMail) == .notConnected)
        settings.connect(.appleMail)
        #expect(await eventually { settings.connectionState(.appleMail) == .needsSystemSettings })

        settings.remindersSyncEnabled = true
        #expect(services.settings.remindersSyncEnabled)
        #expect(await eventually { broker.calls.contains(.request(.reminders)) })
    }

    @Test func nothingIsRequestedUntilAsked() async {
        let broker = FakePermissionBroker()
        let connections = BrokerConnections(broker: broker)
        await connections.refresh()
        #expect(!broker.calls.contains { if case .request = $0 { true } else { false } })
        #expect(connections.connectionState(.appleCalendar) == .notConnected)
    }

    @Test func liveScreenModelsFillEverySlot() {
        let screens = ScreenModels.live(services: AppServices.preview())
        #expect(screens.library != nil)
        #expect(screens.memory != nil)
        #expect(screens.settings != nil)
    }
}
