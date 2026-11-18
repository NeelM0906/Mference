import Foundation
import Mference
import MferenceAppCore
import MferenceEngine
import Synchronization
import Testing

/// A sleeper the test releases: each call waits for the current latch.
final class TestSleeper: Sendable {
    private let state = Mutex<(latch: FakeEngineLatch, calls: Int)>((FakeEngineLatch(), 0))

    var calls: Int { state.withLock { $0.calls } }

    /// Ends every sleep running now; later sleeps wait for the next fire.
    func fire() {
        let latch = state.withLock { s -> FakeEngineLatch in
            defer { s.latch = FakeEngineLatch() }
            return s.latch
        }
        latch.open()
    }

    var sleep: IdleUnloadControl.Sleep {
        { [self] _ in
            let latch = state.withLock { s -> FakeEngineLatch in
                s.calls += 1
                return s.latch
            }
            try await latch.wait()
        }
    }
}

/// Mutable discovery for refresh tests.
final class TestDiscovery: Sendable {
    private let state: Mutex<(installed: [AppModel], runs: Int, hold: DispatchSemaphore?, started: DispatchSemaphore?)>

    init(_ installed: [AppModel]) { state = Mutex((installed, 0, nil, nil)) }

    var runs: Int { state.withLock { $0.runs } }
    func install(_ model: AppModel) { state.withLock { $0.installed.append(model) } }

    /// The next run reads the installs, signals `started`, then blocks until
    /// `hold` is signalled before returning what it read.
    func holdNextRun(started: DispatchSemaphore, hold: DispatchSemaphore) {
        state.withLock { $0.hold = hold; $0.started = started }
    }

    func discover() -> EngineCatalog {
        let (installed, hold, started) = state.withLock { s -> ([AppModel], DispatchSemaphore?, DispatchSemaphore?) in
            s.runs += 1
            defer { s.hold = nil; s.started = nil }
            return (s.installed, s.hold, s.started)
        }
        started?.signal()
        hold?.wait()
        return AppServices.fakeCatalog(installed)
    }
}

struct Harness {
    let service: EngineService
    let journal: FakeEngineJournal
    let native: FakeEngineBackend
    let llama: FakeEngineBackend
    let counter: HostCounter

    init(installed: [AppModel] = AppModel.allCases,
         scripts: [String: FakeEngineScript] = [:],
         discovery: TestDiscovery? = nil,
         idleUnload: IdleUnloadControl = IdleUnloadControl(delay: nil),
         maxQueuedRequests: Int? = nil,
         physicalMemory: UInt64 = 32 << 30) {
        let journal = FakeEngineJournal()
        let native = FakeEngineBackend(engine: .mference, journal: journal, scripts: scripts)
        let llama = FakeEngineBackend(engine: .llamaCpp, journal: journal, scripts: scripts)
        let discovery = discovery ?? TestDiscovery(installed)
        let counter = HostCounter()
        self.service = EngineService(discover: { discovery.discover() }, makeHost: { catalog, observer in
            counter.increment()
            return EngineHost(catalog: catalog, backends: [native, llama],
                              idleUnloadAfter: EngineService.defaultIdleUnload,
                              maxQueuedRequests: maxQueuedRequests,
                              sleep: idleUnload.sleeper, observer: observer)
        }, physicalMemory: physicalMemory)
        self.journal = journal
        self.native = native
        self.llama = llama
        self.counter = counter
    }
}

final class HostCounter: Sendable {
    private let value = Mutex(0)
    var count: Int { value.withLock { $0 } }
    func increment() { value.withLock { $0 += 1 } }
}

/// Waits (up to a few seconds) for a status matching `predicate` on a fresh
/// subscription; returns it, or nil on timeout.
func waitForStatus(_ service: EngineService,
                   _ predicate: @escaping @Sendable (EngineServiceStatus) -> Bool) async -> EngineServiceStatus? {
    await withTaskGroup(of: EngineServiceStatus?.self) { group in
        group.addTask {
            for await status in service.statusUpdates() where predicate(status) { return status }
            return nil
        }
        group.addTask {
            try? await Task.sleep(for: .seconds(5))
            return nil
        }
        let first = await group.next() ?? nil
        group.cancelAll()
        return first
    }
}

func collect(_ stream: GenerationStream) async throws -> [GenerationUpdate] {
    var updates: [GenerationUpdate] = []
    for try await update in stream { updates.append(update) }
    return updates
}

let hello = [ChatTurn(role: .user, content: "hi")]

@Suite struct EngineServiceHardeningTests {
    // MARK: Discovery

    @Test func discoveryWaitsForFirstUseAndRunsOnce() async {
        let discovery = TestDiscovery([.quick])
        let harness = Harness(discovery: discovery)
        #expect(discovery.runs == 0)
        #expect(harness.service.status == .empty)
        async let first = harness.service.models()
        async let second = harness.service.models()
        _ = await (first, second)
        #expect(discovery.runs == 1)
        #expect(harness.counter.count == 0, "listing models does not build a host")
    }

    // MARK: Status under swaps

    @Test func statusFollowsASwapReleasingFirst() async throws {
        let started = FakeEngineLatch()
        let gate = FakeEngineLatch()
        let harness = Harness(scripts: [AppModel.maple.modelID: FakeEngineScript(loadStarted: started,
                                                                                  loadGate: gate)])
        try await harness.service.load(.quick)
        #expect(harness.service.status.residentModel == .quick)
        let swap = Task { try await harness.service.load(.maple) }
        try await started.wait()
        let loading = await waitForStatus(harness.service) { $0.phase == .loading }
        #expect(loading?.targetModel == .maple)
        #expect(loading?.residentModel == nil)
        let rows = await harness.service.models()
        #expect(rows.first { $0.model == .quick }?.availability == .ready)
        #expect(rows.first { $0.model == .maple }?.availability == .loading)
        gate.open()
        try await swap.value
        let ready = await waitForStatus(harness.service) { $0.phase == .ready }
        #expect(ready?.residentModel == .maple)
        // Never two models resident, and Quick released before Maple loaded.
        #expect(harness.journal.maxLiveModels == 1)
        let entries = harness.journal.entries
        let unloadQuick = entries.firstIndex(of: .unloaded(AppModel.quick.modelID))
        let loadMaple = entries.firstIndex(of: .loadStarted(AppModel.maple.modelID))
        #expect(unloadQuick != nil && loadMaple != nil && unloadQuick! < loadMaple!)
    }

    @Test func loadingIsVisibleAndNotCancellable() async throws {
        let started = FakeEngineLatch()
        let gate = FakeEngineLatch()
        let harness = Harness(scripts: [AppModel.deep.modelID: FakeEngineScript(loadStarted: started, loadGate: gate)])
        let load = Task { try await harness.service.load(.deep) }
        try await started.wait()
        let loading = await waitForStatus(harness.service) { $0.phase == .loading }
        #expect(loading?.targetModel == .deep)
        #expect(await harness.service.models().first { $0.model == .deep }?.availability == .loading)
        // Cancelling the caller does not abandon the load.
        load.cancel()
        gate.open()
        _ = try? await load.value
        let ready = await waitForStatus(harness.service) { $0.phase == .ready }
        #expect(ready?.residentModel == .deep)
    }

    // MARK: Idle unload

    @Test func idleUnloadUsesTheControlAndPublishesEmpty() async throws {
        let sleeper = TestSleeper()
        let control = IdleUnloadControl(delay: .seconds(600), sleep: sleeper.sleep)
        let harness = Harness(idleUnload: control)
        _ = try await collect(harness.service.generate(model: .quick, turns: hello, options: GenerationOptions()))
        #expect(harness.service.status.residentModel == .quick)
        // The timer is running; nothing unloads until it fires.
        while sleeper.calls == 0 { await Task.yield() }
        #expect(harness.journal.entries.contains(.unloaded(AppModel.quick.modelID)) == false)
        sleeper.fire()
        let empty = await waitForStatus(harness.service) { $0.phase == .empty }
        #expect(empty != nil)
        #expect(empty?.targetModel == nil)
        #expect(harness.journal.entries.contains(.unloaded(AppModel.quick.modelID)))
    }

    @Test func idleUnloadNeverKeepsTheModelUntilTheSettingChanges() async throws {
        let sleeper = TestSleeper()
        let control = IdleUnloadControl(delay: nil, sleep: sleeper.sleep)
        let harness = Harness(idleUnload: control)
        try await harness.service.load(.minicpm5)
        // Firing the clock does nothing while the setting is "never".
        sleeper.fire()
        for _ in 0..<50 { await Task.yield() }
        #expect(sleeper.calls == 0)
        #expect(harness.service.status.residentModel == .minicpm5)
        control.setDelay(.seconds(300))
        while sleeper.calls == 0 { await Task.yield() }
        sleeper.fire()
        #expect(await waitForStatus(harness.service) { $0.phase == .empty } != nil)
    }

    @Test func activityRestartsTheIdleTimer() async throws {
        let sleeper = TestSleeper()
        let control = IdleUnloadControl(delay: .seconds(600), sleep: sleeper.sleep)
        let harness = Harness(idleUnload: control)
        _ = try await collect(harness.service.generate(model: .quick, turns: hello, options: GenerationOptions()))
        while sleeper.calls == 0 { await Task.yield() }
        // A second request cancels the first timer and starts a new one.
        _ = try await collect(harness.service.generate(model: .quick, turns: hello, options: GenerationOptions()))
        while sleeper.calls < 2 { await Task.yield() }
        #expect(harness.service.status.residentModel == .quick)
        #expect(harness.journal.entries.filter { $0 == .loadStarted(AppModel.quick.modelID) }.count == 1)
    }

    // MARK: One generation at a time

    @Test func secondRequestQueuesAndRunsAfterTheFirst() async throws {
        let reached = FakeEngineLatch()
        let gate = FakeEngineLatch()
        let held = FakeEngineScript(events: [.content("a"), .content("b")], holdAfterEvents: 1,
                                    holdReached: reached, holdGate: gate)
        let harness = Harness(scripts: [AppModel.quick.modelID: held])
        let first = Task { try await collect(harness.service.generate(model: .quick, turns: hello,
                                                                      options: GenerationOptions())) }
        try await reached.wait()
        let second = Task { try await collect(harness.service.generate(model: .quick, turns: hello,
                                                                       options: GenerationOptions())) }
        let queued = await waitForStatus(harness.service) { $0.queuedRequests == 1 }
        #expect(queued?.isGenerating == true)
        gate.open()
        let a = try await first.value
        let b = try await second.value
        #expect(a.last.map { if case .completed = $0 { true } else { false } } == true)
        #expect(b.last.map { if case .completed = $0 { true } else { false } } == true)
        #expect(harness.journal.maxConcurrentGenerations == 1)
        let idle = await waitForStatus(harness.service) { !$0.isGenerating && $0.queuedRequests == 0 }
        #expect(idle != nil)
    }

    @Test func fullQueueMapsToBusy() async throws {
        let reached = FakeEngineLatch()
        let gate = FakeEngineLatch()
        let held = FakeEngineScript(holdAfterEvents: 0, holdReached: reached, holdGate: gate)
        let harness = Harness(scripts: [AppModel.quick.modelID: held], maxQueuedRequests: 0)
        let first = Task { try await collect(harness.service.generate(model: .quick, turns: hello,
                                                                      options: GenerationOptions())) }
        try await reached.wait()
        await #expect(throws: EngineServiceError.busy) {
            _ = try await collect(harness.service.generate(model: .quick, turns: hello, options: GenerationOptions()))
        }
        gate.open()
        _ = try await first.value
    }

    // MARK: Cancellation

    @Test func cancelActiveGenerationStopsWithCancellationError() async throws {
        let reached = FakeEngineLatch()
        let held = FakeEngineScript(events: [.content("a"), .content("b")], holdAfterEvents: 1,
                                    holdReached: reached, holdGate: FakeEngineLatch())
        let harness = Harness(scripts: [AppModel.quick.modelID: held])
        let run = Task { try await collect(harness.service.generate(model: .quick, turns: hello,
                                                                    options: GenerationOptions())) }
        try await reached.wait()
        await harness.service.cancelActiveGeneration()
        await #expect(throws: CancellationError.self) { _ = try await run.value }
        #expect(harness.journal.entries.contains(.generationCancelled(AppModel.quick.modelID)))
        // The model stays resident for the next message.
        #expect(harness.service.status.residentModel == .quick)
    }

    @Test func cancellingTheConsumingTaskThrowsInsteadOfEndingQuietly() async throws {
        let reached = FakeEngineLatch()
        let held = FakeEngineScript(events: [.content("a"), .content("b")], holdAfterEvents: 1,
                                    holdReached: reached, holdGate: FakeEngineLatch())
        let harness = Harness(scripts: [AppModel.quick.modelID: held])
        let run = Task { try await collect(harness.service.generate(model: .quick, turns: hello,
                                                                    options: GenerationOptions())) }
        try await reached.wait()
        run.cancel()
        await #expect(throws: CancellationError.self) { _ = try await run.value }
    }

    // MARK: Refresh after a download

    @Test func refreshWhileEmptyPicksUpADownloadAtOnce() async throws {
        let discovery = TestDiscovery([.quick])
        let harness = Harness(discovery: discovery)
        try await harness.service.load(.quick)
        try await harness.service.unload()
        guard case .notInstalled = await harness.service.models().first(where: { $0.model == .deep })?.availability
        else { Issue.record("Deep should start missing"); return }
        discovery.install(.deep)
        await harness.service.refreshCatalog()
        #expect(await harness.service.models().first { $0.model == .deep }?.availability == .ready)
        #expect(harness.counter.count == 2)
        try await harness.service.load(.deep)
        #expect(harness.service.status.residentModel == .deep)
    }

    @Test func refreshWhileLoadedSwapsHostsOnlyWhenNeeded() async throws {
        let discovery = TestDiscovery([.quick])
        let harness = Harness(discovery: discovery)
        try await harness.service.load(.quick)
        discovery.install(.deep)
        await harness.service.refreshCatalog()
        // The row shows at once; the loaded host is kept for now.
        #expect(await harness.service.models().first { $0.model == .deep }?.availability == .ready)
        #expect(await harness.service.models().first { $0.model == .quick }?.availability == .loaded)
        #expect(harness.counter.count == 1)
        // Quick is served by the old host without a reload.
        _ = try await collect(harness.service.generate(model: .quick, turns: hello, options: GenerationOptions()))
        #expect(harness.journal.entries.filter { $0 == .loadStarted(AppModel.quick.modelID) }.count == 1)
        // Deep needs the new host: the old one releases Quick first.
        let updates = try await collect(harness.service.generate(model: .deep, turns: hello,
                                                                 options: GenerationOptions()))
        guard case .completed = updates.last else { Issue.record("Deep did not complete"); return }
        #expect(harness.counter.count == 2)
        #expect(harness.journal.maxLiveModels == 1)
        #expect(harness.service.status.residentModel == .deep)
        // Back to the old model: the new host swaps Deep out for Quick, and
        // no third host is built.
        let back = try await collect(harness.service.generate(model: .quick, turns: hello,
                                                              options: GenerationOptions()))
        guard case .completed = back.last else { Issue.record("Quick did not complete"); return }
        #expect(harness.counter.count == 2)
        #expect(harness.service.status.residentModel == .quick)
        // Never more than one model across the old and the new host.
        #expect(harness.journal.maxLiveModels == 1)
        #expect(harness.journal.liveModels == 1)
    }

    @Test func replacementWaitsForARequestHoldingTheOldHost() async throws {
        let reached = FakeEngineLatch()
        let gate = FakeEngineLatch()
        let held = FakeEngineScript(events: [.content("a"), .content("b")], holdAfterEvents: 1,
                                    holdReached: reached, holdGate: gate)
        let discovery = TestDiscovery([.quick])
        let harness = Harness(scripts: [AppModel.quick.modelID: held], discovery: discovery)
        // Quick holds the old host mid-answer.
        let quick = Task { try await collect(harness.service.generate(model: .quick, turns: hello,
                                                                      options: GenerationOptions())) }
        try await reached.wait()
        discovery.install(.deep)
        await harness.service.refreshCatalog()
        // Deep needs the new host; the replacement must wait for Quick.
        let deep = Task { try await collect(harness.service.generate(model: .deep, turns: hello,
                                                                     options: GenerationOptions())) }
        for _ in 0..<200 { await Task.yield() }
        #expect(harness.counter.count == 1, "no new host while Quick still holds the old one")
        #expect(!harness.journal.entries.contains(.generationCancelled(AppModel.quick.modelID)))
        gate.open()
        let q = try await quick.value
        let d = try await deep.value
        guard case .completed = q.last, case .completed = d.last else {
            Issue.record("a request did not complete"); return
        }
        #expect(!harness.journal.entries.contains(.generationCancelled(AppModel.quick.modelID)))
        #expect(harness.counter.count == 2)
        #expect(harness.journal.maxLiveModels == 1)
        #expect(harness.journal.maxConcurrentGenerations == 1)
    }

    @Test func shutdownIsNotBlockedByADrainingReplacement() async throws {
        let reached = FakeEngineLatch()
        let held = FakeEngineScript(events: [.content("a"), .content("b")], holdAfterEvents: 1,
                                    holdReached: reached, holdGate: FakeEngineLatch())
        let discovery = TestDiscovery([.quick])
        let harness = Harness(scripts: [AppModel.quick.modelID: held], discovery: discovery)
        let quick = Task { try await collect(harness.service.generate(model: .quick, turns: hello,
                                                                      options: GenerationOptions())) }
        try await reached.wait()
        discovery.install(.deep)
        await harness.service.refreshCatalog()
        let deep = Task { try await collect(harness.service.generate(model: .deep, turns: hello,
                                                                     options: GenerationOptions())) }
        for _ in 0..<200 { await Task.yield() }
        // Quick never resumes on its own; quitting must still finish.
        await harness.service.shutdown()
        await #expect(throws: CancellationError.self) { _ = try await quick.value }
        await #expect(throws: (any Error).self) { _ = try await deep.value }
        #expect(harness.journal.liveModels == 0)
    }

    @Test func overlappingRefreshesKeepTheNewestCatalog() async throws {
        let discovery = TestDiscovery([.quick])
        let harness = Harness(discovery: discovery)
        _ = await harness.service.models()
        // The first refresh reads "Quick only" and stalls.
        let started = DispatchSemaphore(value: 0)
        let hold = DispatchSemaphore(value: 0)
        discovery.holdNextRun(started: started, hold: hold)
        let first = Task { await harness.service.refreshCatalog() }
        await withCheckedContinuation { continuation in
            DispatchQueue.global().async { started.wait(); continuation.resume() }
        }
        // Deep lands; a second refresh sees it and finishes first.
        discovery.install(.deep)
        await harness.service.refreshCatalog()
        #expect(await harness.service.models().first { $0.model == .deep }?.availability == .ready)
        hold.signal()
        await first.value
        #expect(await harness.service.models().first { $0.model == .deep }?.availability == .ready)
    }

    @Test func catalogChangesBumpTheRevision() async throws {
        let discovery = TestDiscovery([.quick])
        let harness = Harness(discovery: discovery)
        _ = await harness.service.models()
        let before = harness.service.status.catalogRevision
        await harness.service.refreshCatalog()
        #expect(harness.service.status.catalogRevision == before, "same catalog, same revision")
        discovery.install(.deep)
        let bumped = Task { await waitForStatus(harness.service) { $0.catalogRevision > before } }
        await harness.service.refreshCatalog()
        #expect(await bumped.value != nil)
    }

    @Test func deepCannotRunOnSixteenGigabytes() async throws {
        let harness = Harness(physicalMemory: 16 << 30)
        guard case .notInstalled(let reason)? = await harness.service.models()
            .first(where: { $0.model == .deep })?.availability else {
            Issue.record("Deep must not be offered on 16 GB"); return
        }
        #expect(reason.contains("16 GB"))
        await #expect(throws: EngineServiceError.self) { try await harness.service.load(.deep) }
        await #expect(throws: EngineServiceError.self) {
            _ = try await collect(harness.service.generate(model: .deep, turns: hello, options: GenerationOptions()))
        }
        #expect(!harness.journal.entries.contains(.loadStarted(AppModel.deep.modelID)))
        // Quick still runs.
        try await harness.service.load(.quick)
    }

    // MARK: Errors

    struct BackendBoom: Error, CustomStringConvertible {
        var description: String { "out of memory" }
    }

    @Test func backendLoadFailureMapsToLoadFailed() async {
        let harness = Harness(scripts: [AppModel.maple.modelID: FakeEngineScript(loadError: BackendBoom())])
        await #expect(throws: EngineServiceError.loadFailed("out of memory")) {
            try await harness.service.load(.maple)
        }
        #expect(harness.service.status.lastLoadFailure == "out of memory")
        #expect(harness.service.status.lastLoadFailureModel == .maple)
        #expect(harness.service.status.phase == .empty)
        await #expect(throws: EngineServiceError.loadFailed("out of memory")) {
            _ = try await collect(harness.service.generate(model: .maple, turns: hello, options: GenerationOptions()))
        }
    }

    @Test func contextOverflowMapsToRequestTooLarge() async {
        let tiny = FakeEngineScript(traits: EngineModelTraits(chatDialect: .chatml, maxContext: 3))
        let harness = Harness(scripts: [AppModel.quick.modelID: tiny])
        let long = [ChatTurn(role: .user, content: "one two three four five")]
        await #expect(throws: EngineServiceError.requestTooLarge("prompt exceeds the configured context")) {
            _ = try await collect(harness.service.generate(model: .quick, turns: long, options: GenerationOptions()))
        }
    }

    @Test func missingModelMapsToNotInstalledOrUnknown() async {
        let harness = Harness(installed: [.quick])
        await #expect(throws: EngineServiceError.unknownModel(AppModel.deep.modelID)) {
            try await harness.service.load(.deep)
        }
    }

    @Test func errorMessagesAreSentences() {
        let all: [EngineServiceError] = [
            .unknownModel("x"), .notInstalled(model: "Deep", reason: "r"), .engineUnavailable,
            .requestTooLarge("m"), .busy, .shuttingDown, .modelUnloaded, .loadFailed("m"),
            .invalidRequest("m"), .generationFailed("m"),
        ]
        for error in all {
            #expect(!error.userMessage.isEmpty)
            #expect(error.userMessage.first?.isUppercase == true)
        }
    }

    // MARK: Shutdown

    @Test func shutdownReleasesAndRefusesWork() async throws {
        let harness = Harness()
        try await harness.service.load(.quick)
        await harness.service.shutdown()
        #expect(harness.journal.entries.contains(.unloaded(AppModel.quick.modelID)))
        #expect(harness.service.status.phase == .empty)
        await #expect(throws: EngineServiceError.shuttingDown) { try await harness.service.load(.quick) }
        // A subscription after shutdown yields the final status and ends.
        var count = 0
        for await _ in harness.service.statusUpdates() { count += 1 }
        #expect(count == 1)
    }
}
