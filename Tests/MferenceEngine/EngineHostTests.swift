import Foundation
import Testing
import Mference
import MferenceEngine

@Suite("EngineHost state machine")
struct EngineHostTests {
    static let request = EngineRequest(
        messages: [MFTokenizer.Message(role: .user, content: "Hello there")],
        maximumCompletionTokens: 16)

    struct Fixture {
        let host: EngineHost
        let backend: FakeEngineBackend
        let events: Collector<EngineHostEvent>
        var journal: [FakeEngineJournalEntry] { backend.journal.entries }
    }

    static func fixture(ids: [String] = ["a", "b"],
                        scripts: [String: FakeEngineScript] = [:],
                        catalog: EngineCatalog? = nil,
                        loadOptions: EngineLoadOptions = EngineLoadOptions(),
                        idleUnloadAfter: Duration? = nil,
                        maxQueuedRequests: Int? = nil,
                        sleeper: ManualSleeper? = nil) -> Fixture {
        let backend = FakeEngineBackend(scripts: scripts)
        let events = Collector<EngineHostEvent>()
        let host = EngineHost(
            catalog: catalog ?? EngineCatalog(models: ids.map { FakeEngineBackend.descriptor(id: $0) }),
            backends: [backend],
            loadOptions: loadOptions,
            idleUnloadAfter: idleUnloadAfter,
            maxQueuedRequests: maxQueuedRequests,
            sleep: { duration in
                if let sleeper { try await sleeper.sleep(duration) } else { try await Task.sleep(for: duration) }
            },
            observer: { events.append($0) })
        return Fixture(host: host, backend: backend, events: events)
    }

    static func run(_ host: EngineHost, _ id: String,
                    onQueued: @escaping @Sendable () -> Void = {},
                    events: Collector<EngineEvent>? = nil) async throws -> EngineCompletion {
        try await host.generate(modelID: id, request: request, onQueued: onQueued) { events?.append($0) }
    }

    @Test func firstRequestLoadsTheModelAndStreamsInOrder() async throws {
        let fixture = Self.fixture()
        #expect(fixture.host.status.phase == .empty)
        #expect(fixture.host.status.health == EngineHealth(status: "ok", modelID: nil))

        let streamed = Collector<EngineEvent>()
        let completion = try await Self.run(fixture.host, "a", events: streamed)

        #expect(streamed.all == [.content("Hello"), .content(" world")])
        #expect(completion.content == "Hello world")
        #expect(completion.finishReason == .stop)
        #expect(completion.usage == EngineUsage(promptTokens: 3, completionTokens: 2))
        #expect(fixture.journal == [.loadStarted("a"), .loadFinished("a"),
                                    .generationStarted("a"), .generationFinished("a")])
        let status = fixture.host.status
        #expect(status.phase == .ready)
        #expect(status.residentModelID == "a")
        #expect(status.activeRequests == 0)
        #expect(status.health == EngineHealth(status: "ok", modelID: "a"))
    }

    @Test func sameModelIsNotReloaded() async throws {
        let fixture = Self.fixture()
        _ = try await Self.run(fixture.host, "a")
        _ = try await Self.run(fixture.host, "a")
        #expect(fixture.journal.count(of: .loadStarted("a")) == 1)
        #expect(fixture.journal.generationsStarted == ["a", "a"])
    }

    @Test func swapReleasesTheOldModelBeforeLoadingTheNext() async throws {
        let fixture = Self.fixture()
        _ = try await Self.run(fixture.host, "a")
        _ = try await Self.run(fixture.host, "b")

        let journal = fixture.journal
        let loadB = try #require(journal.index(of: .loadStarted("b")))
        let unloadA = try #require(journal.index(of: .unloaded("a")))
        let freedA = try #require(journal.index(of: .deinitialized("a")))
        #expect(unloadA < loadB)
        // The host keeps no reference past the release: the object is freed
        // before the next load begins, not merely asked to unload.
        #expect(freedA < loadB)
        #expect(fixture.backend.journal.maxLiveModels == 1)
        #expect(fixture.host.status.residentModelID == "b")

        let lifecycle = fixture.events.all.filter {
            if case .loadFinished = $0 { false } else { true }
        }
        #expect(lifecycle == [
            .loadStarted(modelID: "a", replacing: nil),
            .unloadStarted(modelID: "a", reason: .swap(to: "b")),
            .unloadFinished(modelID: "a", reason: .swap(to: "b")),
            .loadStarted(modelID: "b", replacing: "a"),
        ])
    }

    @Test func healthReportsTheTargetWhileASwapLoads() async throws {
        let started = FakeEngineLatch()
        let gate = FakeEngineLatch()
        let fixture = Self.fixture(scripts: ["b": FakeEngineScript(loadStarted: started, loadGate: gate)])
        _ = try await Self.run(fixture.host, "a")

        let host = fixture.host
        let pending = Task { try await Self.run(host, "b") }
        try await started.wait()
        let status = host.status
        #expect(status.phase == .loading)
        #expect(status.targetModelID == "b")
        #expect(status.residentModelID == nil)
        #expect(status.health == EngineHealth(status: "loading", modelID: "b"))
        #expect(fixture.journal.contains(.unloaded("a")))

        gate.open()
        _ = try await pending.value
        #expect(host.status.health == EngineHealth(status: "ok", modelID: "b"))
    }

    @Test func concurrentRequestsRunOneAtATimeInArrivalOrder() async throws {
        let reached = FakeEngineLatch()
        let gate = FakeEngineLatch()
        let fixture = Self.fixture(scripts: [
            "a": FakeEngineScript(holdAfterEvents: 1, holdReached: reached, holdGate: gate),
        ])
        let host = fixture.host
        let queuedCalls = Collector<Int>()

        let first = Task { try await Self.run(host, "a") }
        try await reached.wait()
        let second = Task { try await Self.run(host, "a", onQueued: { queuedCalls.append(2) }) }
        try await eventually("second request queued") { host.status.queuedRequests == 1 }
        let third = Task { try await Self.run(host, "b", onQueued: { queuedCalls.append(3) }) }
        try await eventually("third request queued") { host.status.queuedRequests == 2 }

        #expect(host.status.activeRequests == 1)
        #expect(fixture.journal.generationsStarted == ["a"])

        gate.open()
        _ = try await first.value
        _ = try await second.value
        _ = try await third.value

        let journal = fixture.journal
        #expect(journal.generationsStarted == ["a", "a", "b"])
        #expect(fixture.backend.journal.maxConcurrentGenerations == 1)
        #expect(fixture.backend.journal.maxLiveModels == 1)
        // No swap under a running request: b loads after both a requests end.
        let lastFinishedA = try #require(journal.lastIndex(of: .generationFinished("a")))
        let loadB = try #require(journal.index(of: .loadStarted("b")))
        #expect(lastFinishedA < loadB)
        #expect(Set(queuedCalls.all) == [2, 3])
        #expect(host.status.queuedRequests == 0)
    }

    @Test func cancellingTheCallerStopsGenerationMidStream() async throws {
        let reached = FakeEngineLatch()
        let gate = FakeEngineLatch()
        let fixture = Self.fixture(scripts: [
            "a": FakeEngineScript(events: [.content("one"), .content("two"), .content("three")],
                                  holdAfterEvents: 1, holdReached: reached, holdGate: gate),
        ])
        let host = fixture.host
        let streamed = Collector<EngineEvent>()
        let request = Task { try await Self.run(host, "a", events: streamed) }
        try await reached.wait()
        request.cancel()

        await #expect(throws: CancellationError.self) { try await request.value }
        #expect(streamed.all == [.content("one")])
        #expect(fixture.journal.contains(.generationCancelled("a")))
        #expect(host.status.phase == .ready)
        #expect(host.status.activeRequests == 0)

        // The model stays resident and serves the next request.
        gate.open()
        let completion = try await Self.run(host, "a")
        #expect(completion.content == "onetwothree")
        #expect(fixture.journal.count(of: .loadStarted("a")) == 1)
    }

    @Test func cancelActiveRequestStopsOnlyTheRunningRequest() async throws {
        let reached = FakeEngineLatch()
        let gate = FakeEngineLatch()
        let fixture = Self.fixture(scripts: [
            "a": FakeEngineScript(holdAfterEvents: 1, holdReached: reached, holdGate: gate),
        ])
        let host = fixture.host
        let running = Task { try await Self.run(host, "a") }
        try await reached.wait()
        let waiting = Task { try await Self.run(host, "a") }
        try await eventually("waiter queued") { host.status.queuedRequests == 1 }

        await host.cancelActiveRequest()
        await #expect(throws: CancellationError.self) { try await running.value }
        gate.open()
        let completion = try await waiting.value
        #expect(completion.content == "Hello world")
    }

    @Test func cancellingAQueuedRequestLeavesTheLine() async throws {
        let reached = FakeEngineLatch()
        let gate = FakeEngineLatch()
        let fixture = Self.fixture(scripts: [
            "a": FakeEngineScript(holdAfterEvents: 1, holdReached: reached, holdGate: gate),
        ])
        let host = fixture.host
        let running = Task { try await Self.run(host, "a") }
        try await reached.wait()
        let queued = Task { try await Self.run(host, "b") }
        try await eventually("waiter queued") { host.status.queuedRequests == 1 }

        queued.cancel()
        await #expect(throws: CancellationError.self) { try await queued.value }
        try await eventually("waiter removed") { host.status.queuedRequests == 0 }

        gate.open()
        _ = try await running.value
        #expect(!fixture.journal.contains(.loadStarted("b")))
        #expect(host.status.residentModelID == "a")
    }

    @Test func loadFailureLeavesTheHostEmpty() async throws {
        let fixture = Self.fixture(scripts: ["b": FakeEngineScript(loadError: InjectedLoadError())])
        _ = try await Self.run(fixture.host, "a")

        await #expect(throws: InjectedLoadError.self) { try await Self.run(fixture.host, "b") }

        let status = fixture.host.status
        #expect(status.phase == .empty)
        #expect(status.residentModelID == nil)
        #expect(status.targetModelID == nil)
        #expect(status.lastLoadFailure?.modelID == "b")
        #expect(status.health == EngineHealth(status: "ok", modelID: nil))
        let journal = fixture.journal
        #expect(try #require(journal.index(of: .unloaded("a"))) < #require(journal.index(of: .loadStarted("b"))))
        #expect(journal.contains(.loadFailed("b")))
        #expect(fixture.backend.journal.liveModels == 0)
        #expect(fixture.events.all.contains {
            if case .loadFailed(modelID: "b", _) = $0 { true } else { false }
        })

        // The next request loads normally and clears the failure.
        _ = try await Self.run(fixture.host, "a")
        #expect(fixture.host.status.residentModelID == "a")
        #expect(fixture.host.status.lastLoadFailure == nil)
    }

    @Test func idleModelIsUnloadedAfterTheTimeout() async throws {
        let sleeper = ManualSleeper()
        let fixture = Self.fixture(idleUnloadAfter: .seconds(300), sleeper: sleeper)
        let host = fixture.host

        _ = try await Self.run(host, "a")
        try await eventually("idle timer armed") { sleeper.requested == 1 }
        // A request before the timer fires re-arms it rather than unloading.
        _ = try await Self.run(host, "a")
        try await eventually("idle timer re-armed") { sleeper.requested == 2 }
        #expect(host.status.residentModelID == "a")

        sleeper.fireAll()
        try await eventually("idle unload") { host.status.phase == .empty }
        #expect(fixture.journal.count(of: .unloaded("a")) == 1)
        #expect(fixture.events.all.contains(.unloadStarted(modelID: "a", reason: .idle)))
        #expect(fixture.backend.journal.liveModels == 0)

        // The next request loads again, and no timer runs with nothing resident.
        _ = try await Self.run(host, "a")
        #expect(fixture.journal.count(of: .loadStarted("a")) == 2)
        sleeper.fireAll()
        await host.shutdown()
    }

    /// A swap has already released the old model, so a load abandoned halfway
    /// would leave the host empty for nothing: the load always finishes.
    @Test func cancellingTheCallerDuringALoadLetsTheLoadFinish() async throws {
        let started = FakeEngineLatch()
        let gate = FakeEngineLatch()
        let fixture = Self.fixture(scripts: ["b": FakeEngineScript(loadStarted: started, loadGate: gate)])
        _ = try await Self.run(fixture.host, "a")
        let host = fixture.host
        let request = Task { try await Self.run(host, "b") }
        try await started.wait()
        request.cancel()
        gate.open()

        await #expect(throws: CancellationError.self) { try await request.value }
        #expect(host.status.phase == .ready)
        #expect(host.status.residentModelID == "b")
        #expect(host.status.lastLoadFailure == nil)
        #expect(fixture.journal.contains(.loadFinished("b")))
        #expect(!fixture.journal.generationsStarted.contains("b"))
        #expect(host.status.activeRequests == 0)
    }

    @Test func cancelActiveRequestReachesARequestThatIsStillLoading() async throws {
        let started = FakeEngineLatch()
        let gate = FakeEngineLatch()
        let fixture = Self.fixture(scripts: ["a": FakeEngineScript(loadStarted: started, loadGate: gate)])
        let host = fixture.host
        let request = Task { try await Self.run(host, "a") }
        try await started.wait()
        #expect(host.status.activeRequests == 1)
        await host.cancelActiveRequest()
        gate.open()

        await #expect(throws: CancellationError.self) { try await request.value }
        #expect(host.status.residentModelID == "a")
        #expect(fixture.journal.generationsStarted.isEmpty)
        // The flag does not leak into the next request.
        let completion = try await Self.run(host, "a")
        #expect(completion.content == "Hello world")
    }

    @Test func shutdownDuringALoadReleasesTheModelOnceLoaded() async throws {
        let started = FakeEngineLatch()
        let gate = FakeEngineLatch()
        let fixture = Self.fixture(scripts: ["a": FakeEngineScript(loadStarted: started, loadGate: gate)])
        let host = fixture.host
        let request = Task { try await Self.run(host, "a") }
        try await started.wait()
        let stopping = Task { await host.shutdown() }
        try await eventually("shutdown begun") { host.status.isShuttingDown }
        gate.open()
        await stopping.value

        await #expect(throws: CancellationError.self) { try await request.value }
        #expect(host.status.phase == .empty)
        #expect(fixture.journal.generationsStarted.isEmpty)
        #expect(fixture.journal.contains(.unloaded("a")))
        #expect(fixture.events.all.contains(.unloadFinished(modelID: "a", reason: .shutdown)))
    }

    @Test func shutdownFromACancelledTaskStillReleasesTheModel() async throws {
        let fixture = Self.fixture()
        try await fixture.host.load("a")
        let host = fixture.host
        let stopping = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            await host.shutdown()
        }
        await stopping.value
        #expect(host.status.phase == .empty)
        #expect(fixture.journal.contains(.unloaded("a")))
        // A second call returns at once; the host stays shut.
        await host.shutdown()
        await #expect(throws: EngineError.shuttingDown) { try await host.load("a") }
    }

    @Test func noIdleUnloadWithoutATimeout() async throws {
        let sleeper = ManualSleeper()
        let fixture = Self.fixture(sleeper: sleeper)
        _ = try await Self.run(fixture.host, "a")
        try await Task.sleep(for: .milliseconds(20))
        #expect(sleeper.requested == 0)
        #expect(fixture.host.status.residentModelID == "a")
    }

    @Test func unknownUnrunnableAndBackendlessModelsAreRefusedWithoutLoading() async throws {
        let catalog = EngineCatalog(
            models: [FakeEngineBackend.descriptor(id: "a"),
                     FakeEngineBackend.descriptor(id: "gguf", engine: .llamaCpp)],
            skipped: [EngineCatalogSkip(location: URL(fileURLWithPath: "/x"),
                                        reason: "incomplete install", modelID: "partial")])
        let fixture = Self.fixture(catalog: catalog)
        await #expect(throws: EngineError.unknownModel("nope")) { try await Self.run(fixture.host, "nope") }
        await #expect(throws: EngineError.modelNotRunnable(id: "partial", reason: "incomplete install")) {
            try await Self.run(fixture.host, "partial")
        }
        await #expect(throws: EngineError.backendUnavailable(.llamaCpp)) {
            try await Self.run(fixture.host, "gguf")
        }
        #expect(fixture.journal.isEmpty)
        #expect(fixture.host.status.phase == .empty)
    }

    @Test func aFullQueueRefusesTheNextRequest() async throws {
        let reached = FakeEngineLatch()
        let gate = FakeEngineLatch()
        let fixture = Self.fixture(scripts: [
            "a": FakeEngineScript(holdAfterEvents: 1, holdReached: reached, holdGate: gate),
        ], maxQueuedRequests: 1)
        let host = fixture.host
        let running = Task { try await Self.run(host, "a") }
        try await reached.wait()
        let queued = Task { try await Self.run(host, "a") }
        try await eventually("waiter queued") { host.status.queuedRequests == 1 }

        await #expect(throws: EngineError.queueFull) { try await Self.run(host, "a") }
        gate.open()
        _ = try await running.value
        _ = try await queued.value
    }

    @Test func rejectedPromptLeavesTheModelResidentAndGeneratesNothing() async throws {
        let fixture = Self.fixture(loadOptions: EngineLoadOptions(nativeMaxContext: 3))
        let traits = try await fixture.host.withModel("a") { $0.traits }
        #expect(traits.maxContext == 3)

        await #expect(throws: EngineError.invalidRequest(message: "prompt exceeds the configured context",
                                                         param: "messages",
                                                         code: "context_length_exceeded")) {
            try await Self.run(fixture.host, "a")
        }
        #expect(fixture.journal.generationsStarted.isEmpty)
        #expect(fixture.host.status.residentModelID == "a")
    }

    @Test func explicitLoadAndUnload() async throws {
        let fixture = Self.fixture()
        try await fixture.host.load("b")
        #expect(fixture.host.status.residentModelID == "b")
        try await fixture.host.unload()
        #expect(fixture.host.status.phase == .empty)
        #expect(fixture.journal == [.loadStarted("b"), .loadFinished("b"), .unloaded("b"), .deinitialized("b")])
        #expect(fixture.events.all.contains(.unloadFinished(modelID: "b", reason: .requested)))
        // Unloading an empty host is a no-op.
        try await fixture.host.unload()
    }

    @Test func shutdownFailsWaitersCancelsTheRunningRequestAndUnloads() async throws {
        let reached = FakeEngineLatch()
        let gate = FakeEngineLatch()
        let fixture = Self.fixture(scripts: [
            "a": FakeEngineScript(holdAfterEvents: 1, holdReached: reached, holdGate: gate),
        ])
        let host = fixture.host
        let running = Task { try await Self.run(host, "a") }
        try await reached.wait()
        let queued = Task { try await Self.run(host, "b") }
        try await eventually("waiter queued") { host.status.queuedRequests == 1 }

        await host.shutdown()
        await #expect(throws: CancellationError.self) { try await running.value }
        await #expect(throws: EngineError.shuttingDown) { try await queued.value }
        #expect(host.status.phase == .empty)
        #expect(fixture.journal.contains(.unloaded("a")))
        await #expect(throws: EngineError.shuttingDown) { try await Self.run(host, "a") }
    }

    @Test func streamYieldsPreparedEventsThenTheCompletion() async throws {
        let fixture = Self.fixture()
        var prepared: Int?
        var events: [EngineEvent] = []
        var completion: EngineCompletion?
        for try await element in fixture.host.stream(modelID: "a", request: Self.request) {
            switch element {
            case .prepared(let tokens): prepared = tokens
            case .event(let event): events.append(event)
            case .completed(let done): completion = done
            }
        }
        #expect(prepared == 3)
        #expect(events == [.content("Hello"), .content(" world")])
        #expect(completion?.content == "Hello world")
    }

    @Test func breakingOutOfARetainedStreamCancelsTheRequest() async throws {
        let gate = FakeEngineLatch()
        let fixture = Self.fixture(scripts: [
            "a": FakeEngineScript(holdAfterEvents: 1, holdGate: gate),
        ])
        // The caller keeps the sequence for its whole lifetime.
        let retained = fixture.host.stream(modelID: "a", request: Self.request)
        for try await element in retained {
            if case .event = element { break }
        }
        try await eventually("generation cancelled") {
            fixture.backend.journal.entries.contains(.generationCancelled("a"))
        }
        try await eventually("request released") { fixture.host.status.activeRequests == 0 }
        // The kept sequence can start a fresh request.
        gate.open()
        var completed: EngineCompletion?
        for try await element in retained {
            if case .completed(let done) = element { completed = done }
        }
        #expect(completed?.content == "Hello world")
    }

    @Test func cancellingTheIteratingTaskThrowsCancellationError() async throws {
        let reached = FakeEngineLatch()
        let gate = FakeEngineLatch()
        let fixture = Self.fixture(scripts: [
            "a": FakeEngineScript(holdAfterEvents: 1, holdReached: reached, holdGate: gate),
        ])
        let host = fixture.host
        let consumer = Task {
            var sawCompletion = false
            for try await element in host.stream(modelID: "a", request: Self.request) {
                if case .completed = element { sawCompletion = true }
            }
            return sawCompletion
        }
        try await reached.wait()
        consumer.cancel()
        await #expect(throws: CancellationError.self) { try await consumer.value }
        try await eventually("generation cancelled") {
            fixture.backend.journal.entries.contains(.generationCancelled("a"))
        }
    }

    @Test func abandoningAStreamCancelsTheRequest() async throws {
        let reached = FakeEngineLatch()
        let gate = FakeEngineLatch()
        let fixture = Self.fixture(scripts: [
            "a": FakeEngineScript(holdAfterEvents: 1, holdReached: reached, holdGate: gate),
        ])
        do {
            var iterator = fixture.host.stream(modelID: "a", request: Self.request).makeAsyncIterator()
            _ = try await iterator.next()
            try await reached.wait()
        }
        try await eventually("generation cancelled") {
            fixture.backend.journal.entries.contains(.generationCancelled("a"))
        }
        try await eventually("request released") { fixture.host.status.activeRequests == 0 }
    }
}
