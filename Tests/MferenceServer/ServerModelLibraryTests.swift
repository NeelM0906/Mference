import Foundation
import Synchronization
import Testing
@testable import Mference
@testable import MferenceServerCore

// MARK: - Fakes

/// Records model lifecycle across loads, so a test can assert that the resident
/// model was released *before* its replacement was loaded rather than after.
private final class LibraryEventLog: Sendable {
    private let entries = Mutex<[String]>([])

    func append(_ event: String) {
        entries.withLock { $0.append(event) }
    }

    var events: [String] {
        entries.withLock { $0 }
    }
}

/// Appends `release <id>` when the model that owns it is deallocated. The
/// library drops its reference before it starts the next load, so the deinit
/// is the observable release point.
private final class ReleaseSentinel {
    private let log: LibraryEventLog
    private let modelID: String

    init(log: LibraryEventLog, modelID: String) {
        self.log = log
        self.modelID = modelID
    }

    deinit { log.append("release \(modelID)") }
}

private actor FakeLibraryModel: ServerLoadedModel {
    nonisolated let chatDialect: ChatDialect
    private let modelID: String
    private let sentinel: ReleaseSentinel
    private let beforeGenerate: @Sendable () async -> Void

    init(modelID: String,
         chatDialect: ChatDialect,
         log: LibraryEventLog,
         beforeGenerate: @escaping @Sendable () async -> Void = {}) {
        self.modelID = modelID
        self.chatDialect = chatDialect
        self.sentinel = ReleaseSentinel(log: log, modelID: modelID)
        self.beforeGenerate = beforeGenerate
    }

    func generate(
        _ prepared: PreparedGeneration,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        await beforeGenerate()
        onEvent(.content(modelID))
        return ServerCompletion(
            content: modelID,
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 3, completionTokens: 1, totalTokens: 4))
    }
}

private func makeIndex(_ pairs: [(id: String, path: String)]) -> ServerLibraryIndex {
    ServerLibraryIndex(entries: pairs.map {
        ServerLibraryEntry(modelID: $0.id,
                           familyModelID: $0.id,
                           basename: $0.id,
                           directory: URL(fileURLWithPath: $0.path, isDirectory: true),
                           family: .qwen36)
    })
}

/// Library whose loads are instant and recorded.
private func makeLibrary(
    _ index: ServerLibraryIndex,
    log: LibraryEventLog,
    maxContext: Int? = ServerArguments.defaultMaxContext,
    beforeLoad: @escaping @Sendable (URL) async -> Void = { _ in },
    beforeGenerate: @escaping @Sendable (String) async -> Void = { _ in }
) -> ServerModelLibrary {
    ServerModelLibrary(index: index, maxContext: maxContext) { directory in
        await beforeLoad(directory)
        let modelID = directory.lastPathComponent
        log.append("load \(modelID)")
        return FakeLibraryModel(modelID: modelID,
                                chatDialect: .chatml,
                                log: log,
                                beforeGenerate: { await beforeGenerate(modelID) })
    }
}

// MARK: - Identifiers

@Suite("Library model identifiers")
struct ServerFamilyModelIDTests {
    /// Every family the runtime knows has its own identifier: library mode
    /// has to be able to name every install it lists or declines.
    @Test func everyKnownFamilyHasAnIdentifier() {
        var identifiers = Set<String>()
        for family in ModelFamily.allCases {
            let identifier = ServerFamilyModelID.modelID(for: family)
            #expect(!identifier.isEmpty)
            #expect(identifiers.insert(identifier).inserted,
                    "\(family.rawValue) reuses identifier \(identifier)")
            #expect(ServerFamilyModelID.modelID(forRawFamily: family.rawValue) == identifier)
        }
    }

    /// The identifiers `ServerArguments.usage` advertises and the ones library
    /// mode derives are the same strings.
    @Test func identifiersMatchTheDocumentedDefaults() {
        #expect(ServerFamilyModelID.modelID(for: .qwen36) == "qwen3.6-35b-a3b")
        #expect(ServerFamilyModelID.modelID(for: .qwen38) == "qwen3.8-27b-4bit")
        #expect(ServerFamilyModelID.modelID(for: .maple) == "maple-preview-2bit-mlx")
        #expect(ServerFamilyModelID.modelID(for: .minicpm5) == "minicpm5-2b-int4g64")
    }

    @Test func unknownRawFamilyHasNoIdentifier() {
        #expect(ServerFamilyModelID.modelID(forRawFamily: "llama9") == nil)
    }

    /// A retired family is not a runtime family any more, so it has no API
    /// identifier either; discovery names its skip by the raw family string.
    @Test func retiredRawFamilyHasNoIdentifier() {
        for raw in ManifestReader.retiredFamilies {
            #expect(ServerFamilyModelID.modelID(forRawFamily: raw) == nil, "\(raw)")
        }
    }
}

// MARK: - Probe

@Suite("Library install probe")
struct ServerLibraryProbeTests {
    @Test func plainDirectoryIsNotAModelDirectory() throws {
        let root = try ServerLibraryFixture.makeRoot("plain")
        let directory = root.appendingPathComponent("notes", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        #expect(ServerLibraryProbe.probe(directory: directory) == .notAModelDirectory)
    }

    @Test func completeInstallIsAccepted() throws {
        let root = try ServerLibraryFixture.makeRoot("complete")
        let directory = try ServerLibraryFixture.makeCompleteInstall(in: root, named: "qwen36")
        #expect(ServerLibraryProbe.probe(directory: directory) == .complete(family: .qwen36))
    }

    /// The repacker holds `<name>.gturbo.install.lock` next to the directory it
    /// is writing, so a complete-looking install under an active lock is still
    /// refused.
    @Test func heldInstallLockRejectsTheDirectory() throws {
        let root = try ServerLibraryFixture.makeRoot("locked")
        let directory = try ServerLibraryFixture.makeCompleteInstall(in: root, named: "qwen36")
        let held = try ServerLibraryFixture.holdInstallLock(in: root, named: "qwen36")
        defer { close(held) }
        guard case .partial(let reason) = ServerLibraryProbe.probe(directory: directory) else {
            Issue.record("a locked install must not be advertised")
            return
        }
        #expect(reason.contains("install.lock"))
    }

    /// `InstallLock` leaves its zero-byte lock file behind after every
    /// completed install, so the file alone must not hide a complete install;
    /// only a held `flock` does.
    @Test func staleInstallLockFileIsIgnored() throws {
        let root = try ServerLibraryFixture.makeRoot("stale-lock")
        let directory = try ServerLibraryFixture.makeCompleteInstall(in: root, named: "qwen36")
        try ServerLibraryFixture.leaveStaleInstallLock(in: root, named: "qwen36")
        #expect(ServerLibraryProbe.probe(directory: directory) == .complete(family: .qwen36))
    }

    @Test func stagingDirectoryIsRejected() throws {
        let root = try ServerLibraryFixture.makeRoot("staging")
        let staging = root.appendingPathComponent("qwen36.gturbo.partial", isDirectory: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        guard case .partial = ServerLibraryProbe.probe(directory: staging) else {
            Issue.record("a .partial staging directory must not be advertised")
            return
        }
    }

    @Test func missingReceiptIsPartialNotComplete() throws {
        let root = try ServerLibraryFixture.makeRoot("no-receipt")
        let directory = try ServerLibraryFixture.makeCompleteInstall(in: root, named: "qwen36")
        try FileManager.default.removeItem(
            at: directory.appendingPathComponent("verified-install.json"))
        guard case .partial = ServerLibraryProbe.probe(directory: directory) else {
            Issue.record("an install without a receipt must not be advertised")
            return
        }
    }

    @Test func missingExpertLayoutIsPartial() throws {
        let root = try ServerLibraryFixture.makeRoot("no-layout")
        let directory = try ServerLibraryFixture.makeCompleteInstall(in: root, named: "qwen36")
        try FileManager.default.removeItem(
            at: directory.appendingPathComponent("packed_experts/layout.json"))
        guard case .partial(let reason) = ServerLibraryProbe.probe(directory: directory) else {
            Issue.record("an install without an expert layout must not be advertised")
            return
        }
        #expect(reason.contains("layout.json"))
    }

    /// A family the repacker installs but no runner executes is reported as
    /// such, not as a corrupt install, so the skip line names the real reason.
    @Test func gatedFamilyIsReportedAsNotRunnable() throws {
        let root = try ServerLibraryFixture.makeRoot("gated")
        let directory = try ServerLibraryFixture.makeGatedInstall(
            in: root, named: "future", family: "not-yet-family")
        guard case .notRunnable(let family, let detail) =
            ServerLibraryProbe.probe(directory: directory,
                                     gatedFamilies: ServerLibraryFixture.gate) else {
            Issue.record("a gated family must be reported as not runnable")
            return
        }
        #expect(family == "not-yet-family")
        #expect(detail.contains("no runner"))
        #expect(detail.contains("someAxis"))
    }

    /// The runtime's gate table ships empty, so a manifest naming a family
    /// with a runner is never "not runnable"; it falls through to the strict
    /// probe like any other install.
    @Test func liftedFamilyIsNotReportedAsGated() throws {
        let root = try ServerLibraryFixture.makeRoot("lifted")
        let directory = try ServerLibraryFixture.makeGatedInstall(
            in: root, named: "qwen36", family: ModelFamily.qwen36.rawValue)
        if case .notRunnable = ServerLibraryProbe.probe(directory: directory) {
            Issue.record("qwen36 has a runner and must not be reported as gated")
        }
    }

    /// An install of a retired family stays on disk but is never served: the
    /// probe names it as not runnable even when everything else about the
    /// install is complete, and a legacy manifest with no `arch.family` (a
    /// Gemma 4 install) is not complete either.
    @Test func retiredFamilyInstallIsNotRunnableAndNeverComplete() throws {
        let root = try ServerLibraryFixture.makeRoot("retired")
        let directory = try ServerLibraryFixture.makeRetiredInstall(
            in: root, named: "flashnext-iq3", family: "qwen38flashnextudiq3xxs")
        guard case .notRunnable(let family, let detail) =
            ServerLibraryProbe.probe(directory: directory) else {
            Issue.record("a retired-family install must be reported as not runnable")
            return
        }
        #expect(family == "qwen38flashnextudiq3xxs")
        #expect(detail.contains("retired family"))

        // No arch.family: the original Gemma 4 format. Not runnable, listed
        // under its modelID, never an incomplete install.
        let legacy = try ServerLibraryFixture.makeRetiredInstall(
            in: root, named: "legacy", family: nil)
        guard case .notRunnable(let legacyID, let legacyDetail) =
            ServerLibraryProbe.probe(directory: legacy) else {
            Issue.record("a manifest without arch.family must be reported as not runnable")
            return
        }
        #expect(legacyID == "test/legacy")
        #expect(legacyDetail.contains("no arch.family") && legacyDetail.contains("retired family"))
    }

    /// Swift-Qwen3.8 is a `qwen38` install, so its family alone would run it
    /// as base Qwen 3.8 and collide with a base install's ID. It is refused by
    /// checkpoint ID instead.
    @Test func retiredSwiftQwenCheckpointIsNotRunnable() throws {
        let root = try ServerLibraryFixture.makeRoot("swift-qwen")
        let directory = try ServerLibraryFixture.makeRetiredInstall(
            in: root, named: "swift-qwen", family: "qwen38",
            modelID: "swift-qwen3.8-27b-int4g64")
        guard case .notRunnable(let identifier, let detail) =
            ServerLibraryProbe.probe(directory: directory) else {
            Issue.record("a Swift-Qwen3.8 install must be reported as not runnable")
            return
        }
        #expect(identifier == "swift-qwen3.8-27b-int4g64")
        #expect(detail.contains("Swift-Qwen3.8") && detail.contains("retired family"))
    }

    @Test func corruptManifestIsPartial() throws {
        let root = try ServerLibraryFixture.makeRoot("corrupt")
        let directory = try ServerLibraryFixture.makeCorruptInstall(in: root, named: "broken")
        guard case .partial = ServerLibraryProbe.probe(directory: directory) else {
            Issue.record("a corrupt manifest must be reported as an incomplete install")
            return
        }
    }
}

// MARK: - Discovery

@Suite("Library discovery")
struct ServerLibraryDiscoveryTests {
    /// Fake probe keyed by basename, so identifier assignment can be exercised
    /// across families without building a manifest for each one.
    private static func probe(_ families: [String: ModelFamily])
        -> @Sendable (URL) -> ServerLibraryProbeResult {
        { url in
            guard let family = families[url.lastPathComponent] else {
                return .notAModelDirectory
            }
            return .complete(family: family)
        }
    }

    private static func children(_ tree: [String: [String]])
        -> @Sendable (URL) -> [URL] {
        { root in
            (tree[root.path] ?? []).map {
                root.appendingPathComponent($0, isDirectory: true)
            }
        }
    }

    @Test func oneInstallPerFamilyKeepsTheBareFamilyIdentifier() {
        let root = URL(fileURLWithPath: "/library", isDirectory: true)
        let index = ServerLibraryDiscovery.discover(
            roots: [root],
            childDirectories: Self.children(["/library": ["maple.gturbo", "qwen36.gturbo"]]),
            probe: Self.probe(["maple.gturbo": .maple, "qwen36.gturbo": .qwen36]))
        #expect(index.entries.map(\.modelID) == ["maple-preview-2bit-mlx", "qwen3.6-35b-a3b"])
        #expect(index.skipped.isEmpty)
    }

    /// The documented rule: two installs of the same family are both suffixed
    /// with their directory basename, and neither keeps the bare identifier.
    @Test func installsSharingAFamilyAreSuffixedWithTheirBasename() {
        let root = URL(fileURLWithPath: "/library", isDirectory: true)
        let index = ServerLibraryDiscovery.discover(
            roots: [root],
            childDirectories: Self.children([
                "/library": ["qwen36.gturbo", "qwen36-ourquant.gturbo", "maple.gturbo"],
            ]),
            probe: Self.probe([
                "qwen36.gturbo": .qwen36,
                "qwen36-ourquant.gturbo": .qwen36,
                "maple.gturbo": .maple,
            ]))
        #expect(index.entries.map(\.modelID) == [
            "maple-preview-2bit-mlx",
            "qwen3.6-35b-a3b@qwen36",
            "qwen3.6-35b-a3b@qwen36-ourquant",
        ])
        #expect(index.entry(for: "qwen3.6-35b-a3b") == nil)
    }

    /// Same family, same basename, two roots: the collision still resolves to
    /// two distinct identifiers rather than one shadowing the other.
    @Test func identicalBasenamesUnderTwoRootsStillGetDistinctIdentifiers() {
        let first = URL(fileURLWithPath: "/a", isDirectory: true)
        let second = URL(fileURLWithPath: "/b", isDirectory: true)
        let index = ServerLibraryDiscovery.discover(
            roots: [first, second],
            childDirectories: Self.children([
                "/a": ["qwen36.gturbo"],
                "/b": ["qwen36.gturbo"],
            ]),
            probe: Self.probe(["qwen36.gturbo": .qwen36]))
        #expect(index.entries.map(\.modelID) == [
            "qwen3.6-35b-a3b@qwen36",
            "qwen3.6-35b-a3b@qwen36#2",
        ])
        #expect(Set(index.entries.map(\.directory.path)).count == 2)
    }

    /// Identifiers depend on the set of installs, not on the order the roots
    /// were listed, so a restart with a reordered `--library` keeps the model
    /// picker's entries pointing at the same models.
    @Test func identifiersDoNotDependOnRootOrder() {
        let tree = ["/a": ["qwen36.gturbo"], "/b": ["qwen36-ourquant.gturbo"]]
        let probe = Self.probe([
            "qwen36.gturbo": .qwen36,
            "qwen36-ourquant.gturbo": .qwen36,
        ])
        let forward = ServerLibraryDiscovery.discover(
            roots: [URL(fileURLWithPath: "/a"), URL(fileURLWithPath: "/b")],
            childDirectories: Self.children(tree),
            probe: probe)
        let reversed = ServerLibraryDiscovery.discover(
            roots: [URL(fileURLWithPath: "/b"), URL(fileURLWithPath: "/a")],
            childDirectories: Self.children(tree),
            probe: probe)
        #expect(forward == reversed)
    }

    @Test func aRootThatIsItselfAnInstallIsListed() {
        let root = URL(fileURLWithPath: "/library/qwen36.gturbo", isDirectory: true)
        let index = ServerLibraryDiscovery.discover(
            roots: [root],
            childDirectories: Self.children([:]),
            probe: Self.probe(["qwen36.gturbo": .qwen36]))
        #expect(index.entries.map(\.modelID) == ["qwen3.6-35b-a3b"])
    }

    /// `--model` alongside `--library` may point outside every root; it still
    /// has to be listed, or the preloaded model would be unreachable.
    @Test func explicitModelDirectoryOutsideEveryRootIsListed() {
        let index = ServerLibraryDiscovery.discover(
            roots: [URL(fileURLWithPath: "/library", isDirectory: true)],
            explicitModelDirectory: URL(fileURLWithPath: "/elsewhere/maple.gturbo",
                                        isDirectory: true),
            childDirectories: Self.children(["/library": ["qwen36.gturbo"]]),
            probe: Self.probe(["qwen36.gturbo": .qwen36, "maple.gturbo": .maple]))
        #expect(index.entries.map(\.modelID) == [
            "maple-preview-2bit-mlx", "qwen3.6-35b-a3b",
        ])
    }

    @Test func theSameDirectoryReachedTwiceIsListedOnce() {
        let index = ServerLibraryDiscovery.discover(
            roots: [URL(fileURLWithPath: "/library", isDirectory: true),
                    URL(fileURLWithPath: "/library/", isDirectory: true)],
            explicitModelDirectory: URL(fileURLWithPath: "/library/qwen36.gturbo",
                                        isDirectory: true),
            childDirectories: Self.children(["/library": ["qwen36.gturbo"]]),
            probe: Self.probe(["qwen36.gturbo": .qwen36]))
        #expect(index.entries.count == 1)
        #expect(index.entries[0].modelID == "qwen3.6-35b-a3b")
    }

    /// End to end over real directories: only the complete install is
    /// advertised, and the other two are recorded with a reason.
    @Test func realRootAdvertisesOnlyCompleteInstalls() throws {
        let root = try ServerLibraryFixture.makeRoot("discovery")
        let complete = try ServerLibraryFixture.makeCompleteInstall(in: root, named: "qwen36")
        try ServerLibraryFixture.makeGatedInstall(in: root, named: "future",
                                                  family: "not-yet-family")
        try ServerLibraryFixture.makeCorruptInstall(in: root, named: "broken")
        let notes = root.appendingPathComponent("notes", isDirectory: true)
        try FileManager.default.createDirectory(at: notes, withIntermediateDirectories: true)

        let index = ServerLibraryDiscovery.discover(
            roots: [root],
            probe: { ServerLibraryProbe.probe(directory: $0,
                                              gatedFamilies: ServerLibraryFixture.gate) })
        #expect(index.entries.map(\.modelID) == ["qwen3.6-35b-a3b"])
        #expect(index.entries[0].directory == complete)
        #expect(index.skipped.count == 2)
        let reasons = index.skipped.map(\.reason).sorted()
        #expect(reasons.contains { $0.contains("not runnable") })
        #expect(reasons.contains { $0.contains("incomplete install") })
        // A directory with no manifest is not a skipped install, just a
        // directory, and must not appear in the report at all.
        #expect(!index.skipped.contains { $0.directory == notes })
    }

    /// A retired-family install in a library root is skipped with its reason
    /// and never listed, while a kept install beside it still is.
    @Test func retiredFamilyInstallIsSkippedNotListed() throws {
        let root = try ServerLibraryFixture.makeRoot("discovery-retired")
        let kept = try ServerLibraryFixture.makeCompleteInstall(in: root, named: "qwen36")
        let retired = try ServerLibraryFixture.makeRetiredInstall(
            in: root, named: "flashnext-iq3", family: "qwen38flashnextudiq3xxs")

        let index = ServerLibraryDiscovery.discover(roots: [root])
        #expect(index.entries.map(\.directory) == [kept])
        #expect(!index.entries.contains { $0.directory == retired })
        let skip = try #require(index.skipped.first { $0.directory == retired })
        #expect(skip.reason.contains("not runnable"))
        #expect(skip.reason.contains("retired family"))
        #expect(skip.modelID == "qwen38flashnextudiq3xxs")
    }

    /// Real discovery over a root with a kept install, a Swift-Qwen3.8
    /// install (kept `qwen38` family, retired checkpoint) and a family-less
    /// Gemma 4 install: only the kept one is listed; the other two are
    /// skipped as not runnable under their own checkpoint IDs, so Swift-Qwen
    /// never takes the base Qwen 3.8 identifier.
    @Test func retiredCheckpointAndFamilylessInstallsAreSkippedByID() throws {
        let root = try ServerLibraryFixture.makeRoot("discovery-retired-ids")
        let swift = try ServerLibraryFixture.makeRetiredInstall(
            in: root, named: "swift-qwen", family: "qwen38",
            modelID: "swift-qwen3.8-27b-int4g64")
        let legacy = try ServerLibraryFixture.makeRetiredInstall(
            in: root, named: "legacy", family: nil, modelID: "gemma-4-26b-a4b-it-qat-q4_0-mlx-aligned")
        let kept = try ServerLibraryFixture.makeCompleteInstall(in: root, named: "qwen36")

        let index = ServerLibraryDiscovery.discover(roots: [root])
        #expect(index.entries.map(\.directory) == [kept])
        #expect(!index.entries.contains { $0.modelID == "swift-qwen3.8-27b-int4g64" || $0.modelID == "qwen3.8-27b-4bit" })
        let swiftSkip = try #require(index.skipped.first { $0.directory == swift })
        #expect(swiftSkip.modelID == "swift-qwen3.8-27b-int4g64")
        #expect(swiftSkip.reason.contains("not runnable") && swiftSkip.reason.contains("Swift-Qwen3.8"))
        let legacySkip = try #require(index.skipped.first { $0.directory == legacy })
        #expect(legacySkip.modelID == "gemma-4-26b-a4b-it-qat-q4_0-mlx-aligned")
        #expect(legacySkip.reason.contains("no arch.family"))
    }
}

// MARK: - Library actor

@Suite("Model library swapping", .serialized)
struct ServerModelLibrarySwapTests {
    @Test func firstResolveLoadsLazilyAndTheSecondReusesIt() async throws {
        let log = LibraryEventLog()
        let library = makeLibrary(
            makeIndex([(id: "alpha", path: "/models/alpha")]),
            log: log)
        #expect(library.snapshot.health.status == "ok")
        #expect(library.snapshot.health.model == nil)

        _ = try await library.resolve(modelID: "alpha")
        _ = try await library.resolve(modelID: "alpha")
        #expect(log.events == ["load alpha"])
        #expect(library.snapshot.health.model == "alpha")
    }

    /// The hard rule: the resident model is released before the replacement is
    /// loaded, so peak memory is one model rather than two, and no second
    /// model process is ever involved.
    @Test func residentModelIsReleasedBeforeTheReplacementLoads() async throws {
        let log = LibraryEventLog()
        let library = makeLibrary(
            makeIndex([(id: "alpha", path: "/models/alpha"),
                       (id: "beta", path: "/models/beta")]),
            log: log)
        _ = try await library.resolve(modelID: "alpha")
        _ = try await library.resolve(modelID: "beta")
        #expect(log.events == ["load alpha", "release alpha", "load beta"])
    }

    @Test func unknownModelIsRejectedWithoutLoading() async throws {
        let log = LibraryEventLog()
        let library = makeLibrary(
            makeIndex([(id: "alpha", path: "/models/alpha")]),
            log: log)
        await #expect(throws: ServerRequestError.unknownModel) {
            _ = try await library.resolve(modelID: "not-installed")
        }
        #expect(log.events.isEmpty)
    }

    /// `/health` has to stay answerable while a load holds the library, because
    /// first-touch SHA-256 verification can run for minutes.
    @Test func healthReportsLoadingWhileASwapIsInFlight() async throws {
        let log = LibraryEventLog()
        let gate = LoadGate()
        let library = makeLibrary(
            makeIndex([(id: "alpha", path: "/models/alpha")]),
            log: log,
            beforeLoad: { _ in await gate.wait() })

        async let resolved: Void = {
            _ = try await library.resolve(modelID: "alpha")
        }()
        await gate.waitUntilEntered()
        let loading = library.snapshot.health
        #expect(loading.status == "loading")
        #expect(loading.model == "alpha")

        await gate.open()
        try await resolved
        #expect(library.snapshot.health.status == "ok")
        #expect(library.snapshot.health.model == "alpha")
    }

    /// A load that fails leaves nothing resident and nothing half-registered,
    /// so the next request can try again rather than inheriting a broken state.
    @Test func failedLoadLeavesNoResidentModel() async throws {
        struct LoadFailure: Error {}
        let attempts = Mutex(0)
        let log = LibraryEventLog()
        let library = ServerModelLibrary(
            index: makeIndex([(id: "alpha", path: "/models/alpha")])
        ) { directory in
            let attempt = attempts.withLock { value -> Int in
                value += 1
                return value
            }
            if attempt == 1 { throw LoadFailure() }
            return FakeLibraryModel(modelID: directory.lastPathComponent,
                                    chatDialect: .chatml,
                                    log: log)
        }
        await #expect(throws: LoadFailure.self) {
            _ = try await library.resolve(modelID: "alpha")
        }
        #expect(library.snapshot.health.status == "ok")
        #expect(library.snapshot.health.model == nil)

        let recovered = try await library.resolve(modelID: "alpha")
        #expect(recovered.modelID == "alpha")
        #expect(library.snapshot.health.model == "alpha")
    }
}

/// Parks a fake load until the test releases it.
private actor LoadGate {
    private var entered = false
    private var opened = false
    private var enteredWaiters: [CheckedContinuation<Void, Never>] = []
    private var openWaiters: [CheckedContinuation<Void, Never>] = []

    func wait() async {
        entered = true
        for waiter in enteredWaiters { waiter.resume() }
        enteredWaiters.removeAll()
        guard !opened else { return }
        await withCheckedContinuation { openWaiters.append($0) }
    }

    func waitUntilEntered() async {
        guard !entered else { return }
        await withCheckedContinuation { enteredWaiters.append($0) }
    }

    func open() {
        opened = true
        for waiter in openWaiters { waiter.resume() }
        openWaiters.removeAll()
    }
}

// MARK: - Arguments

@Suite("Library arguments")
struct ServerLibraryArgumentTests {
    /// Absent `--library` is the untouched single-model shape.
    @Test func libraryIsOffByDefaultAndModelStaysRequired() throws {
        let single = try ServerArguments.parse(["--model", "model.gturbo"])
        #expect(single.library == nil)
        #expect(single.model == "model.gturbo")
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse(["--port", "8080"])
        }
    }

    /// Bare `--library` means the default library roots; it must not swallow the
    /// flag that follows it.
    @Test func bareLibraryRequestsDefaultRootsWithoutEatingTheNextFlag() throws {
        let arguments = try ServerArguments.parse(["--library", "--port", "9001"])
        let library = try #require(arguments.library)
        #expect(library.includesDefaultRoots)
        #expect(library.roots.isEmpty)
        #expect(arguments.port == 9001)
        #expect(arguments.model == nil)
    }

    @Test func bareLibraryAtTheEndOfTheArgumentListIsAccepted() throws {
        let arguments = try ServerArguments.parse(["--library"])
        let library = try #require(arguments.library)
        #expect(library.includesDefaultRoots)
    }

    @Test func libraryRootsRepeatAndCombineWithTheDefaults() throws {
        let arguments = try ServerArguments.parse([
            "--library", "/a",
            "--library", "/b",
            "--library",
        ])
        let library = try #require(arguments.library)
        #expect(library.roots == ["/a", "/b"])
        #expect(library.includesDefaultRoots)
        let resolved = library.resolvedRoots(
            currentDirectoryURL: URL(fileURLWithPath: "/cwd", isDirectory: true),
            defaultRoots: { [URL(fileURLWithPath: "/defaults", isDirectory: true)] })
        #expect(resolved.map(\.path) == ["/a", "/b", "/defaults"])
    }

    @Test func relativeLibraryRootsResolveAgainstTheWorkingDirectory() throws {
        let arguments = try ServerArguments.parse(["--library", "scratch"])
        let library = try #require(arguments.library)
        let resolved = library.resolvedRoots(
            currentDirectoryURL: URL(fileURLWithPath: "/checkout", isDirectory: true),
            defaultRoots: { [] })
        #expect(resolved.map(\.path) == ["/checkout/scratch"])
    }

    @Test func duplicateRootsAreScannedOnce() throws {
        let arguments = try ServerArguments.parse([
            "--library", "/a",
            "--library", "/a",
        ])
        let library = try #require(arguments.library)
        #expect(library.resolvedRoots(defaultRoots: { [] }).map(\.path) == ["/a"])
    }

    /// `--model` alongside `--library` is a preload, not a mode switch.
    @Test func modelAlongsideLibraryIsAccepted() throws {
        let arguments = try ServerArguments.parse([
            "--library", "/a",
            "--model", "/a/qwen36.gturbo",
        ])
        #expect(arguments.library != nil)
        #expect(arguments.model == "/a/qwen36.gturbo")
    }

    /// One override cannot name several models, so the combination is refused
    /// rather than silently applying to whichever loads first.
    @Test func modelIDOverrideIsRefusedInLibraryMode() throws {
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse(["--library", "/a", "--model-id", "custom"])
        }
    }

    @Test func emptyLibraryRootIsRejected() throws {
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse(["--library", ""])
        }
    }

    @Test func usageDocumentsLibraryMode() {
        #expect(ServerArguments.usage.contains("--library"))
        #expect(ServerArguments.usage.contains("Mference.libraryRoot"))
    }
}

// MARK: - HTTP surface

@Suite("Library mode HTTP", .serialized)
struct LibraryHTTPServerTests {
    private func post(port: Int, body: String) async throws -> (Data, HTTPURLResponse) {
        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(body.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        return (data, response as! HTTPURLResponse)
    }

    @Test func modelsListsEveryDiscoveredInstall() async throws {
        let log = LibraryEventLog()
        let library = makeLibrary(
            makeIndex([(id: "alpha", path: "/models/alpha"),
                       (id: "beta", path: "/models/beta")]),
            log: log)
        let server = MferenceHTTPServer(library: library, queueLimit: 1)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        let data = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/v1/models")!).0
        let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let models = try #require(object["data"] as? [[String: Any]])
        #expect(models.compactMap { $0["id"] as? String } == ["alpha", "beta"])
        // Listing must not load anything.
        #expect(log.events.isEmpty)

        try await server.shutdown()
    }

    /// `max_model_len` (vLLM's field) is the context each model runs with:
    /// the configured length, or with `max` its family's native context.
    @Test func modelsReportTheContextEachModelRunsWith() {
        let entries = [
            ServerLibraryEntry(modelID: "qwen", familyModelID: "qwen3.6-35b-a3b",
                               basename: "qwen36.gturbo",
                               directory: URL(fileURLWithPath: "/models/qwen36.gturbo"), family: .qwen36),
            ServerLibraryEntry(modelID: "maple", familyModelID: "maple-preview-2bit-mlx",
                               basename: "maple.gturbo",
                               directory: URL(fileURLWithPath: "/models/maple.gturbo"), family: .maple),
        ]
        func lengths(_ maxContext: Int?) -> [String: Int] {
            let library = ServerModelLibrary(index: ServerLibraryIndex(entries: entries),
                                             maxContext: maxContext) { _ in
                Issue.record("listing must not load")
                throw CancellationError()
            }
            return Dictionary(uniqueKeysWithValues: library.snapshot.modelList.data.map {
                ($0.id, $0.maxModelLen)
            })
        }
        #expect(lengths(nil) == ["qwen": 262_144, "maple": 128_000])
        #expect(lengths(32_768) == ["qwen": 32_768, "maple": 32_768])
    }

    /// The listing comes from the startup index, so it answers while a load
    /// holds the library and loads nothing itself.
    @Test func modelsAnswerWhileALoadIsInFlight() async throws {
        let log = LibraryEventLog()
        let gate = LoadGate()
        let library = makeLibrary(makeIndex([(id: "alpha", path: "/models/alpha")]), log: log,
                                  maxContext: nil, beforeLoad: { _ in await gate.wait() })
        let server = MferenceHTTPServer(library: library, queueLimit: 1)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        async let resolved: Void = {
            _ = try await library.resolve(modelID: "alpha")
        }()
        await gate.waitUntilEntered()

        let data = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/v1/models")!).0
        #expect(String(decoding: data, as: UTF8.self).contains(#""max_model_len":262144"#))
        let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let models = try #require(object["data"] as? [[String: Any]])
        #expect(models.first?["max_model_len"] as? Int == 262_144)
        #expect(log.events.isEmpty, "the parked load is the only one and has not finished")

        await gate.open()
        try await resolved
        try await server.shutdown()
    }

    /// The unknown-model envelope is the one single-model mode already returns,
    /// and it still arrives as a real `404` because the check needs no load.
    @Test func unknownModelReturnsTheUnchanged404Envelope() async throws {
        let log = LibraryEventLog()
        let library = makeLibrary(makeIndex([(id: "alpha", path: "/models/alpha")]), log: log)
        let server = MferenceHTTPServer(library: library, queueLimit: 1)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        let (data, response) = try await post(
            port: port,
            body: #"{"model":"missing","messages":[{"role":"user","content":"hi"}]}"#)
        #expect(response.statusCode == 404)
        let text = String(decoding: data, as: UTF8.self)
        #expect(text.contains("model_not_found"))
        #expect(text.contains("requested model is not available"))
        #expect(log.events.isEmpty)

        try await server.shutdown()
    }

    /// Health stays answerable while a request is blocked on a load, and
    /// reports the model it is loading.
    @Test func healthReportsLoadingWhileARequestWaitsForASwap() async throws {
        let log = LibraryEventLog()
        let gate = LoadGate()
        let library = makeLibrary(makeIndex([(id: "alpha", path: "/models/alpha")]),
                                  log: log,
                                  beforeLoad: { _ in await gate.wait() })
        let server = MferenceHTTPServer(library: library, queueLimit: 1)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        async let completion = post(
            port: port,
            body: #"{"model":"alpha","messages":[{"role":"user","content":"hi"}]}"#)
        await gate.waitUntilEntered()
        let health = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/health")!).0
        let object = try #require(JSONSerialization.jsonObject(with: health) as? [String: Any])
        #expect(object["status"] as? String == "loading")
        #expect(object["model"] as? String == "alpha")

        await gate.open()
        let (_, response) = try await completion
        #expect(response.statusCode == 200)

        let settled = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/health")!).0
        let settledObject = try #require(
            JSONSerialization.jsonObject(with: settled) as? [String: Any])
        #expect(settledObject["status"] as? String == "ok")
        #expect(settledObject["model"] as? String == "alpha")

        try await server.shutdown()
    }

    /// A request naming a model that is not resident swaps in place: the old
    /// model is released, the new one loaded, and the response names the model
    /// that actually answered.
    @Test func completionForAnotherModelSwapsInPlace() async throws {
        let log = LibraryEventLog()
        let library = makeLibrary(
            makeIndex([(id: "alpha", path: "/models/alpha"),
                       (id: "beta", path: "/models/beta")]),
            log: log)
        let server = MferenceHTTPServer(library: library, queueLimit: 1)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        for model in ["alpha", "beta"] {
            let (data, response) = try await post(
                port: port,
                body: #"{"model":"\#(model)","messages":[{"role":"user","content":"hi"}]}"#)
            #expect(response.statusCode == 200)
            let object = try #require(
                JSONSerialization.jsonObject(with: data) as? [String: Any])
            #expect(object["model"] as? String == model)
            let choices = try #require(object["choices"] as? [[String: Any]])
            let message = try #require(choices[0]["message"] as? [String: Any])
            #expect(message["content"] as? String == model)
        }
        #expect(log.events == ["load alpha", "release alpha", "load beta"])

        try await server.shutdown()
    }

    @Test func streamingCompletionNamesTheResolvedModel() async throws {
        let log = LibraryEventLog()
        let library = makeLibrary(makeIndex([(id: "alpha", path: "/models/alpha")]), log: log)
        let server = MferenceHTTPServer(library: library, queueLimit: 1)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        let (data, response) = try await post(port: port, body: #"""
        {"model":"alpha","messages":[{"role":"user","content":"hi"}],
         "stream":true,"stream_options":{"include_usage":true}}
        """#)
        #expect(response.statusCode == 200)
        let text = String(decoding: data, as: UTF8.self)
        #expect(text.contains(#""model":"alpha""#))
        #expect(text.contains(#""content":"alpha""#))
        #expect(text.contains(#""finish_reason":"stop""#))
        #expect(text.hasSuffix("data: [DONE]\n\n"))

        try await server.shutdown()
    }

    /// Validation runs against the dialect of the model that was swapped in, so
    /// an unsupported parameter is still a `400` for a request that did not
    /// have to queue.
    @Test func unsupportedParameterStillReturns400() async throws {
        let log = LibraryEventLog()
        let library = makeLibrary(makeIndex([(id: "alpha", path: "/models/alpha")]), log: log)
        let server = MferenceHTTPServer(library: library, queueLimit: 1)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        let (data, response) = try await post(
            port: port,
            body: #"{"model":"alpha","messages":[{"role":"user","content":"hi"}],"n":2}"#)
        #expect(response.statusCode == 400)
        #expect(String(decoding: data, as: UTF8.self).contains("only n=1 is supported"))

        try await server.shutdown()
    }

    @Test func malformedBodyStillReturns400WithoutTouchingTheLibrary() async throws {
        let log = LibraryEventLog()
        let library = makeLibrary(makeIndex([(id: "alpha", path: "/models/alpha")]), log: log)
        let server = MferenceHTTPServer(library: library, queueLimit: 1)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        let (data, response) = try await post(port: port, body: "{not json")
        #expect(response.statusCode == 400)
        #expect(String(decoding: data, as: UTF8.self).contains("invalid_json"))
        #expect(log.events.isEmpty)

        try await server.shutdown()
    }

    /// The sequencing claim: a request that needs a different model does not
    /// unload anything while a generation is still running. The swap happens
    /// inside the requesting turn, so the in-flight generation and everything
    /// already queued ahead of it finish on the old model first.
    @Test func swapWaitsForTheRunningGenerationToDrain() async throws {
        let log = LibraryEventLog()
        let generation = LoadGate()
        let library = makeLibrary(
            makeIndex([(id: "alpha", path: "/models/alpha"),
                       (id: "beta", path: "/models/beta")]),
            log: log,
            beforeGenerate: { modelID in
                guard modelID == "alpha" else { return }
                await generation.wait()
            })
        let server = MferenceHTTPServer(library: library, queueLimit: 4)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        async let running = post(
            port: port,
            body: #"{"model":"alpha","messages":[{"role":"user","content":"hi"}]}"#)
        await generation.waitUntilEntered()

        async let queued = post(
            port: port,
            body: #"{"model":"beta","messages":[{"role":"user","content":"hi"}]}"#)
        // Wait until the beta request is actually queued behind alpha, so the
        // assertion below is about a real pending swap rather than a request
        // that has not arrived yet.
        var waited = 0
        while await server.queuedRequestCount == 0, waited < 200 {
            try await Task.sleep(for: .milliseconds(10))
            waited += 1
        }
        #expect(await server.queuedRequestCount == 1)
        // alpha is mid-generation: nothing has been released or loaded for beta.
        #expect(log.events == ["load alpha"])

        await generation.open()
        let (_, runningResponse) = try await running
        let (queuedData, queuedResponse) = try await queued
        #expect(runningResponse.statusCode == 200)
        #expect(queuedResponse.statusCode == 200)
        let object = try #require(
            JSONSerialization.jsonObject(with: queuedData) as? [String: Any])
        #expect(object["model"] as? String == "beta")
        #expect(log.events == ["load alpha", "release alpha", "load beta"])

        try await server.shutdown()
    }

    /// A queued *streaming* request has already had `200` and the SSE head
    /// committed by the time its turn comes, so a validation failure is
    /// reported in-band — the mechanism `docs/OPENAI_SERVER.md` describes for
    /// post-commit failures — rather than as a status it can no longer send.
    @Test func queuedStreamingRequestReportsValidationFailureInBand() async throws {
        let log = LibraryEventLog()
        let generation = LoadGate()
        let library = makeLibrary(
            makeIndex([(id: "alpha", path: "/models/alpha")]),
            log: log,
            beforeGenerate: { _ in await generation.wait() })
        let server = MferenceHTTPServer(library: library, queueLimit: 4)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        async let running = post(
            port: port,
            body: #"{"model":"alpha","messages":[{"role":"user","content":"hi"}]}"#)
        await generation.waitUntilEntered()

        async let rejected = post(port: port, body: #"""
        {"model":"alpha","messages":[{"role":"user","content":"hi"}],"stream":true,"n":2}
        """#)
        var waited = 0
        while await server.queuedRequestCount == 0, waited < 200 {
            try await Task.sleep(for: .milliseconds(10))
            waited += 1
        }
        await generation.open()

        _ = try await running
        let (data, response) = try await rejected
        #expect(response.statusCode == 200)
        let text = String(decoding: data, as: UTF8.self)
        #expect(text.contains("only n=1 is supported"))
        #expect(text.hasSuffix("data: [DONE]\n\n"))

        try await server.shutdown()
    }

    /// A preloaded model is resident before the first request, so that request
    /// pays no load at all.
    @Test func preloadedModelServesTheFirstRequestWithoutLoading() async throws {
        let log = LibraryEventLog()
        let library = makeLibrary(
            makeIndex([(id: "alpha", path: "/models/alpha"),
                       (id: "beta", path: "/models/beta")]),
            log: log)
        try await library.preload(modelID: "beta")
        #expect(log.events == ["load beta"])

        let server = MferenceHTTPServer(library: library, queueLimit: 1)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        let (_, response) = try await post(
            port: port,
            body: #"{"model":"beta","messages":[{"role":"user","content":"hi"}]}"#)
        #expect(response.statusCode == 200)
        #expect(log.events == ["load beta"])

        try await server.shutdown()
    }
}
