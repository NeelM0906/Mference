import Foundation
import Testing
import Mference
@testable import MferenceEngine

@Suite("EngineCatalogDiscovery")
struct EngineCatalogDiscoveryTests {
    static let library = URL(fileURLWithPath: "/lib", isDirectory: true)
    static let gguf = URL(fileURLWithPath: "/lib/gguf", isDirectory: true)

    static func dir(_ path: String) -> URL { URL(fileURLWithPath: path, isDirectory: true).standardizedFileURL }
    static func file(_ path: String) -> URL { URL(fileURLWithPath: path).standardizedFileURL }

    static let quickFile = file("/lib/gguf/Qwen3.5-4B-UD-Q4_K_XL.gguf")
    static let deepFile = file("/lib/gguf/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf")
    static let bonsaiFile = file("/lib/gguf/Ternary-Bonsai-2.5-27B-Q2_0-mtp.gguf")

    static func meta(mtp: Bool, context: Int? = 262_144) -> EngineGGUFMetadata {
        EngineGGUFMetadata(architecture: "qwen35", name: "x", contextLength: context,
                           tensorCount: 10, nextNTensorCount: mtp ? 4 : 0)
    }

    /// A fake disk: `children` per root, probe results per directory, GGUF
    /// files per directory and their headers.
    static func fileSystem(children: [String: [String]],
                           probes: [String: EngineInstallProbeResult],
                           ggufFiles: [String: [String]] = [:],
                           headers: [String: Result<EngineGGUFMetadata, EngineGGUFReader.ReadError>] = [:])
        -> EngineCatalogDiscovery.FileSystem {
        EngineCatalogDiscovery.FileSystem(
            childDirectories: { root in (children[root.path] ?? []).map(dir) },
            ggufFiles: { directory in (ggufFiles[directory.path] ?? []).map(file) },
            probe: { probes[$0.path] ?? .notAModelDirectory },
            checkpointID: { _ in nil },
            readGGUF: { url in
                guard let header = headers[url.path] else { throw EngineGGUFReader.ReadError.notGGUF }
                return try header.get()
            },
            size: { _ in 42 })
    }

    static let standardDisk = fileSystem(
        children: ["/lib": ["/lib/gguf", "/lib/maple.gturbo", "/lib/minicpm5.gturbo",
                            "/lib/minicpm5-1b.gturbo", "/lib/half.gturbo", "/lib/qwen36.gturbo", "/lib/qwen38.gturbo",
                            "/lib/inkling2mini.gturbo"]],
        probes: ["/lib/qwen36.gturbo": .complete(family: .qwen36),
                 "/lib/inkling2mini.gturbo": .complete(family: .inkling2Loop),
                 "/lib/minicpm5.gturbo": .complete(family: .minicpm5),
                 "/lib/minicpm5-1b.gturbo": .complete(family: .minicpm5_1b),
                 "/lib/maple.gturbo": .complete(family: .maple),
                 "/lib/qwen38.gturbo": .complete(family: .qwen38),
                 "/lib/half.gturbo": .partial("receipt missing")],
        ggufFiles: ["/lib/gguf": [quickFile.path, deepFile.path, bonsaiFile.path,
                                  "/lib/gguf/Other-Model.Q8_0.gguf"]],
        headers: [quickFile.path: .success(meta(mtp: false)),
                  deepFile.path: .success(meta(mtp: true)),
                  bonsaiFile.path: .success(meta(mtp: true)),
                  "/lib/gguf/Other-Model.Q8_0.gguf": .success(meta(mtp: false, context: 8_192))])

    /// A Mac above every product model's memory floor.
    static let roomy: UInt64 = 64 << 30

    @Test func productScopeListsExactlyTheProductModels() throws {
        let catalog = EngineCatalogDiscovery.discover(roots: [Self.library], ggufDirectories: [Self.gguf],
                                                      fileSystem: Self.standardDisk, physicalMemory: Self.roomy)
        #expect(catalog.models.map(\.id) == [
            "bonsai-2.5-27b-q2_0", "inkling-2-mini-ud-q4_k_xl", "maple-preview-2bit-mlx", "minicpm5-1b-mlx-4bit",
            "minicpm5-2b-int4g64", "qwen3.5-4b", "qwen3.6-35b-a3b", "qwen3.8-27b",
        ])
        #expect(Set(catalog.models.compactMap(\.product)) == Set(EngineProductSlot.allCases))
        // The two MiniCPM5 sizes are separate product models side by side.
        let small = try #require(catalog.descriptor(for: "minicpm5-1b-mlx-4bit"))
        #expect(small.product == .minicpm5_1b)
        #expect(small.engine == .mference)
        #expect(small.format == .gturbo(family: .minicpm5_1b, checkpointID: nil))
        #expect(small.location.lastPathComponent == "minicpm5-1b.gturbo")
        #expect(small.contextLength == 131_072)
        #expect(catalog.descriptor(for: "minicpm5-2b-int4g64")?.product == .minicpm5)
        let deep = try #require(catalog.descriptor(for: "qwen3.8-27b"))
        #expect(deep.engine == .llamaCpp)
        #expect(deep.location == Self.deepFile)
        #expect(deep.contextLength == 262_144)
        #expect(deep.displayName == EngineProductSlot.deep.displayName)
        let bonsai = try #require(catalog.descriptor(for: "bonsai-2.5-27b-q2_0"))
        #expect(bonsai.product == .bonsai27b && bonsai.engine == .llamaCpp)
        #expect(bonsai.location == Self.bonsaiFile)
        let qwen36 = try #require(catalog.descriptor(for: "qwen3.6-35b-a3b"))
        #expect(qwen36.engine == .mference)
        #expect(qwen36.format == .gturbo(family: .qwen36, checkpointID: nil))
        #expect(qwen36.contextLength == ModelFamily.qwen36.maximumContext)
        #expect(qwen36.sizeBytes == 42)
        #expect(catalog.models.filter { $0.hostContextCap != nil }.map(\.id) == [])
        let inkling = try #require(catalog.descriptor(for: "inkling-2-mini-ud-q4_k_xl"))
        #expect(inkling.product == .inkling2Mini && inkling.engine == .mference)
        #expect(inkling.displayName == "Inkling 2 Mini")
        #expect(inkling.format == .gturbo(family: .inkling2Loop, checkpointID: nil))
        #expect(inkling.location == Self.dir("/lib/inkling2mini.gturbo"))

        // Native Qwen 3.8 is found but not offered, and says so when requested.
        #expect(catalog.descriptor(for: "qwen3.8-27b-4bit") == nil)
        #expect(catalog.unavailableReason(for: "qwen3.8-27b-4bit")?
            .contains("not in the product model list") == true)
        #expect(catalog.skipped.contains { $0.location == Self.dir("/lib/half.gturbo") })
        #expect(catalog.skipped.contains { $0.location.lastPathComponent == "Other-Model.Q8_0.gguf" })
        // A declined GGUF keeps its identifier, so a request for it is told why.
        #expect(catalog.unavailableReason(for: "other-model.q8_0")?
            .contains("not in the product model list") == true)
        do {
            _ = try catalog.resolve("other-model.q8_0")
            Issue.record("a declined GGUF resolved")
        } catch let error as EngineError {
            guard case .modelNotRunnable(let id, _) = error else {
                Issue.record("expected modelNotRunnable, got \(error)")
                return
            }
            #expect(id == "other-model.q8_0")
        }
    }

    @Test func allScopeAlsoListsEverythingElseRunnable() throws {
        let catalog = EngineCatalogDiscovery.discover(roots: [Self.library], ggufDirectories: [Self.gguf],
                                                      scope: .all, fileSystem: Self.standardDisk,
                                                      physicalMemory: Self.roomy)
        #expect(catalog.models.map(\.id) == [
            "bonsai-2.5-27b-q2_0", "inkling-2-mini-ud-q4_k_xl", "maple-preview-2bit-mlx", "minicpm5-1b-mlx-4bit",
            "minicpm5-2b-int4g64", "other-model.q8_0", "qwen3.5-4b",
            "qwen3.6-35b-a3b", "qwen3.8-27b", "qwen3.8-27b-4bit",
        ])
        let other = try #require(catalog.descriptor(for: "other-model.q8_0"))
        #expect(other.product == nil)
        #expect(other.contextLength == 8_192)
        #expect(catalog.descriptor(for: "qwen3.8-27b-4bit")?.product == nil)
    }

    @Test func theEarliestRootAndTheCanonicalLabelWinADuplicateProductInstall() {
        let disk = Self.fileSystem(
            children: ["/first": ["/first/aaa.gturbo", "/first/qwen36.gturbo"],
                       "/second": ["/second/0.gturbo"]],
            probes: ["/first/aaa.gturbo": .complete(family: .qwen36),
                     "/first/qwen36.gturbo": .complete(family: .qwen36),
                     "/second/0.gturbo": .complete(family: .qwen36)])
        let product = EngineCatalogDiscovery.discover(roots: [Self.dir("/first"), Self.dir("/second")],
                                                      ggufDirectories: [], fileSystem: disk)
        #expect(product.models.map(\.id) == ["qwen3.6-35b-a3b"])
        #expect(product.models.first?.location == Self.dir("/first/qwen36.gturbo"))
        #expect(product.skipped.count == 2)
        #expect(product.skipped.allSatisfy { $0.reason.contains("/first/qwen36.gturbo") && $0.modelID == nil })

        let all = EngineCatalogDiscovery.discover(roots: [Self.dir("/first"), Self.dir("/second")],
                                                  ggufDirectories: [], scope: .all, fileSystem: disk)
        #expect(all.models.map(\.id) == ["qwen3.6-35b-a3b", "qwen3.6-35b-a3b@0", "qwen3.6-35b-a3b@aaa"])
    }

    @Test func productGGUFFilesMustBeReadableAndMTPSlotsMustCarryMTP() {
        let disk = Self.fileSystem(
            children: [:], probes: [:],
            ggufFiles: ["/g": ["/g/Qwen3.5-4B-UD-Q4_K_XL.gguf", "/g/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf",
                               "/g/Ternary-Bonsai-2.5-27B-Q2_0-mtp.gguf"]],
            headers: ["/g/Qwen3.5-4B-UD-Q4_K_XL.gguf": .failure(.truncated),
                      "/g/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf": .success(Self.meta(mtp: false)),
                      "/g/Ternary-Bonsai-2.5-27B-Q2_0-mtp.gguf": .success(Self.meta(mtp: false))])
        let catalog = EngineCatalogDiscovery.discover(roots: [], ggufDirectories: [Self.dir("/g")],
                                                      fileSystem: disk)
        #expect(catalog.models.isEmpty)
        #expect(catalog.unavailableReason(for: "qwen3.5-4b")?.contains("unreadable GGUF") == true)
        #expect(catalog.unavailableReason(for: "qwen3.8-27b")?.contains("MTP") == true)
        #expect(catalog.unavailableReason(for: "bonsai-2.5-27b-q2_0")?.contains("MTP") == true)
        #expect(EngineProductSlot.allCases.filter(\.requiresMTP) == [.deep, .bonsai27b])
    }

    @Test func aGGUFWithoutContextLengthOrSplitIntoShardsIsSkipped() {
        let disk = Self.fileSystem(
            children: [:], probes: [:],
            ggufFiles: ["/g": ["/g/Qwen3.5-4B-UD-Q4_K_XL.gguf", "/g/big-00001-of-00003.gguf"]],
            headers: ["/g/Qwen3.5-4B-UD-Q4_K_XL.gguf": .success(Self.meta(mtp: false, context: nil)),
                      "/g/big-00001-of-00003.gguf": .success(Self.meta(mtp: false))])
        let catalog = EngineCatalogDiscovery.discover(roots: [], ggufDirectories: [Self.dir("/g")],
                                                      scope: .all, fileSystem: disk)
        #expect(catalog.models.isEmpty)
        #expect(catalog.unavailableReason(for: "qwen3.5-4b")?.contains("context length") == true)
        #expect(catalog.unavailableReason(for: "big-00001-of-00003")?.contains("split") == true)
    }

    @Test func theFirstCopyOfAProductGGUFWins() {
        let disk = Self.fileSystem(
            children: [:], probes: [:],
            ggufFiles: ["/a": ["/a/Qwen3.5-4B-UD-Q4_K_XL.gguf"], "/b": ["/b/Qwen3.5-4B-UD-Q4_K_XL.gguf"]],
            headers: ["/a/Qwen3.5-4B-UD-Q4_K_XL.gguf": .success(Self.meta(mtp: false)),
                      "/b/Qwen3.5-4B-UD-Q4_K_XL.gguf": .success(Self.meta(mtp: false))])
        let catalog = EngineCatalogDiscovery.discover(roots: [], ggufDirectories: [Self.dir("/a"), Self.dir("/b")],
                                                      fileSystem: disk)
        #expect(catalog.models.map(\.location.path) == ["/a/Qwen3.5-4B-UD-Q4_K_XL.gguf"])
        #expect(catalog.skipped.map(\.location.path) == ["/b/Qwen3.5-4B-UD-Q4_K_XL.gguf"])
    }

    /// Inkling 2 Mini is offered from 32 GB up (launch scope): below that its
    /// installs are skipped in either scope with the reason under its
    /// identifier, and a request for it is told why.
    @Test func inkling2MiniNeedsThirtyTwoGigabytes() throws {
        let disk = Self.fileSystem(
            children: ["/first": ["/first/inkling2mini.gturbo", "/first/aaa.gturbo"],
                       "/second": ["/second/inkling2mini.gturbo"]],
            probes: ["/first/inkling2mini.gturbo": .complete(family: .inkling2Loop),
                     "/first/aaa.gturbo": .complete(family: .inkling2Loop),
                     "/second/inkling2mini.gturbo": .complete(family: .inkling2Loop)])
        let roots = [Self.dir("/first"), Self.dir("/second")]
        let reason = "Needs a Mac with 32 GB of memory or more."
        for memory: UInt64 in [16 << 30, 24 << 30, (32 << 30) - 1] {
            for scope in [EngineCatalogScope.product, .all] {
                let catalog = EngineCatalogDiscovery.discover(roots: roots, ggufDirectories: [], scope: scope,
                                                              fileSystem: disk, physicalMemory: memory)
                #expect(catalog.models.isEmpty)
                #expect(catalog.skipped.count == 3)
                #expect(catalog.skipped.allSatisfy { $0.reason == reason && $0.modelID == "inkling-2-mini-ud-q4_k_xl" })
                #expect(catalog.unavailableReason(for: "inkling-2-mini-ud-q4_k_xl") == reason)
            }
        }
        // From 32 GB: the earliest root, then the installer's label, wins.
        for memory: UInt64 in [32 << 30, 64 << 30] {
            let catalog = EngineCatalogDiscovery.discover(roots: roots, ggufDirectories: [],
                                                          fileSystem: disk, physicalMemory: memory)
            #expect(catalog.models.map(\.id) == ["inkling-2-mini-ud-q4_k_xl"])
            #expect(catalog.models.first?.location == Self.dir("/first/inkling2mini.gturbo"))
            #expect(catalog.skipped.count == 2)
        }
        // Under 48 GB its context is capped at 16,384 (any request, `max`
        // included); from 48 GB it runs to its native context.
        let at32 = try #require(EngineCatalogDiscovery.discover(
            roots: roots, ggufDirectories: [], fileSystem: disk, physicalMemory: 32 << 30).models.first)
        #expect(at32.hostContextCap == 16_384 && at32.contextLength == 262_144)
        #expect(at32.resolvedContext(requested: nil) == 16_384)
        #expect(at32.resolvedContext(requested: 16_384) == 16_384)
        #expect(at32.resolvedContext(requested: 8_192) == 8_192)
        #expect(at32.resolvedContext(requested: 65_536) == 16_384)
        let at64 = try #require(EngineCatalogDiscovery.discover(
            roots: roots, ggufDirectories: [], fileSystem: disk, physicalMemory: 64 << 30).models.first)
        #expect(at64.hostContextCap == nil && at64.resolvedContext(requested: nil) == 262_144)
        // Below the floor, the floor is the reason a request gets, even with a
        // gated copy found first.
        let mixed = Self.fileSystem(
            children: ["/lib": ["/lib/a.gturbo", "/lib/inkling2mini.gturbo"]],
            probes: ["/lib/a.gturbo": .notRunnable(family: "inkling2Loop", detail: "missing axes"),
                     "/lib/inkling2mini.gturbo": .complete(family: .inkling2Loop)])
        let small = EngineCatalogDiscovery.discover(roots: [Self.library], ggufDirectories: [],
                                                    fileSystem: mixed, physicalMemory: 16 << 30)
        #expect(small.unavailableReason(for: "inkling-2-mini-ud-q4_k_xl") == reason)
        // The runtime sizes this slot before loading (32K context), and only
        // this slot.
        let estimate = EngineProductSlot.inkling2Mini.estimatedResidentBytes(installBytes: 17_662_322_836,
                                                                             physicalMemory: 32 << 30)
        #expect(estimate == Inkling2MemoryFootprint.estimate(installBytes: 17_662_322_836,
                                                             physicalMemory: 32 << 30)?.bytes(contextTokens: 32_768))
        #expect(EngineProductSlot.allCases.filter {
            $0.estimatedResidentBytes(installBytes: 1 << 30, physicalMemory: 64 << 30) != nil
        } == [.inkling2Mini])
        // Only this slot has a floor of the engine's own.
        #expect(EngineProductSlot.allCases.filter { $0.minimumPhysicalMemory != nil } == [.inkling2Mini])
        #expect(EngineProductSlot.inkling2Mini.memoryFloorReason(physicalMemory: 32 << 30) == nil)
        #expect(EngineProductSlot.deep.memoryFloorReason(physicalMemory: 8 << 30) == nil)
    }

    /// The Unsloth file the native install is made from is a GGUF named like
    /// the slot's identifier; it never takes that identifier, in either
    /// scope, whether or not the native install is there.
    @Test func aGGUFNamedLikeANativeProductModelNeverTakesItsIdentifier() {
        let file = "/lib/gguf/Inkling-2-Mini-UD-Q4_K_XL.gguf"
        #expect(EngineCatalogDiscovery.ggufModelID(fileName: "Inkling-2-Mini-UD-Q4_K_XL.gguf")
                == EngineProductSlot.inkling2Mini.modelID)
        let disk = Self.fileSystem(
            children: [:], probes: [:],
            ggufFiles: ["/lib/gguf": [file]],
            headers: [file: .success(Self.meta(mtp: false))])
        let product = EngineCatalogDiscovery.discover(roots: [], ggufDirectories: [Self.gguf],
                                                      fileSystem: disk, physicalMemory: Self.roomy)
        #expect(product.models.isEmpty)
        #expect(product.skipped.map(\.modelID) == [nil])
        #expect(product.unavailableReason(for: "inkling-2-mini-ud-q4_k_xl") == nil)
        for memory: UInt64 in [16 << 30, Self.roomy] {
            let all = EngineCatalogDiscovery.discover(roots: [], ggufDirectories: [Self.gguf], scope: .all,
                                                      fileSystem: disk, physicalMemory: memory)
            #expect(all.models.isEmpty)
            #expect(all.skipped.first?.reason == "another model already uses the identifier inkling-2-mini-ud-q4_k_xl")
        }
    }

    @Test func gatedAndPartialInstallsAreReported() {
        let disk = Self.fileSystem(
            children: ["/lib": ["/lib/x.gturbo", "/lib/y.gturbo"]],
            probes: ["/lib/x.gturbo": .notRunnable(family: "minicpm5", detail: "missing axes q"),
                     "/lib/y.gturbo": .partial("no receipt")])
        let catalog = EngineCatalogDiscovery.discover(roots: [Self.library], ggufDirectories: [], fileSystem: disk)
        #expect(catalog.models.isEmpty)
        #expect(catalog.unavailableReason(for: "minicpm5-2b-int4g64")?.hasPrefix("not runnable") == true)
        #expect(catalog.skipped.contains { $0.reason == "incomplete install: no receipt" && $0.modelID == nil })
    }

    @Test func anExplicitModelIsConsideredFirst() {
        let disk = Self.fileSystem(
            children: [:], probes: [:],
            ggufFiles: ["/g": ["/g/Qwen3.5-4B-UD-Q4_K_XL.gguf"]],
            headers: ["/g/Qwen3.5-4B-UD-Q4_K_XL.gguf": .success(Self.meta(mtp: false)),
                      "/elsewhere/Qwen3.5-4B-UD-Q4_K_XL.gguf": .success(Self.meta(mtp: false))])
        let catalog = EngineCatalogDiscovery.discover(
            roots: [], ggufDirectories: [Self.dir("/g")],
            explicitModel: Self.file("/elsewhere/Qwen3.5-4B-UD-Q4_K_XL.gguf"), fileSystem: disk)
        #expect(catalog.descriptor(for: "qwen3.5-4b")?.location.path == "/elsewhere/Qwen3.5-4B-UD-Q4_K_XL.gguf")
    }

    @Test func defaultRootsKeepTheServerOrderAndGGUFAddsItsDirectory() {
        let home = Self.dir("/Users/someone")
        let roots = EngineCatalogDiscovery.defaultRoots(
            userDefaults: UserDefaults(suiteName: "engine-tests-\(UUID().uuidString)")!,
            environment: [EngineCatalogDiscovery.libraryRootEnvironmentKey: "/custom"],
            executableURL: nil,
            currentDirectoryURL: Self.dir("/nowhere"),
            applicationSupportURL: Self.dir("/Users/someone/Library/Application Support"),
            homeDirectoryURL: home,
            fileExists: { _ in false })
        #expect(roots.map(\.path) == ["/custom", "/Users/someone/llm-models",
                                      "/Users/someone/Library/Application Support/Mference"])
        #expect(EngineCatalogDiscovery.defaultGGUFDirectories(roots: roots, homeDirectoryURL: home).map(\.path)
                == roots.map(\.path) + ["/Users/someone/llm-models/gguf"])
    }

    @Test func probeTellsOrdinaryStagingAndLockedDirectoriesApart() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("engine-probe-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let plain = root.appendingPathComponent("plain", isDirectory: true)
        let staging = root.appendingPathComponent("m.gturbo.partial", isDirectory: true)
        try FileManager.default.createDirectory(at: plain, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        #expect(EngineInstallProbe.probe(directory: plain) == .notAModelDirectory)
        #expect(EngineInstallProbe.probe(directory: staging) == .partial("in-progress install staging directory"))
        // A completed install leaves a stale, unlocked lock file behind.
        let stale = root.appendingPathComponent("plain.install.lock")
        try Data().write(to: stale)
        #expect(!EngineInstallProbe.installLockIsHeld(at: stale))
        #expect(EngineInstallProbe.probe(directory: plain) == .notAModelDirectory)
    }

    /// An install of a retired family is left on disk but never served: the
    /// probe names it as not runnable, even the developer catalog scope does
    /// not list it, and a request for it is told why. A legacy manifest with
    /// no `arch.family` (a Gemma 4 install) is not complete either.
    @Test func aRetiredFamilyInstallIsNotRunnableAndNeverListed() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("engine-retired-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        func install(_ name: String, arch: [String: Any], modelID: String? = nil) throws -> URL {
            let directory = root.appendingPathComponent("\(name).gturbo", isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let manifest: [String: Any] = ["modelID": modelID ?? "test/\(name)", "arch": arch]
            try JSONSerialization.data(withJSONObject: manifest)
                .write(to: directory.appendingPathComponent("manifest.json"))
            return directory
        }
        let retired = try install("flashnext-iq3", arch: ["family": "qwen38flashnextudiq3xxs"])
        let legacy = try install("legacy", arch: [:])
        let swiftQwen = try install("swift-qwen", arch: ["family": "qwen38"],
                                    modelID: "swift-qwen3.8-27b-int4g64")

        guard case .notRunnable(let family, let detail) = EngineInstallProbe.probe(directory: retired) else {
            Issue.record("a retired-family install must be reported as not runnable")
            return
        }
        #expect(family == "qwen38flashnextudiq3xxs")
        #expect(detail.contains("retired family"))
        // No arch.family (the original Gemma 4 format): not runnable, under
        // its modelID.
        guard case .notRunnable(let legacyID, let legacyDetail) =
            EngineInstallProbe.probe(directory: legacy) else {
            Issue.record("a manifest without arch.family must be reported as not runnable")
            return
        }
        #expect(legacyID == "test/legacy")
        #expect(legacyDetail.contains("no arch.family"))
        // Swift-Qwen3.8 carries the kept qwen38 family; refused by checkpoint ID.
        guard case .notRunnable(let swiftID, let swiftDetail) =
            EngineInstallProbe.probe(directory: swiftQwen) else {
            Issue.record("a Swift-Qwen3.8 install must be reported as not runnable")
            return
        }
        #expect(swiftID == "swift-qwen3.8-27b-int4g64")
        #expect(swiftDetail.contains("Swift-Qwen3.8"))

        let catalog = EngineCatalogDiscovery.discover(roots: [root], ggufDirectories: [], scope: .all,
                                                      fileSystem: .live)
        #expect(catalog.models.isEmpty)
        let reason = try #require(catalog.unavailableReason(for: "qwen38flashnextudiq3xxs"))
        #expect(reason.hasPrefix("not runnable"))
        #expect(reason.contains("retired family"))
    }

    /// Every family's identifier, pinned to the server's spelling at the time
    /// of the lift (`ServerFamilyModelID`), so a change on either side shows.
    @Test func nativeIdentifiersMatchTheServer() {
        let expected: [ModelFamily: String] = [
            .qwen36: "qwen3.6-35b-a3b",
            .qwen38: "qwen3.8-27b-4bit",
            .maple: "maple-preview-2bit-mlx",
            .minicpm5: "minicpm5-2b-int4g64",
            .minicpm5_1b: "minicpm5-1b-mlx-4bit",
            .inkling2Loop: "inkling-2-mini-ud-q4_k_xl",
        ]
        #expect(Set(expected.keys) == Set(ModelFamily.allCases))
        for (family, id) in expected {
            #expect(EngineNativeModelID.modelID(for: family) == id)
        }
        // No kept family has a checkpoint variant with its own identifier.
        #expect(EngineNativeModelID.modelID(for: .qwen38, checkpointID: CheckpointIdentity.baseQwen38)
                == CheckpointIdentity.baseQwen38)
        #expect(EngineNativeModelID.modelID(for: .qwen36, checkpointID: "test/qwen36-requant")
                == "qwen3.6-35b-a3b")
        #expect(EngineNativeModelID.modelID(for: .qwen36) == EngineProductSlot.qwen36.modelID)
        #expect(EngineNativeModelID.modelID(for: .minicpm5) == EngineProductSlot.minicpm5.modelID)
        #expect(EngineNativeModelID.modelID(for: .minicpm5_1b) == EngineProductSlot.minicpm5_1b.modelID)
        #expect(EngineNativeModelID.modelID(for: .maple) == EngineProductSlot.maple.modelID)
        #expect(EngineNativeModelID.modelID(for: .inkling2Loop) == EngineProductSlot.inkling2Mini.modelID)
        #expect(EngineCatalogDiscovery.ggufModelID(fileName: "Foo-Bar.Q4.gguf") == "foo-bar.q4")
    }

    @Test func aGGUFNamedLikeAProductModelNeverDisplacesIt() {
        let disk = Self.fileSystem(
            children: [:], probes: [:],
            ggufFiles: ["/a": ["/a/qwen3.8-27b.gguf"], "/b": ["/b/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"]],
            headers: ["/a/qwen3.8-27b.gguf": .success(Self.meta(mtp: false)),
                      "/b/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf": .success(Self.meta(mtp: true))])
        let catalog = EngineCatalogDiscovery.discover(roots: [], ggufDirectories: [Self.dir("/a"), Self.dir("/b")],
                                                      scope: .all, fileSystem: disk)
        #expect(catalog.models.map(\.location.path) == ["/b/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"])
        #expect(catalog.skipped.contains { $0.location.path == "/a/qwen3.8-27b.gguf" })
    }

    @Test func aSymlinkedGGUFIsListedAndMeasuredThroughItsTarget() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("engine-gguf-link-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let target = root.appendingPathComponent("blob")
        try Data(count: 1_234).write(to: target)
        let link = root.appendingPathComponent("Model.gguf")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: target)
        #expect(EngineCatalogDiscovery.ggufFiles(in: root).map(\.lastPathComponent) == ["Model.gguf"])
        #expect(EngineCatalogDiscovery.bytesOnDisk(link) == 1_234)
    }
}
