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
        children: ["/lib": ["/lib/flashnext.gturbo", "/lib/gguf", "/lib/maple.gturbo", "/lib/minicpm5.gturbo",
                            "/lib/half.gturbo", "/lib/qwen36.gturbo"]],
        probes: ["/lib/qwen36.gturbo": .complete(family: .qwen36),
                 "/lib/minicpm5.gturbo": .complete(family: .minicpm5),
                 "/lib/maple.gturbo": .complete(family: .maple),
                 "/lib/flashnext.gturbo": .complete(family: .qwen38flashnextudiq3xxs),
                 "/lib/half.gturbo": .partial("receipt missing")],
        ggufFiles: ["/lib/gguf": [quickFile.path, deepFile.path, "/lib/gguf/Other-Model.Q8_0.gguf"]],
        headers: [quickFile.path: .success(meta(mtp: false)),
                  deepFile.path: .success(meta(mtp: true)),
                  "/lib/gguf/Other-Model.Q8_0.gguf": .success(meta(mtp: false, context: 8_192))])

    @Test func productScopeListsExactlyTheFiveProductModels() throws {
        let catalog = EngineCatalogDiscovery.discover(roots: [Self.library], ggufDirectories: [Self.gguf],
                                                      fileSystem: Self.standardDisk)
        #expect(catalog.models.map(\.id) == [
            "maple-preview-2bit-mlx", "minicpm5-2b-int4g64", "qwen3.5-4b", "qwen3.6-35b-a3b", "qwen3.8-27b",
        ])
        #expect(Set(catalog.models.compactMap(\.product)) == Set(EngineProductSlot.allCases))
        let deep = try #require(catalog.descriptor(for: "qwen3.8-27b"))
        #expect(deep.engine == .llamaCpp)
        #expect(deep.location == Self.deepFile)
        #expect(deep.contextLength == 262_144)
        #expect(deep.displayName == EngineProductSlot.deep.displayName)
        let qwen36 = try #require(catalog.descriptor(for: "qwen3.6-35b-a3b"))
        #expect(qwen36.engine == .mference)
        #expect(qwen36.format == .gturbo(family: .qwen36, checkpointID: nil))
        #expect(qwen36.contextLength == ModelFamily.qwen36.maximumContext)
        #expect(qwen36.sizeBytes == 42)

        // Flash-Next is found but not offered, and says so when requested.
        #expect(catalog.descriptor(for: "unsloth-qwen3.8-flash-next-ud-iq3_xxs") == nil)
        #expect(catalog.unavailableReason(for: "unsloth-qwen3.8-flash-next-ud-iq3_xxs")?
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
                                                      scope: .all, fileSystem: Self.standardDisk)
        #expect(catalog.models.map(\.id) == [
            "maple-preview-2bit-mlx", "minicpm5-2b-int4g64", "other-model.q8_0", "qwen3.5-4b",
            "qwen3.6-35b-a3b", "qwen3.8-27b", "unsloth-qwen3.8-flash-next-ud-iq3_xxs",
        ])
        let other = try #require(catalog.descriptor(for: "other-model.q8_0"))
        #expect(other.product == nil)
        #expect(other.contextLength == 8_192)
        #expect(catalog.descriptor(for: "unsloth-qwen3.8-flash-next-ud-iq3_xxs")?.product == nil)
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

    @Test func productGGUFFilesMustBeReadableAndDeepMustCarryMTP() {
        let disk = Self.fileSystem(
            children: [:], probes: [:],
            ggufFiles: ["/g": ["/g/Qwen3.5-4B-UD-Q4_K_XL.gguf", "/g/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"]],
            headers: ["/g/Qwen3.5-4B-UD-Q4_K_XL.gguf": .failure(.truncated),
                      "/g/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf": .success(Self.meta(mtp: false))])
        let catalog = EngineCatalogDiscovery.discover(roots: [], ggufDirectories: [Self.dir("/g")],
                                                      fileSystem: disk)
        #expect(catalog.models.isEmpty)
        #expect(catalog.unavailableReason(for: "qwen3.5-4b")?.contains("unreadable GGUF") == true)
        #expect(catalog.unavailableReason(for: "qwen3.8-27b")?.contains("MTP") == true)
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

    /// Every family's identifier, pinned to the server's spelling at the time
    /// of the lift (`ServerFamilyModelID`), so a change on either side shows.
    @Test func nativeIdentifiersMatchTheServer() {
        let expected: [ModelFamily: String] = [
            .gemma4: "gemma-4-26b-a4b-it",
            .qwen36: "qwen3.6-35b-a3b",
            .qwen38: "qwen3.8-27b-4bit",
            .deepseekV4Flash: "deepseek-v4-flash-2bit-dq",
            .inklingSmall: "inkling-small-4bit",
            .maple: "maple-preview-2bit-mlx",
            .flint: "flint-40b-a2.5b-jang-4m",
            .qwen38flashnext: "qwen3.8-flash-next-int4g64",
            .qwen38flashnextudiq3xxs: "unsloth-qwen3.8-flash-next-ud-iq3_xxs",
            .minicpm5: "minicpm5-2b-int4g64",
            .glm53Flash: "glm-5.3-flash-mlx-mixed-4-8bit",
        ]
        #expect(Set(expected.keys) == Set(ModelFamily.allCases))
        for (family, id) in expected {
            #expect(EngineNativeModelID.modelID(for: family) == id)
        }
        #expect(EngineNativeModelID.modelID(for: .gemma4, checkpointID: CheckpointIdentity.gemma4QAT)
                == CheckpointIdentity.gemma4QAT)
        #expect(EngineNativeModelID.modelID(for: .qwen38, checkpointID: CheckpointIdentity.swiftQwen38)
                == CheckpointIdentity.swiftQwen38)
        #expect(EngineNativeModelID.modelID(for: .qwen36) == EngineProductSlot.qwen36.modelID)
        #expect(EngineNativeModelID.modelID(for: .minicpm5) == EngineProductSlot.minicpm5.modelID)
        #expect(EngineNativeModelID.modelID(for: .maple) == EngineProductSlot.maple.modelID)
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
