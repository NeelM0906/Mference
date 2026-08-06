import Foundation
import Mference
import Testing

@testable import MferenceAppCore

@Suite struct ModelLibraryTests {
    private let rootA = URL(fileURLWithPath: "/library/a", isDirectory: true)
    private let rootB = URL(fileURLWithPath: "/library/b", isDirectory: true)

    private func url(_ root: URL, _ name: String) -> URL {
        root.appendingPathComponent(name, isDirectory: true)
    }

    @Test func everyShippedModelGetsAnEntry() {
        let catalog = ModelLibrary.catalog(
            roots: [rootA],
            listSubdirectories: { _ in [] },
            probe: { _ in nil })

        #expect(catalog.map(\.descriptor) == ModelLibrary.shippedDescriptors)
        #expect(catalog.allSatisfy { !$0.isInstalled })
    }

    @Test func detectsInstalledFamiliesByManifestNotDirectoryName() {
        // dsv4.gturbo does not match the canonical deepseekv4flash.gturbo
        // name; detection has to come from the manifest's declared family.
        let dsv4 = url(rootA, "dsv4.gturbo")
        let qwen = url(rootA, "qwen36.gturbo")
        let catalog = ModelLibrary.catalog(
            roots: [rootA],
            listSubdirectories: { _ in [dsv4, qwen] },
            probe: { directory in
                switch directory {
                case dsv4: (.deepseekV4Flash, .complete)
                case qwen: (.qwen36, .complete)
                default: nil
                }
            })

        #expect(catalog.first { $0.descriptor.family == .deepseekV4Flash }?.installedURL == dsv4)
        #expect(catalog.first { $0.descriptor.family == .qwen36 }?.installedURL == qwen)
        #expect(catalog.first { $0.descriptor.family == .gemma4 }?.installedURL == nil)
    }

    @Test func earlierRootWinsAndIncompleteInstallsAreSkipped() {
        let partialInA = url(rootA, "gemma4.gturbo")
        let completeInA = url(rootA, "qwen36.gturbo")
        let completeInB = url(rootB, "qwen36.gturbo")
        let unknownInB = url(rootB, "mystery.gturbo")
        let catalog = ModelLibrary.catalog(
            roots: [rootA, rootB],
            listSubdirectories: { root in
                root == rootA ? [partialInA, completeInA] : [completeInB, unknownInB]
            },
            probe: { directory in
                switch directory {
                case partialInA: (.gemma4, .partial("truncated"))
                case completeInA, completeInB: (.qwen36, .complete)
                default: nil
                }
            })

        #expect(catalog.first { $0.descriptor.family == .qwen36 }?.installedURL == completeInA)
        #expect(catalog.first { $0.descriptor.family == .gemma4 }?.installedURL == nil)
    }

    @Test func scanResolvesSymlinkedLibraryEntriesToTheirRealLocation() throws {
        let base = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("ModelLibraryTests-\(UUID().uuidString)", isDirectory: true)
        let realInstall = base.appendingPathComponent("real/model.gturbo", isDirectory: true)
        let libraryRoot = base.appendingPathComponent("library", isDirectory: true)
        let fileManager = FileManager.default
        try fileManager.createDirectory(at: realInstall, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: libraryRoot, withIntermediateDirectories: true)
        defer { try? fileManager.removeItem(at: base) }
        try fileManager.createSymbolicLink(
            at: libraryRoot.appendingPathComponent("model.gturbo", isDirectory: true),
            withDestinationURL: realInstall)

        let listed = ModelLibrary.gturboSubdirectories(of: libraryRoot)
        #expect(listed.map(\.lastPathComponent) == ["model.gturbo"])
        #expect(listed.first?.path == realInstall.resolvingSymlinksInPath().path)
    }

    @Test func candidateRootsPutOverrideFirstRememberedParentLastAndDeduplicate() throws {
        let suiteName = "ModelLibraryTests-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suiteName))
        defer { defaults.removePersistentDomain(forName: suiteName) }
        defaults.set("/custom/library", forKey: ModelLibrary.rootStorageKey)

        let remembered = URL(fileURLWithPath: "/custom/library/qwen36.gturbo", isDirectory: true)
        let roots = ModelLibrary.candidateRoots(
            rememberedDirectory: remembered,
            userDefaults: defaults)

        #expect(roots.first?.path == "/custom/library")
        #expect(roots.filter { $0.path == "/custom/library" }.count == 1)
        #expect(Set(roots.map(\.path)).count == roots.count)
    }
}
