import Foundation
@testable import MferenceAppCore
import MferenceEngine
import Testing

@Suite struct ModelsImportTests {
    /// A tiny "GGUF" and the pinned source it matches.
    static func fixture(_ model: AppModel = .quick, bytes: Int = 64_000, seed: UInt8 = 3)
        -> (data: Data, source: DownloadSource) {
        let data = Data("GGUF".utf8) + fakeModelBytes(bytes - 4, seed: seed)
        let name = model == .quick ? DownloadSource.quick.fileName : DownloadSource.deep.fileName
        let source = DownloadSource(model: model, repository: "test/test", revision: "abc", fileName: name,
                                    sizeBytes: UInt64(data.count), sha256: sha256Hex(data), licence: "Apache 2.0")
        return (data, source)
    }

    static func importer(_ destination: URL, _ sources: [DownloadSource]) -> ModelFileImporter {
        ModelFileImporter(destination: destination, sources: sources, availableBytes: { _ in 1 << 40 })
    }

    static func visibleFiles(_ directory: URL) -> [String] {
        ((try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? []).sorted()
    }

    @Test func validFileIsCopiedAndTheOriginalKept() async throws {
        let (data, source) = Self.fixture()
        let picked = temporaryDirectory()
        let destination = temporaryDirectory()
        defer {
            try? FileManager.default.removeItem(at: picked)
            try? FileManager.default.removeItem(at: destination)
        }
        // The person's file may have any name.
        let original = picked.appending(path: "my-quick-model.gguf")
        try data.write(to: original)

        let model = try await Self.importer(destination, [source]).importFile(at: original)
        #expect(model == .quick)
        #expect(try Data(contentsOf: destination.appending(path: source.fileName)) == data)
        #expect(try Data(contentsOf: original) == data, "the original is untouched")
        #expect(Self.visibleFiles(destination) == [source.fileName], "no staging file is left")
    }

    @Test func notAGGUFIsRefused() async throws {
        let (_, source) = Self.fixture()
        let picked = temporaryDirectory()
        let destination = temporaryDirectory()
        defer {
            try? FileManager.default.removeItem(at: picked)
            try? FileManager.default.removeItem(at: destination)
        }
        let file = picked.appending(path: "notes.gguf")
        try Data(repeating: 7, count: Int(source.sizeBytes)).write(to: file)
        await #expect(throws: ModelImportError.notGGUF) {
            try await Self.importer(destination, [source]).importFile(at: file)
        }
        #expect(Self.visibleFiles(destination).isEmpty)
    }

    @Test func unknownGGUFIsRefused() async throws {
        let (_, source) = Self.fixture()
        let picked = temporaryDirectory()
        let destination = temporaryDirectory()
        defer {
            try? FileManager.default.removeItem(at: picked)
            try? FileManager.default.removeItem(at: destination)
        }
        // Same size, different contents: the hash decides.
        let (other, _) = Self.fixture(seed: 9)
        let file = picked.appending(path: source.fileName)
        try other.write(to: file)
        await #expect(throws: ModelImportError.notAProductModel) {
            try await Self.importer(destination, [source]).importFile(at: file)
        }
        // A different size is refused without hashing.
        let small = picked.appending(path: "small.gguf")
        try (Data("GGUF".utf8) + Data(count: 10)).write(to: small)
        await #expect(throws: ModelImportError.notAProductModel) {
            try await Self.importer(destination, [source]).importFile(at: small)
        }
        #expect(Self.visibleFiles(destination).isEmpty)
    }

    @Test func identicalFileInPlaceIsLeftAlone() async throws {
        let (data, source) = Self.fixture()
        let picked = temporaryDirectory()
        let destination = temporaryDirectory()
        defer {
            try? FileManager.default.removeItem(at: picked)
            try? FileManager.default.removeItem(at: destination)
        }
        let original = picked.appending(path: "q.gguf")
        try data.write(to: original)
        let installed = destination.appending(path: source.fileName)
        try await Self.importer(destination, [source]).importFile(at: original)
        let before = try FileManager.default.attributesOfItem(atPath: installed.path)[.modificationDate] as? Date
        try await Self.importer(destination, [source]).importFile(at: original)
        let after = try FileManager.default.attributesOfItem(atPath: installed.path)[.modificationDate] as? Date
        #expect(before == after)
        #expect(Self.visibleFiles(destination) == [source.fileName])
    }

    @Test func aDifferentFileUnderTheNameIsNeverReplaced() async throws {
        let (data, source) = Self.fixture()
        let picked = temporaryDirectory()
        let destination = temporaryDirectory()
        defer {
            try? FileManager.default.removeItem(at: picked)
            try? FileManager.default.removeItem(at: destination)
        }
        let original = picked.appending(path: "q.gguf")
        try data.write(to: original)
        // The person's own file, or a damaged one: either way it stays.
        let theirs = Data("GGUF their own build".utf8)
        let installed = destination.appending(path: source.fileName)
        try theirs.write(to: installed)
        do {
            try await Self.importer(destination, [source]).importFile(at: original)
            Issue.record("must not replace an existing file")
        } catch let error as ModelImportError {
            #expect(error == .nameTaken(existing: installed))
            #expect(error.fileToShow == installed)
            #expect(error.errorDescription?.hasPrefix("Another file already uses this name in") == true)
        }
        #expect(try Data(contentsOf: installed) == theirs)
        #expect(Self.visibleFiles(destination) == [source.fileName], "no staging file is left")
    }

    @Test @MainActor func libraryShowsAnImportedModelInstalled() async throws {
        let (data, source) = Self.fixture()
        let picked = temporaryDirectory()
        let destination = temporaryDirectory()
        defer {
            try? FileManager.default.removeItem(at: picked)
            try? FileManager.default.removeItem(at: destination)
        }
        let original = picked.appending(path: "quick.gguf")
        try data.write(to: original)
        let file = destination.appending(path: source.fileName)
        let engine = EngineService(discover: {
            AppServices.fakeCatalog(FileManager.default.fileExists(atPath: file.path) ? [.quick] : [])
        }, makeHost: { catalog, observer in
            EngineHost(catalog: catalog, backends: [FakeEngineBackend(engine: .llamaCpp)], observer: observer)
        })
        let downloads = DownloadManager(destination: destination, sources: [.quick: source], allowedHosts: [])
        let library = ModelLibrary(engine: engine, downloads: downloads,
                                   settings: AppSettings(store: InMemorySettingsStore()), memoryFit: bigMac, chipName: nil)
        await library.refresh()
        #expect(library.row(for: .quick)?.state == .downloadable)
        #expect(try await library.importModelFile(at: original) == .quick)
        #expect(library.row(for: .quick)?.state == .installed)
        #expect(FileManager.default.fileExists(atPath: original.path))
    }
}
