import Foundation
import ImageIO
@testable import MferenceAppCore
import Photos
import Synchronization
import Testing

// The Photos library as a Find source, model-free: a fake library whose
// items are plain PNGs, the hashing embedder, the fake permission broker.

/// A Photos library over plain picture files, keyed by made-up identifiers.
final class FakePhotoLibrary: FindPhotoLibrary {
    private let items: Mutex<[(id: String, file: URL, date: Date)]>
    private let handler = Mutex<(@Sendable () -> Void)?>(nil)

    init(_ files: [URL]) {
        items = Mutex(files.enumerated().map { ("ASSET-\($0.offset)/L0/001", $0.element,
                                                Date(timeIntervalSince1970: 1_700_000_000 + Double($0.offset))) })
    }

    func remove(_ index: Int) {
        items.withLock { _ = $0.remove(at: index) }
        handler.withLock { $0 }?()
    }

    func entries() async -> [FindCrawlEntry] {
        items.withLock { items in
            items.map { item in
                FindCrawlEntry(url: FindPhotos.url(identifier: item.id,
                                                   name: FindPhotos.name(kind: .image, isScreenshot: false, date: item.date)),
                               kind: .image, size: 1, modified: item.date, inode: 0, device: 0)
            }
        }
    }

    func readableFile(for url: URL, kind: FindKind, maxPixel: Int) async -> (url: URL, isTemporary: Bool)? {
        let id = FindPhotos.identifier(of: url)
        return items.withLock { items in items.first { $0.id == id }.map { ($0.file, false) } }
    }

    func observeChanges(_ handler: @escaping @Sendable () -> Void) {
        self.handler.withLock { $0 = handler }
    }

    func picture(for url: URL, maxPixel: Int) async -> CGImage? {
        guard let file = await readableFile(for: url, kind: .image, maxPixel: maxPixel)?.url,
              let source = CGImageSourceCreateWithURL(file as CFURL, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }
}

@Suite struct FindPhotosTests {
    @Test func libraryPathsRoundTripTheirIdentifier() {
        let url = FindPhotos.url(identifier: "9F0C-12/L0/001", name: "Photo 2024-05-03 at 18.22.00.heic")
        #expect(FindPhotos.isLibraryItem(url))
        #expect(FindPhotos.identifier(of: url) == "9F0C-12/L0/001")
        #expect(url.lastPathComponent == "Photo 2024-05-03 at 18.22.00.heic")
        #expect(!FindPhotos.isLibraryItem(URL(filePath: "/Users/me/Pictures/a.png")))
        #expect(FindPhotos.identifier(of: URL(filePath: "/Users/me/a.png")) == nil)
    }

    @Test func libraryNamesSayWhatAndWhen() {
        let date = Date(timeIntervalSince1970: 0)
        #expect(FindPhotos.name(kind: .video, isScreenshot: false, date: date).hasPrefix("Video "))
        #expect(FindPhotos.name(kind: .image, isScreenshot: true, date: date).hasPrefix("Screenshot "))
        #expect(FindPhotos.name(kind: .image, isScreenshot: false, date: nil) == "Photo.heic")
    }

    @Test func photosPermissionMapsLikeTheOthers() {
        #expect(LivePermissionBroker.state(PHAuthorizationStatus.authorized) == .granted)
        #expect(LivePermissionBroker.state(PHAuthorizationStatus.limited) == .granted)
        #expect(LivePermissionBroker.state(PHAuthorizationStatus.denied) == .needsSystemSettings)
        #expect(LivePermissionBroker.state(PHAuthorizationStatus.notDetermined) == .notDetermined)
        #expect(LivePermissionBroker.settingsURL(for: .photos).absoluteString.hasSuffix("Privacy_Photos"))
    }

    @Test func readerSendsFilesToTheBaseAndLibraryItemsThroughTheLibrary() async throws {
        let folder = try FindTempFolder()
        let red = try FindMediaFixtures.png(folder.url.appending(path: "red.png"), red: 1, green: 0, blue: 0)
        let library = FakePhotoLibrary([red])
        let reader = FindLibraryReader(base: InProcessFindExtractor(), library: library)
        let item = try #require(await library.entries().first?.url)
        let options = FindExtractOptions(ocr: false, media: true)
        #expect(await reader.extract(item, kind: .image, options: options).chunks.count == 1)
        #expect(await reader.extract(red, kind: .image, options: options).chunks.count == 1)
        let missing = FindPhotos.url(identifier: "GONE", name: "Photo.heic")
        if case .skipped = await reader.extract(missing, kind: .image, options: options) {} else {
            Issue.record("an item not on this Mac was not skipped")
        }
        let prepared = await reader.prepare(item, kind: .image, pieces: [FindMediaPiece(modality: .image, tokens: 70)])
        let file = try #require(prepared.first ?? nil)
        defer { try? FileManager.default.removeItem(at: file) }
        #expect(FileManager.default.fileExists(atPath: file.path))
        #expect(await reader.prepare(missing, kind: .image, pieces: [FindMediaPiece(modality: .image)]) == [nil])
    }

    @Test func libraryItemsAreIndexedSearchedAndDropped() async throws {
        let folder = try FindTempFolder()
        let pictures = try FindTempFolder()
        let red = try FindMediaFixtures.png(pictures.url.appending(path: "a.png"), red: 0.9, green: 0.1, blue: 0.1)
        let blue = try FindMediaFixtures.png(pictures.url.appending(path: "b.png"), red: 0.1, green: 0.1, blue: 0.9)
        try FindMediaFixtures.fillers(in: folder)
        let library = FakePhotoLibrary([red, blue])
        let embedder = HashingTextEmbedder(media: true)
        let indexer = FindIndexer(
            index: try FindIndex(directory: nil), extractor: FindLibraryReader(base: InProcessFindExtractor(), library: library),
            embedder: embedder,
            configuration: .init(roots: [folder.url], rules: FindCrawlRules(readImages: false, readMedia: true), photos: true),
            photoLibrary: library)
        try await indexer.syncAll()
        await indexer.waitUntilIdle()
        let index = await indexer.index
        #expect(try await index.fileCount(root: FindPhotos.rootPath) == 2)
        #expect(await indexer.publish().learnedFiles == 32)

        let results = try await FindSearcher(index: index, embedder: embedder).search(FindQuery("red", kind: .images))
        let hit = try #require(results.strong.first)
        #expect(FindPhotos.isLibraryItem(hit.url))
        #expect(FindPhotos.identifier(of: hit.url) == "ASSET-0/L0/001")
        #expect(hit.reasons.contains(.picture))

        // Deleted in Photos: gone from Find.
        library.remove(0)
        try await indexer.syncPhotos()
        #expect(try await index.fileCount(root: FindPhotos.rootPath) == 1)

        // The library turned off: all of it leaves the index; folders stay.
        try await indexer.setConfiguration(.init(roots: [folder.url], rules: FindCrawlRules(readMedia: true)))
        #expect(try await index.fileCount(root: FindPhotos.rootPath) == 0)
        #expect(await indexer.publish().totalFiles == 30)
    }
}

@MainActor
@Suite struct FindPhotosServiceTests {
    func service(_ broker: FakePermissionBroker, library: FakePhotoLibrary) -> FindService {
        let settings = FindSettings(store: InMemorySettingsStore())
        settings.learnOnlyOnPower = false
        return FindService(settings: settings, dependencies: .init(
            indexDirectory: nil, model: PresentFindModel(), mediaModel: PresentFindModel(),
            makeEmbedder: { HashingTextEmbedder(media: $1 != nil) },
            extractor: FindLibraryReader(base: InProcessFindExtractor(), library: library), watchFolders: false,
            photoLibrary: library, permissions: broker))
    }

    @Test func turningOnAsksForPhotosThenReadsTheLibrary() async throws {
        let folder = try FindTempFolder()
        let pictures = try FindTempFolder()
        let red = try FindMediaFixtures.png(pictures.url.appending(path: "a.png"), red: 0.9, green: 0.1, blue: 0.1)
        let broker = FakePermissionBroker()
        let find = service(broker, library: FakePhotoLibrary([red]))
        await find.turnOn(roots: [folder.url])
        await find.setPhotosLibrary(true)
        #expect(broker.calls.contains(.request(.photos)))
        #expect(find.photosAccess == .granted)
        #expect(find.settings.photosLibrary)
        #expect(find.photosCount == 1)
        await find.setPhotosLibrary(false)
        #expect(!find.settings.photosLibrary)
        await find.turnOff()
    }

    @Test func aRefusalLeavesItOffAndPointsToSystemSettings() async throws {
        let folder = try FindTempFolder()
        let broker = FakePermissionBroker(answers: [.photos: .needsSystemSettings])
        let find = service(broker, library: FakePhotoLibrary([]))
        await find.turnOn(roots: [folder.url])
        await find.setPhotosLibrary(true)
        #expect(!find.settings.photosLibrary)
        #expect(find.photosAccess == .needsSystemSettings)
        find.openPhotosSettings()
        for _ in 0..<50 where !broker.calls.contains(.openSystemSettings(.photos)) {
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(broker.calls.contains(.openSystemSettings(.photos)))
        await find.turnOff()
    }

    @Test func accessTakenAwayTurnsTheLibraryOff() async throws {
        let folder = try FindTempFolder()
        let broker = FakePermissionBroker()
        let find = service(broker, library: FakePhotoLibrary([]))
        await find.turnOn(roots: [folder.url])
        await find.setPhotosLibrary(true)
        #expect(find.settings.photosLibrary)
        broker.setState(.needsSystemSettings, for: .photos)
        await find.refreshPhotosAccess()
        #expect(!find.settings.photosLibrary)
        await find.turnOff()
    }
}
