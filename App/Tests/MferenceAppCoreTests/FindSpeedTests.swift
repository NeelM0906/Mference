import AppKit
import CoreText
import Foundation
import ImageIO
@testable import MferenceAppCore
import Synchronization
import Testing
import UniformTypeIdentifiers

// Find's indexing speed-ups, model-free: several readers, learning while
// reading, the quick text check, pictures prepared once, batched learning,
// and hurrying through a large backlog.

/// A reader that takes `delay` per file and records what it was asked.
final class SlowReader: FindExtracting {
    let delay: Duration
    private let state = Mutex((inFlight: 0, maxInFlight: 0, hurry: [Bool](), relaxed: 0))
    let parallelism: Int

    init(delay: Duration, parallelism: Int = 1) {
        self.delay = delay
        self.parallelism = parallelism
    }

    var maxInFlight: Int { state.withLock { $0.maxInFlight } }
    var hurryCalls: [Bool] { state.withLock { $0.hurry } }
    var relaxed: Int { state.withLock { $0.relaxed } }

    func extract(_ url: URL, kind: FindKind, options: FindExtractOptions) async -> FindExtraction {
        state.withLock {
            $0.inFlight += 1
            $0.maxInFlight = max($0.maxInFlight, $0.inFlight)
        }
        try? await Task.sleep(for: delay)
        state.withLock { $0.inFlight -= 1 }
        return await FindMediaFixtures.extract(url, kind: kind, options: options)
    }

    func prepare(_ url: URL, kind: FindKind, pieces: [FindMediaPiece]) async -> [URL?] {
        await FindMediaFixtures.prepare(url, kind: kind, pieces: pieces, into: FindMediaFolder.url)
    }

    func shutdown() async {}
    func setHurry(_ hurry: Bool) async { state.withLock { $0.hurry.append(hurry) } }
    func relax() async { state.withLock { $0.relaxed += 1 } }
}

/// The hashing embedder that records the size of each media call.
final class CountingEmbedder: FindTextEmbedding {
    let base = HashingTextEmbedder(media: true)
    let mediaCalls = Mutex<[Int]>([])
    var fingerprint: String { base.fingerprint }
    var supportsMedia: Bool { true }
    var mediaFingerprint: String { base.mediaFingerprint }
    func embed(documents: [FindEmbeddingInput]) async throws -> [[Float]] { try await base.embed(documents: documents) }
    func embed(query: String) async throws -> [Float] { try await base.embed(query: query) }
    func unload() async {}
    func embed(media: [FindMediaData]) async throws -> [[Float]?] {
        mediaCalls.withLock { $0.append(media.count) }
        return try await base.embed(media: media)
    }
}

@Suite struct FindSpeedTests {
    @Test func thePoolReadsSeveralFilesAtOnceButEachReaderOneAtATime() async throws {
        let folder = try FindTempFolder()
        let readers = (0..<3).map { _ in SlowReader(delay: .milliseconds(30)) }
        let pool = FindReaderPool(readers)
        #expect(pool.parallelism == 3)
        let files = try (0..<9).map { try folder.write("n\($0).txt", "note \($0)") }
        await withTaskGroup(of: Void.self) { group in
            for file in files {
                group.addTask { _ = await pool.extract(file, kind: .text, options: FindExtractOptions()) }
            }
        }
        #expect(readers.allSatisfy { $0.maxInFlight == 1 })
        #expect(readers.allSatisfy { $0.maxInFlight >= 1 }, "every reader took work")
        await pool.setHurry(true)
        #expect(readers.allSatisfy { $0.hurryCalls == [true] })
        #expect(FindReaderPool.size(cores: 4) == 1 && FindReaderPool.size(cores: 10) == 2
            && FindReaderPool.size(cores: 24) == 3)
    }

    @Test func theIndexerKeepsTheReadersBusy() async throws {
        let folder = try FindTempFolder()
        for index in 0..<12 { try folder.write("n\(index).txt", "Note \(index) about the garden and tomatoes.") }
        let reader = SlowReader(delay: .milliseconds(20), parallelism: 3)
        let indexer = FindIndexer(index: try FindIndex(directory: nil), extractor: reader, embedder: HashingTextEmbedder(),
                                  configuration: .init(roots: [folder.url], rules: FindCrawlRules()))
        try await indexer.syncAll()
        await indexer.waitUntilIdle()
        #expect(reader.maxInFlight == 3)
        #expect(await indexer.publish().learnedFiles == 12)
        #expect(reader.relaxed >= 1, "readers are told when the work is done")
    }

    @Test func learningRunsWhileReadingGoesOn() async throws {
        let folder = try FindTempFolder()
        for index in 0..<40 { try folder.write("n\(index).txt", "Note \(index): a plan for the garden this spring.") }
        let learnedWhileReading = Mutex(false)
        let indexer = FindIndexer(
            index: try FindIndex(directory: nil), extractor: SlowReader(delay: .milliseconds(15)),
            embedder: HashingTextEmbedder(), configuration: .init(roots: [folder.url], rules: FindCrawlRules()),
            onStatus: { status in
                if status.learnedFiles > 0, status.readFiles + status.skippedFiles < status.totalFiles {
                    learnedWhileReading.withLock { $0 = true }
                }
            })
        await indexer.setReadBatchForTests(4)
        try await indexer.syncAll()
        await indexer.waitUntilIdle()
        #expect(learnedWhileReading.withLock { $0 }, "nothing was learned until every file was read")
        #expect(await indexer.publish().learnedFiles == 40)
    }

    @Test func aBigBacklogHurriesAndASmallOneDoesNot() async throws {
        let folder = try FindTempFolder()
        for index in 0...FindIndexer.hurryBacklog { try folder.write("n\(index).txt", "Note \(index).") }
        let reader = SlowReader(delay: .zero)
        let indexer = FindIndexer(index: try FindIndex(directory: nil), extractor: reader, embedder: HashingTextEmbedder(),
                                  configuration: .init(roots: [folder.url], rules: FindCrawlRules()))
        try await indexer.syncAll()
        await indexer.waitUntilIdle()
        #expect(reader.hurryCalls.first == true)
        #expect(reader.hurryCalls.last == false, "back to background once caught up")
    }

    @Test func photosWithoutTextSkipTheFullRead() async throws {
        let folder = try FindTempFolder()
        let plain = try FindMediaFixtures.png(folder.url.appending(path: "IMG_1.png"), red: 0.2, green: 0.5, blue: 0.3)
        #expect(await FindMediaFixtures.extract(plain, kind: .image, options: FindExtractOptions(ocr: true)).chunks.isEmpty)
        let sign = try Self.textImage(folder.url.appending(path: "IMG_2.png"), lines: ["PARKING PERMIT", "ZONE 4 ONLY"])
        let read = await FindMediaFixtures.extract(sign, kind: .image, options: FindExtractOptions(ocr: true)).chunks
        #expect(read.first?.fromOCR == true)
        #expect(read.first?.text.uppercased().contains("PARKING") == true)
    }

    @Test func picturesPreparedWhileReadingAreLearnedWithoutDecodingAgain() async throws {
        let folder = try FindTempFolder()
        for index in 0..<5 {
            try FindMediaFixtures.png(folder.url.appending(path: "IMG_\(index).png"), red: 0.9, green: 0.1, blue: 0.1)
        }
        let embedder = CountingEmbedder()
        let indexer = FindIndexer(index: try FindIndex(directory: nil), extractor: InProcessFindExtractor(),
                                  embedder: embedder,
                                  configuration: .init(roots: [folder.url], rules: FindCrawlRules(readImages: false,
                                                                                                  readMedia: true)))
        try await indexer.syncAll()
        // Read only: each photo is prepared on the way.
        #expect(try await indexer.readStep())
        let index = await indexer.index
        #expect(try await index.preparedBacklog() == 5)
        let pending = try await index.pendingChunks(limit: 10, includeMedia: true)
        let prepared = pending.compactMap { $0.media?.prepared }
        #expect(prepared.count == 5 && prepared.allSatisfy { FileManager.default.fileExists(atPath: $0) })
        // Learn: one batched call, the prepared files used and removed.
        await indexer.waitUntilIdle()
        #expect(embedder.mediaCalls.withLock { $0 } == [5])
        #expect(try await index.preparedBacklog() == 0)
        #expect(prepared.allSatisfy { !FileManager.default.fileExists(atPath: $0) })
        #expect(await indexer.publish().learnedFiles == 5)
    }

    @Test func picturesAreNotPreparedAheadWhileLearningWaits() async throws {
        let folder = try FindTempFolder()
        try FindMediaFixtures.png(folder.url.appending(path: "IMG_1.png"), red: 0.9, green: 0.1, blue: 0.1)
        let indexer = FindIndexer(index: try FindIndex(directory: nil), extractor: InProcessFindExtractor(),
                                  embedder: CountingEmbedder(),
                                  configuration: .init(roots: [folder.url], rules: FindCrawlRules(readImages: false,
                                                                                                  readMedia: true)),
                                  gate: FindIndexer.Gate(learning: { "Waiting for the power adapter" }))
        try await indexer.syncAll()
        #expect(try await indexer.readStep())
        #expect(try await indexer.index.preparedBacklog() == 0, "nothing piles up on disk while learning waits")
    }

    @Test func aReReadFileDropsItsPreparedPicture() async throws {
        let folder = try FindTempFolder()
        let photo = try FindMediaFixtures.png(folder.url.appending(path: "IMG_1.png"), red: 0.9, green: 0.1, blue: 0.1)
        let indexer = FindIndexer(index: try FindIndex(directory: nil), extractor: InProcessFindExtractor(),
                                  embedder: CountingEmbedder(),
                                  configuration: .init(roots: [folder.url], rules: FindCrawlRules(readImages: false,
                                                                                                  readMedia: true)))
        try await indexer.syncAll()
        #expect(try await indexer.readStep())
        let index = await indexer.index
        let prepared = try #require(try await index.pendingChunks(limit: 5, includeMedia: true).first?.media?.prepared)
        let id = try #require(try await index.file(path: photo.standardizedFileURL.path)?.id)
        try await index.storeText(fileID: id, name: "IMG_1.png", chunks: [], contentHash: nil)
        #expect(!FileManager.default.fileExists(atPath: prepared))
    }

    /// Black text on white, the size of a sign in a photo.
    static func textImage(_ url: URL, lines: [String]) throws -> URL {
        let width = 1_200, height = 800
        let context = try #require(CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                                             space: CGColorSpaceCreateDeviceRGB(),
                                             bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        context.setFillColor(CGColor(gray: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        for (index, line) in lines.enumerated() {
            let text = CTLineCreateWithAttributedString(NSAttributedString(string: line, attributes: [
                .font: NSFont.boldSystemFont(ofSize: 80), .foregroundColor: NSColor.black]))
            context.textPosition = CGPoint(x: 80, y: CGFloat(height - 220 - index * 160))
            CTLineDraw(text, context)
        }
        let image = try #require(context.makeImage())
        let destination = try #require(CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString,
                                                                        1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        #expect(CGImageDestinationFinalize(destination))
        return url
    }
}

extension FindIndexer {
    func setReadBatchForTests(_ count: Int) {
        readBatch = count
    }
}

@Suite struct FindPreparedOrderTests {
    @Test func preparedPicturesAreLearnedFirst() async throws {
        let index = try FindIndex(directory: nil)
        let older = try await index.upsertFile(path: "/a/new.png", root: "/a", kind: .image, size: 1,
                                               modified: .now, inode: 1, device: 1, lastUsed: .now)
        let newer = try await index.upsertFile(path: "/a/old.png", root: "/a", kind: .image, size: 1,
                                               modified: .distantPast, inode: 2, device: 1, lastUsed: .distantPast)
        try await index.storeText(fileID: older, name: "new.png",
                                  chunks: [.init(location: nil, text: "", media: FindMediaPiece(modality: .image, tokens: 70))],
                                  contentHash: nil)
        try await index.storeText(fileID: newer, name: "old.png", chunks: [.init(
            location: nil, text: "", media: FindMediaPiece(modality: .image, tokens: 70, prepared: "/tmp/x.fmd"))],
                                  contentHash: nil)
        // The recently used photo would come first, but the prepared one waits on disk.
        let first = try await index.pendingChunks(limit: 1, includeMedia: true)
        #expect(first.first?.fileID == newer)
        #expect(first.first?.media?.prepared == "/tmp/x.fmd")
    }
}

@Suite struct FindPreparedCleanupTests {
    /// A picture this process prepared before its first Find start is this
    /// run's, so Find's start-up cleanup keeps it. (Tests run side by side:
    /// one test's Find start must not delete what another just prepared.)
    @Test @MainActor func cleanupKeepsWhatThisProcessPrepared() throws {
        let manager = FileManager.default
        try manager.createDirectory(at: FindMediaFolder.ready, withIntermediateDirectories: true)
        let file = FindMediaFolder.ready.appending(path: "cleanup-\(UUID().uuidString).fmd")
        try Data([1]).write(to: file)
        defer { try? manager.removeItem(at: file) }
        FindService.removeEarlierPrepared()
        #expect(manager.fileExists(atPath: file.path), "a file prepared by this process was removed")
    }

    @Test @MainActor func theProcessStartTimeIsRead() throws {
        let start = try #require(FindService.processStart())
        #expect(start <= Date())
        #expect(Date().timeIntervalSince(start) < 24 * 3_600, "a test process is not a day old")
    }
}
