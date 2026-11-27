import CoreGraphics
import Foundation
import ImageIO
@testable import MferenceAppCore
import Testing
import UniformTypeIdentifiers

// Find's photos, video and audio, model-free: the hashing embedder "sees"
// pictures by colour and "hears" sound as loud or quiet.

enum FindMediaFixtures {
    /// A solid-colour PNG.
    @discardableResult
    static func png(_ url: URL, red: CGFloat, green: CGFloat, blue: CGFloat, width: Int = 320,
                    height: Int = 240) throws -> URL {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let context = try #require(CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                                             bytesPerRow: 0, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                             bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        context.setFillColor(CGColor(srgbRed: red, green: green, blue: blue, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        let image = try #require(context.makeImage())
        let destination = try #require(CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString,
                                                                        1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        #expect(CGImageDestinationFinalize(destination))
        return url
    }

    /// A 16 kHz mono 16-bit WAV: a 440 Hz tone of `amplitude` for `loud`
    /// seconds, then silence up to `seconds`.
    @discardableResult
    static func wav(_ url: URL, seconds: Double, loud: Double, amplitude: Double) throws -> URL {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let rate = 16_000
        let count = Int(seconds * Double(rate))
        var samples = [Int16](repeating: 0, count: count)
        for index in 0..<min(count, Int(loud * Double(rate))) {
            samples[index] = Int16(sin(Double(index) * 2 * .pi * 440 / Double(rate)) * amplitude * 32_767)
        }
        var data = Data()
        func text(_ string: String) { data.append(contentsOf: Array(string.utf8)) }
        func u32(_ value: UInt32) { withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) } }
        func u16(_ value: UInt16) { withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) } }
        text("RIFF"); u32(UInt32(36 + count * 2)); text("WAVE")
        text("fmt "); u32(16); u16(1); u16(1); u32(UInt32(rate)); u32(UInt32(rate * 2)); u16(2); u16(16)
        text("data"); u32(UInt32(count * 2))
        samples.withUnsafeBytes { data.append(contentsOf: $0) }
        try data.write(to: url)
        return url
    }

    /// Everyday notes, so meaning scores have a realistic spread.
    static func fillers(in folder: FindTempFolder, count: Int = 30) throws {
        let topics = ["oats milk apples", "legs day gym", "call the bank", "flight to lisbon", "car oil change",
                      "garden tomatoes basil", "book club novel", "running log knee", "roman aqueducts lecture",
                      "wedding gift pasta maker"]
        for index in 0..<count {
            try folder.write("Notes/note \(index).txt", "Note \(index): \(topics[index % topics.count]) and more plans.")
        }
    }

    /// `FindExtractor.extract` off the cooperative pool. Vision and the
    /// AVFoundation bridge need a free pool thread to finish, so a test that
    /// blocks one per read hangs the whole run once the pool is full.
    static func extract(_ url: URL, kind: FindKind, options: FindExtractOptions) async -> FindExtraction {
        await BlockingWork.run { FindExtractor.extract(url, kind: kind, options: options) }
    }

    /// `FindMedia.prepare` off the cooperative pool, for the same reason.
    static func prepare(_ url: URL, kind: FindKind, pieces: [FindMediaPiece], into folder: URL) async -> [URL?] {
        await BlockingWork.run { FindMedia.prepare(url, kind: kind, pieces: pieces, into: folder) }
    }
}

@Suite struct FindMediaPlanTests {
    @Test func videoAndAudioAreKindsOfTheirOwn() {
        #expect(FindKind.of(URL(filePath: "/a/Zoo day.MOV")) == .video)
        #expect(FindKind.of(URL(filePath: "/a/memo.m4a")) == .audio)
        #expect(FindKind.of(URL(filePath: "/a/song.mp3"))?.filter == .media)
        #expect(FindKind.video.isMedia && !FindKind.image.isMedia)
        #expect(FindKindFilter.media.includes(.audio) && !FindKindFilter.media.includes(.image))
    }

    @Test func piecesSayWhereTheyAre() {
        #expect(FindMedia.clock(42.9) == "0:42")
        #expect(FindMedia.clock(3_723) == "1:02:03")
        #expect(FindMediaPiece(modality: .image, start: 75).location == "1:15")
        #expect(FindMediaPiece(modality: .audio, start: 75, end: 105).location == "1:15–1:45")
        #expect(FindMediaPiece(modality: .image, tokens: 140).location == nil)
    }

    @Test func audioWindowsOverlapAndCoverTheEnd() {
        let ranges = FindMedia.windowRanges(duration: 70)
        #expect(ranges.map(\.start) == [0, 25, 50])
        #expect(ranges.map(\.end) == [30, 55, 70])
        #expect(FindMedia.windowRanges(duration: 1).count == 1)
        #expect(FindMedia.windowRanges(duration: 200_000).count <= FindMedia.maxWindows)
    }

    @Test func framesAreEveryFewSecondsAndCapped() {
        #expect(FindMedia.frameTimes(duration: 10) == [0, 3, 6, 9])
        #expect(FindMedia.frameTimes(duration: 36_000).count <= FindMedia.maxFrames)
    }

    @Test func screenshotsGetMoreDetailThanPhotos() {
        #expect(FindMedia.imageTokens(for: URL(filePath: "/a/Screenshot 2026-08-19 at 21.03.png")) == FindMedia.screenshotTokens)
        #expect(FindMedia.imageTokens(for: URL(filePath: "/a/IMG_2041.heic")) == FindMedia.photoTokens)
    }

    @Test func encoderSizeFillsTheBudgetInWholeTiles() {
        let size = FindMedia.encoderSize(width: 640, height: 488, tokens: 140)
        #expect(size?.width == 624 && size?.height == 480)
        #expect(FindMedia.encoderSize(width: 0, height: 10, tokens: 140) == nil)
    }

    @Test func mediaDataRoundTrips() throws {
        let folder = try FindTempFolder()
        let image = FindMediaData.image(width: 2, height: 1, rgb: [1, 2, 3, 4, 5, 6])
        try image.write(to: folder.url.appending(path: "i.fmd"))
        #expect(try FindMediaData(contentsOf: folder.url.appending(path: "i.fmd")) == image)
        let audio = FindMediaData.audio(samples: [0.5, -0.25, 1])
        try audio.write(to: folder.url.appending(path: "a.fmd"))
        #expect(try FindMediaData(contentsOf: folder.url.appending(path: "a.fmd")) == audio)
        try Data("nope".utf8).write(to: folder.url.appending(path: "x.fmd"))
        #expect(throws: (any Error).self) { try FindMediaData(contentsOf: folder.url.appending(path: "x.fmd")) }
    }

    @Test func averageHashTellsShotsApart() throws {
        func image(_ draw: (CGContext) -> Void) throws -> CGImage {
            let context = try #require(CGContext(data: nil, width: 64, height: 64, bitsPerComponent: 8, bytesPerRow: 0,
                                                 space: CGColorSpaceCreateDeviceRGB(),
                                                 bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
            draw(context)
            return try #require(context.makeImage())
        }
        let left = try image { context in
            context.setFillColor(CGColor(gray: 1, alpha: 1))
            context.fill(CGRect(x: 0, y: 0, width: 32, height: 64))
        }
        let top = try image { context in
            context.setFillColor(CGColor(gray: 1, alpha: 1))
            context.fill(CGRect(x: 0, y: 32, width: 64, height: 32))
        }
        let a = try #require(FindMedia.averageHash(left)), b = try #require(FindMedia.averageHash(top))
        #expect(a == FindMedia.averageHash(left))
        #expect((a ^ b).nonzeroBitCount > FindMedia.sameShotBits)
    }

    @Test func photoIsPlannedThenPreparedAtTheEncoderSize() async throws {
        let folder = try FindTempFolder()
        let photo = try FindMediaFixtures.png(folder.url.appending(path: "IMG_1.png"), red: 1, green: 0, blue: 0)
        let extraction = await FindMediaFixtures.extract(photo, kind: .image, options: FindExtractOptions(ocr: false, media: true))
        let chunks = extraction.chunks
        #expect(chunks.count == 1)
        let piece = try #require(chunks.first?.media)
        #expect(piece.modality == .image && piece.tokens == FindMedia.photoTokens)
        // Without media on, a photo has no pieces.
        #expect(await FindMediaFixtures.extract(photo, kind: .image, options: FindExtractOptions(ocr: false)).chunks.isEmpty)

        let files = await FindMediaFixtures.prepare(photo, kind: .image, pieces: [piece],
                                                   into: folder.url.appending(path: "prep"))
        let file = try #require(files.first ?? nil)
        guard case .image(let width, let height, let rgb) = try FindMediaData(contentsOf: file) else {
            Issue.record("not an image")
            return
        }
        #expect(width % 48 == 0 && height % 48 == 0)
        #expect((width / 48) * (height / 48) <= FindMedia.photoTokens)
        #expect(rgb.count == width * height * 3)
        #expect(rgb[0] > 200 && rgb[1] < 40 && rgb[2] < 40)
    }

    @Test func audioIsPlannedInWindowsWithSilenceSkipped() async throws {
        let folder = try FindTempFolder()
        // Sound for the first 20 s of 70: the window from 50 s is silent.
        let wav = try FindMediaFixtures.wav(folder.url.appending(path: "memo.wav"), seconds: 70, loud: 26,
                                             amplitude: 0.5)
        let chunks = await FindMediaFixtures.extract(wav, kind: .audio, options: FindExtractOptions(media: true)).chunks
        #expect(chunks.compactMap(\.media?.start) == [0, 25])
        #expect(chunks.first?.location == "0:00–0:30")
        let files = await FindMediaFixtures.prepare(wav, kind: .audio, pieces: chunks.compactMap(\.media),
                                                   into: folder.url.appending(path: "prep"))
        let file = try #require(files.first ?? nil)
        guard case .audio(let samples) = try FindMediaData(contentsOf: file) else {
            Issue.record("not audio")
            return
        }
        #expect(abs(samples.count - 30 * 16_000) < 1_600)
        // Without media on, audio is name-only.
        #expect(await FindMediaFixtures.extract(wav, kind: .audio, options: FindExtractOptions()).chunks.isEmpty)
    }

    @Test func aDamagedVideoIsSkipped() async throws {
        let folder = try FindTempFolder()
        let file = try folder.write("broken.mp4", "not a video at all")
        if case .skipped = await FindMediaFixtures.extract(file, kind: .video, options: FindExtractOptions(media: true)) {
        } else {
            Issue.record("a damaged video was not skipped")
        }
    }
}

@Suite struct FindMediaSearchTests {
    func indexer(_ folder: FindTempFolder, embedder: any FindTextEmbedding) throws -> FindIndexer {
        FindIndexer(index: try FindIndex(directory: nil), extractor: InProcessFindExtractor(), embedder: embedder,
                    configuration: .init(roots: [folder.url], rules: FindCrawlRules(readImages: false, readMedia: true)))
    }

    @Test func picturesAndSoundAreFoundByWhatTheyAre() async throws {
        let folder = try FindTempFolder()
        try FindMediaFixtures.png(folder.url.appending(path: "Camera/IMG_1.png"), red: 0.9, green: 0.1, blue: 0.1)
        try FindMediaFixtures.png(folder.url.appending(path: "Camera/IMG_2.png"), red: 0.1, green: 0.2, blue: 0.9)
        try FindMediaFixtures.wav(folder.url.appending(path: "Memos/memo 1.wav"), seconds: 8, loud: 8, amplitude: 0.6)
        try FindMediaFixtures.wav(folder.url.appending(path: "Memos/memo 2.wav"), seconds: 8, loud: 8, amplitude: 0.02)
        try FindMediaFixtures.fillers(in: folder)
        let embedder = HashingTextEmbedder(media: true)
        let indexer = try indexer(folder, embedder: embedder)
        try await indexer.syncAll()
        await indexer.waitUntilIdle()
        let status = await indexer.publish()
        #expect(status.learnedFiles == status.totalFiles)

        let searcher = FindSearcher(index: await indexer.index, embedder: embedder)
        let red = try await searcher.search(FindQuery("red"))
        #expect(red.strong.first?.name == "IMG_1.png")
        #expect(red.strong.first?.reasons.contains(.picture) == true)
        let blue = try await searcher.search(FindQuery("blue", kind: .images))
        #expect(blue.strong.map(\.name) == ["IMG_2.png"])
        let loud = try await searcher.search(FindQuery("loud"))
        let hit = try #require(loud.strong.first)
        #expect(hit.name == "memo 1.wav")
        #expect(hit.reasons.contains(.sound))
        #expect(hit.time == 0 && hit.location == "0:00–0:08")
        #expect(hit.passage.contains("sounds like"))
        // The media filter keeps only video and audio.
        let media = try await searcher.search(FindQuery("loud", kind: .media))
        #expect(media.results.allSatisfy { $0.kind.isMedia })
    }

    @Test func mediaWaitsForAnEmbedderThatHasTheEncoders() async throws {
        let folder = try FindTempFolder()
        try FindMediaFixtures.png(folder.url.appending(path: "IMG_1.png"), red: 0.9, green: 0.1, blue: 0.1)
        try folder.write("note.txt", "Tenant may keep one cat with a refundable deposit and written notice.")
        let indexer = try indexer(folder, embedder: HashingTextEmbedder())
        try await indexer.syncAll()
        await indexer.waitUntilIdle()
        var status = await indexer.publish()
        #expect(status.readFiles == 2)
        #expect(status.learnedFiles == 1, "the photo waits for the media encoders")
        await indexer.setEmbedder(HashingTextEmbedder(media: true))
        await indexer.waitUntilIdle()
        status = await indexer.publish()
        #expect(status.learnedFiles == 2)
    }

    @Test func undecodablePiecesStopWaiting() async throws {
        let folder = try FindTempFolder()
        try folder.write("IMG_9.png", "not really a picture")
        let indexer = try indexer(folder, embedder: HashingTextEmbedder(media: true))
        try await indexer.syncAll()
        await indexer.waitUntilIdle()
        let status = await indexer.publish()
        #expect(status.learnedFiles == 1, "a photo that cannot be decoded is still done (findable by name)")
        #expect(status.phase == .upToDate)
    }

    @Test func eachModalityIsScoredOnItsOwnLevel() {
        // A text query sits nearer every text passage than any picture; the
        // one picture that matches must still stand out.
        func vector(_ values: [Int: Float]) -> [Float] {
            var vector = [Float](repeating: 0, count: FindVectors.dimensions)
            for (index, value) in values { vector[index] = value }
            return FindVectors.prepare(vector)
        }
        var cache = VectorCache()
        for index in 0..<40 {
            let record = FindVectors.encode(vector([0: 0.5 + Float(index % 5) * 0.01, 10 + index: 0.85]))
            record.withUnsafeBytes { cache.append(chunkID: Int64(index), fileID: Int64(index), modality: 0, record: $0) }
        }
        for index in 0..<40 {
            let level: Float = index == 7 ? 0.35 : 0.05 + Float(index % 4) * 0.01
            let record = FindVectors.encode(vector([0: level, 100 + index: 1]))
            record.withUnsafeBytes {
                cache.append(chunkID: Int64(100 + index), fileID: Int64(100 + index), modality: 1, record: $0)
            }
        }
        let hits = cache.search(FindVectors.Query(vector([0: 1])), candidates: 600)
        let picture = hits.first { $0.chunkID == 107 }
        #expect(picture.map { $0.z >= FindSearcher.meaningGate } == true, "z \(picture?.z ?? 0)")
        #expect(hits.first?.chunkID == 107)
    }
}

@MainActor
@Suite struct FindMediaServiceTests {
    @Test func turningMediaOnReadsPhotosAgain() async throws {
        let folder = try FindTempFolder()
        try FindMediaFixtures.png(folder.url.appending(path: "IMG_1.png"), red: 0.9, green: 0.1, blue: 0.1)
        try FindMediaFixtures.fillers(in: folder)
        let settings = FindSettings(store: InMemorySettingsStore())
        settings.readImages = false
        settings.learnOnlyOnPower = false
        let find = FindService.preview(settings: settings)
        await find.turnOn(roots: [folder.url])
        func settle() async throws {
            for _ in 0..<300 {
                let status = find.status
                if status.totalFiles > 0, status.learnedFiles + status.skippedFiles == status.totalFiles,
                   status.phase == .upToDate { return }
                try await Task.sleep(for: .milliseconds(20))
            }
        }
        try await settle()
        #expect(await find.search(FindQuery("red")).strong.first?.name != "IMG_1.png")

        await find.setReadMedia(true)
        #expect(settings.readMedia)
        // The media model "downloads", the embedder switches, the photo is read again.
        for _ in 0..<300 where await find.search(FindQuery("red")).strong.first?.name != "IMG_1.png" {
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(find.mediaModelState.isReady)
        let red = await find.search(FindQuery("red"))
        #expect(red.strong.first?.name == "IMG_1.png")
        #expect(red.strong.first?.reasons.contains(.picture) == true)

        await find.setReadMedia(false)
        try await settle()
        #expect(await find.search(FindQuery("red")).strong.first?.name != "IMG_1.png")
        await find.turnOff()
    }

    @Test func readMediaIsOffByDefaultAndSaved() {
        let store = InMemorySettingsStore()
        let settings = FindSettings(store: store)
        #expect(!settings.readMedia)
        settings.readMedia = true
        #expect(FindSettings(store: store).readMedia)
    }
}
