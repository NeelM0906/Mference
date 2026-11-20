import AppKit
import CoreText
import Foundation
@testable import MferenceAppCore
import Testing

// Find, model-free: the hashing embedder stands in for EmbeddingGemma 2.

/// A temporary folder removed at the end of a test.
final class FindTempFolder {
    let url: URL

    init() throws {
        url = FileManager.default.temporaryDirectory.appending(path: "find-tests-\(UUID().uuidString)",
                                                                directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    }

    deinit { try? FileManager.default.removeItem(at: url) }

    @discardableResult
    func write(_ relative: String, _ text: String) throws -> URL {
        let file = url.appending(path: relative)
        try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(text.utf8).write(to: file)
        return file
    }
}

@Suite struct FindTextTests {
    @Test func shortTextIsOneChunk() {
        #expect(FindText.chunks("One short paragraph.") == ["One short paragraph."])
    }

    @Test func longTextSplitsWithOverlapAndLimit() {
        let sentence = "The quick brown fox jumps over the lazy dog. "
        let text = String(repeating: sentence, count: 200)
        let chunks = FindText.chunks(text)
        #expect(chunks.count > 4)
        #expect(chunks.allSatisfy { $0.count <= FindText.chunkCharacters })
        // Neighbours overlap: the end of one appears at the start of the next.
        let tail = String(chunks[0].suffix(40))
        #expect(chunks[1].contains(tail.split(separator: " ").dropFirst().prefix(3).joined(separator: " ")))
        #expect(FindText.chunks(text, limit: 2).count == 2)
    }

    @Test func proseGateKeepsNumberDumpsOut() {
        #expect(FindText.isProse("Tenant may keep one cat with a deposit."))
        #expect(!FindText.isProse("12 34 56 78 90 11 22 33 44 55 66"))
        #expect(!FindText.isProse("Hi there"))
    }

    @Test func queryWordsDropStopWordsAndRepeats() {
        #expect(FindText.queryWords("the lease clause about the pets, Lease") == ["lease", "clause", "pets"])
    }

    @Test func snippetPicksTheSentenceWithTheQueryWords() {
        let chunk = "Section 13 covers paint. Tenant may keep one cat with a deposit. Section 15 covers noise."
        let snippet = FindText.snippet(of: chunk, queryWords: ["cat", "deposit"])
        #expect(snippet.hit.contains("one cat"))
        #expect(snippet.before.contains("paint"))
        #expect(snippet.after.contains("noise"))
    }
}

@Suite struct FindVectorTests {
    @Test func int8RecordsKeepTheCosine() {
        var generator = SystemRandomNumberGenerator()
        let a = FindVectors.prepare((0..<768).map { _ in Float.random(in: -1...1, using: &generator) })
        let b = FindVectors.prepare(zip(a, (0..<768).map { _ in Float.random(in: -0.3...0.3, using: &generator) })
            .map { $0 + $1 })
        let exact = zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
        let record = FindVectors.encode(b)
        #expect(record.count == FindVectors.recordBytes)
        let approximate = record.withUnsafeBytes { FindVectors.cosine(FindVectors.Query(a), record: $0) }
        #expect(abs(approximate - exact) < 0.02)
        let same = FindVectors.encode(a).withUnsafeBytes { FindVectors.hamming(FindVectors.Query(a), record: $0) }
        #expect(same == 0)
    }

    @Test func prepareTruncatesAndNormalises() {
        let vector = FindVectors.prepare([Float](repeating: 2, count: 768))
        #expect(vector.count == FindVectors.dimensions)
        #expect(abs(vector.reduce(0) { $0 + $1 * $1 } - 1) < 1e-4)
    }
}

@Suite struct FindIndexTests {
    func index() throws -> FindIndex { try FindIndex(directory: nil) }

    @discardableResult
    func add(_ index: FindIndex, _ path: String, _ text: String, kind: FindKind = .text,
             inode: Int64 = 0) async throws -> Int64 {
        let id = try await index.upsertFile(path: path, root: "/r", kind: kind, size: Int64(text.count),
                                            modified: Date(timeIntervalSince1970: 1_000), inode: inode, device: 1,
                                            lastUsed: nil)
        try await index.storeText(fileID: id, name: URL(filePath: path).lastPathComponent,
                                  chunks: [FindIndex.ChunkInput(location: nil, text: text)], contentHash: nil)
        return id
    }

    @Test func wordsAndNamesAreSearchable() async throws {
        let index = try index()
        let lease = try await add(index, "/r/Lease agreement.pdf", "Tenant may keep one cat with a refundable deposit.")
        _ = try await add(index, "/r/Recipe.txt", "Whisk the eggs with sugar until pale.")
        #expect(try await index.filesContaining("deposit", isPrefix: false) == [lease])
        #expect(try await index.filesContaining("depo", isPrefix: true) == [lease])
        #expect(try await index.filesContaining("lease", isPrefix: false, column: "name") == [lease])
        let hits = try await index.wordHits(["eggs"], lastIsPrefix: false)
        #expect(hits.count == 1)
    }

    @Test func changedFilesGoBackToPendingAndMovesKeepText() async throws {
        let index = try index()
        let id = try await add(index, "/r/a.txt", "Tenant may keep one cat with a refundable deposit.", inode: 42)
        #expect(try await index.file(path: "/r/a.txt")?.state == .read)
        try await index.move(fileID: id, to: "/r/moved.txt", root: "/r")
        #expect(try await index.file(path: "/r/moved.txt")?.id == id)
        #expect(try await index.filesContaining("moved", isPrefix: false, column: "name") == [id])
        try await index.upsertFile(path: "/r/moved.txt", root: "/r", kind: .text, size: 999, modified: .now,
                                   inode: 42, device: 1, lastUsed: nil)
        #expect(try await index.file(path: "/r/moved.txt")?.state == .pending)
    }

    @Test func skippedFilesAreFoundByName() async throws {
        let index = try index()
        let id = try await index.upsertFile(path: "/r/2025 Tax Return.pdf", root: "/r", kind: .pdf, size: 1,
                                            modified: .now, inode: 1, device: 1, lastUsed: nil)
        try await index.markSkipped(fileID: id, reason: "Password-protected: searched by name only.")
        #expect(try await index.filesContaining("tax", isPrefix: false) == [id])
        #expect(try await index.skippedFiles().first?.reason.contains("Password") == true)
    }

    @Test func vectorsMarkFilesLearnedAndRankByMeaning() async throws {
        let index = try index()
        let embedder = HashingTextEmbedder()
        let lease = try await add(index, "/r/lease.txt", "Tenant may keep one cat with a refundable deposit.")
        _ = try await add(index, "/r/recipe.txt", "Whisk the eggs with sugar until pale and fluffy.")
        _ = try await add(index, "/r/router.txt", "Changed the admin password and isolated the guest network.")
        let pending = try await index.pendingChunks(limit: 10)
        #expect(pending.count == 3)
        let vectors = try await embedder.embed(documents: pending.map { FindEmbeddingInput(title: $0.title, text: $0.text) })
        try await index.storeVectors(zip(pending, vectors).map {
            (chunkID: $0.chunkID, fileID: $0.fileID, vector: FindVectors.prepare($1))
        })
        #expect(try await index.pendingChunks(limit: 10).isEmpty)
        #expect(try await index.counts().learned == 3)
        let query = try await embedder.embed(query: "pets allowed")
        let hits = try await index.meaningHits(query)
        #expect(hits.first?.fileID == lease)
    }

    @Test func numberDumpsGetNoVector() async throws {
        let index = try index()
        _ = try await add(index, "/r/numbers.csv", "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16")
        #expect(try await index.pendingChunks(limit: 10).isEmpty)
        #expect(try await index.file(path: "/r/numbers.csv")?.state == .learned)
    }
}

@Suite struct FindSearcherTests {
    /// A small learned index: a lease, a recipe, a copy of the lease, a
    /// screenshot's OCR text, and a sheet.
    func corpus() async throws -> FindIndex {
        let index = try FindIndex(directory: nil)
        let embedder = HashingTextEmbedder()
        func add(_ path: String, _ kind: FindKind, _ text: String, ocr: Bool = false, hash: String? = nil) async throws {
            let id = try await index.upsertFile(path: path, root: "/r", kind: kind, size: 10, modified: .now,
                                                inode: Int64(path.hashValue), device: 1, lastUsed: nil)
            try await index.storeText(fileID: id, name: URL(filePath: path).lastPathComponent,
                                      chunks: [FindIndex.ChunkInput(location: "Page 4", text: text, fromOCR: ocr)],
                                      contentHash: hash)
        }
        try await add("/r/Lease.pdf", .pdf, "Tenant may keep one cat or one dog with a refundable deposit.", hash: "L")
        try await add("/r/Lease copy.pdf", .pdf, "Tenant may keep one cat or one dog with a refundable deposit.", hash: "L")
        try await add("/r/Recipe.docx", .document, "Whisk the eggs with sugar until pale and fluffy, then bake.")
        try await add("/r/Screenshot 1.png", .image, "Network Cedar-Guest password hunter2 welcome", ocr: true)
        try await add("/r/Q3 close.xlsx", .sheet, "Total revenue 4,218,400 for the third quarter of the year.")
        // A realistic spread: the meaning gate is a z-score against the
        // whole index, so the test index needs more than five passages.
        // (Words chosen so the hashing embedder's buckets do not collide
        // with the queries below.)
        for number in 0..<60 {
            try await add("/r/g\(number)/grocery.txt", .text, "Grocery list apples bread milk butter")
        }
        let pending = try await index.pendingChunks(limit: 500)
        #expect(try await index.pendingChunks(limit: 500).count == 65)
        let vectors = try await embedder.embed(documents: pending.map { FindEmbeddingInput(title: $0.title, text: $0.text) })
        try await index.storeVectors(zip(pending, vectors).map {
            (chunkID: $0.chunkID, fileID: $0.fileID, vector: FindVectors.prepare($1))
        })
        return index
    }

    @Test func meaningFindsTheLeaseWithoutItsWords() async throws {
        let searcher = FindSearcher(index: try await corpus(), embedder: HashingTextEmbedder())
        let results = try await searcher.search(FindQuery("pets allowed"))
        let first = try #require(results.strong.first)
        #expect(first.name.hasPrefix("Lease"))
        #expect(first.reasons.contains(.meaning))
        #expect(first.copies == 1, "the copy collapses into the first")
        #expect(first.location == "Page 4")
        #expect(!results.wordsOnly)
    }

    @Test func screenshotTextIsFoundByWords() async throws {
        let searcher = FindSearcher(index: try await corpus(), embedder: HashingTextEmbedder())
        let results = try await searcher.search(FindQuery("screenshot with the wifi password"))
        let first = try #require(results.strong.first)
        #expect(first.name == "Screenshot 1.png")
        #expect(first.reasons.contains(.textInImage))
    }

    @Test func wordsWhileTypingUseAPrefix() async throws {
        let searcher = FindSearcher(index: try await corpus(), embedder: nil)
        let results = try await searcher.searchWords(FindQuery("revenue quar"))
        #expect(results.wordsOnly)
        #expect(results.strong.first?.name == "Q3 close.xlsx")
    }

    @Test func noCloseMatchReturnsNothing() async throws {
        let searcher = FindSearcher(index: try await corpus(), embedder: HashingTextEmbedder())
        let results = try await searcher.search(FindQuery("orbital mechanics lecture"))
        #expect(results.strong.isEmpty)
    }

    @Test func kindFilterApplies() async throws {
        let searcher = FindSearcher(index: try await corpus(), embedder: HashingTextEmbedder())
        let results = try await searcher.search(FindQuery("pets allowed", kind: .images))
        #expect(results.results.allSatisfy { $0.kind == .image })
    }
}

@Suite struct FindExtractorTests {
    @Test func plainTextAndCodeLocations() throws {
        let folder = try FindTempFolder()
        let notes = try folder.write("notes.md", "# Trip\nPack the passport and the charger.")
        guard case .text(let passages) = FindExtractor.extract(notes, kind: .text, options: .init()) else {
            Issue.record("not read")
            return
        }
        #expect(passages.first?.text.contains("passport") == true)
        let code = try folder.write("a.swift", (1...300).map { "let value\($0) = \($0) // retry backoff" }.joined(separator: "\n"))
        guard case .text(let lines) = FindExtractor.extract(code, kind: .code, options: .init()) else {
            Issue.record("not read")
            return
        }
        #expect(lines.first?.location?.hasPrefix("Lines 1–") == true)
        #expect(lines.count > 1)
    }

    @Test func binaryFilesAreSkipped() throws {
        let folder = try FindTempFolder()
        let file = folder.url.appending(path: "blob.txt")
        try Data([0, 1, 2, 0, 0, 3]).write(to: file)
        guard case .skipped = FindExtractor.extract(file, kind: .text, options: .init()) else {
            Issue.record("binary file was read as text")
            return
        }
    }

    @Test func richDocumentsAndHTML() throws {
        let folder = try FindTempFolder()
        let text = NSAttributedString(string: "The renovation contract ends after ten weeks.")
        let docx = folder.url.appending(path: "contract.docx")
        try text.data(from: NSRange(location: 0, length: text.length),
                      documentAttributes: [.documentType: NSAttributedString.DocumentType.officeOpenXML]).write(to: docx)
        let rtf = folder.url.appending(path: "contract.rtf")
        try text.data(from: NSRange(location: 0, length: text.length),
                      documentAttributes: [.documentType: NSAttributedString.DocumentType.rtf]).write(to: rtf)
        let html = try folder.write("page.html", "<html><head><style>x{}</style></head><body><p>Hot tub hours &amp; rules</p><script>bad()</script></body></html>")
        for url in [docx, rtf] {
            guard case .text(let passages) = FindExtractor.extract(url, kind: .document, options: .init()) else {
                Issue.record("\(url.lastPathComponent) not read")
                continue
            }
            #expect(passages.map(\.text).joined().contains("ten weeks"))
        }
        guard case .text(let page) = FindExtractor.extract(html, kind: .document, options: .init()) else {
            Issue.record("html not read")
            return
        }
        let body = page.map(\.text).joined()
        #expect(body.contains("Hot tub hours & rules"))
        #expect(!body.contains("bad()"))
    }

    @Test func pdfPagesKeepTheirNumbers() throws {
        let folder = try FindTempFolder()
        let url = folder.url.appending(path: "lease.pdf")
        var box = CGRect(x: 0, y: 0, width: 612, height: 792)
        let context = try #require(CGContext(url as CFURL, mediaBox: &box, nil))
        for page in ["13. ALTERATIONS. Tenant shall not paint the walls without consent of the landlord at any time.",
                     "14. ANIMALS. Tenant may keep one cat with a refundable deposit and written notice to landlord."] {
            context.beginPDFPage(nil)
            let line = CTLineCreateWithAttributedString(NSAttributedString(string: page,
                attributes: [.font: NSFont.systemFont(ofSize: 9)]))
            context.textPosition = CGPoint(x: 36, y: 700)
            CTLineDraw(line, context)
            context.endPDFPage()
        }
        context.closePDF()
        guard case .text(let passages) = FindExtractor.extract(url, kind: .pdf, options: FindExtractOptions(ocr: false)) else {
            Issue.record("pdf not read")
            return
        }
        #expect(passages.count == 2)
        #expect(passages[1].location == "Page 2")
        #expect(passages[1].text.contains("cat"))
    }

    @Test func spreadsheetsReadSharedStringsAndSheetNames() throws {
        let folder = try FindTempFolder()
        try folder.write("x/[Content_Types].xml", "<Types/>")
        try folder.write("x/xl/workbook.xml", #"<workbook><sheets><sheet name="Summary" sheetId="1"/></sheets></workbook>"#)
        try folder.write("x/xl/sharedStrings.xml", "<sst><si><t>Total revenue</t></si></sst>")
        try folder.write("x/xl/worksheets/sheet1.xml",
                         #"<worksheet><sheetData><row><c t="s"><v>0</v></c><c><v>4218400</v></c></row></sheetData></worksheet>"#)
        let xlsx = folder.url.appending(path: "Q3.xlsx")
        let zip = Process()
        zip.executableURL = URL(filePath: "/usr/bin/zip")
        zip.currentDirectoryURL = folder.url.appending(path: "x")
        zip.arguments = ["-qr", xlsx.path, "."]
        try zip.run()
        zip.waitUntilExit()
        guard case .text(let passages) = FindExtractor.extract(xlsx, kind: .sheet, options: .init()) else {
            Issue.record("xlsx not read")
            return
        }
        #expect(passages.first?.location == "Sheet “Summary”")
        #expect(passages.first?.text == "Total revenue\t4218400")
    }
}

@Suite struct FindCrawlerTests {
    @Test func rulesSkipNoiseAndCodeProjects() throws {
        let folder = try FindTempFolder()
        try folder.write("Lease.pdf", "x")
        try folder.write("Deck.key", "x")
        try folder.write("id.pem", "x")
        try folder.write("node_modules/pkg/readme.md", "x")
        try folder.write(".hidden/notes.txt", "x")
        try folder.write("Project/.git/HEAD", "x")
        try folder.write("Project/README.md", "x")
        try folder.write("Docs/plan.md", "x")
        try folder.write("Docs/script.py", "x")
        let names = Set(FindCrawler.crawl(root: folder.url, rules: FindCrawlRules()).map(\.url.lastPathComponent))
        #expect(names == ["Lease.pdf", "Deck.key", "plan.md"])
        #expect(FindCrawler.isInsideSkippedFolder(folder.url.appending(path: "node_modules/a/b.md"), root: folder.url,
                                                  rules: FindCrawlRules()))
        #expect(!FindCrawler.isInsideSkippedFolder(folder.url.appending(path: "Docs/b.md"), root: folder.url,
                                                   rules: FindCrawlRules()))
    }
}
