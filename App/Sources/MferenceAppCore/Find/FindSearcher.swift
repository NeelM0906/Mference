import Foundation

/// Ranks files for a query by words and meaning (pipeline board, query path).
///
/// Fusion follows what DigUp measured to beat reciprocal-rank fusion: a
/// file qualifies when its best passage is unusually close in meaning for
/// this query (z ≥ `meaningGate`, against the query's cosine spread over the
/// index) or when it holds at least half the query's words; its score is
/// `wMeaning·max(z, 0) + wWords·coverage^1.5`, plus small bonuses for a name
/// match and for the kind a query names ("screenshot", "pdf"). Results
/// under half the top score are folded as weaker matches, and copies with
/// the same text collapse into one.
public struct FindSearcher: Sendable {
    public let index: FindIndex
    public let embedder: (any FindTextEmbedding)?

    /// z-score a meaning hit needs to count on its own.
    public static let meaningGate: Float = 2.0
    /// Word coverage a file needs to count on words alone.
    public static let wordGate = 0.5
    /// A result under this fraction of the top score is folded away.
    public static let weakFraction = 0.5

    public init(index: FindIndex, embedder: (any FindTextEmbedding)?) {
        self.index = index
        self.embedder = embedder
    }

    /// Words only, for each keystroke: the last word is matched as a prefix.
    public func searchWords(_ query: FindQuery, now: Date = .now) async throws -> FindResults {
        try await search(query, vector: nil, lastIsPrefix: true, now: now)
    }

    /// Words and meaning. `vector` is the query's embedding when the caller
    /// already has it (embedded while typing); otherwise it is computed here.
    public func search(_ query: FindQuery, vector: [Float]? = nil, now: Date = .now) async throws -> FindResults {
        var vector = vector
        if vector == nil, let embedder, try await index.hasVectors {
            vector = try await embedder.embed(query: FindEmbeddingPrompts.query(query.text))
        }
        return try await search(query, vector: vector, lastIsPrefix: false, now: now)
    }

    private func search(_ query: FindQuery, vector: [Float]?, lastIsPrefix: Bool, now: Date) async throws
        -> FindResults {
        let started = Date()
        let words = FindText.queryWords(query.text)
        guard !words.isEmpty || vector != nil else { return FindResults(query: query.text, results: [],
                                                                         wordsOnly: vector == nil, seconds: 0) }
        // Queries made of ids and numbers lean on words.
        let idsOnly = !words.isEmpty && words.allSatisfy { $0.contains(where: \.isNumber) }
        let meaningWeight: Double = idsOnly ? 0.5 : 1
        let wordWeight: Double = idsOnly ? 4 : 2

        // Words: which files hold which query words, and the name matches.
        var wordsByFile: [Int64: Int] = [:]
        var nameFiles = Set<Int64>()
        for (offset, word) in words.enumerated() {
            let prefix = lastIsPrefix && offset == words.count - 1
            for file in try await index.filesContaining(word, isPrefix: prefix) { wordsByFile[file, default: 0] += 1 }
            nameFiles.formUnion(try await index.filesContaining(word, isPrefix: prefix, column: "name"))
        }

        // Meaning: the best passage per file.
        var meaning: [Int64: FindIndex.MeaningHit] = [:]
        if let vector {
            for hit in try await index.meaningHits(vector) where hit.z > (meaning[hit.fileID]?.z ?? -.infinity) {
                meaning[hit.fileID] = hit
            }
        }

        let kindHint = Self.kindNamed(in: words)
        var candidates = Set(wordsByFile.keys).union(meaning.keys)
        let files = try await index.files(ids: Array(candidates))
        let earliest = query.modified.earliest(now: now)
        candidates = candidates.filter { id in
            guard let file = files[id] else { return false }
            if !query.kind.includes(file.kind) { return false }
            if let earliest, file.modified < earliest { return false }
            if let folder = query.folder.map({ FindPaths.normalize($0.resolvingSymlinksInPath().path) }),
               !(file.path == folder || file.path.hasPrefix(folder.hasSuffix("/") ? folder : folder + "/")) {
                return false
            }
            return true
        }

        struct Scored {
            var fileID: Int64
            var score: Double
            var z: Float?
            var coverage: Double
        }
        var scored: [Scored] = []
        for id in candidates {
            let coverage = words.isEmpty ? 0 : Double(wordsByFile[id] ?? 0) / Double(words.count)
            let z = meaning[id]?.z
            let meaningCounts = (z ?? 0) >= Self.meaningGate
            guard meaningCounts || coverage >= Self.wordGate || nameFiles.contains(id) else { continue }
            var score = meaningWeight * Double(max(z ?? 0, 0)) + wordWeight * pow(coverage, 1.5)
            if nameFiles.contains(id) { score += 0.5 }
            if let kindHint, files[id]?.kind.filter == kindHint {
                score += 1
            }
            scored.append(Scored(fileID: id, score: score, z: z, coverage: coverage))
        }
        scored.sort { $0.score > $1.score }

        // Passages for the top files.
        var results: [FindResult] = []
        var seenHashes: [String: Int] = [:]
        let top = scored.prefix(query.limit * 2)
        var chunkFor: [Int64: Int64] = [:]
        for item in top {
            let usesMeaning = (item.z ?? 0) >= Self.meaningGate
            if usesMeaning, let hit = meaning[item.fileID] {
                chunkFor[item.fileID] = hit.chunkID
            } else if let chunk = try await index.bestWordChunk(fileID: item.fileID, words: words,
                                                               lastIsPrefix: lastIsPrefix) {
                chunkFor[item.fileID] = chunk
            }
        }
        let chunks = try await index.chunks(ids: Array(chunkFor.values))
        let topScore = top.first?.score ?? 0
        for item in top {
            guard let file = files[item.fileID] else { continue }
            if let hash = file.contentHash, let existing = seenHashes[hash] {
                results[existing].copies += 1
                continue
            }
            guard results.count < query.limit else { break }
            let chunk = chunkFor[item.fileID].flatMap { chunks[$0] }
            let modality = chunk?.modality ?? .text
            let meant = (item.z ?? 0) >= Self.meaningGate
            var reasons: [FindMatchReason] = []
            if chunk?.fromOCR == true { reasons.append(.textInImage) }
            switch modality {
            case .image where meant: reasons.append(.picture)
            case .audio where meant: reasons.append(.sound)
            case .text where meant: reasons.append(.meaning)
            default: break
            }
            if item.coverage > 0, chunk?.fromOCR != true, modality == .text { reasons.append(.words) }
            if nameFiles.contains(item.fileID) { reasons.append(.name) }
            let text = modality == .text ? chunk?.text ?? "" : Self.describe(chunk, kind: file.kind)
            let snippet = text.isEmpty || modality != .text
                ? (before: "", hit: "", after: "") : FindText.snippet(of: text, queryWords: words)
            if let hash = file.contentHash { seenHashes[hash] = results.count }
            results.append(FindResult(url: file.url, kind: file.kind, modified: file.modified,
                                      location: chunk?.location, before: snippet.before, hit: snippet.hit,
                                      after: snippet.after, reasons: reasons, score: item.score,
                                      isWeak: item.score < topScore * Self.weakFraction, passage: text,
                                      time: modality == .text ? nil : chunk?.start))
        }
        return FindResults(query: query.text, results: results, wordsOnly: vector == nil,
                           seconds: Date().timeIntervalSince(started))
    }

    /// What a media match is, in words (the preview and chat's Files tool
    /// show it in place of a passage).
    static func describe(_ chunk: FindIndex.StoredChunk?, kind: FindKind) -> String {
        guard let chunk else { return "" }
        let at = chunk.location.map { " at \($0)" } ?? ""
        switch (chunk.modality, kind) {
        case (.image, .video): return "Matched what the video shows\(at)."
        case (.image, _): return "Matched what the picture shows."
        case (.audio, .video): return "Matched what the video sounds like\(at)."
        case (.audio, _): return "Matched what the recording sounds like\(at)."
        case (.text, _): return chunk.text
        }
    }

    /// The kind a query names, if any ("that screenshot with…" → images).
    static func kindNamed(in words: [String]) -> FindKindFilter? {
        for word in words {
            switch word {
            case "screenshot", "screenshots", "photo", "photos", "image", "images", "picture", "pictures", "scan":
                return .images
            case "video", "videos", "clip", "clips", "movie", "film", "footage", "recording", "recordings", "audio",
                 "song", "songs", "music", "podcast", "voice", "memo", "sound":
                return .media
            case "pdf", "pdfs", "document", "documents", "doc", "docs":
                return .documents
            case "slides", "slide", "deck", "presentation", "spreadsheet", "sheet", "sheets", "excel":
                return .slidesAndSheets
            case "code", "function", "swift", "python":
                return .code
            default:
                continue
            }
        }
        return nil
    }
}
