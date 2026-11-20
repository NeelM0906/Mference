import Foundation

/// Find's one index file: an SQLite database with the files, their chunks,
/// an FTS5 table for words and the chunks' vectors as small BLOBs (int8 plus
/// a 1-bit code, `FindVectors.recordBytes` each).
///
/// Vectors are scanned by brute force from a packed copy in memory, loaded
/// on the first meaning search and dropped by `trimMemory()` when Find is
/// idle: a personal library is 10k–1M passages, which is 3–300 MB at 292
/// bytes each and a few milliseconds of popcounts, so there is no ANN index.
public actor FindIndex {
    /// Schema version; a change starts a fresh index.
    static let schemaVersion = "2"

    public struct FileRecord: Sendable, Hashable {
        public var id: Int64
        public var path: String
        public var root: String
        public var kind: FindKind
        public var size: Int64
        public var modified: Date
        public var inode: Int64
        public var device: Int64
        public var state: State
        public var reason: String?
        public var contentHash: String?
        public var lastUsed: Date

        public enum State: String, Sendable, Hashable {
            /// Seen, not read yet.
            case pending
            /// Words indexed; some passages still wait for meaning.
            case read
            /// Words and meaning both indexed.
            case learned
            /// Could not be read; `reason` says why.
            case skipped
        }

        public var url: URL { URL(filePath: path) }
    }

    /// One passage to store.
    public struct ChunkInput: Sendable, Hashable {
        public var location: String?
        public var text: String
        /// Read from an image by OCR (shown as "Text in image").
        public var fromOCR: Bool
        /// A photo, frame or stretch of sound learned by what it shows or
        /// sounds like; `text` is empty.
        public var media: FindMediaPiece?

        public init(location: String?, text: String, fromOCR: Bool = false, media: FindMediaPiece? = nil) {
            self.location = location
            self.text = text
            self.fromOCR = fromOCR
            self.media = media
        }
    }

    /// A passage waiting for its vector.
    public struct PendingChunk: Sendable, Hashable {
        public var chunkID: Int64
        public var fileID: Int64
        public var title: String
        public var text: String
        /// Set for a media piece (embedded from the file, not from `text`).
        public var media: FindMediaPiece?
        public var path: String
        public var kind: FindKind
    }

    public struct StoredChunk: Sendable, Hashable {
        public var chunkID: Int64
        public var fileID: Int64
        public var location: String?
        public var text: String
        public var fromOCR: Bool
        public var modality: FindModality
        /// Seconds into a video or audio file.
        public var start: Double?
    }

    public struct WordHit: Sendable, Hashable {
        public var chunkID: Int64
        public var fileID: Int64
        /// bm25: lower is better.
        public var rank: Double
    }

    public struct MeaningHit: Sendable, Hashable {
        public var chunkID: Int64
        public var fileID: Int64
        public var cosine: Float
        /// Standard scores against this query's cosine spread.
        public var z: Float
    }

    private let db: FindSQLite
    private let location: URL?
    private var cache: VectorCache?

    /// Opens (or creates) the index at `directory/find.sqlite`; nil keeps it
    /// in memory (tests).
    public init(directory: URL?) throws {
        if let directory {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            location = directory.appending(path: "find.sqlite")
        } else {
            location = nil
        }
        db = try FindSQLite(path: location?.path)
        try Self.migrate(db)
    }

    private static func migrate(_ db: FindSQLite) throws {
        try db.execute("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA foreign_keys=OFF;")
        try db.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        var version: String?
        try db.query("SELECT value FROM meta WHERE key='schema'") { version = $0.text(0) }
        if let version, version != schemaVersion {
            try db.execute("DROP TABLE IF EXISTS files; DROP TABLE IF EXISTS chunks; DROP TABLE IF EXISTS chunk_fts;")
        }
        try db.execute("""
            CREATE TABLE IF NOT EXISTS files(
              id INTEGER PRIMARY KEY,
              path TEXT NOT NULL UNIQUE,
              root TEXT NOT NULL,
              kind TEXT NOT NULL,
              size INTEGER NOT NULL,
              mtime REAL NOT NULL,
              inode INTEGER NOT NULL,
              device INTEGER NOT NULL,
              state TEXT NOT NULL,
              reason TEXT,
              content_hash TEXT,
              last_used REAL NOT NULL DEFAULT 0);
            CREATE INDEX IF NOT EXISTS files_inode ON files(device, inode);
            CREATE INDEX IF NOT EXISTS files_root ON files(root);
            CREATE TABLE IF NOT EXISTS chunks(
              id INTEGER PRIMARY KEY,
              file_id INTEGER NOT NULL,
              ord INTEGER NOT NULL,
              loc TEXT,
              ocr INTEGER NOT NULL,
              wants_vector INTEGER NOT NULL,
              vec BLOB,
              modality INTEGER NOT NULL DEFAULT 0,
              t0 REAL,
              t1 REAL,
              tokens INTEGER NOT NULL DEFAULT 0);
            CREATE INDEX IF NOT EXISTS chunks_file ON chunks(file_id);
            CREATE INDEX IF NOT EXISTS chunks_pending ON chunks(wants_vector) WHERE wants_vector = 1 AND vec IS NULL;
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
              name, body, tokenize = 'porter unicode61 remove_diacritics 2');
            """)
        // Added within schema 2: where a photo prepared while reading waits.
        var columns: Set<String> = []
        try db.query("PRAGMA table_info(chunks)") { columns.insert($0.text(1)) }
        if !columns.contains("prep") { try db.execute("ALTER TABLE chunks ADD COLUMN prep TEXT") }
        try db.run("INSERT OR REPLACE INTO meta(key, value) VALUES('schema', ?)", [.text(schemaVersion)])
    }

    // MARK: Meta

    public func meta(_ key: String) throws -> String? {
        var value: String?
        try db.query("SELECT value FROM meta WHERE key=?", [.text(key)]) { value = $0.text(0) }
        return value
    }

    public func setMeta(_ key: String, _ value: String?) throws {
        if let value {
            try db.run("INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", [.text(key), .text(value)])
        } else {
            try db.run("DELETE FROM meta WHERE key=?", [.text(key)])
        }
    }

    // MARK: Files

    private static let fileColumns =
        "id, path, root, kind, size, mtime, inode, device, state, reason, content_hash, last_used"

    private static func fileRecord(_ row: FindSQLite.Row) -> FileRecord {
        FileRecord(id: row.int(0), path: row.text(1), root: row.text(2),
                   kind: FindKind(rawValue: row.text(3)) ?? .text, size: row.int(4),
                   modified: Date(timeIntervalSince1970: row.double(5)), inode: row.int(6), device: row.int(7),
                   state: FileRecord.State(rawValue: row.text(8)) ?? .pending,
                   reason: row.isNull(9) ? nil : row.text(9), contentHash: row.isNull(10) ? nil : row.text(10),
                   lastUsed: Date(timeIntervalSince1970: row.double(11)))
    }

    public func file(path: String) throws -> FileRecord? {
        var record: FileRecord?
        try db.query("SELECT \(Self.fileColumns) FROM files WHERE path=?", [.text(path)]) {
            record = Self.fileRecord($0)
        }
        return record
    }

    public func files(ids: [Int64]) throws -> [Int64: FileRecord] {
        var records: [Int64: FileRecord] = [:]
        for batch in stride(from: 0, to: ids.count, by: 500) {
            let slice = ids[batch..<min(ids.count, batch + 500)]
            let marks = slice.map { _ in "?" }.joined(separator: ",")
            try db.query("SELECT \(Self.fileColumns) FROM files WHERE id IN (\(marks))", slice.map { .int($0) }) {
                let record = Self.fileRecord($0)
                records[record.id] = record
            }
        }
        return records
    }

    /// Every file under `root`, by path.
    public func files(root: String) throws -> [String: FileRecord] {
        var records: [String: FileRecord] = [:]
        try db.query("SELECT \(Self.fileColumns) FROM files WHERE root=?", [.text(root)]) {
            let record = Self.fileRecord($0)
            records[record.path] = record
        }
        return records
    }

    /// Files under `root`.
    public func fileCount(root: String) throws -> Int {
        Int(try db.scalarInt("SELECT COUNT(*) FROM files WHERE root=?", [.text(root)]) ?? 0)
    }

    public func roots() throws -> [String] {
        var roots: [String] = []
        try db.query("SELECT DISTINCT root FROM files") { roots.append($0.text(0)) }
        return roots
    }

    /// Files whose text has not been read yet: images, video and audio last
    /// (their OCR and sampling are the slow part, and Vision's first start can
    /// take tens of seconds), then most recently used first.
    public func unreadFiles(limit: Int) throws -> [FileRecord] {
        var records: [FileRecord] = []
        try db.query("""
            SELECT \(Self.fileColumns) FROM files WHERE state='pending'
            ORDER BY (kind IN ('image', 'video', 'audio')), last_used DESC, mtime DESC LIMIT ?
            """, [.int(Int64(limit))]) { records.append(Self.fileRecord($0)) }
        return records
    }

    /// Adds a file seen on disk, or updates its attributes. A changed size or
    /// date sends it back to `pending`. Returns its id.
    @discardableResult
    public func upsertFile(path: String, root: String, kind: FindKind, size: Int64, modified: Date, inode: Int64,
                           device: Int64, lastUsed: Date?) throws -> Int64 {
        if let existing = try file(path: path) {
            let changed = existing.size != size || abs(existing.modified.timeIntervalSince1970
                - modified.timeIntervalSince1970) > 0.5
            try db.run("""
                UPDATE files SET root=?, kind=?, size=?, mtime=?, inode=?, device=?, last_used=?,
                state = CASE WHEN ? THEN 'pending' ELSE state END,
                reason = CASE WHEN ? THEN NULL ELSE reason END
                WHERE id=?
                """, [.text(root), .text(kind.rawValue), .int(size), .double(modified.timeIntervalSince1970),
                      .int(inode), .int(device), .double((lastUsed ?? existing.lastUsed).timeIntervalSince1970),
                      .int(changed ? 1 : 0), .int(changed ? 1 : 0), .int(existing.id)])
            return existing.id
        }
        try db.run("""
            INSERT INTO files(path, root, kind, size, mtime, inode, device, state, last_used)
            VALUES(?, ?, ?, ?, ?, ?, ?, 'pending', ?)
            """, [.text(path), .text(root), .text(kind.rawValue), .int(size), .double(modified.timeIntervalSince1970),
                  .int(inode), .int(device), .double((lastUsed ?? modified).timeIntervalSince1970)])
        return db.lastInsertRowID
    }

    /// A file that moved: same inode, new path. Its text and vectors stay.
    public func move(fileID: Int64, to path: String, root: String) throws {
        try db.run("UPDATE files SET path=?, root=? WHERE id=?", [.text(path), .text(root), .int(fileID)])
        let name = URL(filePath: path).lastPathComponent
        // The name lives in the first chunk's FTS row.
        var first: Int64?
        try db.query("SELECT id FROM chunks WHERE file_id=? ORDER BY ord LIMIT 1", [.int(fileID)]) { first = $0.int(0) }
        if let first {
            var body = ""
            try db.query("SELECT body FROM chunk_fts WHERE rowid=?", [.int(first)]) { body = $0.text(0) }
            try db.run("DELETE FROM chunk_fts WHERE rowid=?", [.int(first)])
            try db.run("INSERT INTO chunk_fts(rowid, name, body) VALUES(?, ?, ?)",
                       [.int(first), .text(Self.searchableName(name)), .text(body)])
        }
    }

    public func removeFiles(ids: [Int64]) throws {
        guard !ids.isEmpty else { return }
        try db.transaction {
            for id in ids {
                try deleteChunks(fileID: id)
                try db.run("DELETE FROM files WHERE id=?", [.int(id)])
            }
        }
        cache = nil
    }

    /// Removes every file under `root` (a folder taken out of Find).
    public func removeRoot(_ root: String) throws {
        var ids: [Int64] = []
        try db.query("SELECT id FROM files WHERE root=?", [.text(root)]) { ids.append($0.int(0)) }
        try removeFiles(ids: ids)
    }

    private func deleteChunks(fileID: Int64) throws {
        var chunkIDs: [Int64] = []
        var prepared: [String] = []
        try db.query("SELECT id, prep FROM chunks WHERE file_id=?", [.int(fileID)]) {
            chunkIDs.append($0.int(0))
            if !$0.isNull(1) { prepared.append($0.text(1)) }
        }
        for path in prepared { try? FileManager.default.removeItem(atPath: path) }
        for chunkID in chunkIDs {
            try db.run("DELETE FROM chunk_fts WHERE rowid=?", [.int(chunkID)])
        }
        try db.run("DELETE FROM chunks WHERE file_id=?", [.int(fileID)])
        if !chunkIDs.isEmpty { cache = nil }
    }

    /// Stores a file's text: its name and passages become searchable by
    /// words at once; prose passages and media pieces wait for meaning. A
    /// file with no text still gets a name-only row.
    public func storeText(fileID: Int64, name: String, chunks: [ChunkInput], contentHash: String?) throws {
        try db.transaction {
            try deleteChunks(fileID: fileID)
            let rows = chunks.isEmpty ? [ChunkInput(location: nil, text: "")] : chunks
            var wantsAny = false
            for (ord, chunk) in rows.enumerated() {
                // OCR text gets meaning too when it reads as prose, so a
                // screenshot matches what it says even without its picture.
                let wants = chunk.media != nil || FindText.isProse(chunk.text)
                wantsAny = wantsAny || wants
                let media = chunk.media
                try db.run("""
                    INSERT INTO chunks(file_id, ord, loc, ocr, wants_vector, modality, t0, t1, tokens, prep)
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [.int(fileID), .int(Int64(ord)), chunk.location.map { .text($0) } ?? .null,
                          .int(chunk.fromOCR ? 1 : 0), .int(wants ? 1 : 0),
                          .int(Int64(media?.modality.rawValue ?? 0)), media?.start.map { .double($0) } ?? .null,
                          media?.end.map { .double($0) } ?? .null, .int(Int64(media?.tokens ?? 0)),
                          media?.prepared.map { .text($0) } ?? .null])
                let chunkID = db.lastInsertRowID
                try db.run("INSERT INTO chunk_fts(rowid, name, body) VALUES(?, ?, ?)",
                           [.int(chunkID), .text(ord == 0 ? Self.searchableName(name) : ""), .text(chunk.text)])
            }
            try db.run("UPDATE files SET state=?, reason=NULL, content_hash=? WHERE id=?",
                       [.text(wantsAny ? FileRecord.State.read.rawValue : FileRecord.State.learned.rawValue),
                        contentHash.map { .text($0) } ?? .null, .int(fileID)])
        }
    }

    /// "Q3_close-final.xlsx" → "Q3 close final xlsx": the name's words.
    static func searchableName(_ name: String) -> String {
        name.replacingOccurrences(of: "_", with: " ").replacingOccurrences(of: "-", with: " ")
            .replacingOccurrences(of: ".", with: " ")
    }

    public func markSkipped(fileID: Int64, reason: String) throws {
        try db.transaction {
            try deleteChunks(fileID: fileID)
            try db.run("UPDATE files SET state='skipped', reason=? WHERE id=?", [.text(reason), .int(fileID)])
            // Still findable by name.
            var name = ""
            try db.query("SELECT path FROM files WHERE id=?", [.int(fileID)]) { name = URL(filePath: $0.text(0)).lastPathComponent }
            try db.run("INSERT INTO chunks(file_id, ord, loc, ocr, wants_vector) VALUES(?, 0, NULL, 0, 0)",
                       [.int(fileID)])
            try db.run("INSERT INTO chunk_fts(rowid, name, body) VALUES(?, ?, '')",
                       [.int(db.lastInsertRowID), .text(Self.searchableName(name))])
        }
    }

    public func skippedFiles(limit: Int = 200) throws -> [FindSkippedFile] {
        var files: [FindSkippedFile] = []
        try db.query("SELECT path, reason FROM files WHERE state='skipped' ORDER BY path LIMIT ?",
                     [.int(Int64(limit))]) {
            files.append(FindSkippedFile(url: URL(filePath: $0.text(0)), reason: $0.text(1)))
        }
        return files
    }

    // MARK: Meaning queue

    /// The first passes of every file come before the rest of long files
    /// (WindowsFinder's audit: cap the first pass per file).
    public static let firstPassChunks = 40

    /// Passages waiting for vectors: pictures already prepared first (their
    /// files wait on disk), then first passes, recently used files first.
    /// Media pieces only with `includeMedia` (an embedder that has the media
    /// encoders).
    public func pendingChunks(limit: Int, includeMedia: Bool = false) throws -> [PendingChunk] {
        var pending: [PendingChunk] = []
        try db.query("""
            SELECT c.id, c.file_id, f.path, chunk_fts.body, c.modality, c.t0, c.t1, c.tokens, f.kind, c.prep FROM chunks c
            JOIN files f ON f.id = c.file_id
            JOIN chunk_fts ON chunk_fts.rowid = c.id
            WHERE c.wants_vector = 1 AND c.vec IS NULL AND (? OR c.modality = 0)
            ORDER BY (c.prep IS NULL), (c.ord >= ?), f.last_used DESC, f.mtime DESC, c.file_id, c.ord
            LIMIT ?
            """, [.int(includeMedia ? 1 : 0), .int(Int64(Self.firstPassChunks)), .int(Int64(limit))]) {
            let path = $0.text(2)
            let name = URL(filePath: path).deletingPathExtension().lastPathComponent
            let modality = FindModality(rawValue: Int($0.int(4))) ?? .text
            let media = modality == .text ? nil : FindMediaPiece(
                modality: modality, start: $0.isNull(5) ? nil : $0.double(5), end: $0.isNull(6) ? nil : $0.double(6),
                tokens: Int($0.int(7)), prepared: $0.isNull(9) ? nil : $0.text(9))
            pending.append(PendingChunk(chunkID: $0.int(0), fileID: $0.int(1), title: name, text: $0.text(3),
                                        media: media, path: path, kind: FindKind(rawValue: $0.text(8)) ?? .text))
        }
        return pending
    }

    /// Pictures prepared while reading that still wait to be learned (the
    /// indexer stops preparing ahead past a limit, to bound the disk used).
    public func preparedBacklog() throws -> Int {
        Int(try db.scalarInt("SELECT COUNT(*) FROM chunks WHERE prep IS NOT NULL AND vec IS NULL") ?? 0)
    }

    /// The prepared file of a piece is used or gone: forget it.
    public func clearPrepared(chunkIDs: [Int64]) throws {
        guard !chunkIDs.isEmpty else { return }
        try db.transaction {
            for id in chunkIDs { try db.run("UPDATE chunks SET prep=NULL WHERE id=?", [.int(id)]) }
        }
    }

    /// Forgets every prepared file (their folder is temporary: emptied at
    /// start, so a piece decodes again when it is learned).
    public func forgetPrepared() throws {
        try db.run("UPDATE chunks SET prep=NULL WHERE prep IS NOT NULL")
    }

    /// Media pieces that could not be decoded: they stop waiting for a
    /// vector (the file stays findable by name and any text).
    public func dropPending(chunkIDs: [Int64]) throws {
        guard !chunkIDs.isEmpty else { return }
        var files = Set<Int64>()
        try db.transaction {
            for id in chunkIDs {
                try db.query("SELECT file_id FROM chunks WHERE id=?", [.int(id)]) { files.insert($0.int(0)) }
                try db.run("UPDATE chunks SET wants_vector=0 WHERE id=?", [.int(id)])
            }
            for fileID in files { try finishIfLearned(fileID) }
        }
    }

    /// Sends files of these kinds back to be read again (a setting that
    /// changes how they are read was switched).
    public func requeue(kinds: [FindKind]) throws {
        guard !kinds.isEmpty else { return }
        let marks = kinds.map { _ in "?" }.joined(separator: ",")
        try db.run("UPDATE files SET state='pending', reason=NULL WHERE kind IN (\(marks))",
                   kinds.map { .text($0.rawValue) })
    }

    private func finishIfLearned(_ fileID: Int64) throws {
        let remaining = try db.scalarInt(
            "SELECT COUNT(*) FROM chunks WHERE file_id=? AND wants_vector=1 AND vec IS NULL", [.int(fileID)]) ?? 0
        if remaining == 0 {
            try db.run("UPDATE files SET state='learned' WHERE id=? AND state='read'", [.int(fileID)])
        }
    }

    /// Stores prepared vectors and marks files whose passages are all done.
    public func storeVectors(_ vectors: [(chunkID: Int64, fileID: Int64, vector: [Float])]) throws {
        guard !vectors.isEmpty else { return }
        var records: [(Int64, Int64, UInt8, Data)] = []
        try db.transaction {
            for item in vectors {
                let data = FindVectors.encode(item.vector)
                try db.run("UPDATE chunks SET vec=? WHERE id=?", [.blob(data), .int(item.chunkID)])
                if db.changes > 0 {
                    let modality = try db.scalarInt("SELECT modality FROM chunks WHERE id=?", [.int(item.chunkID)]) ?? 0
                    records.append((item.chunkID, item.fileID, UInt8(truncatingIfNeeded: modality), data))
                }
            }
            for fileID in Set(vectors.map(\.fileID)) { try finishIfLearned(fileID) }
        }
        cache?.append(records)
    }

    /// Drops every vector (the model changed); files go back to `read`.
    public func clearVectors() throws {
        try db.transaction {
            try db.run("UPDATE chunks SET vec=NULL")
            try db.run("UPDATE files SET state='read' WHERE state='learned' AND id IN (SELECT file_id FROM chunks WHERE wants_vector=1)")
        }
        cache = nil
    }

    // MARK: Status

    public func counts() throws -> (total: Int, read: Int, learned: Int, skipped: Int, vectors: Int) {
        var total = 0, read = 0, learned = 0, skipped = 0
        try db.query("SELECT state, COUNT(*) FROM files GROUP BY state") {
            let count = Int($0.int(1))
            total += count
            switch FileRecord.State(rawValue: $0.text(0)) {
            case .read: read += count
            case .learned: read += count; learned += count
            case .skipped: skipped += count
            case .pending, nil: break
            }
        }
        let vectors = Int(try db.scalarInt("SELECT COUNT(*) FROM chunks WHERE vec IS NOT NULL") ?? 0)
        return (total, read, learned, skipped, vectors)
    }

    /// Bytes on disk (database plus its WAL).
    public func sizeOnDisk() -> UInt64 {
        guard let location else { return 0 }
        return [location.path, location.path + "-wal", location.path + "-shm"].reduce(0) { total, path in
            total + ((try? FileManager.default.attributesOfItem(atPath: path)[.size] as? NSNumber)?.uint64Value ?? 0)
        }
    }

    // MARK: Search

    /// FTS5 syntax for words: each quoted, OR-ed, the last one as a prefix
    /// (it may still be being typed).
    static func matchExpression(_ words: [String], lastIsPrefix: Bool, column: String? = nil) -> String? {
        let terms = words.enumerated().compactMap { index, word -> String? in
            let cleaned = word.replacingOccurrences(of: "\"", with: "")
            guard !cleaned.isEmpty else { return nil }
            let prefix = lastIsPrefix && index == words.count - 1 && cleaned.count >= 2 ? "*" : ""
            return "\"\(cleaned)\"\(prefix)"
        }
        guard !terms.isEmpty else { return nil }
        let joined = terms.joined(separator: " OR ")
        if let column { return "\(column) : (\(joined))" }
        return joined
    }

    /// The best word matches, best first.
    public func wordHits(_ words: [String], lastIsPrefix: Bool, limit: Int = 300) throws -> [WordHit] {
        guard let expression = Self.matchExpression(words, lastIsPrefix: lastIsPrefix) else { return [] }
        var hits: [WordHit] = []
        try db.query("""
            SELECT chunk_fts.rowid, bm25(chunk_fts, 3.0, 1.0) AS rank, c.file_id FROM chunk_fts
            JOIN chunks c ON c.id = chunk_fts.rowid
            WHERE chunk_fts MATCH ? ORDER BY rank LIMIT ?
            """, [.text(expression), .int(Int64(limit))]) {
            hits.append(WordHit(chunkID: $0.int(0), fileID: $0.int(2), rank: $0.double(1)))
        }
        return hits
    }

    /// Files containing `word` (anywhere, name included).
    public func filesContaining(_ word: String, isPrefix: Bool, column: String? = nil,
                                limit: Int = 20_000) throws -> Set<Int64> {
        guard let expression = Self.matchExpression([word], lastIsPrefix: isPrefix, column: column) else { return [] }
        var ids = Set<Int64>()
        try db.query("""
            SELECT DISTINCT c.file_id FROM chunk_fts JOIN chunks c ON c.id = chunk_fts.rowid
            WHERE chunk_fts MATCH ? LIMIT ?
            """, [.text(expression), .int(Int64(limit))]) { ids.insert($0.int(0)) }
        return ids
    }

    /// The passages with these ids.
    public func chunks(ids: [Int64]) throws -> [Int64: StoredChunk] {
        var chunks: [Int64: StoredChunk] = [:]
        for batch in stride(from: 0, to: ids.count, by: 500) {
            let slice = ids[batch..<min(ids.count, batch + 500)]
            let marks = slice.map { _ in "?" }.joined(separator: ",")
            try db.query("""
                SELECT c.id, c.file_id, c.loc, chunk_fts.body, c.ocr, c.modality, c.t0 FROM chunks c
                JOIN chunk_fts ON chunk_fts.rowid = c.id WHERE c.id IN (\(marks))
                """, slice.map { .int($0) }) {
                chunks[$0.int(0)] = StoredChunk(chunkID: $0.int(0), fileID: $0.int(1),
                                                location: $0.isNull(2) ? nil : $0.text(2), text: $0.text(3),
                                                fromOCR: $0.int(4) == 1,
                                                modality: FindModality(rawValue: Int($0.int(5))) ?? .text,
                                                start: $0.isNull(6) ? nil : $0.double(6))
            }
        }
        return chunks
    }

    /// The best passage of each file by word rank, for files with no meaning hit.
    public func bestWordChunk(fileID: Int64, words: [String], lastIsPrefix: Bool) throws -> Int64? {
        var best: Int64?
        if let expression = Self.matchExpression(words, lastIsPrefix: lastIsPrefix) {
            try db.query("""
                SELECT chunk_fts.rowid FROM chunk_fts JOIN chunks c ON c.id = chunk_fts.rowid
                WHERE chunk_fts MATCH ? AND c.file_id = ? ORDER BY bm25(chunk_fts, 3.0, 1.0) LIMIT 1
                """, [.text(expression), .int(fileID)]) { best = $0.int(0) }
        }
        if best == nil {
            try db.query("SELECT id FROM chunks WHERE file_id=? ORDER BY ord LIMIT 1", [.int(fileID)]) { best = $0.int(0) }
        }
        return best
    }

    public var hasVectors: Bool {
        get throws { (try db.scalarInt("SELECT EXISTS(SELECT 1 FROM chunks WHERE vec IS NOT NULL)") ?? 0) == 1 }
    }

    /// The passages closest in meaning to `query`: a 1-bit scan over every
    /// vector keeps the `candidates` nearest of each modality, int8 rescoring
    /// orders them, and each gets a z-score against this query's cosine
    /// spread (from a sample of the whole index), so a query with no good
    /// answer has no high z. Text, picture and sound vectors sit at different
    /// cosine levels for the same query, so each is first centred on its own
    /// mean (DigUp measured this beats one scale for all).
    public func meaningHits(_ query: [Float], candidates: Int = 600) throws -> [MeaningHit] {
        let cache = try loadedCache()
        guard cache.count > 0 else { return [] }
        let prepared = FindVectors.Query(FindVectors.prepare(query))
        return cache.search(prepared, candidates: candidates)
    }

    /// Frees the in-memory vector copy (Find idle).
    public func trimMemory() {
        cache = nil
    }

    private func loadedCache() throws -> VectorCache {
        if let cache { return cache }
        var loaded = VectorCache()
        try db.query("SELECT id, file_id, vec, modality FROM chunks WHERE vec IS NOT NULL") { row in
            row.withBlob(2) { bytes in
                guard bytes.count == FindVectors.recordBytes else { return }
                loaded.append(chunkID: row.int(0), fileID: row.int(1),
                              modality: UInt8(truncatingIfNeeded: row.int(3)), record: bytes)
            }
        }
        cache = loaded
        return loaded
    }

    /// Closes the database (Find turned off or index deleted).
    public func close() {
        cache = nil
        db.close()
    }

    /// Deletes the index files. The actor is unusable afterwards.
    public func deleteFiles() {
        close()
        guard let location else { return }
        for suffix in ["", "-wal", "-shm"] {
            try? FileManager.default.removeItem(atPath: location.path + suffix)
        }
    }
}

/// Every vector packed back to back, with their chunk and file ids and
/// modality (`FindModality.rawValue`).
struct VectorCache {
    private(set) var records = Data()
    private(set) var chunkIDs: [Int64] = []
    private(set) var fileIDs: [Int64] = []
    private(set) var modalities: [UInt8] = []

    /// Modalities with fewer sampled vectors than this use the overall mean.
    static let minimumGroup = 20

    var count: Int { chunkIDs.count }

    mutating func append(chunkID: Int64, fileID: Int64, modality: UInt8 = 0, record: UnsafeRawBufferPointer) {
        records.append(contentsOf: record)
        chunkIDs.append(chunkID)
        fileIDs.append(fileID)
        modalities.append(min(modality, 2))
    }

    mutating func append(_ items: [(Int64, Int64, UInt8, Data)]) {
        for (chunkID, fileID, modality, data) in items {
            data.withUnsafeBytes { append(chunkID: chunkID, fileID: fileID, modality: modality, record: $0) }
        }
    }

    func search(_ query: FindVectors.Query, candidates: Int) -> [FindIndex.MeaningHit] {
        let size = FindVectors.recordBytes
        return records.withUnsafeBytes { all -> [FindIndex.MeaningHit] in
            func record(_ index: Int) -> UnsafeRawBufferPointer {
                UnsafeRawBufferPointer(rebasing: all[(index * size)..<((index + 1) * size)])
            }
            // 1-bit pass: Hamming distance of every record, then a histogram
            // cut per modality that keeps about `candidates` of the nearest
            // of each (pictures sit farther from a text query than text does).
            var distances = [UInt16](repeating: 0, count: count)
            var histograms = [[Int]](repeating: [Int](repeating: 0, count: FindVectors.dimensions + 1), count: 3)
            for index in 0..<count {
                let distance = FindVectors.hamming(query, record: record(index))
                distances[index] = UInt16(distance)
                histograms[Int(modalities[index])][distance] += 1
            }
            let cutoffs = histograms.map { histogram -> Int in
                var cutoff = 0
                var kept = 0
                while cutoff < histogram.count {
                    kept += histogram[cutoff]
                    if kept >= candidates { break }
                    cutoff += 1
                }
                return cutoff
            }
            // This query's cosine levels over a sample of the index: each
            // modality's mean, and the spread of all once centred.
            let stride = max(1, count / 2_048)
            var sums = [Float](repeating: 0, count: 3)
            var counts = [Float](repeating: 0, count: 3)
            var sampled: [(UInt8, Float)] = []
            for index in Swift.stride(from: 0, to: count, by: stride) {
                let cosine = FindVectors.cosine(query, record: record(index))
                sums[Int(modalities[index])] += cosine
                counts[Int(modalities[index])] += 1
                sampled.append((modalities[index], cosine))
            }
            let overall = sums.reduce(0, +) / max(1, counts.reduce(0, +))
            let means = (0..<3).map { counts[$0] >= Float(Self.minimumGroup) ? sums[$0] / counts[$0] : overall }
            var centredSum: Float = 0
            var centredSquares: Float = 0
            for (modality, cosine) in sampled {
                let centred = cosine - means[Int(modality)]
                centredSum += centred
                centredSquares += centred * centred
            }
            let n = Float(max(1, sampled.count))
            let centredMean = centredSum / n
            let deviation = max(0.02, (centredSquares / n - centredMean * centredMean).squareRoot())
            // int8 rescoring of the kept candidates.
            var hits: [FindIndex.MeaningHit] = []
            hits.reserveCapacity(min(count, candidates * 3))
            for index in 0..<count where Int(distances[index]) <= cutoffs[Int(modalities[index])] {
                let cosine = FindVectors.cosine(query, record: record(index))
                let z = (cosine - means[Int(modalities[index])] - centredMean) / deviation
                hits.append(FindIndex.MeaningHit(chunkID: chunkIDs[index], fileID: fileIDs[index], cosine: cosine,
                                                 z: z))
            }
            hits.sort { $0.z > $1.z }
            return Array(hits.prefix(candidates * 2))
        }
    }
}
