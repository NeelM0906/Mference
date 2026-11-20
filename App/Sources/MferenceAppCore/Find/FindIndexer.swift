import Foundation

/// Keeps the index in step with the folders (pipeline board, index path):
/// syncs folders, reads new and changed files so their words are searchable,
/// then learns their meaning in the background.
///
/// Reading runs whenever there is work; learning only while `learningGate`
/// says nothing stands in the way (Chat answering, battery, heat). Both
/// stop while Chat is answering.
public actor FindIndexer {
    public struct Configuration: Sendable, Hashable {
        public var roots: [URL]
        public var rules: FindCrawlRules
        /// Read the Photos library too (`FindPhotos`).
        public var photos: Bool

        public init(roots: [URL], rules: FindCrawlRules, photos: Bool = false) {
            self.roots = roots.map { URL(filePath: FindPaths.normalize($0.resolvingSymlinksInPath().path),
                                         directoryHint: .isDirectory) }
            self.rules = rules
            self.photos = photos
        }
    }

    /// Why work must wait, or nil when it may go on.
    public struct Gate: Sendable {
        /// Reading (extraction) may run; nil to go on.
        public var reading: @Sendable () async -> String?
        /// Learning (embedding) may run; nil to go on.
        public var learning: @Sendable () async -> String?

        public init(reading: @escaping @Sendable () async -> String? = { nil },
                    learning: @escaping @Sendable () async -> String? = { nil }) {
            self.reading = reading
            self.learning = learning
        }
    }

    public let index: FindIndex
    private let extractor: any FindExtracting
    private var embedder: (any FindTextEmbedding)?
    private var configuration: Configuration
    private let gate: Gate
    private let onStatus: @Sendable (FindIndexStatus) -> Void
    private let recentFiles: @Sendable ([URL]) -> [String: Date]
    private let photoLibrary: (any FindPhotoLibrary)?

    private var worker: Task<Void, Never>?
    private var wake: AsyncStream<Void>.Continuation?
    private var phase: FindIndexStatus.Phase = .upToDate
    private var pauseReason: String?
    /// Files read per step, passages embedded per step.
    public var readBatch = 16
    public var learnBatch = 32
    /// How long a paused gate waits before asking again.
    public var gatePoll: Duration = .seconds(5)

    public init(index: FindIndex, extractor: any FindExtracting, embedder: (any FindTextEmbedding)?,
                configuration: Configuration, gate: Gate = Gate(),
                recentFiles: @escaping @Sendable ([URL]) -> [String: Date] = { _ in [:] },
                photoLibrary: (any FindPhotoLibrary)? = nil,
                onStatus: @escaping @Sendable (FindIndexStatus) -> Void = { _ in }) {
        self.index = index
        self.extractor = extractor
        self.embedder = embedder
        self.configuration = configuration
        self.gate = gate
        self.recentFiles = recentFiles
        self.photoLibrary = photoLibrary
        self.onStatus = onStatus
    }

    // MARK: Control

    /// Starts the background loop (idempotent).
    public func start() {
        guard worker == nil else { return }
        let (stream, continuation) = AsyncStream<Void>.makeStream(bufferingPolicy: .bufferingNewest(1))
        wake = continuation
        worker = Task(priority: .utility) { [weak self] in
            for await _ in stream {
                guard let self, !Task.isCancelled else { return }
                await self.drain()
            }
        }
        kick()
    }

    /// Stops the loop and the extraction helper.
    public func stop() async {
        worker?.cancel()
        wake?.finish()
        worker = nil
        wake = nil
        await extractor.shutdown()
    }

    /// Wakes the loop to look for work.
    public func kick() {
        wake?.yield()
    }

    public func setEmbedder(_ embedder: (any FindTextEmbedding)?) {
        self.embedder = embedder
        kick()
    }

    public func setConfiguration(_ configuration: Configuration) async throws {
        var removed = Set(self.configuration.roots.map(\.path)).subtracting(configuration.roots.map(\.path))
        if self.configuration.photos, !configuration.photos { removed.insert(FindPhotos.rootPath) }
        self.configuration = configuration
        for root in removed { try await index.removeRoot(root) }
        try await syncAll()
    }

    // MARK: Sync

    /// Crawls every folder and brings the index in step: new files, changed
    /// files, moves (same inode at a new path) and deletions.
    public func syncAll() async throws {
        setPhase(.scanning)
        let configuration = self.configuration
        let recent = recentFiles(configuration.roots)
        for root in configuration.roots {
            let entries = await Task.detached(priority: .utility) {
                FindCrawler.crawl(root: root, rules: configuration.rules)
            }.value
            try await apply(entries, root: root, recent: recent)
        }
        if configuration.photos { try await syncPhotos(publishing: false) }
        // Folders no longer chosen.
        var chosen = Set(configuration.roots.map(\.path))
        if configuration.photos { chosen.insert(FindPhotos.rootPath) }
        for root in try await index.roots() where !chosen.contains(root) {
            try await index.removeRoot(root)
        }
        setPhase(.upToDate)
        await publish()
        kick()
    }

    /// Brings the Photos library's items in step (added, edited, deleted).
    public func syncPhotos(publishing: Bool = true) async throws {
        guard configuration.photos, let photoLibrary else { return }
        let entries = await photoLibrary.entries()
        try await apply(entries, root: FindPhotos.root, recent: [:])
        if publishing {
            await publish()
            kick()
        }
    }

    private func apply(_ entries: [FindCrawlEntry], root: URL, recent: [String: Date]) async throws {
        var known = try await index.files(root: root.path)
        var missing = known
        var unmatched: [FindCrawlEntry] = []
        for entry in entries {
            if missing.removeValue(forKey: entry.url.path) != nil {
                try await index.upsertFile(path: entry.url.path, root: root.path, kind: entry.kind, size: entry.size,
                                           modified: entry.modified, inode: entry.inode, device: entry.device,
                                           lastUsed: recent[entry.url.path])
            } else {
                unmatched.append(entry)
            }
        }
        // Moves: a new path whose inode belongs to a file that disappeared.
        var byInode: [String: FindIndex.FileRecord] = [:]
        for record in missing.values where record.inode != 0 {
            byInode["\(record.device):\(record.inode)"] = record
        }
        for entry in unmatched {
            if let moved = byInode.removeValue(forKey: "\(entry.device):\(entry.inode)"),
               moved.size == entry.size {
                try await index.move(fileID: moved.id, to: entry.url.path, root: root.path)
                missing.removeValue(forKey: moved.path)
            }
            try await index.upsertFile(path: entry.url.path, root: root.path, kind: entry.kind, size: entry.size,
                                       modified: entry.modified, inode: entry.inode, device: entry.device,
                                       lastUsed: recent[entry.url.path])
        }
        try await index.removeFiles(ids: missing.values.map(\.id))
        known.removeAll()
    }

    /// Applies changed paths from the folder watcher.
    public func update(paths: [String]) async throws {
        let configuration = self.configuration
        for path in Set(paths) {
            let url = URL(filePath: FindPaths.normalize(path)).standardizedFileURL
            guard let root = configuration.roots.first(where: {
                url.path == $0.path || url.path.hasPrefix($0.path.hasSuffix("/") ? $0.path : $0.path + "/")
            }) else { continue }
            if let entry = FindCrawler.entry(for: url, rules: configuration.rules),
               !FindCrawler.isInsideSkippedFolder(url, root: root, rules: configuration.rules) {
                try await index.upsertFile(path: entry.url.path, root: root.path, kind: entry.kind, size: entry.size,
                                           modified: entry.modified, inode: entry.inode, device: entry.device,
                                           lastUsed: .now)
            } else if !FileManager.default.fileExists(atPath: url.path),
                      let record = try await index.file(path: url.path) {
                try await index.removeFiles(ids: [record.id])
            } else if FileManager.default.fileExists(atPath: url.path) {
                // A folder: renamed or moved in. Sync it as a whole.
                var isDirectory: ObjCBool = false
                if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue,
                   !FindCrawler.isInsideSkippedFolder(url.appending(path: "x"), root: root, rules: configuration.rules) {
                    let entries = FindCrawler.crawl(root: url, rules: configuration.rules)
                    for entry in entries {
                        try await index.upsertFile(path: entry.url.path, root: root.path, kind: entry.kind,
                                                   size: entry.size, modified: entry.modified, inode: entry.inode,
                                                   device: entry.device, lastUsed: nil)
                    }
                }
            }
        }
        await publish()
        kick()
    }

    // MARK: Work

    /// Runs until there is nothing left to read or learn (or a gate stops it).
    ///
    /// Reading and learning run side by side: reading is CPU and Vision
    /// work, learning is GPU work, and a file read a moment ago can be
    /// learned while the next ones are read, so meaning matches arrive as
    /// the index grows rather than at the end.
    func drain() async {
        let priority: TaskPriority = hurry ? .userInitiated : .utility
        readingNow = true
        await withTaskGroup(of: Void.self) { group in
            group.addTask(priority: priority) { await self.readLoop() }
            group.addTask(priority: priority) { await self.learnLoop() }
        }
        if !Task.isCancelled { setPhase(.upToDate) }
        await extractor.relax()
        await publish()
    }

    /// Whether the read loop of the running drain is still going.
    private var readingNow = false

    private func readLoop() async {
        defer { readingNow = false }
        while !Task.isCancelled {
            do {
                if try await readStep() { continue }
            } catch {
                // An index error stops this round; the next kick retries.
            }
            return
        }
    }

    private func learnLoop() async {
        while !Task.isCancelled {
            do {
                if try await learnStep() { continue }
            } catch {
                return
            }
            // Nothing to learn yet: done, unless reading may still add some.
            if !readingNow {
                if (try? await learnStep()) == true { continue }
                return
            }
            try? await Task.sleep(for: .milliseconds(250))
        }
    }

    /// Reads one batch of unread files, `parallelism` at a time. False when
    /// there was none to read.
    func readStep() async throws -> Bool {
        let files = try await index.unreadFiles(limit: max(readBatch, extractor.parallelism * 8))
        guard !files.isEmpty else { return false }
        if let reason = await gate.reading() {
            setPhase(.paused, reason: reason)
            await publish()
            try await Task.sleep(for: gatePoll)
            return true
        }
        setPhase(.reading)
        // Photos are decoded for the encoders while they are open, when
        // learning will follow soon (the gate is open) and not too many wait.
        let prepareAhead: Bool
        if configuration.rules.readMedia, embedder?.supportsMedia == true, await gate.learning() == nil {
            prepareAhead = (try await index.preparedBacklog()) < Self.preparedLimit
        } else {
            prepareAhead = false
        }
        let options = FindExtractOptions(ocr: configuration.rules.readImages, media: configuration.rules.readMedia,
                                         prepareMedia: prepareAhead)
        let extractor = self.extractor
        let width = max(1, extractor.parallelism)
        try await withThrowingTaskGroup(of: (FindIndex.FileRecord, FindExtraction).self) { group in
            var next = files.makeIterator()
            for _ in 0..<width {
                guard let file = next.next() else { break }
                group.addTask { (file, await extractor.extract(file.url, kind: file.kind, options: options)) }
            }
            while let (file, extraction) = try await group.next() {
                switch extraction {
                case .text:
                    try await index.storeText(fileID: file.id, name: file.url.lastPathComponent,
                                              chunks: extraction.chunks, contentHash: extraction.contentHash)
                case .skipped(let reason):
                    try await index.markSkipped(fileID: file.id, reason: reason)
                }
                if !Task.isCancelled, let file = next.next() {
                    group.addTask { (file, await extractor.extract(file.url, kind: file.kind, options: options)) }
                }
            }
        }
        await publish()
        return !Task.isCancelled
    }

    /// Pictures prepared ahead that may wait at once (about 0.8 MB each).
    static let preparedLimit = 400
    /// A backlog this large (files to read or learn) makes Find hurry.
    static let hurryBacklog = 200
    private var hurry = false

    /// Sends files of these kinds back to be read again.
    public func requeue(kinds: [FindKind]) async throws {
        try await index.requeue(kinds: kinds)
        await publish()
        kick()
    }

    /// Embeds one batch of passages. False when there is none, no model, or
    /// the gate says wait.
    func learnStep() async throws -> Bool {
        guard let embedder else { return false }
        let pending = try await index.pendingChunks(limit: learnBatch, includeMedia: embedder.supportsMedia)
        guard !pending.isEmpty else { return false }
        if let reason = await gate.learning() {
            setPhase(.paused, reason: reason)
            await publish()
            // Wait for a kick or the next poll, without spinning.
            try await Task.sleep(for: gatePoll)
            return true
        }
        setPhase(.learning)
        let text = pending.filter { $0.media == nil }
        if !text.isEmpty {
            let inputs = text.map { FindEmbeddingInput(title: $0.title, text: $0.text) }
            let vectors = try await embedder.embed(documents: inputs)
            guard vectors.count == text.count else { return false }
            try await index.storeVectors(zip(text, vectors).map {
                (chunkID: $0.chunkID, fileID: $0.fileID, vector: FindVectors.prepare($1))
            })
        }
        // Media: pictures prepared while reading are loaded as they are; the
        // rest are decoded by the readers (several files at once). Then all
        // of them go to the encoders in one batched call.
        var loaded: [(FindIndex.PendingChunk, FindMediaData)] = []
        var failed: [Int64] = []
        var usedPrepared: [Int64] = []
        var toPrepare: [Int64: [FindIndex.PendingChunk]] = [:]
        var order: [Int64] = []
        for chunk in pending where chunk.media != nil {
            if let path = chunk.media?.prepared {
                usedPrepared.append(chunk.chunkID)
                defer { try? FileManager.default.removeItem(atPath: path) }
                if let data = try? FindMediaData(contentsOf: URL(filePath: path)) {
                    loaded.append((chunk, data))
                    continue
                }
            }
            if toPrepare[chunk.fileID] == nil { order.append(chunk.fileID) }
            toPrepare[chunk.fileID, default: []].append(chunk)
        }
        try await index.clearPrepared(chunkIDs: usedPrepared)
        let extractor = self.extractor
        // Photos-library pictures are prepared in this process, beside the
        // readers, so more can be in flight than there are readers.
        let width = max(1, extractor.parallelism) * 2
        await withTaskGroup(of: [(FindIndex.PendingChunk, URL?)].self) { group in
            var next = order.makeIterator()
            func add() {
                guard let fileID = next.next(), let chunks = toPrepare[fileID], let first = chunks.first else { return }
                group.addTask {
                    let files = await extractor.prepare(URL(filePath: first.path), kind: first.kind,
                                                        pieces: chunks.compactMap(\.media))
                    return chunks.enumerated().map { ($0.element, $0.offset < files.count ? files[$0.offset] : nil) }
                }
            }
            for _ in 0..<width { add() }
            while let results = await group.next() {
                for (chunk, file) in results {
                    if let file, let data = try? FindMediaData(contentsOf: file) {
                        loaded.append((chunk, data))
                    } else {
                        failed.append(chunk.chunkID)
                    }
                    if let file { try? FileManager.default.removeItem(at: file) }
                }
                if !Task.isCancelled { add() }
            }
        }
        if !loaded.isEmpty {
            let vectors = try await embedder.embed(media: loaded.map(\.1))
            var stored: [(chunkID: Int64, fileID: Int64, vector: [Float])] = []
            for (index, item) in loaded.enumerated() {
                if index < vectors.count, let vector = vectors[index], !vector.isEmpty {
                    stored.append((item.0.chunkID, item.0.fileID, FindVectors.prepare(vector)))
                } else {
                    failed.append(item.0.chunkID)
                }
            }
            try await index.storeVectors(stored)
        }
        try await index.dropPending(chunkIDs: failed)
        return true
    }

    // MARK: Status

    private func setPhase(_ phase: FindIndexStatus.Phase, reason: String? = nil) {
        self.phase = phase
        pauseReason = reason
    }

    /// The status now, and tells `onStatus`.
    @discardableResult
    public func publish() async -> FindIndexStatus {
        let counts = (try? await index.counts()) ?? (total: 0, read: 0, learned: 0, skipped: 0, vectors: 0)
        let pending = counts.total - counts.read - counts.skipped
        // A big backlog (the first index, a large import) runs at
        // user-initiated priority; the trickle afterwards in the background.
        let shouldHurry = pending + (counts.read - counts.learned) > Self.hurryBacklog
        if shouldHurry != hurry {
            hurry = shouldHurry
            await extractor.setHurry(shouldHurry)
            await embedder?.setHurry(shouldHurry)
        }
        let phase: FindIndexStatus.Phase
        if self.phase == .paused || self.phase == .scanning {
            phase = self.phase
        } else if pending > 0 {
            phase = .reading
        } else if counts.read > counts.learned, embedder != nil {
            phase = .learning
        } else {
            phase = .upToDate
        }
        let status = FindIndexStatus(phase: phase, totalFiles: counts.total, readFiles: counts.read,
                                     learnedFiles: counts.learned, skippedFiles: counts.skipped,
                                     pauseReason: phase == .paused ? pauseReason : nil,
                                     indexBytes: await index.sizeOnDisk())
        onStatus(status)
        return status
    }

    /// Waits until reading and learning have nothing left (tests).
    public func waitUntilIdle() async {
        await drain()
    }
}

/// One spelling per path. FSEvents reports real paths (`/private/var/…`)
/// while a folder reached through the `/var`, `/tmp` or `/etc` links is
/// crawled under the short form; both become the short form.
public enum FindPaths {
    public static func normalize(_ path: String) -> String {
        for prefix in ["/private/var/", "/private/tmp/", "/private/etc/"] where path.hasPrefix(prefix) {
            return String(path.dropFirst("/private".count))
        }
        return path
    }
}
