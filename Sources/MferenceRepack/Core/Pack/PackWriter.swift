import CryptoKit
import Darwin
import Foundation

/// Options for `MFPackWriter.write`.
public struct MFPackWriteOptions: Sendable {
    /// Product label, e.g. `"quick"`; recorded in the manifest and receipt.
    public var label: String
    /// Continue a partial write (`<output>.partial` + `<output>.resume.json`)
    /// instead of refusing it. Tensors the checkpoint lists are re-hashed and
    /// kept; everything else is written.
    public var resume: Bool
    /// The tile layout for a tensor of a type with `rows` x `blocksPerRow`
    /// blocks. The default is `MFPack.LayoutTable`; a layout that does not fit
    /// falls back to `.gguf`. The manifest records `LayoutTable.version` only
    /// when every tensor follows the table, 0 otherwise.
    public var layout: @Sendable (MFPack.GGMLType, Int, Int) -> MFPack.TileLayout
    /// Source bytes read per `pread`.
    public var readChunkBytes: Int
    /// Upper bound of each scratch buffer (read chunk, row-group buffers).
    public var scratchLimitBytes: Int
    /// fsync `weights.bin` and advance the resume checkpoint after at least
    /// this many newly written bytes.
    public var checkpointEveryBytes: UInt64
    /// Progress lines.
    public var log: (@Sendable (String) -> Void)?
    /// Test hook: stop with `MFPackError.outputState` once this many tensors
    /// were written and checkpointed in this run (simulates an interruption).
    public var stopAfterTensors: Int?

    public init(label: String, resume: Bool = false,
                layout: @escaping @Sendable (MFPack.GGMLType, Int, Int) -> MFPack.TileLayout = {
                    MFPack.LayoutTable.layout(for: $0, rows: $1, blocksPerRow: $2)
                },
                readChunkBytes: Int = 8 << 20, scratchLimitBytes: Int = 64 << 20,
                checkpointEveryBytes: UInt64 = 256 << 20, log: (@Sendable (String) -> Void)? = nil,
                stopAfterTensors: Int? = nil) {
        self.label = label
        self.resume = resume
        self.layout = layout
        self.readChunkBytes = readChunkBytes
        self.scratchLimitBytes = scratchLimitBytes
        self.checkpointEveryBytes = checkpointEveryBytes
        self.log = log
        self.stopAfterTensors = stopAfterTensors
    }
}

/// What a write did.
public struct MFPackWriteResult: Sendable {
    public var directory: String
    public var manifest: MFPack.Manifest
    public var receipt: MFPack.Receipt
    /// Tensors kept from a previous partial write.
    public var resumedTensors: Int
    /// Tensor bytes written in this run.
    public var writtenBytes: UInt64
    /// Largest scratch buffer used (bounded by `scratchLimitBytes`).
    public var scratchHighWaterBytes: Int
}

/// The production `.mfpack` v1 writer (D3, `docs/MFERENCE_PACK.md`): a GGUF's
/// blocks rearranged into each tensor's tile layout with no requantization,
/// hash-checked against the source.
///
/// - Install lock `<output>.install.lock`; the pack is built in
///   `<output>.partial` and renamed into place only after verification.
/// - Bounded memory: the source is read with `pread` in `readChunkBytes`
///   chunks; a tiled tensor goes through two row-group buffers. No tensor is
///   ever held whole in memory.
/// - One sequential pass hashes the whole source file (`source.sha256`) and
///   every tensor's source and packed bytes (SHA-256).
/// - Resume: `<output>.resume.json` lists tensors whose bytes were fsynced;
///   it is tied to the source file's identity and the write plan.
/// - Verification re-reads `weights.bin`: every packed hash, every tensor
///   unpacked back to GGUF order against its source hash, zero padding.
/// - `verified-install.json` is written last.
///
/// For the same GGUF, label and layouts, `weights.bin`, `manifest.json`,
/// `tokenizer.json` and `chat_template.jinja` are byte-identical to the
/// reference writer's (`Native.ReferencePackWriter`).
public enum MFPackWriter {
    public static let writerName = "MferenceRepack"

    struct Planned {
        var info: MFPack.GGUFHeader.TensorInfo
        var format: MFPack.QuantBlockFormat
        var rows: Int
        var blocksPerRow: Int
        var layout: MFPack.TileLayout
        /// Offset in `weights.bin`.
        var packOffset: UInt64
        /// Absolute offset in the GGUF.
        var sourceOffset: Int
        var byteCount: Int { info.byteCount }
        var permuter: MFPack.TilePermuter {
            MFPack.TilePermuter(layout: layout, format: format, rows: rows, blocksPerRow: blocksPerRow)
        }
    }

    struct PlanRecord: Codable {
        var name: String
        var type: UInt32
        var shape: [Int]
        var layout: MFPack.TileLayout
        var packOffset: UInt64
        var sourceOffset: Int
        var byteCount: Int
    }

    // MARK: Write

    @discardableResult
    public static func write(gguf: String, output: String, options: MFPackWriteOptions) throws -> MFPackWriteResult {
        let lock = try InstallLock.acquire(outputDirectory: output)
        // Hold the flock until the write returns (ARC may otherwise release it early).
        defer { withExtendedLifetime(lock) {} }
        let paths = lock.paths
        guard try Posix.entryKind(paths.finalDirectory) == .absent else {
            throw MFPackError.outputState("\(paths.finalDirectory) exists; refusing to overwrite it")
        }
        let log = options.log ?? { _ in }
        log("reading GGUF header of \(gguf)")
        let header = try MFPack.GGUFHeader(path: gguf)
        let config = try header.qwen35Config()
        let tokenizer = try header.tokenizerData()
        let template = header.string("tokenizer.chat_template") ?? ""
        let architecture = header.string("general.architecture") ?? ""

        // Plan: tensors in GGUF tensor-info order, each at a 4096-aligned offset.
        var plan: [Planned] = []
        var cursor: UInt64 = 0
        var followsTable = true
        for info in header.tensors {
            guard let format = MFPack.QuantBlockFormat.of(info.type) else {
                throw MFPackError.unsupported("\(info.name): \(info.type)")
            }
            let rows = info.shape.dropFirst().reduce(1, *)
            let blocksPerRow = info.shape[0] / format.elements
            var layout = options.layout(info.type, rows, blocksPerRow)
            if !layout.fits(rows: rows, blocksPerRow: blocksPerRow) { layout = .gguf }
            if layout != MFPack.LayoutTable.layout(for: info.type, rows: rows, blocksPerRow: blocksPerRow) {
                followsTable = false
            }
            cursor = (cursor + MFPack.pageAlignment - 1) / MFPack.pageAlignment * MFPack.pageAlignment
            plan.append(Planned(info: info, format: format, rows: rows, blocksPerRow: blocksPerRow, layout: layout,
                                packOffset: cursor, sourceOffset: header.dataOffset + Int(info.offset)))
            cursor += UInt64(info.byteCount)
        }
        let weightsByteCount = cursor
        let planData = try MFPack.encoder().encode(plan.map {
            PlanRecord(name: $0.info.name, type: $0.info.type.rawValue, shape: $0.info.shape, layout: $0.layout,
                       packOffset: $0.packOffset, sourceOffset: $0.sourceOffset, byteCount: $0.byteCount)
        })

        // Source identity for the resume checkpoint.
        var st = stat()
        guard stat(gguf, &st) == 0 else { throw MFPackError.io("stat \(gguf): errno \(errno)") }
        guard Int(st.st_size) == header.fileSize else { throw MFPackError.integrity("\(gguf) changed while reading") }
        var fresh = MFPack.ResumeCheckpoint(
            label: options.label, sourceByteCount: UInt64(st.st_size),
            sourceModifiedNanoseconds: Int64(st.st_mtimespec.tv_sec) * 1_000_000_000 + Int64(st.st_mtimespec.tv_nsec),
            sourceInode: UInt64(st.st_ino), planSHA256: sha256Hex(planData),
            weightsByteCount: weightsByteCount, completed: [:])

        let partial = paths.partialDirectory
        let partialKind = try Posix.entryKind(partial)
        if partialKind != .absent && !options.resume {
            throw MFPackError.outputState("a partial pack exists at \(partial); resume it or discard it")
        }
        var checkpoint = fresh
        if partialKind == .directory, options.resume,
           try Posix.entryKind(paths.checkpointFile) == .regular {
            let saved: MFPack.ResumeCheckpoint
            do {
                saved = try JSONDecoder().decode(MFPack.ResumeCheckpoint.self,
                                                 from: Data(contentsOf: URL(fileURLWithPath: paths.checkpointFile)))
            } catch {
                throw MFPackError.integrity("\(paths.checkpointFile) is unreadable (\(error)); discard the partial pack")
            }
            guard saved.matches(fresh) else {
                throw MFPackError.integrity("the partial pack was made from a different source, label or plan; "
                                            + "discard it")
            }
            checkpoint = saved
        } else {
            if partialKind != .absent { try FileManager.default.removeItem(atPath: partial) }
            try Posix.mkdirP(partial)
            try? FileManager.default.removeItem(atPath: paths.checkpointFile)
            fresh.completed = [:]
            checkpoint = fresh
            try save(checkpoint, to: paths.checkpointFile)
        }

        let weightsPath = (partial as NSString).appendingPathComponent("weights.bin")
        let out = open(weightsPath, O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0o644)
        guard out >= 0 else { throw MFPackError.io("open \(weightsPath): errno \(errno)") }
        defer { close(out) }
        _ = fcntl(out, F_NOCACHE, 1)
        try Posix.ftruncate(out, path: weightsPath, size: weightsByteCount)

        let readScratch = BoundedScratch(limitBytes: options.scratchLimitBytes)
        let groupIn = BoundedScratch(limitBytes: options.scratchLimitBytes)
        let groupOut = BoundedScratch(limitBytes: options.scratchLimitBytes)
        let chunk = max(1, min(options.readChunkBytes, options.scratchLimitBytes))

        // Keep only checkpointed tensors whose packed bytes still hash right.
        var resumed = 0
        if !checkpoint.completed.isEmpty {
            for t in plan {
                guard let done = checkpoint.completed[t.info.name] else { continue }
                let packed = try hashFileRange(fd: out, path: weightsPath, offset: t.packOffset,
                                               count: t.byteCount, scratch: readScratch, chunk: chunk)
                if packed == done.packed {
                    resumed += 1
                } else {
                    log("resume: \(t.info.name) packed bytes do not match the checkpoint; rewriting it")
                    checkpoint.completed[t.info.name] = nil
                }
            }
            log("resume: keeping \(resumed) of \(plan.count) tensors")
        }
        let remaining = plan.filter { checkpoint.completed[$0.info.name] == nil }.reduce(UInt64(0)) {
            $0 + UInt64($1.byteCount)
        }
        _ = try DiskSpaceChecker.requireAvailable(path: partial, bytes: remaining)

        // One sequential pass over the source.
        let src = open(gguf, O_RDONLY | O_CLOEXEC)
        guard src >= 0 else { throw MFPackError.io("open \(gguf): errno \(errno)") }
        defer { close(src) }
        _ = fcntl(src, F_NOCACHE, 1)
        var fileHasher = SHA256()
        var position = 0
        func hashSource(upTo end: Int) throws {
            while position < end {
                let n = min(chunk, end - position)
                let buf = try readScratch.ensure(n)
                try Posix.preadAll(fd: src, path: gguf, buf: buf.baseAddress!, count: n, offset: UInt64(position))
                fileHasher.update(bufferPointer: UnsafeRawBufferPointer(start: buf.baseAddress, count: n))
                position += n
            }
        }
        var pending: [String: MFPack.ResumeCheckpoint.Hashes] = [:]
        var bytesSinceSync: UInt64 = 0
        var writtenBytes: UInt64 = 0
        var writtenThisRun = 0
        var hashes: [String: MFPack.ResumeCheckpoint.Hashes] = checkpoint.completed
        func sync() throws {
            try Posix.fsync(out, path: weightsPath)
            checkpoint.completed.merge(pending) { _, new in new }
            pending.removeAll()
            try save(checkpoint, to: paths.checkpointFile)
            bytesSinceSync = 0
        }
        let started = Date()
        var lastLog = Date.distantPast
        for t in plan.sorted(by: { $0.sourceOffset < $1.sourceOffset }) {
            try hashSource(upTo: t.sourceOffset)
            let done = checkpoint.completed[t.info.name]
            var sourceHasher = SHA256()
            var packedHasher = SHA256()
            let permuter = t.permuter
            let groupBytes = permuter.groupBytes
            if done == nil && !permuter.isIdentity {
                guard groupBytes <= options.scratchLimitBytes else {
                    throw MFPackError.scratch("\(t.info.name): a \(t.layout) row group is \(groupBytes) bytes, "
                                              + "over the \(options.scratchLimitBytes)-byte bound")
                }
                _ = try groupIn.ensure(groupBytes)
                _ = try groupOut.ensure(groupBytes)
            }
            var fill = 0
            var group = 0
            var emitted: UInt64 = 0
            func emit(_ bytes: UnsafeRawBufferPointer) throws {
                packedHasher.update(bufferPointer: bytes)
                try Posix.pwriteAll(fd: out, path: weightsPath, buf: bytes.baseAddress!, count: bytes.count,
                                    offset: t.packOffset + emitted)
                emitted += UInt64(bytes.count)
            }
            let end = t.sourceOffset + t.byteCount
            while position < end {
                let n = min(chunk, end - position)
                let buf = try readScratch.ensure(n)
                try Posix.preadAll(fd: src, path: gguf, buf: buf.baseAddress!, count: n, offset: UInt64(position))
                let bytes = UnsafeRawBufferPointer(start: buf.baseAddress, count: n)
                fileHasher.update(bufferPointer: bytes)
                sourceHasher.update(bufferPointer: bytes)
                position += n
                guard done == nil else { continue }
                if permuter.isIdentity {
                    try emit(bytes)
                    continue
                }
                var consumed = 0
                while consumed < n {
                    let take = min(groupBytes - fill, n - consumed)
                    let gin = try groupIn.ensure(groupBytes)
                    (gin.baseAddress! + fill).copyMemory(from: bytes.baseAddress! + consumed, byteCount: take)
                    fill += take
                    consumed += take
                    if fill == groupBytes {
                        let gout = try groupOut.ensure(groupBytes)
                        permuter.pack(group: group, src: UnsafeRawPointer(gin.baseAddress!), dst: gout.baseAddress!)
                        try emit(UnsafeRawBufferPointer(start: gout.baseAddress, count: groupBytes))
                        fill = 0
                        group += 1
                    }
                }
            }
            let sourceHex = hex(sourceHasher.finalize())
            if let done {
                guard sourceHex == done.source else {
                    throw MFPackError.integrity("\(t.info.name): the source changed since the partial write; "
                                                + "discard the partial pack")
                }
                continue
            }
            guard fill == 0, emitted == UInt64(t.byteCount) else {
                throw MFPackError.integrity("\(t.info.name): wrote \(emitted) of \(t.byteCount) bytes")
            }
            let h = MFPack.ResumeCheckpoint.Hashes(source: sourceHex, packed: hex(packedHasher.finalize()))
            pending[t.info.name] = h
            hashes[t.info.name] = h
            bytesSinceSync += UInt64(t.byteCount)
            writtenBytes += UInt64(t.byteCount)
            writtenThisRun += 1
            if bytesSinceSync >= options.checkpointEveryBytes { try sync() }
            if let stop = options.stopAfterTensors, writtenThisRun >= stop {
                try sync()
                throw MFPackError.outputState("stopped after \(writtenThisRun) tensors (test hook)")
            }
            if Date().timeIntervalSince(lastLog) > 10 {
                lastLog = Date()
                log("wrote \(writtenThisRun) tensors, \(writtenBytes >> 20) MiB in "
                    + "\(Int(Date().timeIntervalSince(started))) s")
            }
        }
        try hashSource(upTo: header.fileSize)
        try sync()
        let sourceSHA = hex(fileHasher.finalize())
        log("source SHA-256 \(sourceSHA); wrote \(writtenBytes >> 20) MiB, kept \(resumed) tensors")

        // Metadata files.
        let entries = try plan.map { t -> MFPack.Manifest.Tensor in
            guard let h = hashes[t.info.name] else { throw MFPackError.integrity("\(t.info.name): no hashes") }
            return MFPack.Manifest.Tensor(name: t.info.name, type: t.info.type, shape: t.info.shape, layout: t.layout,
                                          offset: t.packOffset, byteCount: UInt64(t.byteCount),
                                          sourceSHA256: h.source, packedSHA256: h.packed)
        }
        let manifest = MFPack.Manifest(
            architecture: architecture, label: options.label,
            source: .init(fileName: (gguf as NSString).lastPathComponent, byteCount: UInt64(header.fileSize),
                          sha256: sourceSHA, ggufVersion: header.version),
            config: config, layoutTableVersion: followsTable ? MFPack.LayoutTable.version : 0,
            tensors: entries, metadata: header.scalarMetadataStrings())
        let encoder = MFPack.encoder()
        let manifestData = try encoder.encode(manifest)
        let tokenizerData = try encoder.encode(tokenizer)
        let templateData = Data(template.utf8)
        try writeFile(manifestData, to: (partial as NSString).appendingPathComponent("manifest.json"))
        try writeFile(tokenizerData, to: (partial as NSString).appendingPathComponent("tokenizer.json"))
        try writeFile(templateData, to: (partial as NSString).appendingPathComponent("chat_template.jinja"))

        log("verifying \(plan.count) tensors")
        try verifyWeights(fd: out, path: weightsPath, plan: plan, entries: entries, size: weightsByteCount,
                          scratch: readScratch, groupIn: groupIn, groupOut: groupOut, chunk: chunk)

        let iso = ISO8601DateFormatter()
        let receipt = MFPack.Receipt(
            format: manifest.format, formatVersion: manifest.formatVersion, label: options.label,
            writer: writerName, sourceSHA256: sourceSHA, manifestSHA256: sha256Hex(manifestData),
            tensorCount: entries.count, weightsByteCount: weightsByteCount, verifiedAt: iso.string(from: Date()),
            tokenizerSHA256: sha256Hex(tokenizerData), chatTemplateSHA256: sha256Hex(templateData))
        try writeFile(try encoder.encode(receipt),
                      to: (partial as NSString).appendingPathComponent("verified-install.json"))
        try Posix.fsyncDirectory(partial)
        try Posix.rename(from: partial, to: paths.finalDirectory)
        try Posix.fsyncDirectory(paths.parentDirectory)
        try? FileManager.default.removeItem(atPath: paths.checkpointFile)
        log("installed \(paths.finalDirectory)")
        let high = max(readScratch.largestEverBytes, groupIn.largestEverBytes, groupOut.largestEverBytes)
        return MFPackWriteResult(directory: paths.finalDirectory, manifest: manifest, receipt: receipt,
                                 resumedTensors: resumed, writtenBytes: writtenBytes, scratchHighWaterBytes: high)
    }

    /// Remove `<output>.partial` and `<output>.resume.json` (never the
    /// installed pack).
    public static func discardPartial(output: String) throws {
        let lock = try InstallLock.acquire(outputDirectory: output)
        defer { withExtendedLifetime(lock) {} }
        if try Posix.entryKind(lock.paths.partialDirectory) != .absent {
            try FileManager.default.removeItem(atPath: lock.paths.partialDirectory)
        }
        if try Posix.entryKind(lock.paths.checkpointFile) != .absent {
            try FileManager.default.removeItem(atPath: lock.paths.checkpointFile)
        }
    }

    // MARK: Verification

    /// Re-read `weights.bin`: size, zero padding, every packed hash, and every
    /// tensor unpacked back to GGUF order against its source hash.
    static func verifyWeights(fd: Int32, path: String, plan: [Planned], entries: [MFPack.Manifest.Tensor],
                              size: UInt64, scratch: BoundedScratch, groupIn: BoundedScratch,
                              groupOut: BoundedScratch, chunk: Int) throws {
        var st = stat()
        guard fstat(fd, &st) == 0, UInt64(st.st_size) == size else {
            throw MFPackError.integrity("\(path) is not \(size) bytes")
        }
        var previousEnd: UInt64 = 0
        for (t, e) in zip(plan, entries) {
            // Padding before the tensor is zero.
            if t.packOffset > previousEnd {
                let n = Int(t.packOffset - previousEnd)
                let buf = try scratch.ensure(n)
                try Posix.preadAll(fd: fd, path: path, buf: buf.baseAddress!, count: n, offset: previousEnd)
                guard UnsafeRawBufferPointer(start: buf.baseAddress, count: n).allSatisfy({ $0 == 0 }) else {
                    throw MFPackError.integrity("non-zero padding before \(t.info.name)")
                }
            }
            previousEnd = t.packOffset + UInt64(t.byteCount)
            let permuter = t.permuter
            if permuter.isIdentity {
                let packed = try hashFileRange(fd: fd, path: path, offset: t.packOffset, count: t.byteCount,
                                               scratch: scratch, chunk: chunk)
                guard packed == e.packedSHA256, packed == e.sourceSHA256 else {
                    throw MFPackError.integrity("\(t.info.name): packed SHA-256 mismatch")
                }
                continue
            }
            var packedHasher = SHA256()
            var sourceHasher = SHA256()
            let g = permuter.groupBytes
            let groupsPerRead = max(1, chunk / g)
            let groups = t.rows / t.layout.rowsPerTile
            var group = 0
            while group < groups {
                let count = min(groupsPerRead, groups - group)
                let buf = try scratch.ensure(count * g)
                try Posix.preadAll(fd: fd, path: path, buf: buf.baseAddress!, count: count * g,
                                   offset: t.packOffset + UInt64(group * g))
                packedHasher.update(bufferPointer: UnsafeRawBufferPointer(start: buf.baseAddress, count: count * g))
                let out = try groupOut.ensure(g)
                for k in 0..<count {
                    permuter.unpack(group: group + k, src: UnsafeRawPointer(buf.baseAddress! + k * g),
                                    dst: out.baseAddress!)
                    sourceHasher.update(bufferPointer: UnsafeRawBufferPointer(start: out.baseAddress, count: g))
                }
                group += count
            }
            guard hex(packedHasher.finalize()) == e.packedSHA256 else {
                throw MFPackError.integrity("\(t.info.name): packed SHA-256 mismatch")
            }
            guard hex(sourceHasher.finalize()) == e.sourceSHA256 else {
                throw MFPackError.integrity("\(t.info.name): unpacked bytes do not match the source")
            }
        }
        guard previousEnd == size else { throw MFPackError.integrity("\(path) has trailing bytes") }
    }

    static func hashFileRange(fd: Int32, path: String, offset: UInt64, count: Int, scratch: BoundedScratch,
                              chunk: Int) throws -> String {
        var hasher = SHA256()
        var done = 0
        while done < count {
            let n = min(chunk, count - done)
            let buf = try scratch.ensure(n)
            try Posix.preadAll(fd: fd, path: path, buf: buf.baseAddress!, count: n, offset: offset + UInt64(done))
            hasher.update(bufferPointer: UnsafeRawBufferPointer(start: buf.baseAddress, count: n))
            done += n
        }
        return hex(hasher.finalize())
    }

    // MARK: Files and hashes

    static func save(_ checkpoint: MFPack.ResumeCheckpoint, to path: String) throws {
        let tmp = path + ".tmp"
        try writeFile(try MFPack.encoder().encode(checkpoint), to: tmp)
        try Posix.rename(from: tmp, to: path)
        try Posix.fsyncDirectory((path as NSString).deletingLastPathComponent)
    }

    /// Create or replace `path` with `data` and fsync it.
    static func writeFile(_ data: Data, to path: String) throws {
        let fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW | O_CLOEXEC, 0o644)
        guard fd >= 0 else { throw MFPackError.io("open \(path): errno \(errno)") }
        defer { close(fd) }
        if !data.isEmpty {
            try data.withUnsafeBytes {
                try Posix.pwriteAll(fd: fd, path: path, buf: $0.baseAddress!, count: $0.count, offset: 0)
            }
        }
        try Posix.fsync(fd, path: path)
    }

    public static func sha256Hex(_ data: Data) -> String {
        var hasher = SHA256()
        data.withUnsafeBytes { hasher.update(bufferPointer: $0) }
        return hex(hasher.finalize())
    }

    static func hex(_ digest: SHA256.Digest) -> String {
        let digits = Array("0123456789abcdef".utf8)
        var out = ""
        out.reserveCapacity(64)
        for byte in digest {
            out.unicodeScalars.append(Unicode.Scalar(digits[Int(byte >> 4)]))
            out.unicodeScalars.append(Unicode.Scalar(digits[Int(byte & 0x0F)]))
        }
        return out
    }
}
