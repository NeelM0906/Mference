import Darwin
import Foundation

extension Native {
    /// Where and how much a `PagedKVStore` may spill cached prefixes to disk.
    public struct PrefixSSDTierConfiguration: Sendable {
        /// Parent directory. The store makes its own subdirectory
        /// `mfkv-<pid>-<uuid>` in it and removes that on deinit.
        public var directory: URL
        /// Most bytes of snapshot files kept at once; the least recently used
        /// files are deleted past it.
        public var budgetBytes: UInt64
        public init(directory: URL, budgetBytes: UInt64) {
            self.directory = directory
            self.budgetBytes = budgetBytes
        }
    }

    /// The SSD tier of the prefix cache: one file per spilled prefix.
    ///
    /// File format `mfkv-snapshot` v1 (all integers little-endian):
    /// - header: magic `MFKVSNP1`, then u32 `version`, u32 `dataOffset`,
    ///   then the KV geometry (`tokensPerPage`, `attentionLayers`, `kvHeads`,
    ///   `headDim`, precision 0 = F16 / 1 = INT8) and the recurrent layout
    ///   (`convKernel`, `convChannels`, `valueHeads`, `keyDim`, `valueDim`,
    ///   GDN layer count) as u32, u64 `pageBytes`, u64 `length` (tokens),
    ///   u32 `kvPages`, u64 `snapshotBytes`, then `length` i32 tokens, then
    ///   one u64 hash per data chunk (each KV page, then each page-sized
    ///   chunk of the snapshot);
    /// - data at `dataOffset` (16 KiB aligned): the `kvPages` KV pages in
    ///   logical order, `pageBytes` each, laid out per `KVPageGeometry`, then
    ///   the recurrent snapshot (`snapshotBytes`: every GDN layer's state per
    ///   `RecurrentLayout`, in GDN-layer order).
    ///
    /// Data moves between the arena's shared pages and the file with
    /// `pwrite`/`pread` (no heap copy: bounded memory), with `F_NOCACHE` so
    /// spills do not fill the unified buffer cache. Every chunk's hash is
    /// checked on load.
    final class MemorySSDTier {
        static let magic: [UInt8] = Array("MFKVSNP1".utf8)
        static let version: UInt32 = 1
        static let dataAlignment = 16 << 10

        let directory: URL
        let budgetBytes: UInt64
        private(set) var usedBytes: UInt64 = 0

        struct Shape: Equatable {
            var geometry: KVPageGeometry
            var layout: RecurrentLayout
            var gdnLayers: Int
        }

        let shape: Shape

        init(configuration: PrefixSSDTierConfiguration, shape: Shape) throws {
            self.shape = shape
            budgetBytes = configuration.budgetBytes
            let fm = FileManager.default
            try fm.createDirectory(at: configuration.directory, withIntermediateDirectories: true)
            Self.removeStaleDirectories(in: configuration.directory)
            directory = configuration.directory.appendingPathComponent(
                "mfkv-\(getpid())-\(UUID().uuidString)", isDirectory: true)
            try fm.createDirectory(at: directory, withIntermediateDirectories: false)
        }

        deinit {
            try? FileManager.default.removeItem(at: directory)
        }

        /// Removes `mfkv-<pid>-*` directories left by processes that no
        /// longer run (a crash skips deinit). Directories of live processes
        /// are never touched.
        static func removeStaleDirectories(in parent: URL) {
            guard let names = try? FileManager.default.contentsOfDirectory(atPath: parent.path) else { return }
            for name in names where name.hasPrefix("mfkv-") {
                let parts = name.split(separator: "-")
                guard parts.count >= 3, let pid = pid_t(parts[1]), pid != getpid() else { continue }
                if kill(pid, 0) == -1 && errno == ESRCH {
                    try? FileManager.default.removeItem(at: parent.appendingPathComponent(name))
                }
            }
        }

        /// 64-bit integrity hash of `count` bytes (not cryptographic).
        static func hash(_ base: UnsafeRawPointer, count: Int) -> UInt64 {
            var h: UInt64 = 0x243F_6A88_85A3_08D3 ^ UInt64(count)
            let words = count / 8
            if words > 0 {
                let w = UnsafeBufferPointer(start: base.assumingMemoryBound(to: UInt64.self), count: words)
                for x in w {
                    h = (h ^ (x &* 0x9E37_79B9_7F4A_7C15))
                    h = ((h << 27) | (h >> 37)) &* 0xBF58_476D_1CE4_E5B9
                }
            }
            for i in (words * 8)..<count {
                h = (h ^ UInt64(base.load(fromByteOffset: i, as: UInt8.self))) &* 0x100_0000_01B3
            }
            return h ^ (h >> 31)
        }

        /// A chunk of data in the arena: `count` bytes at `base`.
        struct Chunk {
            var base: UnsafeMutableRawPointer
            var count: Int
        }

        struct Header {
            var length: Int
            var kvPages: Int
            var snapshotBytes: Int
            var tokens: [Int32]
            var hashes: [UInt64]
            var dataOffset: Int
        }

        private func headerBytes(tokens: [Int32], kvPages: Int, snapshotBytes: Int,
                                 hashes: [UInt64]) -> [UInt8] {
            var out = Self.magic
            func u32(_ v: Int) { withUnsafeBytes(of: UInt32(v).littleEndian) { out += $0 } }
            func u64(_ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { out += $0 } }
            let g = shape.geometry, l = shape.layout
            let fixed = 8 + 4 * 2 + 4 * 5 + 4 * 6 + 8 + 8 + 4 + 8
            let unaligned = fixed + tokens.count * 4 + hashes.count * 8
            let dataOffset = (unaligned + Self.dataAlignment - 1) / Self.dataAlignment * Self.dataAlignment
            u32(Int(Self.version))
            u32(dataOffset)
            for v in [g.tokensPerPage, g.attentionLayers, g.kvHeads, g.headDim, g.precision == .int8 ? 1 : 0] { u32(v) }
            for v in [l.convKernel, l.convChannels, l.valueHeads, l.keyDim, l.valueDim, shape.gdnLayers] { u32(v) }
            u64(UInt64(g.pageBytes))
            u64(UInt64(tokens.count))
            u32(kvPages)
            u64(UInt64(snapshotBytes))
            for t in tokens { withUnsafeBytes(of: t.littleEndian) { out += $0 } }
            for h in hashes { u64(h) }
            out += [UInt8](repeating: 0, count: dataOffset - out.count)
            return out
        }

        enum TierError: Error, CustomStringConvertible {
            case io(String)
            case format(String)
            var description: String {
                switch self {
                case .io(let s): return "SSD tier I/O: \(s)"
                case .format(let s): return "SSD tier file: \(s)"
                }
            }
        }

        private static func writeAll(_ fd: Int32, _ base: UnsafeRawPointer, _ count: Int, at offset: Int) throws {
            var done = 0
            while done < count {
                let n = pwrite(fd, base.advanced(by: done), count - done, off_t(offset + done))
                if n < 0 {
                    if errno == EINTR { continue }
                    throw TierError.io("pwrite: \(String(cString: strerror(errno)))")
                }
                done += n
            }
        }

        private static func readAll(_ fd: Int32, _ base: UnsafeMutableRawPointer, _ count: Int, at offset: Int) throws {
            var done = 0
            while done < count {
                let n = pread(fd, base.advanced(by: done), count - done, off_t(offset + done))
                if n < 0 {
                    if errno == EINTR { continue }
                    throw TierError.io("pread: \(String(cString: strerror(errno)))")
                }
                if n == 0 { throw TierError.format("truncated at byte \(offset + done)") }
                done += n
            }
        }

        /// Bytes a file for this prefix would take.
        func fileBytes(length: Int, kvPages: Int, snapshotBytes: Int, chunks: Int) -> UInt64 {
            UInt64(headerBytes(tokens: [Int32](repeating: 0, count: length), kvPages: kvPages,
                               snapshotBytes: snapshotBytes, hashes: [UInt64](repeating: 0, count: chunks)).count
                   + kvPages * shape.geometry.pageBytes + snapshotBytes)
        }

        /// Writes one prefix: `kv` (one chunk per KV page, `pageBytes` each)
        /// then `snapshot` (page-sized chunks totalling `snapshotBytes`).
        /// Returns the file and its size.
        func write(id: UInt64, tokens: [Int32], kv: [Chunk], snapshot: [Chunk]) throws -> (URL, UInt64) {
            let snapshotBytes = snapshot.reduce(0) { $0 + $1.count }
            let hashes = (kv + snapshot).map { Self.hash($0.base, count: $0.count) }
            let header = headerBytes(tokens: tokens, kvPages: kv.count, snapshotBytes: snapshotBytes, hashes: hashes)
            let url = directory.appendingPathComponent("prefix-\(id).mfkv")
            let temporary = directory.appendingPathComponent("prefix-\(id).mfkv.partial")
            let fd = open(temporary.path, O_WRONLY | O_CREAT | O_TRUNC, 0o600)
            guard fd >= 0 else { throw TierError.io("open \(temporary.path): \(String(cString: strerror(errno)))") }
            var closed = false
            defer {
                if !closed { close(fd) }
            }
            _ = fcntl(fd, F_NOCACHE, 1)
            do {
                try header.withUnsafeBytes { try Self.writeAll(fd, $0.baseAddress!, header.count, at: 0) }
                var offset = header.count
                for c in kv + snapshot {
                    try Self.writeAll(fd, c.base, c.count, at: offset)
                    offset += c.count
                }
                close(fd)
                closed = true
                guard rename(temporary.path, url.path) == 0 else {
                    throw TierError.io("rename: \(String(cString: strerror(errno)))")
                }
                let bytes = UInt64(offset)
                usedBytes += bytes
                return (url, bytes)
            } catch {
                if !closed { close(fd); closed = true }
                unlink(temporary.path)
                throw error
            }
        }

        /// Reads and checks a file's header against this tier's shape and
        /// the expected tokens.
        func readHeader(_ url: URL, expectedTokens: [Int32]) throws -> Header {
            let fd = open(url.path, O_RDONLY)
            guard fd >= 0 else { throw TierError.io("open \(url.lastPathComponent): \(String(cString: strerror(errno)))") }
            defer { close(fd) }
            let fixed = 8 + 4 * 2 + 4 * 5 + 4 * 6 + 8 + 8 + 4 + 8
            var head = [UInt8](repeating: 0, count: fixed)
            try head.withUnsafeMutableBytes { try Self.readAll(fd, $0.baseAddress!, fixed, at: 0) }
            guard Array(head[0..<8]) == Self.magic else { throw TierError.format("bad magic") }
            var cursor = 8
            func u32() -> Int {
                defer { cursor += 4 }
                return Int(UInt32(littleEndian: head.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: cursor, as: UInt32.self) }))
            }
            func u64() -> UInt64 {
                defer { cursor += 8 }
                return UInt64(littleEndian: head.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: cursor, as: UInt64.self) })
            }
            guard u32() == Int(Self.version) else { throw TierError.format("version") }
            let dataOffset = u32()
            let g = shape.geometry, l = shape.layout
            let geometry = [u32(), u32(), u32(), u32(), u32()]
            let layout = [u32(), u32(), u32(), u32(), u32(), u32()]
            guard geometry == [g.tokensPerPage, g.attentionLayers, g.kvHeads, g.headDim, g.precision == .int8 ? 1 : 0],
                  layout == [l.convKernel, l.convChannels, l.valueHeads, l.keyDim, l.valueDim, shape.gdnLayers],
                  u64() == UInt64(g.pageBytes) else {
                throw TierError.format("geometry or layout differs from the store's")
            }
            let length = Int(u64())
            let kvPages = u32()
            let snapshotBytes = Int(u64())
            guard length == expectedTokens.count, kvPages == (length + g.tokensPerPage - 1) / g.tokensPerPage else {
                throw TierError.format("length \(length), \(kvPages) pages")
            }
            let snapshotChunks = (snapshotBytes + g.pageBytes - 1) / g.pageBytes
            let variable = length * 4 + (kvPages + snapshotChunks) * 8
            guard dataOffset >= fixed + variable else { throw TierError.format("data offset") }
            var rest = [UInt8](repeating: 0, count: variable)
            try rest.withUnsafeMutableBytes { try Self.readAll(fd, $0.baseAddress!, variable, at: fixed) }
            let tokens: [Int32] = rest.withUnsafeBytes { raw in
                (0..<length).map { Int32(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 4, as: Int32.self)) }
            }
            guard tokens == expectedTokens else { throw TierError.format("tokens differ from the cached prefix") }
            let hashes: [UInt64] = rest.withUnsafeBytes { raw in
                (0..<(kvPages + snapshotChunks)).map {
                    UInt64(littleEndian: raw.loadUnaligned(fromByteOffset: length * 4 + $0 * 8, as: UInt64.self))
                }
            }
            return Header(length: length, kvPages: kvPages, snapshotBytes: snapshotBytes, tokens: tokens,
                          hashes: hashes, dataOffset: dataOffset)
        }

        /// Reads the data chunks of `header` into `destinations` (one per KV
        /// page, then one per snapshot chunk) and checks every hash.
        func readData(_ url: URL, header: Header, into destinations: [Chunk]) throws {
            guard destinations.count == header.hashes.count else { throw TierError.format("chunk count") }
            let fd = open(url.path, O_RDONLY)
            guard fd >= 0 else { throw TierError.io("open \(url.lastPathComponent): \(String(cString: strerror(errno)))") }
            defer { close(fd) }
            _ = fcntl(fd, F_NOCACHE, 1)
            var offset = header.dataOffset
            for (c, expected) in zip(destinations, header.hashes) {
                try Self.readAll(fd, c.base, c.count, at: offset)
                guard Self.hash(c.base, count: c.count) == expected else {
                    throw TierError.format("hash mismatch at byte \(offset)")
                }
                offset += c.count
            }
        }

        func remove(_ url: URL, bytes: UInt64) {
            unlink(url.path)
            usedBytes -= min(bytes, usedBytes)
        }
    }
}
