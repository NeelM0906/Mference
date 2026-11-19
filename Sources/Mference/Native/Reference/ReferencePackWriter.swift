import CryptoKit
import Foundation

extension Native {
    /// The slow reference `.mfpack` v1 writer (D3): rearranges each GGUF
    /// tensor's blocks into its tile layout with no requantization, hashes
    /// the source and packed bytes, and writes the install directory
    /// (`PackManifest` documents the layout). It exists so the reference path
    /// can load Quick; W1 owns the production writer in `MferenceRepack`.
    ///
    /// Memory is bounded: tensor bytes stream from the mapped GGUF through a
    /// scratch buffer of one tile row group, never a whole tensor.
    public enum ReferencePackWriter {
        /// Contents of `verified-install.json`, written last.
        public struct Receipt: Codable, Equatable, Sendable {
            public var format: String
            public var formatVersion: Int
            public var label: String
            public var writer: String
            public var sourceSHA256: String
            /// SHA-256 of `manifest.json` as written.
            public var manifestSHA256: String
            public var tensorCount: Int
            public var weightsByteCount: UInt64
            public var verifiedAt: String
        }

        public static let pageAlignment: UInt64 = 4096
        static let streamChunk = 32 << 20

        /// Write `gguf` as an `.mfpack` directory at `directory` (which must not
        /// exist). `layout` chooses each tensor's tile layout; a layout that
        /// does not fit the tensor falls back to `.gguf`, as the contract says.
        /// `layoutTableVersion` records `PackLayoutTable.version` when every
        /// tensor is in the table's layout, and 0 when `layout` chose others.
        /// On failure the partially written directory is removed.
        @discardableResult
        public static func write(gguf url: URL, to directory: URL, label: String,
                                 layout: (GGMLType, _ rows: Int, _ blocksPerRow: Int) -> TileLayout = {
                                     PackLayoutTable.layout(for: $0, rows: $1, blocksPerRow: $2)
                                 },
                                 log: ((String) -> Void)? = nil) throws -> PackManifest {
            let fm = FileManager.default
            guard !fm.fileExists(atPath: directory.path) else {
                throw NativeError.integrity("refusing to overwrite \(directory.path)")
            }
            let gguf = try ReferenceGGUFFile(url: url)
            log?("hashing source \(url.lastPathComponent) (\(gguf.fileSize) bytes)")
            let sourceSHA = sha256Hex(gguf.fileBytes)
            let config = try gguf.qwen35Config()
            let tokenizer = try gguf.tokenizerData()
            let template = gguf.string("tokenizer.chat_template") ?? ""

            try fm.createDirectory(at: directory, withIntermediateDirectories: false)
            var completed = false
            defer { if !completed { try? fm.removeItem(at: directory) } }
            var followsTable = true
            let weightsURL = directory.appendingPathComponent("weights.bin")
            guard fm.createFile(atPath: weightsURL.path, contents: nil) else {
                throw NativeError.resource("cannot create \(weightsURL.path)")
            }
            let out = try FileHandle(forWritingTo: weightsURL)
            var written: UInt64 = 0
            var entries: [PackManifest.Tensor] = []
            do {
                for info in gguf.tensors {
                    guard let format = QuantBlockFormat.of(info.type) else {
                        throw NativeError.unsupported("\(info.name): \(info.type)")
                    }
                    let cols = info.shape[0]
                    let rows = info.shape.dropFirst().reduce(1, *)
                    let blocksPerRow = cols / format.elements
                    var chosen = layout(info.type, rows, blocksPerRow)
                    if !chosen.fits(rows: rows, blocksPerRow: blocksPerRow) { chosen = .gguf }
                    if chosen != PackLayoutTable.layout(for: info.type, rows: rows, blocksPerRow: blocksPerRow) {
                        followsTable = false
                    }

                    let aligned = (written + pageAlignment - 1) / pageAlignment * pageAlignment
                    if aligned > written {
                        try out.write(contentsOf: Data(count: Int(aligned - written)))
                        written = aligned
                    }
                    let source = gguf.bytes(of: info)
                    var packedHasher = SHA256()
                    try pack(source: source, format: format, rows: rows, blocksPerRow: blocksPerRow,
                             layout: chosen) { chunk in
                        packedHasher.update(bufferPointer: chunk)
                        try out.write(contentsOf: Data(chunk))
                    }
                    entries.append(PackManifest.Tensor(
                        name: info.name, type: info.type, shape: info.shape, layout: chosen,
                        offset: written, byteCount: UInt64(info.byteCount),
                        sourceSHA256: sha256Hex(source),
                        packedSHA256: hex(packedHasher.finalize())))
                    written += UInt64(info.byteCount)
                }
                try out.synchronize()
                try out.close()
            } catch {
                try? out.close()
                throw error
            }

            let manifest = PackManifest(
                architecture: gguf.string("general.architecture") ?? "",
                label: label,
                source: .init(fileName: url.lastPathComponent, byteCount: UInt64(gguf.fileSize),
                              sha256: sourceSHA, ggufVersion: gguf.version),
                config: config,
                layoutTableVersion: followsTable ? PackLayoutTable.version : 0,
                tensors: entries,
                metadata: gguf.scalarMetadataStrings())
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
            let manifestData = try encoder.encode(manifest)
            try manifestData.write(to: directory.appendingPathComponent("manifest.json"))
            try encoder.encode(tokenizer).write(to: directory.appendingPathComponent("tokenizer.json"))
            try Data(template.utf8).write(to: directory.appendingPathComponent("chat_template.jinja"))

            log?("verifying \(entries.count) tensors")
            try verify(directory: directory, manifest: manifest)

            let receipt = Receipt(format: "mfpack", formatVersion: 1, label: label, writer: "reference",
                                  sourceSHA256: sourceSHA, manifestSHA256: sha256Hex(manifestData),
                                  tensorCount: entries.count, weightsByteCount: written,
                                  verifiedAt: ISO8601DateFormatter().string(from: Date()))
            try encoder.encode(receipt).write(to: directory.appendingPathComponent("verified-install.json"))
            completed = true
            return manifest
        }

        /// Stream the packed bytes of one tensor through `emit`, one tile row
        /// group at a time (a row group's tiles are contiguous in the pack).
        static func pack(source: UnsafeRawBufferPointer, format: QuantBlockFormat, rows: Int, blocksPerRow: Int,
                         layout: TileLayout, emit: (UnsafeRawBufferPointer) throws -> Void) throws {
            if layout == .gguf {
                var start = 0
                while start < source.count {
                    let n = min(streamChunk, source.count - start)
                    try emit(UnsafeRawBufferPointer(rebasing: source[start ..< start + n]))
                    start += n
                }
                return
            }
            let rowBytes = blocksPerRow * format.bytes
            let groupBytes = layout.rowsPerTile * rowBytes
            var scratch = [UInt8](repeating: 0, count: groupBytes)
            try scratch.withUnsafeMutableBytes { dst in
                for group in 0 ..< rows / layout.rowsPerTile {
                    let groupBase = group * groupBytes
                    for rInGroup in 0 ..< layout.rowsPerTile {
                        let row = group * layout.rowsPerTile + rInGroup
                        for block in 0 ..< blocksPerRow {
                            let src = TileLayout.ggufOffset(row: row, block: block, byteInBlock: 0,
                                                            blocksPerRow: blocksPerRow, format: format)
                            var fieldStart = 0
                            let fields = layout.planes ? format.fields : [BlockField(name: "block", bytes: format.bytes)]
                            for field in fields {
                                let to = layout.packedOffset(row: row, block: block, byteInBlock: fieldStart,
                                                             blocksPerRow: blocksPerRow, format: format) - groupBase
                                dst.baseAddress!.advanced(by: to).copyMemory(
                                    from: source.baseAddress!.advanced(by: src + fieldStart), byteCount: field.bytes)
                                fieldStart += field.bytes
                            }
                        }
                    }
                    try emit(UnsafeRawBufferPointer(dst))
                }
            }
        }

        /// Re-read `weights.bin` and check every tensor's packed hash and the
        /// unpacked (GGUF-order) hash against the manifest.
        public static func verify(directory: URL, manifest: PackManifest) throws {
            let mapping = try ReferenceMappedFile(url: directory.appendingPathComponent("weights.bin"))
            for t in manifest.tensors {
                guard t.offset % pageAlignment == 0, t.offset + t.byteCount <= UInt64(mapping.size) else {
                    throw NativeError.integrity("\(t.name): bad offset \(t.offset)")
                }
                let packed = UnsafeRawBufferPointer(start: mapping.base + Int(t.offset), count: Int(t.byteCount))
                guard sha256Hex(packed) == t.packedSHA256 else {
                    throw NativeError.integrity("\(t.name): packed SHA-256 mismatch")
                }
                if t.layout == .gguf {
                    guard t.packedSHA256 == t.sourceSHA256 else {
                        throw NativeError.integrity("\(t.name): identity layout but hashes differ")
                    }
                    continue
                }
                guard let format = QuantBlockFormat.of(t.type) else { throw NativeError.unsupported("\(t.type)") }
                let rows = t.shape.dropFirst().reduce(1, *)
                let blocksPerRow = t.shape[0] / format.elements
                var hasher = SHA256()
                var block = [UInt8](repeating: 0, count: format.bytes)
                for row in 0..<rows {
                    for b in 0..<blocksPerRow {
                        block.withUnsafeMutableBufferPointer { blk in
                            ReferenceDequant.gatherBlock(base: packed.baseAddress!, layout: t.layout, format: format,
                                                         row: row, block: b, blocksPerRow: blocksPerRow,
                                                         into: blk.baseAddress!)
                        }
                        hasher.update(data: block)
                    }
                }
                guard hex(hasher.finalize()) == t.sourceSHA256 else {
                    throw NativeError.integrity("\(t.name): unpacked bytes do not match the source")
                }
            }
        }

        public static func sha256Hex(_ bytes: UnsafeRawBufferPointer) -> String {
            var hasher = SHA256()
            var start = 0
            while start < bytes.count {
                let n = min(streamChunk, bytes.count - start)
                hasher.update(bufferPointer: UnsafeRawBufferPointer(rebasing: bytes[start ..< start + n]))
                start += n
            }
            return hex(hasher.finalize())
        }

        public static func sha256Hex(_ data: Data) -> String {
            data.withUnsafeBytes { sha256Hex($0) }
        }

        static func hex(_ digest: SHA256.Digest) -> String {
            digest.map { String(format: "%02x", $0) }.joined()
        }
    }

    /// A read-only file mapping.
    public final class ReferenceMappedFile {
        public let base: UnsafeRawPointer
        public let size: Int
        /// Bytes mapped (`size` rounded up to the page size).
        public let mappedLength: Int

        public init(url: URL) throws {
            let fd = open(url.path, O_RDONLY)
            guard fd >= 0 else { throw NativeError.missing(url.path) }
            defer { close(fd) }
            var st = stat()
            guard fstat(fd, &st) == 0 else { throw NativeError.resource("stat \(url.path)") }
            size = Int(st.st_size)
            let page = Int(getpagesize())
            mappedLength = max(page, (size + page - 1) / page * page)
            guard let p = mmap(nil, mappedLength, PROT_READ, MAP_PRIVATE, fd, 0), p != MAP_FAILED else {
                throw NativeError.resource("mmap \(url.path)")
            }
            base = UnsafeRawPointer(p)
        }

        deinit { munmap(UnsafeMutableRawPointer(mutating: base), mappedLength) }
    }
}
