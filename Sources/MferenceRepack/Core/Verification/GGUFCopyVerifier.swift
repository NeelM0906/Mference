import Darwin
import Foundation

/// Proves an install copied from a GGUF holds the GGUF's bytes: for every
/// tensor in the file's header, the bytes at its place in the install
/// (`model_weights.bin` through the resident index, or each expert's slice of
/// `packed_experts/layer_LL.bin` through `layout.json`) equal the bytes at its
/// place in the GGUF. Reads both sides through one bounded tile; holds no
/// tensor in memory.
public enum GGUFCopyVerifier {
    public struct Result: Sendable, Equatable {
        public let residentTensors: Int
        public let expertTensors: Int
        public let bytesCompared: UInt64
    }

    static let tileBytes = 8 << 20

    public static func run(installDirectory: String, ggufPath: String) throws -> Result {
        let ggufFD = try Posix.openReadNoFollow(ggufPath)
        defer { close(ggufFD) }
        let ggufSize = try Posix.fileSize(fd: ggufFD, path: ggufPath)
        let (header, _) = try LocalGGUFSource.readHeader(path: ggufPath, size: ggufSize)
        try requireTiledDataSection(header: header, fileSize: ggufSize, path: ggufPath)

        let weightsPath = (installDirectory as NSString).appendingPathComponent("model_weights.bin")
        let resident = try readResidentIndex(path: weightsPath)
        let layout = try readLayout(installDirectory: installDirectory)

        let tile = UnsafeMutableRawBufferPointer.allocate(byteCount: tileBytes * 2, alignment: 16_384)
        defer { tile.deallocate() }
        var residentCount = 0, expertCount = 0
        var compared: UInt64 = 0
        var openFiles: [String: Int32] = [:]
        defer { openFiles.values.forEach { close($0) } }
        func fd(_ path: String) throws -> Int32 {
            if let open = openFiles[path] { return open }
            let opened = try Posix.openReadNoFollow(path)
            openFiles[path] = opened
            return opened
        }
        func compare(_ name: String, sourceOffset: UInt64, size: UInt64,
                     installPath: String, installOffset: UInt64) throws {
            let installFD = try fd(installPath)
            var done: UInt64 = 0
            while done < size {
                let count = Int(min(UInt64(tileBytes), size - done))
                let a = tile.baseAddress!, b = tile.baseAddress!.advanced(by: tileBytes)
                try Posix.preadAll(fd: ggufFD, path: ggufPath, buf: a, count: count, offset: sourceOffset + done)
                try Posix.preadAll(fd: installFD, path: installPath, buf: b, count: count,
                                   offset: installOffset + done)
                guard memcmp(a, b, count) == 0 else {
                    throw RepackError.configurationInvalid(
                        detail: "\(name): install bytes differ from the GGUF in [\(done), \(done + UInt64(count)))")
                }
                done += UInt64(count)
            }
            compared += size
        }

        for t in header.tensors {
            try Task.checkCancellation()
            if let (layer, role) = Inkling2GGUFPlanner.expertRole(t.name) {
                guard let entry = layout[layer], let slice = entry.tensors[role] else {
                    throw RepackError.missingTensor(name: "\(t.name) in layout.json")
                }
                guard slice.format == t.type.name else {
                    throw RepackError.dtypeMismatch(name: t.name,
                                                    detail: "layout.json says \(slice.format ?? "nil")")
                }
                guard slice.size * UInt64(entry.experts) == t.sizeBytes else {
                    throw RepackError.shapeMismatch(name: t.name, detail: "expert slices do not tile the tensor")
                }
                let path = ((installDirectory as NSString).appendingPathComponent("packed_experts")
                    as NSString).appendingPathComponent(entry.file)
                for e in 0..<UInt64(entry.experts) {
                    try compare("\(t.name)[\(e)]", sourceOffset: t.absoluteOffset + e * slice.size,
                                size: slice.size, installPath: path,
                                installOffset: e * entry.stride + slice.offset)
                }
                expertCount += 1
            } else {
                guard let entry = resident[t.name] else {
                    throw RepackError.missingTensor(name: "\(t.name) in model_weights.bin")
                }
                let shape = RepackPlanner.padTo4(t.ne.reversed())
                guard entry.size == t.sizeBytes, entry.dtype == t.type.residentDtype, entry.shape == shape else {
                    throw RepackError.dtypeMismatch(name: t.name,
                                                    detail: "install holds \(entry.size) bytes of dtype \(entry.dtype)")
                }
                try compare(t.name, sourceOffset: t.absoluteOffset, size: t.sizeBytes,
                            installPath: weightsPath, installOffset: entry.offset)
                residentCount += 1
            }
        }
        guard residentCount == resident.count else {
            throw RepackError.configurationInvalid(
                detail: "model_weights.bin holds \(resident.count) tensors, the GGUF \(residentCount)")
        }
        return Result(residentTensors: residentCount, expertTensors: expertCount, bytesCompared: compared)
    }

    struct ResidentEntryRef { let dtype: UInt8; let offset: UInt64; let size: UInt64; let shape: [UInt32] }

    /// Independent of the parser's offsets being right: the tensors, in file
    /// order, must start at the data section and follow each other with
    /// nothing but alignment padding, up to the end of the file. A wrong data
    /// offset or tensor size in the parser cannot then go unnoticed by a
    /// comparison that uses those same offsets on both sides.
    static func requireTiledDataSection(header: GGUF.Header, fileSize: UInt64, path: String) throws {
        let sorted = header.tensors.sorted { $0.absoluteOffset < $1.absoluteOffset }
        var cursor = header.dataOffset
        for t in sorted {
            guard t.absoluteOffset == cursor else {
                throw RepackError.ggufInvalid(path: path,
                    detail: "\(t.name) starts at \(t.absoluteOffset), expected \(cursor) (tensors do not tile the data)")
            }
            let end = t.absoluteOffset + t.sizeBytes
            cursor = (end + header.alignment - 1) / header.alignment * header.alignment
            if t.name == sorted.last?.name { cursor = end }
        }
        guard cursor <= fileSize, fileSize - cursor < header.alignment else {
            throw RepackError.ggufInvalid(path: path, detail: "data ends at \(cursor), file at \(fileSize)")
        }
    }

    /// Name -> (dtype, absolute offset, size) from the resident index.
    static func readResidentIndex(path: String) throws -> [String: ResidentEntryRef] {
        let fd = try Posix.openReadNoFollow(path)
        defer { close(fd) }
        var head = [UInt8](repeating: 0, count: GTurboBinary.indexHeaderBytes)
        try head.withUnsafeMutableBytes {
            try Posix.preadAll(fd: fd, path: path, buf: $0.baseAddress!, count: $0.count, offset: 0)
        }
        func u64(_ b: [UInt8], _ o: Int) -> UInt64 { (0..<8).reduce(0) { $0 | UInt64(b[o + $1]) << (8 * $1) } }
        func u32(_ b: [UInt8], _ o: Int) -> UInt32 { (0..<4).reduce(0) { $0 | UInt32(b[o + $1]) << (8 * $1) } }
        let indexSize = u64(head, 0), count = u64(head, 16)
        guard indexSize <= 64 << 20, count <= 1 << 20,
              UInt64(GTurboBinary.indexHeaderBytes) + count * UInt64(GTurboBinary.indexEntryBytes) <= indexSize else {
            throw RepackError.configurationInvalid(detail: "model_weights.bin index is implausible")
        }
        var region = [UInt8](repeating: 0, count: Int(indexSize))
        try region.withUnsafeMutableBytes {
            try Posix.preadAll(fd: fd, path: path, buf: $0.baseAddress!, count: $0.count, offset: 0)
        }
        var out: [String: ResidentEntryRef] = [:]
        for i in 0..<Int(count) {
            let p = GTurboBinary.indexHeaderBytes + i * GTurboBinary.indexEntryBytes
            let nameOffset = Int(u32(region, p)), nameLength = Int(region[p + 4]) | Int(region[p + 5]) << 8
            guard nameOffset + nameLength <= region.count else {
                throw RepackError.configurationInvalid(detail: "resident index name out of range")
            }
            let name = String(decoding: region[nameOffset..<(nameOffset + nameLength)], as: UTF8.self)
            out[name] = ResidentEntryRef(dtype: region[p + 6], offset: u64(region, p + 8), size: u64(region, p + 16),
                                         shape: (0..<4).map { u32(region, p + 24 + 4 * $0) })
        }
        return out
    }

    struct LayerRef {
        let file: String
        let stride: UInt64
        let experts: Int
        let tensors: [String: (offset: UInt64, size: UInt64, format: String?)]
    }

    /// Layer -> file, stride, expert count and expert-0 slices from `layout.json`.
    static func readLayout(installDirectory: String) throws -> [Int: LayerRef] {
        let path = ((installDirectory as NSString).appendingPathComponent("packed_experts")
            as NSString).appendingPathComponent("layout.json")
        guard let root = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: path)))
                as? [String: Any],
              let top = (root["expertStride"] as? NSNumber)?.uint64Value,
              let layers = root["layers"] as? [[String: Any]] else {
            throw RepackError.configurationInvalid(detail: "layout.json is malformed")
        }
        var out: [Int: LayerRef] = [:]
        for layer in layers {
            guard let index = layer["layer"] as? Int, let file = layer["file"] as? String,
                  let experts = layer["experts"] as? [[String: Any]], let first = experts.first,
                  let tensors = first["tensors"] as? [String: [String: Any]] else { continue }
            let stride = (layer["expertStride"] as? NSNumber)?.uint64Value ?? top
            var slices: [String: (offset: UInt64, size: UInt64, format: String?)] = [:]
            for (role, t) in tensors {
                guard let o = (t["offset"] as? NSNumber)?.uint64Value,
                      let s = (t["size"] as? NSNumber)?.uint64Value else {
                    throw RepackError.configurationInvalid(detail: "layout.json tensor \(role) is malformed")
                }
                slices[role] = (o, s, t["format"] as? String)
            }
            for (e, expert) in experts.enumerated() {
                guard (expert["offset"] as? NSNumber)?.uint64Value == UInt64(e) * stride else {
                    throw RepackError.configurationInvalid(detail: "layer \(index) expert \(e) is off its stride")
                }
            }
            out[index] = LayerRef(file: file, stride: stride, experts: experts.count, tensors: slices)
        }
        return out
    }
}
