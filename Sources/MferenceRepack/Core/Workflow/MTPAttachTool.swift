import Foundation

/// Attaches a checkpoint's native MTP (multi-token-prediction) draft layer to
/// an existing Qwen 3.8 `.gturbo` install.
///
/// The BF16 `mtp.*` tensors from the HF shard are appended to
/// `model_weights.bin` as resident tensors: projections are quantized to
/// INT4 affine group-64 in the MLX layout (`Int4AffineEncoder`, bit-exact
/// with the runtime's reference quantizer) so the existing INT4 GEMV
/// kernels consume them; norm vectors stay BF16 with the HF checkpoint's
/// zero-centered convention converted to full form (`w + 1`, mirroring
/// mlx-vlm's `Qwen3_5MTPDraftModel.sanitize`). Tensor names keep the source
/// `mtp.` prefix, which the runtime probes at load; installs without them
/// are unaffected.
///
/// The resident index is rewritten in place when the new entry table still
/// fits the existing page-padded index region (the common case); otherwise
/// the whole weights file is rebuilt with a grown index. `manifest.json`'s
/// size/sha256 entry is refreshed, and any stale verified-install receipt is
/// removed so it cannot bind to the old manifest.
public enum MTPAttachTool {

    public struct Result: Sendable {
        public let tensorCount: Int
        public let appendedBytes: UInt64
        public let weightsFileBytes: UInt64
        public let rewroteWeightsFile: Bool
    }

    private static let headerBytes = 24
    private static let entryBytes = 72
    private static let pageBytes: UInt64 = 16_384

    private struct RawIndexEntry {
        var name: String
        var dtype: UInt8
        var fileOffset: UInt64
        var sizeBytes: UInt64
        var shape: (UInt32, UInt32, UInt32, UInt32)
        var scaleOffset: UInt64
        var scaleSize: UInt64
        var biasOffset: UInt64
        var biasSize: UInt64
    }

    private struct ExpectedTensor {
        let name: String            // without the "mtp." prefix
        let rows: Int
        let cols: Int?              // nil => BF16 norm vector of `rows` elements
        let addOne: Bool            // zero-centered norm -> full form
    }

    public static func run(gturboDirectory: String, shardPath: String) throws -> Result {
        let weightsPath = (gturboDirectory as NSString)
            .appendingPathComponent("model_weights.bin")
        let manifestPath = (gturboDirectory as NSString)
            .appendingPathComponent("manifest.json")

        // -- Manifest arch dims drive shape validation.
        let manifestData = try Posix.readBoundedData(manifestPath, maximumBytes: 8 << 20)
        guard var manifestRoot = try JSONSerialization
                .jsonObject(with: manifestData) as? [String: Any],
              let arch = manifestRoot["arch"] as? [String: Any],
              let family = arch["family"] as? String, family == "qwen38",
              let d = arch["hiddenSize"] as? Int,
              let f = arch["ffnIntermediate"] as? Int,
              let numHeads = arch["numHeads"] as? Int,
              let numKVHeads = arch["numFullKVHeads"] as? Int,
              let headDim = arch["fullHeadDim"] as? Int else {
            throw RepackError.configurationInvalid(
                detail: "\(manifestPath) is not a Qwen 3.8 manifest")
        }
        // -- Every manifest file must exist at its recorded size BEFORE the
        // irreversible append: a pre-existing missing/truncated sidecar would
        // otherwise only surface in the post-attach receipt pass, when a
        // retry is already rejected as "already attached".
        if let files = manifestRoot["files"] as? [String: Any] {
            for (name, entry) in files {
                let path = (gturboDirectory as NSString).appendingPathComponent(name)
                let recorded = (entry as? [String: Any])?["size"] as? Int
                let actual = (try? FileManager.default
                    .attributesOfItem(atPath: path)[.size] as? Int) ?? nil
                guard let recorded, let actual, recorded == actual else {
                    throw RepackError.configurationInvalid(detail:
                        "install file \(name) is missing or has size " +
                        "\(actual.map(String.init) ?? "unknown") (manifest records " +
                        "\(recorded.map(String.init) ?? "unknown")); repair the " +
                        "install before attaching MTP tensors")
                }
            }
        }

        let qDim = numHeads * headDim
        let kvDim = numKVHeads * headDim
        let expected: [ExpectedTensor] = [
            ExpectedTensor(name: "fc.weight", rows: d, cols: 2 * d, addOne: false),
            ExpectedTensor(name: "pre_fc_norm_embedding.weight", rows: d, cols: nil, addOne: true),
            ExpectedTensor(name: "pre_fc_norm_hidden.weight", rows: d, cols: nil, addOne: true),
            ExpectedTensor(name: "norm.weight", rows: d, cols: nil, addOne: true),
            ExpectedTensor(name: "layers.0.input_layernorm.weight", rows: d, cols: nil, addOne: true),
            ExpectedTensor(name: "layers.0.post_attention_layernorm.weight", rows: d, cols: nil, addOne: true),
            ExpectedTensor(name: "layers.0.self_attn.q_proj.weight", rows: 2 * qDim, cols: d, addOne: false),
            ExpectedTensor(name: "layers.0.self_attn.k_proj.weight", rows: kvDim, cols: d, addOne: false),
            ExpectedTensor(name: "layers.0.self_attn.v_proj.weight", rows: kvDim, cols: d, addOne: false),
            ExpectedTensor(name: "layers.0.self_attn.o_proj.weight", rows: d, cols: qDim, addOne: false),
            ExpectedTensor(name: "layers.0.self_attn.q_norm.weight", rows: headDim, cols: nil, addOne: true),
            ExpectedTensor(name: "layers.0.self_attn.k_norm.weight", rows: headDim, cols: nil, addOne: true),
            ExpectedTensor(name: "layers.0.mlp.gate_proj.weight", rows: f, cols: d, addOne: false),
            ExpectedTensor(name: "layers.0.mlp.up_proj.weight", rows: f, cols: d, addOne: false),
            ExpectedTensor(name: "layers.0.mlp.down_proj.weight", rows: d, cols: f, addOne: false),
        ]

        // -- Existing resident index.
        let weightsFd = try Posix.openExistingRW(weightsPath)
        defer { close(weightsFd) }
        let oldFileSize = try Posix.fileSize(fd: weightsFd, path: weightsPath)
        var (oldIndexSize, oldResidentSize, oldEntries) =
            try readIndex(fd: weightsFd, path: weightsPath)
        guard oldIndexSize + oldResidentSize == oldFileSize else {
            throw RepackError.configurationInvalid(detail:
                "\(weightsPath) size \(oldFileSize) != index \(oldIndexSize) + resident \(oldResidentSize)")
        }
        guard !oldEntries.contains(where: { $0.name.hasPrefix("mtp.") }) else {
            throw RepackError.configurationInvalid(
                detail: "\(gturboDirectory) already carries mtp.* tensors")
        }

        // -- Source shard.
        let shardFd = try Posix.openRead(shardPath)
        defer { close(shardFd) }
        let shardSize = try Posix.fileSize(fd: shardFd, path: shardPath)
        var headerLenBytes = [UInt8](repeating: 0, count: 8)
        try headerLenBytes.withUnsafeMutableBytes {
            try Posix.preadAll(fd: shardFd, path: shardPath,
                               buf: $0.baseAddress!, count: 8, offset: 0)
        }
        let headerLen = headerLenBytes.withUnsafeBytes {
            $0.load(as: UInt64.self).littleEndian
        }
        guard headerLen > 0, headerLen < Safetensors.maxHeaderBytes else {
            throw RepackError.safetensorsHeaderTooLarge(path: shardPath, size: headerLen)
        }
        var headerData = Data(count: Int(headerLen))
        try headerData.withUnsafeMutableBytes {
            try Posix.preadAll(fd: shardFd, path: shardPath,
                               buf: $0.baseAddress!, count: Int(headerLen), offset: 8)
        }
        let header = try Safetensors.parseHeaderBytes(path: shardPath,
                                                      fileSize: shardSize,
                                                      headerBytes: headerData)
        var byName: [String: SourceTensor] = [:]
        for t in header.tensors where t.name.hasPrefix("mtp.") {
            byName[t.name] = t
        }

        // -- Quantize / convert into an appended-payload staging file plan.
        // Payload is written straight after the current file end; entries are
        // built against absolute offsets in the (possibly shifted) new file.
        var appended = Data()
        appended.reserveCapacity(1 << 20)
        var newEntries: [RawIndexEntry] = []
        let payloadBase = oldFileSize

        for spec in expected {
            let fullName = "mtp.\(spec.name)"
            guard let tensor = byName[fullName] else {
                throw RepackError.missingTensor(name: fullName)
            }
            guard tensor.dtype == .bf16 else {
                throw RepackError.dtypeMismatch(name: fullName,
                    detail: "expected BF16, got \(tensor.dtype)")
            }
            if let cols = spec.cols {
                guard tensor.shape == [UInt64(spec.rows), UInt64(cols)] else {
                    throw RepackError.shapeMismatch(name: fullName,
                        detail: "expected [\(spec.rows), \(cols)], got \(tensor.shape)")
                }
                guard cols % Int4AffineEncoder.groupSize == 0 else {
                    throw RepackError.shapeMismatch(name: fullName,
                        detail: "cols \(cols) not a multiple of \(Int4AffineEncoder.groupSize)")
                }
                let weightOffset = payloadBase + UInt64(appended.count)
                var scales = [UInt16]()
                var biases = [UInt16]()
                let groups = cols / Int4AffineEncoder.groupSize
                scales.reserveCapacity(spec.rows * groups)
                biases.reserveCapacity(spec.rows * groups)
                // Row-batched: read BF16 rows, widen, quantize, append.
                let batchRows = max(1, (8 << 20) / (cols * 2))
                var floats = [Float](repeating: 0, count: batchRows * cols)
                var raw = [UInt16](repeating: 0, count: batchRows * cols)
                var row = 0
                while row < spec.rows {
                    let take = min(batchRows, spec.rows - row)
                    let byteCount = take * cols * 2
                    try raw.withUnsafeMutableBytes {
                        try Posix.preadAll(fd: shardFd, path: shardPath,
                                           buf: $0.baseAddress!, count: byteCount,
                                           offset: tensor.absoluteOffset
                                               + UInt64(row * cols * 2))
                    }
                    for i in 0..<(take * cols) {
                        floats[i] = Float(bitPattern: UInt32(raw[i]) << 16)
                    }
                    let encoded = floats.withUnsafeBufferPointer {
                        Int4AffineEncoder.encodeTensor(
                            UnsafeBufferPointer(rebasing: $0.prefix(take * cols)),
                            rowLength: cols)
                    }
                    appended.append(contentsOf: encoded.packed)
                    scales.append(contentsOf: encoded.scales)
                    biases.append(contentsOf: encoded.biases)
                    row += take
                }
                let scaleOffset = payloadBase + UInt64(appended.count)
                scales.withUnsafeBufferPointer {
                    appended.append(UnsafeRawBufferPointer($0).bindMemory(to: UInt8.self))
                }
                let biasOffset = payloadBase + UInt64(appended.count)
                biases.withUnsafeBufferPointer {
                    appended.append(UnsafeRawBufferPointer($0).bindMemory(to: UInt8.self))
                }
                newEntries.append(RawIndexEntry(
                    name: fullName, dtype: 0,
                    fileOffset: weightOffset,
                    sizeBytes: UInt64(spec.rows * cols / 2),
                    shape: (UInt32(spec.rows), UInt32(cols), 0, 0),
                    scaleOffset: scaleOffset,
                    scaleSize: UInt64(spec.rows * groups * 2),
                    biasOffset: biasOffset,
                    biasSize: UInt64(spec.rows * groups * 2)))
            } else {
                guard tensor.shape == [UInt64(spec.rows)] else {
                    throw RepackError.shapeMismatch(name: fullName,
                        detail: "expected [\(spec.rows)], got \(tensor.shape)")
                }
                var raw = [UInt16](repeating: 0, count: spec.rows)
                try raw.withUnsafeMutableBytes {
                    try Posix.preadAll(fd: shardFd, path: shardPath,
                                       buf: $0.baseAddress!, count: spec.rows * 2,
                                       offset: tensor.absoluteOffset)
                }
                if spec.addOne {
                    for i in 0..<raw.count {
                        let value = Float(bitPattern: UInt32(raw[i]) << 16) + 1.0
                        raw[i] = Int4AffineEncoder.bf16Bits(value)
                    }
                }
                let offset = payloadBase + UInt64(appended.count)
                raw.withUnsafeBufferPointer {
                    appended.append(UnsafeRawBufferPointer($0).bindMemory(to: UInt8.self))
                }
                newEntries.append(RawIndexEntry(
                    name: fullName, dtype: 1,
                    fileOffset: offset,
                    sizeBytes: UInt64(spec.rows * 2),
                    shape: (UInt32(spec.rows), 0, 0, 0),
                    scaleOffset: 0, scaleSize: 0,
                    biasOffset: 0, biasSize: 0))
            }
        }

        // -- New index image.
        let allEntries = oldEntries + newEntries
        let rawIndexBytes = headerBytes + allEntries.count * entryBytes
            + allEntries.reduce(0) { $0 + $1.name.utf8.count }
        let fitsInPlace = UInt64(rawIndexBytes) <= oldIndexSize
        let newIndexSize = fitsInPlace
            ? oldIndexSize
            : ((UInt64(rawIndexBytes) + pageBytes - 1) / pageBytes) * pageBytes
        let shift = newIndexSize - oldIndexSize
        let newResidentSize = oldResidentSize + UInt64(appended.count)

        var shifted = allEntries
        if shift > 0 {
            for i in 0..<shifted.count {
                shifted[i].fileOffset += shift
                if shifted[i].scaleSize > 0 { shifted[i].scaleOffset += shift }
                if shifted[i].biasSize > 0 { shifted[i].biasOffset += shift }
            }
        }
        let indexImage = encodeIndex(entries: shifted,
                                     indexSize: newIndexSize,
                                     residentSize: newResidentSize)

        let rewrote = !fitsInPlace
        if fitsInPlace {
            try appended.withUnsafeBytes {
                try Posix.pwriteAll(fd: weightsFd, path: weightsPath,
                                    buf: $0.baseAddress!, count: appended.count,
                                    offset: oldFileSize)
            }
            try indexImage.withUnsafeBytes {
                try Posix.pwriteAll(fd: weightsFd, path: weightsPath,
                                    buf: $0.baseAddress!, count: indexImage.count,
                                    offset: 0)
            }
            try Posix.fsync(weightsFd, path: weightsPath)
        } else {
            // Grown index: rebuild the file with every payload byte shifted.
            let tmpPath = weightsPath + ".mtp-attach.tmp"
            let outFd = try Posix.openCreateRW(tmpPath)
            defer { close(outFd) }
            try indexImage.withUnsafeBytes {
                try Posix.pwriteAll(fd: outFd, path: tmpPath,
                                    buf: $0.baseAddress!, count: indexImage.count,
                                    offset: 0)
            }
            let tile = 8 << 20
            let buf = UnsafeMutableRawBufferPointer.allocate(byteCount: tile, alignment: 16_384)
            defer { buf.deallocate() }
            var copied: UInt64 = 0
            while copied < oldResidentSize {
                let take = Int(min(UInt64(tile), oldResidentSize - copied))
                try Posix.preadAll(fd: weightsFd, path: weightsPath,
                                   buf: buf.baseAddress!, count: take,
                                   offset: oldIndexSize + copied)
                try Posix.pwriteAll(fd: outFd, path: tmpPath,
                                    buf: buf.baseAddress!, count: take,
                                    offset: newIndexSize + copied)
                copied += UInt64(take)
            }
            try appended.withUnsafeBytes {
                try Posix.pwriteAll(fd: outFd, path: tmpPath,
                                    buf: $0.baseAddress!, count: appended.count,
                                    offset: newIndexSize + oldResidentSize)
            }
            try Posix.fsync(outFd, path: tmpPath)
            try Posix.rename(from: tmpPath, to: weightsPath)
        }

        // -- Manifest refresh.
        let newSize = newIndexSize + newResidentSize
        let sha = try Sha256Stream.hashFile(path: weightsPath, tileBytes: 8 << 20)
        guard var files = manifestRoot["files"] as? [String: Any] else {
            throw RepackError.configurationInvalid(detail: "manifest.json has no files map")
        }
        files["model_weights.bin"] = ["size": Int(newSize), "sha256": sha]
        manifestRoot["files"] = files
        let newManifest = try JSONSerialization.data(
            withJSONObject: manifestRoot,
            options: [.sortedKeys, .withoutEscapingSlashes])
        try Posix.atomicWrite(newManifest, to: manifestPath,
                              durableIn: gturboDirectory)
        // The attach invalidated the old receipt; regenerate it so installs
        // that gate on verified-install.json (the Mac app) keep working
        // without a manual --verify-install pass. The attach itself is
        // complete at this point, so a receipt failure must say so and point
        // at the retryable step rather than reading as a failed attach.
        let receiptPath = (gturboDirectory as NSString)
            .appendingPathComponent("verified-install.json")
        try? FileManager.default.removeItem(atPath: receiptPath)
        do {
            _ = try VerifiedInstallTool.run(
                options: VerifyInstallOptions(inputGTurbo: gturboDirectory))
        } catch {
            throw RepackError.configurationInvalid(detail:
                "MTP tensors were attached, but regenerating the install " +
                "receipt failed (\(error)); repair the install and run " +
                "--verify-install --input-gturbo \(gturboDirectory)")
        }

        return Result(tensorCount: newEntries.count,
                      appendedBytes: UInt64(appended.count),
                      weightsFileBytes: newSize,
                      rewroteWeightsFile: rewrote)
    }

    // MARK: - Index encode/decode

    private static func readIndex(fd: Int32, path: String)
        throws -> (indexSize: UInt64, residentSize: UInt64, entries: [RawIndexEntry]) {
        var header = [UInt8](repeating: 0, count: headerBytes)
        try header.withUnsafeMutableBytes {
            try Posix.preadAll(fd: fd, path: path, buf: $0.baseAddress!,
                               count: headerBytes, offset: 0)
        }
        func u64(_ bytes: [UInt8], _ off: Int) -> UInt64 {
            var v: UInt64 = 0
            for i in (0..<8).reversed() { v = (v << 8) | UInt64(bytes[off + i]) }
            return v
        }
        let indexSize = u64(header, 0)
        let residentSize = u64(header, 8)
        let entryCount = Int(u64(header, 16))
        guard indexSize >= UInt64(headerBytes + entryCount * entryBytes) else {
            throw RepackError.configurationInvalid(detail: "\(path): corrupt index header")
        }
        var region = [UInt8](repeating: 0, count: Int(indexSize))
        try region.withUnsafeMutableBytes {
            try Posix.preadAll(fd: fd, path: path, buf: $0.baseAddress!,
                               count: Int(indexSize), offset: 0)
        }
        var entries: [RawIndexEntry] = []
        entries.reserveCapacity(entryCount)
        try region.withUnsafeBytes { raw in
            let base = raw.baseAddress!
            func u64p(_ p: UnsafeRawPointer, _ off: Int) -> UInt64 {
                var v: UInt64 = 0
                let bytes = p.advanced(by: off).assumingMemoryBound(to: UInt8.self)
                for i in (0..<8).reversed() { v = (v << 8) | UInt64(bytes[i]) }
                return v
            }
            func u32p(_ p: UnsafeRawPointer, _ off: Int) -> UInt32 {
                var v: UInt32 = 0
                let bytes = p.advanced(by: off).assumingMemoryBound(to: UInt8.self)
                for i in (0..<4).reversed() { v = (v << 8) | UInt32(bytes[i]) }
                return v
            }
            for i in 0..<entryCount {
                let p = base.advanced(by: headerBytes + i * entryBytes)
                let nameOffset = Int(u32p(p, 0))
                let nameLength = Int(UInt16(p.advanced(by: 4)
                    .assumingMemoryBound(to: UInt8.self)[0])
                    | (UInt16(p.advanced(by: 5).assumingMemoryBound(to: UInt8.self)[0]) << 8))
                guard nameOffset >= headerBytes,
                      nameOffset + nameLength <= Int(indexSize) else {
                    throw RepackError.configurationInvalid(
                        detail: "\(path): entry \(i) name out of range")
                }
                let name = String(decoding: UnsafeRawBufferPointer(
                    start: base.advanced(by: nameOffset), count: nameLength), as: UTF8.self)
                entries.append(RawIndexEntry(
                    name: name,
                    dtype: p.advanced(by: 6).assumingMemoryBound(to: UInt8.self)[0],
                    fileOffset: u64p(p, 8),
                    sizeBytes: u64p(p, 16),
                    shape: (u32p(p, 24), u32p(p, 28), u32p(p, 32), u32p(p, 36)),
                    scaleOffset: u64p(p, 40),
                    scaleSize: u64p(p, 48),
                    biasOffset: u64p(p, 56),
                    biasSize: u64p(p, 64)))
            }
        }
        return (indexSize, residentSize, entries)
    }

    private static func encodeIndex(entries: [RawIndexEntry],
                                    indexSize: UInt64,
                                    residentSize: UInt64) -> Data {
        var image = Data(count: Int(indexSize))
        image.withUnsafeMutableBytes { raw in
            let base = raw.baseAddress!
            GTurboBinary.writeIndexHeader(into: base,
                                          indexSize: indexSize,
                                          residentSize: residentSize,
                                          entryCount: UInt64(entries.count))
            let stringTableBase = headerBytes + entries.count * entryBytes
            var nameCursor = 0
            for (i, entry) in entries.enumerated() {
                let nameOffset = UInt32(stringTableBase + nameCursor)
                let nameBytes = Array(entry.name.utf8)
                memcpy(base.advanced(by: stringTableBase + nameCursor),
                       nameBytes, nameBytes.count)
                nameCursor += nameBytes.count
                let dst = base.advanced(by: headerBytes + i * entryBytes)
                var off = 0
                writeU32LE(dst, &off, nameOffset)
                writeU16LE(dst, &off, UInt16(entry.name.utf8.count))
                writeU8(dst, &off, entry.dtype)
                writeU8(dst, &off, 0)
                writeU64LE(dst, &off, entry.fileOffset)
                writeU64LE(dst, &off, entry.sizeBytes)
                writeU32LE(dst, &off, entry.shape.0)
                writeU32LE(dst, &off, entry.shape.1)
                writeU32LE(dst, &off, entry.shape.2)
                writeU32LE(dst, &off, entry.shape.3)
                writeU64LE(dst, &off, entry.scaleOffset)
                writeU64LE(dst, &off, entry.scaleSize)
                writeU64LE(dst, &off, entry.biasOffset)
                writeU64LE(dst, &off, entry.biasSize)
            }
        }
        return image
    }

    private static func writeU64LE(_ buf: UnsafeMutableRawPointer, _ off: inout Int, _ v: UInt64) {
        var x = v.littleEndian
        withUnsafeBytes(of: &x) { src in
            memcpy(buf.advanced(by: off), src.baseAddress!, 8)
        }
        off += 8
    }

    private static func writeU32LE(_ buf: UnsafeMutableRawPointer, _ off: inout Int, _ v: UInt32) {
        var x = v.littleEndian
        withUnsafeBytes(of: &x) { src in
            memcpy(buf.advanced(by: off), src.baseAddress!, 4)
        }
        off += 4
    }

    private static func writeU16LE(_ buf: UnsafeMutableRawPointer, _ off: inout Int, _ v: UInt16) {
        var x = v.littleEndian
        withUnsafeBytes(of: &x) { src in
            memcpy(buf.advanced(by: off), src.baseAddress!, 2)
        }
        off += 2
    }

    private static func writeU8(_ buf: UnsafeMutableRawPointer, _ off: inout Int, _ v: UInt8) {
        buf.advanced(by: off).assumingMemoryBound(to: UInt8.self)[0] = v
        off += 1
    }
}
