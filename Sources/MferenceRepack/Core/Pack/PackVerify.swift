import CryptoKit
import Darwin
import Foundation

/// Checks of an installed `.mfpack` against itself and against its source GGUF.
public enum MFPackVerifier {
    public struct RoundTripReport: Sendable, Equatable {
        public var tensors: Int
        public var bytes: UInt64
        /// Tensors stored in a non-identity tile layout.
        public var tiledTensors: Int
    }

    /// Every tensor of the pack at `directory`, unpacked to GGUF block order,
    /// compared byte for byte (`memcmp`, not only hashes) with the same tensor
    /// in the GGUF at `gguf`; also the tensor list, types, shapes, the source
    /// size and the receipt. Streams both files in bounded chunks.
    public static func roundTrip(pack directory: String, gguf: String,
                                 chunkBytes: Int = 8 << 20) throws -> RoundTripReport {
        let (manifest, _) = try readVerified(directory: directory)
        let header = try MFPack.GGUFHeader(path: gguf)
        guard UInt64(header.fileSize) == manifest.source.byteCount else {
            throw MFPackError.integrity("\(gguf) is \(header.fileSize) bytes, the pack's source was "
                                        + "\(manifest.source.byteCount)")
        }
        guard header.tensors.map(\.name) == manifest.tensors.map(\.name) else {
            throw MFPackError.integrity("the pack's tensor list differs from the GGUF's")
        }
        let weightsPath = (directory as NSString).appendingPathComponent("weights.bin")
        let packFD = open(weightsPath, O_RDONLY | O_CLOEXEC)
        guard packFD >= 0 else { throw MFPackError.io("open \(weightsPath): errno \(errno)") }
        defer { close(packFD) }
        let srcFD = open(gguf, O_RDONLY | O_CLOEXEC)
        guard srcFD >= 0 else { throw MFPackError.io("open \(gguf): errno \(errno)") }
        defer { close(srcFD) }
        _ = fcntl(packFD, F_NOCACHE, 1)
        _ = fcntl(srcFD, F_NOCACHE, 1)
        let packScratch = BoundedScratch(limitBytes: max(chunkBytes, 1 << 20) * 2)
        let srcScratch = BoundedScratch(limitBytes: max(chunkBytes, 1 << 20) * 2)
        let unpackScratch = BoundedScratch(limitBytes: max(chunkBytes, 1 << 20) * 2)
        var report = RoundTripReport(tensors: 0, bytes: 0, tiledTensors: 0)
        for (info, t) in zip(header.tensors, manifest.tensors) {
            guard info.type == t.type, info.shape == t.shape, UInt64(info.byteCount) == t.byteCount,
                  let format = MFPack.QuantBlockFormat.of(t.type) else {
                throw MFPackError.integrity("\(t.name): type, shape or size differs from the GGUF")
            }
            let rows = t.shape.dropFirst().reduce(1, *)
            let blocksPerRow = t.shape[0] / format.elements
            let permuter = MFPack.TilePermuter(layout: t.layout, format: format, rows: rows,
                                               blocksPerRow: blocksPerRow)
            let unit = permuter.isIdentity ? 1 : permuter.groupBytes
            let step = max(unit, chunkBytes / unit * unit)
            let sourceBase = UInt64(header.dataOffset) + info.offset
            var done = 0
            let count = Int(t.byteCount)
            while done < count {
                let n = min(step, count - done)
                let packed = try packScratch.ensure(n)
                let source = try srcScratch.ensure(n)
                try Posix.preadAll(fd: packFD, path: weightsPath, buf: packed.baseAddress!, count: n,
                                   offset: t.offset + UInt64(done))
                try Posix.preadAll(fd: srcFD, path: gguf, buf: source.baseAddress!, count: n,
                                   offset: sourceBase + UInt64(done))
                var ggufOrder = UnsafeRawPointer(packed.baseAddress!)
                if !permuter.isIdentity {
                    let out = try unpackScratch.ensure(n)
                    let firstGroup = done / unit
                    for k in 0 ..< n / unit {
                        permuter.unpack(group: firstGroup + k, src: UnsafeRawPointer(packed.baseAddress! + k * unit),
                                        dst: out.baseAddress! + k * unit)
                    }
                    ggufOrder = UnsafeRawPointer(out.baseAddress!)
                }
                guard memcmp(ggufOrder, source.baseAddress!, n) == 0 else {
                    throw MFPackError.integrity("\(t.name): bytes \(done) ..< \(done + n) differ from the GGUF")
                }
                done += n
            }
            report.tensors += 1
            report.bytes += t.byteCount
            if !permuter.isIdentity { report.tiledTensors += 1 }
        }
        return report
    }

    /// `manifest.json` and `verified-install.json` of an installed pack, with
    /// the receipt checked against the manifest, `weights.bin` size and the
    /// metadata files' hashes.
    public static func readVerified(directory: String) throws -> (MFPack.Manifest, MFPack.Receipt) {
        func path(_ name: String) -> URL { URL(fileURLWithPath: (directory as NSString).appendingPathComponent(name)) }
        let receipt: MFPack.Receipt
        let manifestData: Data
        do {
            receipt = try JSONDecoder().decode(MFPack.Receipt.self, from: Data(contentsOf: path("verified-install.json")))
            manifestData = try Data(contentsOf: path("manifest.json"))
        } catch {
            throw MFPackError.integrity("\(directory) is not a complete pack: \(error)")
        }
        guard MFPackWriter.sha256Hex(manifestData) == receipt.manifestSHA256 else {
            throw MFPackError.integrity("manifest.json does not match verified-install.json")
        }
        let manifest = try JSONDecoder().decode(MFPack.Manifest.self, from: manifestData)
        if let want = receipt.tokenizerSHA256,
           MFPackWriter.sha256Hex(try Data(contentsOf: path("tokenizer.json"))) != want {
            throw MFPackError.integrity("tokenizer.json does not match verified-install.json")
        }
        if let want = receipt.chatTemplateSHA256,
           MFPackWriter.sha256Hex(try Data(contentsOf: path("chat_template.jinja"))) != want {
            throw MFPackError.integrity("chat_template.jinja does not match verified-install.json")
        }
        var st = stat()
        guard stat(path("weights.bin").path, &st) == 0, UInt64(st.st_size) == receipt.weightsByteCount,
              receipt.tensorCount == manifest.tensors.count, receipt.sourceSHA256 == manifest.source.sha256 else {
            throw MFPackError.integrity("verified-install.json does not describe this pack")
        }
        return (manifest, receipt)
    }
}
