import Foundation

extension MFPack {
    /// One tensor's GGUF → pack permutation (`TileLayout.packedOffset`),
    /// streamed one tile row group at a time. A row group's tiles are
    /// contiguous in the pack, so a group is the unit of both directions;
    /// memory is two group buffers from bounded scratch, never a tensor.
    struct TilePermuter {
        let layout: TileLayout
        let format: QuantBlockFormat
        let rows: Int
        let blocksPerRow: Int
        /// Bytes of one row group (`rowsPerTile` rows).
        let groupBytes: Int

        init(layout: TileLayout, format: QuantBlockFormat, rows: Int, blocksPerRow: Int) {
            self.layout = layout
            self.format = format
            self.rows = rows
            self.blocksPerRow = blocksPerRow
            groupBytes = layout.rowsPerTile * blocksPerRow * format.bytes
        }

        var isIdentity: Bool { layout == .gguf }

        private var fields: [BlockField] {
            layout.planes ? format.fields : [BlockField(name: "block", bytes: format.bytes)]
        }

        /// Pack one row group: `src` in GGUF order → `dst` in pack order.
        func pack(group: Int, src: UnsafeRawPointer, dst: UnsafeMutableRawPointer) {
            permute(group: group, gguf: UnsafeMutableRawPointer(mutating: src), packed: dst, toPacked: true)
        }

        /// Unpack one row group: `src` in pack order → `dst` in GGUF order.
        func unpack(group: Int, src: UnsafeRawPointer, dst: UnsafeMutableRawPointer) {
            permute(group: group, gguf: dst, packed: UnsafeMutableRawPointer(mutating: src), toPacked: false)
        }

        private func permute(group: Int, gguf: UnsafeMutableRawPointer, packed: UnsafeMutableRawPointer,
                             toPacked: Bool) {
            let groupBase = group * groupBytes
            let fields = fields
            for rInGroup in 0 ..< layout.rowsPerTile {
                let row = group * layout.rowsPerTile + rInGroup
                for block in 0 ..< blocksPerRow {
                    let g = (rInGroup * blocksPerRow + block) * format.bytes
                    var fieldStart = 0
                    for field in fields {
                        let p = layout.packedOffset(row: row, block: block, byteInBlock: fieldStart,
                                                    blocksPerRow: blocksPerRow, format: format) - groupBase
                        if toPacked {
                            (packed + p).copyMemory(from: gguf + g + fieldStart, byteCount: field.bytes)
                        } else {
                            (gguf + g + fieldStart).copyMemory(from: packed + p, byteCount: field.bytes)
                        }
                        fieldStart += field.bytes
                    }
                }
            }
        }
    }

    /// `<output>.resume.json`: what a resumed write may skip. A tensor is
    /// listed only after its packed bytes were fsynced.
    struct ResumeCheckpoint: Codable, Equatable {
        struct Hashes: Codable, Equatable {
            var source: String
            var packed: String
        }

        var format = "mfpack-resume"
        var version = 1
        var label: String
        var sourceByteCount: UInt64
        var sourceModifiedNanoseconds: Int64
        var sourceInode: UInt64
        /// SHA-256 of the canonical write plan (names, types, shapes, layouts,
        /// source and pack offsets): a resume must replay the same plan.
        var planSHA256: String
        var weightsByteCount: UInt64
        var completed: [String: Hashes]

        func matches(_ other: ResumeCheckpoint) -> Bool {
            format == other.format && version == other.version && label == other.label
                && sourceByteCount == other.sourceByteCount
                && sourceModifiedNanoseconds == other.sourceModifiedNanoseconds
                && sourceInode == other.sourceInode && planSHA256 == other.planSHA256
                && weightsByteCount == other.weightsByteCount
        }
    }
}
