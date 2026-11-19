extension Native {
    /// How a 2-D quantized matrix's GGUF blocks are arranged in an `.mfpack`
    /// (D3: an exact byte permutation of the GGUF blocks, no requantization).
    ///
    /// A matrix has `rows` output rows (ggml `ne[1]`) of `cols` inputs (ggml
    /// `ne[0]`), so `blocksPerRow = cols / format.elements` blocks of
    /// `format.bytes` each. In GGUF order block `b` of row `r` sits at byte
    /// `(r * blocksPerRow + b) * blockBytes`.
    ///
    /// A layout groups `rowsPerTile` consecutive rows and `blocksPerTile`
    /// consecutive blocks of each of those rows into one tile, stores tiles in
    /// (row group, block chunk) order, and inside a tile stores row-major
    /// (row in tile, block in chunk). With `planes`, the tile is additionally
    /// split by block field: all of the tile's field-0 bytes (in tile block
    /// order), then all field-1 bytes, and so on (`QuantBlockFormat.fields`).
    ///
    /// `TileLayout.gguf` (1, 1, no planes) is the identity. The permutation is
    /// defined by `packedOffset(...)`; the reference repacker, the production
    /// writer (W1) and every kernel (W2) implement exactly this function.
    /// Which layout each quant type uses on each GPU family is
    /// `KernelCalibration`'s choice and is recorded per tensor in the manifest.
    public struct TileLayout: Codable, Hashable, Sendable {
        public var rowsPerTile: Int
        public var blocksPerTile: Int
        public var planes: Bool

        public init(rowsPerTile: Int, blocksPerTile: Int, planes: Bool) {
            self.rowsPerTile = rowsPerTile
            self.blocksPerTile = blocksPerTile
            self.planes = planes
        }

        /// The identity layout: plain GGUF row-major blocks.
        public static let gguf = TileLayout(rowsPerTile: 1, blocksPerTile: 1, planes: false)

        /// Whether this layout can hold a `rows` x `blocksPerRow` matrix: both
        /// tile dimensions must divide the matrix exactly (no padding, so the
        /// packed size equals the GGUF size). A writer stores a tensor that
        /// fails this check in `.gguf` and records that in the manifest.
        public func fits(rows: Int, blocksPerRow: Int) -> Bool {
            rowsPerTile >= 1 && blocksPerTile >= 1
                && rows % rowsPerTile == 0 && blocksPerRow % blocksPerTile == 0
        }

        /// Normative definition of the permutation: the packed byte offset of
        /// byte `byteInBlock` of block `block` in row `row`.
        /// Precondition: `fits(rows:blocksPerRow:)`, `0 <= byteInBlock < format.bytes`.
        public func packedOffset(row: Int, block: Int, byteInBlock: Int,
                                 blocksPerRow: Int, format: QuantBlockFormat) -> Int {
            let r = rowsPerTile, k = blocksPerTile, bb = format.bytes
            let chunksPerRow = blocksPerRow / k
            let tileIndex = (row / r) * chunksPerRow + (block / k)
            let tileBytes = r * k * bb
            let blockInTile = (row % r) * k + (block % k)
            let tileBase = tileIndex * tileBytes
            guard planes else { return tileBase + blockInTile * bb + byteInBlock }
            var fieldStart = 0
            var planeBase = 0
            for field in format.fields {
                if byteInBlock < fieldStart + field.bytes {
                    return tileBase + planeBase + blockInTile * field.bytes + (byteInBlock - fieldStart)
                }
                fieldStart += field.bytes
                planeBase += r * k * field.bytes
            }
            preconditionFailure("byteInBlock \(byteInBlock) outside a \(bb)-byte block")
        }

        /// GGUF byte offset of the same byte (the identity layout's offset).
        public static func ggufOffset(row: Int, block: Int, byteInBlock: Int,
                                      blocksPerRow: Int, format: QuantBlockFormat) -> Int {
            (row * blocksPerRow + block) * format.bytes + byteInBlock
        }
    }
}
