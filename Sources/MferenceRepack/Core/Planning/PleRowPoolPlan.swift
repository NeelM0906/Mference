import Foundation

/// Layout of a per-layer n-gram embedding (PLE) **row-lookup pool** — the
/// third instantiation of the fixed-aperture idea, after expert slots and KV
/// pages: a table far too large for RAM whose rows are fetched a handful at a
/// time by a token-derived index.
///
/// # Why a new pool kind
///
/// The source ships the table split across `shard_0 .. shard_{N-1}` BF16
/// tensors. Neither existing `.gturbo` region fits it: the resident file is
/// mapped whole, and the packed-expert pool addresses fixed page-aligned blobs
/// by router index. This pool is strictly **additive** to the on-disk contract
/// (spec 2026-08-08 non-goals allow additive sidecar groups): existing families
/// emit no `ple/` directory and no `plePool` manifest block, so their installs
/// stay byte-identical.
///
/// # Storage: BF16 in practice
///
/// Group-64 quantization needs the row width to be a multiple of 64.
/// Qwen3.8-Flash-Next's table is **160 wide** (16 n-gram heads x 160 =
/// `ple_embed_dim` 2560; verified against the shard headers at `de4b8e4d`), so
/// its rows **cannot** be int4 group-64 quantized cleanly and the pool stores
/// them as BF16. The rule is width-driven rather than hard-coded — a table
/// whose width the group size divides is quantized — but the production width
/// lands on BF16, which is why the installed pool is ~102 GB rather than the
/// ~29 GB the Day-0 dossier estimated.
///
/// # Geometry
///
/// ```
/// row record   = BF16:  rowDim * 2 bytes
///                INT4:  [ rowDim/2 packed | BF16 scales | BF16 biases ]
/// rowStride    = that record, dense (no padding between rows)
/// rowsPerBlock = floor(pageBytes / rowStride)           (at least 1)
/// blockStride  = pageBytes rounded up to hold one block (== pageBytes when
///                rowStride <= pageBytes)
/// ```
///
/// At the production width: `rowStride` 320 B, `rowsPerBlock` 51,
/// `blockStride` 16384, slack 64 B per block (0.4%).
///
/// A block is page-aligned and holds `rowsPerBlock` dense records followed by
/// slack, so **no record ever straddles a page**: one row costs exactly one
/// page fault, and a cached page serves `rowsPerBlock` neighbouring rows. That
/// is the property an LFU row cache needs, and it is why rows are *not*
/// individually page-aligned — at 320 B per row that would inflate the ~102 GB
/// table to ~5 TB.
///
/// Each source shard gets its own page-aligned region, so a shard's rows never
/// straddle a block boundary belonging to another shard. That keeps the install
/// plan to two byte-range copies per shard (the whole-block run, then the tail)
/// instead of one per block — 256 copies rather than 6.3 million — and it keeps
/// row addressing a division rather than a search. The manifest publishes the
/// per-shard region bases and row counts.
///
/// Row `i` of shard `s` lives at:
/// `shardRegionOffset[s] + (i / rowsPerBlock) * blockStride + (i % rowsPerBlock) * rowStride`
struct PleRowPoolShardPlan: Sendable {
    let shardIndex: Int
    let sourceTensor: SourceTensor
    let rows: Int
    /// Page-aligned start of this shard's region inside the pool file.
    let regionOffset: UInt64
    let regionBytes: UInt64
    /// Blocks that are completely filled; each consumes `blockStride` bytes.
    let fullBlocks: Int
    /// Rows in the trailing partial block (0 when the shard divides evenly).
    let tailRows: Int
}

struct PleRowPoolPlan: Sendable {
    /// How a row is stored. Chosen from the table's width, not hard-coded:
    /// group-64 quantization needs `rowDim % 64 == 0`, which Flash-Next's
    /// 160-wide table does not satisfy.
    enum Storage: String, Sendable, Equatable {
        case int4AffineG64
        case bf16
    }

    /// Text layer the PLE module belongs to.
    let layerIndex: Int
    let path: String
    let relativePath: String
    /// Source tensor base name the pool was built from, for the manifest.
    let sourceTensorPrefix: String
    let rowDim: Int
    let storage: Storage
    /// 4 when quantized, 16 when the rows stay BF16.
    let bits: Int
    /// 0 when the rows stay BF16 (no grouping applies).
    let groupSize: Int
    let rowWeightBytes: UInt64
    /// Per companion component (scales; biases match). 0 when BF16.
    let rowCompanionBytes: UInt64
    let rowStride: UInt64
    let rowsPerBlock: Int
    let blockStride: UInt64
    let totalRows: Int
    let fileSize: UInt64
    let shards: [PleRowPoolShardPlan]

    /// BF16 bytes one source row occupies.
    var rowSourceBytes: Int { rowDim * 2 }

    /// Transform for a run of whole blocks: quantize-and-pad, or copy-and-pad.
    var blockTransform: RangeCopyTransform {
        switch storage {
        case .int4AffineG64:
            return .quantizeInt4G64RowBlocks(rowSourceBytes: rowSourceBytes,
                                             rowsPerBlock: rowsPerBlock,
                                             blockStride: blockStride)
        case .bf16:
            return .bf16RowBlocks(rowSourceBytes: rowSourceBytes,
                                  rowsPerBlock: rowsPerBlock,
                                  blockStride: blockStride)
        }
    }

    /// Transform for the trailing partial block, which is dense and sits at a
    /// block base. BF16 rows are byte-identical to their source, so the tail
    /// needs no transform at all.
    var tailTransform: RangeCopyTransform {
        switch storage {
        case .int4AffineG64:
            return .quantizeInt4G64Rows(rowSourceBytes: rowSourceBytes)
        case .bf16:
            return .identity
        }
    }

    /// Build the geometry for one layer's pool from its ordered source shards.
    /// `shards` must be in shard-index order; `rowDim` is the table's row width.
    static func make(layerIndex: Int,
                     outputDir: String,
                     sourceTensorPrefix: String,
                     rowDim: Int,
                     shards: [(index: Int, tensor: SourceTensor, rows: Int)]) throws
        -> PleRowPoolPlan {
        guard rowDim > 0 else {
            throw RepackError.shapeMismatch(
                name: sourceTensorPrefix,
                detail: "PLE n-gram row width must be positive, got \(rowDim)")
        }
        guard !shards.isEmpty else {
            throw RepackError.configurationInvalid(
                detail: "PLE pool for layer \(layerIndex) has no n-gram shards")
        }
        // Quantize only when the group size divides the row width. Flash-Next's
        // 160-wide table does not, so its pool is BF16.
        let storage: Storage = StreamingInt4Quantizer.isQuantizableRowDim(rowDim)
            ? .int4AffineG64
            : .bf16
        let rowWeightBytes: UInt64
        let rowCompanionBytes: UInt64
        switch storage {
        case .int4AffineG64:
            let groups = rowDim / StreamingInt4Quantizer.groupSize
            rowWeightBytes = UInt64(rowDim / 2)
            rowCompanionBytes = UInt64(groups
                * StreamingInt4Quantizer.companionBytesPerGroup)
        case .bf16:
            rowWeightBytes = UInt64(rowDim * 2)
            rowCompanionBytes = 0
        }
        let rowStride = rowWeightBytes + 2 * rowCompanionBytes
        let page = Layout.pageBytes
        let rowsPerBlock = rowStride <= page ? Int(page / rowStride) : 1
        let blockStride = rowStride <= page
            ? page
            : ((rowStride + page - 1) / page) * page

        var shardPlans: [PleRowPoolShardPlan] = []
        shardPlans.reserveCapacity(shards.count)
        var cursor: UInt64 = 0
        var totalRows = 0
        for shard in shards {
            guard shard.rows > 0 else {
                throw RepackError.shapeMismatch(
                    name: shard.tensor.name,
                    detail: "PLE n-gram shard has no rows")
            }
            let blocks = (shard.rows + rowsPerBlock - 1) / rowsPerBlock
            let regionBytes = UInt64(blocks) * blockStride
            let fullBlocks = shard.rows / rowsPerBlock
            let tailRows = shard.rows % rowsPerBlock
            shardPlans.append(PleRowPoolShardPlan(shardIndex: shard.index,
                                                  sourceTensor: shard.tensor,
                                                  rows: shard.rows,
                                                  regionOffset: cursor,
                                                  regionBytes: regionBytes,
                                                  fullBlocks: fullBlocks,
                                                  tailRows: tailRows))
            cursor += regionBytes
            totalRows += shard.rows
        }

        let directory = (outputDir as NSString).appendingPathComponent("ple")
        let name = String(format: "layer_%02d_ngram_rows.bin", layerIndex)
        return PleRowPoolPlan(
            layerIndex: layerIndex,
            path: (directory as NSString).appendingPathComponent(name),
            relativePath: "ple/" + name,
            sourceTensorPrefix: sourceTensorPrefix,
            rowDim: rowDim,
            storage: storage,
            bits: storage == .int4AffineG64 ? 4 : 16,
            groupSize: storage == .int4AffineG64 ? StreamingInt4Quantizer.groupSize : 0,
            rowWeightBytes: rowWeightBytes,
            rowCompanionBytes: rowCompanionBytes,
            rowStride: rowStride,
            rowsPerBlock: rowsPerBlock,
            blockStride: blockStride,
            totalRows: totalRows,
            fileSize: cursor,
            shards: shardPlans)
    }
}
