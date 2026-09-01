import Testing
import Foundation
import Metal
@testable import Mference

/// The PLE n-gram row pool: geometry validation, block/offset arithmetic, and
/// the bounded LFU row cache in front of it.
///
/// The fixture pool (`FlashNextToySynthetic.Pool`) is deliberately tiny but
/// shaped exactly like production: rows tile page-aligned blocks, the last
/// block of each shard is short, shard regions are consecutive, and the slack
/// bytes at the end of every block are filled with `0xAB` — so a reader that
/// mis-computes a block offset reads recognizable garbage rather than a
/// plausible neighbouring row.
@Suite struct PleRowPoolTests {

    typealias Pool = FlashNextToySynthetic.Pool

    private static func openPool(cacheRows: Int = 64) throws -> (PleRowPool, URL) {
        let dir = try FlashNextToySynthetic.write()
        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(directoryURL: dir, device: device,
                                   expecting: .qwen38FlashNextToy())
        let pool = try model.openPleRowPool(layer: FlashNextToySynthetic.pleLayer,
                                            cacheRows: cacheRows)
        return (pool, dir)
    }

    // MARK: - Geometry

    @Test func geometryDerivesShardPrefixesAndHeadCount() throws {
        let dir = try FlashNextToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: dir) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(directoryURL: dir, device: device,
                                   expecting: .qwen38FlashNextToy())
        let geometry = try model.plePoolGeometry(layer: FlashNextToySynthetic.pleLayer)

        #expect(geometry.rows == Pool.totalRows)
        #expect(geometry.rowDim == Pool.rowDim)
        #expect(geometry.storage == .bf16)
        #expect(geometry.shards.count == Pool.shardCount)
        #expect(geometry.shards.map(\.firstRow) == [0, Pool.rowsPerShard])
        // 16 heads x 160 = 2560 in production; 4 x 16 = 64 here.
        #expect(geometry.ngramHeads == Pool.ngramHeads)
    }

    /// Row addressing is a division and two multiplies against the manifest's
    /// own numbers. Checking every row against an independently written
    /// expression is what keeps a future geometry change from silently
    /// shifting every lookup.
    @Test func rowOffsetsMatchTheBlockArithmetic() throws {
        let dir = try FlashNextToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: dir) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(directoryURL: dir, device: device,
                                   expecting: .qwen38FlashNextToy())
        let geometry = try model.plePoolGeometry(layer: FlashNextToySynthetic.pleLayer)

        for row in 0..<Pool.totalRows {
            let shard = row / Pool.rowsPerShard
            let local = row % Pool.rowsPerShard
            let expected = UInt64(shard * Pool.shardBytes
                + (local / Pool.rowsPerBlock) * Pool.blockStride
                + (local % Pool.rowsPerBlock) * Pool.rowStride)
            #expect(try geometry.fileOffset(row: row) == expected, "row \(row)")
        }
        #expect(throws: ModelError.plePoolRowOutOfRange(row: Pool.totalRows,
                                                        rows: Pool.totalRows)) {
            _ = try geometry.fileOffset(row: Pool.totalRows)
        }
        #expect(throws: ModelError.self) { _ = try geometry.fileOffset(row: -1) }
    }

    @Test func geometryRefusesAnUnknownPoolKind() throws {
        let dir = try FlashNextToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifestURL = dir.appendingPathComponent("manifest.json")
        var root = try JSONSerialization.jsonObject(
            with: Data(contentsOf: manifestURL)) as! [String: Any]
        var pool = root["plePool"] as! [String: Any]
        pool["kind"] = "rowLookupPoolV2"
        root["plePool"] = pool
        try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys])
            .write(to: manifestURL)

        #expect(throws: ModelError.self) {
            _ = try ManifestReader.load(directoryURL: dir,
                                        expecting: .qwen38FlashNextToy())
        }
    }

    /// Every self-consistency relation the row reader depends on is a load-time
    /// gate, not a per-read assumption.
    @Test func geometryRefusesInconsistentBlockTiling() throws {
        var block = FlashNextToySynthetic.plePoolBlock()
        var layer = (block["layers"] as! [[String: Any]])[0]

        func decode(_ layer: [String: Any]) throws -> ManifestPlePoolLayer {
            let data = try JSONSerialization.data(withJSONObject: layer)
            return try JSONDecoder().decode(ManifestPlePoolLayer.self, from: data)
        }
        // Baseline decodes and validates.
        _ = try PleRowPoolGeometry(layer: try decode(layer), hiddenSize: 64)

        // Rows of this stride cannot fit that many per block.
        var tooManyRows = layer
        tooManyRows["rowsPerBlock"] = Pool.rowsPerBlock + 1
        #expect(throws: ModelError.self) {
            _ = try PleRowPoolGeometry(layer: try decode(tooManyRows), hiddenSize: 64)
        }
        // The record and the stride must agree.
        var badStride = layer
        badStride["rowStride"] = Pool.rowStride + 1
        #expect(throws: ModelError.self) {
            _ = try PleRowPoolGeometry(layer: try decode(badStride), hiddenSize: 64)
        }
        // Shard rows must sum to the declared total.
        var shortRows = layer
        shortRows["rows"] = Pool.totalRows + 1
        #expect(throws: ModelError.self) {
            _ = try PleRowPoolGeometry(layer: try decode(shortRows), hiddenSize: 64)
        }
        // A BF16 pool that claims scale/bias companions is not a BF16 pool.
        var withCompanions = layer
        withCompanions["rowScaleBytes"] = 8
        #expect(throws: ModelError.self) {
            _ = try PleRowPoolGeometry(layer: try decode(withCompanions), hiddenSize: 64)
        }
        // 160 does not divide 2560 -> the head split would be fractional.
        #expect(throws: ModelError.self) {
            _ = try PleRowPoolGeometry(layer: try decode(layer), hiddenSize: 70)
        }
        layer["rowDim"] = Pool.rowDim
        block["layers"] = [layer]
    }

    // MARK: - Row reads

    @Test func readsEveryRowIncludingBlockBoundariesAndTails() throws {
        let (pool, dir) = try Self.openPool()
        defer { try? FileManager.default.removeItem(at: dir) }

        for row in 0..<Pool.totalRows {
            let values = try pool.readRow(row)
            #expect(values.count == Pool.rowDim)
            let expected = (0..<Pool.rowDim).map { Pool.value(row: row, column: $0) }
            #expect(values == expected, "row \(row)")
        }
    }

    /// The rows that sit at the seams: the first and last of a block, the
    /// first row of the short tail block, the last row of a shard, and the
    /// first row of the next shard. These are exactly the indices a
    /// division-vs-modulo slip gets wrong.
    @Test func seamRowsReadCorrectly() throws {
        let (pool, dir) = try Self.openPool()
        defer { try? FileManager.default.removeItem(at: dir) }

        let lastRowOfFirstBlock = Pool.rowsPerBlock - 1
        let firstRowOfSecondBlock = Pool.rowsPerBlock
        let firstRowOfTailBlock = 2 * Pool.rowsPerBlock
        let lastRowOfFirstShard = Pool.rowsPerShard - 1
        let firstRowOfSecondShard = Pool.rowsPerShard
        let lastRow = Pool.totalRows - 1

        for row in [0, lastRowOfFirstBlock, firstRowOfSecondBlock,
                    firstRowOfTailBlock, lastRowOfFirstShard,
                    firstRowOfSecondShard, lastRow] {
            let values = try pool.readRow(row)
            #expect(values[0] == Pool.value(row: row, column: 0), "row \(row) head")
            #expect(values[Pool.rowDim - 1]
                    == Pool.value(row: row, column: Pool.rowDim - 1), "row \(row) tail")
        }
    }

    /// A token's PLE embedding is `ngramHeads` rows concatenated in head order.
    @Test func embeddingConcatenatesOneRowPerNgramHead() throws {
        let (pool, dir) = try Self.openPool()
        defer { try? FileManager.default.removeItem(at: dir) }

        let rows = [3, 11, 21, 39]
        let embedding = try pool.readEmbedding(rows: rows)
        #expect(embedding.count == Pool.ngramHeads * Pool.rowDim)
        for (head, row) in rows.enumerated() {
            let slice = Array(embedding[(head * Pool.rowDim)..<((head + 1) * Pool.rowDim)])
            #expect(slice == (0..<Pool.rowDim).map { Pool.value(row: row, column: $0) })
        }
        #expect(throws: ModelError.self) { _ = try pool.readEmbedding(rows: [0, 1]) }
    }

    @Test func readingPastTheLastRowRefuses() throws {
        let (pool, dir) = try Self.openPool()
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(throws: ModelError.plePoolRowOutOfRange(row: Pool.totalRows,
                                                        rows: Pool.totalRows)) {
            _ = try pool.readRow(Pool.totalRows)
        }
    }

    /// Opening the pool cross-checks the installed `ngram_heads_*` tables
    /// against the geometry: a table of the wrong length would otherwise show
    /// up only as a wrong row index at decode.
    @Test func openingRefusesHeadTablesThatDoNotSpanThePool() throws {
        let dir = try FlashNextToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifestURL = dir.appendingPathComponent("manifest.json")
        var root = try JSONSerialization.jsonObject(
            with: Data(contentsOf: manifestURL)) as! [String: Any]
        var pool = root["plePool"] as! [String: Any]
        var layer = (pool["layers"] as! [[String: Any]])[0]
        // Halve the row count the manifest claims: the head tables now run off
        // the end of the pool. Shard rows are halved to keep the sum honest,
        // so the failure is the head-span check and not the tiling check.
        layer["rows"] = Pool.totalRows / 2
        layer["shards"] = (layer["shards"] as! [[String: Any]]).map { shard -> [String: Any] in
            var shard = shard
            shard["rows"] = Pool.rowsPerShard / 2
            return shard
        }
        pool["layers"] = [layer]
        root["plePool"] = pool
        try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys])
            .write(to: manifestURL)

        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(directoryURL: dir, device: device,
                                   expecting: .qwen38FlashNextToy())
        #expect(throws: ModelError.self) {
            _ = try model.openPleRowPool(layer: FlashNextToySynthetic.pleLayer)
        }
    }

    // MARK: - LFU row cache

    @Test func rereadingARowIsServedFromTheCache() throws {
        let (pool, dir) = try Self.openPool()
        defer { try? FileManager.default.removeItem(at: dir) }

        _ = try pool.readRow(7)
        #expect(pool.cacheStatistics.misses == 1)
        #expect(pool.cacheStatistics.hits == 0)
        for _ in 0..<5 { _ = try pool.readRow(7) }
        #expect(pool.cacheStatistics.hits == 5)
        #expect(pool.cacheStatistics.misses == 1)
        #expect(pool.cacheStatistics.resident == 1)
        #expect(pool.cachedUseCount(of: 7) == 6)
    }

    /// The cache is bounded and evicts by use count, not by recency: a row read
    /// once loses to a row read many times, however recently.
    @Test func evictionDropsTheLeastFrequentlyUsedRow() throws {
        let (pool, dir) = try Self.openPool(cacheRows: 4)
        defer { try? FileManager.default.removeItem(at: dir) }

        // Rows 0-2 are hot; row 3 is touched once.
        for _ in 0..<10 { for row in 0..<3 { _ = try pool.readRow(row) } }
        _ = try pool.readRow(3)
        #expect(pool.residentRows() == [0, 1, 2, 3])
        #expect(pool.cacheStatistics.evictions == 0)

        // The fifth distinct row must evict the coldest, which is row 3.
        _ = try pool.readRow(4)
        #expect(pool.cacheStatistics.evictions == 1)
        #expect(pool.residentRows() == [0, 1, 2, 4])
        for row in 0..<3 {
            #expect(pool.cachedUseCount(of: row) == 10, "hot row \(row) was evicted")
        }
    }

    /// The cache never grows past its budget, whatever the access pattern.
    @Test func cacheStaysWithinItsRowBudget() throws {
        let (pool, dir) = try Self.openPool(cacheRows: 5)
        defer { try? FileManager.default.removeItem(at: dir) }

        for row in 0..<Pool.totalRows {
            _ = try pool.readRow(row)
            #expect(pool.cacheStatistics.resident <= 5)
        }
        #expect(pool.cacheStatistics.resident == 5)
        #expect(pool.cacheStatistics.evictions == Pool.totalRows - 5)
        // Evicted rows still read correctly — the cache is transparent.
        #expect(try pool.readRow(0) == (0..<Pool.rowDim).map {
            Pool.value(row: 0, column: $0)
        })
    }

    /// A one-row cache is legal and still correct: the degenerate case a
    /// budget-tuning experiment will reach for.
    @Test func aSingleRowCacheStaysCorrect() throws {
        let (pool, dir) = try Self.openPool(cacheRows: 1)
        defer { try? FileManager.default.removeItem(at: dir) }
        for row in [5, 6, 5, 6, 5] {
            #expect(try pool.readRow(row) == (0..<Pool.rowDim).map {
                Pool.value(row: row, column: $0)
            })
        }
        #expect(pool.cacheStatistics.resident == 1)
    }

    /// The production default is a deliberate, documented size, not an
    /// accident: ~20 MB at the installed 320-byte row.
    @Test func defaultCacheBudgetIsAboutTwentyMegabytes() {
        let productionRowStride = 320
        let bytes = PleRowPool.defaultCacheRows * productionRowStride
        #expect(PleRowPool.defaultCacheRows == 65_536)
        #expect(bytes == 20_971_520)
    }
}
