import Foundation
import Testing
@testable import Mference
import MferenceValidationSupport

/// The IQ4_NL PLE row pool of a WS4 GGUF install (U1 §3c): the runtime row
/// decoder against llama.cpp's order, the `iq4nl` geometry rules, and real row
/// reads from a small pool laid out like the installed one (160-value rows of
/// five native blocks, page blocks of whole rows, slack filled with 0xAB).
@Suite struct PleRowPoolIQ4NLTests {
    static let rowDim = 160, rowBytes = 90, blockStride = 16_384
    static let rowsPerBlock = blockStride / rowBytes  // 182, as installed

    static func layer(rows: Int, overrides: [String: Any] = [:]) -> [String: Any] {
        let blocks = (rows + rowsPerBlock - 1) / rowsPerBlock
        var layer: [String: Any] = [
            "layer": 1, "file": "ple/layer_01_ngram_rows.bin", "rows": rows, "rowDim": rowDim,
            "storage": "iq4nl", "weightBits": 4, "groupSize": 32,
            "rowWeightBytes": rowBytes, "rowScaleBytes": 0, "rowBiasBytes": 0, "rowStride": rowBytes,
            "rowsPerBlock": rowsPerBlock, "blockStride": blockStride,
            "fileSize": blocks * blockStride,
            "shards": [["shard": 0, "rows": rows, "offset": 0, "size": blocks * blockStride]],
        ]
        for (k, v) in overrides { layer[k] = v }
        return layer
    }

    static func decode(_ layer: [String: Any]) throws -> ManifestPlePoolLayer {
        try JSONDecoder().decode(ManifestPlePoolLayer.self, from: JSONSerialization.data(withJSONObject: layer))
    }

    @Test func iq4NLRowDecoderMatchesReferenceBits() {
        var rng = SeedTree(0x5052_4C34).key("iq4nl-rows")
        let bytes = randomGGUFBlocks(.iq4_nl, count: 2_000, rng: &rng)
        let (got, want) = bytes.withUnsafeBytes {
            (GGUFBlockDecode.dequantizeIQ4NL($0), GGUFReferenceDequant.iq4NL($0))
        }
        #expect(got.count == want.count)
        #expect(zip(got, want).allSatisfy { $0.bitPattern == $1.bitPattern })
    }

    @Test func geometryAcceptsTheInstalledShapeAndRefusesOthers() throws {
        let g = try PleRowPoolGeometry(layer: Self.decode(Self.layer(rows: 1_000)), hiddenSize: 2560)
        #expect(g.storage == .iq4nl)
        #expect(g.ngramHeads == 16)
        for bad: [String: Any] in [["rowWeightBytes": 80, "rowStride": 80], ["rowScaleBytes": 10, "rowStride": 100],
                                   ["groupSize": 64], ["weightBits": 8], ["rowDim": 150]] {
            #expect(throws: ModelError.self, "\(bad)") {
                _ = try PleRowPoolGeometry(layer: Self.decode(Self.layer(rows: 1_000, overrides: bad)), hiddenSize: 2560)
            }
        }
    }

    @Test func readsRowsAcrossBlockBoundaries() throws {
        let rows = 2 * Self.rowsPerBlock + 17
        var rng = SeedTree(0x5052_4C35).key("iq4nl-pool")
        let blocks = (rows + Self.rowsPerBlock - 1) / Self.rowsPerBlock
        var file = [UInt8](repeating: 0xAB, count: blocks * Self.blockStride)
        var expected = [[Float]]()
        for r in 0..<rows {
            let row = randomGGUFBlocks(.iq4_nl, count: Self.rowDim / 32, rng: &rng)
            let at = (r / Self.rowsPerBlock) * Self.blockStride + (r % Self.rowsPerBlock) * Self.rowBytes
            file.replaceSubrange(at..<(at + Self.rowBytes), with: row)
            expected.append(row.withUnsafeBytes { GGUFReferenceDequant.iq4NL($0) })
        }
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("ple-iq4nl-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try FileManager.default.createDirectory(at: dir.appendingPathComponent("ple"), withIntermediateDirectories: true)
        try Data(file).write(to: dir.appendingPathComponent("ple/layer_01_ngram_rows.bin"))

        let geometry = try PleRowPoolGeometry(layer: Self.decode(Self.layer(rows: rows)), hiddenSize: 2560)
        let pool = try PleRowPool(directoryURL: dir, geometry: geometry, cacheRows: 8)
        for r in [0, 1, Self.rowsPerBlock - 1, Self.rowsPerBlock, 2 * Self.rowsPerBlock + 16, 5, 5] {
            let got = try pool.readRow(r)
            #expect(got.map(\.bitPattern) == expected[r].map(\.bitPattern), "row \(r)")
        }
        #expect(throws: ModelError.self) { _ = try pool.readRow(rows) }
    }
}
