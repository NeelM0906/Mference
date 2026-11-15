import Foundation
import Testing
@testable import MferenceRepackCore

/// Plans the real Unsloth UD-IQ3_XXS layout from the committed tensor table.
@Suite
struct FlashNextGGUFPlannerTests {
    private static func realPlan(outputDir: String = "/out") throws -> (RepackPlan, [GGUF.TensorInfo]) {
        let (md, tensors) = try GGUFArchInfoTests.load()
        let arch = try ArchInfo.fromGGUFQwen4Exp(metadata: md, tensors: tensors, path: "unsloth")
        // Regroup by shard using the table's own shard numbers (1-based).
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: root.appendingPathComponent(
            "docs/families/qwen38flashnext-unsloth-ud-iq3_xxs.tensors.json"))) as? [String: Any]
        let rows = try #require(json?["tensors"] as? [[String: Any]])
        var shardOf: [String: Int] = [:]
        for r in rows { shardOf[r["name"] as! String] = (r["shard"] as! NSNumber).intValue }
        var headers: [GGUF.Header] = []
        var paths: [String] = []
        for shard in 1...4 {
            let ts = tensors.filter { shardOf[$0.name] == shard }
            headers.append(GGUF.Header(version: 3, keys: [], metadata: shard == 1 ? md : [:], tensors: ts,
                                       alignment: 32, headerBytes: 0, dataOffset: 0))
            paths.append("UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-0000\(shard)-of-00004.gguf")
        }
        return (try FlashNextGGUFPlanner.plan(headers: headers, shardPaths: paths, arch: arch,
                                              outputDir: outputDir, matchedModelID: nil), tensors)
    }

    @Test func perLayerExpertStridesMatchTheU1Record() throws {
        let (plan, _) = try Self.realPlan()
        #expect(plan.layers.count == 48)
        var byStride: [UInt64: [Int]] = [:]
        for l in plan.layers { byStride[l.expertStride, default: []].append(l.layerIndex) }
        #expect(byStride[1_523_712] == [8, 10, 13])            // IQ2_S / Q2_0
        #expect(byStride[2_670_592] == [47])                   // IQ4_XS / IQ4_NL
        #expect(byStride[2_179_072]?.count == 21)              // IQ3_XXS / IQ4_NL
        #expect(byStride[1_982_464]?.count == 23)              // IQ2_S / IQ4_NL
        for l in plan.layers {
            #expect(l.expertsPerLayer == 512 && l.expertStride % 16_384 == 0)
            #expect(l.subTensors.map(\.role) == ["gate", "up", "down"])
            #expect(l.subTensors.allSatisfy { $0.offsetInExpertBlob % 256 == 0 })
        }
    }

    @Test func pleTableBecomesAnIQ4NLRowPool() throws {
        let (plan, _) = try Self.realPlan()
        let pool = try #require(plan.plePools.first)
        #expect(plan.plePools.count == 1)
        #expect(pool.layerIndex == 1 && pool.storage == .iq4nl && pool.rowDim == 160)
        #expect(pool.rowWeightBytes == 90 && pool.rowStride == 90 && pool.rowsPerBlock == 182)
        #expect(pool.totalRows == 320_001_536)
        #expect(pool.fileSize == 1_758_251 * 16_384)
        #expect(pool.blockTransform == .bf16RowBlocks(rowSourceBytes: 90, rowsPerBlock: 182, blockStride: 16_384))
    }

    @Test func everySourceByteIsCopiedOnceVerbatim() throws {
        let (plan, tensors) = try Self.realPlan()
        let resident = plan.resident.entries
        #expect(resident.count == 1224 - 48 * 3 - 1)
        #expect(resident.allSatisfy { $0.fileOffset % 256 == 0 && $0.weightTransform == .identity })
        let q6 = try #require(resident.first { $0.name == "token_embd.weight" })
        #expect(q6.dtype == 0x40 | 14 && q6.logicalShape4 == [248_320, 2560, 0, 0])

        let copies = try RangeCopyPlanner.plan(repackPlan: plan, rangeChunkBytes: 64 << 20)
        let sourceBytes = copies.scalarCopies.reduce(UInt64(0)) { $0 + $1.size }
        #expect(sourceBytes == tensors.reduce(UInt64(0)) { $0 + $1.sizeBytes })
        #expect(sourceBytes == 83_615_413_760)
    }
}
