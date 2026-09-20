import CryptoKit
import Foundation
import Metal
import Testing
@testable import Mference

extension GemmaQATInstalledExecutionTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["MFERENCE_GEMMA_QAT_PREFILL_ATTENTION_REFERENCE"] != nil))
    func installedPrefillAttentionMatchesFrozenSourceValues() throws {
        let env = ProcessInfo.processInfo.environment
        let root = URL(fileURLWithPath: try #require(env["MFERENCE_GEMMA_QAT_PREFILL_ATTENTION_REFERENCE"]))
        func frozen(_ name: String, _ digest: String) throws -> Data {
            let data = try Data(contentsOf: root.appendingPathComponent(name), options: .mappedIfSafe)
            try #require(SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() == digest)
            return data
        }
        struct Entry: Decodable { let position: Int, layer: Int, stage: String, offset: Int, count: Int }
        struct Trace: Decodable { let entries: [Entry] }
        struct Case: Decodable {
            let layer: Int, headDim: Int, start: Int, queryCount: Int, window: Int, offset: Int, count: Int
            let inputSha256: [String]
        }
        let entries = try JSONDecoder().decode(Trace.self, from: frozen("trace.json",
            "176709159a8174ffaafc51f29e8cf749c6d805ac07c47de956a34f4b9391873c")).entries
        let native = try frozen("native.f32", "272545a9817f292134db91fd68b14bba779e6042ef6ea8b3d7ecd4a71d7e3be6")
        let expected = try frozen("prefill-attention.f16", "7a33f54a368c3bcab3481fa4c6f18bccda07fd921948ffa352258f390cb664cb")
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let cases = try decoder.decode([Case].self, from: frozen("prefill-attention.json",
            "b1b0008f7558ab0a52cd0d7c4b4418f489127787a4e5ff16f2a1b93211430d68"))
        try #require(cases.count == 41)
        let manifest = try ManifestReader.load(directoryURL: URL(fileURLWithPath:
            try #require(env["MFERENCE_GEMMA_QAT_GTURBO"])), expecting: .gemma4_26B_A4B)
        try #require(manifest.modelID == CheckpointIdentity.gemma4QAT)
        let context = try MetalContext()
        let attention = try PrefillAttention(context: context, gemmaQATMaxContext: 4100)
        func snapshot(_ layer: Int, _ position: Int, _ stage: String) throws -> [Float16] {
            let entry = try #require(entries.first { $0.layer == layer && $0.position == position % 6 && $0.stage == stage })
            return try native.withUnsafeBytes { bytes in
                try (0..<entry.count).map {
                    let value = bytes.loadUnaligned(fromByteOffset: entry.offset + $0 * 4, as: Float.self)
                    let half = Float16(value)
                    try #require(Float(half) == value)
                    return half
                }
            }
        }
        func buffer(_ data: Data) throws -> MTLBuffer {
            try #require(data.withUnsafeBytes {
                context.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
            })
        }
        let guardBytes = Data(repeating: 0xa5, count: 64)
        var results: [String: Int] = [:]
        var total = 0, totalDifferences = 0
        for item in cases {
            let end = item.start + item.queryCount
            let kvHeads = item.headDim == 256 ? 8 : 2
            let width = item.headDim * 16, kvWidth = item.headDim * kvHeads
            let queries = try (item.start..<end).map { try snapshot(item.layer, $0, "query") }
            let histories = try ["key", "value"].map { stage in
                try (0..<end).map { try snapshot(item.layer, $0, stage) }
            }
            for (rows, digest) in zip([queries] + histories, item.inputSha256) {
                let bytes = rows.flatMap { $0 }.withUnsafeBytes { Data($0) }
                try #require(SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined() == digest)
            }
            for padded in [false, true] {
                for ring in (item.headDim == 256 ? [false, true] : [false]) {
                    let capacity = ring ? min(end, item.window + 128) : 0
                    let qStride = width + (padded ? 16 : 0)
                    let kvStride = kvWidth + (padded ? 16 : 0)
                    let qRows = queries.flatMap { $0 + [Float16](repeating: .nan, count: qStride - width) }
                    let q = try buffer(guardBytes + qRows.withUnsafeBytes { Data($0) } + guardBytes)
                    let inputs = try histories.map { history -> MTLBuffer in
                        var physical = [Float16](repeating: .nan, count: (ring ? capacity : end) * kvStride)
                        for position in 0..<end {
                            let offset = (ring ? position % capacity : position) * kvStride
                            physical.replaceSubrange(offset..<offset + kvWidth, with: history[position])
                        }
                        return try buffer(guardBytes + physical.withUnsafeBytes { Data($0) } + guardBytes)
                    }
                    let outBytes = Data(repeating: 0xa5, count: item.queryCount * qStride * 2)
                    let output = try buffer(guardBytes + outBytes + guardBytes)
                    let cb = try #require(context.queue.makeCommandBuffer())
                    let params = PrefillAttentionParams(startPosition: UInt32(item.start),
                        queryCount: UInt32(item.queryCount), headDim: UInt32(item.headDim),
                        numQHeads: 16, numKVHeads: UInt32(kvHeads), kvValidCount: UInt32(end),
                        slidingWindow: UInt32(item.window), kvTokenStrideElements: UInt32(kvStride),
                        qTokenStrideElements: UInt32(qStride), oTokenStrideElements: UInt32(qStride), scale: 1)
                    attention.encodeCausal(commandBuffer: cb, q: q, qOffset: 64,
                        k: inputs[0], kOffset: 64, v: inputs[1], vOffset: 64,
                        out: output, outOffset: 64, params: params, kvRingCapacity: UInt32(capacity),
                        path: padded ? .fullTensorOps2DPreferred : .causalTiled)
                    cb.commit()
                    cb.waitUntilCompleted()
                    try #require(cb.status == .completed, "\(String(describing: cb.error))")
                    let values = output.contents().advanced(by: 64).assumingMemoryBound(to: UInt16.self)
                    var differences = 0
                    for row in 0..<item.queryCount {
                        expected.withUnsafeBytes { bytes in
                            for d in 0..<width {
                                if values[row * qStride + d] != bytes.loadUnaligned(
                                    fromByteOffset: item.offset + (row * width + d) * 2, as: UInt16.self) {
                                    differences += 1
                                }
                            }
                        }
                        for d in width..<qStride { try #require(values[row * qStride + d] == 0xa5a5) }
                    }
                    try #require(Data(bytes: output.contents(), count: 64) == guardBytes)
                    try #require(Data(bytes: output.contents().advanced(by: 64 + outBytes.count), count: 64) == guardBytes)
                    results["\(item.layer)/\(item.start)/\(padded ? "padded" : "packed")/\(ring ? "ring" : "linear")"] = differences
                    totalDifferences += differences
                    total += item.count
                }
            }
        }
        try JSONSerialization.data(withJSONObject: ["values": total, "different": totalDifferences,
            "cases": results], options: [.prettyPrinted, .sortedKeys])
            .write(to: root.appendingPathComponent("prefill-attention-native.json"))
        print("[qat-prefill-attention] values=\(total) different=\(totalDifferences) variants=\(results.count)")
        #expect(totalDifferences == 0)
    }
}
