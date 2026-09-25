import Foundation
import Metal
import Testing
@testable import Mference

extension GemmaQATKernelTests {
    @Test func nativeGreedyHeadChoosesTheCorrectGroup32Row() throws {
        let context = try MetalContext()
        for n in [704, 2112, 2816] {
            // Row 0 is all ones, row 1 all twos, other rows zero. Reading the
            // scale table with a group-64 row stride instead makes rows 2/3
            // win. This proves the winning token, not just plausible output.
            let weights = try Self.buffer([UInt8](repeating: 0x11, count: 19 * n / 2), context: context)
            let groupCount = n / 32
            let scaleBits = (0..<19 * groupCount).map { i -> UInt16 in
                i < groupCount ? 0x3F80 : (i < 2 * groupCount ? 0x4000 : 0)
            }
            let scales = try Self.buffer(scaleBits, context: context)
            let biases = try Self.buffer([UInt16](repeating: 0, count: scaleBits.count), context: context)
            let hidden = try Self.buffer([Float16](repeating: 1, count: n), context: context)
            let norm = try Self.buffer([UInt16](repeating: 0x3F80, count: n), context: context)
            let token = try Self.buffer([UInt32.max], context: context)
            let head = try LMHeadChainInt4(context: context, maxD: n, maxVocab: 19, groupSize: 32)
            for vocab in [19, 17] {
                let cb = try #require(context.queue.makeCommandBuffer())
                head.encodeGreedyDecode(commandBuffer: cb, hidden: hidden, normWeight: norm,
                    weights: weights, scales: scales, biases: biases, outToken: token,
                    d: UInt32(n), vocab: UInt32(vocab), rmsEps: 0)
                cb.commit()
                cb.waitUntilCompleted()
                try #require(cb.status == .completed, "\(String(describing: cb.error))")
                #expect(token.contents().load(as: UInt32.self) == 1, "N=\(n), vocab=\(vocab)")
            }
        }
    }

    @Test func nativeAffinePrefillRowsAndTails() throws {
        let context = try MetalContext()
        let embed = try PrefillEmbedLookupInt4(context: context, groupSize: 32)
        let projections = try [false, true].map {
            try PrefillInt4QMM(context: context, decodeOrder: $0, groupSize: 32)
        }
        let tokens: [UInt32] = [10, 0, 9, 1, 2, 7, 3]
        let tokenBuffer = try Self.buffer(tokens, context: context, offset: 4)
        for n in [32, 96, 704, 2112, 2816] {
            let matrix = Matrix(rows: 11, columns: n, groupSize: 32)
            let weights = try Self.buffer(matrix.packed, context: context, offset: 2)
            let scales = try Self.buffer(matrix.scales, context: context, offset: 6)
            let biases = try Self.buffer(matrix.biases, context: context, offset: 10)
            let inputs = (0..<7 * n).map { Float16(Float(($0 * 3) % 17 - 8) / 32) }
            let x = try Self.buffer(inputs, context: context, offset: 8)
            let rows = try Self.buffer([Float16](repeating: .nan, count: 7 * n), context: context, offset: 4)
            let outputs = try projections.map { _ in
                try Self.buffer([Float16](repeating: .nan, count: 7 * 11), context: context, offset: 4)
            }
            let cb = try #require(context.queue.makeCommandBuffer())
            embed.encode(commandBuffer: cb, table: weights, tableOffset: 2,
                scales: scales, scalesOffset: 6, biases: biases, biasesOffset: 10,
                tokens: tokenBuffer, tokensOffset: 4, out: rows, outOffset: 4,
                t: 7, d: UInt32(n), outScale: 2)
            for (projection, output) in zip(projections, outputs) {
                projection.encode(commandBuffer: cb, weights: weights, weightsOffset: 2,
                    scales: scales, scalesOffset: 6, biases: biases, biasesOffset: 10,
                    x: x, xOffset: 8, y: output, yOffset: 4, t: 7, n: 11, k: n)
            }
            cb.commit()
            cb.waitUntilCompleted()
            try #require(cb.status == .completed, "\(String(describing: cb.error))")
            let expectedRows = tokens.flatMap { token in
                (0..<n).map { Float16(matrix.value(row: Int(token), column: $0) * 2) }
            }
            let actualRows = rows.contents().advanced(by: 4).assumingMemoryBound(to: Float16.self)
            #expect(Array(UnsafeBufferPointer(start: actualRows, count: 7 * n)) == expectedRows,
                    "batched embedding N=\(n)")
            let expected = (0..<7).flatMap { t in
                (0..<11).map { r in
                    Float16((0..<n).reduce(0.0) {
                        $0 + matrix.value(row: r, column: $1) * Double(inputs[t * n + $1])
                    })
                }
            }
            for (mode, output) in outputs.enumerated() {
                let actual = output.contents().advanced(by: 4).assumingMemoryBound(to: Float16.self)
                #expect(Array(UnsafeBufferPointer(start: actual, count: 77)) == expected,
                        "batched projection mode=\(mode), N=\(n)")
            }
        }
    }

    @Test func nativeAffineFusedQKVAndOutputHeads() throws {
        let context = try MetalContext()
        for n in [96, 704, 2112, 2816] {
            let matrix = Matrix(rows: 19, columns: n, groupSize: 32)
            let weights = try Self.buffer(matrix.packed, context: context, offset: 2)
            let scales = try Self.buffer(matrix.scales, context: context, offset: 6)
            let biases = try Self.buffer(matrix.biases, context: context, offset: 10)
            let ones = try Self.buffer([Float16](repeating: 1, count: 3 * n), context: context)
            let norm = try Self.buffer([UInt16](repeating: 0x3F80, count: n), context: context, offset: 2)
            let q = try Self.buffer([Float16](repeating: .nan, count: 11), context: context)
            let k = try Self.buffer([Float16](repeating: .nan, count: 5), context: context)
            let v = try Self.buffer([Float16](repeating: .nan, count: 5), context: context)
            let logits = try Self.buffer([Float16](repeating: .nan, count: 19), context: context)
            let token = try Self.buffer([UInt32.max], context: context)
            let qkv = try FusedQKVGEMV(context: context, groupSize: 32)
            let head = try LMHeadChainInt4(context: context, maxD: n, maxVocab: 19, groupSize: 32)
            let prefillHead = try PrefillFinalRowHeadInt4(context: context, maxD: n, groupSize: 32)
            let expected = (0..<19).map { r in
                (0..<n).reduce(0.0) { $0 + matrix.value(row: r, column: $1) }
            }
            for vocab in [19, 17] { // specialized and runtime-sized greedy pipelines
                let cb = try #require(context.queue.makeCommandBuffer())
                qkv.encode(commandBuffer: cb,
                    qWeights: weights, qWeightsOffset: 2, qScales: scales, qScalesOffset: 6,
                    qBiases: biases, qBiasesOffset: 10,
                    kWeights: weights, kWeightsOffset: 2 + 2 * n / 2,
                    kScales: scales, kScalesOffset: 6 + 2 * n / 32 * 2,
                    kBiases: biases, kBiasesOffset: 10 + 2 * n / 32 * 2,
                    vWeights: weights, vWeightsOffset: 2 + 5 * n / 2,
                    vScales: scales, vScalesOffset: 6 + 5 * n / 32 * 2,
                    vBiases: biases, vBiasesOffset: 10 + 5 * n / 32 * 2,
                    x: ones, qOut: q, kOut: k, vOut: v, qRows: 11, kvRows: 5, n: UInt32(n))
                head.encodeGreedyDecode(commandBuffer: cb, hidden: ones,
                    normWeight: norm, normOffset: 2, weights: weights, weightsOffset: 2,
                    scales: scales, scalesOffset: 6, biases: biases, biasesOffset: 10,
                    outToken: token, d: UInt32(n), vocab: UInt32(vocab), rmsEps: 0)
                prefillHead.encodeLogits(commandBuffer: cb, hiddenBlock: ones, row: 2,
                    rowStrideElements: n, normWeight: norm, normWeightOffset: 2,
                    weights: weights, weightsOffset: 2, scales: scales, scalesOffset: 6,
                    biases: biases, biasesOffset: 10, logits: logits,
                    d: UInt32(n), vocab: UInt32(vocab), rmsEps: 0)
                cb.commit()
                cb.waitUntilCompleted()
                try #require(cb.status == .completed, "\(String(describing: cb.error))")
                for (buffer, range) in [(q, 0..<11), (k, 2..<7), (v, 5..<10), (logits, 0..<vocab)] {
                    let actual = buffer.contents().assumingMemoryBound(to: Float16.self)
                    #expect(Array(UnsafeBufferPointer(start: actual, count: range.count)) ==
                            expected[range].map(Float16.init), "N=\(n), rows=\(range)")
                }
                let maximum = try #require(expected.prefix(vocab).max())
                let expectedToken = try #require(expected.prefix(vocab).firstIndex(of: maximum))
                #expect(token.contents().load(as: UInt32.self) == UInt32(expectedToken),
                        "greedy group-32 head N=\(n), vocab=\(vocab)")
            }
        }
    }
}
