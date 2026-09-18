import Metal
import Testing
@testable import Mference

@Suite struct Glm53DeviceSelectionTests {
    @Test func selectionMatchesStableCPUAcrossCutoverAndTies() throws {
        let context = try MetalContext()
        let kernels = try Glm53PrefillKernels(context: context, swigluLimit: 0)
        let shapes = [(1, 4, 0, 13), (4, 16, 8, 17), (4, 2048, 2040, 64),
                      (8, 2048, 8192, 65)]
        for (pool, topK, base, rows) in shapes {
            let pools = (base + rows) / pool
            let stride = topK + pool
            for pattern in 0..<3 {
                var scores = [Float](repeating: 0, count: rows * pools)
                for i in scores.indices {
                    if pattern == 1 { scores[i] = Float((i * 37 + 11) % 19 - 9) }
                    if pattern == 2 {
                        switch i % 5 {
                        case 0: scores[i] = .infinity
                        case 1: scores[i] = -.infinity
                        case 2: scores[i] = -0.0
                        default: scores[i] = 0
                        }
                    }
                }
                let scoreBuffer = try #require(context.device.makeBuffer(bytes: scores,
                    length: scores.count * 4, options: .storageModeShared))
                let selected = try #require(context.device.makeBuffer(length: (rows * stride + 1) * 4,
                    options: .storageModeShared))
                let counts = try #require(context.device.makeBuffer(length: (rows + 1) * 4,
                    options: .storageModeShared))
                let marks = try #require(context.device.makeBuffer(length: rows * pools,
                    options: .storageModeShared))
                let output = selected.contents().assumingMemoryBound(to: UInt32.self)
                let sizes = counts.contents().assumingMemoryBound(to: UInt32.self)
                for tail in [false, true] {
                    output[rows * stride] = 0xDEADBEEF
                    sizes[rows] = 0xDEADBEEF
                    let cb = try #require(context.queue.makeCommandBuffer())
                    try kernels.encodePoolSelection(commandBuffer: cb, scores: scoreBuffer,
                        selected: selected, counts: counts, marks: marks, pools: pools,
                        selectionStride: stride, base: base, tokens: rows, poolSize: pool,
                        topK: topK, includeTail: tail)
                    cb.commit()
                    cb.waitUntilCompleted()
                    try #require(cb.error == nil)
                    #expect(output[rows * stride] == 0xDEADBEEF)
                    #expect(sizes[rows] == 0xDEADBEEF)
                    for row in 0..<rows {
                        let visible = base + row + 1
                        if visible <= topK {
                            #expect(sizes[row] == Glm53Kernels.attendAll)
                        } else {
                            let expected = Glm53Selection.selectTokens(
                                poolScores: Array(scores[(row * pools)..<((row + 1) * pools)]),
                                cached: visible, kPool: pool, indexTopK: topK, alwaysSelectTail: tail)
                            let count = Int(sizes[row])
                            try #require(count <= stride)
                            let actual = Array(UnsafeBufferPointer(start: output + row * stride, count: count))
                            #expect(actual == expected.map(UInt32.init),
                                "pool=\(pool) topK=\(topK) visible=\(visible) pattern=\(pattern) tail=\(tail)")
                        }
                    }
                }
            }
        }
    }
}
