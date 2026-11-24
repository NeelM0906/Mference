import Foundation
import Metal
import Testing
@testable import Mference

/// The runner's head at Mini's shape on synthetic data: the final depth mix
/// of five 2,304-wide states (one row), the output norm, and the Q8_0 head
/// over 201,024 rows (a 492 MB table), with the mix from the runner's
/// one-threadgroup kernel or from K3's one-row `Inkling2DepthMix`. Each
/// sample repeats the work `repeats` times in one serial encoder, so the GPU
/// stays busy; the figure is the median per repetition of GPU time over 15
/// samples after 3 warm-up ones. Model-free, but it
/// allocates the half-gigabyte table, so it runs only with
/// `MFERENCE_INKLING2_HEAD_BENCH` set (and, as a GPU benchmark, under
/// `lock.sh build`).
@Suite("Inkling2 head benchmark", .serialized)
struct Inkling2HeadBenchmark {
    static var enabled: Bool { ProcessInfo.processInfo.environment["MFERENCE_INKLING2_HEAD_BENCH"] != nil }

    @Test(.enabled(if: enabled, "needs MFERENCE_INKLING2_HEAD_BENCH (allocates a 492 MB table)"))
    func headWithEitherDepthMix() throws {
        let context = try MetalContext()
        let device = context.device
        let kernels = try Inkling2RunnerKernels(device: device)
        let mix = try Inkling2DepthMix(library: Inkling2KernelLibrary(context: context))
        let d = 2304, stateCount = 5, vocab = 200_058, rows = 201_024, rowBytes = d / 32 * 34
        func floats(_ count: Int, _ seed: Int) throws -> MTLBuffer {
            let b = try #require(device.makeBuffer(length: count * 4, options: .storageModeShared))
            let p = b.contents().bindMemory(to: Float.self, capacity: count)
            for i in 0..<count { p[i] = Float((i * 7919 + seed * 104_729) % 2001 - 1000) / 1000 }
            return b
        }
        let arena = try floats(stateCount * d, 1)
        let query = try floats(d, 2)
        let norm = try floats(d, 3)
        let finalRow = try #require(device.makeBuffer(length: 2 * d * 4, options: .storageModeShared))
        let table = try #require(device.makeBuffer(length: rows * rowBytes, options: .storageModeShared))
        memset(table.contents(), 0x11, rows * rowBytes)   // FP16 scales of 0x1111, codes 17
        let logits = try #require(device.makeBuffer(length: rows * 2, options: .storageModeShared))

        func time(repeats: Int = 1, _ body: (MTLComputeCommandEncoder) throws -> Void) throws -> Double {
            var samples: [Double] = []
            for i in 0..<18 {
                let cb = try #require(context.queue.makeCommandBuffer())
                let enc = try #require(cb.makeComputeCommandEncoder())
                for _ in 0..<repeats { try body(enc) }
                enc.endEncoding()
                cb.commit()
                cb.waitUntilCompleted()
                #expect(cb.status == .completed)
                if i >= 3 { samples.append((cb.gpuEndTime - cb.gpuStartTime) / Double(repeats)) }
            }
            return samples.sorted()[samples.count / 2]
        }
        func runnerMix(_ enc: MTLComputeCommandEncoder) {
            kernels.encodeDepthMix(enc, arena: arena, stateOffsets: (0..<stateCount).map { $0 * d }, stateStride: d,
                                   queries: query, queriesOffset: 0, queryCount: 1, out: finalRow, outStride: d,
                                   outQueryStride: d, d: d, tokens: 1)
        }
        func oneRowMix(_ enc: MTLComputeCommandEncoder) throws {
            try mix.encode(encoder: enc, states: (0..<stateCount).map { (Inkling2BufferRef(arena, offset: $0 * d * 4), d) },
                           queries: Inkling2BufferRef(query), queryCount: 1, output: Inkling2BufferRef(finalRow),
                           outputStride: d, rows: 1, width: d, eps: Inkling2RunnerKernels.residualNormEps)
        }
        func tail(_ enc: MTLComputeCommandEncoder) {
            kernels.encodeRMSNorm(enc, x: finalRow, xStride: d, weight: norm, weightOffset: 0,
                                  eps: Inkling2RunnerKernels.residualNormEps, out: finalRow, outOffset: d * 4,
                                  outStride: d, d: d, tokens: 1)
            kernels.encodeHead(enc, x: finalRow, xOffset: d * 4, table: table, tableOffset: 0, rowBytes: rowBytes,
                               rows: vocab, softcap: 0, logits: logits, d: d)
        }

        // Both mixes agree (the rest of the head reads the same row).
        func mixed(_ body: (MTLComputeCommandEncoder) throws -> Void) throws -> [Float] {
            _ = try time(body)
            let p = finalRow.contents().bindMemory(to: Float.self, capacity: d)
            return (0..<d).map { p[$0] }
        }
        let a = try mixed { runnerMix($0) }, b = try mixed { try oneRowMix($0) }
        let mixDifference = zip(a, b).map { abs($0 - $1) }.max()! / max(a.map(abs).max()!, 1e-6)

        let runnerMixOnly = try time(repeats: 200) { runnerMix($0) }
        let oneRowMixOnly = try time(repeats: 200) { try oneRowMix($0) }
        let headRunnerMix = try time(repeats: 20) { runnerMix($0); tail($0) }
        let headOneRowMix = try time(repeats: 20) { try oneRowMix($0); tail($0) }
        let tailOnly = try time(repeats: 20) { tail($0) }
        print(String(format: "inkling2 head bench: mix alone, runner kernel %.4f ms, one-row kernel %.4f ms; "
                     + "norm and head matvec %.3f ms; whole head with the runner's mix %.3f ms, with the one-row "
                     + "mix %.3f ms; mix outputs differ by %.2e (relative)",
                     runnerMixOnly * 1e3, oneRowMixOnly * 1e3, tailOnly * 1e3, headRunnerMix * 1e3,
                     headOneRowMix * 1e3, mixDifference))
        #expect(mixDifference < 1e-4)
    }
}
