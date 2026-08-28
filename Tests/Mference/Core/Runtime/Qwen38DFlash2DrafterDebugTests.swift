import Foundation
import Metal
import Testing

@testable import Mference

/// Bring-up diagnostic: prints NaN fraction and value range for every
/// intermediate drafter buffer after the round-1 fixture. Gated like the
/// parity suite; carries no assertions beyond "the run completes".
@Suite struct Qwen38DFlash2DrafterDebugTests {

    @Test func dumpIntermediateStats() throws {
        guard ProcessInfo.processInfo.environment["MFERENCE_DFLASH2_DEBUG"] == "1"
        else { return }
        let context = try MetalContext()
        let cfg = try #require(ArchConfig.knownArchitectures[.qwen38])
        let root = { var u = URL(fileURLWithPath: #filePath)
                     for _ in 0..<5 { u.deleteLastPathComponent() }
                     return u }()
        let dir = root.appendingPathComponent("scratch/dflash2-qwen38")
        let fixtures = root.appendingPathComponent("scratch/dflash2-fixtures")
        let dummy = context.device.makeBuffer(length: 64, options: .storageModeShared)!
        let view = TensorView(buffer: dummy, offset: 0, length: 64,
                              scaleOffset: 0, scaleLength: 0,
                              biasOffset: 0, biasLength: 0,
                              shape: (0, 0, 0, 0), dtype: 0)
        let drafter = try Qwen38DFlash2Drafter(context: context, directory: dir,
                                               embedding: view, lmHead: view,
                                               targetConfig: cfg,
                                               precision: .bf16)
        func floats(_ name: String) throws -> [Float] {
            let data = try Data(contentsOf: fixtures.appendingPathComponent(name))
            return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }
        let ctx = try floats("f1_ctx.f32")
        let blk = try floats("f1_block.f32")
        let tapWidth = drafter.tapOrdinal.count * drafter.config.hidden_size
        let tapDst = drafter.tapStaging.contents()
            .bindMemory(to: Float16.self, capacity: 3 * tapWidth)
        for i in 0..<ctx.count { tapDst[i] = Float16(ctx[i]) }
        drafter.commitTapRows(3)
        let blockDst = drafter.parityBlockHidden.contents()
            .bindMemory(to: Float.self, capacity: blk.count)
        for i in 0..<blk.count { blockDst[i] = blk[i] }
        try drafter.runCoreForParity(blockTokens: 8)

        func stats(_ name: String, _ buf: MTLBuffer, count: Int) {
            let p = buf.contents().bindMemory(to: Float16.self, capacity: count)
            var nans = 0
            var lo = Float.infinity, hi = -Float.infinity
            for i in 0..<count {
                let v = Float(p[i])
                if v.isNaN { nans += 1; continue }
                lo = min(lo, v); hi = max(hi, v)
            }
            print("[dflash2-debug] \(name): nan=\(nans)/\(count) range=[\(lo), \(hi)]")
        }
        for (name, buf) in drafter.parityProbe.sorted(by: { $0.key < $1.key }) {
            stats(name, buf, count: min(buf.length / 2, 8 * 17408))
        }
        stats("normedFinal", drafter.parityNormedOut, count: 8 * 5120)

        // Isolated draft-round latency: the core forward with no target
        // model in memory. Bounds the drafter's own compute cost.
        for round in 0..<5 {
            let t0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            try drafter.runCoreForParity(blockTokens: 8)
            let ms = Double(clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - t0) / 1e6
            print("[dflash2-debug] core round \(round): \(String(format: "%.1f", ms)) ms")
        }
    }
}
