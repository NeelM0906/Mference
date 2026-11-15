import Foundation
import Metal
import Testing
@testable import Mference

/// **Measurement**, not a gate (WS4 U1 §3a spike): GPU cost of expanding the
/// Unsloth UD-IQ3_XXS routed experts from native GGUF blocks into LUT4 slots,
/// for decode misses (1, 4, 10 experts of one layer, one command buffer, as
/// cb2 would carry them) and for prefill (128 experts per batch, scaled to a
/// 512-expert layer and a 48-layer sweep). Random block bytes at the install's
/// shapes: gate and up 2560 × 640, down 640 × 2560. Env-gated:
/// `MFERENCE_WS4_LUT4_BENCH=1`. Prints one line per case.
@Suite(.serialized) struct GGUFLUT4ExpandBenchmarks {
    private static var enabled: Bool { ProcessInfo.processInfo.environment["MFERENCE_WS4_LUT4_BENCH"] == "1" }

    static let projectionWeights = 2560 * 640
    static let page = 16_384

    /// A layer mix: gate/up type, down type, and how many of the 48 layers use it.
    struct Mix { let name: String; let gateUp: GGUFLUT4.SourceType; let down: GGUFLUT4.SourceType; let layers: Int }
    static let mixes = [
        Mix(name: "IQ2_S + IQ4_NL", gateUp: .iq2_s, down: .iq4_nl, layers: 23),
        Mix(name: "IQ3_XXS + IQ4_NL", gateUp: .iq3_xxs, down: .iq4_nl, layers: 21),
        Mix(name: "IQ2_S + Q2_0", gateUp: .iq2_s, down: .q2_0, layers: 3),
        Mix(name: "IQ4_XS + IQ4_NL", gateUp: .iq4_xs, down: .iq4_nl, layers: 1),
    ]

    static func srcBytes(_ t: GGUFLUT4.SourceType) -> Int { projectionWeights / t.blockWeights * t.blockBytes }

    struct Section { let index: Int; let d: Int; let m: Int; let end: Int }
    static func sections(_ t: GGUFLUT4.SourceType, at start: Int) -> Section {
        let p = GGUFLUT4.parameters(t), n = projectionWeights
        let d = start + n / 2, m = d + 2 * (n / p.bd)
        return Section(index: start, d: d, m: m, end: m + (p.bm == 0 ? 0 : n / p.bm))
    }

    /// Per-expert source and slot geometry of a mix.
    struct Geometry {
        let srcStride: Int, slotStride: Int
        let gate: Section, up: Section, down: Section
        init(_ mix: Mix) {
            gate = GGUFLUT4ExpandBenchmarks.sections(mix.gateUp, at: 0)
            up = GGUFLUT4ExpandBenchmarks.sections(mix.gateUp, at: gate.end)
            down = GGUFLUT4ExpandBenchmarks.sections(mix.down, at: up.end)
            let page = GGUFLUT4ExpandBenchmarks.page
            slotStride = (down.end + page - 1) / page * page
            let raw = 2 * GGUFLUT4ExpandBenchmarks.srcBytes(mix.gateUp) + GGUFLUT4ExpandBenchmarks.srcBytes(mix.down)
            srcStride = (raw + page - 1) / page * page
        }
    }

    static func randomBuffer(_ device: MTLDevice, _ count: Int) -> MTLBuffer {
        let b = device.makeBuffer(length: count, options: .storageModeShared)!
        let p = b.contents().bindMemory(to: UInt32.self, capacity: count / 4)
        var x: UInt32 = 0x9E37_79B9
        for i in 0..<(count / 4) { x = x &* 1_664_525 &+ 1_013_904_223; p[i] = x }
        return b
    }

    /// Encodes the expansion of `experts` consecutive experts: one dispatch for
    /// the gate/up type (two jobs per expert) and one for the down type.
    static func encode(_ kernel: GGUFLUT4Expand, _ mix: Mix, _ g: Geometry, experts: Int,
                       src: MTLBuffer, dst: MTLBuffer, cb: MTLCommandBuffer) {
        var gateUp = [GGUFLUT4Expand.Job](), down = [GGUFLUT4Expand.Job]()
        let gu = srcBytes(mix.gateUp)
        func job(_ s: Section, _ type: GGUFLUT4.SourceType, src: Int, slot: Int) -> GGUFLUT4Expand.Job {
            GGUFLUT4Expand.Job(srcOffset: UInt64(src), dstIndexOffset: UInt64(slot + s.index),
                               dstDOffset: UInt64(slot + s.d), dstMOffset: UInt64(slot + s.m),
                               blocks: UInt32(projectionWeights / type.blockWeights))
        }
        for e in 0..<experts {
            let s = e * g.srcStride, slot = e * g.slotStride
            gateUp.append(job(g.gate, mix.gateUp, src: s, slot: slot))
            gateUp.append(job(g.up, mix.gateUp, src: s + gu, slot: slot))
            down.append(job(g.down, mix.down, src: s + 2 * gu, slot: slot))
        }
        kernel.encode(commandBuffer: cb, type: mix.gateUp, src: src, dst: dst, jobs: gateUp)
        kernel.encode(commandBuffer: cb, type: mix.down, src: src, dst: dst, jobs: down)
    }

    /// Median GPU and wall milliseconds over `iterations` command buffers.
    static func time(_ ctx: MetalContext, iterations: Int, _ body: (MTLCommandBuffer) -> Void) throws -> (gpu: Double, wall: Double) {
        let warm = try #require(ctx.queue.makeCommandBuffer())
        body(warm); warm.commit(); warm.waitUntilCompleted()
        var gpu = [Double](), wall = [Double]()
        for _ in 0..<iterations {
            let cb = try #require(ctx.queue.makeCommandBuffer())
            let start = Date()
            body(cb); cb.commit(); cb.waitUntilCompleted()
            wall.append(Date().timeIntervalSince(start) * 1e3)
            gpu.append((cb.gpuEndTime - cb.gpuStartTime) * 1e3)
        }
        return (gpu.sorted()[iterations / 2], wall.sorted()[iterations / 2])
    }

    @Test func expansionCostForDecodeMissesAndPrefill() throws {
        guard Self.enabled else { return }
        let ctx = try MetalContext()
        let kernel = try GGUFLUT4Expand(context: ctx)
        let batch = 128
        print("  [lut4bench] gate/up 2560x640, down 640x2560; median of 30 (decode) / 10 (prefill) command buffers")
        for mix in Self.mixes {
            let g = Geometry(mix)
            let src = Self.randomBuffer(ctx.device, batch * g.srcStride)
            let dst = try #require(ctx.device.makeBuffer(length: batch * g.slotStride, options: .storageModeShared))
            print(String(format: "  [lut4bench] %@ (%d layers): disk %d B/expert, slot %d B/expert",
                         mix.name as NSString, mix.layers, g.srcStride, g.slotStride))
            for experts in [1, 4, 10] {
                let t = try Self.time(ctx, iterations: 30) {
                    Self.encode(kernel, mix, g, experts: experts, src: src, dst: dst, cb: $0)
                }
                print(String(format: "  [lut4bench]   decode %2d misses: gpu %.3f ms (%.1f us/expert), wall %.3f ms",
                             experts, t.gpu, t.gpu * 1e3 / Double(experts), t.wall))
            }
            let t = try Self.time(ctx, iterations: 10) {
                Self.encode(kernel, mix, g, experts: batch, src: src, dst: dst, cb: $0)
            }
            let perExpert = t.gpu / Double(batch)
            let outGBs = Double(batch * g.slotStride) / (t.gpu / 1e3) / 1e9
            print(String(format: "  [lut4bench]   prefill %d experts: gpu %.3f ms (%.1f us/expert, %.1f GB/s slot writes); layer x512 %.1f ms",
                         batch, t.gpu, perExpert * 1e3, outGBs, perExpert * 512))
        }
    }
}
