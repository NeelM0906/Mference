import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

/// The decode step's dispatches that are not K3's, timed at one token, as the runner encodes
/// them: K4's runner kernels (dual RMS norm, final depth mix, output norm, head, embedding) and
/// K2's MoE kernels (router, expert gate-up and down). Each case chains its dispatch 48 times in
/// one serial encoder (a token's visits), cycles 24 weight sets where the weights are per layer
/// (so the SLC cannot serve repeats), and is interleaved with the others over 10 rounds; the
/// minimum per dispatch is reported with the threadgroups it runs and the bytes it reads.
/// Env-gated: `MFERENCE_INKLING2_KBENCH=1` with `MFERENCE_INKLING2_KBENCH_ONLY=audit`.
@Suite(.serialized) struct Inkling2DecodeAuditBenchmarks {
    private static var enabled: Bool {
        let env = ProcessInfo.processInfo.environment
        let only = env["MFERENCE_INKLING2_KBENCH_ONLY"] ?? ""
        return env["MFERENCE_INKLING2_KBENCH"] == "1" && (only.isEmpty || only.split(separator: ",").contains("audit"))
    }

    struct Case {
        let name: String
        let threadgroups: Int
        let bytes: Int
        let perToken: Int
        let dispatches: Int          // dispatches per call
        let encode: (MTLComputeCommandEncoder, Int) -> Void   // (encoder, call index)
    }

    static func floats(_ device: MTLDevice, _ count: Int, seed: UInt64, scale: Float = 0.05) -> MTLBuffer {
        let buffer = device.makeBuffer(length: max(1, count) * 4, options: .storageModeShared)!
        let p = buffer.contents().bindMemory(to: Float.self, capacity: count)
        var s = seed
        for i in 0..<count {
            s = s &* 6364136223846793005 &+ 1442695040888963407
            p[i] = (Float(s >> 40) / Float(1 << 24) * 2 - 1) * scale
        }
        return buffer
    }

    static func time(_ context: MetalContext, _ cases: [Case], calls: Int, rounds: Int = 10) -> [Double] {
        var best = [Double](repeating: .infinity, count: cases.count)
        for round in 0...rounds {
            for (i, c) in cases.enumerated() {
                let cb = context.queue.makeCommandBuffer()!
                let e = cb.makeComputeCommandEncoder()!
                for k in 0..<calls { c.encode(e, k) }
                e.endEncoding()
                cb.commit()
                cb.waitUntilCompleted()
                #expect(cb.status == .completed, "\(c.name)")
                if round > 0 { best[i] = min(best[i], (cb.gpuEndTime - cb.gpuStartTime) / Double(calls)) }
            }
        }
        return best
    }

    static func report(_ cases: [Case], _ seconds: [Double]) {
        for (c, s) in zip(cases, seconds) {
            let us = s * 1e6
            let rate = c.bytes > 0 ? String(format: "%.1f GB/s", Double(c.bytes) / s / 1e9) : "-"
            print(String(format: "[ink2-audit] %@ | threadgroups %ld | %.3f MB | %.1f us per call (%ld dispatch%@) | x%ld per token = %.2f ms | %@",
                         c.name, c.threadgroups, Double(c.bytes) / 1e6, us, c.dispatches, c.dispatches == 1 ? "" : "es",
                         c.perToken, us * Double(c.perToken) / 1000, rate))
        }
    }

    @Test(.enabled(if: enabled)) func decodeStepAudit() throws {
        let context = try MetalContext()
        let device = context.device
        let runner = try Inkling2RunnerKernels(device: device)
        let moe = try Inkling2MoEKernels(context: context)
        let shape = Inkling2MoEShape.inkling2Mini
        let d = 2304, sets = 24, calls = 48

        // Runner kernels and the router.
        let x = Self.floats(device, 8 * d, seed: 1, scale: 2)
        let normWeights = Self.floats(device, sets * d, seed: 2, scale: 1)
        let out = device.makeBuffer(length: 8 * d * 4, options: .storageModeShared)!
        let outHalf = device.makeBuffer(length: 8 * d * 2, options: .storageModeShared)!
        let routers = Self.floats(device, sets * 128 * d, seed: 3)
        let bias = Self.floats(device, 128, seed: 4, scale: 0.1)
        let ids = device.makeBuffer(length: 64 * 4, options: .storageModeShared)!
        let routeWeights = device.makeBuffer(length: 64 * 4, options: .storageModeShared)!
        let arena = Self.floats(device, 8 * d, seed: 5, scale: 2)
        let queries = Self.floats(device, 4 * d, seed: 6, scale: 1)
        let tokenIDs = device.makeBuffer(length: 64 * 4, options: .storageModeShared)!
        let tp = tokenIDs.contents().bindMemory(to: UInt32.self, capacity: 64)
        for i in 0..<64 { tp[i] = UInt32((i * 40_009) % 201_024) }
        // The head and the embedding table: Q8_0 [201,024][2,304], one block pattern repeated.
        let vocab = 201_024, rowBytes = d / 32 * 34
        let table = device.makeBuffer(length: vocab * rowBytes, options: .storageModeShared)!
        do {
            var chunk = [UInt8](repeating: 0, count: 34 * 1024)
            var s: UInt64 = 7
            let scale = Float16(0.01).bitPattern
            for b in 0..<1024 {
                chunk[b * 34] = UInt8(truncatingIfNeeded: scale)
                chunk[b * 34 + 1] = UInt8(truncatingIfNeeded: scale >> 8)
                for j in 0..<32 { s = s &* 6364136223846793005 &+ 1442695040888963407; chunk[b * 34 + 2 + j] = UInt8(truncatingIfNeeded: s >> 33) }
            }
            let base = table.contents()
            var off = 0
            chunk.withUnsafeBytes { src in
                while off < table.length {
                    let n = min(src.count, table.length - off)
                    memcpy(base + off, src.baseAddress!, n)
                    off += n
                }
            }
        }
        let logits = device.makeBuffer(length: vocab * 2, options: .storageModeShared)!

        let decodeRouter = try Inkling2DecodeRouter(library: Inkling2KernelLibrary(context: context))
        let routerLogits = device.makeBuffer(length: 128 * 4, options: .storageModeShared)!
        let small: [Case] = [
            Case(name: "router (K2 encodeRouter)", threadgroups: 1, bytes: 128 * d * 4 + d * 4, perToken: 48, dispatches: 1) { e, k in
                moe.encodeRouter(e, shape: shape, x: x, xStride: d, weight: routers, weightOffset: (k % sets) * 128 * d * 4,
                                 bias: bias, ids: ids, routeWeights: routeWeights, tokens: 1)
            },
            Case(name: "router, decode form (K3 Inkling2DecodeRouter)", threadgroups: 64 + 1, bytes: 128 * d * 4 + d * 4,
                 perToken: 48, dispatches: 2) { e, k in
                decodeRouter.encode(encoder: e, x: .init(x), weight: .init(routers, offset: (k % sets) * 128 * d * 4),
                                    bias: .init(bias), logits: .init(routerLogits), ids: .init(ids),
                                    routeWeights: .init(routeWeights))
            },
            Case(name: "dual RMS norm (K4 encodeRMSNormDual)", threadgroups: 1, bytes: 2 * d * 4, perToken: 48, dispatches: 1) { e, k in
                runner.encodeRMSNormDual(e, x: x, xStride: d, weight: normWeights, weightOffset: (k % sets) * d * 4,
                                         eps: 1e-6, out: out, outStride: d, outHalf: outHalf, halfStride: d, d: d, tokens: 1)
            },
            Case(name: "final depth mix, 6 states (K4 encodeDepthMix)", threadgroups: 1, bytes: 7 * d * 4, perToken: 1, dispatches: 1) { e, _ in
                runner.encodeDepthMix(e, arena: arena, stateOffsets: (0..<6).map { $0 * d }, stateStride: d,
                                      queries: queries, queryCount: 1, out: out, outStride: d, outQueryStride: d,
                                      d: d, tokens: 1)
            },
            Case(name: "output RMS norm (K4 encodeRMSNorm)", threadgroups: 1, bytes: 2 * d * 4, perToken: 1, dispatches: 1) { e, _ in
                runner.encodeRMSNorm(e, x: x, xStride: d, weight: normWeights, weightOffset: 0, eps: 1e-6,
                                     out: out, outStride: d, d: d, tokens: 1)
            },
            Case(name: "embedding (K4 encodeEmbedding)", threadgroups: 1, bytes: rowBytes + d * 4, perToken: 1, dispatches: 1) { e, k in
                runner.encodeEmbedding(e, ids: tokenIDs, idsOffset: (k % 64) * 4, table: table, tableOffset: 0,
                                       rowBytes: rowBytes, weight: normWeights, weightOffset: 0, eps: 1e-6,
                                       out: out, outStride: d, d: d, tokens: 1)
            },
        ]
        Self.report(small, Self.time(context, small, calls: calls))

        let head = [Case(name: "head, 201,024 rows Q8_0 (K4 encodeHead)", threadgroups: (vocab + 31) / 32,
                         bytes: vocab * rowBytes + d * 4, perToken: 1, dispatches: 1) { e, _ in
            runner.encodeHead(e, x: out, table: table, tableOffset: 0, rowBytes: rowBytes, rows: vocab,
                              softcap: 0, logits: logits, d: d)
        }]
        Self.report(head, Self.time(context, head, calls: 4, rounds: 6))

        // Expert decode: a full 128-expert pool per encoding, 8 random distinct experts per call.
        for encoding in [Inkling2MoEEncoding.nvfp4, .q8_0] {
            let blob = shape.expertBytes(encoding)
            let stride = (blob + 16383) / 16384 * 16384
            let poolBuffer = device.makeBuffer(length: stride * shape.experts, options: .storageModeShared)!
            var rng = SeedTree(0xA11).key("audit-\(encoding)")
            let blocks = Inkling2SyntheticPool.randomBlocks(encoding: encoding, bytes: blob, rng: &rng)
            blocks.withUnsafeBytes { src in
                for e in 0..<shape.experts { memcpy(poolBuffer.contents() + e * stride, src.baseAddress!, blob) }
            }
            let scales = Self.floats(device, 2 * shape.experts, seed: 9, scale: 0.01)
            let pool = try Inkling2ExpertPool(buffer: poolBuffer, encoding: encoding, shape: shape, expertStride: stride,
                                              gateUpScales: scales, downScales: scales,
                                              downScalesOffset: shape.experts * 4)
            let routes = Inkling2MoETestSupport.randomRoutes(tokens: 64, experts: shape.experts, topK: shape.topK, seed: 11)
            let routeIDs = device.makeBuffer(bytes: routes.ids, length: routes.ids.count * 4, options: .storageModeShared)!
            let routeW = device.makeBuffer(bytes: routes.weights, length: routes.weights.count * 4, options: .storageModeShared)!
            let xHalf = device.makeBuffer(length: d * 2, options: .storageModeShared)!
            memset(xHalf.contents(), 0, d * 2)
            let activation = device.makeBuffer(length: shape.topK * shape.f * 2, options: .storageModeShared)!
            let moeOut = device.makeBuffer(length: d * 4, options: .storageModeShared)!
            let label = encoding == .nvfp4 ? "NVFP4 (20 layers)" : "Q8_0 (4 layers)"
            let expert = [Case(name: "experts \(label), gate-up + down (K2 encodeDecode)",
                               threadgroups: shape.f / 16 * shape.topK + d / 8, bytes: shape.topK * blob,
                               perToken: encoding == .nvfp4 ? 40 : 8, dispatches: 2) { e, k in
                moe.encodeDecode(e, shape: shape, x: xHalf, ids: routeIDs, idsOffset: (k % 64) * shape.topK * 4,
                                 routeWeights: routeW, routeWeightsOffset: (k % 64) * shape.topK * 4, pool: pool,
                                 activation: activation, out: moeOut)
            }]
            Self.report(expert, Self.time(context, expert, calls: calls))
        }
    }
}
