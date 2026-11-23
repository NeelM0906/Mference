import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

/// GPU time of the Inkling 2 routed-expert path at the model's shape (D 2304,
/// F 1152, 128 experts, top-8) on synthetic weights: router, route plan,
/// grouped gate-up and down, reduce, for chunks of 512, 1,024 and 2,048 rows,
/// and the one-token decode step. Env-gated: `MFERENCE_INKLING2_MOE_KBENCH=1`.
/// Prints one line per measurement and asserts only that the work completed.
@Suite(.serialized) struct Inkling2MoEKernelBenchmarks {
    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["MFERENCE_INKLING2_MOE_KBENCH"] == "1"
    }

    private static let shape = Inkling2MoEShape.inkling2Mini
    private static let repeats = 7

    /// A full layer of 128 experts: eight distinct random blobs repeated, so
    /// every expert has its own addresses without minutes of fixture work.
    private static func pool(_ device: MTLDevice, _ encoding: Inkling2MoEEncoding) throws -> Inkling2ExpertPool {
        let blobBytes = shape.expertBytes(encoding)
        let stride = (blobBytes + 16383) / 16384 * 16384
        guard let buffer = device.makeBuffer(length: stride * shape.experts, options: .storageModeShared) else {
            throw PrefillGroupedRoutedMoEError.allocationFailed("bench pool")
        }
        var rng = SeedTree(0xB2).key("bench-\(encoding)")
        let base = buffer.contents().bindMemory(to: UInt8.self, capacity: buffer.length)
        let distinct = (0..<8).map { _ in Inkling2SyntheticPool.randomBlocks(encoding: encoding, bytes: blobBytes, rng: &rng) }
        for e in 0..<shape.experts {
            distinct[e % 8].withUnsafeBufferPointer { (base + e * stride).update(from: $0.baseAddress!, count: blobBytes) }
        }
        let scales: [Float] = (0..<(2 * shape.experts)).map { _ in encoding == .nvfp4 ? rng.uniform(0.006, 0.02) : 1 }
        let scaleBuffer = scales.withUnsafeBytes {
            device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
        }
        return try Inkling2ExpertPool(buffer: buffer, encoding: encoding, shape: shape, expertStride: stride,
                                      gateUpScales: scaleBuffer, downScales: scaleBuffer,
                                      downScalesOffset: shape.experts * 4)
    }

    /// Uniform routes, or a skewed load where expert popularity falls as 1/(i+1)^0.7.
    private static func routes(tokens: Int, skewed: Bool, seed: UInt64) -> (ids: [UInt32], weights: [Float]) {
        guard skewed else {
            return Inkling2MoETestSupport.randomRoutes(tokens: tokens, experts: shape.experts, topK: shape.topK, seed: seed)
        }
        var rng = SeedTree(seed).key("skewed")
        let popularity = (0..<shape.experts).map { 1 / pow(Float($0 + 1), 0.7) }
        let total = popularity.reduce(0, +)
        var ids: [UInt32] = [], weights: [Float] = []
        for _ in 0..<tokens {
            var chosen: [UInt32] = []
            while chosen.count < shape.topK {
                var u = rng.uniform(0, total), e = 0
                while e < shape.experts - 1 && u >= popularity[e] { u -= popularity[e]; e += 1 }
                if !chosen.contains(UInt32(e)) { chosen.append(UInt32(e)) }
            }
            ids += chosen
            weights += (0..<shape.topK).map { _ in rng.uniform(0.2, 2) }
        }
        return (ids, weights)
    }

    private static func best(_ context: MetalContext, _ body: (MTLComputeCommandEncoder) -> Void) -> Double {
        var best = Double.infinity
        for _ in 0..<repeats {
            let cb = Inkling2MoETestSupport.run(context, body)
            #expect(cb.status == .completed, "\(String(describing: cb.error))")
            best = min(best, cb.gpuEndTime - cb.gpuStartTime)
        }
        return best
    }

    @Test(.enabled(if: enabled))
    func groupedExperts() throws {
        let context = try MetalContext()
        let kernels = try Inkling2MoEKernels(context: context)
        let device = context.device
        print("[inkling2-moe-kbench] device=\(device.name) apple10=\(device.supportsApple10TensorOps)")
        let pools = try Inkling2MoEEncoding.allCases.map { try Self.pool(device, $0) }
        let maxTokens = 2048
        let scratch = try Inkling2MoEPrefillScratch(device: device, shape: Self.shape, maxTokens: maxTokens)
        print("[inkling2-moe-kbench] scratch at 2048 rows: \(Inkling2MoEPrefillScratch.bytes(shape: Self.shape, maxTokens: maxTokens) / 1_048_576) MiB")
        let (hidden, _) = Inkling2MoETestSupport.halfBuffer(device, rows: maxTokens, stride: Self.shape.d, d: Self.shape.d,
                                                            seed: 0xB3, range: 1)
        let out = device.makeBuffer(length: maxTokens * Self.shape.d * 4, options: .storageModeShared)!
        // Router inputs: FP32 rows and a random F32 router.
        var rng = SeedTree(0xB4).key("router")
        let x = (0..<(maxTokens * Self.shape.d)).map { _ in rng.uniform(-1, 1) }
        let routerWeight = (0..<(128 * Self.shape.d)).map { _ in rng.uniform(-0.05, 0.05) }
        let bias = (0..<128).map { _ in rng.uniform(-0.1, 0.1) }
        func buffer(_ v: [Float]) -> MTLBuffer {
            v.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
        }
        let xBuffer = buffer(x), wBuffer = buffer(routerWeight), bBuffer = buffer(bias)
        let routerIds = device.makeBuffer(length: maxTokens * 32, options: .storageModeShared)!
        let routerWeights = device.makeBuffer(length: maxTokens * 32, options: .storageModeShared)!

        // Warm-up: bring the GPU to its sustained clock before the first number.
        let warm = Self.routes(tokens: maxTokens, skewed: false, seed: 0xB8)
        Inkling2MoETestSupport.write(warm.ids, to: scratch.ids)
        Inkling2MoETestSupport.write(warm.weights, to: scratch.routeWeights)
        let warmStart = Date()
        while Date().timeIntervalSince(warmStart) < 3 {
            _ = Inkling2MoETestSupport.run(context) {
                kernels.encodePrefill($0, hidden: hidden, hiddenStride: Self.shape.d, pool: pools[0],
                                      scratch: scratch, tokens: maxTokens, out: out, outStride: Self.shape.d)
            }
        }

        for tokens in [512, 1024, 2048] {
            let router = Self.best(context) {
                kernels.encodeRouter($0, shape: Self.shape, x: xBuffer, xStride: Self.shape.d, weight: wBuffer,
                                     bias: bBuffer, ids: routerIds, routeWeights: routerWeights, tokens: tokens)
            }
            print(String(format: "[inkling2-moe-kbench] rows=%d router %.3f ms", tokens, router * 1000))
            for skewed in [false, true] {
                let r = Self.routes(tokens: tokens, skewed: skewed, seed: 0xB5 + UInt64(tokens))
                Inkling2MoETestSupport.write(r.ids, to: scratch.ids)
                Inkling2MoETestSupport.write(r.weights, to: scratch.routeWeights)
                for pool in pools {
                    let plan = Self.best(context) { kernels.encodeRoutePlan($0, scratch: scratch, tokens: tokens) }
                    let experts = Self.best(context) {
                        kernels.encodeGroupedExperts($0, hidden: hidden, hiddenStride: Self.shape.d, pool: pool, scratch: scratch)
                    }
                    let reduce = Self.best(context) {
                        kernels.encodeReduce($0, scratch: scratch, tokens: tokens, out: out, outStride: Self.shape.d)
                    }
                    let total = Self.best(context) {
                        kernels.encodePrefill($0, hidden: hidden, hiddenStride: Self.shape.d, pool: pool,
                                              scratch: scratch, tokens: tokens, out: out, outStride: Self.shape.d)
                    }
                    let work = Int(Inkling2MoETestSupport.read(scratch.workCount, as: UInt32.self, count: 1)[0])
                    let flops = 2.0 * Double(tokens * Self.shape.topK) * Double(Self.shape.macsPerPair)
                    print(String(format: "[inkling2-moe-kbench] rows=%d routes=%@ enc=%@ blocks=%d plan %.3f ms experts %.3f ms (%.2f TFLOP/s) reduce %.3f ms total %.3f ms (%.2f TFLOP/s)",
                                 tokens, skewed ? "skewed" : "uniform", pool.encoding == .nvfp4 ? "nvfp4" : "q8_0", work,
                                 plan * 1000, experts * 1000, flops / experts / 1e12, reduce * 1000,
                                 total * 1000, flops / total / 1e12))
                }
            }
        }
    }

    @Test(.enabled(if: enabled))
    func decodeStep() throws {
        let context = try MetalContext()
        let kernels = try Inkling2MoEKernels(context: context)
        let device = context.device
        let (x, _) = Inkling2MoETestSupport.halfBuffer(device, rows: 1, stride: Self.shape.d, d: Self.shape.d, seed: 0xB6, range: 1)
        let ids = device.makeBuffer(length: 32, options: .storageModeShared)!
        let weights = device.makeBuffer(length: 32, options: .storageModeShared)!
        let r = Inkling2MoETestSupport.randomRoutes(tokens: 1, experts: Self.shape.experts, topK: 8, seed: 0xB7)
        Inkling2MoETestSupport.write(r.ids, to: ids)
        Inkling2MoETestSupport.write(r.weights, to: weights)
        let activation = device.makeBuffer(length: 8 * Self.shape.f * 2, options: .storageModeShared)!
        let out = device.makeBuffer(length: Self.shape.d * 4, options: .storageModeShared)!
        for encoding in Inkling2MoEEncoding.allCases {
            let pool = try Self.pool(device, encoding)
            let warmStart = Date()
            while Date().timeIntervalSince(warmStart) < 1 {
                _ = Inkling2MoETestSupport.run(context) {
                    kernels.encodeDecode($0, shape: Self.shape, x: x, ids: ids, routeWeights: weights, pool: pool,
                                         activation: activation, out: out)
                }
            }
            let t = Self.best(context) {
                kernels.encodeDecode($0, shape: Self.shape, x: x, ids: ids, routeWeights: weights, pool: pool,
                                     activation: activation, out: out)
            }
            let bytes = Double(8 * Self.shape.expertBytes(encoding))
            print(String(format: "[inkling2-moe-kbench] decode enc=%@ %.3f ms (%.0f GB/s of expert weights)",
                         encoding == .nvfp4 ? "nvfp4" : "q8_0", t * 1000, bytes / t / 1e9))
        }
    }
}
