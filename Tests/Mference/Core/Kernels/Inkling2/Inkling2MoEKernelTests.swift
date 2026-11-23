import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

/// The Inkling 2 routed-expert kernels against the CPU references in
/// `MferenceValidationSupport` (`Inkling2ExpertRef`, `Inkling2RouterRef`).
/// The kernels read FP16 inputs, so the references get the same FP16-rounded
/// inputs; the remaining differences are half activations between the two
/// products, half-rounded Q8_0 weights in the grouped tiles, and summation order.
@Suite(.serialized) struct Inkling2MoEKernelTests {
    static let small = Inkling2MoEShape(d: 256, f: 128, experts: 8, topK: 4)

    /// Bound for the routed sum against the FP32 reference: FP16 activations
    /// (rel 2^-11) feed a 128-deep FP32 product, plus FP16-rounded Q8_0 weights.
    static let routedTolerance: Float = 4e-3

    private static func expectedRows(_ synthetic: Inkling2SyntheticPool, hidden: [[Float]],
                                     ids: [UInt32], weights: [Float], topK: Int) -> [[Float]] {
        let experts = (0..<synthetic.shape.experts).map { synthetic.expert($0) }
        return hidden.indices.map { t in
            Inkling2ExpertRef.routedSum(
                experts: { experts[$0] },
                ids: (0..<topK).map { Int(ids[t * topK + $0]) },
                weights: Array(weights[(t * topK)..<((t + 1) * topK)]),
                x: hidden[t])
        }
    }

    @Test(arguments: [(Inkling2MoEEncoding.nvfp4, 40), (.nvfp4, 200), (.q8_0, 40), (.q8_0, 200)])
    func prefillMatchesReference(_ c: (encoding: Inkling2MoEEncoding, tokens: Int)) throws {
        let context = try MetalContext()
        let kernels = try Inkling2MoEKernels(context: context)
        let shape = Self.small
        let synthetic = try Inkling2SyntheticPool.make(device: context.device, encoding: c.encoding, shape: shape,
                                                       seed: 0x2A01, quantized: true)
        let stride = shape.d + 8
        let (hidden, rows) = Inkling2MoETestSupport.halfBuffer(context.device, rows: c.tokens, stride: stride,
                                                               d: shape.d, seed: 0x2A02)
        let routes = Inkling2MoETestSupport.randomRoutes(tokens: c.tokens, experts: shape.experts,
                                                         topK: shape.topK, seed: 0x2A03)
        let scratch = try Inkling2MoEPrefillScratch(device: context.device, shape: shape, maxTokens: c.tokens + 3)
        Inkling2MoETestSupport.write(routes.ids, to: scratch.ids)
        Inkling2MoETestSupport.write(routes.weights, to: scratch.routeWeights)
        let outStride = shape.d + 4
        let out = context.device.makeBuffer(length: c.tokens * outStride * 4, options: .storageModeShared)!
        let cb = Inkling2MoETestSupport.run(context) { encoder in
            kernels.encodePrefill(encoder, hidden: hidden, hiddenStride: stride, pool: synthetic.pool,
                                  scratch: scratch, tokens: c.tokens, out: out, outStride: outStride)
        }
        #expect(cb.status == .completed, "\(String(describing: cb.error))")
        let got = Inkling2MoETestSupport.read(out, as: Float.self, count: c.tokens * outStride)
        let want = Self.expectedRows(synthetic, hidden: rows, ids: routes.ids, weights: routes.weights, topK: shape.topK)
        var worst: Float = 0
        for t in 0..<c.tokens {
            let row = Array(got[(t * outStride)..<(t * outStride + shape.d)])
            worst = max(worst, RelError.compute(actual: row, reference: want[t]))
        }
        #expect(worst < Self.routedTolerance, "worst row rel err \(worst)")
    }

    @Test func routePlanGroupsEveryPairOnce() throws {
        let context = try MetalContext()
        let kernels = try Inkling2MoEKernels(context: context)
        let shape = Inkling2MoEShape(d: 256, f: 128, experts: 16, topK: 8)
        let tokens = 300
        // Skewed: expert 3 takes most tokens, so it needs several blocks.
        var routes = Inkling2MoETestSupport.randomRoutes(tokens: tokens, experts: shape.experts, topK: shape.topK, seed: 0x2A04)
        for t in 0..<tokens where !routes.ids[(t * 8)..<(t * 8 + 8)].contains(3) { routes.ids[t * 8] = 3 }
        let scratch = try Inkling2MoEPrefillScratch(device: context.device, shape: shape, maxTokens: tokens)
        Inkling2MoETestSupport.write(routes.ids, to: scratch.ids)
        Inkling2MoETestSupport.write(routes.weights, to: scratch.routeWeights)
        let cb = Inkling2MoETestSupport.run(context) { kernels.encodeRoutePlan($0, scratch: scratch, tokens: tokens) }
        #expect(cb.status == .completed)

        let pairCount = tokens * shape.topK
        let pairs = Inkling2MoETestSupport.read(scratch.pairs, as: Inkling2RoutedPair.self, count: pairCount)
        let workCount = Int(Inkling2MoETestSupport.read(scratch.workCount, as: UInt32.self, count: 1)[0])
        let work = Inkling2MoETestSupport.read(scratch.work, as: Inkling2WorkBlock.self, count: workCount)
        let dispatch = Inkling2MoETestSupport.read(scratch.dispatch, as: UInt32.self, count: 6)
        #expect(dispatch == [UInt32(shape.f / 64), UInt32(workCount), 1, UInt32(shape.d / 128), UInt32(workCount), 1])

        var counts = [Int](repeating: 0, count: shape.experts)
        for id in routes.ids { counts[Int(id)] += 1 }
        #expect(workCount == counts.reduce(0) { $0 + ($1 + 63) / 64 })
        var seen = Set<UInt32>()
        var covered = 0
        for block in work {
            #expect(block.rowCount > 0 && block.rowCount <= 64)
            for i in 0..<Int(block.rowCount) {
                let pair = pairs[Int(block.pairStart) + i]
                let p = pair.token * UInt32(shape.topK) + pair.rank
                #expect(routes.ids[Int(p)] == block.expert, "pair \(p) in expert \(block.expert)'s block")
                #expect(pair.weight == routes.weights[Int(p)])
                #expect(seen.insert(p).inserted, "pair \(p) twice")
                covered += 1
            }
        }
        #expect(covered == pairCount)
    }

    @Test func routerMatchesReference() throws {
        let context = try MetalContext()
        let kernels = try Inkling2MoEKernels(context: context)
        let shape = Inkling2MoEShape(d: 256, f: 128, experts: 128, topK: 8)
        let tokens = 37
        var rng = SeedTree(0x2A05).key("router")
        let weight = (0..<(128 * shape.d)).map { _ in rng.uniform(-0.1, 0.1) }
        let bias = (0..<128).map { _ in rng.uniform(-0.2, 0.2) }
        let xStride = shape.d + 4
        let x = (0..<(tokens * xStride)).map { _ in rng.uniform(-1, 1) }
        func buffer(_ v: [Float]) -> MTLBuffer {
            v.withUnsafeBytes { context.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
        }
        let ids = context.device.makeBuffer(length: tokens * 8 * 4, options: .storageModeShared)!
        let weights = context.device.makeBuffer(length: tokens * 8 * 4, options: .storageModeShared)!
        let cb = Inkling2MoETestSupport.run(context) {
            kernels.encodeRouter($0, shape: shape, x: buffer(x), xStride: xStride, weight: buffer(weight),
                                 bias: buffer(bias), ids: ids, routeWeights: weights, tokens: tokens)
        }
        #expect(cb.status == .completed)
        let gotIds = Inkling2MoETestSupport.read(ids, as: UInt32.self, count: tokens * 8)
        let gotWeights = Inkling2MoETestSupport.read(weights, as: Float.self, count: tokens * 8)
        for t in 0..<tokens {
            let row = Array(x[(t * xStride)..<(t * xStride + shape.d)])
            let logits = Inkling2RouterRef.logits(weight: weight, experts: 128, x: row)
            let route = Inkling2RouterRef.route(logits: logits, selectionBias: bias)
            #expect(gotIds[(t * 8)..<(t * 8 + 8)].map(Int.init) == route.ids, "token \(t)")
            let err = RelError.compute(actual: Array(gotWeights[(t * 8)..<(t * 8 + 8)]), reference: route.weights)
            #expect(err < 1e-5, "token \(t) weights rel err \(err)")
        }
    }

    @Test(arguments: [Inkling2MoEEncoding.nvfp4, .q8_0])
    func decodeMatchesReference(encoding: Inkling2MoEEncoding) throws {
        let context = try MetalContext()
        let kernels = try Inkling2MoEKernels(context: context)
        let shape = Inkling2MoEShape(d: 256, f: 128, experts: 16, topK: 8)
        let synthetic = try Inkling2SyntheticPool.make(device: context.device, encoding: encoding, shape: shape,
                                                       seed: 0x2A06, quantized: true)
        let (x, rows) = Inkling2MoETestSupport.halfBuffer(context.device, rows: 1, stride: shape.d, d: shape.d, seed: 0x2A07)
        let routes = Inkling2MoETestSupport.randomRoutes(tokens: 1, experts: shape.experts, topK: shape.topK, seed: 0x2A08)
        let ids = context.device.makeBuffer(length: 32, options: .storageModeShared)!
        let weights = context.device.makeBuffer(length: 32, options: .storageModeShared)!
        Inkling2MoETestSupport.write(routes.ids, to: ids)
        Inkling2MoETestSupport.write(routes.weights, to: weights)
        let activation = context.device.makeBuffer(length: shape.topK * shape.f * 2, options: .storageModeShared)!
        let out = context.device.makeBuffer(length: shape.d * 4, options: .storageModeShared)!
        let cb = Inkling2MoETestSupport.run(context) {
            kernels.encodeDecode($0, shape: shape, x: x, ids: ids, routeWeights: weights, pool: synthetic.pool,
                                 activation: activation, out: out)
        }
        #expect(cb.status == .completed)
        let got = Inkling2MoETestSupport.read(out, as: Float.self, count: shape.d)
        let want = Self.expectedRows(synthetic, hidden: rows, ids: routes.ids, weights: routes.weights, topK: shape.topK)[0]
        let err = RelError.compute(actual: got, reference: want)
        #expect(err < Self.routedTolerance, "rel err \(err)")
    }

    /// Prefill and decode of the same token agree closely: both read FP16
    /// inputs and store FP16 activations; they differ in summation order and
    /// in Q8_0 weight rounding (the grouped tiles hold half weights).
    @Test(arguments: [Inkling2MoEEncoding.nvfp4, .q8_0])
    func prefillRowMatchesDecode(encoding: Inkling2MoEEncoding) throws {
        let context = try MetalContext()
        let kernels = try Inkling2MoEKernels(context: context)
        let shape = Inkling2MoEShape(d: 256, f: 128, experts: 16, topK: 8)
        let synthetic = try Inkling2SyntheticPool.make(device: context.device, encoding: encoding, shape: shape,
                                                       seed: 0x2A09, quantized: true)
        let tokens = 9
        let (hidden, _) = Inkling2MoETestSupport.halfBuffer(context.device, rows: tokens, stride: shape.d, d: shape.d, seed: 0x2A0A)
        let routes = Inkling2MoETestSupport.randomRoutes(tokens: tokens, experts: shape.experts, topK: shape.topK, seed: 0x2A0B)
        let scratch = try Inkling2MoEPrefillScratch(device: context.device, shape: shape, maxTokens: tokens)
        Inkling2MoETestSupport.write(routes.ids, to: scratch.ids)
        Inkling2MoETestSupport.write(routes.weights, to: scratch.routeWeights)
        let prefillOut = context.device.makeBuffer(length: tokens * shape.d * 4, options: .storageModeShared)!
        let decodeOut = context.device.makeBuffer(length: tokens * shape.d * 4, options: .storageModeShared)!
        let activation = context.device.makeBuffer(length: shape.topK * shape.f * 2, options: .storageModeShared)!
        _ = Inkling2MoETestSupport.run(context) { encoder in
            kernels.encodePrefill(encoder, hidden: hidden, hiddenStride: shape.d, pool: synthetic.pool,
                                  scratch: scratch, tokens: tokens, out: prefillOut, outStride: shape.d)
        }
        for t in 0..<tokens {
            _ = Inkling2MoETestSupport.run(context) {
                kernels.encodeDecode($0, shape: shape, x: hidden, xOffset: t * shape.d * 2,
                                     ids: scratch.ids, idsOffset: t * shape.topK * 4,
                                     routeWeights: scratch.routeWeights, routeWeightsOffset: t * shape.topK * 4,
                                     pool: synthetic.pool, activation: activation, out: decodeOut, outOffset: t * shape.d * 4)
            }
        }
        let a = Inkling2MoETestSupport.read(prefillOut, as: Float.self, count: tokens * shape.d)
        let b = Inkling2MoETestSupport.read(decodeOut, as: Float.self, count: tokens * shape.d)
        let err = RelError.compute(actual: a, reference: b)
        #expect(err < (encoding == .nvfp4 ? 2e-3 : Self.routedTolerance), "rel err \(err)")
    }

    /// Production shape (D 2304, F 1152) with random blocks: one grouped
    /// prefill against the reference for a few tokens of 16 experts.
    @Test(.fullTier, arguments: [Inkling2MoEEncoding.nvfp4, .q8_0])
    func prefillMatchesReferenceAtModelShape(encoding: Inkling2MoEEncoding) throws {
        let context = try MetalContext()
        let kernels = try Inkling2MoEKernels(context: context)
        let shape = Inkling2MoEShape(d: 2304, f: 1152, experts: 16, topK: 8)
        let synthetic = try Inkling2SyntheticPool.make(device: context.device, encoding: encoding, shape: shape,
                                                       seed: 0x2A0C, quantized: false)
        let tokens = 12
        let (hidden, rows) = Inkling2MoETestSupport.halfBuffer(context.device, rows: tokens, stride: shape.d,
                                                               d: shape.d, seed: 0x2A0D, range: 1)
        let routes = Inkling2MoETestSupport.randomRoutes(tokens: tokens, experts: shape.experts, topK: shape.topK, seed: 0x2A0E)
        let scratch = try Inkling2MoEPrefillScratch(device: context.device, shape: shape, maxTokens: tokens)
        Inkling2MoETestSupport.write(routes.ids, to: scratch.ids)
        Inkling2MoETestSupport.write(routes.weights, to: scratch.routeWeights)
        let out = context.device.makeBuffer(length: tokens * shape.d * 4, options: .storageModeShared)!
        let cb = Inkling2MoETestSupport.run(context) { encoder in
            kernels.encodePrefill(encoder, hidden: hidden, hiddenStride: shape.d, pool: synthetic.pool,
                                  scratch: scratch, tokens: tokens, out: out, outStride: shape.d)
        }
        #expect(cb.status == .completed)
        let got = Inkling2MoETestSupport.read(out, as: Float.self, count: tokens * shape.d)
        let want = Self.expectedRows(synthetic, hidden: rows, ids: routes.ids, weights: routes.weights, topK: shape.topK)
        let err = RelError.compute(actual: got, reference: want.flatMap { $0 })
        #expect(err < Self.routedTolerance, "rel err \(err)")
    }
}
