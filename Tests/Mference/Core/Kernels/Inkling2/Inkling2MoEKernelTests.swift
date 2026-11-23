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

    /// Bound for the routed sum against the FP32 reference: one FP16 rounding
    /// of the activation between two FP32 products (rel 2^-11 each), plus
    /// FP16-rounded Q8_0 weights in the grouped tiles.
    static let routedTolerance: Float = Tolerance.fp16Reduction

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

    /// 128 experts runs the planner's threadgroup as four simdgroups (thread
    /// 0's prefix sums published by the barrier); 16 as one.
    @Test(arguments: [16, 128])
    func routePlanGroupsEveryPairOnce(experts: Int) throws {
        let context = try MetalContext()
        let kernels = try Inkling2MoEKernels(context: context)
        let shape = Inkling2MoEShape(d: 256, f: 128, experts: experts, topK: 8)
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
        let counts2 = Inkling2MoETestSupport.read(scratch.workCount, as: UInt32.self, count: 2)
        let (fullCount, tailCount) = (Int(counts2[0]), Int(counts2[1]))
        let all = Inkling2MoETestSupport.read(scratch.work, as: Inkling2WorkBlock.self,
                                              count: scratch.fullBlockCapacity + shape.experts)
        let full = Array(all[0..<fullCount])
        let tail = Array(all[scratch.fullBlockCapacity..<(scratch.fullBlockCapacity + tailCount)])
        let dispatch = Inkling2MoETestSupport.read(scratch.dispatch, as: UInt32.self, count: 12)
        let gu = UInt32(shape.f / 64), dn = UInt32(shape.d / 128)
        #expect(dispatch == [gu, UInt32(fullCount), 1, gu, UInt32(tailCount), 1,
                             dn, UInt32(fullCount), 1, dn, UInt32(tailCount), 1])

        // An expert's remainder of 33-63 rows is one more full block; 1-32 rows a tail block.
        var counts = [Int](repeating: 0, count: shape.experts)
        for id in routes.ids { counts[Int(id)] += 1 }
        #expect(fullCount == counts.reduce(0) { $0 + $1 / 64 + ($1 % 64 > 32 ? 1 : 0) })
        #expect(tailCount == counts.filter { $0 % 64 > 0 && $0 % 64 <= 32 }.count)
        for block in tail { #expect(block.rowCount > 0 && block.rowCount <= 32) }
        var seen = Set<UInt32>()
        var covered = 0
        for block in full + tail {
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
        let tiedToken = 33   // an all-zero row: every logit 0, so the ties pick experts 0-7
        var rng = SeedTree(0x2A05).key("router")
        let weight = (0..<(128 * shape.d)).map { _ in rng.uniform(-0.1, 0.1) }
        let bias = (0..<128).map { _ in rng.uniform(-0.2, 0.2) }
        let xStride = shape.d + 4
        var x = (0..<(tokens * xStride)).map { _ in rng.uniform(-1, 1) }
        for i in 0..<xStride { x[tiedToken * xStride + i] = 0 }
        let tiedBias = bias.map { _ in Float(0) }
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
        // Ties need a zero bias too: route the tied row again on its own.
        let tiedIds = context.device.makeBuffer(length: 32, options: .storageModeShared)!
        let tiedWeights = context.device.makeBuffer(length: 32, options: .storageModeShared)!
        _ = Inkling2MoETestSupport.run(context) {
            kernels.encodeRouter($0, shape: shape, x: buffer(x), xOffset: tiedToken * xStride * 4, xStride: xStride,
                                 weight: buffer(weight), bias: buffer(tiedBias), ids: tiedIds,
                                 routeWeights: tiedWeights, tokens: 1)
        }
        #expect(Inkling2MoETestSupport.read(tiedIds, as: UInt32.self, count: 8) == [0, 1, 2, 3, 4, 5, 6, 7])
        #expect(Inkling2MoETestSupport.read(tiedWeights, as: Float.self, count: 8) == [Float](repeating: 1, count: 8))
        let gotIds = Inkling2MoETestSupport.read(ids, as: UInt32.self, count: tokens * 8)
        let gotWeights = Inkling2MoETestSupport.read(weights, as: Float.self, count: tokens * 8)
        for t in 0..<tokens {
            let row = Array(x[(t * xStride)..<(t * xStride + shape.d)])
            let logits = Inkling2RouterRef.logits(weight: weight, experts: 128, x: row)
            let route = Inkling2RouterRef.route(logits: logits, selectionBias: bias)
            #expect(gotIds[(t * 8)..<(t * 8 + 8)].map(Int.init) == route.ids, "token \(t)")
            let err = RelError.compute(actual: Array(gotWeights[(t * 8)..<(t * 8 + 8)]), reference: route.weights)
            #expect(err < Tolerance.identity, "token \(t) weights rel err \(err)")
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

    /// NVFP4 scale bytes over all 256 values (0x00, 0x7F, sign bit,
    /// subnormal and exponent-15 bytes; tile values up to 6 x 480), as the
    /// real file's bytes 95-126 sit at the top of the range. Per-expert
    /// scales near 1e-4, as in the file, bring the products back to O(1).
    @Test func nvfp4EveryScaleByteMatchesReference() throws {
        let context = try MetalContext()
        let kernels = try Inkling2MoEKernels(context: context)
        let shape = Inkling2MoEShape(d: 256, f: 128, experts: 8, topK: 4)
        let synthetic = try Inkling2SyntheticPool.make(device: context.device, encoding: .nvfp4, shape: shape,
                                                       seed: 0x2A0F, quantized: false, fullScaleRange: true,
                                                       expertScales: 5e-5...2e-4)
        let tokens = 70
        let (hidden, rows) = Inkling2MoETestSupport.halfBuffer(context.device, rows: tokens, stride: shape.d,
                                                               d: shape.d, seed: 0x2A10)
        let routes = Inkling2MoETestSupport.randomRoutes(tokens: tokens, experts: shape.experts, topK: shape.topK, seed: 0x2A11)
        let scratch = try Inkling2MoEPrefillScratch(device: context.device, shape: shape, maxTokens: tokens)
        Inkling2MoETestSupport.write(routes.ids, to: scratch.ids)
        Inkling2MoETestSupport.write(routes.weights, to: scratch.routeWeights)
        let out = context.device.makeBuffer(length: tokens * shape.d * 4, options: .storageModeShared)!
        let decodeOut = context.device.makeBuffer(length: shape.d * 4, options: .storageModeShared)!
        let activation = context.device.makeBuffer(length: shape.topK * shape.f * 2, options: .storageModeShared)!
        _ = Inkling2MoETestSupport.run(context) { encoder in
            kernels.encodePrefill(encoder, hidden: hidden, hiddenStride: shape.d, pool: synthetic.pool,
                                  scratch: scratch, tokens: tokens, out: out, outStride: shape.d)
        }
        _ = Inkling2MoETestSupport.run(context) {
            kernels.encodeDecode($0, shape: shape, x: hidden, ids: scratch.ids, routeWeights: scratch.routeWeights,
                                 pool: synthetic.pool, activation: activation, out: decodeOut)
        }
        let want = Self.expectedRows(synthetic, hidden: rows, ids: routes.ids, weights: routes.weights, topK: shape.topK)
        let got = Inkling2MoETestSupport.read(out, as: Float.self, count: tokens * shape.d)
        let err = RelError.compute(actual: got, reference: want.flatMap { $0 })
        #expect(err < Self.routedTolerance, "prefill rel err \(err)")
        let decodeErr = RelError.compute(actual: Inkling2MoETestSupport.read(decodeOut, as: Float.self, count: shape.d),
                                         reference: want[0])
        #expect(decodeErr < Self.routedTolerance, "decode rel err \(decodeErr)")
        #expect(!got.contains { !$0.isFinite })
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

    /// The input-quantization flag end to end against the reference's switch,
    /// through both grouped-GEMM variants, with the same FP16 inputs and
    /// FP16-rounded activations. The quantizer kernel itself is checked bit
    /// for bit below; here the activations come from
    /// different summation orders and a fast `tanh`, and a coarse quantizer
    /// turns an ulp on a rounding boundary into a whole step (E2M1: up to a
    /// third of the block maximum; Q8_0: 1/127). So both modes keep the
    /// aggregate (L1) error within `Tolerance.fp16Reduction`; the largest
    /// element error has the same bound for Q8_0 blocks and 0.1 for the
    /// vendor rule.
    @Test(arguments: [(Inkling2MoEEncoding.nvfp4, Inkling2MoEInputQuantization.q8_0Blocks),
                      (.nvfp4, .vendorRequired), (.q8_0, .q8_0Blocks), (.q8_0, .vendorRequired)],
          Inkling2MoEGEMMVariant.allCases)
    func inputQuantizationMatchesReference(_ c: (encoding: Inkling2MoEEncoding, mode: Inkling2MoEInputQuantization),
                                           variant: Inkling2MoEGEMMVariant) throws {
        let context = try MetalContext()
        let kernels = try Inkling2MoEKernels(context: context)
        guard kernels.availableVariants.contains(variant) else { return }
        kernels.variant = variant
        let shape = Self.small
        let synthetic = try Inkling2SyntheticPool.make(device: context.device, encoding: c.encoding, shape: shape,
                                                       seed: 0x2A20, quantized: true)
        let tokens = 40
        let (hidden, rows) = Inkling2MoETestSupport.halfBuffer(context.device, rows: tokens, stride: shape.d, d: shape.d, seed: 0x2A21)
        let routes = Inkling2MoETestSupport.randomRoutes(tokens: tokens, experts: shape.experts, topK: shape.topK, seed: 0x2A22)
        let scratch = try Inkling2MoEPrefillScratch(device: context.device, shape: shape, maxTokens: tokens)
        Inkling2MoETestSupport.write(routes.ids, to: scratch.ids)
        Inkling2MoETestSupport.write(routes.weights, to: scratch.routeWeights)
        let out = context.device.makeBuffer(length: tokens * shape.d * 4, options: .storageModeShared)!
        let decodeOut = context.device.makeBuffer(length: shape.d * 4, options: .storageModeShared)!
        let activation = context.device.makeBuffer(length: shape.topK * shape.f * 2, options: .storageModeShared)!
        let inputScratch = context.device.makeBuffer(length: shape.d * 2, options: .storageModeShared)!
        let cb = Inkling2MoETestSupport.run(context) { encoder in
            kernels.encodePrefill(encoder, hidden: hidden, hiddenStride: shape.d, pool: synthetic.pool, scratch: scratch,
                                  tokens: tokens, out: out, outStride: shape.d, inputQuantization: c.mode)
        }
        #expect(cb.status == .completed)
        _ = Inkling2MoETestSupport.run(context) {
            kernels.encodeDecode($0, shape: shape, x: hidden, ids: scratch.ids, routeWeights: scratch.routeWeights,
                                 pool: synthetic.pool, activation: activation, out: decodeOut,
                                 inputQuantization: c.mode, inputScratch: (inputScratch, 0))
        }
        let mode: Inkling2ExpertInputQuantization = c.mode == .q8_0Blocks ? .q8_0Blocks : .vendorRequired
        let experts = (0..<shape.experts).map { synthetic.expert($0) }
        let want: [[Float]] = rows.indices.map { t in
            var sum = [Float](repeating: 0, count: shape.d)
            for r in 0..<shape.topK {
                let expert = experts[Int(routes.ids[t * shape.topK + r])]
                let w = routes.weights[t * shape.topK + r]
                let act = Inkling2ExpertRef.activation(expert, x: rows[t], mode: mode).map { Float(Float16($0)) }
                let y = mode == .vendorRequired
                    ? Inkling2ExpertRef.down(expert, activation: act, preScale: w, mode: mode)
                    : Inkling2ExpertRef.down(expert, activation: act, mode: mode).map { $0 * w }
                for i in 0..<shape.d { sum[i] += y[i] }
            }
            return sum
        }
        let vendor = c.mode == .vendorRequired
        let bound = vendor ? 0.1 : Tolerance.fp16Reduction
        let got = Inkling2MoETestSupport.read(out, as: Float.self, count: tokens * shape.d)
        let flat = want.flatMap { $0 }
        let err = RelError.compute(actual: got, reference: flat)
        #expect(err < bound, "prefill rel err \(err)")
        let l1 = zip(got, flat).reduce(Float(0)) { $0 + abs($1.0 - $1.1) } / flat.reduce(Float(0)) { $0 + abs($1) }
        #expect(l1 < Tolerance.fp16Reduction, "prefill L1 rel err \(l1)")
        let decoded = Inkling2MoETestSupport.read(decodeOut, as: Float.self, count: shape.d)
        let decodeErr = RelError.compute(actual: decoded, reference: want[0])
        #expect(decodeErr < bound, "decode rel err \(decodeErr)")
        let decodeL1 = zip(decoded, want[0]).reduce(Float(0)) { $0 + abs($1.0 - $1.1) } / want[0].reduce(Float(0)) { $0 + abs($1) }
        #expect(decodeL1 < Tolerance.fp16Reduction, "decode L1 rel err \(decodeL1)")
        // The pass changes the result: the unquantized output differs.
        let plain = Self.expectedRows(synthetic, hidden: rows, ids: routes.ids, weights: routes.weights, topK: shape.topK)
        #expect(RelError.compute(actual: got, reference: plain.flatMap { $0 }) > 1e-4)
    }

    /// The input-quantization kernel on its own, bit for bit against the
    /// reference quantizer on the same FP16 inputs (rounded to FP16 as the
    /// kernel stores them): many rows, with and without row weights (read at
    /// a 16-byte stride as from `I2Pair`), at small and production widths.
    @Test(arguments: [(Inkling2MoEEncoding.nvfp4, Inkling2MoEInputQuantization.vendorRequired),
                      (.nvfp4, .q8_0Blocks), (.q8_0, .vendorRequired)], [128, 1152, 2304])
    func inputQuantizationKernelMatchesReference(_ c: (encoding: Inkling2MoEEncoding, mode: Inkling2MoEInputQuantization),
                                                 width: Int) throws {
        let context = try MetalContext()
        let kernels = try Inkling2MoEKernels(context: context)
        let rows = width == 128 ? 160 : 24
        let (source, values) = Inkling2MoETestSupport.halfBuffer(context.device, rows: rows, stride: width, d: width,
                                                                 seed: 0x2A30 + UInt64(width), range: 3)
        var rng = SeedTree(0x2A31).key("row-weights")
        let weights = (0..<rows).map { _ in rng.uniform(0.2, 2) }
        var records = [Float](repeating: 0, count: rows * 4)
        for r in 0..<rows { records[r * 4 + 2] = weights[r] }
        let weightBuffer = records.withUnsafeBytes {
            context.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
        }
        let mode: Inkling2ExpertInputQuantization = c.mode == .q8_0Blocks ? .q8_0Blocks : .vendorRequired
        let encoding: Inkling2ExpertEncoding = c.encoding == .nvfp4 ? .nvfp4 : .q8_0
        for weighted in [false, true] {
            let destination = context.device.makeBuffer(length: rows * width * 2, options: .storageModeShared)!
            _ = Inkling2MoETestSupport.run(context) {
                kernels.encodeInputQuantization($0, encoding: c.encoding, mode: c.mode, source: source, sourceStride: width,
                                                destination: destination, destinationStride: width, rows: rows, width: width,
                                                rowWeights: weighted ? (weightBuffer, 8, 4) : nil)
            }
            let got = Inkling2MoETestSupport.read(destination, as: Float16.self, count: rows * width).map(\.bitPattern)
            var mismatches = 0, firstRow = -1
            for r in 0..<rows {
                let input = weighted ? values[r].map { $0 * weights[r] } : values[r]
                let want = Inkling2ExpertRef.quantizedInput(input, encoding: encoding, mode: mode).map { Float16($0).bitPattern }
                let row = Array(got[(r * width)..<((r + 1) * width)])
                let bad = zip(row, want).filter { $0 != $1 }.count
                if bad > 0 && firstRow < 0 { firstRow = r }
                mismatches += bad
            }
            #expect(mismatches == 0, "weighted \(weighted): \(mismatches) values differ, first in row \(firstRow)")
        }
    }

    /// The router with fewer experts than its 128 lanes (K4's toy has 16):
    /// rows past the count are never chosen.
    @Test func routerWithSixteenExpertsMatchesReference() throws {
        let context = try MetalContext()
        let kernels = try Inkling2MoEKernels(context: context)
        let shape = Inkling2MoEShape(d: 128, f: 64, experts: 16, topK: 4)
        let tokens = 45
        var rng = SeedTree(0x2A44).key("router16")
        let weight = (0..<(16 * shape.d)).map { _ in rng.uniform(-0.2, 0.2) }
        let bias = (0..<16).map { _ in rng.uniform(-0.2, 0.2) }
        let x = (0..<(tokens * shape.d)).map { _ in rng.uniform(-1, 1) }
        func buffer(_ v: [Float]) -> MTLBuffer {
            v.withUnsafeBytes { context.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
        }
        let ids = context.device.makeBuffer(length: tokens * 16, options: .storageModeShared)!
        let weights = context.device.makeBuffer(length: tokens * 16, options: .storageModeShared)!
        _ = Inkling2MoETestSupport.run(context) {
            kernels.encodeRouter($0, shape: shape, x: buffer(x), xStride: shape.d, weight: buffer(weight),
                                 bias: buffer(bias), ids: ids, routeWeights: weights, tokens: tokens)
        }
        let gotIds = Inkling2MoETestSupport.read(ids, as: UInt32.self, count: tokens * 4)
        let gotWeights = Inkling2MoETestSupport.read(weights, as: Float.self, count: tokens * 4)
        for t in 0..<tokens {
            let row = Array(x[(t * shape.d)..<((t + 1) * shape.d)])
            let route = Inkling2RouterRef.route(logits: Inkling2RouterRef.logits(weight: weight, experts: 16, x: row),
                                                selectionBias: bias, topK: 4)
            #expect(gotIds[(t * 4)..<(t * 4 + 4)].map(Int.init) == route.ids, "token \(t)")
            #expect(RelError.compute(actual: Array(gotWeights[(t * 4)..<(t * 4 + 4)]), reference: route.weights)
                    < Tolerance.identity)
        }
    }

    /// The tensor-ops (`matmul2d`) grouped kernels against the reference,
    /// where this GPU builds them (full and tail blocks at 200 tokens).
    @Test(arguments: [Inkling2MoEEncoding.nvfp4, .q8_0])
    func tensorOpsMatchReference(encoding: Inkling2MoEEncoding) throws {
        let context = try MetalContext()
        let kernels = try Inkling2MoEKernels(context: context)
        guard kernels.availableVariants.contains(.tensorOps) else {
            print("[inkling2-moe] tensor-ops kernels not built on \(context.device.name); simdgroup path only")
            return
        }
        kernels.variant = .tensorOps
        let shape = Self.small
        let synthetic = try Inkling2SyntheticPool.make(device: context.device, encoding: encoding, shape: shape,
                                                       seed: 0x2A40, quantized: true)
        let tokens = 200
        let (hidden, rows) = Inkling2MoETestSupport.halfBuffer(context.device, rows: tokens, stride: shape.d, d: shape.d, seed: 0x2A41)
        let routes = Inkling2MoETestSupport.randomRoutes(tokens: tokens, experts: shape.experts, topK: shape.topK, seed: 0x2A42)
        let scratch = try Inkling2MoEPrefillScratch(device: context.device, shape: shape, maxTokens: tokens)
        Inkling2MoETestSupport.write(routes.ids, to: scratch.ids)
        Inkling2MoETestSupport.write(routes.weights, to: scratch.routeWeights)
        let out = context.device.makeBuffer(length: tokens * shape.d * 4, options: .storageModeShared)!
        let cb = Inkling2MoETestSupport.run(context) { encoder in
            kernels.encodePrefill(encoder, hidden: hidden, hiddenStride: shape.d, pool: synthetic.pool,
                                  scratch: scratch, tokens: tokens, out: out, outStride: shape.d)
        }
        #expect(cb.status == .completed, "\(String(describing: cb.error))")
        let got = Inkling2MoETestSupport.read(out, as: Float.self, count: tokens * shape.d)
        let want = Self.expectedRows(synthetic, hidden: rows, ids: routes.ids, weights: routes.weights, topK: shape.topK)
        let err = RelError.compute(actual: got, reference: want.flatMap { $0 })
        #expect(err < Self.routedTolerance, "rel err \(err)")
    }

    @Test func calibrationPicksAnAvailableVariantAndKeepsIt() throws {
        let context = try MetalContext()
        let kernels = try Inkling2MoEKernels(context: context)
        let shape = Self.small
        let synthetic = try Inkling2SyntheticPool.make(device: context.device, encoding: .nvfp4, shape: shape,
                                                       seed: 0x2A43, quantized: true)
        let scratch = try Inkling2MoEPrefillScratch(device: context.device, shape: shape, maxTokens: 96)
        let first = try Inkling2MoECalibration.calibrate(context: context, kernels: kernels, pool: synthetic.pool,
                                                         scratch: scratch)
        #expect(kernels.availableVariants.contains(first.variant))
        #expect(kernels.variant == first.variant)
        if kernels.availableVariants.count > 1 {
            #expect(Set(first.seconds.keys) == Set(kernels.availableVariants))
            #expect(first.seconds.values.allSatisfy { $0 > 0 && $0.isFinite })
        }
        let other = try Inkling2MoEKernels(context: context)
        let second = try Inkling2MoECalibration.calibrate(context: context, kernels: other, pool: synthetic.pool,
                                                          scratch: scratch)
        #expect(second.variant == first.variant && other.variant == first.variant)
    }

    /// The pipelined loop (default) and the plain loop compute the same
    /// thing in the same order: bit-identical outputs, both encodings, with
    /// full and tail blocks.
    @Test(arguments: [Inkling2MoEEncoding.nvfp4, .q8_0])
    func pipelinedLoopMatchesPlainLoop(encoding: Inkling2MoEEncoding) throws {
        let context = try MetalContext()
        let pipelined = try Inkling2MoEKernels(context: context, pipelined: true)
        let plain = try Inkling2MoEKernels(context: context, pipelined: false)
        let shape = Self.small
        let synthetic = try Inkling2SyntheticPool.make(device: context.device, encoding: encoding, shape: shape,
                                                       seed: 0x2A50, quantized: true)
        let tokens = 200
        let (hidden, _) = Inkling2MoETestSupport.halfBuffer(context.device, rows: tokens, stride: shape.d, d: shape.d, seed: 0x2A51)
        let routes = Inkling2MoETestSupport.randomRoutes(tokens: tokens, experts: shape.experts, topK: shape.topK, seed: 0x2A52)
        let scratch = try Inkling2MoEPrefillScratch(device: context.device, shape: shape, maxTokens: tokens)
        Inkling2MoETestSupport.write(routes.ids, to: scratch.ids)
        Inkling2MoETestSupport.write(routes.weights, to: scratch.routeWeights)
        var outputs: [[Float]] = []
        for kernels in [pipelined, plain] {
            let out = context.device.makeBuffer(length: tokens * shape.d * 4, options: .storageModeShared)!
            _ = Inkling2MoETestSupport.run(context) { encoder in
                kernels.encodePrefill(encoder, hidden: hidden, hiddenStride: shape.d, pool: synthetic.pool,
                                      scratch: scratch, tokens: tokens, out: out, outStride: shape.d)
            }
            outputs.append(Inkling2MoETestSupport.read(out, as: Float.self, count: tokens * shape.d))
        }
        #expect(outputs[0] == outputs[1])
    }
}
