import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

/// The prefill tail blocks (an expert's last 1-32 routed rows) multiply only the 8-row fragments
/// that hold a routed row (`FC_I2_TAIL_SKIP`). The fragments in use run the same multiplies in the
/// same order, so a prefill with the skip writes the same bits as one with the plain loop
/// (`tailFragmentSkip: false`): every activation row (keyed by its pair's token and rank) and every
/// output row, both encodings. Token
/// counts give all-tail plans, full blocks with tails, and partly filled full blocks; the small
/// shape also checks the CPU reference.
@Suite(.serialized) struct Inkling2MoETailTests {
    private static func compare(shape: Inkling2MoEShape, encoding: Inkling2MoEEncoding, tokens: Int,
                                seed: UInt64, reference: Bool, expectsTails: Bool = true,
                                fixedRoutes: (ids: [UInt32], weights: [Float])? = nil) throws {
        let context = try MetalContext()
        let skipping = try Inkling2MoEKernels(context: context)
        let plain = try Inkling2MoEKernels(context: context, tailFragmentSkip: false)
        let synthetic = try Inkling2SyntheticPool.make(device: context.device, encoding: encoding, shape: shape,
                                                       seed: seed, quantized: reference)
        let (hidden, rows) = Inkling2MoETestSupport.halfBuffer(context.device, rows: tokens, stride: shape.d,
                                                               d: shape.d, seed: seed &+ 1)
        let routes = fixedRoutes ?? Inkling2MoETestSupport.randomRoutes(tokens: tokens, experts: shape.experts,
                                                                        topK: shape.topK, seed: seed &+ 2)
        var outputs: [[UInt32]] = [], activations: [[UInt16]] = []
        var tails = 0
        for kernels in [plain, skipping] {
            let scratch = try Inkling2MoEPrefillScratch(device: context.device, shape: shape, maxTokens: tokens)
            Inkling2MoETestSupport.write(routes.ids, to: scratch.ids)
            Inkling2MoETestSupport.write(routes.weights, to: scratch.routeWeights)
            let out = context.device.makeBuffer(length: tokens * shape.d * 4, options: .storageModeShared)!
            memset(out.contents(), 0xCD, out.length)
            memset(scratch.activation.contents(), 0xAB, scratch.activation.length)
            let cb = Inkling2MoETestSupport.run(context) {
                kernels.encodePrefill($0, hidden: hidden, hiddenStride: shape.d, pool: synthetic.pool,
                                      scratch: scratch, tokens: tokens, out: out, outStride: shape.d)
            }
            #expect(cb.status == .completed)
            outputs.append(Inkling2MoETestSupport.read(out, as: UInt32.self, count: tokens * shape.d))
            // Activation rows follow the plan's pair order, which the scatter's atomics leave
            // unfixed: key each row by its pair's (token, rank).
            let pairCount = tokens * shape.topK
            let pairWords = Inkling2MoETestSupport.read(scratch.pairs, as: UInt32.self, count: pairCount * 4)
            let raw = Inkling2MoETestSupport.read(scratch.activation, as: UInt16.self, count: pairCount * shape.f)
            var keyed = [UInt16](repeating: 0, count: pairCount * shape.f)
            for row in 0..<pairCount {
                let key = Int(pairWords[row * 4]) * shape.topK + Int(pairWords[row * 4 + 1])
                for c in 0..<shape.f { keyed[key * shape.f + c] = raw[row * shape.f + c] }
            }
            activations.append(keyed)
            tails = Int(Inkling2MoETestSupport.read(scratch.workCount, as: UInt32.self, count: 2)[1])
        }
        let outDiff = zip(outputs[0], outputs[1]).filter { $0 != $1 }.count
        let actDiff = zip(activations[0], activations[1]).filter { $0 != $1 }.count
        if expectsTails { #expect(tails > 0, "the plan has no tail blocks") }
        #expect(actDiff == 0, "\(encoding) \(tokens) tokens: \(actDiff) activation halves differ")
        #expect(outDiff == 0, "\(encoding) \(tokens) tokens: \(outDiff) outputs differ")
        print("[ink2-moe-tail] \(encoding) d \(shape.d) f \(shape.f) experts \(shape.experts) top-\(shape.topK), "
              + "\(tokens) tokens, \(tails) tail blocks: \(actDiff) activations and \(outDiff) outputs differ")
        guard reference else { return }
        let experts = (0..<shape.experts).map { synthetic.expert($0) }
        let got = outputs[1].map { Float(bitPattern: $0) }
        var want: [Float] = []
        for t in 0..<tokens {
            want += Inkling2ExpertRef.routedSum(experts: { experts[$0] },
                                                ids: (0..<shape.topK).map { Int(routes.ids[t * shape.topK + $0]) },
                                                weights: Array(routes.weights[(t * shape.topK)..<((t + 1) * shape.topK)]),
                                                x: rows[t])
        }
        let err = RelError.computeStrict(actual: got, reference: want)
        #expect(err < Tolerance.fp16Reduction, "rel err \(err)")
    }

    /// 40 tokens x top-4 over 16 experts: about 10 rows each, all tails. 300 tokens: full blocks
    /// and tails. Against the CPU reference too.
    @Test(arguments: [(Inkling2MoEEncoding.nvfp4, 40), (.nvfp4, 300), (.q8_0, 40), (.q8_0, 300)])
    func smallShapeSameBitsAndReference(_ c: (encoding: Inkling2MoEEncoding, tokens: Int)) throws {
        try Self.compare(shape: Inkling2MoEShape(d: 256, f: 128, experts: 16, topK: 4), encoding: c.encoding,
                         tokens: c.tokens, seed: 0x7A11 + UInt64(c.tokens), reference: true)
    }

    /// Every tail size: top-1 over 32 experts, expert e taking exactly e + 1 tokens, so the tail
    /// blocks hold 1 to 32 rows and the fragment switch takes each of its four cases.
    @Test(arguments: [Inkling2MoEEncoding.nvfp4, .q8_0])
    func everyTailSizeSameBitsAndReference(_ encoding: Inkling2MoEEncoding) throws {
        var ids: [UInt32] = []
        for e in 0..<32 { ids += [UInt32](repeating: UInt32(e), count: e + 1) }
        try Self.compare(shape: Inkling2MoEShape(d: 256, f: 128, experts: 32, topK: 1), encoding: encoding,
                         tokens: ids.count, seed: 0x7A31, reference: true,
                         fixedRoutes: (ids, [Float](repeating: 1, count: ids.count)))
    }

    /// Inkling 2 Mini's d and f, 16 experts, top-8: 100 tokens (about 50 rows each: partly filled
    /// full blocks and tails) and 300 (two full blocks and a tail each). Full tier: the random
    /// pools are 72 MB (NVFP4) and 135 MB (Q8_0) per case in a debug build.
    @Test(.fullTier, arguments: [(Inkling2MoEEncoding.nvfp4, 100), (.nvfp4, 300), (.q8_0, 100), (.q8_0, 300)])
    func miniWidthsSameBits(_ c: (encoding: Inkling2MoEEncoding, tokens: Int)) throws {
        try Self.compare(shape: Inkling2MoEShape(d: 2304, f: 1152, experts: 16, topK: 8), encoding: c.encoding,
                         tokens: c.tokens, seed: 0x7A21 + UInt64(c.tokens), reference: false,
                         expectsTails: c.tokens != 100)
    }
}
