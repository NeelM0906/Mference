import Testing
import Foundation
import MferenceValidationSupport

/// The expert, activation-quantizer and router references, each against an
/// independent scalar computation in Double or a property of the format.
@Suite struct Inkling2ExpertReferenceTests {
    static func makeExpert(_ encoding: Inkling2ExpertEncoding, d: Int, f: Int,
                           rng: inout SplitMix64, gateUpScale: Float, downScale: Float) -> Inkling2ExpertWeights {
        var gateUp: [UInt8] = [], down: [UInt8] = []
        for _ in 0..<(2 * f) { gateUp += encoding.quantize((0..<d).map { _ in rng.uniform(-6, 6) }) }
        for _ in 0..<d { down += encoding.quantize((0..<f).map { _ in rng.uniform(-6, 6) }) }
        return Inkling2ExpertWeights(encoding: encoding, d: d, f: f, gateUp: gateUp, down: down,
                                     gateUpScale: gateUpScale, downScale: downScale)
    }

    /// Double-precision scalar expert, written from the formula.
    static func scalarExpert(_ e: Inkling2ExpertWeights, x: [Float]) -> [Double] {
        let gu = e.decodedGateUp(), dn = e.decodedDown()
        var act = [Double](repeating: 0, count: e.f)
        for i in 0..<e.f {
            var g = 0.0, u = 0.0
            for k in 0..<e.d {
                g += Double(gu[i * e.d + k]) * Double(x[k])
                u += Double(gu[(e.f + i) * e.d + k]) * Double(x[k])
            }
            g *= Double(e.gateUpScale); u *= Double(e.gateUpScale)
            act[i] = 4 * tanh(g / 4) / (1 + exp(-g)) * 25 * tanh(u / 25)
        }
        return (0..<e.d).map { n in
            var y = 0.0
            for i in 0..<e.f { y += Double(dn[n * e.f + i]) * act[i] }
            return y * Double(e.downScale)
        }
    }

    @Test(arguments: [Inkling2ExpertEncoding.nvfp4, .q8_0])
    func expertMatchesScalarFormula(encoding: Inkling2ExpertEncoding) {
        var rng = SeedTree(0x1D2).key("expert-\(encoding)")
        // Scales of the order of the real ones (amax / denominator ~ 1e-3).
        let expert = Self.makeExpert(encoding, d: 256, f: 128, rng: &rng,
                                     gateUpScale: 1.7e-3, downScale: 2.3e-3)
        let x = (0..<256).map { _ in rng.uniform(-4, 4) }
        let got = Inkling2ExpertRef.forward(expert, x: x)
        let want = Self.scalarExpert(expert, x: x)
        let err = RelError.compute(actual: got, reference: want.map(Float.init))
        #expect(err < 1e-5, "rel err \(err)")
    }

    /// The per-expert scale is applied to the FP32 product before the
    /// activation: scaling it by 2 is not the same as scaling the output.
    @Test func gateUpScaleIsAppliedBeforeTheActivation() {
        var rng = SeedTree(0x1D3).key("scale-order")
        let a = Self.makeExpert(.nvfp4, d: 128, f: 64, rng: &rng, gateUpScale: 1e-2, downScale: 1)
        let b = Inkling2ExpertWeights(encoding: .nvfp4, d: 128, f: 64, gateUp: a.gateUp, down: a.down,
                                      gateUpScale: 2e-2, downScale: 1)
        let x = (0..<128).map { _ in rng.uniform(-1, 1) }
        let ya = Inkling2ExpertRef.forward(a, x: x), yb = Inkling2ExpertRef.forward(b, x: x)
        let ratios = zip(yb, ya).filter { abs($0.1) > 1e-3 }.map { $0.0 / $0.1 }
        #expect(ratios.contains { abs($0 - 2) > 0.05 }, "the activation is not linear in the scale")
    }

    @Test func situMatchesDefinition() {
        for (g, u) in [(Float(0), Float(0)), (1, 1), (-3, 7), (40, -80), (-200, 300), (2.5, 0.1)] {
            let want = 4 * tanh(Double(g) / 4) / (1 + exp(-Double(g))) * 25 * tanh(Double(u) / 25)
            #expect(abs(Double(Inkling2ExpertRef.situ(gate: g, up: u)) - want) <= 1e-5 * max(1, abs(want)))
        }
        // Bounded by 4 * 25 whatever the inputs.
        #expect(abs(Inkling2ExpertRef.situ(gate: 1e6, up: -1e6)) <= 100)
    }

    @Test func routedSumWeightsAndOrder() {
        var rng = SeedTree(0x1D4).key("routed")
        let experts = (0..<4).map { _ in
            Self.makeExpert(.q8_0, d: 64, f: 64, rng: &rng, gateUpScale: 0.01, downScale: 0.01)
        }
        let x = (0..<64).map { _ in rng.uniform(-1, 1) }
        let ids = [2, 0, 3], weights: [Float] = [3.5, 2.5, 2]
        let got = Inkling2ExpertRef.routedSum(experts: { experts[$0] }, ids: ids, weights: weights, x: x)
        var want = [Float](repeating: 0, count: 64)
        for (r, id) in ids.enumerated() {
            let y = Inkling2ExpertRef.forward(experts[id], x: x)
            for i in 0..<64 { want[i] += y[i] * weights[r] }
        }
        #expect(RelError.compute(actual: got, reference: want) < 1e-6)
    }

    @Test func q8_0InputModeRoundsTheInput() {
        var rng = SeedTree(0x1D5).key("q8-input")
        let x = (0..<256).map { _ in rng.uniform(-3, 3) }
        let q = Inkling2ExpertRef.quantizedInput(x, encoding: .nvfp4, mode: .q8_0Blocks)
        #expect(q == Inkling2Q8_0.roundTrip(x))
        #expect(q != x)
        #expect(Inkling2ExpertRef.quantizedInput(x, encoding: .nvfp4, mode: .none) == x)
    }

    @Test func roundingHelpersAreNearestEven() {
        #expect(Inkling2ActivationQuant.roundE2M1(0.25) == 0)
        #expect(Inkling2ActivationQuant.roundE2M1(0.26) == 0.5)
        #expect(Inkling2ActivationQuant.roundE2M1(0.75) == 1)
        #expect(Inkling2ActivationQuant.roundE2M1(1.25) == 1)
        #expect(Inkling2ActivationQuant.roundE2M1(2.5) == 2)
        #expect(Inkling2ActivationQuant.roundE2M1(3.5) == 4)
        #expect(Inkling2ActivationQuant.roundE2M1(5) == 4)
        #expect(Inkling2ActivationQuant.roundE2M1(-5.5) == -6)
        #expect(Inkling2ActivationQuant.roundE2M1(100) == 6)
        // E4M3 spacing is 2^(e-3); subnormal spacing 2^-9.
        #expect(Inkling2ActivationQuant.roundE4M3(1.0625) == 1)      // tie to even
        #expect(Inkling2ActivationQuant.roundE4M3(1.1875) == 1.25)   // tie to even
        #expect(Inkling2ActivationQuant.roundE4M3(448) == 448)
        #expect(Inkling2ActivationQuant.roundE4M3(3 * pow(2, -10)) == pow(2, -8))
    }

    @Test(arguments: [UInt64(7), 8, 9])
    func fourOverSixPicksTheSmallerError(seed: UInt64) {
        var rng = SeedTree(seed).key("4over6")
        let x = (0..<1152).map { _ in rng.uniform(-1, 1) * (rng.uniform(0, 1) < 0.05 ? 20 : 1) }
        let q = Inkling2ActivationQuant.nvfp4FourOverSix(x)
        let outer = q.rowMax / 1536
        #expect(q.rowMax == x.map(abs).max()!)
        for b in 0..<(1152 / 16) {
            // Every value is an E2M1 code times one E4M3 scale.
            let block = q.inner[(b * 16)..<((b + 1) * 16)]
            let scale = block.map(abs).max()! / (q.usedFourRange[b] ? 4 : 6)
            if scale > 0 {
                #expect(Inkling2ActivationQuant.roundE4M3(scale) == scale, "block \(b) scale")
                for v in block { #expect(Inkling2ActivationQuant.roundE2M1(v / scale) * scale == v) }
            }
            // Its error is no larger than the 6-range candidate's.
            let error = (0..<16).reduce(Float(0)) { $0 + abs(x[b * 16 + $1] - q.inner[b * 16 + $1] * outer) }
            let blockMax = x[(b * 16)..<((b + 1) * 16)].map(abs).max()!
            let s6 = Inkling2ActivationQuant.roundE4M3(min(blockMax * (1536 / q.rowMax) / 6, 448))
            let e6 = (0..<16).reduce(Float(0)) {
                let v = x[b * 16 + $1]
                return $0 + abs(v - Inkling2ActivationQuant.roundE2M1(v / (s6 * outer)) * s6 * outer)
            }
            #expect(error <= e6 * (1 + 1e-5) + 1e-7, "block \(b)")
        }
        // Restored input: within half the widest E2M1 gap of each block's
        // scale (blockMax / 6, E4M3-rounded), for either range.
        let restored = Inkling2ExpertRef.quantizedInput(x, encoding: .nvfp4, mode: .vendorRequired)
        for b in 0..<(1152 / 16) {
            let blockMax = x[(b * 16)..<((b + 1) * 16)].map(abs).max()!
            for i in (b * 16)..<((b + 1) * 16) {
                #expect(abs(restored[i] - x[i]) <= blockMax / 6 * (1 + 1.0 / 16) * 1.0001, "element \(i)")
            }
        }
    }

    @Test func mxfp8RoundTripIsBounded() {
        var rng = SeedTree(0x1D6).key("mxfp8")
        let x = (0..<2304).map { _ in rng.uniform(-50, 50) }
        let q = Inkling2ActivationQuant.mxfp8(x)
        for b in 0..<(2304 / 32) {
            let amax = x[(b * 32)..<((b + 1) * 32)].map(abs).max()!
            for i in (b * 32)..<((b + 1) * 32) {
                // Three mantissa bits: half an ulp is at most 1/16 of the
                // value's binade, and the block scale keeps every value normal
                // or within one subnormal step of 2^-9 * scale.
                let bound = max(abs(x[i]) / 16, amax / 448 * pow(2, -8))
                #expect(abs(q[i] - x[i]) <= bound * 1.0001, "element \(i)")
            }
        }
        #expect(Inkling2ActivationQuant.mxfp8([Float](repeating: 0, count: 32)) == [Float](repeating: 0, count: 32))
    }

    @Test func routerTopKAndWeights() {
        var rng = SeedTree(0x1D7).key("router")
        let logits = (0..<128).map { _ in rng.uniform(-4, 4) }
        let bias = (0..<128).map { _ in rng.uniform(-0.5, 0.5) }
        let route = Inkling2RouterRef.route(logits: logits, selectionBias: bias)
        #expect(route.ids.count == 8 && Set(route.ids).count == 8)
        let probs = logits.map { 1 / (1 + exp(-$0)) }
        let selection = zip(probs, bias).map(+)
        let threshold = route.ids.map { selection[$0] }.min()!
        #expect(selection.enumerated().filter { !route.ids.contains($0.offset) }.allSatisfy { $0.element <= threshold })
        #expect(abs(route.weights.reduce(0, +) - 8) < 1e-5)
        for (i, id) in route.ids.enumerated() {
            #expect(abs(route.weights[i] / route.weights[0] - probs[id] / probs[route.ids[0]]) < 1e-5)
        }
        // A tie takes the lower expert.
        let tied = Inkling2RouterRef.route(logits: [Float](repeating: 0, count: 16),
                                           selectionBias: [Float](repeating: 0, count: 16), topK: 3)
        #expect(tied.ids == [0, 1, 2])
    }
}
