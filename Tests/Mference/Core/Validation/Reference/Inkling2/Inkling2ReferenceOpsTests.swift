import Foundation
import Testing
import MferenceValidationSupport

/// Hand-computed cases for each operation of the Inkling 2 CPU reference,
/// so the whole-model tests rest on checked pieces.
@Suite("Inkling2ReferenceOps")
struct Inkling2ReferenceOpsTests {
    typealias Op = Inkling2ReferenceOps

    private func close(_ a: Float, _ b: Float, _ tol: Float = 1e-6) -> Bool { abs(a - b) <= tol }

    @Test func shortConvStaticTapsWeightTheCurrentTokenLast() {
        // tail oldest first: x[t-3] = 1, x[t-2] = 2, x[t-1] = 3; x[t] = 4.
        let out = Op.shortConv([4], tail: [[1], [2], [3]], weights: [0.1, 0.2, 0.3, 0.4])
        let conv: Float = 0.1 * 1 + 0.2 * 2 + 0.3 * 3 + 0.4 * 4
        #expect(close(out[0], (4 + conv) / Float(5).squareRoot()))
    }

    @Test func shortConvDynamicTapsUseLagOrderAndTheWiderDivisor() {
        let staticW: [Float] = [0.1, 0.2, 0.3, 0.4]
        let dyn: [Float] = [1, 0.5, 0.25, 0.125]   // lag 0 weights x[t]
        let out = Op.shortConv([4], tail: [[1], [2], [3]], weights: staticW, dynamicTaps: dyn, heads: 1)
        let conv: Float = (0.1 * 1 + 0.2 * 2 + 0.3 * 3 + 0.4 * 4) + (1 * 4 + 0.5 * 3 + 0.25 * 2 + 0.125 * 1)
        #expect(close(out[0], (4 + conv) / 3, 1e-5))
    }

    @Test func shortConvDynamicTapsAreSharedPerHead() {
        // Two channels per head, two heads: head 1's taps must not touch head 0.
        let x: [Float] = [1, 1, 1, 1]
        let tail = [[Float]](repeating: [0, 0, 0, 0], count: 3)
        let staticW = [Float](repeating: 0, count: 16)
        let dyn: [Float] = [0, 0, 0, 0,  2, 0, 0, 0]   // head 1, lag 0 = 2
        let out = Op.shortConv(x, tail: tail, weights: staticW, dynamicTaps: dyn, heads: 2)
        #expect(close(out[0], 1 / 3) && close(out[1], 1 / 3))
        #expect(close(out[2], 3 / 3) && close(out[3], 3 / 3))
    }

    @Test func advanceDropsTheOldestInput() {
        #expect(Op.advance(tail: [[1], [2], [3]], with: [4]) == [[2], [3], [4]])
    }

    @Test func l2ClipScalesOnlyLongVectors() {
        let clipped = Op.l2Clip([3, 4], threshold: 2.5)
        #expect(close(clipped[0], 1.5, 1e-5) && close(clipped[1], 2.0, 1e-5))
        #expect(Op.l2Clip([3, 4], threshold: 10) == [3, 4])
        #expect(Op.l2ClipActive([3, 4], threshold: 2.5))
        #expect(!Op.l2ClipActive([3, 4], threshold: 10))
        // The per-channel scale applies before the norm is taken: [6, 8] has
        // norm 10 and is clipped to 7; scaling after a clip of [3, 4] (norm 5,
        // unclipped) would give [6, 8].
        let scaled = Op.l2Clip([3, 4], scale: [2, 2], threshold: 7)
        #expect(close(scaled[0], 4.2, 1e-5) && close(scaled[1], 5.6, 1e-5))
    }

    @Test func logScaleStartsAtTheFloor() {
        #expect(Op.logScale(position: 0, floor: 4, alpha: 0.1) == 1)
        #expect(Op.logScale(position: 3, floor: 4, alpha: 0.1) == 1)
        #expect(close(Op.logScale(position: 7, floor: 4, alpha: 0.1), 1 + 0.1 * Float(log(2.0))))
    }

    @Test func mixOfOneStateIsThatStateNormalized() {
        let s: [Float] = [1, 2, 3, 4]
        let out = Op.mix(states: [s], queries: [[5, -1, 0, 2]])
        let n = Op.rmsNorm(s, eps: Op.residualNormEps)
        for i in 0..<4 { #expect(close(out[0][i], n[i])) }
    }

    @Test func mixWithAZeroQueryAveragesTheNormalizedStates() {
        let a: [Float] = [1, 0, 0, 0]
        let b: [Float] = [0, 2, 0, 0]
        let out = Op.mix(states: [a, b], queries: [[0, 0, 0, 0], [0, 0, 0, 0]])
        let na = Op.rmsNorm(a, eps: Op.residualNormEps)
        let nb = Op.rmsNorm(b, eps: Op.residualNormEps)
        #expect(out.count == 2)
        for i in 0..<4 { #expect(close(out[1][i], (na[i] + nb[i]) / 2)) }
    }

    @Test func mixScoresWithInverseSqrtHidden() {
        // States already unit-RMS; query picks state a with score 4/2 = 2.
        let a: [Float] = [1, 1, 1, 1]
        let b: [Float] = [1, 1, -1, -1]
        let out = Op.mix(states: [a, b], queries: [[1, 1, 1, 1]])
        let pa = Float(exp(2.0) / (exp(2.0) + exp(0.0)))
        for i in 0..<4 { #expect(close(out[0][i], pa * a[i] + (1 - pa) * b[i], 1e-5)) }
    }

    @Test func silentControllerReturnsItsInput() {
        let zero = { (r: Int, c: Int) in Inkling2DenseMatrix(rows: r, cols: c, values: [Float](repeating: 0, count: r * c)) }
        let x: [Float] = [0.5, -1, 2, 3]
        let m: [Float] = [9, 9, 9, 9]
        let out = Op.control(x, message: m,
                             down: Inkling2DenseMatrix(rows: 2, cols: 8, values: (0..<16).map { Float($0) / 10 }),
                             gain: zero(4, 2), forget: zero(4, 2))
        #expect(out == x)
    }

    @Test func controllerMixesWithTanhGates() {
        // rms(x) = [√2, 0], rms(m) = [0, √2]; z = [g0, 2·g3] = [√2, 2√2] with
        // the [rms(x); rms(m)] order ([0, 0] the other way round). Gain reads
        // z1 and forget reads z0, so swapping either the halves or the two
        // gates changes the result.
        let x: [Float] = [1, 0]
        let m: [Float] = [0, 3]
        var downValues = [Float](repeating: 0, count: 2 * 4)
        downValues[0] = 1          // z0 = g[0] = rms(x)[0]
        downValues[4 + 3] = 2      // z1 = 2 g[3] = 2 rms(m)[1]
        let down = Inkling2DenseMatrix(rows: 2, cols: 4, values: downValues)
        let gain = Inkling2DenseMatrix(rows: 2, cols: 2, values: [0, 1, 0, 1])
        let forget = Inkling2DenseMatrix(rows: 2, cols: 2, values: [1, 0, 1, 0])
        let out = Op.control(x, message: m, down: down, gain: gain, forget: forget)
        let z0 = Float(2).squareRoot(), z1 = 2 * Float(2).squareRoot()
        #expect(close(out[0], (1 - tanh(z0)) * 1, 1e-5))
        #expect(close(out[1], tanh(z1) * 3, 1e-5))
    }

    @Test func attendHeadWithoutSinkIsSoftmax() {
        let out = Op.attendHead(query: [1, 0], keys: [[1, 0], [0, 1]], values: [[10, 0], [0, 10]],
                                biases: [0, 0], sink: -1e30, scale: 1)
        let p = Float(exp(1.0) / (exp(1.0) + 1))
        #expect(close(out[0], 10 * p, 1e-4) && close(out[1], 10 * (1 - p), 1e-4))
    }

    @Test func attendHeadSinkAbsorbsMass() {
        // One key with logit 0 and a sink logit 0: half the mass is dropped.
        let out = Op.attendHead(query: [0, 0], keys: [[1, 1]], values: [[4, 2]],
                                biases: [0], sink: 0, scale: 1)
        #expect(close(out[0], 2) && close(out[1], 1))
        // A bias shifts the key's logit: bias ln(3) leaves 3/4 of the mass.
        let biased = Op.attendHead(query: [0, 0], keys: [[1, 1]], values: [[4, 2]],
                                   biases: [Float(log(3.0))], sink: 0, scale: 1)
        #expect(close(biased[0], 3, 1e-5))
    }

    @Test func relativeBiasIsZeroPastTheExtent() {
        // extent 3 rows of relDim 2.
        let table: [Float] = [1, 2,  3, 4,  5, 6]
        #expect(Op.relativeBias(table: table, relState: [1, 1], distance: 0) == 3)
        #expect(Op.relativeBias(table: table, relState: [1, 1], distance: 2) == 11)
        #expect(Op.relativeBias(table: table, relState: [1, 1], distance: 3) == 0)
        #expect(Op.relativeBias(table: table, relState: [1, 1], distance: -1) == 0)
    }

    @Test func routerBiasSelectsButDoesNotWeight() {
        // Scores sigmoid(logit): expert 0 highest, expert 2 lowest; a bias lifts expert 2.
        let router = Inkling2DenseMatrix(rows: 3, cols: 1, values: [2, 1, -1])
        let plain = Op.route([1], router: router, bias: [0, 0, 0], topK: 2, routeScale: 8)
        #expect(plain.experts == [0, 1])
        let biased = Op.route([1], router: router, bias: [0, 0, 5], topK: 2, routeScale: 8)
        #expect(biased.experts == [2, 0])
        let s0 = Op.sigmoid(2), s2 = Op.sigmoid(-1)
        #expect(close(biased.weights[0], s2 / (s0 + s2) * 8, 1e-5))
        #expect(close(biased.weights.reduce(0, +), 8, 1e-5))
    }

    @Test func routerTiesGoToTheLowerIndex() {
        let router = Inkling2DenseMatrix(rows: 4, cols: 1, values: [1, 1, 1, 1])
        #expect(Op.route([1], router: router, bias: [0, 0, 0, 0], topK: 2, routeScale: 1).experts == [0, 1])
    }

    @Test func expertActivationUsesTheRawGateInTheSigmoid() {
        // ff = 1: gateUp rows (gate, up) = (3, 5) for x = [1]; down = [2].
        let gateUp = Inkling2DenseMatrix(rows: 2, cols: 1, values: [3, 5])
        let down = Inkling2DenseMatrix(rows: 1, cols: 1, values: [2])
        let y = Op.expert([1], gateUp: gateUp, down: down, gateSoftcap: 2, upSoftcap: 4)
        let act = Float(2 * tanh(1.5)) * Op.sigmoid(3) * Float(4 * tanh(1.25))
        #expect(close(y[0], 2 * act, 1e-5))
    }

    @Test func moeAccumulatesWeightedExperts() {
        let gateUp: [any Inkling2ReferenceMatrix] = [
            Inkling2DenseMatrix(rows: 2, cols: 1, values: [1, 1]),
            Inkling2DenseMatrix(rows: 2, cols: 1, values: [2, 1]),
        ]
        let down: [any Inkling2ReferenceMatrix] = [
            Inkling2DenseMatrix(rows: 1, cols: 1, values: [1]),
            Inkling2DenseMatrix(rows: 1, cols: 1, values: [1]),
        ]
        let route = Inkling2Route(experts: [1, 0], weights: [3, 5])
        let y = Op.moe([1], route: route, gateUp: gateUp, down: down, gateSoftcap: 0, upSoftcap: 0)
        let e0 = Op.sigmoid(1) * 1, e1 = 2 * Op.sigmoid(2) * 1
        #expect(close(y[0], 5 * e0 + 3 * e1, 1e-5))
    }

    @Test func softcapAndRmsNorm() {
        #expect(close(Op.softcap(30, 30), Float(30 * tanh(1.0))))
        #expect(Op.softcap(5, 0) == 5)
        let n = Op.rmsNorm([3, 4], weight: [1, 2], eps: 0)
        let rms = Float((12.5).squareRoot())
        #expect(close(n[0], 3 / rms) && close(n[1], 8 / rms))
    }
}
