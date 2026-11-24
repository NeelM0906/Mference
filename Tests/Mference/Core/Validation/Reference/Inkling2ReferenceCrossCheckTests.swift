import Testing
import Foundation
import MferenceValidationSupport

/// K3's batched operation references (`Inkling2AttentionRef`, `Inkling2NormRef`,
/// `Inkling2ShortConvRef`, `Inkling2LoopRef`) against K4's per-token
/// `Inkling2ReferenceOps` on shared random inputs: two independent
/// transcriptions of upstream must agree.
@Suite struct Inkling2ReferenceCrossCheckTests {
    typealias Ops = Inkling2ReferenceOps

    static func random(_ count: Int, _ rng: inout SplitMix64, _ lo: Float = -1, _ hi: Float = 1) -> [Float] {
        (0..<count).map { _ in rng.uniform(lo, hi) }
    }

    static func close(_ a: [Float], _ b: [Float], _ tolerance: Float = 1e-5) -> Bool {
        RelError.computeStrict(actual: a, reference: b) < tolerance
    }

    @Test func clipsNormsAndLogScaleAgree() {
        var rng = SeedTree(0x5a01).key("cross-norm")
        let rows = 3, heads = 4, width = 16
        var x = Self.random(rows * heads * width, &rng, -3, 3)
        for i in 0..<width { x[width + i] *= 5 }
        let scale = Self.random(heads * width, &rng, 0.5, 2)
        let ours = Inkling2NormRef.l2Clip(x, rows: rows, heads: heads, width: width, scale: scale, threshold: 5)
        let rms = Inkling2NormRef.unweightedRMS(x, rows: rows, heads: heads, width: width)
        for v in 0..<(rows * heads) {
            let vector = Array(x[(v * width)..<((v + 1) * width)])
            let headScale = Array(scale[((v % heads) * width)..<((v % heads + 1) * width)])
            #expect(Self.close(Array(ours[(v * width)..<((v + 1) * width)]),
                               Ops.l2Clip(vector, scale: headScale, threshold: 5)))
            #expect(Self.close(Array(rms[(v * width)..<((v + 1) * width)]), Ops.rmsNorm(vector, eps: 1e-6)))
        }
        let weight = Self.random(heads * width, &rng, 0.2, 2)
        let weighted = Inkling2NormRef.weightedRMS(x, rows: rows, width: heads * width, weight: weight)
        for t in 0..<rows {
            let row = Array(x[(t * heads * width)..<((t + 1) * heads * width)])
            #expect(Self.close(Array(weighted[(t * heads * width)..<((t + 1) * heads * width)]),
                               Ops.rmsNorm(row, weight: weight, eps: 1e-6)))
        }
        for position in [0, 7, 8191, 8192, 40_000, 1_048_575] {
            let ours = Inkling2AttentionRef.LogScale().tau(position: position)
            #expect(abs(Float(ours) - Ops.logScale(position: position, floor: 8192, alpha: 0.1)) < 1e-6)
        }
    }

    @Test(arguments: [false, true])
    func shortConvolutionsAgree(dynamic: Bool) {
        var rng = SeedTree(0x5a02).key("cross-conv-\(dynamic)")
        let heads = 2, channels = 6, rows = 7, kernel = 4
        let weight = Self.random(channels * kernel, &rng)
        let taps = Self.random(rows * heads * kernel, &rng)
        let history = Self.random((kernel - 1) * channels, &rng)
        let x = Self.random(rows * channels, &rng, -2, 2)
        let ours = dynamic
            ? Inkling2ShortConvRef.dynamicConv(x, weight: weight, dynamic: taps, heads: heads, history: history,
                                               rows: rows, channels: channels)
            : Inkling2ShortConvRef.staticConv(x, weight: weight, history: history, rows: rows, channels: channels)
        var tail = (0..<(kernel - 1)).map { Array(history[($0 * channels)..<(($0 + 1) * channels)]) }
        for t in 0..<rows {
            let row = Array(x[(t * channels)..<((t + 1) * channels)])
            let step = Ops.shortConv(row, tail: tail, weights: weight,
                                     dynamicTaps: dynamic ? Array(taps[(t * heads * kernel)..<((t + 1) * heads * kernel)]) : nil,
                                     heads: heads)
            #expect(Self.close(Array(ours.output[(t * channels)..<((t + 1) * channels)]), step))
            tail = Ops.advance(tail: tail, with: row)
        }
        #expect(ours.history == tail.flatMap { $0 })
    }

    @Test(arguments: [Int?(5), nil])
    func attentionAgrees(window: Int?) {
        var rng = SeedTree(0x5a03).key("cross-attn-\(window ?? 0)")
        let hq = 6, hkv = 2, d = 8, r = 4, extent = 6, rows = 4, start = 9
        let q = Self.random(rows * hq * d, &rng)
        let k = Self.random((start + rows) * hkv * d, &rng)
        let v = Self.random((start + rows) * hkv * d, &rng)
        let rel = Self.random(rows * hq * r, &rng)
        let table = Self.random(extent * r, &rng, -2, 2)
        let sink = Self.random(rows * hq, &rng)
        let logScale = window == nil ? Inkling2AttentionRef.LogScale(floor: 4, alpha: 0.2) : nil
        let ours = Inkling2AttentionRef.attend(q: q, k: k, v: v, rel: rel, table: table, sink: sink,
                                               queryCount: rows, startPosition: start, numQHeads: hq,
                                               numKVHeads: hkv, headDim: d, dRel: r, relExtent: extent,
                                               window: window, logScale: logScale)
        for i in 0..<rows {
            let position = start + i
            let tau = window == nil ? Ops.logScale(position: position, floor: 4, alpha: 0.2) : 1
            let first = window.map { max(0, position - $0 + 1) } ?? 0
            for h in 0..<hq {
                let g = h / (hq / hkv)
                let query = (0..<d).map { q[(i * hq + h) * d + $0] * tau }
                let relState = (0..<r).map { rel[(i * hq + h) * r + $0] * tau }
                let keys = (first...position).map { p in (0..<d).map { k[(p * hkv + g) * d + $0] } }
                let values = (first...position).map { p in (0..<d).map { v[(p * hkv + g) * d + $0] } }
                let biases = (first...position).map { Ops.relativeBias(table: table, relState: relState, distance: position - $0) }
                let theirs = Ops.attendHead(query: query, keys: keys, values: values, biases: biases,
                                            sink: sink[i * hq + h], scale: 1 / Float(d).squareRoot())
                #expect(Self.close(Array(ours[((i * hq + h) * d)..<((i * hq + h + 1) * d)]), theirs))
            }
        }
    }

    @Test func depthMixAndMessageGateAgree() {
        var rng = SeedTree(0x5a04).key("cross-loop")
        let rows = 3, hidden = 16, rank = 4
        let states = (0..<3).map { _ in Self.random(rows * hidden, &rng, -3, 3) }
        let queries = Self.random(2 * hidden, &rng, -2, 2)
        let mixed = Inkling2LoopRef.depthMix(states: states, queries: queries, queryCount: 2, rows: rows, hidden: hidden)
        for t in 0..<rows {
            let theirs = Ops.mix(states: states.map { Array($0[(t * hidden)..<((t + 1) * hidden)]) },
                                 queries: [Array(queries[..<hidden]), Array(queries[hidden...])])
            for j in 0..<2 {
                #expect(Self.close(Array(mixed[((t * 2 + j) * hidden)..<((t * 2 + j + 1) * hidden)]), theirs[j]))
            }
        }
        let x = states[0], m = states[1]
        let down = Self.random(rank * 2 * hidden, &rng)
        let forget = Self.random(hidden * rank, &rng)
        let gain = Self.random(hidden * rank, &rng)
        let gated = Inkling2LoopRef.messageGate(x: x, message: m, down: down, forget: forget, gain: gain,
                                                rows: rows, hidden: hidden, rank: rank)
        for t in 0..<rows {
            let theirs = Ops.control(Array(x[(t * hidden)..<((t + 1) * hidden)]),
                                     message: Array(m[(t * hidden)..<((t + 1) * hidden)]),
                                     down: Inkling2DenseMatrix(rows: rank, cols: 2 * hidden, values: down),
                                     gain: Inkling2DenseMatrix(rows: hidden, cols: rank, values: gain),
                                     forget: Inkling2DenseMatrix(rows: hidden, cols: rank, values: forget))
            #expect(Self.close(Array(gated[(t * hidden)..<((t + 1) * hidden)]), theirs))
        }
    }
}
