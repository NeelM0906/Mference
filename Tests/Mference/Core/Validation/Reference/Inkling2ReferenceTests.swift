import Testing
import Foundation
import MferenceValidationSupport

/// Checks of the Inkling 2 attention-side CPU references against independent
/// formulations: the llama.cpp folded-sink logits, the generic attention
/// reference, the vendor's lag-ordered dynamic convolution, and algebraic
/// properties (window and extent masking, chunk invariance, tau).
@Suite struct Inkling2ReferenceTests {
    static func random(_ count: Int, _ rng: inout SplitMix64, _ lo: Float = -1, _ hi: Float = 1) -> [Float] {
        (0..<count).map { _ in rng.uniform(lo, hi) }
    }

    struct AttentionCase {
        var hq = 6, hkv = 2, d = 16, dRel = 4
        var extent = 12
        var t = 5, start = 9
        var q: [Float] = [], k: [Float] = [], v: [Float] = [], rel: [Float] = [], table: [Float] = [], sink: [Float] = []

        init(seed: UInt64, t: Int = 5, start: Int = 9, extent: Int = 12) {
            self.t = t
            self.start = start
            self.extent = extent
            var rng = SeedTree(seed).key("ink2-attn")
            q = Inkling2ReferenceTests.random(t * hq * d, &rng)
            k = Inkling2ReferenceTests.random((start + t) * hkv * d, &rng)
            v = Inkling2ReferenceTests.random((start + t) * hkv * d, &rng)
            rel = Inkling2ReferenceTests.random(t * hq * dRel, &rng)
            table = Inkling2ReferenceTests.random(extent * dRel, &rng, -2, 2)
            sink = Inkling2ReferenceTests.random(t * hq, &rng, -1, 2)
        }

        func run(window: Int?, logScale: Inkling2AttentionRef.LogScale? = nil) -> [Float] {
            Inkling2AttentionRef.attend(q: q, k: k, v: v, rel: rel, table: table, sink: sink,
                                        queryCount: t, startPosition: start,
                                        numQHeads: hq, numKVHeads: hkv, headDim: d, dRel: dRel,
                                        relExtent: extent, window: window, logScale: logScale)
        }
    }

    /// llama.cpp `inkling2`: `table = P_pad^T r - sink`, gathered by distance
    /// (index `extent` is the zero pad row), added to the mask, softmax with a
    /// zero sink logit. FP32 throughout.
    static func foldedSink(_ c: AttentionCase, window: Int?) -> [Float] {
        let group = c.hq / c.hkv
        let scale = 1 / Float(c.d).squareRoot()
        var out = [Float](repeating: 0, count: c.q.count)
        for i in 0..<c.t {
            let pos = c.start + i
            for h in 0..<c.hq {
                var table = [Float](repeating: 0, count: c.extent + 1)
                for e in 0..<c.extent {
                    var acc: Float = 0
                    for r in 0..<c.dRel { acc += c.rel[(i * c.hq + h) * c.dRel + r] * c.table[e * c.dRel + r] }
                    table[e] = acc
                }
                let s = c.sink[i * c.hq + h]
                for e in 0...c.extent { table[e] -= s }
                var logits = [Float](repeating: -.infinity, count: pos + 1)
                for j in 0...pos {
                    let distance = pos - j
                    if let window, distance >= window { continue }
                    var dot: Float = 0
                    for x in 0..<c.d { dot += c.q[(i * c.hq + h) * c.d + x] * c.k[(j * c.hkv + h / group) * c.d + x] }
                    logits[j] = dot * scale + table[min(distance, c.extent)]
                }
                let m = max(logits.max()!, 0)
                var denominator = exp(0 - m)
                var acc = [Float](repeating: 0, count: c.d)
                for j in 0...pos where logits[j] > -.infinity {
                    let w = exp(logits[j] - m)
                    denominator += w
                    for x in 0..<c.d { acc[x] += w * c.v[(j * c.hkv + h / group) * c.d + x] }
                }
                for x in 0..<c.d { out[(i * c.hq + h) * c.d + x] = acc[x] / denominator }
            }
        }
        return out
    }

    @Test("attention matches llama.cpp's folded-sink formulation", arguments: [
        (Int?(nil), UInt64(0x2a01)), (Int?(4), UInt64(0x2a02)), (Int?(12), UInt64(0x2a03)), (Int?(30), UInt64(0x2a04)),
    ])
    func attentionMatchesFoldedSink(window: Int?, seed: UInt64) {
        let c = AttentionCase(seed: seed)
        let reference = c.run(window: window)
        let folded = Self.foldedSink(c, window: window)
        #expect(RelError.computeStrict(actual: folded, reference: reference) < 1e-5)
    }

    @Test func attentionWithoutBiasOrSinkMatchesGenericReference() {
        var c = AttentionCase(seed: 0x2a10, t: 1, start: 20)
        c.rel = [Float](repeating: 0, count: c.rel.count)
        c.sink = [Float](repeating: -1e30, count: c.sink.count)
        for window in [Int?(nil), 7] {
            let ours = c.run(window: window)
            let generic = AttentionRef.apply(q: c.q, k: c.k, v: c.v, headDim: c.d,
                                             numQHeads: c.hq, numKVHeads: c.hkv,
                                             seqLen: c.start + 1, window: window)
            #expect(RelError.computeStrict(actual: ours, reference: generic) < 1e-5)
        }
    }

    @Test func slidingWindowIgnoresKeysOutsideTheWindow() {
        var c = AttentionCase(seed: 0x2a20, t: 3, start: 10)
        let window = 4
        let before = c.run(window: window)
        // Key positions 0 ... start - window are outside every query's window.
        let stride = c.hkv * c.d
        for i in 0..<((c.start - window + 1) * stride) { c.k[i] = 50; c.v[i] = -50 }
        #expect(c.run(window: window) == before)
        // A key inside the window does matter.
        c.v[(c.start + 2) * stride] += 1
        #expect(c.run(window: window) != before)
    }

    @Test func globalExtentGivesFarKeysZeroBias() {
        // With extent 3 every key three or more positions back gets bias 0:
        // equal to a bias table whose rows past the extent are zero.
        var c = AttentionCase(seed: 0x2a30, t: 2, start: 8, extent: 3)
        let short = c.run(window: nil)
        var wide = c
        wide.extent = 40
        wide.table = c.table + [Float](repeating: 0, count: (40 - 3) * c.dRel)
        #expect(RelError.computeStrict(actual: wide.run(window: nil), reference: short) < 1e-6)
        c.table[0] += 1
        #expect(c.run(window: nil) != short)
    }

    @Test func rowsAreChunkInvariantAndKeyBaseOnlyShiftsStorage() {
        let c = AttentionCase(seed: 0x2a40, t: 6, start: 7)
        let window = 5
        let whole = c.run(window: window)
        // Second half as its own chunk, passing only the keys it can see.
        let split = 3
        let firstKey = c.start + split - window + 1
        let stride = c.hkv * c.d
        let tail = Inkling2AttentionRef.attend(
            q: Array(c.q[(split * c.hq * c.d)...]),
            k: Array(c.k[(firstKey * stride)...]), v: Array(c.v[(firstKey * stride)...]),
            rel: Array(c.rel[(split * c.hq * c.dRel)...]), table: c.table,
            sink: Array(c.sink[(split * c.hq)...]),
            queryCount: c.t - split, startPosition: c.start + split, keyBase: firstKey,
            numQHeads: c.hq, numKVHeads: c.hkv, headDim: c.d, dRel: c.dRel,
            relExtent: c.extent, window: window, logScale: nil)
        #expect(Array(whole[(split * c.hq * c.d)...]) == tail)
    }

    @Test func logScaleEqualsScalingQueryAndRelativeStates() {
        var c = AttentionCase(seed: 0x2a50, t: 4, start: 11)
        let logScale = Inkling2AttentionRef.LogScale(floor: 4, alpha: 0.3)
        let scaled = c.run(window: nil, logScale: logScale)
        for i in 0..<c.t {
            let tau = Float(logScale.tau(position: c.start + i))
            for j in 0..<(c.hq * c.d) { c.q[i * c.hq * c.d + j] *= tau }
            for j in 0..<(c.hq * c.dRel) { c.rel[i * c.hq * c.dRel + j] *= tau }
        }
        #expect(RelError.computeStrict(actual: c.run(window: nil), reference: scaled) < 1e-5)
        #expect(logScale.tau(position: 2) == 1)
        #expect(abs(Inkling2AttentionRef.LogScale().tau(position: 131_071) - (1 + 0.1 * log(16.0))) < 1e-6)
    }

    @Test func sinkOnlyAbsorbsMass() {
        // One visible key: out = v * e^l / (e^l + e^s).
        var c = AttentionCase(seed: 0x2a60, t: 1, start: 0)
        c.sink = [Float](repeating: 0.25, count: c.sink.count)
        let out = c.run(window: nil)
        let group = c.hq / c.hkv
        for h in 0..<c.hq {
            var dot: Float = 0
            for x in 0..<c.d { dot += c.q[h * c.d + x] * c.k[(h / group) * c.d + x] }
            var bias: Float = 0
            for r in 0..<c.dRel { bias += c.rel[h * c.dRel + r] * c.table[r] }
            let logit = dot / Float(c.d).squareRoot() + bias
            let p = 1 / (1 + exp(0.25 - logit))
            for x in 0..<c.d {
                #expect(abs(out[h * c.d + x] - p * c.v[(h / group) * c.d + x]) < 1e-5)
            }
        }
    }

    @Test func l2ClipShortensOnlyLongVectors() {
        var rng = SeedTree(0x2b01).key("clip")
        let heads = 3, width = 8
        var x = Self.random(2 * heads * width, &rng)
        for i in 0..<width { x[heads * width + i] *= 40 }   // row 1, head 0: long
        let scale = Self.random(heads * width, &rng, 0.5, 1.5)
        let out = Inkling2NormRef.l2Clip(x, rows: 2, heads: heads, width: width, scale: scale, threshold: 5)
        for row in 0..<2 {
            for h in 0..<heads {
                let base = (row * heads + h) * width
                let scaled = (0..<width).map { x[base + $0] * scale[h * width + $0] }
                let norm = scaled.reduce(0) { $0 + $1 * $1 }.squareRoot()
                let outNorm = (0..<width).map { out[base + $0] }.reduce(0) { $0 + $1 * $1 }.squareRoot()
                if norm <= 5 {
                    for i in 0..<width { #expect(abs(out[base + i] - scaled[i]) < 1e-6) }
                } else {
                    #expect(abs(outNorm - 5 * norm / (norm + 1e-6)) < 1e-4)
                }
            }
        }
        let rms = Inkling2NormRef.unweightedRMS(x, rows: 2, heads: heads, width: width)
        for v in 0..<(2 * heads) {
            let ms = (0..<width).map { rms[v * width + $0] }.reduce(0) { $0 + $1 * $1 } / Float(width)
            #expect(abs(ms - 1) < 1e-3)
        }
    }

    @Test func staticConvImpulseResponseAndChunking() {
        var rng = SeedTree(0x2c01).key("conv")
        let channels = 5, rows = 9, kernel = 4
        let weight = Self.random(channels * kernel, &rng)
        var impulse = [Float](repeating: 0, count: rows * channels)
        impulse[2] = 1   // row 0, channel 2
        let response = Inkling2ShortConvRef.staticConv(
            impulse, weight: weight, history: Inkling2ShortConvRef.zeroHistory(channels: channels),
            rows: rows, channels: channels)
        let rootFive = Float(5).squareRoot()
        for t in 0..<rows {
            var expected: Float = 0
            if t < kernel {
                let tap: Float = weight[2 * kernel + kernel - 1 - t]
                let identity: Float = t == 0 ? 1 : 0
                expected = (tap + identity) / rootFive
            }
            #expect(abs(response.output[t * channels + 2] - expected) < 1e-6)
        }

        let x = Self.random(rows * channels, &rng)
        let history = Self.random((kernel - 1) * channels, &rng)
        let whole = Inkling2ShortConvRef.staticConv(x, weight: weight, history: history, rows: rows, channels: channels)
        let a = Inkling2ShortConvRef.staticConv(Array(x[..<(4 * channels)]), weight: weight, history: history,
                                                rows: 4, channels: channels)
        let b = Inkling2ShortConvRef.staticConv(Array(x[(4 * channels)...]), weight: weight, history: a.history,
                                                rows: rows - 4, channels: channels)
        #expect(a.output + b.output == whole.output)
        #expect(b.history == whole.history)
        #expect(whole.history == Array(x[((rows - 3) * channels)...]))
    }

    @Test func dynamicConvMatchesVendorLagOrderedFormulation() {
        var rng = SeedTree(0x2c10).key("dynconv")
        let heads = 2, headWidth = 3, channels = heads * headWidth, rows = 6, kernel = 4
        let ggmlWeight = Self.random(channels * kernel, &rng)
        let dynamic = Self.random(rows * heads * kernel, &rng)
        let history = Self.random((kernel - 1) * channels, &rng)
        let x = Self.random(rows * channels, &rng)
        let ours = Inkling2ShortConvRef.dynamicConv(x, weight: ggmlWeight, dynamic: dynamic, heads: heads,
                                                    history: history, rows: rows, channels: channels)
        // Vendor: static_weight[lag][c] (lag 0 = current token) is the ggml
        // weight flipped back; taps = static + dynamic; output / sqrt(2K+1).
        func past(_ t: Int, _ lag: Int, _ c: Int) -> Float {
            let n = t - lag
            return n >= 0 ? x[n * channels + c] : history[(kernel - 1 + n) * channels + c]
        }
        for t in 0..<rows {
            for c in 0..<channels {
                var acc: Float = 0
                for lag in 0..<kernel {
                    let tap = ggmlWeight[c * kernel + (kernel - 1 - lag)] + dynamic[(t * heads + c / headWidth) * kernel + lag]
                    acc += tap * past(t, lag, c)
                }
                let vendor = (acc + x[t * channels + c]) / 3
                #expect(abs(ours.output[t * channels + c] - vendor) < 1e-5)
            }
        }
    }

    @Test func q8RoundTripAndStackedProjection() {
        var rng = SeedTree(0x2d01).key("q8")
        let inputs = 64, tokens = 3
        let rowsA = 5, rowsB = 2
        let wA = Self.random(rowsA * inputs, &rng, -0.3, 0.3)
        let wB = Self.random(rowsB * inputs, &rng, -2, 2)
        let qA = Inkling2Q8Ref.quantize(rows: rowsA, inputs: inputs, values: wA)
        let qB = Inkling2Q8Ref.quantize(rows: rowsB, inputs: inputs, values: wB)
        #expect(qA.count == rowsA * 2 * 34)
        let decoded = Inkling2Q8Ref.dequantize(qA, rows: rowsA, inputs: inputs)
        for block in 0..<(rowsA * inputs / 32) {
            let slice = wA[(block * 32)..<(block * 32 + 32)]
            let step = slice.map(abs).max()! / 127
            // Rounding to the step (d/2) plus the half rounding of d itself (127 d 2^-11).
            for i in slice.indices { #expect(abs(decoded[i] - wA[i]) <= step * 0.57 + 1e-6) }
        }
        // A hand-built block: d = 0.5 (0x3800), q = 0, 1, -2, 3, ...
        var block: [UInt8] = [0x00, 0x38]
        for j in 0..<32 { block.append(UInt8(bitPattern: Int8(j % 2 == 0 ? j : -j))) }
        let values = Inkling2Q8Ref.dequantize(block, rows: 1, inputs: 32)
        for j in 0..<32 { #expect(values[j] == 0.5 * Float(j % 2 == 0 ? j : -j)) }

        let x1 = Self.random(tokens * inputs, &rng)
        let x2 = Self.random(tokens * inputs, &rng)
        let stacked = Inkling2Q8Ref.stackedProjection(
            [.init(input: x1, weight: qA, rows: rowsA), .init(input: x2, weight: qB, rows: rowsB)],
            tokens: tokens, inputs: inputs)
        let decodedB = Inkling2Q8Ref.dequantize(qB, rows: rowsB, inputs: inputs)
        for t in 0..<tokens {
            for r in 0..<(rowsA + rowsB) {
                let (w, x, row) = r < rowsA ? (decoded, x1, r) : (decodedB, x2, r - rowsA)
                var dot: Float = 0
                for i in 0..<inputs { dot += w[row * inputs + i] * x[t * inputs + i] }
                #expect(abs(stacked[t * (rowsA + rowsB) + r] - dot) < 1e-5)
            }
        }
    }

    @Test func depthMixAndMessageGate() {
        var rng = SeedTree(0x2e01).key("loop")
        let rows = 3, hidden = 8, rank = 4
        let a = Self.random(rows * hidden, &rng, -3, 3)
        let b = Self.random(rows * hidden, &rng, -3, 3)
        let normA = Inkling2NormRef.unweightedRMS(a, rows: rows, heads: 1, width: hidden)
        let normB = Inkling2NormRef.unweightedRMS(b, rows: rows, heads: 1, width: hidden)
        let queries = Self.random(2 * hidden, &rng, -2, 2)

        let single = Inkling2LoopRef.depthMix(states: [a], queries: queries, queryCount: 2, rows: rows, hidden: hidden)
        for t in 0..<rows {
            for q in 0..<2 {
                for d in 0..<hidden {
                    #expect(abs(single[(t * 2 + q) * hidden + d] - normA[t * hidden + d]) < 1e-6)
                }
            }
        }
        let zero = Inkling2LoopRef.depthMix(states: [a, b], queries: [Float](repeating: 0, count: hidden),
                                            queryCount: 1, rows: rows, hidden: hidden)
        for i in 0..<(rows * hidden) { #expect(abs(zero[i] - (normA[i] + normB[i]) / 2) < 1e-6) }
        #expect(Inkling2LoopRef.depthMix(states: [a, b], queries: queries, queryCount: 2, rows: rows, hidden: hidden)
                == Inkling2LoopRef.depthMix(states: [b, a], queries: queries, queryCount: 2, rows: rows, hidden: hidden))

        let down = Self.random(rank * 2 * hidden, &rng)
        let forget = Self.random(hidden * rank, &rng)
        let gain = Self.random(hidden * rank, &rng)
        let zeros = [Float](repeating: 0, count: hidden * rank)
        #expect(Inkling2LoopRef.messageGate(x: a, message: b, down: down, forget: zeros, gain: zeros,
                                            rows: rows, hidden: hidden, rank: rank) == a)
        let gated = Inkling2LoopRef.messageGate(x: a, message: b, down: down, forget: forget, gain: gain,
                                                rows: rows, hidden: hidden, rank: rank)
        for t in 0..<rows {
            var joined = [Float]()
            joined += (0..<hidden).map { normA[t * hidden + $0] }
            joined += (0..<hidden).map { normB[t * hidden + $0] }
            let z = (0..<rank).map { r in (0..<(2 * hidden)).reduce(Float(0)) { $0 + down[r * 2 * hidden + $1] * joined[$1] } }
            for d in 0..<hidden {
                let f = tanh((0..<rank).reduce(Float(0)) { $0 + forget[d * rank + $1] * z[$1] })
                let g = tanh((0..<rank).reduce(Float(0)) { $0 + gain[d * rank + $1] * z[$1] })
                let expected = a[t * hidden + d] - f * a[t * hidden + d] + g * b[t * hidden + d]
                #expect(abs(gated[t * hidden + d] - expected) < 1e-5)
            }
        }
    }
}
