import Foundation
import Metal

extension Native {
    /// The fast `Sampler`: the chain the engine's llama.cpp backend builds
    /// (`MferenceEngine/LlamaSampler.swift`) in llama.cpp `common_sampler`
    /// order: logit bias, penalties, then greedy at temperature <= 0, else
    /// top-k, top-p, min-p, temperature and a seeded `mt19937` draw.
    ///
    /// Token choices equal `ReferenceSampler`'s (and so llama.cpp's) for every
    /// setting: every stage keeps the reference's arithmetic and accumulation
    /// order. What is faster is the work around it:
    /// - one reused scratch row instead of a candidate array per call;
    /// - penalties and biases applied to the listed tokens only;
    /// - greedy as a vectorized first-maximum scan;
    /// - top-k as a bounded heap instead of a full sort;
    /// - top-p without top-k sorts only as many leading candidates as the
    ///   nucleus needs (doubling until the cumulative probability reaches
    ///   `topP`, falling back to the full sort);
    /// - a ring buffer for the penalty window.
    ///
    /// Ordering among equal logits is the reference's (ties to the lower token
    /// id), so the sorted prefix is unique and a partial selection yields the
    /// same candidates as a full sort.
    ///
    /// `encodeChoice` is a two-pass GPU argmax (`native_sampling`): ties go to
    /// the lowest id and NaN never wins, except that a NaN in row 0 chooses 0,
    /// which is what the CPU first-maximum scan does.
    public final class ChainSampler: Sampler {
        public private(set) var params = SamplingParams(temperature: 0, topK: nil, topP: nil, minP: 0,
                                                        repetitionPenalty: 1, presencePenalty: 0,
                                                        frequencyPenalty: 0, repeatLastN: 0, seed: nil)
        private var rng = SamplingMT19937(seed: 0)
        /// Penalty window: a ring of `repeatLastN` tokens when positive, the
        /// whole history (counts only) when negative.
        private var window: [Int32] = []
        private var windowHead = 0
        private var windowCount = 0
        private var counts: [Int32: Int] = [:]
        /// Scratch: the biased, penalized logits of the current row.
        private var work: [Float] = []
        /// Scratch: exp values for the unsorted top-p softmax.
        private var exps: [Float] = []

        private let partialPipeline: MTLComputePipelineState?
        private let finalPipeline: MTLComputePipelineState?
        /// GPU-only scratch for the argmax partials (one queue runs steps in
        /// order, so no ring is needed).
        private let partials: MTLBuffer?

        /// Shader module and kernel names (`Metal/Native/Sampling/native_sampling.metal`).
        public static let module = "native_sampling"
        static let maxPartials = 64
        static let threadsPerGroup = 256
        static let rowsPerGroup = 4096

        /// `context` is needed only for `encodeChoice`.
        public init(context: MetalContext?) throws {
            if let context {
                partialPipeline = try context.modulePipeline(module: Self.module, name: "native_argmax_partial")
                finalPipeline = try context.modulePipeline(module: Self.module, name: "native_argmax_final")
                guard let buffer = context.device.makeBuffer(length: Self.maxPartials * 8, options: .storageModePrivate)
                else { throw NativeError.resource("sampling partials") }
                buffer.label = "native.sampling.partials"
                partials = buffer
            } else {
                partialPipeline = nil
                finalPipeline = nil
                partials = nil
            }
        }

        var penalised: Bool {
            params.repeatLastN != 0
                && (params.repetitionPenalty != 1 || params.presencePenalty != 0 || params.frequencyPenalty != 0)
        }

        /// llama.cpp `LLAMA_DEFAULT_SEED` (and no seed) draws a random seed.
        public static func resolvedSeed(_ seed: UInt64?) -> UInt32 {
            guard let seed, UInt32(truncatingIfNeeded: seed) != 0xFFFF_FFFF else {
                return UInt32.random(in: 0..<UInt32.max)
            }
            return UInt32(truncatingIfNeeded: seed)
        }

        public func reset(params: SamplingParams) {
            self.params = params
            counts.removeAll(keepingCapacity: true)
            windowHead = 0
            windowCount = 0
            if params.repeatLastN > 0 {
                window = [Int32](repeating: 0, count: params.repeatLastN)
            } else {
                window = []
            }
            rng = SamplingMT19937(seed: Self.resolvedSeed(params.seed))
        }

        public func accept(_ token: Int32) {
            let lastN = params.repeatLastN
            guard lastN != 0 else { return }
            counts[token, default: 0] += 1
            guard lastN > 0 else { return }
            if windowCount >= lastN {
                // Drop the oldest (after counting the new one, as the reference).
                let old = window[windowHead]
                if let c = counts[old] {
                    if c <= 1 { counts[old] = nil } else { counts[old] = c - 1 }
                }
                window[windowHead] = token
                windowHead = (windowHead + 1) % lastN
            } else {
                window[(windowHead + windowCount) % lastN] = token
                windowCount += 1
            }
        }

        public var canChooseOnGPU: Bool { params.isGreedy && !penalised && params.logitBias.isEmpty }

        // MARK: CPU chain

        struct Candidate {
            var id: Int32
            var logit: Float
            var p: Float
        }

        /// The reference order: higher logit first, ties to the lower id.
        @inline(__always)
        static func before(_ a: Candidate, _ b: Candidate) -> Bool {
            a.logit != b.logit ? a.logit > b.logit : a.id < b.id
        }

        static func sortDescending(_ c: inout [Candidate]) {
            c.sort(by: before)
        }

        /// The first index holding the maximum (`v > best` scan from index 0):
        /// NaN never wins, except a NaN at 0, which keeps 0.
        static func firstArgmax(_ p: UnsafePointer<Float>, count n: Int) -> Int {
            var best = 0
            var bestValue = p[0]
            var i = 1
            let raw = UnsafeRawPointer(p)
            while i + 16 <= n {
                let v = raw.loadUnaligned(fromByteOffset: i * 4, as: SIMD16<Float>.self)
                if any(v .> SIMD16(repeating: bestValue)) {
                    for j in i ..< i + 16 where p[j] > bestValue {
                        bestValue = p[j]
                        best = j
                    }
                }
                i += 16
            }
            while i < n {
                if p[i] > bestValue {
                    bestValue = p[i]
                    best = i
                }
                i += 1
            }
            return best
        }

        /// The first `k` candidates of the reference order over `p[0..<n]`,
        /// sorted (a bounded min-heap of the best `k`, then a sort).
        static func topCandidates(_ p: UnsafePointer<Float>, count n: Int, k: Int) -> [Candidate] {
            precondition(k >= 1 && k <= n)
            if k * 2 >= n {
                var all = (0..<n).map { Candidate(id: Int32($0), logit: p[$0], p: 0) }
                sortDescending(&all)
                all.removeLast(n - k)
                return all
            }
            // heap[0] is the worst kept candidate (the last in reference order).
            var heap: [Candidate] = []
            heap.reserveCapacity(k)
            @inline(__always) func worse(_ a: Candidate, _ b: Candidate) -> Bool { before(b, a) }
            func siftUp(_ start: Int) {
                var c = start
                while c > 0 {
                    let parent = (c - 1) / 2
                    guard worse(heap[c], heap[parent]) else { break }
                    heap.swapAt(c, parent)
                    c = parent
                }
            }
            func siftDown() {
                var i = 0
                let count = heap.count
                while true {
                    let l = 2 * i + 1, r = l + 1
                    var m = i
                    if l < count && worse(heap[l], heap[m]) { m = l }
                    if r < count && worse(heap[r], heap[m]) { m = r }
                    if m == i { return }
                    heap.swapAt(i, m)
                    i = m
                }
            }
            for i in 0..<k {
                heap.append(Candidate(id: Int32(i), logit: p[i], p: 0))
                siftUp(i)
            }
            // A later index beats the worst kept one only with a strictly
            // greater logit (its id is larger than every kept id).
            var threshold = heap[0].logit
            var i = k
            let raw = UnsafeRawPointer(p)
            func offer(_ j: Int) {
                let v = p[j]
                guard v > threshold else { return }
                heap[0] = Candidate(id: Int32(j), logit: v, p: 0)
                siftDown()
                threshold = heap[0].logit
            }
            while i + 16 <= n {
                let v = raw.loadUnaligned(fromByteOffset: i * 4, as: SIMD16<Float>.self)
                if any(v .> SIMD16(repeating: threshold)) {
                    for j in i ..< i + 16 { offer(j) }
                }
                i += 16
            }
            while i < n {
                offer(i)
                i += 1
            }
            sortDescending(&heap)
            return heap
        }

        public func sample(logits: UnsafeBufferPointer<Float>) -> Int32 {
            let n = logits.count
            precondition(n > 0, "empty logits row")
            if work.count < n { work = [Float](repeating: 0, count: n) }
            return work.withUnsafeMutableBufferPointer { w in
                let x = w.baseAddress!
                x.update(from: logits.baseAddress!, count: n)
                for (id, bias) in params.logitBias where id >= 0 && Int(id) < n {
                    x[Int(id)] += bias
                }
                if penalised {
                    let repetition = params.repetitionPenalty
                    let frequency = params.frequencyPenalty
                    let presence = params.presencePenalty
                    for (id, count) in counts where id >= 0 && Int(id) < n {
                        var l = x[Int(id)]
                        if l <= 0 {
                            l *= repetition
                        } else {
                            l /= repetition
                        }
                        l -= Float(count) * frequency + Float(count > 0 ? 1 : 0) * presence
                        x[Int(id)] = l
                    }
                }
                if params.temperature <= 0 {
                    return Int32(Self.firstArgmax(x, count: n))
                }
                return sampleTruncated(x, count: n)
            }
        }

        /// Top-k, top-p, min-p, temperature and the seeded draw over the
        /// biased, penalized row `x`.
        private func sampleTruncated(_ x: UnsafeMutablePointer<Float>, count n: Int) -> Int32 {
            // `cur` holds the candidates once they are sorted (or filtered);
            // nil means "every token, in id order, values in x".
            var cur: [Candidate]?
            var sorted = false
            if let k = params.topK, k > 0 {
                cur = Self.topCandidates(x, count: n, k: min(k, n))
                sorted = true
            }
            if let p = params.topP, p < 1 {
                if var c = cur {
                    // Softmax over the sorted candidates, then the cut.
                    let maxL = c[0].logit
                    var cum: Float = 0
                    for i in c.indices {
                        let e = expf(c[i].logit - maxL)
                        c[i].p = e
                        cum += e
                    }
                    for i in c.indices { c[i].p /= cum }
                    c.removeLast(c.count - Self.nucleusLength(c, p))
                    cur = c
                } else {
                    cur = unsortedNucleus(x, count: n, p: p)
                }
                sorted = true
            }
            if params.minP > 0 {
                if var c = cur {
                    // Sorted: keep the prefix within minP of the first.
                    let minLogit = c[0].logit + logf(params.minP)
                    var i = 1
                    while i < c.count {
                        if c[i].logit < minLogit && i >= 1 { break }
                        i += 1
                    }
                    c.removeLast(c.count - i)
                    cur = c
                } else {
                    var maxL = -Float.greatestFiniteMagnitude
                    for i in 0..<n { maxL = max(maxL, x[i]) }
                    let minLogit = maxL + logf(params.minP)
                    var kept: [Candidate] = []
                    for i in 0..<n where x[i] >= minLogit {
                        kept.append(Candidate(id: Int32(i), logit: x[i], p: 0))
                    }
                    if !kept.isEmpty {
                        cur = kept
                    } else {
                        var all = (0..<n).map { Candidate(id: Int32($0), logit: x[$0], p: 0) }
                        Self.sortDescending(&all)
                        sorted = true
                        let minLogit = all[0].logit + logf(params.minP)
                        var i = 1
                        while i < all.count {
                            if all[i].logit < minLogit && i >= 1 { break }
                            i += 1
                        }
                        all.removeLast(all.count - i)
                        cur = all
                    }
                }
            }
            let temperature = params.temperature
            guard var c = cur else {
                // Every token, in id order: temperature, then the draw.
                for i in 0..<n { x[i] /= temperature }
                if n == 1 {
                    _ = rng.nextCanonical()
                    return 0
                }
                var maxL = x[0]
                for i in 1..<n { maxL = max(maxL, x[i]) }
                var sumCum: Double = 0
                for i in 0..<n {
                    let e = expf(x[i] - maxL)
                    x[i] = e
                    sumCum += Double(e)
                }
                let target = sumCum * rng.nextCanonical()
                var running: Double = 0
                for i in 0..<n {
                    running += Double(x[i])
                    if running >= target { return Int32(i) }
                }
                return Int32(n - 1)
            }
            for i in c.indices { c[i].logit /= temperature }
            if c.count == 1 {
                _ = rng.nextCanonical()
                return c[0].id
            }
            var maxL = c[0].logit
            if !sorted { for e in c.dropFirst() { maxL = max(maxL, e.logit) } }
            var sumCum: Double = 0
            for i in c.indices {
                let e = expf(c[i].logit - maxL)
                c[i].p = e
                sumCum += Double(e)
            }
            let target = sumCum * rng.nextCanonical()
            var running: Double = 0
            for e in c {
                running += Double(e.p)
                if running >= target { return e.id }
            }
            return c[c.count - 1].id
        }

        /// Candidates kept by the top-p cut over sorted candidates whose `p`
        /// is set: the shortest prefix whose cumulative probability reaches
        /// `p` (all of them if it never does).
        static func nucleusLength(_ c: [Candidate], _ p: Float) -> Int {
            var running: Float = 0
            for i in c.indices {
                running += c[i].p
                if running >= p && i + 1 >= 1 { return i + 1 }
            }
            return c.count
        }

        /// Top-p with no top-k: the softmax runs over every token in id order
        /// (the reference's accumulation order), then only as many leading
        /// candidates of the reference order are selected and sorted as the
        /// cut needs.
        private func unsortedNucleus(_ x: UnsafeMutablePointer<Float>, count n: Int, p: Float) -> [Candidate] {
            if exps.count < n { exps = [Float](repeating: 0, count: n) }
            return exps.withUnsafeMutableBufferPointer { eb in
                let e = eb.baseAddress!
                var maxL = x[0]
                for i in 1..<n { maxL = max(maxL, x[i]) }
                var cum: Float = 0
                for i in 0..<n {
                    let v = expf(x[i] - maxL)
                    e[i] = v
                    cum += v
                }
                var m = min(n, 1024)
                while true {
                    var c = Self.topCandidates(x, count: n, k: m)
                    for i in c.indices { c[i].p = e[Int(c[i].id)] / cum }
                    var running: Float = 0
                    for i in c.indices {
                        running += c[i].p
                        if running >= p && i + 1 >= 1 {
                            c.removeLast(c.count - (i + 1))
                            return c
                        }
                    }
                    if m == n { return c }
                    m = min(n, m * 8)
                }
            }
        }

        // MARK: GPU choice

        public func encodeChoice(_ encoder: MTLComputeCommandEncoder, logits: TensorRef, output: TokenSlot) throws {
            guard canChooseOnGPU else { throw NativeError.unsupported("GPU choice needs greedy, neutral params") }
            guard let partialPipeline, let finalPipeline, let partials else {
                throw NativeError.resource("ChainSampler built without a Metal context")
            }
            guard logits.dtype == .f32, logits.rows == 1, logits.cols >= 1, output.count >= 1 else {
                throw NativeError.invalidShape("encodeChoice takes one F32 logits row")
            }
            var count = UInt32(logits.cols)
            let groups = min(Self.maxPartials, max(1, (logits.cols + Self.rowsPerGroup - 1) / Self.rowsPerGroup))
            var groupCount = UInt32(groups)
            encoder.setComputePipelineState(partialPipeline)
            encoder.setBuffer(logits.buffer, offset: logits.offset, index: 0)
            encoder.setBuffer(partials, offset: 0, index: 1)
            encoder.setBytes(&count, length: 4, index: 2)
            encoder.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: Self.threadsPerGroup, height: 1, depth: 1))
            encoder.setComputePipelineState(finalPipeline)
            encoder.setBuffer(logits.buffer, offset: logits.offset, index: 0)
            encoder.setBuffer(partials, offset: 0, index: 1)
            encoder.setBytes(&groupCount, length: 4, index: 2)
            encoder.setBuffer(output.buffer, offset: output.offset, index: 3)
            encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: Self.maxPartials, height: 1, depth: 1))
        }
    }

    /// Plain decoding: no drafts (the program's only strategy).
    public final class PlainDecoding: DecodingStrategy {
        public init() {}
        public let name = "plain"
        public let maxDraftRows = 0
        public func reset(prompt: [Int32]) {}
        public func draft(history: [Int32]) -> [Int32] { [] }
        public func observe(proposed: Int, accepted: Int) {}
    }
}
