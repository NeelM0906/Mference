import Foundation
import Metal

extension Native {
    /// `std::mt19937` (32-bit Mersenne Twister) as libc++ implements it.
    public struct ReferenceMT19937 {
        private var state = [UInt32](repeating: 0, count: 624)
        private var index = 624

        public init(seed: UInt32) {
            state[0] = seed
            for i in 1..<624 {
                state[i] = 1_812_433_253 &* (state[i - 1] ^ (state[i - 1] >> 30)) &+ UInt32(i)
            }
            index = 624
        }

        public mutating func next() -> UInt32 {
            if index >= 624 {
                for i in 0..<624 {
                    let y = (state[i] & 0x8000_0000) | (state[(i + 1) % 624] & 0x7FFF_FFFF)
                    var v = state[(i + 397) % 624] ^ (y >> 1)
                    if y & 1 != 0 { v ^= 0x9908_B0DF }
                    state[i] = v
                }
                index = 0
            }
            var y = state[index]
            index += 1
            y ^= y >> 11
            y ^= (y << 7) & 0x9D2C_5680
            y ^= (y << 15) & 0xEFC6_0000
            y ^= y >> 18
            return y
        }

        /// libc++ `std::uniform_real_distribution<double>(0, 1)`:
        /// `generate_canonical<double, 53>` from two draws.
        public mutating func nextCanonical() -> Double {
            let r = 4_294_967_296.0
            var sum = Double(next())
            sum += Double(next()) * r
            return sum / (r * r)
        }
    }

    /// The reference `Sampler`: the chain the engine's llama.cpp backend
    /// builds (`MferenceEngine/LlamaSampler.swift`) in `common_sampler` order:
    /// logit bias, penalties, then greedy at temperature <= 0, else top-k,
    /// top-p, min-p, temperature and a seeded `mt19937` draw. Each stage is a
    /// port of `src/llama-sampler.cpp` at the pin, including its float/double
    /// accumulation order.
    ///
    /// One documented difference: where llama.cpp sorts with `std::sort` /
    /// `std::partial_sort` (unspecified order among equal logits), this sorts
    /// equal logits by token id. Choices differ only when candidates tie
    /// exactly.
    public final class ReferenceSampler: Sampler {
        public private(set) var params = SamplingParams(temperature: 0, topK: nil, topP: nil, minP: 0,
                                                        repetitionPenalty: 1, presencePenalty: 0,
                                                        frequencyPenalty: 0, repeatLastN: 0, seed: nil)
        private var rng = ReferenceMT19937(seed: 0)
        private var history: [Int32] = []
        private var counts: [Int32: Int] = [:]
        private let argmax: MTLComputePipelineState?

        /// `context` is needed only for `encodeChoice`.
        public init(context: MetalContext?) throws {
            argmax = try context?.modulePipeline(module: ReferenceMetal.module, name: "ref_argmax")
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
            history = []
            counts = [:]
            rng = ReferenceMT19937(seed: Self.resolvedSeed(params.seed))
        }

        public func accept(_ token: Int32) {
            guard params.repeatLastN != 0 else { return }
            counts[token, default: 0] += 1
            if params.repeatLastN > 0 && history.count >= params.repeatLastN {
                let old = history.removeFirst()
                counts[old]! -= 1
                if counts[old] == 0 { counts[old] = nil }
            }
            history.append(token)
        }

        public var canChooseOnGPU: Bool { params.isGreedy && !penalised && params.logitBias.isEmpty }

        struct Candidate {
            var id: Int32
            var logit: Float
            var p: Float
        }

        static func sortDescending(_ c: inout [Candidate]) {
            c.sort { $0.logit != $1.logit ? $0.logit > $1.logit : $0.id < $1.id }
        }

        public func sample(logits: UnsafeBufferPointer<Float>) -> Int32 {
            var cur = (0..<logits.count).map { Candidate(id: Int32($0), logit: logits[$0], p: 0) }
            var sorted = false
            for (id, bias) in params.logitBias where id >= 0 && Int(id) < cur.count {
                cur[Int(id)].logit += bias
            }
            if penalised {
                for i in cur.indices {
                    guard let count = counts[cur[i].id] else { continue }
                    if cur[i].logit <= 0 {
                        cur[i].logit *= params.repetitionPenalty
                    } else {
                        cur[i].logit /= params.repetitionPenalty
                    }
                    cur[i].logit -= Float(count) * params.frequencyPenalty + Float(count > 0 ? 1 : 0) * params.presencePenalty
                }
            }
            if params.temperature <= 0 {
                var best = 0
                for i in 1..<cur.count where cur[i].logit > cur[best].logit { best = i }
                return cur[best].id
            }
            if let k = params.topK, k > 0 {
                let kk = min(k, cur.count)
                Self.sortDescending(&cur)
                cur.removeLast(cur.count - kk)
                sorted = true
            }
            if let p = params.topP, p < 1 {
                // Softmax over the candidates in their current order.
                var maxL = cur[0].logit
                if !sorted { for c in cur.dropFirst() { maxL = max(maxL, c.logit) } }
                var cum: Float = 0
                for i in cur.indices {
                    let e = expf(cur[i].logit - maxL)
                    cur[i].p = e
                    cum += e
                }
                for i in cur.indices { cur[i].p /= cum }
                if !sorted { Self.sortDescending(&cur) }
                var running: Float = 0
                var last = cur.count
                for i in cur.indices {
                    running += cur[i].p
                    if running >= p && i + 1 >= 1 {
                        last = i + 1
                        break
                    }
                }
                cur.removeLast(cur.count - last)
                sorted = true
            }
            if params.minP > 0 && !cur.isEmpty {
                var applied = false
                if !sorted {
                    var maxL = -Float.greatestFiniteMagnitude
                    for c in cur { maxL = max(maxL, c.logit) }
                    let minLogit = maxL + logf(params.minP)
                    let kept = cur.filter { $0.logit >= minLogit }
                    if !kept.isEmpty {
                        cur = kept
                        applied = true
                    }
                }
                if !applied {
                    if !sorted {
                        Self.sortDescending(&cur)
                        sorted = true
                    }
                    let minLogit = cur[0].logit + logf(params.minP)
                    var i = 1
                    while i < cur.count {
                        if cur[i].logit < minLogit && i >= 1 { break }
                        i += 1
                    }
                    cur.removeLast(cur.count - i)
                }
            }
            for i in cur.indices { cur[i].logit /= params.temperature }
            // dist
            if cur.count == 1 {
                _ = rng.nextCanonical()
                return cur[0].id
            }
            var maxL = cur[0].logit
            if !sorted { for c in cur.dropFirst() { maxL = max(maxL, c.logit) } }
            var sumCum: Double = 0
            for i in cur.indices {
                let e = expf(cur[i].logit - maxL)
                cur[i].p = e
                sumCum += Double(e)
            }
            let target = sumCum * rng.nextCanonical()
            var running: Double = 0
            for c in cur {
                running += Double(c.p)
                if running >= target { return c.id }
            }
            return cur[cur.count - 1].id
        }

        public func encodeChoice(_ encoder: MTLComputeCommandEncoder, logits: TensorRef, output: TokenSlot) throws {
            guard canChooseOnGPU else { throw NativeError.unsupported("GPU choice needs greedy, neutral params") }
            guard let argmax else { throw NativeError.resource("ReferenceSampler built without a Metal context") }
            guard logits.dtype == .f32, logits.rows == 1, output.count >= 1 else {
                throw NativeError.invalidShape("encodeChoice takes one F32 logits row")
            }
            encoder.setBuffer(logits.buffer, offset: logits.offset, index: 0)
            encoder.setBuffer(output.buffer, offset: output.offset, index: 1)
            ReferenceMetal.setBytes(encoder, UInt32(logits.cols), index: 2)
            encoder.setComputePipelineState(argmax)
            encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        }
    }

    /// Plain decoding: no drafts.
    public final class ReferencePlainDecoding: DecodingStrategy {
        public init() {}
        public let name = "plain"
        public let maxDraftRows = 0
        public func reset(prompt: [Int32]) {}
        public func draft(history: [Int32]) -> [Int32] { [] }
        public func observe(proposed: Int, accepted: Int) {}
    }
}
