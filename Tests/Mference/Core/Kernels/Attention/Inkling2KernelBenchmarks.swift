import Foundation
import Metal
import Testing
@testable import Mference

/// GPU time of the Inkling 2 attention-side kernels at the model's shapes on
/// synthetic data (checkpoint C0 evidence). Env-gated:
/// `MFERENCE_INKLING2_KBENCH=1`. Prints `[ink2-kbench]` lines; asserts only
/// that the command buffers complete.
@Suite(.serialized) struct Inkling2KernelBenchmarks {
    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["MFERENCE_INKLING2_KBENCH"] == "1"
    }

    static func halfBuffer(_ device: MTLDevice, _ count: Int, salt: Int, scale: Float = 0.5) -> MTLBuffer {
        let buffer = device.makeBuffer(length: max(1, count) * 2, options: .storageModeShared)!
        let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: count)
        for i in 0..<count { pointer[i] = Float16(Float((i &* 31 &+ salt) % 97) / 97 * 2 * scale - scale) }
        return buffer
    }

    static func floatBuffer(_ device: MTLDevice, _ count: Int, salt: Int, scale: Float = 0.5) -> MTLBuffer {
        let buffer = device.makeBuffer(length: max(1, count) * 4, options: .storageModeShared)!
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count { pointer[i] = Float((i &* 37 &+ salt) % 101) / 101 * 2 * scale - scale }
        return buffer
    }

    /// Q8_0 rows with pseudo-random quants and a fixed scale per block.
    static func q8Buffer(_ device: MTLDevice, rows: Int, inputs: Int) -> MTLBuffer {
        let rowBytes = inputs / 32 * 34
        let buffer = device.makeBuffer(length: rows * rowBytes, options: .storageModeShared)!
        let bytes = buffer.contents().bindMemory(to: UInt8.self, capacity: rows * rowBytes)
        var state: UInt64 = 0x9E37_79B9_7F4A_7C15
        let scaleBits = Float16(0.004).bitPattern
        for block in 0..<(rows * inputs / 32) {
            let base = block * 34
            bytes[base] = UInt8(truncatingIfNeeded: scaleBits)
            bytes[base + 1] = UInt8(truncatingIfNeeded: scaleBits >> 8)
            for j in 0..<32 {
                state = state &* 6364136223846793005 &+ 1442695040888963407
                bytes[base + 2 + j] = UInt8(truncatingIfNeeded: state >> 33)
            }
        }
        return buffer
    }

    /// Best and median GPU seconds of `repeats` single-command-buffer runs after one warm-up.
    static func time(_ ctx: MetalContext, repeats: Int = 5,
                     _ encode: (MTLCommandBuffer) throws -> Void) rethrows -> (best: Double, median: Double) {
        var samples: [Double] = []
        for run in 0...repeats {
            let cb = ctx.queue.makeCommandBuffer()!
            try encode(cb)
            cb.commit()
            cb.waitUntilCompleted()
            #expect(cb.status == .completed)
            if run > 0 { samples.append(cb.gpuEndTime - cb.gpuStartTime) }
        }
        samples.sort()
        return (samples[0], samples[samples.count / 2])
    }

    private static func only(_ name: String) -> Bool {
        let filter = ProcessInfo.processInfo.environment["MFERENCE_INKLING2_KBENCH_ONLY"] ?? ""
        return filter.isEmpty || filter.split(separator: ",").contains(Substring(name))
    }

    @Test(.enabled(if: enabled && only("projections"))) func projections() throws {
        let ctx = try MetalContext()
        let projection = try Inkling2Projection(library: Inkling2KernelLibrary(context: ctx))
        let hidden = 2304
        // One weight buffer: Q8_0 rows [q 3072 | rel 384 | gate 3072 | k 1024 | v 1024],
        // then BF16 rows [sink 24 | k taps 32 | v taps 32], as the file stores them.
        let rowBytes = hidden / 32 * 34
        let q8Rows = 3072 + 384 + 3072 + 1024 + 1024
        let bf16Base = (q8Rows * rowBytes + 63) / 64 * 64
        let weights = ctx.device.makeBuffer(length: bf16Base + 88 * hidden * 2, options: .storageModeShared)!
        let packed = Self.q8Buffer(ctx.device, rows: q8Rows, inputs: hidden)
        memcpy(weights.contents(), packed.contents(), q8Rows * rowBytes)
        let bf = (weights.contents() + bf16Base).bindMemory(to: UInt16.self, capacity: 88 * hidden)
        for i in 0..<(88 * hidden) { bf[i] = UInt16(truncatingIfNeeded: (Float((i % 61) - 30) * 0.001).bitPattern >> 16) }
        let wo = Self.q8Buffer(ctx.device, rows: hidden, inputs: 3072)
        let maxT = 2048
        let input = Self.halfBuffer(ctx.device, 3 * maxT * hidden, salt: 3)
        let output = ctx.device.makeBuffer(length: maxT * 8664 * 4, options: .storageModeShared)!
        let attnIn = Self.halfBuffer(ctx.device, maxT * 3072, salt: 5)
        let woOut = ctx.device.makeBuffer(length: maxT * hidden * 4, options: .storageModeShared)!
        func q8(_ row: Int, _ rows: Int, input: Int, column: Int) -> Inkling2ProjectionSegment {
            .init(weightOffset: row * rowBytes, inputOffset: input * maxT * hidden, inputStride: hidden,
                  rows: rows, outColumn: column)
        }
        func bf16(_ row: Int, _ rows: Int, input: Int, column: Int) -> Inkling2ProjectionSegment {
            .init(weightOffset: bf16Base + row * hidden * 2, inputOffset: input * maxT * hidden, inputStride: hidden,
                  rows: rows, outColumn: column, format: .bf16)
        }
        // Window visit, pass 1: one input for every projection.
        let windowSegments = [q8(0, 3072, input: 0, column: 0), q8(3072, 384, input: 0, column: 3072),
                              bf16(0, 24, input: 0, column: 3456), q8(3456, 3072, input: 0, column: 3480),
                              q8(6528, 1024, input: 0, column: 6552), q8(7552, 1024, input: 0, column: 7576)]
        // Global visit (and loop passes 2-3): separate q, k, v inputs; k and v carry their BF16 taps.
        let globalSegments = [q8(0, 3072, input: 0, column: 0), q8(3072, 384, input: 0, column: 3072),
                              bf16(0, 24, input: 0, column: 3456), q8(3456, 3072, input: 0, column: 3480),
                              q8(6528, 1024, input: 1, column: 6552), bf16(24, 32, input: 1, column: 7576),
                              q8(7552, 1024, input: 2, column: 7608), bf16(56, 32, input: 2, column: 8632)]
        var runs: [(Int, Inkling2Projection.Path)] = [(1, .rows), (8, .rows)]
        for tokens in [512, 1024, 2048] {
            runs.append((tokens, .tiled))
            runs.append((tokens, .tiled64))
            if projection.supportsTensorOps { runs.append((tokens, .tensorOps)) }
        }
        for (tokens, path) in runs {
            // Decode-sized products chain 48 visits in one command buffer, as a decode step
            // does; one dispatch per buffer would mostly time the GPU clock ramp.
            let chain = path == .rows ? 48 : 1
            for (label, segments) in [("window-qkv", windowSegments), ("global-qkv", globalSegments)] {
                let rows = segments.reduce(0) { $0 + $1.rows }
                let t = Self.time(ctx) { cb in
                    for _ in 0..<chain {
                        projection.encode(commandBuffer: cb, weights: weights, input: input, output: output,
                                          tokens: tokens, inputs: hidden, outStride: rows, segments: segments, path: path)
                    }
                }
                Self.report("\(label)-\(path.rawValue)", tokens: tokens, rows: rows, inputs: hidden,
                            time: (t.best / Double(chain), t.median / Double(chain)))
            }
            let t = Self.time(ctx) { cb in
                for _ in 0..<chain {
                    projection.encode(commandBuffer: cb, weights: wo, input: attnIn, output: woOut, tokens: tokens,
                                      inputs: 3072, outStride: hidden,
                                      segments: [.init(weightOffset: 0, inputStride: 3072, rows: hidden, outColumn: 0)],
                                      path: path)
                }
            }
            Self.report("output-\(path.rawValue)", tokens: tokens, rows: hidden, inputs: 3072,
                        time: (t.best / Double(chain), t.median / Double(chain)))
        }
    }

    static func report(_ label: String, tokens: Int, rows: Int, inputs: Int, time: (best: Double, median: Double)) {
        let flops = 2.0 * Double(rows) * Double(inputs) * Double(tokens)
        let bytes = Double(rows) * Double(inputs) * 34 / 32
        print(String(format: "[ink2-kbench] proj %@ T=%d rows=%d K=%d best=%.3f ms median=%.3f ms %.2f TFLOP/s %.1f GB/s(weights) %.2f us/token",
                     label, tokens, rows, inputs, time.best * 1e3, time.median * 1e3,
                     flops / time.best / 1e12, bytes / time.best / 1e9, time.best * 1e6 / Double(tokens)))
    }

    @Test(.enabled(if: enabled && only("attention"))) func attention() throws {
        let ctx = try MetalContext()
        let library = try Inkling2KernelLibrary(context: ctx)
        let prefill = try Inkling2PrefillAttention(library: library)
        let decode = try Inkling2DecodeAttention(library: library)
        let maxT = 2048, maxKeys = 32768
        let q = Self.halfBuffer(ctx.device, maxT * 3072, salt: 1, scale: 0.15)
        let kLinear = Self.halfBuffer(ctx.device, maxKeys * 1024, salt: 2, scale: 0.15)
        let vLinear = Self.halfBuffer(ctx.device, maxKeys * 1024, salt: 3)
        let rel = Self.floatBuffer(ctx.device, maxT * 384, salt: 4)
        let table = Self.floatBuffer(ctx.device, 1024 * 16, salt: 5)
        let sink = Self.floatBuffer(ctx.device, maxT * 24, salt: 6)
        let gate = Self.floatBuffer(ctx.device, maxT * 3072, salt: 7)
        let out = ctx.device.makeBuffer(length: maxT * 3072 * 2, options: .storageModeShared)!
        let partials = ctx.device.makeBuffer(length: Inkling2DecodeAttention.partialsLength(keys: maxKeys),
                                             options: .storageModeShared)!
        let ringCapacity = 1024 + maxT
        let ring = Inkling2KVView(keys: .init(kLinear), values: .init(vLinear), ringCapacity: ringCapacity)
        let linear = Inkling2KVView(keys: .init(kLinear), values: .init(vLinear), ringCapacity: 0)

        func args(rows: Int, start: Int, cache: Inkling2KVView, mask: Inkling2AttentionMask) -> Inkling2PrefillAttention.Arguments {
            .init(queries: .init(q), cache: cache, relative: .init(rel), table: .init(table), sink: .init(sink),
                  gate: .init(gate), output: .init(out), rows: rows, startPosition: start, mask: mask)
        }
        // Window visits at a full window (start 4096).
        for rows in [512, 1024, 2048] {
            for tile in Inkling2PrefillAttention.Variant.allCases where prefill.supports(tile) {
                let a = args(rows: rows, start: 4096, cache: ring, mask: .slidingDefault)
                let t = Self.time(ctx) { prefill.encode(commandBuffer: $0, a, variant: tile) }
                Self.reportAttention("window", rows: rows, start: 4096, window: 1024, tile: tile.rawValue, time: t)
            }
        }
        // Global visits whose chunk ends at 1K, 4K and 32K keys.
        for keys in [1024, 4096, 32768] {
            for rows in [512, 1024, 2048] where rows <= keys {
                for tile in Inkling2PrefillAttention.Variant.allCases where prefill.supports(tile) {
                    let a = args(rows: rows, start: keys - rows, cache: linear, mask: .globalDefault)
                    let t = Self.time(ctx) { prefill.encode(commandBuffer: $0, a, variant: tile) }
                    Self.reportAttention("global", rows: rows, start: keys - rows, window: nil, tile: tile.rawValue, time: t)
                }
            }
        }
        // Decode steps.
        for (label, position, cache, mask) in [("window", 4095, ring, Inkling2AttentionMask.slidingDefault),
                                               ("global", 1023, linear, .globalDefault),
                                               ("global", 4095, linear, .globalDefault),
                                               ("global", 32767, linear, .globalDefault)] {
            let a = Inkling2DecodeAttention.Arguments(
                queries: .init(q), cache: cache, relative: .init(rel), table: .init(table), sink: .init(sink),
                gate: .init(gate), output: .init(out), partials: .init(partials), position: position, mask: mask)
            // 48 steps per command buffer (a decode token's visits), per-step time reported.
            let chained = Self.time(ctx, repeats: 5) { cb in for _ in 0..<48 { decode.encode(commandBuffer: cb, a) } }
            let t = (best: chained.best / 48, median: chained.median / 48)
            let keys = Inkling2DecodeAttention.keyRange(position: position, mask: mask).count
            let bytes = Double(keys) * 1024 * 2 * 2
            print(String(format: "[ink2-kbench] decode-attn %@ position=%d keys=%d best=%.1f us median=%.1f us %.1f GB/s(kv)",
                         label, position, keys, t.best * 1e6, t.median * 1e6, bytes / t.best / 1e9))
        }
    }

    static func reportAttention(_ label: String, rows: Int, start: Int, window: Int?, tile: String,
                                time: (best: Double, median: Double)) {
        var pairs = 0.0
        for i in 0..<rows {
            let position = start + i
            pairs += Double(window.map { min($0, position + 1) } ?? (position + 1))
        }
        let flops = pairs * 24 * 128 * 4
        print(String(format: "[ink2-kbench] attn %@ rows=%d keys_end=%d variant=%@ best=%.3f ms median=%.3f ms %.2f TFLOP/s %.2f us/token",
                     label, rows, start + rows, tile, time.best * 1e3, time.median * 1e3,
                     flops / time.best / 1e12, time.best * 1e6 / Double(rows)))
    }

    @Test(.enabled(if: enabled && only("epilogue"))) func epilogueAndConvolutions() throws {
        let ctx = try MetalContext()
        let library = try Inkling2KernelLibrary(context: ctx)
        let epilogue = try Inkling2QKVEpilogue(library: library)
        let tails = try Inkling2ConvTail(library: library)
        let conv = try Inkling2ShortConv(library: library)
        let norm = try Inkling2InputNorm(library: library)
        let rows = 2048, stride = 8664
        let y = Self.floatBuffer(ctx.device, rows * stride, salt: 1)
        let small = Self.floatBuffer(ctx.device, 3072 * 4, salt: 2)
        let tailBuffer = Self.floatBuffer(ctx.device, 3 * (1024 * 2 + 2304 * 2), salt: 3)
        let qOut = ctx.device.makeBuffer(length: rows * 3072 * 2, options: .storageModeShared)!
        let relOut = ctx.device.makeBuffer(length: rows * 384 * 4, options: .storageModeShared)!
        let kc = ctx.device.makeBuffer(length: (1024 + rows) * 1024 * 2, options: .storageModeShared)!
        let vc = ctx.device.makeBuffer(length: (1024 + rows) * 1024 * 2, options: .storageModeShared)!
        let cache = Inkling2KVView(keys: .init(kc), values: .init(vc), ringCapacity: 1024 + rows)
        for (label, columns) in [
            ("window", Inkling2QKVEpilogue.Columns(queries: 0, relative: 3072, keys: 6552, values: 7576)),
            ("global", Inkling2QKVEpilogue.Columns(queries: 0, relative: 3072, keys: 6552, values: 7608,
                                                   keyTaps: 7576, valueTaps: 8632)),
        ] {
            let t = Self.time(ctx) { cb in
                epilogue.encode(commandBuffer: cb, .init(
                    projections: .init(y), projectionStride: stride, columns: columns,
                    queryScale: .init(small), relativeScale: .init(small), keyConv: .init(small), valueConv: .init(small),
                    keyTail: .init(tailBuffer), valueTail: .init(tailBuffer, offset: 3 * 1024 * 4),
                    queries: .init(qOut), relative: .init(relOut), cache: cache, rows: rows, startPosition: 4096))
            }
            print(String(format: "[ink2-kbench] qkv-epilogue %@ T=%d best=%.3f ms median=%.3f ms", label, rows,
                         t.best * 1e3, t.median * 1e3))
        }
        let kvTail = Self.time(ctx) { cb in
            tails.encode(commandBuffer: cb, x: .init(y), xStride: stride, rows: rows, tails: .init(tailBuffer),
                         runs: [.init(xColumn: 6552, tailOffset: 0, channels: 1024),
                                .init(xColumn: 7576, tailOffset: 3072, channels: 1024)])
        }
        print(String(format: "[ink2-kbench] conv-tail k+v T=%d best=%.3f ms", rows, kvTail.best * 1e3))
        let x = Self.floatBuffer(ctx.device, rows * 2304, salt: 4)
        let residual = Self.floatBuffer(ctx.device, rows * 2304, salt: 5)
        let weight = Self.floatBuffer(ctx.device, 2304 * 4, salt: 6)
        let post = Self.time(ctx) { cb in
            conv.encode(commandBuffer: cb, x: .init(x), xStride: 2304, weight: .init(weight),
                        tail: .init(tailBuffer, offset: 6144 * 4), residual: .init(residual), residualStride: 2304,
                        output: .init(residual), outputStride: 2304, rows: rows)
        }
        print(String(format: "[ink2-kbench] post-conv+residual T=%d best=%.3f ms", rows, post.best * 1e3))
        let normOut = ctx.device.makeBuffer(length: rows * 2304 * 2, options: .storageModeShared)!
        let n = Self.time(ctx) { cb in
            norm.encode(commandBuffer: cb, x: .init(x), xStride: 2304, weight: .init(weight), output: .init(normOut),
                        outputStride: 2304, rows: rows)
        }
        print(String(format: "[ink2-kbench] input-norm T=%d best=%.3f ms", rows, n.best * 1e3))
    }

    @Test(.enabled(if: enabled && only("loop"))) func loopOps() throws {
        let ctx = try MetalContext()
        let library = try Inkling2KernelLibrary(context: ctx)
        let mix = try Inkling2DepthMix(library: library)
        let gate = try Inkling2MessageGateKernels(library: library)
        let projection = try Inkling2Projection(library: library)
        let rows = 2048, width = 2304, rank = 64
        let states = (0..<6).map { Self.floatBuffer(ctx.device, rows * width, salt: $0) }
        let queries = Self.floatBuffer(ctx.device, 4 * width, salt: 9, scale: 0.05)
        let mixOut = ctx.device.makeBuffer(length: rows * 4 * width * 4, options: .storageModeShared)!
        for (count, q) in [(6, 3), (6, 1), (3, 1), (2, 4)] {
            let t = try Self.time(ctx) { cb in
                try mix.encode(commandBuffer: cb, states: states.prefix(count).map { (.init($0), width) },
                               queries: .init(queries), queryCount: q, output: .init(mixOut), outputStride: q * width, rows: rows)
            }
            print(String(format: "[ink2-kbench] depth-mix states=%d queries=%d T=%d best=%.3f ms %.2f us/token",
                         count, q, rows, t.best * 1e3, t.best * 1e6 / Double(rows)))
        }
        // Three attention controllers of one loop visit.
        let h = ctx.device.makeBuffer(length: 3 * rows * 2 * width * 4, options: .storageModeShared)!
        let z = ctx.device.makeBuffer(length: rows * 3 * rank * 4, options: .storageModeShared)!
        let fgOut = ctx.device.makeBuffer(length: rows * 2 * width * 4, options: .storageModeShared)!
        let down = Self.floatBuffer(ctx.device, 3 * rank * 2 * width, salt: 11, scale: 0.02)
        let fgWeights = Self.floatBuffer(ctx.device, 2 * width * rank, salt: 12, scale: 0.1)
        let out = ctx.device.makeBuffer(length: rows * width * 4, options: .storageModeShared)!
        let tInputs = Self.time(ctx) { cb in
            for c in 0..<3 {
                gate.encodeInputs(commandBuffer: cb, x: .init(states[c]), xStride: width, message: .init(states[c + 3]),
                                  messageStride: width, output: .init(h, offset: c * rows * 2 * width * 4),
                                  outputStride: 2 * width, rows: rows)
            }
        }
        let tDown = Self.time(ctx) { cb in
            projection.encodeFloat(commandBuffer: cb, weights: down, input: h, output: z, tokens: rows, inputs: 2 * width,
                                   outStride: 3 * rank, segments: (0..<3).map { c in
                                       .init(weightOffset: c * rank * 2 * width * 4, inputOffset: c * rows * 2 * width,
                                             inputStride: 2 * width, rows: rank, outColumn: c * rank, format: .f32)
                                   })
        }
        let tGates = Self.time(ctx) { cb in
            for c in 0..<3 {
                projection.encodeFloat(commandBuffer: cb, weights: fgWeights, input: z, output: fgOut, tokens: rows,
                                       inputs: rank, outStride: 2 * width,
                                       segments: [.init(weightOffset: 0, inputOffset: c * rank, inputStride: 3 * rank,
                                                        rows: width, outColumn: 0, format: .f32),
                                                  .init(weightOffset: width * rank * 4, inputOffset: c * rank,
                                                        inputStride: 3 * rank, rows: width, outColumn: width, format: .f32)])
            }
        }
        let tCombine = Self.time(ctx) { cb in
            for c in 0..<3 {
                gate.encodeCombine(commandBuffer: cb, x: .init(states[c]), xStride: width, message: .init(states[c + 3]),
                                   messageStride: width, gates: .init(fgOut), gatesStride: 2 * width,
                                   forgetColumn: 0, gainColumn: width, output: .init(out), outputStride: width, rows: rows)
            }
        }
        let total = tInputs.best + tDown.best + tGates.best + tCombine.best
        print(String(format: "[ink2-kbench] message-gate x3 T=%d inputs=%.3f ms down=%.3f ms forget+gain=%.3f ms combine=%.3f ms total=%.2f us/token",
                     rows, tInputs.best * 1e3, tDown.best * 1e3, tGates.best * 1e3, tCombine.best * 1e3,
                     total * 1e6 / Double(rows)))
    }
}
