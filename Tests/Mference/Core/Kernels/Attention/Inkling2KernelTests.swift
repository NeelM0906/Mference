import Testing
import Foundation
import Metal
@testable import Mference
import MferenceValidationSupport

/// Shared helpers for the Inkling 2 attention-side kernel tests.
enum Inkling2TestSupport {
    static func buffer(_ device: MTLDevice, floats: [Float]) -> MTLBuffer {
        floats.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: max($0.count, 4), options: .storageModeShared)! }
    }
    static func buffer(_ device: MTLDevice, halves: [Float16]) -> MTLBuffer {
        halves.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: max($0.count, 4), options: .storageModeShared)! }
    }
    static func buffer(_ device: MTLDevice, bytes: [UInt8]) -> MTLBuffer {
        bytes.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: max($0.count, 4), options: .storageModeShared)! }
    }
    static func readFloats(_ buffer: MTLBuffer, count: Int, offset: Int = 0) -> [Float] {
        let p = (buffer.contents() + offset).bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: p, count: count))
    }
    static func readHalves(_ buffer: MTLBuffer, count: Int, offset: Int = 0) -> [Float] {
        let p = (buffer.contents() + offset).bindMemory(to: Float16.self, capacity: count)
        return (0..<count).map { Float(p[$0]) }
    }
    static func random(_ count: Int, _ rng: inout SplitMix64, _ lo: Float = -1, _ hi: Float = 1) -> [Float] {
        (0..<count).map { _ in rng.uniform(lo, hi) }
    }
    static func halfRounded(_ values: [Float]) -> [Float] { values.map { Float(Float16($0)) } }

    static func run(_ context: MetalContext, _ body: (MTLCommandBuffer) throws -> Void) rethrows {
        let cb = context.queue.makeCommandBuffer()!
        try body(cb)
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.status == .completed, "command buffer: \(String(describing: cb.error))")
    }
}

@Suite(.serialized) struct Inkling2ProjectionTests {
    /// Four weights (Q8_0, BF16 like the sink projection, Q8_0, F32), two
    /// inputs, a padded output: partial row tiles (70, 24, 40 rows), partial
    /// token tiles, and columns outside every segment untouched.
    @Test(arguments: [(37, 96, Inkling2Projection.Path.tiled), (64, 2304, .tiled),
                      (1, 2304, .rows), (2, 96, .rows), (3, 96, .rows), (8, 2304, .rows), (5, 2304, .tiled),
                      (37, 2304, .tensorOps), (64, 128, .tensorOps),
                      (37, 96, .tiled64), (130, 2304, .tiled64)])
    func stackedProjectionMatchesReference(tokens: Int, inputs: Int, path: Inkling2Projection.Path) throws {
        let ctx = try MetalContext()
        let library = try Inkling2KernelLibrary(context: ctx)
        let projection = try Inkling2Projection(library: library)
        if path == .tensorOps && !projection.supportsTensorOps { return }
        var rng = SeedTree(0x3a00 + UInt64(tokens * 7 + inputs)).key("ink2-proj")
        let rows = [70, 24, 64, 40]
        let sinkValues = Inkling2Q8Ref.bf16Rounded(Inkling2TestSupport.random(rows[1] * inputs, &rng, -0.2, 0.2))
        let f32Values = Inkling2TestSupport.random(rows[3] * inputs, &rng, -0.2, 0.2)
        let weights = [
            Inkling2Q8Ref.quantize(rows: rows[0], inputs: inputs,
                                   values: Inkling2TestSupport.random(rows[0] * inputs, &rng, -0.2, 0.2)),
            Inkling2Q8Ref.bf16(sinkValues),
            Inkling2Q8Ref.quantize(rows: rows[2], inputs: inputs,
                                   values: Inkling2TestSupport.random(rows[2] * inputs, &rng, -0.2, 0.2)),
            Inkling2Q8Ref.f32(f32Values),
        ]
        // Weight buffer: an 8-byte lead-in, then the weights, each at a 16-byte boundary.
        var blob = [UInt8](repeating: 0xAB, count: 8)
        var offsets: [Int] = []
        for w in weights {
            while blob.count % 16 != 0 { blob.append(0xCD) }
            offsets.append(blob.count)
            blob += w
        }
        let x1 = Inkling2TestSupport.halfRounded(Inkling2TestSupport.random(tokens * inputs, &rng, -2, 2))
        let x2 = Inkling2TestSupport.halfRounded(Inkling2TestSupport.random(tokens * inputs, &rng, -2, 2))
        let inputBuffer = Inkling2TestSupport.buffer(ctx.device, halves: (x1 + x2).map { Float16($0) })
        let outStride = rows.reduce(0, +) + 5     // columns 94 ..< 99 belong to no segment
        let sentinel: Float = -12345
        let out = Inkling2TestSupport.buffer(ctx.device, floats: [Float](repeating: sentinel, count: tokens * outStride))
        let segments = [
            Inkling2ProjectionSegment(weightOffset: offsets[0], inputOffset: 0, inputStride: inputs, rows: rows[0], outColumn: 0),
            Inkling2ProjectionSegment(weightOffset: offsets[1], inputOffset: 0, inputStride: inputs, rows: rows[1], outColumn: 70,
                                      format: .bf16),
            Inkling2ProjectionSegment(weightOffset: offsets[2], inputOffset: tokens * inputs, inputStride: inputs,
                                      rows: rows[2], outColumn: 94 + 5),
            Inkling2ProjectionSegment(weightOffset: offsets[3], inputOffset: tokens * inputs, inputStride: inputs,
                                      rows: rows[3], outColumn: 163, format: .f32),
        ]
        Inkling2TestSupport.run(ctx) { cb in
            let taken = projection.encode(commandBuffer: cb, weights: Inkling2TestSupport.buffer(ctx.device, bytes: blob),
                                          input: inputBuffer, output: out, tokens: tokens, inputs: inputs,
                                          outStride: outStride, segments: segments, path: path)
            #expect(taken == path)
        }
        let got = Inkling2TestSupport.readFloats(out, count: tokens * outStride)
        let reference = Inkling2Q8Ref.stackedProjection(
            [.init(input: x1, weight: weights[0], rows: rows[0]),
             .init(input: x1, weight: weights[1], rows: rows[1], format: .bf16),
             .init(input: x2, weight: weights[2], rows: rows[2]),
             .init(input: x2, weight: weights[3], rows: rows[3], format: .f32)], tokens: tokens, inputs: inputs)
        var actual = [Float](), expected = [Float]()
        for t in 0..<tokens {
            for c in 0..<outStride {
                let value = got[t * outStride + c]
                if (94..<99).contains(c) {
                    #expect(value == sentinel, "column \(c) outside every segment was written")
                    continue
                }
                actual.append(value)
                expected.append(reference[t * 198 + (c < 94 ? c : c - 5)])
            }
        }
        // Weights decode to half in the tiled paths (d*q rounded); FP32 in the rows path.
        let tolerance: Float = path == .rows ? 1e-4 : 2e-3
        #expect(RelError.computeStrict(actual: actual, reference: expected) < tolerance)
    }

    /// The 64 x 64 and 64 x 32 tiles run the same fragment products in the same K order, so
    /// they write the same bits; the 64 x 64 kernel's edge fragments (stored through a
    /// per-simdgroup scratch) included. Q8_0, BF16 and F32 rows; partial row tiles (70, 24, 100,
    /// 40 rows) and partial token tiles (37, 130 and 1,001 tokens cut an 8 x 8 fragment); columns
    /// outside every segment untouched.
    @Test(arguments: [37, 130, 1001])
    func tiled64MatchesTiledBitForBit(tokens: Int) throws {
        let ctx = try MetalContext()
        let projection = try Inkling2Projection(library: Inkling2KernelLibrary(context: ctx))
        let inputs = 2304
        var rng = SeedTree(0x3b00 + UInt64(tokens)).key("ink2-proj-bits")
        let rows = [70, 24, 100, 40]
        let weights = [
            Inkling2Q8Ref.quantize(rows: rows[0], inputs: inputs,
                                   values: Inkling2TestSupport.random(rows[0] * inputs, &rng, -0.2, 0.2)),
            Inkling2Q8Ref.bf16(Inkling2Q8Ref.bf16Rounded(Inkling2TestSupport.random(rows[1] * inputs, &rng, -0.2, 0.2))),
            Inkling2Q8Ref.quantize(rows: rows[2], inputs: inputs,
                                   values: Inkling2TestSupport.random(rows[2] * inputs, &rng, -0.2, 0.2)),
            Inkling2Q8Ref.f32(Inkling2TestSupport.random(rows[3] * inputs, &rng, -0.2, 0.2)),
        ]
        var blob = [UInt8](repeating: 0xAB, count: 8)
        var offsets: [Int] = []
        for w in weights {
            while blob.count % 16 != 0 { blob.append(0xCD) }
            offsets.append(blob.count)
            blob += w
        }
        let weightBuffer = Inkling2TestSupport.buffer(ctx.device, bytes: blob)
        let x = Inkling2TestSupport.random(2 * tokens * inputs, &rng, -2, 2).map { Float16($0) }
        let inputBuffer = Inkling2TestSupport.buffer(ctx.device, halves: x)
        let outStride = rows.reduce(0, +) + 5     // columns 194 ..< 199 belong to no segment
        var columns: [Int] = []
        var column = 0
        for (i, r) in rows.enumerated() {
            columns.append(column)
            column += r + (i == 2 ? 5 : 0)
        }
        let formats: [Inkling2WeightFormat] = [.q8_0, .bf16, .q8_0, .f32]
        let segments = (0..<4).map { i in
            Inkling2ProjectionSegment(weightOffset: offsets[i], inputOffset: i < 2 ? 0 : tokens * inputs,
                                      inputStride: inputs, rows: rows[i], outColumn: columns[i], format: formats[i])
        }
        var bits: [[UInt32]] = []
        for path in [Inkling2Projection.Path.tiled, .tiled64] {
            let out = Inkling2TestSupport.buffer(ctx.device, floats: [Float](repeating: -12345, count: tokens * outStride))
            Inkling2TestSupport.run(ctx) { cb in
                projection.encode(commandBuffer: cb, weights: weightBuffer, input: inputBuffer, output: out,
                                  tokens: tokens, inputs: inputs, outStride: outStride, segments: segments, path: path)
            }
            bits.append(Inkling2TestSupport.readFloats(out, count: tokens * outStride).map(\.bitPattern))
        }
        let differing = zip(bits[0], bits[1]).filter { $0 != $1 }.count
        #expect(differing == 0, "\(differing) of \(tokens * outStride) outputs differ between the 64 x 32 and 64 x 64 tiles")
        let untouched = (0..<tokens).allSatisfy { t in (194..<199).allSatisfy { bits[1][t * outStride + $0] == Float(-12345).bitPattern } }
        #expect(untouched, "a column outside every segment was written")
    }
}

@Suite(.serialized) struct Inkling2PrefillAttentionTests {
    struct Case {
        var rows: Int
        var start: Int
        var mask: Inkling2AttentionMask
        var ringCapacity: Int
        var relExtent: Int
        var gate: Bool
    }

    static let cases: [Case] = [
        // Window 40 over a ring of 90 that has wrapped; extent 24 < window.
        Case(rows: 45, start: 150, mask: .window(40), ringCapacity: 90, relExtent: 24, gate: true),
        // Chunk at sequence start, window longer than the history.
        Case(rows: 33, start: 0, mask: .window(40), ringCapacity: 80, relExtent: 40, gate: false),
        // Global, tau active (floor 32), far keys past the extent.
        Case(rows: 37, start: 70, mask: .global(logFloor: 32, logAlpha: 0.1), ringCapacity: 0, relExtent: 24, gate: true),
        // One row (decode shape) on a global visit.
        Case(rows: 1, start: 100, mask: .global(logFloor: 64, logAlpha: 0.3), ringCapacity: 0, relExtent: 24, gate: true),
    ]

    @Test(arguments: cases.indices, Inkling2PrefillAttention.Variant.allCases)
    func prefillAttentionMatchesReference(index: Int, variant: Inkling2PrefillAttention.Variant) throws {
        let c = Self.cases[index]
        let ctx = try MetalContext()
        let attention = try Inkling2PrefillAttention(library: Inkling2KernelLibrary(context: ctx))
        // GPUs without tensor support (CI runners) build only the simdgroup variants.
        if !attention.supports(variant) { return }
        let hq = 24, hkv = 8, d = 128, r = 16
        var rng = SeedTree(0x3b00 + UInt64(index)).key("ink2-attn")
        let positions = c.start + c.rows
        let q = Inkling2TestSupport.halfRounded(Inkling2TestSupport.random(c.rows * hq * d, &rng, -0.6, 0.6))
        let k = Inkling2TestSupport.halfRounded(Inkling2TestSupport.random(positions * hkv * d, &rng, -0.6, 0.6))
        let v = Inkling2TestSupport.halfRounded(Inkling2TestSupport.random(positions * hkv * d, &rng, -1, 1))
        let rel = Inkling2TestSupport.random(c.rows * hq * r, &rng, -0.7, 0.7)
        let table = Inkling2TestSupport.random(c.relExtent * r, &rng, -1, 1)
        let sink = Inkling2TestSupport.random(c.rows * hq, &rng, -1, 2)
        let gate = Inkling2TestSupport.random(c.rows * hq * d, &rng, -3, 3)

        // Cache: logical position p at slot p % capacity (later positions overwrite).
        let slots = c.ringCapacity == 0 ? positions : c.ringCapacity
        var kc = [Float16](repeating: 0, count: slots * hkv * d)
        var vc = [Float16](repeating: 0, count: slots * hkv * d)
        for p in 0..<positions {
            let slot = c.ringCapacity == 0 ? p : p % c.ringCapacity
            for i in 0..<(hkv * d) {
                kc[slot * hkv * d + i] = Float16(k[p * hkv * d + i])
                vc[slot * hkv * d + i] = Float16(v[p * hkv * d + i])
            }
        }
        let outBuffer = ctx.device.makeBuffer(length: c.rows * hq * d * 2, options: .storageModeShared)!
        let args = Inkling2PrefillAttention.Arguments(
            queries: .init(Inkling2TestSupport.buffer(ctx.device, halves: q.map { Float16($0) })),
            cache: .init(keys: .init(Inkling2TestSupport.buffer(ctx.device, halves: kc)),
                         values: .init(Inkling2TestSupport.buffer(ctx.device, halves: vc)),
                         ringCapacity: c.ringCapacity),
            relative: .init(Inkling2TestSupport.buffer(ctx.device, floats: rel)),
            table: .init(Inkling2TestSupport.buffer(ctx.device, floats: table)),
            sink: .init(Inkling2TestSupport.buffer(ctx.device, floats: sink)),
            gate: c.gate ? .init(Inkling2TestSupport.buffer(ctx.device, floats: gate)) : nil,
            output: .init(outBuffer), rows: c.rows, startPosition: c.start, mask: c.mask, relExtent: c.relExtent)
        Inkling2TestSupport.run(ctx) { attention.encode(commandBuffer: $0, args, variant: variant) }

        var window: Int?
        var logScale: Inkling2AttentionRef.LogScale?
        switch c.mask {
        case .window(let w): window = w
        case .global(let floor, let alpha): logScale = .init(floor: floor, alpha: alpha)
        }
        var reference = Inkling2AttentionRef.attend(
            q: q, k: k, v: v, rel: rel, table: table, sink: sink,
            queryCount: c.rows, startPosition: c.start, relExtent: c.relExtent,
            window: window, logScale: logScale)
        if c.gate { reference = Inkling2AttentionRef.sigmoidGate(reference, gateLogits: gate) }
        let got = Inkling2TestSupport.readHalves(outBuffer, count: c.rows * hq * d)
        #expect(RelError.computeStrict(actual: got, reference: reference) < Tolerance.fp16Reduction)
    }
}
