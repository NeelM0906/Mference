import Testing
import Foundation
import Metal
@testable import Mference
import MferenceValidationSupport

/// The attention block of one Inkling 2 loop-pass visit at production widths
/// on synthetic weights (a 16-key window keeps the rings small). Its steps, in
/// order, all through the `encode(encoder:)` forms:
/// 1. the block-residual depth mix;
/// 2. the q/k/v message gates: the many-row path on a prefill chunk (inputs,
///    the FP32 down product, then per controller forget and gain and the
///    combine), the decode form on a decode step;
/// 3. the three input norms and the stacked projection;
/// 4. the epilogue (tails in place on a decode step, the tail kernel after a chunk);
/// 5. attention;
/// 6. the output projection and the post-attention convolution.
///
/// `encode` puts the steps either into one compute encoder per kernel, or all
/// into one serial encoder.
final class Inkling2VisitFixture {
    typealias Step = (MTLComputeCommandEncoder) throws -> Void
    static let d = 2304
    static let maxRows = 9
    static let window = 16

    struct Segment {
        let offset: Int, rows: Int, format: Inkling2WeightFormat, input: Int, column: Int
    }

    /// One run's own outputs and state; every buffer starts zeroed.
    final class Run {
        let state: Inkling2AttentionState
        let mixOut, acts, gateIn, zRows, fg, gated, h, y, q, rel, att, o, post, partials, z: MTLBuffer

        init(_ device: MTLDevice, stride: Int) throws {
            state = try Inkling2AttentionState(device: device, configuration: .init(
                maxContext: 64, maxChunkTokens: Inkling2VisitFixture.maxRows, window: Inkling2VisitFixture.window))
            func zeroed(_ bytes: Int) -> MTLBuffer {
                let buffer = device.makeBuffer(length: bytes, options: .storageModeShared)!
                memset(buffer.contents(), 0, bytes)
                return buffer
            }
            let d = Inkling2VisitFixture.d, rows = Inkling2VisitFixture.maxRows
            mixOut = zeroed(rows * 3 * d * 4)
            acts = zeroed(9 * d * 4)            // decode: [mix x (up to 3) | messages (3) | gated (3)]
            gateIn = zeroed(3 * rows * 2 * d * 4)   // prefill gate path: [3][rows][rms(x); rms(m)]
            zRows = zeroed(rows * 3 * 64 * 4)
            fg = zeroed(rows * 2 * d * 4)
            gated = zeroed(3 * rows * d * 4)
            h = zeroed(3 * rows * d * 2)
            y = zeroed(rows * stride * 4)
            q = zeroed(rows * 3072 * 2)
            rel = zeroed(rows * 384 * 4)
            att = zeroed(rows * 3072 * 2)
            o = zeroed(rows * d * 4)
            post = zeroed((rows + 1) * d * 4)
            partials = zeroed(Inkling2DecodeAttention.partialsLength(keys: 64))
            z = zeroed(3 * 64 * 4)
        }

        /// Every output buffer, by name, for a byte comparison.
        var outputs: [(String, MTLBuffer)] {
            [("mix", mixOut), ("decode gates", acts), ("gate inputs", gateIn), ("gate down", zRows),
             ("forget and gain", fg), ("gated", gated), ("norms", h), ("projections", y), ("queries", q),
             ("relative", rel), ("attention", att), ("output projection", o), ("post conv", post)]
        }
    }

    let ctx: MetalContext
    let global: Bool
    let slot: Int
    let stride: Int
    let columns: Inkling2QKVEpilogue.Columns
    let queryCount: Int
    let mix: Inkling2DepthMix
    let gate: Inkling2MessageGateKernels
    let norm: Inkling2InputNorm
    let projection: Inkling2Projection
    let epilogue: Inkling2QKVEpilogue
    let tails: Inkling2ConvTail
    let prefill: Inkling2PrefillAttention
    let decode: Inkling2DecodeAttention
    let conv: Inkling2ShortConv
    let segments: [Segment]
    let projWeights: MTLBuffer, outWeights: MTLBuffer, gateWeights: MTLBuffer
    let normWeights: [MTLBuffer]
    let qClip: MTLBuffer, relClip: MTLBuffer, kConv: MTLBuffer, vConv: MTLBuffer, relBias: MTLBuffer, postConv: MTLBuffer
    let states: [MTLBuffer]
    let queries: MTLBuffer
    let messages: [Float]
    let prefillMessages: MTLBuffer

    init(global: Bool) throws {
        typealias B = Inkling2KernelBenchmarks
        let d = Self.d
        ctx = try MetalContext()
        let device = ctx.device
        let library = try Inkling2KernelLibrary(context: ctx)
        mix = try Inkling2DepthMix(library: library)
        gate = try Inkling2MessageGateKernels(library: library)
        norm = try Inkling2InputNorm(library: library)
        projection = try Inkling2Projection(library: library)
        epilogue = try Inkling2QKVEpilogue(library: library)
        tails = try Inkling2ConvTail(library: library)
        prefill = try Inkling2PrefillAttention(library: library)
        decode = try Inkling2DecodeAttention(library: library)
        conv = try Inkling2ShortConv(library: library)
        self.global = global
        slot = global ? 5 : 0
        stride = global ? 8664 : 8600
        let columns: Inkling2QKVEpilogue.Columns = global
            ? .init(queries: 0, relative: 3072, keys: 6552, values: 7608, keyTaps: 7576, valueTaps: 8632)
            : .init(queries: 0, relative: 3072, keys: 6552, values: 7576)
        self.columns = columns
        queryCount = global ? 3 : 1

        // Stacked weights: Q8_0 segments with a fixed block scale, BF16 rows, each at a 16-byte boundary.
        var layout: [(rows: Int, format: Inkling2WeightFormat, input: Int, column: Int)] = [
            (3072, .q8_0, 0, 0), (384, .q8_0, 0, 3072), (24, .bf16, 0, 3456), (3072, .q8_0, 0, 3480),
            (1024, .q8_0, 1, columns.keys), (1024, .q8_0, 2, columns.values),
        ]
        if global { layout += [(32, .bf16, 1, 7576), (32, .bf16, 2, 8632)] }
        var segments: [Segment] = []
        var size = 0
        for l in layout {
            segments.append(Segment(offset: size, rows: l.rows, format: l.format, input: l.input, column: l.column))
            size += l.rows * (l.format == .q8_0 ? d / 32 * 34 : d * 2)
            size = (size + 15) / 16 * 16
        }
        self.segments = segments
        let stacked = device.makeBuffer(length: size, options: .storageModeShared)!
        projWeights = stacked
        let bytes = stacked.contents().bindMemory(to: UInt8.self, capacity: size)
        var seed: UInt64 = 0x51ED_0000 &+ (global ? 1 : 0)
        for s in segments {
            if s.format == .q8_0 {
                Self.fillQ8(bytes + s.offset, rows: s.rows, inputs: d, seed: &seed)
            } else {
                let values = (0..<(s.rows * d)).map { Float(($0 &* 29 &+ s.column) % 89) / 89 * 0.2 - 0.1 }
                let encoded = Inkling2Q8Ref.bf16(values)
                for (i, b) in encoded.enumerated() { bytes[s.offset + i] = b }
            }
        }
        outWeights = B.q8Buffer(device, rows: d, inputs: 3072)
        normWeights = (0..<3).map { B.floatBuffer(device, d, salt: 11 + $0, scale: 1) }
        qClip = B.floatBuffer(device, 24 * 128, salt: 3, scale: 2)
        relClip = B.floatBuffer(device, 24 * 16, salt: 5, scale: 3)
        kConv = B.floatBuffer(device, 1024 * 4, salt: 7, scale: 0.5)
        vConv = B.floatBuffer(device, 1024 * 4, salt: 9, scale: 0.5)
        relBias = B.floatBuffer(device, 1024 * 16, salt: 13, scale: 0.5)
        postConv = B.floatBuffer(device, d * 4, salt: 17, scale: 0.5)
        states = (0..<3).map { B.floatBuffer(device, (Self.maxRows + 1) * d, salt: 19 + $0, scale: 2) }
        queries = B.floatBuffer(device, 3 * d, salt: 23, scale: 1)
        gateWeights = B.floatBuffer(device, 3 * (64 * 2 * d + 2 * d * 64), salt: 29, scale: 0.05)
        messages = (0..<(3 * d)).map { Float(($0 &* 41) % 103) / 103 * 4 - 2 }
        prefillMessages = B.floatBuffer(device, 3 * Self.maxRows * d, salt: 31, scale: 2)
    }

    static func fillQ8(_ p: UnsafeMutablePointer<UInt8>, rows: Int, inputs: Int, seed: inout UInt64) {
        let scale = Float16(0.004).bitPattern
        for block in 0..<(rows * inputs / 32) {
            let base = block * 34
            p[base] = UInt8(truncatingIfNeeded: scale)
            p[base + 1] = UInt8(truncatingIfNeeded: scale >> 8)
            for word in 0..<4 {
                seed = seed &* 6364136223846793005 &+ 1442695040888963407
                var v = seed
                for k in 0..<8 {
                    p[base + 2 + word * 8 + k] = UInt8(truncatingIfNeeded: v >> 24)
                    v >>= 8
                }
            }
        }
    }

    func makeRun() throws -> Run {
        let run = try Run(ctx.device, stride: stride)
        let acts = run.acts.contents().bindMemory(to: Float.self, capacity: 9 * Self.d)
        for (i, m) in messages.enumerated() { acts[3 * Self.d + i] = m }
        return run
    }

    /// The kernels of one chunk (`rows` rows at position `row0`), in order. Call after `ensureCapacity`.
    func steps(_ r: Run, row0: Int, rows: Int) -> [Step] {
        let d = Self.d, decodeStep = rows == 1
        let q = queryCount, maxRows = Self.maxRows
        let kv = r.state.kv(slot: slot), vt = r.state.tails(slot: slot), mask = r.state.mask(slot: slot)
        let controllerFloats = 64 * 2 * d + 2 * d * 64
        func downOffset(_ c: Int) -> Int { c * controllerFloats * 4 }
        func forgetOffset(_ c: Int) -> Int { (c * controllerFloats + 64 * 2 * d) * 4 }
        func gainOffset(_ c: Int) -> Int { (c * controllerFloats + 64 * 2 * d + d * 64) * 4 }
        var steps: [Step] = []
        steps.append { e in
            try self.mix.encode(encoder: e, states: self.states.map { (ref: Inkling2BufferRef($0, offset: row0 * d * 4), stride: d) },
                                queries: .init(self.queries), queryCount: q,
                                output: decodeStep ? .init(r.acts) : .init(r.mixOut), outputStride: q * d, rows: rows)
        }
        if decodeStep {
            let controllers = (0..<3).map { c in
                Inkling2MessageGateKernels.DecodeController(
                    down: downOffset(c), forget: forgetOffset(c), gain: gainOffset(c),
                    x: self.global ? c * d : 0, message: (3 + c) * d, output: (6 + c) * d)
            }
            steps.append { e in
                self.gate.encodeDecode(encoder: e, weights: self.gateWeights, activations: r.acts, z: .init(r.z),
                                       controllers: controllers)
            }
        } else {
            func x(_ c: Int) -> Inkling2BufferRef { .init(r.mixOut, offset: (self.global ? c : 0) * d * 4) }
            func message(_ c: Int) -> Inkling2BufferRef { .init(self.prefillMessages, offset: c * maxRows * d * 4) }
            for c in 0..<3 {
                steps.append { e in
                    self.gate.encodeInputs(encoder: e, x: x(c), xStride: q * d, message: message(c), messageStride: d,
                                           output: .init(r.gateIn, offset: c * maxRows * 2 * d * 4),
                                           outputStride: 2 * d, rows: rows)
                }
            }
            steps.append { e in
                self.projection.encodeFloat(encoder: e, weights: self.gateWeights, input: r.gateIn, output: r.zRows,
                                            tokens: rows, inputs: 2 * d, outStride: 3 * 64,
                                            segments: (0..<3).map { c in
                                                Inkling2ProjectionSegment(weightOffset: downOffset(c),
                                                                          inputOffset: c * maxRows * 2 * d,
                                                                          inputStride: 2 * d, rows: 64,
                                                                          outColumn: c * 64, format: .f32)
                                            })
            }
            for c in 0..<3 {
                steps.append { e in
                    self.projection.encodeFloat(encoder: e, weights: self.gateWeights, input: r.zRows, output: r.fg,
                                                tokens: rows, inputs: 64, outStride: 2 * d,
                                                segments: [.init(weightOffset: forgetOffset(c), inputOffset: c * 64,
                                                                 inputStride: 3 * 64, rows: d, outColumn: 0, format: .f32),
                                                           .init(weightOffset: gainOffset(c), inputOffset: c * 64,
                                                                 inputStride: 3 * 64, rows: d, outColumn: d, format: .f32)])
                    self.gate.encodeCombine(encoder: e, x: x(c), xStride: q * d, message: message(c), messageStride: d,
                                            gates: .init(r.fg), gatesStride: 2 * d, forgetColumn: 0, gainColumn: d,
                                            output: .init(r.gated, offset: c * maxRows * d * 4), outputStride: d,
                                            rows: rows)
                }
            }
        }
        for c in 0..<3 {
            let input: Inkling2BufferRef = decodeStep ? .init(r.acts, offset: (6 + c) * d * 4)
                                                      : .init(r.gated, offset: c * maxRows * d * 4)
            let weight = normWeights[global ? c : 0]
            steps.append { e in
                self.norm.encode(encoder: e, x: input, xStride: d, weight: .init(weight),
                                 output: .init(r.h, offset: c * maxRows * d * 2), outputStride: d, rows: rows)
            }
        }
        let projectionSegments = segments.map { s in
            Inkling2ProjectionSegment(weightOffset: s.offset, inputOffset: s.input * Self.maxRows * d,
                                      inputStride: d, rows: s.rows, outColumn: s.column, format: s.format)
        }
        steps.append { e in
            self.projection.encode(encoder: e, weights: self.projWeights, input: r.h, output: r.y, tokens: rows,
                                   inputs: d, outStride: self.stride, segments: projectionSegments)
        }
        steps.append { e in
            self.epilogue.encode(encoder: e, .init(
                projections: .init(r.y), projectionStride: self.stride, columns: self.columns,
                queryScale: .init(self.qClip), relativeScale: .init(self.relClip),
                keyConv: .init(self.kConv), valueConv: .init(self.vConv), keyTail: vt.keys, valueTail: vt.values,
                queries: .init(r.q), relative: .init(r.rel), cache: kv, rows: rows, startPosition: row0,
                advanceTails: decodeStep))
        }
        if !decodeStep {
            steps.append { e in
                self.tails.encode(encoder: e, x: .init(r.y), xStride: self.stride, rows: rows, tails: vt.keys,
                                  runs: vt.keyValueRuns(keyColumn: self.columns.keys, valueColumn: self.columns.values))
            }
        }
        steps.append { e in
            if decodeStep {
                self.decode.encode(encoder: e, .init(
                    queries: .init(r.q), cache: kv, relative: .init(r.rel), table: .init(self.relBias),
                    sink: .init(r.y, offset: 3456 * 4), gate: .init(r.y, offset: 3480 * 4), output: .init(r.att),
                    partials: .init(r.partials), position: row0, mask: mask))
            } else {
                self.prefill.encode(encoder: e, .init(
                    queries: .init(r.q), cache: kv, relative: .init(r.rel), table: .init(self.relBias),
                    sink: .init(r.y, offset: 3456 * 4), sinkStride: self.stride,
                    gate: .init(r.y, offset: 3480 * 4), gateStride: self.stride,
                    output: .init(r.att), rows: rows, startPosition: row0, mask: mask))
            }
        }
        steps.append { e in
            self.projection.encode(encoder: e, weights: self.outWeights, input: r.att, output: r.o, tokens: rows,
                                   inputs: 3072, outStride: d,
                                   segments: [.init(weightOffset: 0, inputOffset: 0, inputStride: 3072, rows: d, outColumn: 0)])
        }
        steps.append { e in
            self.conv.encode(encoder: e, x: .init(r.o), xStride: d, weight: .init(self.postConv), tail: vt.attentionOutput,
                             residual: .init(self.states[0], offset: row0 * d * 4), residualStride: d,
                             output: .init(r.post, offset: row0 * d * 4), outputStride: d, rows: rows,
                             advanceTail: decodeStep)
        }
        if !decodeStep {
            steps.append { e in
                self.tails.encode(encoder: e, x: .init(r.o), xStride: d, rows: rows, tails: vt.attentionOutput,
                                  runs: [.init(xColumn: 0, tailOffset: 0, channels: d)])
            }
        }
        return steps
    }

    /// Encodes one chunk into `cb`: one compute encoder per kernel, or every kernel in one serial encoder.
    func encode(_ r: Run, row0: Int, rows: Int, oneEncoder: Bool, into cb: MTLCommandBuffer) throws {
        try r.state.ensureCapacity(tokens: row0 + rows)
        let list = steps(r, row0: row0, rows: rows)
        if oneEncoder {
            let encoder = cb.makeComputeCommandEncoder()!
            for step in list { try step(encoder) }
            encoder.endEncoding()
        } else {
            for step in list {
                let encoder = cb.makeComputeCommandEncoder()!
                try step(encoder)
                encoder.endEncoding()
            }
        }
    }
}

/// Every K3 kernel through its `encode(encoder:)` form: a whole visit's chain
/// in one serial encoder, with no barriers, writes exactly the bytes that one
/// encoder per kernel writes (outputs, KV rows, tails). The other suites
/// check each kernel's values through the `encode(commandBuffer:)` wrappers.
@Suite(.serialized) struct Inkling2SerialEncoderTests {
    static func bytes(_ buffer: MTLBuffer, offset: Int = 0, count: Int? = nil) -> [UInt8] {
        Array(UnsafeRawBufferPointer(start: buffer.contents() + offset, count: count ?? buffer.length - offset))
    }

    @Test(arguments: [false, true])
    func visitInOneEncoderMatchesOneEncoderPerKernel(global: Bool) throws {
        let f = try Inkling2VisitFixture(global: global)
        let perKernel = try f.makeRun(), shared = try f.makeRun()
        // A 9-row prefill chunk, then a decode step (with the message gates and in-place tails).
        for (row0, rows) in [(0, Inkling2VisitFixture.maxRows), (Inkling2VisitFixture.maxRows, 1)] {
            let cb = f.ctx.queue.makeCommandBuffer()!
            try f.encode(perKernel, row0: row0, rows: rows, oneEncoder: false, into: cb)
            try f.encode(shared, row0: row0, rows: rows, oneEncoder: true, into: cb)
            cb.commit()
            cb.waitUntilCompleted()
            #expect(cb.status == .completed)
            perKernel.state.advance(by: rows)
            shared.state.advance(by: rows)
            let step = rows == 1 ? "decode step" : "prefill chunk"
            for ((name, a), (_, b)) in zip(perKernel.outputs, shared.outputs) {
                #expect(Self.bytes(a) == Self.bytes(b), "\(name) after the \(step)")
            }
            let total = row0 + rows
            let kvA = perKernel.state.kv(slot: f.slot), kvB = shared.state.kv(slot: f.slot)
            #expect(Self.bytes(kvA.keys.buffer, offset: kvA.keys.offset, count: total * 2048)
                    == Self.bytes(kvB.keys.buffer, offset: kvB.keys.offset, count: total * 2048), "keys after the \(step)")
            #expect(Self.bytes(kvA.values.buffer, offset: kvA.values.offset, count: total * 2048)
                    == Self.bytes(kvB.values.buffer, offset: kvB.values.offset, count: total * 2048), "values after the \(step)")
            let tailBytes = Inkling2VisitTails.floatsPerVisit * 4
            let tA = perKernel.state.tails(slot: f.slot).keys, tB = shared.state.tails(slot: f.slot).keys
            #expect(Self.bytes(tA.buffer, offset: tA.offset, count: tailBytes)
                    == Self.bytes(tB.buffer, offset: tB.offset, count: tailBytes), "tails after the \(step)")
        }
        // The chain ran: every row of the post-attention output is finite and not all zero.
        let post = Inkling2TestSupport.readFloats(shared.post, count: (Inkling2VisitFixture.maxRows + 1) * Inkling2VisitFixture.d)
        let finite = post.allSatisfy { $0.isFinite }
        #expect(finite)
        for row in 0...Inkling2VisitFixture.maxRows {
            let slice = post[(row * Inkling2VisitFixture.d)..<((row + 1) * Inkling2VisitFixture.d)]
            let written = slice.contains { $0 != 0 }
            #expect(written, "post-attention row \(row) was not written")
        }
    }
}
