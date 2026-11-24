import Testing
import Foundation
import Metal
@testable import Mference
import MferenceValidationSupport

/// K3's kernels on the real Inkling 2 Mini weights: the attention block of one
/// window layer (0) and one global layer (5), and the four message gates of
/// core layer 6. Each stage is checked against its CPU reference with the
/// previous stage's GPU output as input, so a failure names its kernel. The
/// inputs are real embedding rows after `token_embd_norm`: 12 prefill rows,
/// then one decode step. Every projection path and prefill variant this GPU
/// builds is checked against the same reference.
///
/// A model run (it maps the install): run it only through `model-hold.sh`,
/// with `MFERENCE_INKLING2_GTURBO` naming the install. It reads resident
/// tensors only (the one dense weight buffer) and opens no expert file. The
/// CPU references copy a projection weight 256 rows at a time (at most
/// 836 KB, about 3 MB dequantized), never a whole one; only the small F32
/// tensors (at most 1.2 MB each) are copied whole. Each stage prints one
/// `[ink2-real]` line.
@Suite(.serialized) struct Inkling2RealWeightsKernelTests {
    typealias S = Inkling2TestSupport
    static let install = ProcessInfo.processInfo.environment["MFERENCE_INKLING2_GTURBO"]
    static let width = 2304
    static let prefillRows = 12
    /// 13 token ids: 12 prefill rows and the decode row.
    static let tokenIDs = [3, 17, 271, 1_024, 4_096, 9_999, 31_337, 65_000, 100_003, 150_001, 190_007, 42, 7_777]
    static let q8Type: UInt8 = 0x48, bf16Type: UInt8 = 1, f32Type: UInt8 = 3

    /// One stage within `bound` (`computeStrict`: a non-finite value fails it).
    static func check(_ stage: String, _ actual: [Float], _ reference: [Float], _ bound: Float) {
        let error = RelError.computeStrict(actual: actual, reference: reference)
        print("[ink2-real] \(stage): rel \(String(format: "%.2e", error)) bound \(String(format: "%.0e", bound))")
        #expect(error < bound, "\(stage): rel \(error)")
    }

    static func expectTensor(_ view: TensorView, dtype: UInt8, bytes: Int, _ name: String) {
        #expect(view.dtype == dtype && Int(view.length) == bytes, "\(name): dtype \(view.dtype), \(view.length) bytes")
    }

    static func loadModel(_ ctx: MetalContext) throws -> Model {
        try Model.load(directoryURL: URL(fileURLWithPath: install!), device: ctx.device,
                       expecting: .inkling2Mini, integrityPolicy: .trustedReceiptWhenValid)
    }

    /// The install's one resident buffer, and its tensors' bytes.
    struct Resident {
        let model: Model
        let buffer: MTLBuffer

        init(_ model: Model) throws {
            self.model = model
            buffer = try model.inkling2AttnQ(layer: 0).buffer
        }

        func ref(_ view: TensorView) throws -> Inkling2BufferRef {
            try #require(view.buffer === buffer, "a tensor outside the one resident buffer")
            return .init(buffer, offset: Int(view.offset))
        }

        func bytes(_ view: TensorView, from start: Int = 0, count: Int? = nil) -> [UInt8] {
            let n = count ?? Int(view.length) - start
            precondition(start >= 0 && start + n <= Int(view.length))
            return Array(UnsafeRawBufferPointer(start: buffer.contents().advanced(by: Int(view.offset) + start), count: n))
        }

        /// A small F32 tensor (norms, clips, taps, the bias table, gate weights).
        func floats(_ view: TensorView) -> [Float] {
            Inkling2Q8Ref.decodeF32(bytes(view), offset: 0, count: Int(view.length) / 4)
        }

        /// `[tokens][rows]` of `input` (`[tokens][inputs]`) through a resident
        /// weight, by `Inkling2Q8Ref.stackedProjection` on 256 rows at a time.
        func project(_ view: TensorView, rows: Int, inputs: Int, format: Inkling2Q8Ref.Format,
                     input: [Float], tokens: Int) -> [Float] {
            let rowBytes: Int
            switch format {
            case .q8_0: rowBytes = Inkling2Q8Ref.rowBytes(inputs: inputs)
            case .bf16: rowBytes = inputs * 2
            case .f32: rowBytes = inputs * 4
            }
            var out = [Float](repeating: 0, count: tokens * rows)
            var start = 0
            while start < rows {
                let count = min(256, rows - start)
                let block = bytes(view, from: start * rowBytes, count: count * rowBytes)
                let part = Inkling2Q8Ref.stackedProjection([.init(input: input, weight: block, rows: count, format: format)],
                                                           tokens: tokens, inputs: inputs)
                for t in 0..<tokens {
                    for r in 0..<count { out[t * rows + start + r] = part[t * count + r] }
                }
                start += count
            }
            return out
        }

        /// Embedding rows of `ids` after `token_embd_norm`, `[ids.count][2304]`.
        func embedded(_ ids: [Int]) throws -> [Float] {
            let table = try model.inkling2Embedding
            let rowBytes = Inkling2Q8Ref.rowBytes(inputs: width)
            var rows: [Float] = []
            for id in ids {
                rows += Inkling2Q8Ref.dequantize(bytes(table, from: id * rowBytes, count: rowBytes), rows: 1, inputs: width)
            }
            return Inkling2NormRef.weightedRMS(rows, rows: ids.count, width: width,
                                               weight: floats(try model.inkling2EmbeddingNorm))
        }
    }

    /// One weight segment of the stacked attention projection.
    struct Segment {
        let view: TensorView
        let rows: Int
        let format: Inkling2Q8Ref.Format
        /// 0 = the query input, 1 = keys, 2 = values (one input on window layers).
        let input: Int
        let column: Int
    }

    /// Columns `column ..< column + count` of every row of a `[rows][stride]` matrix.
    static func columns(_ matrix: [Float], stride: Int, _ column: Int, _ count: Int, rows: Range<Int>) -> [Float] {
        rows.flatMap { t in matrix[(t * stride + column)..<(t * stride + column + count)] }
    }

    @Test(.enabled(if: install != nil), arguments: [0, 5])
    func attentionBlockMatchesReferenceOnRealWeights(layer: Int) throws {
        let ctx = try MetalContext()
        let r = try Resident(try Self.loadModel(ctx))
        let model = r.model
        let library = try Inkling2KernelLibrary(context: ctx)
        let norm = try Inkling2InputNorm(library: library)
        let projection = try Inkling2Projection(library: library)
        let epilogue = try Inkling2QKVEpilogue(library: library)
        let tails = try Inkling2ConvTail(library: library)
        let prefill = try Inkling2PrefillAttention(library: library)
        let decode = try Inkling2DecodeAttention(library: library)
        let conv = try Inkling2ShortConv(library: library)

        let global = Inkling2VisitPlan.isGlobal(layer: layer)
        let width = Self.width, T = Self.prefillRows, total = T + 1
        let stride = global ? 8664 : 8600
        let columns = global
            ? Inkling2QKVEpilogue.Columns(queries: 0, relative: 3072, keys: 6552, values: 7608, keyTaps: 7576, valueTaps: 8632)
            : Inkling2QKVEpilogue.Columns(queries: 0, relative: 3072, keys: 6552, values: 7576)
        let sinkColumn = 3456, gateColumn = 3480
        let tag = "L\(layer)"

        // Tensors, checked against the formats and sizes the kernels assume.
        var segments = [
            Segment(view: try model.inkling2AttnQ(layer: layer), rows: 3072, format: .q8_0, input: 0, column: 0),
            Segment(view: try model.inkling2AttnRel(layer: layer), rows: 384, format: .q8_0, input: 0, column: 3072),
            Segment(view: try model.inkling2AttnSinkProj(layer: layer), rows: 24, format: .bf16, input: 0, column: sinkColumn),
            Segment(view: try model.inkling2AttnGate(layer: layer), rows: 3072, format: .q8_0, input: 0, column: gateColumn),
            Segment(view: try model.inkling2AttnK(layer: layer), rows: 1024, format: .q8_0, input: global ? 1 : 0,
                    column: columns.keys),
            Segment(view: try model.inkling2AttnV(layer: layer), rows: 1024, format: .q8_0, input: global ? 2 : 0,
                    column: columns.values),
        ]
        if global {
            segments.append(Segment(view: try model.inkling2AttnKConvProj(layer: layer), rows: 32, format: .bf16,
                                    input: 1, column: columns.keyTaps!))
            segments.append(Segment(view: try model.inkling2AttnVConvProj(layer: layer), rows: 32, format: .bf16,
                                    input: 2, column: columns.valueTaps!))
        }
        for s in segments {
            let bytes = s.format == .q8_0 ? s.rows * Inkling2Q8Ref.rowBytes(inputs: width) : s.rows * width * 2
            Self.expectTensor(s.view, dtype: s.format == .q8_0 ? Self.q8Type : Self.bf16Type, bytes: bytes,
                              "\(tag) segment at column \(s.column)")
        }
        let normViews = global
            ? [try model.inkling2AttnQInputNorm(layer: layer), try model.inkling2AttnKInputNorm(layer: layer),
               try model.inkling2AttnVInputNorm(layer: layer)]
            : [try model.inkling2AttnNorm(layer: layer)]
        let qClip = try model.inkling2AttnQClip(layer: layer), relClip = try model.inkling2AttnRelClip(layer: layer)
        let kConv = try model.inkling2AttnKConv(layer: layer), vConv = try model.inkling2AttnVConv(layer: layer)
        let relBias = try model.inkling2AttnRelBias(layer: layer), postConv = try model.inkling2AttnPostConv(layer: layer)
        let attnOutput = try model.inkling2AttnOutput(layer: layer)
        for view in normViews { Self.expectTensor(view, dtype: Self.f32Type, bytes: width * 4, "\(tag) input norm") }
        Self.expectTensor(qClip, dtype: Self.f32Type, bytes: 24 * 128 * 4, "\(tag) attn_q_clip")
        Self.expectTensor(relClip, dtype: Self.f32Type, bytes: 24 * 16 * 4, "\(tag) attn_rel_clip")
        Self.expectTensor(kConv, dtype: Self.f32Type, bytes: 1024 * 4 * 4, "\(tag) attn_k_conv")
        Self.expectTensor(vConv, dtype: Self.f32Type, bytes: 1024 * 4 * 4, "\(tag) attn_v_conv")
        Self.expectTensor(relBias, dtype: Self.f32Type, bytes: 1024 * 16 * 4, "\(tag) attn_rel_bias")
        Self.expectTensor(postConv, dtype: Self.f32Type, bytes: width * 4 * 4, "\(tag) attn_post_conv")
        Self.expectTensor(attnOutput, dtype: Self.q8Type, bytes: width * Inkling2Q8Ref.rowBytes(inputs: 3072),
                          "\(tag) attn_output")

        let normRefs = try normViews.map { try r.ref($0) }
        let qClipRef = try r.ref(qClip), relClipRef = try r.ref(relClip)
        let kConvRef = try r.ref(kConv), vConvRef = try r.ref(vConv)
        let relBiasRef = try r.ref(relBias), postConvRef = try r.ref(postConv)
        _ = try r.ref(attnOutput)
        for s in segments { _ = try r.ref(s.view) }
        func projectionSegments(row0: Int) -> [Inkling2ProjectionSegment] {
            segments.map { s in
                Inkling2ProjectionSegment(weightOffset: Int(s.view.offset), inputOffset: (s.input * total + row0) * width,
                                          inputStride: width, rows: s.rows, outColumn: s.column,
                                          format: s.format == .q8_0 ? .q8_0 : .bf16)
            }
        }
        func outputSegment(row0: Int) -> [Inkling2ProjectionSegment] {
            [.init(weightOffset: Int(attnOutput.offset), inputOffset: row0 * 3072, inputStride: 3072, rows: width, outColumn: 0)]
        }

        let x = try r.embedded(Self.tokenIDs)
        func shared(_ bytes: Int) -> MTLBuffer { ctx.device.makeBuffer(length: bytes, options: .storageModeShared)! }
        let xBuffer = S.buffer(ctx.device, floats: x)
        let h = shared(3 * total * width * 2)          // FP16 [3][total][2304]: the q, k and v inputs
        let y = shared(total * stride * 4)
        let qOut = shared(total * 3072 * 2), rel = shared(total * 384 * 4)
        let att = shared(total * 3072 * 2), o = shared(total * width * 4), post = shared(total * width * 4)
        let partials = shared(Inkling2DecodeAttention.partialsLength(keys: 64))
        let state = try Inkling2AttentionState(device: ctx.device, configuration: .init(maxContext: 64, maxChunkTokens: T))
        let slot = layer
        var paths: [Inkling2Projection.Path] = []

        // 12 prefill rows, then one decode step, as a runner encodes them.
        for (row0, rows) in [(0, T), (T, 1)] {
            let step = rows == 1
            try state.ensureCapacity(tokens: row0 + rows)
            let kv = state.kv(slot: slot), vt = state.tails(slot: slot), mask = state.mask(slot: slot)
            S.run(ctx) { cb in
                for (i, weight) in normRefs.enumerated() {
                    norm.encode(commandBuffer: cb, x: .init(xBuffer, offset: row0 * width * 4), xStride: width, weight: weight,
                                output: .init(h, offset: (i * total + row0) * width * 2), outputStride: width, rows: rows)
                }
                paths.append(projection.encode(commandBuffer: cb, weights: r.buffer, input: h, output: y,
                                               outputOffset: row0 * stride * 4, tokens: rows, inputs: width,
                                               outStride: stride, segments: projectionSegments(row0: row0)))
                epilogue.encode(commandBuffer: cb, .init(
                    projections: .init(y, offset: row0 * stride * 4), projectionStride: stride, columns: columns,
                    queryScale: qClipRef, relativeScale: relClipRef, keyConv: kConvRef, valueConv: vConvRef,
                    keyTail: vt.keys, valueTail: vt.values,
                    queries: .init(qOut, offset: row0 * 3072 * 2), relative: .init(rel, offset: row0 * 384 * 4),
                    cache: kv, rows: rows, startPosition: row0, advanceTails: step))
                if !step {
                    tails.encode(commandBuffer: cb, x: .init(y), xStride: stride, rows: rows, tails: vt.keys,
                                 runs: vt.keyValueRuns(keyColumn: columns.keys, valueColumn: columns.values))
                    prefill.encode(commandBuffer: cb, .init(
                        queries: .init(qOut), cache: kv, relative: .init(rel), table: relBiasRef,
                        sink: .init(y, offset: sinkColumn * 4), sinkStride: stride,
                        gate: .init(y, offset: gateColumn * 4), gateStride: stride,
                        output: .init(att), rows: rows, startPosition: 0, mask: mask))
                } else {
                    decode.encode(commandBuffer: cb, .init(
                        queries: .init(qOut, offset: row0 * 3072 * 2), cache: kv,
                        relative: .init(rel, offset: row0 * 384 * 4), table: relBiasRef,
                        sink: .init(y, offset: (row0 * stride + sinkColumn) * 4),
                        gate: .init(y, offset: (row0 * stride + gateColumn) * 4),
                        output: .init(att, offset: row0 * 3072 * 2), partials: .init(partials),
                        position: row0, mask: mask))
                }
                paths.append(projection.encode(commandBuffer: cb, weights: r.buffer, input: att, output: o,
                                               outputOffset: row0 * width * 4, tokens: rows, inputs: 3072,
                                               outStride: width, segments: outputSegment(row0: row0)))
                conv.encode(commandBuffer: cb, x: .init(o, offset: row0 * width * 4), xStride: width, weight: postConvRef,
                            tail: vt.attentionOutput, residual: nil, output: .init(post, offset: row0 * width * 4),
                            outputStride: width, rows: rows, advanceTail: step)
                if !step {
                    tails.encode(commandBuffer: cb, x: .init(o), xStride: width, rows: rows, tails: vt.attentionOutput,
                                 runs: [.init(xColumn: 0, tailOffset: 0, channels: width)])
                }
            }
            state.advance(by: rows)
        }
        print("[ink2-real] \(tag) paths: prefill \(paths[0].rawValue)/\(paths[1].rawValue), decode \(paths[2].rawValue)/\(paths[3].rawValue)")
        func bound(_ path: Inkling2Projection.Path) -> Float { path == .rows ? 1e-4 : 2e-3 }
        let prefillRange = 0..<T, decodeRange = T..<total

        // 1. Input norms.
        let hGot = (0..<normViews.count).map { i in S.readHalves(h, count: total * width, offset: i * total * width * 2) }
        for (i, view) in normViews.enumerated() {
            Self.check("\(tag) input norm \(i)", hGot[i],
                       Inkling2NormRef.weightedRMS(x, rows: total, width: width, weight: r.floats(view)), 2e-3)
        }

        // 2. Stacked projection, from the GPU's normed inputs.
        var yRef = [Float](repeating: 0, count: total * stride)
        for s in segments {
            let part = r.project(s.view, rows: s.rows, inputs: width, format: s.format, input: hGot[s.input], tokens: total)
            for t in 0..<total {
                for c in 0..<s.rows { yRef[t * stride + s.column + c] = part[t * s.rows + c] }
            }
        }
        let yGot = S.readFloats(y, count: total * stride)
        Self.check("\(tag) projection prefill (\(paths[0].rawValue))", Array(yGot[0..<(T * stride)]),
                   Array(yRef[0..<(T * stride)]), bound(paths[0]))
        Self.check("\(tag) projection decode (\(paths[2].rawValue))", Array(yGot[(T * stride)...]),
                   Array(yRef[(T * stride)...]), bound(paths[2]))
        var allPaths: [Inkling2Projection.Path] = [.tiled, .tiled64, .rows]
        if projection.supportsTensorOps { allPaths.append(.tensorOps) }
        for path in allPaths {
            let tokens = path == .rows ? Inkling2Projection.maxRowsPathTokens : T
            let scratch = S.buffer(ctx.device, floats: [Float](repeating: .nan, count: tokens * stride))
            S.run(ctx) { cb in
                _ = projection.encode(commandBuffer: cb, weights: r.buffer, input: h, output: scratch, tokens: tokens,
                                      inputs: width, outStride: stride, segments: projectionSegments(row0: 0), path: path)
            }
            Self.check("\(tag) projection path \(path.rawValue), \(tokens) rows", S.readFloats(scratch, count: tokens * stride),
                       Array(yRef[0..<(tokens * stride)]), bound(path))
        }

        // 3. Epilogue (clips, K/V convolutions over all 13 rows, V norm, KV store), from the GPU's projections.
        func cols(_ column: Int, _ count: Int) -> [Float] {
            Self.columns(yGot, stride: stride, column, count, rows: 0..<total)
        }
        let qRef = Inkling2NormRef.l2Clip(cols(columns.queries, 3072), rows: total, heads: 24, width: 128,
                                          scale: r.floats(qClip), threshold: Inkling2RefShape.qkClip)
        let relRef = Inkling2NormRef.l2Clip(cols(columns.relative, 384), rows: total, heads: 24, width: 16,
                                            scale: r.floats(relClip), threshold: Inkling2RefShape.relClip)
        let zero = Inkling2ShortConvRef.zeroHistory(channels: 1024)
        let kc: [Float], vc: [Float]
        if global {
            kc = Inkling2ShortConvRef.dynamicConv(cols(columns.keys, 1024), weight: r.floats(kConv),
                                                  dynamic: cols(columns.keyTaps!, 32), heads: 8, history: zero,
                                                  rows: total, channels: 1024).output
            vc = Inkling2ShortConvRef.dynamicConv(cols(columns.values, 1024), weight: r.floats(vConv),
                                                  dynamic: cols(columns.valueTaps!, 32), heads: 8, history: zero,
                                                  rows: total, channels: 1024).output
        } else {
            kc = Inkling2ShortConvRef.staticConv(cols(columns.keys, 1024), weight: r.floats(kConv), history: zero,
                                                 rows: total, channels: 1024).output
            vc = Inkling2ShortConvRef.staticConv(cols(columns.values, 1024), weight: r.floats(vConv), history: zero,
                                                 rows: total, channels: 1024).output
        }
        let kRef = Inkling2NormRef.l2Clip(kc, rows: total, heads: 8, width: 128, scale: nil, threshold: Inkling2RefShape.qkClip)
        let vRef = Inkling2NormRef.unweightedRMS(vc, rows: total, heads: 8, width: 128)
        let cache = state.kv(slot: slot)
        let qGot = S.readHalves(qOut, count: total * 3072), relGot = S.readFloats(rel, count: total * 384)
        let kGot = S.readHalves(cache.keys.buffer, count: total * 1024, offset: cache.keys.offset)
        let vGot = S.readHalves(cache.values.buffer, count: total * 1024, offset: cache.values.offset)
        Self.check("\(tag) epilogue queries", qGot, qRef, 2e-3)
        Self.check("\(tag) epilogue relative", relGot, relRef, 1e-5)
        Self.check("\(tag) epilogue keys (13 rows, decode row after in-place tails)", kGot, kRef, 2e-3)
        Self.check("\(tag) epilogue values", vGot, vRef, 2e-3)

        // 4. Attention, from the GPU's queries, cache and projections.
        let table = r.floats(relBias)
        let sinks = cols(sinkColumn, 24), gates = cols(gateColumn, 3072)
        func attentionRef(_ rows: Range<Int>) -> [Float] {
            let keys = rows.upperBound
            let out = Inkling2AttentionRef.attend(
                q: Array(qGot[(rows.lowerBound * 3072)..<(rows.upperBound * 3072)]),
                k: Array(kGot[0..<(keys * 1024)]), v: Array(vGot[0..<(keys * 1024)]),
                rel: Array(relGot[(rows.lowerBound * 384)..<(rows.upperBound * 384)]), table: table,
                sink: Array(sinks[(rows.lowerBound * 24)..<(rows.upperBound * 24)]),
                queryCount: rows.count, startPosition: rows.lowerBound,
                window: global ? nil : Inkling2RefShape.slidingWindow, logScale: global ? .init() : nil)
            return Inkling2AttentionRef.sigmoidGate(out, gateLogits: Array(gates[(rows.lowerBound * 3072)..<(rows.upperBound * 3072)]))
        }
        let attGot = S.readHalves(att, count: total * 3072)
        let prefillAttention = attentionRef(prefillRange)
        Self.check("\(tag) prefill attention (default variant)", Array(attGot[0..<(T * 3072)]), prefillAttention,
                   Tolerance.fp16Reduction)
        Self.check("\(tag) decode attention", Array(attGot[(T * 3072)...]), attentionRef(decodeRange), Tolerance.fp16Reduction)
        for variant in Inkling2PrefillAttention.Variant.allCases where prefill.supports(variant) {
            let scratch = S.buffer(ctx.device, halves: [Float16](repeating: .nan, count: T * 3072))
            S.run(ctx) { cb in
                prefill.encode(commandBuffer: cb, .init(
                    queries: .init(qOut), cache: cache, relative: .init(rel), table: relBiasRef,
                    sink: .init(y, offset: sinkColumn * 4), sinkStride: stride,
                    gate: .init(y, offset: gateColumn * 4), gateStride: stride,
                    output: .init(scratch), rows: T, startPosition: 0, mask: state.mask(slot: slot)), variant: variant)
            }
            Self.check("\(tag) prefill attention variant \(variant.rawValue)", S.readHalves(scratch, count: T * 3072),
                       prefillAttention, Tolerance.fp16Reduction)
        }

        // 5. Output projection, from the GPU's attention output.
        let oRef = r.project(attnOutput, rows: width, inputs: 3072, format: .q8_0, input: attGot, tokens: total)
        let oGot = S.readFloats(o, count: total * width)
        Self.check("\(tag) output projection prefill (\(paths[1].rawValue))", Array(oGot[0..<(T * width)]),
                   Array(oRef[0..<(T * width)]), bound(paths[1]))
        Self.check("\(tag) output projection decode (\(paths[3].rawValue))", Array(oGot[(T * width)...]),
                   Array(oRef[(T * width)...]), bound(paths[3]))

        // 6. Post-attention convolution over all 13 rows, from the GPU's output projection.
        let postRef = Inkling2ShortConvRef.staticConv(oGot, weight: r.floats(postConv),
                                                      history: Inkling2ShortConvRef.zeroHistory(channels: width),
                                                      rows: total, channels: width).output
        Self.check("\(tag) post-attention convolution (decode row after in-place tail)",
                   S.readFloats(post, count: total * width), postRef, 1e-5)

        // 7. The tails the decode step advanced in place: the inputs of rows 10, 11, 12, oldest first, exactly.
        let advanced = state.tails(slot: slot)
        let last3 = (T - 2)..<total
        #expect(S.readFloats(advanced.keys.buffer, count: 3 * 1024, offset: advanced.keys.offset)
                == Self.columns(yGot, stride: stride, columns.keys, 1024, rows: last3), "\(tag) key tail after decode")
        #expect(S.readFloats(advanced.values.buffer, count: 3 * 1024, offset: advanced.values.offset)
                == Self.columns(yGot, stride: stride, columns.values, 1024, rows: last3), "\(tag) value tail after decode")
        #expect(S.readFloats(advanced.attentionOutput.buffer, count: 3 * width, offset: advanced.attentionOutput.offset)
                == Array(oGot[((T - 2) * width)...]), "\(tag) attention-output tail after decode")
    }

    /// The q, k, v and MoE message gates of core layer 6 with the real F32
    /// controller weights: the many-row path (12 rows) and the decode form
    /// (one row, four controllers in one call).
    @Test(.enabled(if: install != nil))
    func messageGatesMatchReferenceOnRealWeights() throws {
        let ctx = try MetalContext()
        let r = try Resident(try Self.loadModel(ctx))
        let library = try Inkling2KernelLibrary(context: ctx)
        let projection = try Inkling2Projection(library: library)
        let gate = try Inkling2MessageGateKernels(library: library)
        let layer = 6, width = Self.width, rank = 64, rows = Self.prefillRows
        let controllers = Inkling2MessageGate.allCases
        let views = try controllers.map { g in
            (down: try r.model.inkling2LoopGateDown(layer: layer, g), forget: try r.model.inkling2LoopGateForget(layer: layer, g),
             gain: try r.model.inkling2LoopGateGain(layer: layer, g))
        }
        for (c, v) in views.enumerated() {
            Self.expectTensor(v.down, dtype: Self.f32Type, bytes: rank * 2 * width * 4, "ctrl \(c) down")
            Self.expectTensor(v.forget, dtype: Self.f32Type, bytes: width * rank * 4, "ctrl \(c) forget")
            Self.expectTensor(v.gain, dtype: Self.f32Type, bytes: width * rank * 4, "ctrl \(c) gain")
            _ = try r.ref(v.down)
            _ = try r.ref(v.forget)
            _ = try r.ref(v.gain)
        }
        // x: the 12 embedded rows; each controller's message: other rows, at its own scale.
        let ids = Array(Self.tokenIDs.prefix(rows))
        let x = try r.embedded(ids)
        let messages = try (0..<controllers.count).map { c in
            try r.embedded(ids.map { ($0 * 7 + 1_000 * (c + 1)) % 190_000 }).map { $0 * Float(c + 1) * 0.5 }
        }
        let refs = (0..<controllers.count).map { c in
            Inkling2LoopRef.messageGate(x: x, message: messages[c], down: r.floats(views[c].down),
                                        forget: r.floats(views[c].forget), gain: r.floats(views[c].gain),
                                        rows: rows, hidden: width, rank: rank)
        }

        // Many-row path: inputs, one down product for all four, forget and gain, combine.
        let n = controllers.count
        let xBuffer = S.buffer(ctx.device, floats: x)
        let mBuffers = messages.map { S.buffer(ctx.device, floats: $0) }
        let h = ctx.device.makeBuffer(length: n * rows * 2 * width * 4, options: .storageModeShared)!
        let z = ctx.device.makeBuffer(length: rows * n * rank * 4, options: .storageModeShared)!
        let fg = ctx.device.makeBuffer(length: rows * 2 * width * 4, options: .storageModeShared)!
        S.run(ctx) { cb in
            for c in 0..<n {
                gate.encodeInputs(commandBuffer: cb, x: .init(xBuffer), xStride: width, message: .init(mBuffers[c]),
                                  messageStride: width, output: .init(h, offset: c * rows * 2 * width * 4),
                                  outputStride: 2 * width, rows: rows)
            }
            projection.encodeFloat(commandBuffer: cb, weights: r.buffer, input: h, output: z, tokens: rows,
                                   inputs: 2 * width, outStride: n * rank,
                                   segments: (0..<n).map { c in
                                       Inkling2ProjectionSegment(weightOffset: Int(views[c].down.offset),
                                                                 inputOffset: c * rows * 2 * width, inputStride: 2 * width,
                                                                 rows: rank, outColumn: c * rank, format: .f32)
                                   })
        }
        for c in 0..<n {
            let out = S.buffer(ctx.device, floats: [Float](repeating: .nan, count: rows * width))
            S.run(ctx) { cb in
                projection.encodeFloat(commandBuffer: cb, weights: r.buffer, input: z, output: fg, tokens: rows, inputs: rank,
                                       outStride: 2 * width,
                                       segments: [.init(weightOffset: Int(views[c].forget.offset), inputOffset: c * rank,
                                                        inputStride: n * rank, rows: width, outColumn: 0, format: .f32),
                                                  .init(weightOffset: Int(views[c].gain.offset), inputOffset: c * rank,
                                                        inputStride: n * rank, rows: width, outColumn: width, format: .f32)])
                gate.encodeCombine(commandBuffer: cb, x: .init(xBuffer), xStride: width, message: .init(mBuffers[c]),
                                   messageStride: width, gates: .init(fg), gatesStride: 2 * width,
                                   forgetColumn: 0, gainColumn: width, output: .init(out), outputStride: width, rows: rows)
            }
            Self.check("gate L6 \(controllers[c]) many-row (\(rows) rows)", S.readFloats(out, count: rows * width), refs[c], 1e-4)
        }

        // Decode form: row 0, four controllers in one call sharing x, each output NaN-filled first
        // (the synthetic test covers an output written over its own x).
        // Activations: [x | m0 | m1 | m2 | m3 | out0 | out1 | out2 | out3].
        let acts = S.buffer(ctx.device, floats: Array(x[0..<width]) + messages.flatMap { Array($0[0..<width]) }
                                                + [Float](repeating: .nan, count: n * width))
        let decodeControllers = (0..<n).map { c in
            Inkling2MessageGateKernels.DecodeController(
                down: Int(views[c].down.offset), forget: Int(views[c].forget.offset), gain: Int(views[c].gain.offset),
                x: 0, message: (1 + c) * width, output: (1 + n + c) * width)
        }
        let zDecode = ctx.device.makeBuffer(length: n * rank * 4, options: .storageModeShared)!
        S.run(ctx) { cb in
            gate.encodeDecode(commandBuffer: cb, weights: r.buffer, activations: acts, z: .init(zDecode),
                              controllers: decodeControllers)
        }
        for c in 0..<n {
            Self.check("gate L6 \(controllers[c]) decode form", S.readFloats(acts, count: width, offset: decodeControllers[c].output * 4),
                       Array(refs[c][0..<width]), 1e-4)
        }
    }
}
