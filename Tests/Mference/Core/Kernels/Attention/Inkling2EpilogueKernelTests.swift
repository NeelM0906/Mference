import Testing
import Foundation
import Metal
@testable import Mference
import MferenceValidationSupport

/// QKV epilogue, convolution tails, the FP32 post convolution, the input norm
/// and decode attention against the CPU references.
@Suite(.serialized) struct Inkling2EpilogueKernelTests {
    typealias S = Inkling2TestSupport

    /// Columns of a window layer's stacked output, then a global layer's
    /// (which adds the 8 x 4 dynamic taps for keys and values).
    static func columns(dynamic: Bool) -> (Inkling2QKVEpilogue.Columns, stride: Int) {
        if dynamic {
            return (.init(queries: 0, relative: 3072, keys: 3480, values: 4536, keyTaps: 4504, valueTaps: 5560), 5596)
        }
        return (.init(queries: 0, relative: 3072, keys: 6552, values: 7576), 8604)
    }

    @Test(arguments: [(2, false), (9, false), (2, true), (9, true)])
    func qkvEpilogueAndTailsMatchReference(rows: Int, dynamic: Bool) throws {
        let ctx = try MetalContext()
        let library = try Inkling2KernelLibrary(context: ctx)
        let epilogue = try Inkling2QKVEpilogue(library: library)
        let tails = try Inkling2ConvTail(library: library)
        var rng = SeedTree(0x3c00 + UInt64(rows) + (dynamic ? 100 : 0)).key("ink2-qkv")
        let (cols, stride) = Self.columns(dynamic: dynamic)
        // Wide enough that most key heads exceed the clip after the convolution.
        let y = S.random(rows * stride, &rng, -8, 8)
        let qScale = S.random(24 * 128, &rng, 0.5, 4)
        let relScale = S.random(24 * 16, &rng, 0.5, 6)
        let kConv = S.random(1024 * 4, &rng, -0.6, 0.6)
        let vConv = S.random(1024 * 4, &rng, -0.6, 0.6)
        let kTail = S.random(3 * 1024, &rng, -3, 3)
        let vTail = S.random(3 * 1024, &rng, -3, 3)
        let capacity = 12, start = 25    // ring wraps inside the chunk
        let qOut = ctx.device.makeBuffer(length: rows * 3072 * 2, options: .storageModeShared)!
        let relOut = ctx.device.makeBuffer(length: rows * 384 * 4, options: .storageModeShared)!
        let kCache = ctx.device.makeBuffer(length: capacity * 1024 * 2, options: .storageModeShared)!
        let vCache = ctx.device.makeBuffer(length: capacity * 1024 * 2, options: .storageModeShared)!
        let tailBuffer = S.buffer(ctx.device, floats: kTail + vTail)
        let yBuffer = S.buffer(ctx.device, floats: y)
        let cache = Inkling2KVView(keys: .init(kCache), values: .init(vCache), ringCapacity: capacity)
        S.run(ctx) { cb in
            epilogue.encode(commandBuffer: cb, .init(
                projections: .init(yBuffer), projectionStride: stride, columns: cols,
                queryScale: .init(S.buffer(ctx.device, floats: qScale)),
                relativeScale: .init(S.buffer(ctx.device, floats: relScale)),
                keyConv: .init(S.buffer(ctx.device, floats: kConv)),
                valueConv: .init(S.buffer(ctx.device, floats: vConv)),
                keyTail: .init(tailBuffer, offset: 0), valueTail: .init(tailBuffer, offset: 3 * 1024 * 4),
                queries: .init(qOut), relative: .init(relOut), cache: cache, rows: rows, startPosition: start))
            tails.encode(commandBuffer: cb, x: .init(yBuffer), xStride: stride, rows: rows, tails: .init(tailBuffer),
                         runs: [.init(xColumn: cols.keys, tailOffset: 0, channels: 1024),
                                .init(xColumn: cols.values, tailOffset: 3 * 1024, channels: 1024)])
        }

        func slice(_ column: Int, _ width: Int) -> [Float] {
            (0..<rows).flatMap { t in y[(t * stride + column)..<(t * stride + column + width)] }
        }
        let qRef = Inkling2NormRef.l2Clip(slice(cols.queries, 3072), rows: rows, heads: 24, width: 128,
                                          scale: qScale, threshold: Inkling2RefShape.qkClip)
        let relRef = Inkling2NormRef.l2Clip(slice(cols.relative, 384), rows: rows, heads: 24, width: 16,
                                            scale: relScale, threshold: Inkling2RefShape.relClip)
        let kConvRef: Inkling2ShortConvRef.Result
        let vConvRef: Inkling2ShortConvRef.Result
        if dynamic {
            kConvRef = Inkling2ShortConvRef.dynamicConv(slice(cols.keys, 1024), weight: kConv,
                                                        dynamic: slice(cols.keyTaps!, 32), heads: 8,
                                                        history: kTail, rows: rows, channels: 1024)
            vConvRef = Inkling2ShortConvRef.dynamicConv(slice(cols.values, 1024), weight: vConv,
                                                        dynamic: slice(cols.valueTaps!, 32), heads: 8,
                                                        history: vTail, rows: rows, channels: 1024)
        } else {
            kConvRef = Inkling2ShortConvRef.staticConv(slice(cols.keys, 1024), weight: kConv, history: kTail,
                                                       rows: rows, channels: 1024)
            vConvRef = Inkling2ShortConvRef.staticConv(slice(cols.values, 1024), weight: vConv, history: vTail,
                                                       rows: rows, channels: 1024)
        }
        let kRef = Inkling2NormRef.l2Clip(kConvRef.output, rows: rows, heads: 8, width: 128, scale: nil,
                                          threshold: Inkling2RefShape.qkClip)
        var clipped = 0
        for v in 0..<(rows * 8) {
            let norm = (0..<128).reduce(Float(0)) { $0 + kConvRef.output[v * 128 + $1] * kConvRef.output[v * 128 + $1] }
            if norm.squareRoot() > Inkling2RefShape.qkClip { clipped += 1 }
        }
        #expect(clipped > rows * 4, "the key clip must be exercised")
        let vRef = Inkling2NormRef.unweightedRMS(vConvRef.output, rows: rows, heads: 8, width: 128)

        #expect(RelError.compute(actual: S.readHalves(qOut, count: rows * 3072), reference: qRef) < 2e-3)
        #expect(RelError.compute(actual: S.readFloats(relOut, count: rows * 384), reference: relRef) < 1e-5)
        var kGot = [Float](), vGot = [Float]()
        for t in 0..<rows {
            let slot = (start + t) % capacity
            kGot += S.readHalves(kCache, count: 1024, offset: slot * 2048)
            vGot += S.readHalves(vCache, count: 1024, offset: slot * 2048)
        }
        #expect(RelError.compute(actual: kGot, reference: kRef) < 2e-3)
        #expect(RelError.compute(actual: vGot, reference: vRef) < 2e-3)
        let tailsGot = S.readFloats(tailBuffer, count: 6 * 1024)
        #expect(tailsGot == kConvRef.history + vConvRef.history)
    }

    /// One row (decode): the epilogue advances the key and value tails itself.
    @Test(arguments: [false, true])
    func qkvEpilogueAdvancesTailsInPlaceForOneRow(dynamic: Bool) throws {
        let ctx = try MetalContext()
        let epilogue = try Inkling2QKVEpilogue(library: Inkling2KernelLibrary(context: ctx))
        var rng = SeedTree(0x3c80 + (dynamic ? 1 : 0)).key("ink2-qkv-decode")
        let (cols, stride) = Self.columns(dynamic: dynamic)
        let y = S.random(stride, &rng, -8, 8)
        let qScale = S.random(24 * 128, &rng, 0.5, 4)
        let relScale = S.random(24 * 16, &rng, 0.5, 6)
        let kConv = S.random(1024 * 4, &rng, -0.6, 0.6)
        let vConv = S.random(1024 * 4, &rng, -0.6, 0.6)
        let kTail = S.random(3 * 1024, &rng, -3, 3)
        let vTail = S.random(3 * 1024, &rng, -3, 3)
        let tailBuffer = S.buffer(ctx.device, floats: kTail + vTail)
        let kCache = ctx.device.makeBuffer(length: 8 * 1024 * 2, options: .storageModeShared)!
        let vCache = ctx.device.makeBuffer(length: 8 * 1024 * 2, options: .storageModeShared)!
        let qOut = ctx.device.makeBuffer(length: 3072 * 2, options: .storageModeShared)!
        let relOut = ctx.device.makeBuffer(length: 384 * 4, options: .storageModeShared)!
        S.run(ctx) { cb in
            epilogue.encode(commandBuffer: cb, .init(
                projections: .init(S.buffer(ctx.device, floats: y)), projectionStride: stride, columns: cols,
                queryScale: .init(S.buffer(ctx.device, floats: qScale)),
                relativeScale: .init(S.buffer(ctx.device, floats: relScale)),
                keyConv: .init(S.buffer(ctx.device, floats: kConv)), valueConv: .init(S.buffer(ctx.device, floats: vConv)),
                keyTail: .init(tailBuffer), valueTail: .init(tailBuffer, offset: 3 * 1024 * 4),
                queries: .init(qOut), relative: .init(relOut),
                cache: Inkling2KVView(keys: .init(kCache), values: .init(vCache), ringCapacity: 8),
                rows: 1, startPosition: 13, advanceTails: true))
        }
        let kIn = Array(y[cols.keys..<(cols.keys + 1024)]), vIn = Array(y[cols.values..<(cols.values + 1024)])
        let kRef: Inkling2ShortConvRef.Result, vRef: Inkling2ShortConvRef.Result
        if dynamic {
            kRef = Inkling2ShortConvRef.dynamicConv(kIn, weight: kConv, dynamic: Array(y[cols.keyTaps!..<(cols.keyTaps! + 32)]),
                                                    heads: 8, history: kTail, rows: 1, channels: 1024)
            vRef = Inkling2ShortConvRef.dynamicConv(vIn, weight: vConv, dynamic: Array(y[cols.valueTaps!..<(cols.valueTaps! + 32)]),
                                                    heads: 8, history: vTail, rows: 1, channels: 1024)
        } else {
            kRef = Inkling2ShortConvRef.staticConv(kIn, weight: kConv, history: kTail, rows: 1, channels: 1024)
            vRef = Inkling2ShortConvRef.staticConv(vIn, weight: vConv, history: vTail, rows: 1, channels: 1024)
        }
        let kExpected = Inkling2NormRef.l2Clip(kRef.output, rows: 1, heads: 8, width: 128, scale: nil,
                                               threshold: Inkling2RefShape.qkClip)
        let vExpected = Inkling2NormRef.unweightedRMS(vRef.output, rows: 1, heads: 8, width: 128)
        #expect(RelError.compute(actual: S.readHalves(kCache, count: 1024, offset: 5 * 2048), reference: kExpected) < 2e-3)
        #expect(RelError.compute(actual: S.readHalves(vCache, count: 1024, offset: 5 * 2048), reference: vExpected) < 2e-3)
        #expect(S.readFloats(tailBuffer, count: 6 * 1024) == kRef.history + vRef.history)
    }

    /// One row (decode): the post convolution advances its tail itself.
    @Test func postConvolutionAdvancesTailInPlaceForOneRow() throws {
        let ctx = try MetalContext()
        let conv = try Inkling2ShortConv(library: Inkling2KernelLibrary(context: ctx))
        var rng = SeedTree(0x3d80).key("ink2-sconv-decode")
        let channels = 2304
        let x = S.random(channels, &rng, -5, 5)
        let weight = S.random(channels * 4, &rng, -0.7, 0.7)
        let tail = S.random(3 * channels, &rng, -5, 5)
        let tailBuffer = S.buffer(ctx.device, floats: tail)
        let out = ctx.device.makeBuffer(length: channels * 4, options: .storageModeShared)!
        S.run(ctx) { cb in
            conv.encode(commandBuffer: cb, x: .init(S.buffer(ctx.device, floats: x)), xStride: channels,
                        weight: .init(S.buffer(ctx.device, floats: weight)), tail: .init(tailBuffer),
                        residual: nil, output: .init(out), outputStride: channels, rows: 1, advanceTail: true)
        }
        let reference = Inkling2ShortConvRef.staticConv(x, weight: weight, history: tail, rows: 1, channels: channels)
        #expect(RelError.compute(actual: S.readFloats(out, count: channels), reference: reference.output) < 1e-5)
        #expect(S.readFloats(tailBuffer, count: 3 * channels) == reference.history)
    }

    @Test(arguments: [1, 2, 7])
    func postConvolutionWithResidualMatchesReference(rows: Int) throws {
        let ctx = try MetalContext()
        let library = try Inkling2KernelLibrary(context: ctx)
        let conv = try Inkling2ShortConv(library: library)
        let tails = try Inkling2ConvTail(library: library)
        var rng = SeedTree(0x3d00 + UInt64(rows)).key("ink2-sconv")
        let channels = 2304
        let x = S.random(rows * channels, &rng, -5, 5)
        let weight = S.random(channels * 4, &rng, -0.7, 0.7)
        let tail = S.random(3 * channels, &rng, -5, 5)
        let residual = S.random(rows * channels, &rng, -2, 2)
        let xBuffer = S.buffer(ctx.device, floats: x)
        let tailBuffer = S.buffer(ctx.device, floats: tail)
        let out = S.buffer(ctx.device, floats: residual)      // in-place residual add
        S.run(ctx) { cb in
            conv.encode(commandBuffer: cb, x: .init(xBuffer), xStride: channels,
                        weight: .init(S.buffer(ctx.device, floats: weight)), tail: .init(tailBuffer),
                        residual: .init(out), residualStride: channels, output: .init(out),
                        outputStride: channels, rows: rows, channels: channels)
            tails.encode(commandBuffer: cb, x: .init(xBuffer), xStride: channels, rows: rows,
                         tails: .init(tailBuffer), runs: [.init(xColumn: 0, tailOffset: 0, channels: channels)])
        }
        let reference = Inkling2ShortConvRef.staticConv(x, weight: weight, history: tail, rows: rows, channels: channels)
        let expected = zip(reference.output, residual).map { $0 + $1 }
        #expect(RelError.compute(actual: S.readFloats(out, count: rows * channels), reference: expected) < 1e-5)
        #expect(S.readFloats(tailBuffer, count: 3 * channels) == reference.history)
    }

    @Test func inputNormMatchesReference() throws {
        let ctx = try MetalContext()
        let norm = try Inkling2InputNorm(library: Inkling2KernelLibrary(context: ctx))
        var rng = SeedTree(0x3e00).key("ink2-norm")
        let rows = 5, width = 2304, stride = 2304 + 8
        let x = S.random(rows * stride, &rng, -40, 40)
        let weight = S.random(width, &rng, 0.2, 2)
        let out = ctx.device.makeBuffer(length: rows * width * 2, options: .storageModeShared)!
        S.run(ctx) { cb in
            norm.encode(commandBuffer: cb, x: .init(S.buffer(ctx.device, floats: x)), xStride: stride,
                        weight: .init(S.buffer(ctx.device, floats: weight)), output: .init(out),
                        outputStride: width, rows: rows)
        }
        let dense = (0..<rows).flatMap { t in x[(t * stride)..<(t * stride + width)] }
        let reference = Inkling2NormRef.weightedRMS(dense, rows: rows, width: width, weight: weight)
        #expect(RelError.compute(actual: S.readHalves(out, count: rows * width), reference: reference) < 2e-3)
    }

    struct DecodeCase: Sendable {
        var position: Int
        var mask: Inkling2AttentionMask
        var ringCapacity: Int
        var relExtent: Int
        var gate: Bool
    }

    static let decodeCases: [DecodeCase] = [
        DecodeCase(position: 230, mask: .window(100), ringCapacity: 140, relExtent: 100, gate: true),
        DecodeCase(position: 40, mask: .window(100), ringCapacity: 140, relExtent: 100, gate: false),
        DecodeCase(position: 150, mask: .global(logFloor: 32, logAlpha: 0.1), ringCapacity: 0, relExtent: 24, gate: true),
        DecodeCase(position: 0, mask: .global(logFloor: 8192, logAlpha: 0.1), ringCapacity: 0, relExtent: 1024, gate: true),
    ]

    @Test(arguments: decodeCases.indices)
    func decodeAttentionMatchesReferenceAndPrefill(index: Int) throws {
        let c = Self.decodeCases[index]
        let ctx = try MetalContext()
        let library = try Inkling2KernelLibrary(context: ctx)
        let decode = try Inkling2DecodeAttention(library: library)
        let prefill = try Inkling2PrefillAttention(library: library)
        var rng = SeedTree(0x3f00 + UInt64(index)).key("ink2-decode")
        let positions = c.position + 1
        let q = S.halfRounded(S.random(24 * 128, &rng, -0.6, 0.6))
        let k = S.halfRounded(S.random(positions * 1024, &rng, -0.6, 0.6))
        let v = S.halfRounded(S.random(positions * 1024, &rng, -1, 1))
        let rel = S.random(24 * 16, &rng, -0.7, 0.7)
        let table = S.random(c.relExtent * 16, &rng, -1, 1)
        let sink = S.random(24, &rng, -1, 2)
        let gate = S.random(24 * 128, &rng, -3, 3)
        let slots = c.ringCapacity == 0 ? positions : c.ringCapacity
        var kc = [Float16](repeating: 0, count: slots * 1024), vc = kc
        for p in 0..<positions {
            let slot = c.ringCapacity == 0 ? p : p % c.ringCapacity
            for i in 0..<1024 {
                kc[slot * 1024 + i] = Float16(k[p * 1024 + i])
                vc[slot * 1024 + i] = Float16(v[p * 1024 + i])
            }
        }
        let cache = Inkling2KVView(keys: .init(S.buffer(ctx.device, halves: kc)),
                                   values: .init(S.buffer(ctx.device, halves: vc)), ringCapacity: c.ringCapacity)
        let qBuffer = S.buffer(ctx.device, halves: q.map { Float16($0) })
        let relBuffer = S.buffer(ctx.device, floats: rel)
        let tableBuffer = S.buffer(ctx.device, floats: table)
        let sinkBuffer = S.buffer(ctx.device, floats: sink)
        let gateBuffer: Inkling2BufferRef? = c.gate ? .init(S.buffer(ctx.device, floats: gate)) : nil
        let decodeOut = ctx.device.makeBuffer(length: 3072 * 2, options: .storageModeShared)!
        let prefillOut = ctx.device.makeBuffer(length: 3072 * 2, options: .storageModeShared)!
        let keys = Inkling2DecodeAttention.keyRange(position: c.position, mask: c.mask).count
        let partials = ctx.device.makeBuffer(length: Inkling2DecodeAttention.partialsLength(keys: keys),
                                             options: .storageModeShared)!
        S.run(ctx) { cb in
            decode.encode(commandBuffer: cb, .init(
                queries: .init(qBuffer), cache: cache, relative: .init(relBuffer), table: .init(tableBuffer),
                sink: .init(sinkBuffer), gate: gateBuffer, output: .init(decodeOut), partials: .init(partials),
                position: c.position, mask: c.mask, relExtent: c.relExtent))
            prefill.encode(commandBuffer: cb, .init(
                queries: .init(qBuffer), cache: cache, relative: .init(relBuffer), table: .init(tableBuffer),
                sink: .init(sinkBuffer), gate: gateBuffer, output: .init(prefillOut), rows: 1,
                startPosition: c.position, mask: c.mask, relExtent: c.relExtent))
        }
        var window: Int?
        var logScale: Inkling2AttentionRef.LogScale?
        switch c.mask {
        case .window(let w): window = w
        case .global(let floor, let alpha): logScale = .init(floor: floor, alpha: alpha)
        }
        var reference = Inkling2AttentionRef.attend(q: q, k: k, v: v, rel: rel, table: table, sink: sink,
                                                    queryCount: 1, startPosition: c.position,
                                                    relExtent: c.relExtent, window: window, logScale: logScale)
        if c.gate { reference = Inkling2AttentionRef.sigmoidGate(reference, gateLogits: gate) }
        #expect(RelError.compute(actual: S.readHalves(decodeOut, count: 3072), reference: reference) < Tolerance.fp16Reduction)
        #expect(RelError.compute(actual: S.readHalves(prefillOut, count: 3072), reference: reference) < Tolerance.fp16Reduction)
    }
}
