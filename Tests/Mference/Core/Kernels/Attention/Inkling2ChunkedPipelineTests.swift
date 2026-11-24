import Testing
import Foundation
import Metal
@testable import Mference
import MferenceValidationSupport

/// The attention block of one window visit and one global visit run through
/// `Inkling2AttentionState` over several prefill chunks and decode steps
/// (epilogue, tail updates, ring or linear KV, prefill or decode attention)
/// must equal the CPU references applied to the whole sequence at once.
@Suite(.serialized) struct Inkling2ChunkedPipelineTests {
    typealias S = Inkling2TestSupport

    @Test(arguments: [0, 5])   // slot 0: window layer 0; slot 5: global layer 5
    func chunksAndDecodeStepsMatchWholeSequenceReference(slot: Int) throws {
        let ctx = try MetalContext()
        let library = try Inkling2KernelLibrary(context: ctx)
        let epilogue = try Inkling2QKVEpilogue(library: library)
        let tails = try Inkling2ConvTail(library: library)
        let prefill = try Inkling2PrefillAttention(library: library)
        let decode = try Inkling2DecodeAttention(library: library)
        let window = 16
        let state = try Inkling2AttentionState(
            device: ctx.device,
            configuration: .init(maxContext: 64, maxChunkTokens: 9, window: window, globalGrowthStep: 16))
        let global = Inkling2VisitPlan.bySlot[slot].isGlobal
        #expect(state.mask(slot: slot) == (global ? .globalDefault : .window(window)))
        // A small log floor so tau moves off 1 inside this short sequence.
        let mask: Inkling2AttentionMask = global ? .global(logFloor: 8, logAlpha: 0.3) : .window(window)

        var rng = SeedTree(0x4a00 + UInt64(slot)).key("ink2-pipeline")
        let chunks = [7, 9, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        let total = chunks.reduce(0, +)
        let stride = 8664
        let columns = global
            ? Inkling2QKVEpilogue.Columns(queries: 0, relative: 3072, keys: 6552, values: 7608, keyTaps: 7576, valueTaps: 8632)
            : Inkling2QKVEpilogue.Columns(queries: 0, relative: 3072, keys: 6552, values: 7608)
        let sinkColumn = 3456, gateColumn = 3480
        let y = S.random(total * stride, &rng, -2, 2)
        let qScale = S.random(24 * 128, &rng, 0.5, 3)
        let relScale = S.random(24 * 16, &rng, 0.5, 4)
        let kConv = S.random(1024 * 4, &rng, -0.5, 0.5)
        let vConv = S.random(1024 * 4, &rng, -0.5, 0.5)
        let table = S.random(1024 * 16, &rng, -0.5, 0.5)

        let qScaleBuffer = S.buffer(ctx.device, floats: qScale)
        let relScaleBuffer = S.buffer(ctx.device, floats: relScale)
        let kConvBuffer = S.buffer(ctx.device, floats: kConv)
        let vConvBuffer = S.buffer(ctx.device, floats: vConv)
        let tableBuffer = S.buffer(ctx.device, floats: table)
        let qBuffer = ctx.device.makeBuffer(length: 9 * 3072 * 2, options: .storageModeShared)!
        let relBuffer = ctx.device.makeBuffer(length: 9 * 384 * 4, options: .storageModeShared)!
        let outBuffer = ctx.device.makeBuffer(length: 9 * 3072 * 2, options: .storageModeShared)!
        let partials = ctx.device.makeBuffer(length: Inkling2DecodeAttention.partialsLength(keys: 64),
                                             options: .storageModeShared)!
        var got: [Float] = []
        var row = 0
        for rows in chunks {
            let position = state.position
            try state.ensureCapacity(tokens: position + rows)
            let yBuffer = S.buffer(ctx.device, floats: Array(y[(row * stride)..<((row + rows) * stride)]))
            let kv = state.kv(slot: slot)
            let vt = state.tails(slot: slot)
            S.run(ctx) { cb in
                epilogue.encode(commandBuffer: cb, .init(
                    projections: .init(yBuffer), projectionStride: stride, columns: columns,
                    queryScale: .init(qScaleBuffer), relativeScale: .init(relScaleBuffer),
                    keyConv: .init(kConvBuffer), valueConv: .init(vConvBuffer),
                    keyTail: vt.keys, valueTail: vt.values,
                    queries: .init(qBuffer), relative: .init(relBuffer), cache: kv, rows: rows, startPosition: position))
                tails.encode(commandBuffer: cb, x: .init(yBuffer), xStride: stride, rows: rows, tails: vt.keys,
                             runs: vt.keyValueRuns(keyColumn: columns.keys, valueColumn: columns.values))
                if rows == 1 {
                    decode.encode(commandBuffer: cb, .init(
                        queries: .init(qBuffer), cache: kv, relative: .init(relBuffer), table: .init(tableBuffer),
                        sink: .init(yBuffer, offset: sinkColumn * 4), gate: .init(yBuffer, offset: gateColumn * 4),
                        output: .init(outBuffer), partials: .init(partials), position: position,
                        mask: mask))
                } else {
                    prefill.encode(commandBuffer: cb, .init(
                        queries: .init(qBuffer), cache: kv, relative: .init(relBuffer), table: .init(tableBuffer),
                        sink: .init(yBuffer, offset: sinkColumn * 4), sinkStride: stride,
                        gate: .init(yBuffer, offset: gateColumn * 4), gateStride: stride,
                        output: .init(outBuffer), rows: rows, startPosition: position, mask: mask))
                }
            }
            got += S.readHalves(outBuffer, count: rows * 3072)
            state.advance(by: rows)
            row += rows
        }

        func slice(_ column: Int, _ width: Int) -> [Float] {
            (0..<total).flatMap { t in y[(t * stride + column)..<(t * stride + column + width)] }
        }
        let q = Inkling2NormRef.l2Clip(slice(columns.queries, 3072), rows: total, heads: 24, width: 128,
                                       scale: qScale, threshold: Inkling2RefShape.qkClip)
        let rel = Inkling2NormRef.l2Clip(slice(columns.relative, 384), rows: total, heads: 24, width: 16,
                                         scale: relScale, threshold: Inkling2RefShape.relClip)
        let zero = Inkling2ShortConvRef.zeroHistory(channels: 1024)
        let kc: [Float], vc: [Float]
        if global {
            kc = Inkling2ShortConvRef.dynamicConv(slice(columns.keys, 1024), weight: kConv, dynamic: slice(columns.keyTaps!, 32),
                                                  heads: 8, history: zero, rows: total, channels: 1024).output
            vc = Inkling2ShortConvRef.dynamicConv(slice(columns.values, 1024), weight: vConv, dynamic: slice(columns.valueTaps!, 32),
                                                  heads: 8, history: zero, rows: total, channels: 1024).output
        } else {
            kc = Inkling2ShortConvRef.staticConv(slice(columns.keys, 1024), weight: kConv, history: zero,
                                                 rows: total, channels: 1024).output
            vc = Inkling2ShortConvRef.staticConv(slice(columns.values, 1024), weight: vConv, history: zero,
                                                 rows: total, channels: 1024).output
        }
        let k = Inkling2NormRef.l2Clip(kc, rows: total, heads: 8, width: 128, scale: nil, threshold: Inkling2RefShape.qkClip)
        let v = Inkling2NormRef.unweightedRMS(vc, rows: total, heads: 8, width: 128)
        var reference = Inkling2AttentionRef.attend(
            q: q, k: k, v: v, rel: rel, table: table, sink: slice(sinkColumn, 24),
            queryCount: total, startPosition: 0, window: global ? nil : window,
            logScale: global ? .init(floor: 8, alpha: 0.3) : nil)
        reference = Inkling2AttentionRef.sigmoidGate(reference, gateLogits: slice(gateColumn, 3072))
        #expect(RelError.computeStrict(actual: got, reference: reference) < Tolerance.fp16ChainedReduction)
        #expect(state.position == total)
    }
}
