import Foundation
import Metal
import Testing
@testable import Mference

/// The reference GEMM kernel against its CPU twin, for every type, layout,
/// entry point and output mode.
@Suite("ReferenceQuantGEMMTests", .serialized)
struct ReferenceQuantGEMMTests {
    static let types: [Native.GGMLType] = [.f32, .f16, .bf16, .q8_0, .q4_K, .q5_K, .q6_K, .iq4_xs, .iq3_xxs, .iq3_s]
    static let layouts: [Native.TileLayout] = [
        .gguf,
        Native.TileLayout(rowsPerTile: 2, blocksPerTile: 1, planes: false),
        Native.TileLayout(rowsPerTile: 4, blocksPerTile: 2, planes: true),
    ]

    /// GPU dequant of every block equals the CPU twin exactly (get_rows over
    /// every row, every layout).
    @Test(arguments: types)
    func getRowsMatchesCPUTwin(type: Native.GGMLType) throws {
        let ctx = try MetalContext()
        let gemm = Native.ReferenceQuantGEMM(context: ctx)
        try gemm.prepare(types: [type])
        let format = try #require(Native.QuantBlockFormat.of(type))
        let rows = 8, cols = format.elements == 1 ? 64 : format.elements * 2
        for layout in Self.layouts {
            let w = try RefTest.randomWeight(ctx.device, type: type, rows: rows, cols: cols, layout: layout, seed: 11)
            let tokens: [UInt32] = [3, 0, 7, 3, 5]
            let tokenBuffer = RefTest.buffer(ctx.device, bytes: tokens.withUnsafeBytes { Array($0) }, at: 8)
            let out = RefTest.tensor(ctx.device, floats: [Float](repeating: .nan, count: tokens.count * cols),
                                     rows: tokens.count, cols: cols)
            try RefTest.run(ctx) {
                try gemm.encodeGetRows($0, weight: w, tokens: Native.TokenSlot(buffer: tokenBuffer, offset: 8,
                                                                               count: tokens.count), output: out)
            }
            let expected = Native.ReferenceCPU.getRows(weight: w, tokens: tokens.map { Int32($0) })
            #expect(RefTest.floats(out).map(\.bitPattern) == expected.map(\.bitPattern), "\(type) \(layout)")
        }
    }

    /// Every entry point and output mode, every layout: GPU equals the CPU
    /// twin bit for bit (same order, no contraction, safe math).
    @Test(arguments: types)
    func gemmMatchesCPUTwin(type: Native.GGMLType) throws {
        let ctx = try MetalContext()
        let gemm = Native.ReferenceQuantGEMM(context: ctx)
        try gemm.prepare(types: [type])
        let format = try #require(Native.QuantBlockFormat.of(type))
        let outRows = 12, cols = format.elements == 1 ? 96 : format.elements * 2
        var rng = RefTest.SplitMix64(99)
        for layout in Self.layouts {
            let w = try RefTest.randomWeight(ctx.device, type: type, rows: outRows, cols: cols, layout: layout, seed: 21)
            for rows in [1, 3, 16, 21] {
                let x = (0..<rows * cols).map { _ in rng.unit() }
                let base = (0..<rows * outRows).map { _ in rng.unit() }
                let input = RefTest.tensor(ctx.device, floats: x, rows: rows, cols: cols, rowStride: cols * 4 + 64,
                                           offset: 256)
                let expected = Native.ReferenceCPU.gemm(weight: w, input: x, rows: rows)
                for mode in [Native.GEMMOutputMode.overwrite, .accumulate] {
                    let out = RefTest.tensor(ctx.device, floats: base, rows: rows, cols: outRows)
                    try RefTest.run(ctx) { enc in
                        switch rows {
                        case 1: try gemm.encodeGEMV(enc, weight: w, input: input, output: out, mode: mode)
                        case 2...16: try gemm.encodeVerify(enc, weight: w, input: input, output: out, mode: mode)
                        default: try gemm.encodePrefill(enc, weight: w, input: input, output: out, mode: mode)
                        }
                    }
                    let want = mode == .overwrite ? expected : zip(base, expected).map { $0 + $1 }
                    #expect(RefTest.floats(out).map(\.bitPattern) == want.map(\.bitPattern),
                            "\(type) \(layout) rows \(rows) \(mode)")
                }
            }
        }
    }

    @Test func entryPointsRejectWrongRowCounts() throws {
        let ctx = try MetalContext()
        let gemm = Native.ReferenceQuantGEMM(context: ctx)
        try gemm.prepare(types: [.q8_0])
        let w = try RefTest.randomWeight(ctx.device, type: .q8_0, rows: 4, cols: 32, seed: 1)
        let two = RefTest.tensor(ctx.device, floats: [Float](repeating: 0, count: 64), rows: 2, cols: 32)
        let out = RefTest.tensor(ctx.device, floats: [Float](repeating: 0, count: 8), rows: 2, cols: 4)
        try RefTest.run(ctx) { enc in
            #expect(throws: Native.NativeError.self) {
                try gemm.encodeGEMV(enc, weight: w, input: two, output: out, mode: .overwrite)
            }
            let one = RefTest.tensor(ctx.device, floats: [Float](repeating: 0, count: 32), rows: 1, cols: 32)
            #expect(throws: Native.NativeError.self) {
                try gemm.encodeVerify(enc, weight: w, input: one, output: out, mode: .overwrite)
            }
        }
        #expect(throws: Native.NativeError.self) { try gemm.prepare(types: [.q2_K]) }
    }

    @Test func calibrationNamesReferenceVariants() {
        let cal = Native.ReferenceKernelCalibration(family: .apple7)
        #expect(cal.variant(for: .q4_K, rows: .gemv) == "reference.q4_K.gemv")
        #expect(cal.variant(for: .iq4_xs, rows: .verify(rows: 4)) == "reference.iq4_xs.verify4")
        #expect(cal.variant(for: .q6_K, rows: .prefill(rows: 512)) == "reference.q6_K.prefill512")
    }
}
