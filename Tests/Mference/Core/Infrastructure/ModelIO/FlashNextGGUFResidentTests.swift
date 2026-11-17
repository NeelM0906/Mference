import Foundation
import Metal
import Testing
@testable import Mference

/// The GGUF install's resident seams (WS4 U4): the throwing dtype whitelist
/// (R5) that turns index entries into Flash-Next matrices, and the F32 → BF16
/// narrowing behind the accessors.
@Suite struct FlashNextGGUFResidentTests {
    static func view(_ buffer: MTLBuffer, dtype: UInt8, rows: Int, columns: Int,
                     length: Int, scale: (Int, Int) = (0, 0), bias: (Int, Int) = (0, 0)) -> TensorView {
        TensorView(buffer: buffer, offset: 0, length: UInt64(length),
                   scaleOffset: UInt64(scale.0), scaleLength: UInt64(scale.1),
                   biasOffset: UInt64(bias.0), biasLength: UInt64(bias.1),
                   shape: (UInt32(rows), UInt32(columns), 0, 0), dtype: dtype)
    }

    @Test func whitelistAcceptsINT8TwoLevelAndBF16() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let buffer = try #require(device.makeBuffer(length: 1 << 20, options: .storageModeShared))
        let rows = 8, columns = 512, n = rows * columns
        // Q6_K: values n, d 2 per 256, m 1 per 16.
        let q6K = try FlashNextWeightMatrix.fromGGUF(
            Self.view(buffer, dtype: 0x80 | 14, rows: rows, columns: columns, length: n,
                      scale: (n, n / 128), bias: (n + n / 128, n / 16)), name: "q6k")
        guard case let .ggufInt8TwoLevel(w) = q6K else { Issue.record("Q6_K: \(q6K)"); return }
        #expect(w.source == .q6_K && w.rows == rows && w.n == columns)
        #expect(w.d.offset == n && w.m?.offset == n + n / 128)
        // Q8_0: d 2 per 32, no m.
        let q8 = try FlashNextWeightMatrix.fromGGUF(
            Self.view(buffer, dtype: 0x80 | 8, rows: rows, columns: columns, length: n,
                      scale: (n, n / 16)), name: "q8")
        guard case let .ggufInt8TwoLevel(w8) = q8 else { Issue.record("Q8_0: \(q8)"); return }
        #expect(w8.source == .q8_0 && w8.m == nil)
        let bf16 = try FlashNextWeightMatrix.fromGGUF(
            Self.view(buffer, dtype: 1, rows: rows, columns: columns, length: 2 * n), name: "bf16")
        guard case .bf16 = bf16 else { Issue.record("BF16: \(bf16)"); return }
    }

    /// Everything else throws instead of reaching a precondition: F32 (which
    /// the accessor must narrow first), vendor INT4/INT8 affine (0), native
    /// GGUF blocks (0x40 | id), an INT8-2L code of another ggml type, and
    /// INT8-2L regions that do not match the shape.
    @Test func whitelistRefusesEverythingElse() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let buffer = try #require(device.makeBuffer(length: 1 << 20, options: .storageModeShared))
        let rows = 8, columns = 512, n = rows * columns
        let refused: [(String, TensorView)] = [
            ("F32", Self.view(buffer, dtype: 3, rows: rows, columns: columns, length: 4 * n)),
            ("affine", Self.view(buffer, dtype: 0, rows: rows, columns: columns, length: n / 2)),
            ("native Q6_K", Self.view(buffer, dtype: 0x40 | 14, rows: rows, columns: columns, length: n / 256 * 210)),
            ("INT8-2L Q4_K", Self.view(buffer, dtype: 0x80 | 12, rows: rows, columns: columns, length: n)),
            ("Q6_K without m", Self.view(buffer, dtype: 0x80 | 14, rows: rows, columns: columns, length: n,
                                         scale: (n, n / 128))),
            ("Q8_0 short d", Self.view(buffer, dtype: 0x80 | 8, rows: rows, columns: columns, length: n,
                                       scale: (n, n / 32))),
        ]
        for (label, view) in refused {
            #expect(throws: ModelError.self, "\(label)") {
                _ = try FlashNextWeightMatrix.fromGGUF(view, name: label)
            }
        }
        // The embedding is native Q6_K rows, and only that.
        let table = Self.view(buffer, dtype: 0x40 | 14, rows: 4, columns: 512, length: 4 * 2 * 210)
        guard case .ggufQ6KRows = try FlashNextWeightMatrix.ggufEmbedding(table, name: "token_embd") else {
            Issue.record("Q6_K table refused"); return
        }
        #expect(throws: ModelError.self) {
            _ = try FlashNextWeightMatrix.ggufEmbedding(
                Self.view(buffer, dtype: 0x80 | 14, rows: rows, columns: columns, length: n,
                          scale: (n, n / 128), bias: (n + n / 128, n / 16)), name: "token_embd")
        }
    }

    /// `.exact` copies BF16-representable F32 bit for bit (signed zero and
    /// infinities included) and throws on the first value that is not;
    /// `.nearestEven` rounds it as `Quantization.bf16Bits` does.
    @Test func narrowingIsExactOrRoundsToNearestEven() throws {
        let exact: [Float] = [1, -0.0, 0, -2.5, .infinity, 3.0517578e-05, Float(bitPattern: 0x3F81_0000)]
        var out = [UInt16](repeating: 0, count: exact.count)
        try exact.withUnsafeBytes { src in
            try out.withUnsafeMutableBufferPointer {
                try Model.narrowF32ToBF16(src.baseAddress!, count: exact.count, into: $0.baseAddress!,
                                          mode: .exact, name: "t")
            }
        }
        #expect(out == exact.map { UInt16($0.bitPattern >> 16) })

        let inexact: [Float] = [1, 1.001, -1.0039062]
        #expect(throws: ModelError.self) {
            try inexact.withUnsafeBytes { src in
                try out.withUnsafeMutableBufferPointer {
                    try Model.narrowF32ToBF16(src.baseAddress!, count: inexact.count, into: $0.baseAddress!,
                                              mode: .exact, name: "t")
                }
            }
        }
        try inexact.withUnsafeBytes { src in
            try out.withUnsafeMutableBufferPointer {
                try Model.narrowF32ToBF16(src.baseAddress!, count: inexact.count, into: $0.baseAddress!,
                                          mode: .nearestEven, name: "t")
            }
        }
        #expect(Array(out.prefix(3)) == inexact.map(Quantization.bf16Bits))
    }
}
