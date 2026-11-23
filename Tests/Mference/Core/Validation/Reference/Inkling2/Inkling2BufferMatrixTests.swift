import Foundation
import Metal
import Testing
import MferenceValidationSupport

/// `Inkling2BufferMatrix.apply`'s vDSP paths against decoding each row
/// (`row`, the ggml decode) and taking a Double dot product: the runner's
/// toy tests compare the GPU with this reference, so the fast product must
/// be right on its own.
@Suite("Inkling2BufferMatrix")
struct Inkling2BufferMatrixTests {

    private func slowApply(_ m: Inkling2BufferMatrix, _ x: [Float]) -> [Float] {
        (0..<m.rows).map { r in
            Float(zip(m.row(r), x).reduce(0.0) { $0 + Double($1.0) * Double($1.1) })
        }
    }

    private func maxRelative(_ got: [Float], _ want: [Float]) -> Float {
        let scale = max(want.map(abs).max()!, 1e-6)
        return zip(got, want).map { abs($0 - $1) }.max()! / scale
    }

    /// Rows past one tile (a partial last tile), per-block scales from the
    /// real quantizer, an offset that is 2- but not 4-byte aligned.
    @Test func q8_0MatchesTheRowDecode() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let rows = 2 * 64 + 3, cols = 2304, offset = 34 * 3
        var rng = SplitMix64(seed: 0x51_08)
        var bytes = [UInt8](repeating: 0, count: offset)
        for _ in 0..<rows {
            // Amplitudes vary by block so every block has its own scale.
            let row = (0..<cols).map { i in rng.uniform(-1, 1) * Float(1 + (i / 32) % 7) }
            bytes += Inkling2Q8_0.quantize(row)
        }
        let buffer = try #require(device.makeBuffer(bytes: bytes, length: bytes.count, options: .storageModeShared))
        let m = Inkling2BufferMatrix(buffer: buffer, offset: offset, rows: rows, cols: cols, format: .q8_0)
        let x = (0..<cols).map { _ in rng.uniform(-2, 2) }
        let err = maxRelative(m.apply(x), slowApply(m, x))
        #expect(err < 1e-5, "max relative error \(err)")
    }

    /// Random codes (every nibble value) and scale bytes from small to
    /// large, including 0x00 and 0x7F (both decode to 0); rows past a tile;
    /// one block per row (the expert down projection) and several.
    @Test(arguments: [64, 2304])
    func nvfp4MatchesTheRowDecode(cols: Int) throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let rows = 64 + 5, rowBytes = Inkling2NVFP4.rowBytes(k: cols)
        var rng = SplitMix64(seed: 0xF4 &+ UInt64(cols))
        var bytes = [UInt8](repeating: 0, count: rows * rowBytes)
        for b in 0..<(rows * rowBytes / Inkling2NVFP4.blockBytes) {
            let block = b * Inkling2NVFP4.blockBytes
            for s in 0..<Inkling2NVFP4.scaleBytesPerBlock {
                let pick = rng.next() % 16
                bytes[block + s] = pick == 0 ? 0 : pick == 1 ? 0x7F : UInt8(40 + rng.next() % 80)
            }
            for j in Inkling2NVFP4.scaleBytesPerBlock..<Inkling2NVFP4.blockBytes {
                bytes[block + j] = UInt8(truncatingIfNeeded: rng.next())
            }
        }
        let buffer = try #require(device.makeBuffer(bytes: bytes, length: bytes.count, options: .storageModeShared))
        let m = Inkling2BufferMatrix(buffer: buffer, offset: 0, rows: rows, cols: cols, format: .nvfp4(scale: 0.75))
        let x = (0..<cols).map { _ in rng.uniform(-2, 2) }
        let err = maxRelative(m.apply(x), slowApply(m, x))
        #expect(err < 1e-5, "max relative error \(err)")
    }

    @Test func f32MatchesTheRowDecode() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let rows = 7, cols = 96, offset = 16
        var rng = SplitMix64(seed: 0xF32)
        var values = [Float](repeating: 0, count: offset / 4)
        values += (0..<(rows * cols)).map { _ in rng.uniform(-1, 1) }
        let buffer = try #require(device.makeBuffer(bytes: values, length: values.count * 4,
                                                    options: .storageModeShared))
        let m = Inkling2BufferMatrix(buffer: buffer, offset: offset, rows: rows, cols: cols, format: .f32)
        let x = (0..<cols).map { _ in rng.uniform(-1, 1) }
        let err = maxRelative(m.apply(x), slowApply(m, x))
        #expect(err < 1e-5, "max relative error \(err)")
    }
}
