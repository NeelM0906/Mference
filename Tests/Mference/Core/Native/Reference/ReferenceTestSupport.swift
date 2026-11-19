import Foundation
import Metal
@testable import Mference

/// Shared helpers for the reference tests: deterministic quant blocks (the
/// same generator as `Scripts/native/reference/ggml_dequant_digests.c`),
/// tensors in any tile layout, and small Metal plumbing.
enum RefTest {
    struct SplitMix64 {
        var state: UInt64
        init(_ seed: UInt64) { state = seed }
        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
        /// Uniform in [-1, 1).
        mutating func unit() -> Float { Float(Double(next() >> 11) / Double(1 << 53) * 2 - 1) }
        mutating func scale16() -> UInt16 {
            let r = next()
            return UInt16(((r & 1) << 15) | (0x1800 + ((r >> 1) % 0x1800)))
        }
    }

    /// Offsets of the F16 scale fields that must hold finite values.
    static func scaleFields(_ type: Native.GGMLType) -> [Int] {
        switch type {
        case .q8_0, .iq4_xs, .iq3_xxs, .iq3_s: return [0]
        case .q4_K, .q5_K: return [0, 2]
        case .q6_K: return [208]
        default: return []
        }
    }

    /// `blocks` random blocks of `type` in GGUF order (finite scales).
    static func randomBlocks(_ type: Native.GGMLType, blocks: Int, seed: UInt64) -> [UInt8] {
        let format = Native.QuantBlockFormat.of(type)!
        var rng = SplitMix64(seed)
        switch type {
        case .f32:
            return (0..<blocks).flatMap { _ in withUnsafeBytes(of: rng.unit().bitPattern.littleEndian) { Array($0) } }
        case .f16:
            return (0..<blocks).flatMap { _ in
                withUnsafeBytes(of: Float16(rng.unit()).bitPattern.littleEndian) { Array($0) }
            }
        case .bf16:
            return (0..<blocks).flatMap { _ in
                withUnsafeBytes(of: UInt16(rng.unit().bitPattern >> 16).littleEndian) { Array($0) }
            }
        default:
            var bytes = (0..<blocks * format.bytes).map { _ in UInt8(truncatingIfNeeded: rng.next()) }
            for b in 0..<blocks {
                for field in scaleFields(type) {
                    let v = rng.scale16()
                    bytes[b * format.bytes + field] = UInt8(v & 0xff)
                    bytes[b * format.bytes + field + 1] = UInt8(v >> 8)
                }
            }
            return bytes
        }
    }

    static func fnv(_ values: [Float]) -> UInt64 {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        for v in values {
            let bits = v.bitPattern
            for b in 0..<4 {
                h ^= UInt64((bits >> (8 * UInt32(b))) & 0xff)
                h = h &* 0x100_0000_01b3
            }
        }
        return h
    }

    static func buffer(_ device: MTLDevice, bytes: [UInt8], at offset: Int = 0, extra: Int = 0) -> MTLBuffer {
        let b = device.makeBuffer(length: max(16, offset + bytes.count + extra), options: .storageModeShared)!
        bytes.withUnsafeBytes { b.contents().advanced(by: offset).copyMemory(from: $0.baseAddress!, byteCount: bytes.count) }
        return b
    }

    static func buffer(_ device: MTLDevice, floats: [Float], at offset: Int = 0) -> MTLBuffer {
        let bytes = floats.withUnsafeBytes { Array($0) }
        return buffer(device, bytes: bytes, at: offset)
    }

    static func floats(_ t: Native.TensorRef) -> [Float] {
        var out: [Float] = []
        for r in 0..<t.rows {
            let p = t.buffer.contents().advanced(by: t.offset + r * t.rowStride).assumingMemoryBound(to: Float.self)
            out += Array(UnsafeBufferPointer(start: p, count: t.cols))
        }
        return out
    }

    static func tensor(_ device: MTLDevice, floats: [Float], rows: Int, cols: Int,
                       rowStride: Int? = nil, offset: Int = 0) -> Native.TensorRef {
        let stride = rowStride ?? cols * 4
        let b = device.makeBuffer(length: max(16, offset + rows * stride), options: .storageModeShared)!
        for r in 0..<rows {
            floats[r * cols ..< (r + 1) * cols].withUnsafeBytes {
                b.contents().advanced(by: offset + r * stride).copyMemory(from: $0.baseAddress!, byteCount: cols * 4)
            }
        }
        return Native.TensorRef(buffer: b, offset: offset, dtype: .f32, rows: rows, cols: cols, rowStride: stride)
    }

    /// A weight tensor holding `gguf` (GGUF-order bytes) packed in `layout`,
    /// placed at `offset` in a fresh buffer.
    static func weight(_ device: MTLDevice, name: String = "w", type: Native.GGMLType, rows: Int, cols: Int,
                       layout: Native.TileLayout = .gguf, gguf: [UInt8], offset: Int = 0) throws -> Native.QuantTensor {
        let format = Native.QuantBlockFormat.of(type)!
        var packed: [UInt8] = []
        try gguf.withUnsafeBytes { src in
            try Native.ReferencePackWriter.pack(source: src, format: format, rows: rows,
                                                blocksPerRow: cols / format.elements, layout: layout) {
                packed += Array($0)
            }
        }
        return Native.QuantTensor(name: name, type: type, shape: rows == 1 ? [cols] : [cols, rows], layout: layout,
                                  buffer: buffer(device, bytes: packed, at: offset), offset: offset,
                                  byteCount: packed.count, sourceSHA256: "")
    }

    static func randomWeight(_ device: MTLDevice, name: String = "w", type: Native.GGMLType, rows: Int, cols: Int,
                             layout: Native.TileLayout = .gguf, seed: UInt64) throws -> Native.QuantTensor {
        let format = Native.QuantBlockFormat.of(type)!
        let blocks = rows * cols / format.elements
        return try weight(device, name: name, type: type, rows: rows, cols: cols, layout: layout,
                          gguf: randomBlocks(type, blocks: blocks, seed: seed))
    }

    static func run(_ ctx: MetalContext, _ body: (MTLComputeCommandEncoder) throws -> Void) throws {
        let cb = ctx.queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        try body(enc)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { throw error }
    }

    static func maxRelativeError(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count)
        let scale = max(b.map(abs).max() ?? 0, 1e-6)
        return zip(a, b).map { abs($0 - $1) }.max().map { $0 / scale } ?? 0
    }
}

extension RefTest {
    /// A small `qwen35`-shaped config: 4 blocks (1 and 3 full attention), K-quant
    /// friendly widths, IMROPE sections that reduce to NEOX.
    static let tiny = Native.Qwen35Config(
        blockCount: 4, hiddenSize: 256, ffnSize: 512, headCount: 4, kvHeadCount: 2, keyLength: 32,
        valueLength: 32, rmsNormEpsilon: 1e-6, ropeFreqBase: 10_000, ropeDimensionCount: 16,
        ropeSections: [3, 3, 2, 0], fullAttentionInterval: 2, ssmConvKernel: 4, ssmStateSize: 32,
        ssmGroupCount: 2, ssmTimeStepRank: 4, ssmInnerSize: 128, contextLength: 4096, nextnPredictLayers: 0,
        vocabSize: 320, tiedEmbeddings: true)

    static func f32Weight(_ device: MTLDevice, _ values: [Float], name: String = "v") -> Native.QuantTensor {
        let bytes = values.withUnsafeBytes { Array($0) }
        return Native.QuantTensor(name: name, type: .f32, shape: [values.count], layout: .gguf,
                                  buffer: buffer(device, bytes: bytes, at: 64), offset: 64, byteCount: bytes.count,
                                  sourceSHA256: "")
    }
}
