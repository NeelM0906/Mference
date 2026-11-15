import Foundation

/// Runtime decoders for the GGUF Q6_K and Q8_0 blocks of a WS4 install, and
/// the INT8-2L resident format the installer expands them into (U1 §3b).
///
/// This is the runtime twin of the installer's `GGUFInt8TwoLevelExpander`
/// (which must stay independent of this module). It is written per element
/// rather than with ggml's loop structure, so the parity test between the two
/// is a cross-check, not a copy compared with itself. Block layouts are those
/// of `ggml/src/ggml-common.h` at llama.cpp b1b770d9:
/// - `block_q8_0`: `d` (fp16), `qs[32]` (int8), 34 bytes;
/// - `block_q6_K`: `ql[128]`, `qh[64]`, `scales[16]` (int8), `d` (fp16), 210 bytes.
public enum GGUFBlockDecode {
    public static let q6KBlockBytes = 210
    public static let q8_0BlockBytes = 34

    /// One INT8-2L block: int8 values, the FP16 block scale bits and, for
    /// Q6_K, the int8 scale per 16 values (empty for Q8_0, where m ≡ 1).
    public struct Int8TwoLevelBlock: Equatable, Sendable {
        public let values: [Int8]
        public let d: UInt16
        public let m: [Int8]
    }

    /// The 6-bit Q6_K code of element `e` (0..<256), already offset by −32.
    @inline(__always)
    static func q6KValue(_ b: UnsafeRawBufferPointer, _ e: Int) -> Int8 {
        let half = e / 128, r = e % 128, k = r / 32, l = r % 32
        let qlByte = b[half * 64 + l + (k & 1) * 32]
        let low4 = k < 2 ? qlByte & 0x0F : qlByte >> 4
        let high2 = (b[128 + half * 32 + l] >> UInt8(2 * k)) & 0x3
        return Int8(Int(low4) | Int(high2) << 4) - 32
    }

    public static func q6KToInt8TwoLevel(_ block: UnsafeRawBufferPointer) -> Int8TwoLevelBlock {
        precondition(block.count == q6KBlockBytes)
        let values = (0..<256).map { q6KValue(block, $0) }
        let m = (0..<16).map { Int8(bitPattern: block[192 + $0]) }
        return Int8TwoLevelBlock(values: values, d: UInt16(block[208]) | UInt16(block[209]) << 8, m: m)
    }

    public static func q8_0ToInt8TwoLevel(_ block: UnsafeRawBufferPointer) -> Int8TwoLevelBlock {
        precondition(block.count == q8_0BlockBytes)
        return Int8TwoLevelBlock(values: (0..<32).map { Int8(bitPattern: block[2 + $0]) },
                                 d: UInt16(block[0]) | UInt16(block[1]) << 8, m: [])
    }

    @inline(__always)
    public static func fp16ToFloat(_ bits: UInt16) -> Float { Float(Float16(bitPattern: bits)) }

    /// An INT8-2L weight: `(fp16(d) · m) · value`, in ggml's evaluation order.
    /// Exact in fp32 (11 + 7 + 6 significant bits).
    @inline(__always)
    public static func int8TwoLevelWeight(d: UInt16, m: Int8, value: Int8) -> Float {
        fp16ToFloat(d) * Float(m) * Float(value)
    }

    /// Dequantizes whole Q6_K blocks (a `token_embd` row, for example).
    public static func dequantizeQ6K(_ blocks: UnsafeRawBufferPointer) -> [Float] {
        precondition(blocks.count % q6KBlockBytes == 0)
        var out: [Float] = []
        out.reserveCapacity(blocks.count / q6KBlockBytes * 256)
        for i in 0..<(blocks.count / q6KBlockBytes) {
            let b = UnsafeRawBufferPointer(rebasing: blocks[(i * q6KBlockBytes)..<((i + 1) * q6KBlockBytes)])
            let t = q6KToInt8TwoLevel(b)
            for e in 0..<256 { out.append(int8TwoLevelWeight(d: t.d, m: t.m[e / 16], value: t.values[e])) }
        }
        return out
    }

    /// Dequantizes whole Q8_0 blocks.
    public static func dequantizeQ8_0(_ blocks: UnsafeRawBufferPointer) -> [Float] {
        precondition(blocks.count % q8_0BlockBytes == 0)
        var out: [Float] = []
        out.reserveCapacity(blocks.count / q8_0BlockBytes * 32)
        for i in 0..<(blocks.count / q8_0BlockBytes) {
            let b = UnsafeRawBufferPointer(rebasing: blocks[(i * q8_0BlockBytes)..<((i + 1) * q8_0BlockBytes)])
            let t = q8_0ToInt8TwoLevel(b)
            for e in 0..<32 { out.append(fp16ToFloat(t.d) * Float(t.values[e])) }
        }
        return out
    }
}
