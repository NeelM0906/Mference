import Foundation

/// Install-time expansion of GGUF Q6_K and Q8_0 blocks into the INT8-2L
/// resident format (WS4 U1 §3b, ruling R4(a)). INT8-2L stores, per tensor,
/// three dense regions in block order:
///
/// - `values`: one int8 per weight, in natural element order;
/// - `d`: the block's FP16 scale bits (per 256 for Q6_K, per 32 for Q8_0);
/// - `m`: Q6_K only, the int8 sub-block scale per 16 weights.
///
/// A weight is `Float(fp16(d)) * Float(m) * Float(value)` (m ≡ 1 for Q8_0),
/// which is exactly ggml's `dequantize_row_q6_K` / `dequantize_row_q8_0` at
/// llama.cpp b1b770d9: every product fits in fp32's 24-bit significand. The
/// expansion is block-local and row-agnostic, so it streams like the other
/// range transforms. The runtime twin is `GGUFBlockDecode` in `Mference`; a
/// parity test locks the two bit for bit.
public enum GGUFInt8TwoLevelExpander {
    public enum Source: String, Sendable, Equatable {
        case q6_K = "q6_K"
        case q8_0 = "q8_0"

        public var blockBytes: Int { self == .q6_K ? 210 : 34 }
        public var blockValues: Int { self == .q6_K ? 256 : 32 }
    }

    public enum Component: String, Sendable, Equatable {
        case values, d, m
    }

    public static func destinationBytesPerBlock(_ source: Source, _ component: Component) -> Int {
        switch (source, component) {
        case (_, .values): source.blockValues
        case (_, .d): 2
        case (.q6_K, .m): 16
        case (.q8_0, .m): 0
        }
    }

    /// One block's INT8-2L parts. `block` is exactly `source.blockBytes` long.
    public static func expandBlock(_ block: UnsafeRawBufferPointer, source: Source)
        -> (values: [Int8], d: UInt16, m: [Int8]) {
        precondition(block.count == source.blockBytes)
        let b = block.bindMemory(to: UInt8.self)
        switch source {
        case .q8_0:
            // block_q8_0: d (fp16), qs[32] (int8).
            let d = UInt16(b[0]) | UInt16(b[1]) << 8
            return ((0..<32).map { Int8(bitPattern: b[2 + $0]) }, d, [])
        case .q6_K:
            // block_q6_K: ql[128], qh[64], scales[16] (int8), d (fp16).
            let ql = 0, qh = 128, sc = 192
            let d = UInt16(b[208]) | UInt16(b[209]) << 8
            var values = [Int8](repeating: 0, count: 256)
            for half in 0..<2 {
                let qlh = ql + half * 64, qhh = qh + half * 32, y = half * 128
                for l in 0..<32 {
                    let h = b[qhh + l]
                    values[y + l]      = Int8(Int(b[qlh + l] & 0xF) | Int((h >> 0) & 3) << 4) - 32
                    values[y + l + 32] = Int8(Int(b[qlh + l + 32] & 0xF) | Int((h >> 2) & 3) << 4) - 32
                    values[y + l + 64] = Int8(Int(b[qlh + l] >> 4) | Int((h >> 4) & 3) << 4) - 32
                    values[y + l + 96] = Int8(Int(b[qlh + l + 32] >> 4) | Int((h >> 6) & 3) << 4) - 32
                }
            }
            // Element e uses scales[e / 16]: the half/is/2k indexing of
            // dequantize_row_q6_K reduces to that.
            return (values, d, (0..<16).map { Int8(bitPattern: b[sc + $0]) })
        }
    }

    /// Rewrites `sourceCount` bytes of whole blocks in `buffer` into one INT8-2L
    /// component and returns the output byte count. Growing components run
    /// back to front and shrinking ones front to back, so no unread block is
    /// overwritten.
    public static func expandInPlace(buffer: UnsafeMutableRawBufferPointer, sourceCount: Int,
                              source: Source, component: Component) throws -> Int {
        let inUnit = source.blockBytes
        let outUnit = destinationBytesPerBlock(source, component)
        guard sourceCount % inUnit == 0 else {
            throw RepackError.configurationInvalid(detail: "INT8-2L input is not whole \(source.rawValue) blocks")
        }
        let blocks = sourceCount / inUnit
        let outCount = blocks * outUnit
        guard outCount <= buffer.count else {
            throw RepackError.scratchExceeded(requested: outCount, limit: buffer.count)
        }
        let base = buffer.baseAddress!
        var temp = [UInt8](repeating: 0, count: inUnit)
        func one(_ i: Int) {
            temp.withUnsafeMutableBytes { t in
                t.copyMemory(from: UnsafeRawBufferPointer(start: base + i * inUnit, count: inUnit))
            }
            let parts = temp.withUnsafeBytes { expandBlock($0, source: source) }
            let dst = base + i * outUnit
            switch component {
            case .values: parts.values.withUnsafeBytes { dst.copyMemory(from: $0.baseAddress!, byteCount: outUnit) }
            case .d:
                dst.storeBytes(of: UInt8(truncatingIfNeeded: parts.d), as: UInt8.self)
                (dst + 1).storeBytes(of: UInt8(truncatingIfNeeded: parts.d >> 8), as: UInt8.self)
            case .m: parts.m.withUnsafeBytes { if outUnit > 0 { dst.copyMemory(from: $0.baseAddress!, byteCount: outUnit) } }
            }
        }
        if outUnit > inUnit {
            for i in stride(from: blocks - 1, through: 0, by: -1) { one(i) }
        } else {
            for i in 0..<blocks { one(i) }
        }
        return outCount
    }
}
