import Foundation

/// CPU references for the two expert weight encodings of the Inkling 2 Mini
/// install (the Unsloth UD-Q4_K_XL file, copied byte for byte): NVFP4 for the
/// experts of layers 0-19 and Q8_0 for layers 20-23. Decoding is byte-exact
/// with ggml at llama.cpp b12635 (`dequantize_row_nvfp4`,
/// `dequantize_row_q8_0`, `ggml_ue4m3_to_fp32` in `ggml-quants.c` and
/// `ggml-impl.h`); the encoders mirror `quantize_row_nvfp4_ref` and
/// `quantize_row_q8_0_ref` and exist only to build synthetic fixtures.
///
/// Rows are stored the way ggml stores a 2-D slice: `k / blockWeights`
/// consecutive blocks, no padding between rows.
public enum Inkling2NVFP4 {
    /// Weights per block (`QK_NVFP4`).
    public static let blockWeights = 64
    /// Weights per block scale (`QK_NVFP4_SUB`).
    public static let subBlockWeights = 16
    /// `block_nvfp4`: four UE4M3 scale bytes, then 32 code bytes.
    public static let blockBytes = 36
    public static let scaleBytesPerBlock = 4

    /// ggml `kvalues_mxfp4` (shared with MXFP4): the E2M1 values doubled.
    /// Codes 8-15 are the negatives of 0-7 (code 8 is a negative zero, decoded 0).
    public static let doubledE2M1: [Int8] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12]

    /// The plain E2M1 magnitude of a code, `doubledE2M1 / 2`.
    public static func e2m1(_ code: UInt8) -> Float {
        Float(doubledE2M1[Int(code & 0xF)]) * 0.5
    }

    public static func rowBytes(k: Int) -> Int {
        precondition(k % blockWeights == 0, "NVFP4 row of \(k) weights is not whole blocks")
        return k / blockWeights * blockBytes
    }

    /// `ggml_ue4m3_to_fp32`: half of the unsigned E4M3 value, so that
    /// `doubledE2M1[code] * ue4m3Half(scale)` is the weight. The sign bit is
    /// ignored and 0x00 / 0x7F decode to 0, as ggml does.
    public static func ue4m3Half(_ x: UInt8) -> Float {
        if x == 0 || x == 0x7F { return 0 }
        let exponent = Int((x >> 3) & 0xF)
        let mantissa = Float(x & 0x7)
        let raw: Float = exponent == 0
            ? Float(sign: .plus, exponent: -9, significand: mantissa)
            : Float(sign: .plus, exponent: exponent - 7, significand: 1 + mantissa / 8)
        return raw * 0.5
    }

    /// The scale itself (twice `ue4m3Half`).
    public static func ue4m3(_ x: UInt8) -> Float { ue4m3Half(x) * 2 }

    /// `ggml_fp32_to_ue4m3`, including its truncating corner cases (no
    /// ties-to-even, saturation at 0x7E).
    public static func ue4m3Encode(_ value: Float) -> UInt8 {
        guard value > 0 else { return 0 }
        let x = min(value, 448)
        let bits = x.bitPattern
        let fp32Exponent = Int((bits >> 23) & 0xFF) - 127
        let fp32Mantissa = Int((bits >> 20) & 0x7)
        var exponent = fp32Exponent + 7
        if exponent <= 0 {
            let mantissa = min(Int(x * 512 + 0.5), 7)
            return mantissa < 1 ? 0 : UInt8(mantissa)
        }
        if exponent >= 15 { return 0x7E }
        var mantissa = fp32Mantissa + Int((bits >> 19) & 1)
        if mantissa > 7 {
            mantissa = 0
            exponent += 1
            if exponent >= 15 { return 0x7E }
        }
        return UInt8((exponent << 3) | mantissa)
    }

    /// Weight `j` (0..<64) of a block: sub-block `j / 16`; inside it byte
    /// `j % 8` holds element `j % 8` in its low nibble and element `j % 8 + 8`
    /// in its high nibble.
    @inline(__always)
    public static func code(block: UnsafeRawBufferPointer, index j: Int) -> UInt8 {
        let sub = j / subBlockWeights, inSub = j % subBlockWeights
        let byte = block[scaleBytesPerBlock + sub * (subBlockWeights / 2) + inSub % 8]
        return inSub < 8 ? byte & 0xF : byte >> 4
    }

    /// `dequantize_row_nvfp4` over `k` weights starting at `bytes`.
    public static func dequantize(_ bytes: UnsafeRawBufferPointer, k: Int) -> [Float] {
        precondition(bytes.count >= rowBytes(k: k))
        var out = [Float](repeating: 0, count: k)
        for b in 0..<(k / blockWeights) {
            let block = UnsafeRawBufferPointer(rebasing: bytes[(b * blockBytes)..<((b + 1) * blockBytes)])
            for s in 0..<(blockWeights / subBlockWeights) {
                let d = ue4m3Half(block[s])
                for j in 0..<(subBlockWeights / 2) {
                    let q = block[scaleBytesPerBlock + s * (subBlockWeights / 2) + j]
                    let base = b * blockWeights + s * subBlockWeights
                    out[base + j] = Float(doubledE2M1[Int(q & 0xF)]) * d
                    out[base + j + subBlockWeights / 2] = Float(doubledE2M1[Int(q >> 4)]) * d
                }
            }
        }
        return out
    }

    public static func dequantize(_ bytes: [UInt8], k: Int) -> [Float] {
        bytes.withUnsafeBytes { dequantize($0, k: k) }
    }

    /// `best_index_mxfp4`: the code whose doubled value times `d` is nearest
    /// to `x`; the first such code wins a tie.
    static func bestIndex(_ x: Float, _ d: Float) -> UInt8 {
        var best = 0
        var bestError = abs(Float(doubledE2M1[0]) * d - x)
        for i in 1..<16 {
            let error = abs(Float(doubledE2M1[i]) * d - x)
            if error < bestError {
                best = i
                bestError = error
            }
        }
        return UInt8(best)
    }

    /// `quantize_row_nvfp4_ref`. Fixture use only.
    public static func quantize(_ row: [Float]) -> [UInt8] {
        let k = row.count
        var out = [UInt8](repeating: 0, count: rowBytes(k: k))
        for b in 0..<(k / blockWeights) {
            for s in 0..<(blockWeights / subBlockWeights) {
                let base = b * blockWeights + s * subBlockWeights
                var amax: Float = 0
                for j in 0..<subBlockWeights where amax < abs(row[base + j]) { amax = abs(row[base + j]) }
                let ue = ue4m3Encode(amax / 6)
                out[b * blockBytes + s] = ue
                let d = ue4m3Half(ue)
                for j in 0..<(subBlockWeights / 2) {
                    let lo = bestIndex(row[base + j], d)
                    let hi = bestIndex(row[base + subBlockWeights / 2 + j], d)
                    out[b * blockBytes + scaleBytesPerBlock + s * (subBlockWeights / 2) + j] = lo | (hi << 4)
                }
            }
        }
        return out
    }
}

/// ggml `block_q8_0`: an FP16 scale, then 32 signed bytes; weight = `q * d`.
public enum Inkling2Q8_0 {
    public static let blockWeights = 32
    public static let blockBytes = 34

    public static func rowBytes(k: Int) -> Int {
        precondition(k % blockWeights == 0, "Q8_0 row of \(k) weights is not whole blocks")
        return k / blockWeights * blockBytes
    }

    /// `dequantize_row_q8_0`.
    public static func dequantize(_ bytes: UnsafeRawBufferPointer, k: Int) -> [Float] {
        precondition(bytes.count >= rowBytes(k: k))
        var out = [Float](repeating: 0, count: k)
        for b in 0..<(k / blockWeights) {
            let base = b * blockBytes
            let bits = UInt16(bytes[base]) | (UInt16(bytes[base + 1]) << 8)
            let d = Float(Float16(bitPattern: bits))
            for j in 0..<blockWeights {
                out[b * blockWeights + j] = Float(Int8(bitPattern: bytes[base + 2 + j])) * d
            }
        }
        return out
    }

    public static func dequantize(_ bytes: [UInt8], k: Int) -> [Float] {
        bytes.withUnsafeBytes { dequantize($0, k: k) }
    }

    /// `quantize_row_q8_0_ref`: `d = amax / 127` stored as FP16, `q = round(x / d)`
    /// with the unrounded `d` (ggml divides by the FP32 scale, not the stored one).
    public static func quantize(_ row: [Float]) -> [UInt8] {
        let k = row.count
        var out = [UInt8](repeating: 0, count: rowBytes(k: k))
        for b in 0..<(k / blockWeights) {
            var amax: Float = 0
            for j in 0..<blockWeights { amax = max(amax, abs(row[b * blockWeights + j])) }
            let d = amax / 127
            let id: Float = d != 0 ? 1 / d : 0
            let bits = Float16(d).bitPattern
            out[b * blockBytes] = UInt8(bits & 0xFF)
            out[b * blockBytes + 1] = UInt8(bits >> 8)
            for j in 0..<blockWeights {
                let q = Int8(roundf(row[b * blockWeights + j] * id))
                out[b * blockBytes + 2 + j] = UInt8(bitPattern: q)
            }
        }
        return out
    }

    /// The input-side Q8_0 round trip ggml's CPU backend applies to the
    /// activations of an NVFP4 or Q8_0 product (`vec_dot_type = Q8_0`):
    /// quantize, then read back with the stored FP16 scale.
    public static func roundTrip(_ row: [Float]) -> [Float] {
        dequantize(quantize(row), k: row.count)
    }
}

/// Encoding of one layer's routed experts in the install.
public enum Inkling2ExpertEncoding: Sendable, Equatable {
    case nvfp4
    case q8_0

    public func rowBytes(k: Int) -> Int {
        switch self {
        case .nvfp4: return Inkling2NVFP4.rowBytes(k: k)
        case .q8_0: return Inkling2Q8_0.rowBytes(k: k)
        }
    }

    public func dequantize(_ bytes: UnsafeRawBufferPointer, k: Int) -> [Float] {
        switch self {
        case .nvfp4: return Inkling2NVFP4.dequantize(bytes, k: k)
        case .q8_0: return Inkling2Q8_0.dequantize(bytes, k: k)
        }
    }

    public func quantize(_ row: [Float]) -> [UInt8] {
        switch self {
        case .nvfp4: return Inkling2NVFP4.quantize(row)
        case .q8_0: return Inkling2Q8_0.quantize(row)
        }
    }
}
