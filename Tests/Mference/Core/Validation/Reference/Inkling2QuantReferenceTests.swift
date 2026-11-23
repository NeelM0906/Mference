import Testing
import Foundation
import MferenceValidationSupport

/// The NVFP4 and Q8_0 references against hand-built blocks and an
/// independent decoding of the formats, so the references are not checked
/// only against themselves.
@Suite struct Inkling2QuantReferenceTests {
    /// UE4M3 from the format definition (bias 7, 3 mantissa bits, subnormal
    /// at exponent 0), written without ggml's code shape.
    private static func ue4m3Definition(_ x: UInt8) -> Float {
        let e = Int((x >> 3) & 0xF), m = Int(x & 7)
        if x == 0 || x == 0x7F { return 0 }
        if e == 0 { return Float(m) / 8 * pow(2, -6) }
        return (1 + Float(m) / 8) * pow(2, Float(e - 7))
    }

    @Test func ue4m3MatchesTheFormatForEveryByte() {
        for x in 0...255 {
            let byte = UInt8(x)
            #expect(Inkling2NVFP4.ue4m3(byte) == Self.ue4m3Definition(byte), "byte \(x)")
            #expect(Inkling2NVFP4.ue4m3Half(byte) * 2 == Inkling2NVFP4.ue4m3(byte))
        }
        #expect(Inkling2NVFP4.ue4m3(0x7E) == 448)
        #expect(Inkling2NVFP4.ue4m3(0x38) == 1)
        #expect(Inkling2NVFP4.ue4m3(0x01) == pow(2, -9))
    }

    @Test func ue4m3EncodeRoundTripsEveryFiniteScale() {
        for x in 1...0x77 {
            let byte = UInt8(x)
            #expect(Inkling2NVFP4.ue4m3Encode(Inkling2NVFP4.ue4m3(byte)) == byte, "byte \(x)")
        }
        // ggml's encoder saturates every value with exponent field 15 to 0x7E.
        for x in 0x78...0x7E {
            #expect(Inkling2NVFP4.ue4m3Encode(Inkling2NVFP4.ue4m3(UInt8(x))) == 0x7E)
        }
        #expect(Inkling2NVFP4.ue4m3Encode(0) == 0)
        #expect(Inkling2NVFP4.ue4m3Encode(-1) == 0)
        #expect(Inkling2NVFP4.ue4m3Encode(1e6) == 0x7E)
    }

    @Test func e2m1TableIsTheFormat() {
        let magnitudes: [Float] = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
        for code in 0..<16 {
            let expected = code < 8 ? magnitudes[code] : -magnitudes[code - 8]
            #expect(Inkling2NVFP4.e2m1(UInt8(code)) == expected)
        }
    }

    /// One block with distinct codes in every position and four scales:
    /// checks the nibble order (low nibble element j, high nibble j + 8 of
    /// each 16-value sub-block) and the per-16 scale.
    @Test func nvfp4BlockLayout() {
        var block = [UInt8](repeating: 0, count: 36)
        let scales: [UInt8] = [0x38, 0x40, 0x30, 0x48]  // 1, 2, 0.5, 4
        for s in 0..<4 { block[s] = scales[s] }
        var codes = [UInt8](repeating: 0, count: 64)
        for j in 0..<64 { codes[j] = UInt8((j * 7 + 3) % 16) }
        for s in 0..<4 {
            for j in 0..<8 {
                block[4 + s * 8 + j] = codes[s * 16 + j] | (codes[s * 16 + j + 8] << 4)
            }
        }
        let decoded = Inkling2NVFP4.dequantize(block, k: 64)
        for j in 0..<64 {
            let expected = Inkling2NVFP4.e2m1(codes[j]) * Inkling2NVFP4.ue4m3(scales[j / 16])
            #expect(decoded[j] == expected, "weight \(j)")
            #expect(block.withUnsafeBytes { Inkling2NVFP4.code(block: $0, index: j) } == codes[j])
        }
    }

    @Test(arguments: [UInt64(1), 2, 3])
    func nvfp4QuantizeRoundTripIsBounded(seed: UInt64) {
        var rng = SeedTree(seed).key("nvfp4-roundtrip")
        let row = (0..<1152).map { _ in rng.uniform(-3, 3) }
        let bytes = Inkling2NVFP4.quantize(row)
        #expect(bytes.count == 1152 / 64 * 36)
        let decoded = Inkling2NVFP4.dequantize(bytes, k: 1152)
        for b in 0..<(1152 / 16) {
            let block = row[(b * 16)..<((b + 1) * 16)]
            let amax = block.map(abs).max()!
            // The scale is amax / 6 within E4M3 rounding (< 1/16 relative);
            // the widest E2M1 gap is 2 (4 to 6), so the error is at most one
            // scale step: amax / 6 * (1 + 1/16).
            let bound = amax / 6 * (1 + 1.0 / 16) + 1e-6
            for i in (b * 16)..<((b + 1) * 16) {
                #expect(abs(decoded[i] - row[i]) <= bound, "element \(i)")
            }
        }
    }

    @Test func q8_0BlockLayout() {
        var block = [UInt8](repeating: 0, count: 34)
        let d = Float16(0.03125)
        block[0] = UInt8(d.bitPattern & 0xFF)
        block[1] = UInt8(d.bitPattern >> 8)
        for j in 0..<32 { block[2 + j] = UInt8(bitPattern: Int8(j * 8 - 128)) }
        let decoded = Inkling2Q8_0.dequantize(block, k: 32)
        for j in 0..<32 { #expect(decoded[j] == Float(j * 8 - 128) * 0.03125) }
    }

    @Test(arguments: [UInt64(4), 5])
    func q8_0QuantizeRoundTripIsBounded(seed: UInt64) {
        var rng = SeedTree(seed).key("q8-roundtrip")
        let row = (0..<2304).map { _ in rng.uniform(-2, 2) }
        let decoded = Inkling2Q8_0.dequantize(Inkling2Q8_0.quantize(row), k: 2304)
        for b in 0..<(2304 / 32) {
            let amax = row[(b * 32)..<((b + 1) * 32)].map(abs).max()!
            // Half a step of amax / 127, plus the FP16 rounding of the scale
            // times the largest code.
            let bound = amax / 127 * 0.5 + amax * 1.0 / 1024 + 1e-6
            for i in (b * 32)..<((b + 1) * 32) {
                #expect(abs(decoded[i] - row[i]) <= bound, "element \(i)")
            }
        }
    }

    @Test func rowBytesMatchTheInstallFile() {
        // The tensor offsets in the UD-Q4_K_XL header: an NVFP4 gate-up
        // expert is 2,985,984 bytes, a down expert 1,492,992; Q8_0 5,640,192 and 2,820,096.
        #expect(2304 * Inkling2NVFP4.rowBytes(k: 2304) == 2_985_984)
        #expect(2304 * Inkling2NVFP4.rowBytes(k: 1152) == 1_492_992)
        #expect(2304 * Inkling2Q8_0.rowBytes(k: 2304) == 5_640_192)
        #expect(2304 * Inkling2Q8_0.rowBytes(k: 1152) == 2_820_096)
    }
}
