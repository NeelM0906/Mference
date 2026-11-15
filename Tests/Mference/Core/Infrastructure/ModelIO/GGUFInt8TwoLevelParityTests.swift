import Foundation
import Testing
@testable import MferenceRepackCore
@testable import Mference

/// WS4 R4(a): the installer's INT8-2L expander (`GGUFInt8TwoLevelExpander`)
/// and the runtime's decoder (`GGUFBlockDecode`) are bit-exact twins, and the
/// INT8-2L weight equals ggml's own dequantization at llama.cpp b1b770d9.
@Suite struct GGUFInt8TwoLevelParityTests {
    /// Deterministic random blocks with a finite FP16 scale.
    private static func blocks(_ source: GGUFInt8TwoLevelExpander.Source, count: Int, seed: UInt64) -> [UInt8] {
        var state = seed
        func next() -> UInt8 {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return UInt8(truncatingIfNeeded: state >> 33)
        }
        var out: [UInt8] = []
        for _ in 0..<count {
            var b = (0..<source.blockBytes).map { _ in next() }
            // FP16 d: sign 0/1, exponent 1...29 (finite, normal), any mantissa.
            let d = UInt16(next() & 0x80) << 8 | UInt16(1 + next() % 29) << 10 | UInt16(next()) << 2
            let at = source == .q6_K ? 208 : 0
            b[at] = UInt8(truncatingIfNeeded: d); b[at + 1] = UInt8(truncatingIfNeeded: d >> 8)
            out += b
        }
        return out
    }

    @Test("Installer and runtime agree per block", arguments: [GGUFInt8TwoLevelExpander.Source.q6_K, .q8_0])
    func perBlockParity(source: GGUFInt8TwoLevelExpander.Source) {
        let raw = Self.blocks(source, count: 64, seed: source == .q6_K ? 0x51 : 0x52)
        for i in 0..<64 {
            raw.withUnsafeBytes { all in
                let b = UnsafeRawBufferPointer(rebasing: all[(i * source.blockBytes)..<((i + 1) * source.blockBytes)])
                let installer = GGUFInt8TwoLevelExpander.expandBlock(b, source: source)
                let runtime = source == .q6_K ? GGUFBlockDecode.q6KToInt8TwoLevel(b)
                                              : GGUFBlockDecode.q8_0ToInt8TwoLevel(b)
                #expect(installer.values == runtime.values, "block \(i)")
                #expect(installer.d == runtime.d && installer.m == runtime.m, "block \(i)")
            }
        }
    }

    @Test("Streaming in-place expansion equals the runtime decode",
          arguments: [GGUFInt8TwoLevelExpander.Source.q6_K, .q8_0])
    func inPlaceParity(source: GGUFInt8TwoLevelExpander.Source) throws {
        let n = 37
        let raw = Self.blocks(source, count: n, seed: 0xA11CE)
        var expectValues: [UInt8] = [], expectD: [UInt8] = [], expectM: [UInt8] = []
        raw.withUnsafeBytes { all in
            for i in 0..<n {
                let b = UnsafeRawBufferPointer(rebasing: all[(i * source.blockBytes)..<((i + 1) * source.blockBytes)])
                let t = source == .q6_K ? GGUFBlockDecode.q6KToInt8TwoLevel(b) : GGUFBlockDecode.q8_0ToInt8TwoLevel(b)
                expectValues += t.values.map { UInt8(bitPattern: $0) }
                expectD += [UInt8(truncatingIfNeeded: t.d), UInt8(truncatingIfNeeded: t.d >> 8)]
                expectM += t.m.map { UInt8(bitPattern: $0) }
            }
        }
        for (component, expected) in [(GGUFInt8TwoLevelExpander.Component.values, expectValues),
                                      (.d, expectD), (.m, expectM)] {
            let capacity = n * max(source.blockBytes, 256)
            let buf = UnsafeMutableRawBufferPointer.allocate(byteCount: capacity, alignment: 16)
            defer { buf.deallocate() }
            raw.withUnsafeBytes { buf.copyMemory(from: $0) }
            let transform = RangeCopyTransform.ggufInt8TwoLevel(source: source, component: component)
            #expect(try transform.destinationByteCount(for: UInt64(raw.count)) == UInt64(expected.count))
            let out = try GGUFInt8TwoLevelExpander.expandInPlace(buffer: buf, sourceCount: raw.count, source: source, component: component)
            #expect(out == expected.count)
            #expect(Array(buf.prefix(out)) == expected, "\(source) \(component)")
        }
    }

    /// Line-by-line transliteration of ggml's `dequantize_row_q6_K` (b1b770d9,
    /// ggml-quants.c:1949), used only as the oracle.
    private static func ggmlDequantizeQ6K(_ x: [UInt8]) -> [Float] {
        var y: [Float] = []
        for i in 0..<(x.count / 210) {
            let base = i * 210
            let d = Float(Float16(bitPattern: UInt16(x[base + 208]) | UInt16(x[base + 209]) << 8))
            var ql = base, qh = base + 128, sc = base + 192
            for _ in stride(from: 0, to: 256, by: 128) {
                var half = [Float](repeating: 0, count: 128)
                for l in 0..<32 {
                    let isx = l / 16
                    let q1 = Int8(Int(x[ql + l] & 0xF) | Int((x[qh + l] >> 0) & 3) << 4) - 32
                    let q2 = Int8(Int(x[ql + l + 32] & 0xF) | Int((x[qh + l] >> 2) & 3) << 4) - 32
                    let q3 = Int8(Int(x[ql + l] >> 4) | Int((x[qh + l] >> 4) & 3) << 4) - 32
                    let q4 = Int8(Int(x[ql + l + 32] >> 4) | Int((x[qh + l] >> 6) & 3) << 4) - 32
                    func s(_ k: Int) -> Float { Float(Int8(bitPattern: x[sc + k])) }
                    half[l] = d * s(isx + 0) * Float(q1)
                    half[l + 32] = d * s(isx + 2) * Float(q2)
                    half[l + 64] = d * s(isx + 4) * Float(q3)
                    half[l + 96] = d * s(isx + 6) * Float(q4)
                }
                y += half
                ql += 64; qh += 32; sc += 8
            }
        }
        return y
    }

    @Test func int8TwoLevelWeightsEqualGgmlDequantization() {
        let raw = Self.blocks(.q6_K, count: 40, seed: 0xD00D)
        let oracle = Self.ggmlDequantizeQ6K(raw)
        let runtime = raw.withUnsafeBytes { GGUFBlockDecode.dequantizeQ6K($0) }
        #expect(runtime.map(\.bitPattern) == oracle.map(\.bitPattern))

        let q8 = Self.blocks(.q8_0, count: 40, seed: 0xBEEF)
        var q8Oracle: [Float] = []
        for i in 0..<40 {   // ggml dequantize_row_q8_0: y = qs[j] * d
            let d = Float(Float16(bitPattern: UInt16(q8[i * 34]) | UInt16(q8[i * 34 + 1]) << 8))
            for j in 0..<32 { q8Oracle.append(Float(Int8(bitPattern: q8[i * 34 + 2 + j])) * d) }
        }
        #expect(q8.withUnsafeBytes { GGUFBlockDecode.dequantizeQ8_0($0) }.map(\.bitPattern) == q8Oracle.map(\.bitPattern))
    }

    @Test func handBuiltQ6KBlockDecodesToKnownValues() {
        var b = [UInt8](repeating: 0, count: 210)
        b[0] = 0x21          // ql[0]: low 1 (element 0), high 2 (element 64)
        b[128] = 0b1110_0111 // qh[0]: element 0 hi 3, 32 hi 1, 64 hi 2, 96 hi 3
        b[192] = UInt8(bitPattern: -3)   // scales[0] covers elements 0..<16
        b[196] = 5                       // scales[4] covers elements 64..<80
        b[208] = 0x00; b[209] = 0x3C     // d = 1.0
        let t = b.withUnsafeBytes { GGUFBlockDecode.q6KToInt8TwoLevel($0) }
        #expect(t.values[0] == Int8((1 | 3 << 4) - 32))     // 49 - 32 = 17
        #expect(t.values[64] == Int8((2 | 2 << 4) - 32))    // 34 - 32 = 2
        #expect(t.values[32] == Int8((0 | 1 << 4) - 32))    // -16
        #expect(t.m[0] == -3 && t.m[4] == 5)
        #expect(GGUFBlockDecode.int8TwoLevelWeight(d: t.d, m: t.m[0], value: t.values[0]) == -51)
    }
}
