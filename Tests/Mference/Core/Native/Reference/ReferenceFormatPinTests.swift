import Foundation
import Testing
@testable import Mference

/// Pins the native contract's ggml definitions and the reference lookup tables
/// to the vendored llama.cpp headers (read as text; no model, no GPU).
@Suite("ReferenceFormatPinTests")
struct ReferenceFormatPinTests {
    static let root = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

    static func header(_ path: String) throws -> String {
        try String(contentsOf: root.appendingPathComponent(path), encoding: .utf8)
    }

    @Test func ggmlTypeRawValuesMatchGGMLHeader() throws {
        let text = try Self.header("Sources/LlamaCpp/include/ggml.h")
        let regex = try NSRegularExpression(pattern: #"^\s*GGML_TYPE_([A-Z0-9_]+)\s*=\s*(\d+),"#,
                                            options: .anchorsMatchLines)
        var values: [String: UInt32] = [:]
        for m in regex.matches(in: text, range: NSRange(text.startIndex..., in: text)) {
            let name = String(text[Range(m.range(at: 1), in: text)!]).lowercased()
            values[name] = UInt32(text[Range(m.range(at: 2), in: text)!])
        }
        #expect(values.count > 30)
        for type in Native.GGMLType.allCases {
            let name = "\(type)".lowercased()
            #expect(values[name] == type.rawValue, "GGML_TYPE_\(name.uppercased())")
        }
    }

    /// Evaluates the right-hand side of ggml-common.h's
    /// `static_assert(sizeof(block_x) == <expr>, ...)`.
    static func evaluate(_ expr: String) throws -> Int {
        var s = expr
        for (k, v) in [("sizeof(ggml_half)", "2"), ("sizeof(uint16_t)", "2"), ("sizeof(uint8_t)", "1"),
                       ("IQ3S_N_SCALE", "(QK_K/64)"), ("K_SCALE_SIZE", "12"), ("QK8_0", "32"), ("QK_K", "256")] {
            s = s.replacingOccurrences(of: k, with: v)
        }
        var chars = Array(s.filter { !$0.isWhitespace })
        var i = 0
        func number() throws -> Int {
            if chars[i] == "(" {
                i += 1
                let v = try sum()
                i += 1
                return v
            }
            var n = 0
            guard i < chars.count, chars[i].isNumber else { throw Native.NativeError.invalidShape(String(chars)) }
            while i < chars.count, let d = chars[i].wholeNumberValue { n = n * 10 + d; i += 1 }
            return n
        }
        func product() throws -> Int {
            var v = try number()
            while i < chars.count, chars[i] == "*" || chars[i] == "/" {
                let op = chars[i]
                i += 1
                let r = try number()
                v = op == "*" ? v * r : v / r
            }
            return v
        }
        func sum() throws -> Int {
            var v = try product()
            while i < chars.count, chars[i] == "+" {
                i += 1
                v += try product()
            }
            return v
        }
        defer { chars = [] }
        return try sum()
    }

    @Test func blockFormatSizesMatchGGMLCommon() throws {
        let text = try Self.header("Sources/LlamaCpp/ggml/src/ggml-common.h")
        let regex = try NSRegularExpression(pattern: #"static_assert\(sizeof\(block_(\w+)\) == ([^,]+),"#)
        var sizes: [String: Int] = [:]
        for m in regex.matches(in: text, range: NSRange(text.startIndex..., in: text)) {
            let name = String(text[Range(m.range(at: 1), in: text)!])
            sizes[name] = try? Self.evaluate(String(text[Range(m.range(at: 2), in: text)!]))
        }
        let checked: [Native.GGMLType] = [.q8_0, .q2_K, .q4_K, .q5_K, .q6_K, .iq4_xs, .iq3_xxs, .iq3_s,
                                          .iq2_xxs, .iq2_xs, .iq2_s, .iq1_m]
        for type in checked {
            let format = try #require(Native.QuantBlockFormat.of(type))
            #expect(sizes["\(type)"] == format.bytes, "block_\(type)")
        }
    }

    /// Every reference lookup table equals the header's table.
    @Test func quantTablesMatchGGMLCommon() throws {
        let text = try Self.header("Sources/LlamaCpp/ggml/src/ggml-common.h")
        func table(_ name: String) throws -> [Int64] {
            let regex = try NSRegularExpression(
                pattern: #"GGML_TABLE_BEGIN\(\w+,\s*"# + name + #",\s*\w+\)(.*?)GGML_TABLE_END"#,
                options: .dotMatchesLineSeparators)
            let m = try #require(regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)))
            let body = String(text[Range(m.range(at: 1), in: text)!])
                .split(separator: "\n").map { $0.split(separator: "/", maxSplits: 1).first ?? "" }.joined(separator: " ")
            let numbers = try NSRegularExpression(pattern: #"-?0x[0-9a-fA-F]+|-?\d+"#)
            return numbers.matches(in: body, range: NSRange(body.startIndex..., in: body)).map {
                let token = String(body[Range($0.range, in: body)!])
                if token.hasPrefix("0x") { return Int64(token.dropFirst(2), radix: 16)! }
                if token.hasPrefix("-0x") { return -Int64(token.dropFirst(3), radix: 16)! }
                return Int64(token)!
            }
        }
        typealias T = Native.ReferenceQuantTables
        #expect(try table("kmask_iq2xs") == T.kmask_iq2xs.map { Int64($0) })
        #expect(try table("ksigns_iq2xs") == T.ksigns_iq2xs.map { Int64($0) })
        #expect(try table("kvalues_iq4nl") == T.kvalues_iq4nl.map { Int64($0) })
        #expect(try table("iq3xxs_grid") == T.iq3xxs_grid.map { Int64($0) })
        #expect(try table("iq3s_grid") == T.iq3s_grid.map { Int64($0) })
    }
}
