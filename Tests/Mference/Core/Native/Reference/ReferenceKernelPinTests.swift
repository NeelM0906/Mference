import Foundation
import Testing
@testable import Mference

/// Pins the reference Metal module's tables to the Swift ones, and the
/// attention reference's IMROPE admissibility rule.
@Suite("ReferenceKernelPinTests")
struct ReferenceKernelPinTests {
    /// The Metal module's copies of the tables equal the Swift ones.
    @Test func metalTablesMatchSwiftTables() throws {
        let text = try ReferenceFormatPinTests.header("Sources/Mference/Metal/Native/Reference/native_reference.metal")
        func table(_ name: String) throws -> [Int64] {
            let regex = try NSRegularExpression(pattern: #"ref_"# + name + #"\[\d+\] = \{(.*?)\};"#,
                                                options: .dotMatchesLineSeparators)
            let m = try #require(regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)))
            let body = String(text[Range(m.range(at: 1), in: text)!])
            return body.split(separator: ",").compactMap {
                let t = $0.trimmingCharacters(in: .whitespacesAndNewlines)
                if t.isEmpty { return nil }
                return t.hasPrefix("0x") ? Int64(t.dropFirst(2), radix: 16) : Int64(t)
            }
        }
        typealias T = Native.ReferenceQuantTables
        #expect(try table("kmask_iq2xs") == T.kmask_iq2xs.map { Int64($0) })
        #expect(try table("ksigns_iq2xs") == T.ksigns_iq2xs.map { Int64($0) })
        #expect(try table("kvalues_iq4nl") == T.kvalues_iq4nl.map { Int64($0) })
        #expect(try table("iq3xxs_grid") == T.iq3xxs_grid.map { Int64($0) })
        #expect(try table("iq3s_grid") == T.iq3s_grid.map { Int64($0) })
    }

    @Test func imropeCheckAcceptsQuickAndRejectsSectionE() throws {
        try Native.ReferenceAttentionOp.checkIMROPE(sections: [11, 11, 10, 0], ropeDims: 64)
        try Native.ReferenceAttentionOp.checkIMROPE(sections: [3, 3, 2, 0], ropeDims: 16)
        #expect(throws: Native.NativeError.self) {
            try Native.ReferenceAttentionOp.checkIMROPE(sections: [11, 11, 9, 1], ropeDims: 64)
        }
    }
}
