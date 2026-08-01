import Foundation
import Testing
@testable import Mference

@Suite("JSONValue decode depth")
struct JSONValueDepthTests {
    /// Wraps `leaf` in `levels` nested single-key objects.
    private func nestedObject(levels: Int, leaf: String = #"{"type":"string"}"#) -> Data {
        var text = leaf
        for _ in 0..<levels { text = #"{"a":\#(text)}"# }
        return Data(text.utf8)
    }

    /// Wraps `[]` in `levels` nested arrays, the shape the crashing request used.
    private func nestedArray(levels: Int) -> Data {
        Data((String(repeating: "[", count: levels)
              + String(repeating: "]", count: levels)).utf8)
    }

    @Test func nestedObjectsBeyondTheDepthLimitAreRejected() throws {
        // Deliberately only just past the limit: a body deep enough to actually
        // overflow would take the test runner down with it.
        let data = nestedObject(levels: JSONValue.maximumDepth + 8)
        #expect(throws: DecodingError.self) {
            try JSONDecoder().decode(JSONValue.self, from: data)
        }
    }

    @Test func nestedArraysBeyondTheDepthLimitAreRejected() throws {
        let data = nestedArray(levels: JSONValue.maximumDepth + 8)
        #expect(throws: DecodingError.self) {
            try JSONDecoder().decode(JSONValue.self, from: data)
        }
    }

    @Test func schemaAtRealisticNestingStillDecodes() throws {
        // Twelve levels is deeper than any tool schema observed in the wild;
        // the limit has to clear it with room to spare.
        let data = nestedObject(levels: 12)
        let decoded = try JSONDecoder().decode(JSONValue.self, from: data)
        var cursor = decoded
        for _ in 0..<12 {
            cursor = try #require(cursor.objectValue?["a"])
        }
        #expect(cursor.objectValue?["type"] == .string("string"))
    }

    @Test func decodeDepthCountsFromTheDocumentRoot() throws {
        // Stack depth is what the limit is protecting, so the keys of an
        // enclosing type count towards it: the same fragment that fits at the
        // root is one level too deep once it is nested inside a request.
        struct Envelope: Decodable {
            let outer: JSONValue
        }
        let fragment = String(
            decoding: nestedObject(levels: JSONValue.maximumDepth - 1), as: UTF8.self)
        _ = try JSONDecoder().decode(JSONValue.self, from: Data(fragment.utf8))
        #expect(throws: DecodingError.self) {
            try JSONDecoder().decode(
                Envelope.self, from: Data(#"{"outer":\#(fragment)}"#.utf8))
        }
    }
}
