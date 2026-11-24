import Testing
import Foundation
@testable import Mference

@Suite struct Sha256VerifierTests {

    /// SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    @Test func hashesEmptyFile() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-sha-empty-\(UUID().uuidString).bin")
        try Data().write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let h = try Sha256Verifier.hashFile(at: url)
        #expect(h == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    }

    @Test func chunkSizeDoesNotAffectDigest() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-sha-2m-\(UUID().uuidString).bin")
        // Two chunks at the default 1 MB chunkBytes — exercises the loop.
        var data = Data(count: 2 << 20)
        for i in 0..<data.count { data[i] = UInt8(i & 0xFF) }
        try data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let small = try Sha256Verifier.hashFile(at: url, chunkBytes: 64 << 10)
        let big   = try Sha256Verifier.hashFile(at: url)
        #expect(small == big, "chunk size must not affect digest")
    }

    @Test func verifyMatches() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-sha-match-\(UUID().uuidString).bin")
        let payload = Data("hello world".utf8)
        try payload.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let hex = try Sha256Verifier.hashFile(at: url)
        try Sha256Verifier.verifyFile(at: url, named: "hello", expectedHex: hex)
    }

    @Test func verifyMismatchThrowsChecksumMismatch() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-sha-bad-\(UUID().uuidString).bin")
        try Data("hello world".utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let wrong = String(repeating: "0", count: 64)
        #expect(throws: ModelError.checksumMismatch(file: "hello")) {
            try Sha256Verifier.verifyFile(at: url, named: "hello", expectedHex: wrong)
        }
    }

    /// `bypassCache` (`F_NOCACHE`) changes how the file is read, never the
    /// digest: equal with and without it, and equal to hashing the bytes in
    /// memory, at sizes that are multiples of neither the chunk nor the page.
    @Test func bypassingTheCacheGivesTheSameDigest() throws {
        let page = Int(getpagesize())
        for size in [0, 1, page - 1, page + 1, (1 << 20) + page + 1, 3 * (1 << 20) + 12_345] {
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("gturbo-sha-nocache-\(size)-\(UUID().uuidString).bin")
            var data = Data(count: size)
            for i in 0..<size { data[i] = UInt8(truncatingIfNeeded: i &* 31 &+ (i >> 9)) }
            try data.write(to: url)
            defer { try? FileManager.default.removeItem(at: url) }
            let expected = Sha256Verifier.hashData(data)
            for chunk in [64 << 10, 1 << 20] {
                #expect(try Sha256Verifier.hashFile(at: url, chunkBytes: chunk) == expected, "\(size) at \(chunk)")
                #expect(try Sha256Verifier.hashFile(at: url, chunkBytes: chunk, bypassCache: true) == expected,
                        "\(size) at \(chunk), uncached")
            }
            try Sha256Verifier.verifyFile(at: url, named: "f", expectedHex: expected, bypassCache: true)
            #expect(throws: ModelError.checksumMismatch(file: "f")) {
                try Sha256Verifier.verifyFile(at: url, named: "f", expectedHex: String(repeating: "0", count: 64),
                                              bypassCache: true)
            }
        }
    }

    /// Exactly one caller reads uncached: the Inkling 2 expert-layer check,
    /// whose pool (15.8 GB) would otherwise pass through the file cache at
    /// load. Every other `hashFile` / `verifyFile` call in `Sources` passes no
    /// `bypassCache`, so the other families read as before.
    @Test func onlyTheInklingExpertLayerCheckBypassesTheCache() throws {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<6 { root.deleteLastPathComponent() }   // .../ModelIO/<file> up to the package root
        let sources = root.appendingPathComponent("Sources")
        #expect(FileManager.default.fileExists(atPath: sources.appendingPathComponent("Mference").path))
        var flagged: [String] = []
        var plain = 0
        let files = try #require(FileManager.default.enumerator(at: sources, includingPropertiesForKeys: nil))
        for case let url as URL in files where url.pathExtension == "swift" {
            let text = try String(contentsOf: url, encoding: .utf8)
            let file = String(url.path.dropFirst(sources.path.count + 1))
            for arguments in Self.verifierCallArguments(in: text) {
                if arguments.contains("bypassCache") {
                    flagged.append(file + ": " + arguments)
                } else {
                    plain += 1
                }
            }
        }
        #expect(flagged.count == 1, "\(flagged)")
        #expect(flagged.first?.hasPrefix("Mference/Runtime/Inference/Inkling2ForwardRunner+Model.swift: ") == true,
                "\(flagged)")
        #expect(flagged.first?.hasSuffix("bypassCache: true") == true, "\(flagged)")
        // The scan sees the existing callers (manifest, weights, layout,
        // streamed layers, the server's and engine's manifest hashes).
        #expect(plain >= 6, "\(plain)")
    }

    /// The argument lists of every `Sha256Verifier.hashFile(` and
    /// `Sha256Verifier.verifyFile(` call in `text`, whitespace collapsed.
    static func verifierCallArguments(in text: String) -> [String] {
        var out: [String] = []
        for marker in ["Sha256Verifier.hashFile(", "Sha256Verifier.verifyFile("] {
            var rest = text[...]
            while let start = rest.range(of: marker) {
                var depth = 1
                var index = start.upperBound
                while index < rest.endIndex, depth > 0 {
                    if rest[index] == "(" { depth += 1 } else if rest[index] == ")" { depth -= 1 }
                    index = rest.index(after: index)
                }
                let arguments = rest[start.upperBound..<rest.index(before: index)]
                out.append(arguments.split(whereSeparator: \.isWhitespace).joined(separator: " "))
                rest = rest[index...]
            }
        }
        return out
    }
}
