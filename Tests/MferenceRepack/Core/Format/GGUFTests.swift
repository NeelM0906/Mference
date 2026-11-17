import Foundation
import Testing
@testable import MferenceRepackCore

/// Builds small GGUF v3 files in memory for parser tests.
struct SyntheticGGUF {
    var version: UInt32 = 3
    var kv: [(String, UInt32, [UInt8])] = []
    var tensors: [(name: String, ne: [UInt64], type: UInt32, offset: UInt64)] = []
    var magic: [UInt8] = Array("GGUF".utf8)

    static func le<T: FixedWidthInteger>(_ v: T) -> [UInt8] {
        (0..<MemoryLayout<T>.size).map { UInt8(truncatingIfNeeded: v >> (8 * $0)) }
    }
    static func str(_ s: String) -> [UInt8] { le(UInt64(s.utf8.count)) + Array(s.utf8) }
    static func rawStr(_ b: [UInt8]) -> [UInt8] { le(UInt64(b.count)) + b }

    mutating func addU32(_ k: String, _ v: UInt32) { kv.append((k, 4, Self.le(v))) }
    mutating func addString(_ k: String, _ v: String) { kv.append((k, 8, Self.str(v))) }
    mutating func addStringArray(_ k: String, _ v: [String]) {
        kv.append((k, 9, Self.le(UInt32(8)) + Self.le(UInt64(v.count)) + v.flatMap(Self.str)))
    }
    mutating func addI64Array(_ k: String, _ v: [Int64]) {
        kv.append((k, 9, Self.le(UInt32(11)) + Self.le(UInt64(v.count)) + v.flatMap { Self.le($0) }))
    }

    func header() -> [UInt8] {
        var b = magic + Self.le(version) + Self.le(UInt64(tensors.count)) + Self.le(UInt64(kv.count))
        for (k, t, v) in kv { b += Self.str(k) + Self.le(t) + v }
        for t in tensors {
            b += Self.str(t.name) + Self.le(UInt32(t.ne.count))
            for d in t.ne { b += Self.le(d) }
            b += Self.le(t.type) + Self.le(t.offset)
        }
        return b
    }

    /// Header padded to `alignment`, followed by `dataBytes` zero bytes.
    func file(alignment: Int = 32, dataBytes: Int) -> Data {
        var h = header()
        while h.count % alignment != 0 { h.append(0) }
        return Data(h + [UInt8](repeating: 0, count: dataBytes))
    }
}

@Suite
struct GGUFTests {
    private func parseComplete(_ data: Data, path: String = "s.gguf") throws -> GGUF.Header {
        switch try GGUF.parseHeader(path: path, fileSize: UInt64(data.count), prefix: data) {
        case .complete(let h): return h
        case .needBytes(let n): Issue.record("unexpected needBytes \(n)"); throw RepackError.testHookStop(stage: "needBytes")
        }
    }

    private func sample() -> SyntheticGGUF {
        var g = SyntheticGGUF()
        g.addString("general.architecture", "qwen4exp")
        g.addU32("qwen4exp.block_count", 48)
        g.addStringArray("tokenizer.ggml.tokens", ["a", "b", "Ġc"])
        g.addI64Array("qwen4exp.ple_layer_multipliers", [35_184_372_088_831, -1])
        // Q6_K [256, 2] = 2 blocks of 210 B = 420 B; IQ4_NL [64, 3] = 6 blocks of 18 B = 108 B.
        g.tensors = [("blk.0.attn_q.weight", [256, 2], 14, 0),
                     ("blk.0.ffn_down_exps.weight", [64, 3], 20, 448)]
        return g
    }

    @Test func parsesMetadataAndTensorTable() throws {
        let g = sample()
        let data = g.file(dataBytes: 448 + 108)
        let h = try parseComplete(data)
        #expect(h.version == 3)
        #expect(h.keys == ["general.architecture", "qwen4exp.block_count", "tokenizer.ggml.tokens",
                           "qwen4exp.ple_layer_multipliers"])
        #expect(h.metadata["general.architecture"]?.string == "qwen4exp")
        #expect(h.metadata["qwen4exp.block_count"]?.uint64 == 48)
        #expect(h.metadata["tokenizer.ggml.tokens"] == .array([.string("a"), .string("b"), .string("Ġc")]))
        #expect(h.metadata["qwen4exp.ple_layer_multipliers"] == .array([.i64(35_184_372_088_831), .i64(-1)]))
        #expect(h.alignment == 32)
        #expect(h.headerBytes == UInt64(g.header().count))
        #expect(h.dataOffset % 32 == 0 && h.dataOffset >= h.headerBytes && h.dataOffset - h.headerBytes < 32)
        #expect(h.tensors.count == 2)
        #expect(h.tensors[0] == GGUF.TensorInfo(name: "blk.0.attn_q.weight", ne: [256, 2], type: .q6_K,
                                                relativeOffset: 0, absoluteOffset: h.dataOffset, sizeBytes: 420))
        #expect(h.tensors[1].type == .iq4_nl)
        #expect(h.tensors[1].absoluteOffset == h.dataOffset + 448)
        #expect(h.tensors[1].sizeBytes == 108)
    }

    @Test func shortPrefixAsksForMoreBytesThenCompletes() throws {
        let data = sample().file(dataBytes: 448 + 108)
        var have = 16
        var rounds = 0
        while true {
            rounds += 1
            #expect(rounds < 20)
            switch try GGUF.parseHeader(path: "s.gguf", fileSize: UInt64(data.count), prefix: data.prefix(have)) {
            case .complete(let h):
                #expect(h.tensors.count == 2)
                return
            case .needBytes(let n):
                #expect(n > have)
                have = n
            }
        }
    }

    @Test func emptyTensorTableIsValid() throws {
        // Shard 1 of a split GGUF carries only metadata.
        var g = SyntheticGGUF()
        g.addU32("split.count", 4)
        let h = try parseComplete(g.file(dataBytes: 0))
        #expect(h.tensors.isEmpty)
        #expect(h.metadata["split.count"]?.uint64 == 4)
    }

    /// Regression (real shard 1): a metadata-only shard whose header ends at
    /// EOF on a non-aligned byte, so the aligned data offset is past the end.
    @Test func headerOnlyShardEndingOffAlignmentIsValid() throws {
        var g = SyntheticGGUF()
        g.addString("general.architecture", "qwen4exp")
        var header = g.header()
        if header.count % 32 == 0 { g.addString("pad", "x"); header = g.header() }
        #expect(header.count % 32 != 0)
        let h = try parseComplete(Data(header))
        #expect(h.tensors.isEmpty && h.dataOffset > UInt64(header.count))
        // With a tensor, a data section past EOF is still rejected.
        g.tensors = [("t", [32], 0, 0)]
        #expect(throws: RepackError.self) { try parseComplete(Data(g.header())) }
    }

    @Test func honoursGeneralAlignment() throws {
        var g = SyntheticGGUF()
        g.addU32("general.alignment", 64)
        g.tensors = [("t", [32], 0, 0)]
        let h = try parseComplete(g.file(alignment: 64, dataBytes: 128))
        #expect(h.alignment == 64)
        #expect(h.dataOffset % 64 == 0)
        #expect(h.tensors[0].sizeBytes == 128)
    }

    @Test func rejectsUnsupportedType() {
        var g = SyntheticGGUF()
        g.tensors = [("q4k", [256], 12, 0)]  // Q4_K is not in the accepted set
        #expect(throws: RepackError.self) { try parseComplete(g.file(dataBytes: 144)) }
    }

    @Test func rejectsPartialBlockRow() {
        var g = SyntheticGGUF()
        g.tensors = [("t", [100], 8, 0)]  // Q8_0 rows must be multiples of 32
        #expect(throws: RepackError.self) { try parseComplete(g.file(dataBytes: 256)) }
    }

    @Test func rejectsTensorPastEndOfFile() {
        var g = SyntheticGGUF()
        g.tensors = [("t", [256, 2], 14, 0)]
        #expect(throws: RepackError.self) { try parseComplete(g.file(dataBytes: 419)) }
    }

    @Test func rejectsOverlapAndMisalignment() {
        var g = SyntheticGGUF()
        g.tensors = [("a", [256, 2], 14, 0), ("b", [32], 8, 416)]
        #expect(throws: RepackError.self) { try parseComplete(g.file(dataBytes: 1024)) }
        g.tensors = [("a", [32], 8, 8)]
        #expect(throws: RepackError.self) { try parseComplete(g.file(dataBytes: 1024)) }
    }

    @Test func rejectsBadMagicVersionUTF8AndDuplicates() {
        var g = SyntheticGGUF()
        g.magic = Array("GGUG".utf8)
        #expect(throws: RepackError.self) { try parseComplete(g.file(dataBytes: 0)) }

        g = SyntheticGGUF()
        g.version = 2
        #expect(throws: RepackError.self) { try parseComplete(g.file(dataBytes: 0)) }

        g = SyntheticGGUF()
        g.kv.append(("k", 8, SyntheticGGUF.rawStr([0xC3, 0x28])))
        #expect(throws: RepackError.self) { try parseComplete(g.file(dataBytes: 0)) }

        g = SyntheticGGUF()
        g.addU32("k", 1)
        g.addU32("k", 2)
        #expect(throws: RepackError.self) { try parseComplete(g.file(dataBytes: 0)) }

        g = SyntheticGGUF()
        g.tensors = [("t", [32], 0, 0), ("t", [32], 0, 128)]
        #expect(throws: RepackError.self) { try parseComplete(g.file(dataBytes: 256)) }
    }

    /// R9: the GGUF label installs by name but is not advertised, and its
    /// pinned `sourceIndexSHA256` is the pin's own fingerprint.
    @Test func ggufSourceIsUnlistedAndPinnedByFingerprint() throws {
        let source = try #require(SupportedModelSource.named("qwen38flashnextudiq3xxs"))
        #expect(!SupportedModelSource.all.contains { $0.name == source.name })
        #expect(source.kind == .gguf)
        let pin = try #require(source.ggufPin)
        #expect(pin.fingerprint == source.sourceIndexSHA256)
        #expect(pin.totalBytes == 83_626_438_426 && pin.shards.count == 4)
        #expect(SourceFingerprint.modelID(forIndexSha256: pin.fingerprint) == source.modelID)
        #expect(source.installOptions(outputDirectory: URL(fileURLWithPath: "/tmp/x"), overwrite: false,
                                      token: nil).ggufPin == pin)
    }

    /// R6: Foundation JSON round-trips full-width Int64 values exactly, so the
    /// PLE hash constants (up to 63 bits) need no decimal-string encoding.
    @Test func int64JSONRoundTripIsExact() throws {
        let values: [Int64] = [Int64.max, Int64.max - 1, (1 << 62) + 12_345, 23_703_573_157_769, -1]
        let data = try JSONSerialization.data(withJSONObject: ["v": values])
        let back = try JSONDecoder().decode([String: [Int64]].self, from: data)
        #expect(back["v"] == values)
    }

    /// The per-expert byte counts in the U0 inventory follow from the type
    /// table: expert gate/up are [2560, 640] and down is [640, 2560].
    @Test func expertBytesMatchTheU0Inventory() throws {
        func expert(_ gateUp: GGMLType, _ down: GGMLType) throws -> UInt64 {
            let gu = try #require(gateUp.rowBytes(elements: 2560)) * 640
            let d = try #require(down.rowBytes(elements: 640)) * 2560
            return 2 * gu + d
        }
        #expect(try expert(.iq2_s, .iq4_nl) == 1_971_200)
        #expect(try expert(.iq3_xxs, .iq4_nl) == 2_176_000)
        #expect(try expert(.iq2_s, .q2_0) == 1_510_400)
        #expect(try expert(.iq4_xs, .iq4_nl) == 2_662_400)
    }
}
