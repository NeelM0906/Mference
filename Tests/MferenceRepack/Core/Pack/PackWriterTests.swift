import Foundation
import Testing
@testable import MferenceRepackCore

/// The production `.mfpack` writer on small synthetic GGUFs: resume after an
/// interruption, the install lock, refusals, bounded scratch, tiled layouts
/// and hostile headers. Byte identity with the reference writer is pinned in
/// `PackWeightStoreTests` (`MferenceTestsCore`, which links both targets).
@Suite("PackWriterTests", .serialized)
struct PackWriterTests {
    /// Minimal GGUF v3 writer.
    struct GGUF {
        var kv: [(String, UInt32, [UInt8])] = []
        var tensors: [(name: String, type: MFPack.GGMLType, shape: [Int], bytes: [UInt8])] = []
        var alignment = 32
        var magic = Array("GGUF".utf8)
        var offsetOverride: [Int: UInt64] = [:]

        static func le<T: FixedWidthInteger>(_ v: T) -> [UInt8] { withUnsafeBytes(of: v.littleEndian) { Array($0) } }
        static func str(_ s: String) -> [UInt8] { le(UInt64(s.utf8.count)) + Array(s.utf8) }
        mutating func u32(_ k: String, _ v: UInt32) { kv.append((k, 4, Self.le(v))) }
        mutating func f32(_ k: String, _ v: Float) { kv.append((k, 6, Self.le(v.bitPattern))) }
        mutating func f64(_ k: String, _ v: Double) { kv.append((k, 12, Self.le(v.bitPattern))) }
        mutating func bool(_ k: String, _ v: Bool) { kv.append((k, 7, [v ? 1 : 0])) }
        mutating func i64(_ k: String, _ v: Int64) { kv.append((k, 11, Self.le(v))) }
        mutating func string(_ k: String, _ v: String) { kv.append((k, 8, Self.str(v))) }
        mutating func strings(_ k: String, _ v: [String]) {
            kv.append((k, 9, Self.le(UInt32(8)) + Self.le(UInt64(v.count)) + v.flatMap(Self.str)))
        }
        mutating func i32s(_ k: String, _ v: [Int32]) {
            kv.append((k, 9, Self.le(UInt32(5)) + Self.le(UInt64(v.count)) + v.flatMap { Self.le($0) }))
        }

        func data() -> Data {
            var out = magic + Self.le(UInt32(3)) + Self.le(UInt64(tensors.count)) + Self.le(UInt64(kv.count))
            for (k, t, v) in kv { out += Self.str(k) + Self.le(t) + v }
            var offset = 0
            var offsets: [Int] = []
            for t in tensors {
                offsets.append(offset)
                offset = (offset + t.bytes.count + alignment - 1) / alignment * alignment
            }
            for (i, (t, o)) in zip(tensors, offsets).enumerated() {
                out += Self.str(t.name) + Self.le(UInt32(t.shape.count))
                for d in t.shape { out += Self.le(UInt64(d)) }
                out += Self.le(t.type.rawValue) + Self.le(offsetOverride[i] ?? UInt64(o))
            }
            while out.count % alignment != 0 { out.append(0) }
            let base = out.count
            for (t, o) in zip(tensors, offsets) {
                while out.count < base + o { out.append(0) }
                out += t.bytes
            }
            return Data(out)
        }
    }

    static func bytes(_ n: Int, seed: UInt64) -> [UInt8] {
        var s = seed &* 0x9E37_79B9_7F4A_7C15 | 1
        return (0..<n).map { _ in
            s ^= s << 13
            s ^= s >> 7
            s ^= s << 17
            return UInt8(truncatingIfNeeded: s)
        }
    }

    /// A `qwen35`-shaped header with a handful of tensors of mixed types.
    static func model(tensors extra: Int = 6) -> GGUF {
        var g = GGUF()
        g.string("general.architecture", "qwen35")
        g.string("general.name", "writer-test")
        g.f64("general.some_f64", 0.1)
        g.bool("general.flag", true)
        g.i64("general.signed", -3)
        for (k, v) in [("block_count", 2), ("context_length", 4096), ("embedding_length", 256),
                       ("feed_forward_length", 512), ("attention.head_count", 4), ("attention.head_count_kv", 2),
                       ("attention.key_length", 32), ("rope.dimension_count", 16), ("ssm.conv_kernel", 4),
                       ("ssm.state_size", 32), ("ssm.group_count", 2), ("ssm.time_step_rank", 4),
                       ("ssm.inner_size", 128)] {
            g.u32("qwen35.\(k)", UInt32(v))
        }
        g.f32("qwen35.rope.freq_base", 10_000)
        g.f32("qwen35.attention.layer_norm_rms_epsilon", 1e-6)
        g.i32s("qwen35.rope.dimension_sections", [3, 3, 2, 0])
        g.string("tokenizer.ggml.model", "gpt2")
        g.string("tokenizer.ggml.pre", "qwen35")
        g.strings("tokenizer.ggml.tokens", (0..<300).map { "t\($0)" })
        g.i32s("tokenizer.ggml.token_type", [Int32](repeating: 1, count: 300))
        g.strings("tokenizer.ggml.merges", ["t 1", "t 2"])
        g.u32("tokenizer.ggml.eos_token_id", 7)
        g.u32("tokenizer.ggml.eot_token_id", 8)
        g.string("tokenizer.chat_template", "{{ messages }}é")
        g.tensors.append(("token_embd.weight", .q6_K, [256, 300], bytes(300 * 210, seed: 1)))
        g.tensors.append(("output_norm.weight", .f32, [256], bytes(1024, seed: 2)))
        let types: [MFPack.GGMLType] = [.q4_K, .q5_K, .q8_0, .iq4_xs, .f16, .q6_K, .bf16, .iq3_s]
        for i in 0..<extra {
            let t = types[i % types.count]
            let f = MFPack.QuantBlockFormat.of(t)!
            let rows = 8 * (1 + i % 3)
            g.tensors.append(("blk.\(i).w", t, [256, rows], bytes(rows * 256 / f.elements * f.bytes, seed: 10 + UInt64(i))))
        }
        return g
    }

    static func tempDir() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("mference-pack-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    static func tiled(_ t: MFPack.GGMLType, _ rows: Int, _ blocks: Int) -> MFPack.TileLayout {
        guard t != .f32 else { return .gguf }
        let l = MFPack.TileLayout(rowsPerTile: 4, blocksPerTile: 1, planes: true)
        return l.fits(rows: rows, blocksPerRow: blocks) ? l : .gguf
    }

    static func files(_ dir: String) throws -> [String: Data] {
        var out: [String: Data] = [:]
        for name in ["weights.bin", "manifest.json", "tokenizer.json", "chat_template.jinja"] {
            out[name] = try Data(contentsOf: URL(fileURLWithPath: (dir as NSString).appendingPathComponent(name)))
        }
        return out
    }

    @Test func writesAVerifiedPack() throws {
        let dir = try Self.tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let gguf = dir.appendingPathComponent("m.gguf")
        try Self.model().data().write(to: gguf)
        let out = dir.appendingPathComponent("m.mfpack").path
        let r = try MFPackWriter.write(gguf: gguf.path, output: out, options: .init(label: "m"))
        let m = r.manifest
        #expect(m.tensors.count == 8 && m.tensors.allSatisfy { $0.offset % 4096 == 0 && $0.layout == .gguf })
        #expect(m.layoutTableVersion == MFPack.LayoutTable.version)
        #expect(m.config.blockCount == 2 && m.config.vocabSize == 300 && m.config.tiedEmbeddings)
        #expect(m.config.valueLength == 32 && m.config.fullAttentionInterval == 4 && m.config.ropeSections == [3, 3, 2, 0])
        #expect(m.metadata["general.some_f64"] == "0.1" && m.metadata["general.flag"] == "true")
        #expect(m.metadata["general.signed"] == "-3" && m.metadata["tokenizer.ggml.pre"] == "qwen35")
        #expect(m.metadata["tokenizer.ggml.eos_token_id"] == nil)
        #expect(m.source.byteCount == UInt64(try Data(contentsOf: gguf).count))
        #expect(m.source.sha256 == MFPackWriter.sha256Hex(try Data(contentsOf: gguf)))
        let (manifest, receipt) = try MFPackVerifier.readVerified(directory: out)
        #expect(manifest == m && receipt == r.receipt)
        let tok = try JSONDecoder().decode(MFPack.TokenizerData.self, from: Self.files(out)["tokenizer.json"]!)
        #expect(tok.eosTokenID == 7 && tok.specialTokenIDs == ["eot": 8] && tok.tokens.count == 300)
        #expect(try Self.files(out)["chat_template.jinja"] == Data("{{ messages }}é".utf8))
        #expect(try MFPackVerifier.roundTrip(pack: out, gguf: gguf.path, chunkBytes: 1000).tensors == 8)
        // No partial or checkpoint left; the lock file stays, as for .gturbo installs.
        #expect(!FileManager.default.fileExists(atPath: out + ".partial"))
        #expect(!FileManager.default.fileExists(atPath: out + ".resume.json"))
        // Never overwrites.
        #expect(throws: MFPackError.self) {
            try MFPackWriter.write(gguf: gguf.path, output: out, options: .init(label: "m"))
        }
    }

    @Test(arguments: [false, true])
    func resumeAfterAnInterruptionMatchesAFreshWrite(tiled: Bool) throws {
        let dir = try Self.tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let gguf = dir.appendingPathComponent("m.gguf")
        try Self.model(tensors: 14).data().write(to: gguf)
        let layout: @Sendable (MFPack.GGMLType, Int, Int) -> MFPack.TileLayout = { type, rows, blocks in
            tiled ? Self.tiled(type, rows, blocks) : MFPack.LayoutTable.layout(for: type, rows: rows, blocksPerRow: blocks)
        }
        func options(resume: Bool = false, stop: Int? = nil) -> MFPackWriteOptions {
            MFPackWriteOptions(label: "m", resume: resume, layout: layout,
                               readChunkBytes: 1500, scratchLimitBytes: 64 << 10, checkpointEveryBytes: 1,
                               stopAfterTensors: stop)
        }
        let fresh = dir.appendingPathComponent("fresh.mfpack").path
        let freshResult = try MFPackWriter.write(gguf: gguf.path, output: fresh, options: options())
        #expect(freshResult.scratchHighWaterBytes <= 64 << 10)
        #expect(freshResult.manifest.tensors.contains { $0.layout != .gguf } == tiled)

        let out = dir.appendingPathComponent("m.mfpack").path
        #expect(throws: MFPackError.outputState("stopped after 5 tensors (test hook)")) {
            try MFPackWriter.write(gguf: gguf.path, output: out, options: options(stop: 5))
        }
        #expect(FileManager.default.fileExists(atPath: out + ".partial"))
        #expect(FileManager.default.fileExists(atPath: out + ".resume.json"))
        // A partial pack is refused without resume.
        #expect(throws: MFPackError.self) { try MFPackWriter.write(gguf: gguf.path, output: out, options: options()) }
        // Resume once more with an interruption, then corrupt a kept tensor:
        // it is detected and rewritten.
        #expect(throws: MFPackError.self) {
            try MFPackWriter.write(gguf: gguf.path, output: out, options: options(resume: true, stop: 3))
        }
        let cp = try JSONDecoder().decode(MFPack.ResumeCheckpoint.self,
                                          from: Data(contentsOf: URL(fileURLWithPath: out + ".resume.json")))
        #expect(cp.completed.count == 8)
        let victim = freshResult.manifest.tensors.first { cp.completed[$0.name] != nil }!
        let weights = URL(fileURLWithPath: out + ".partial/weights.bin")
        var d = try Data(contentsOf: weights)
        d[Int(victim.offset) + 2] ^= 0xFF
        try d.write(to: weights)
        let r = try MFPackWriter.write(gguf: gguf.path, output: out, options: options(resume: true))
        #expect(r.resumedTensors == 7)
        var rewritten: UInt64 = 0
        for t in freshResult.manifest.tensors where cp.completed[t.name] == nil || t.name == victim.name {
            rewritten += t.byteCount
        }
        #expect(r.writtenBytes == rewritten)
        #expect(try Self.files(out) == Self.files(fresh))
        #expect(r.manifest == freshResult.manifest)
        #expect(try MFPackVerifier.roundTrip(pack: out, gguf: gguf.path).tensors == 16)
    }

    @Test func resumeRefusesAChangedSourceAndDiscardWorks() throws {
        let dir = try Self.tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let gguf = dir.appendingPathComponent("m.gguf")
        try Self.model().data().write(to: gguf)
        let out = dir.appendingPathComponent("m.mfpack").path
        let opts = MFPackWriteOptions(label: "m", checkpointEveryBytes: 1, stopAfterTensors: 2)
        #expect(throws: MFPackError.self) { try MFPackWriter.write(gguf: gguf.path, output: out, options: opts) }
        // A different label (or source) does not resume into this partial pack.
        #expect(throws: MFPackError.self) {
            try MFPackWriter.write(gguf: gguf.path, output: out, options: .init(label: "other", resume: true))
        }
        var g = Self.model()
        g.tensors[0].bytes[0] ^= 1
        try g.data().write(to: gguf)
        #expect(throws: MFPackError.self) {
            try MFPackWriter.write(gguf: gguf.path, output: out, options: .init(label: "m", resume: true))
        }
        try MFPackWriter.discardPartial(output: out)
        #expect(!FileManager.default.fileExists(atPath: out + ".partial"))
        #expect(!FileManager.default.fileExists(atPath: out + ".resume.json"))
        let r = try MFPackWriter.write(gguf: gguf.path, output: out, options: .init(label: "m"))
        #expect(r.resumedTensors == 0)
    }

    @Test func installLockExcludesASecondWriter() throws {
        let dir = try Self.tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let gguf = dir.appendingPathComponent("m.gguf")
        try Self.model().data().write(to: gguf)
        let out = dir.appendingPathComponent("m.mfpack").path
        let held = try InstallLock.acquire(outputDirectory: out)
        #expect(throws: RepackError.self) {
            try MFPackWriter.write(gguf: gguf.path, output: out, options: .init(label: "m"))
        }
        withExtendedLifetime(held) {}
    }

    @Test func scratchBoundsTheRowGroup() throws {
        let dir = try Self.tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let gguf = dir.appendingPathComponent("m.gguf")
        try Self.model().data().write(to: gguf)
        // token_embd's 4-row group is 4 * 210 bytes: over a 512-byte bound.
        let opts = MFPackWriteOptions(label: "m", layout: { Self.tiled($0, $1, $2) }, readChunkBytes: 4096,
                                      scratchLimitBytes: 512)
        #expect(throws: MFPackError.self) {
            try MFPackWriter.write(gguf: gguf.path, output: dir.appendingPathComponent("m.mfpack").path, options: opts)
        }
    }

    @Test func hostileHeadersAreRejected() throws {
        let dir = try Self.tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        func rejects(_ g: GGUF, _ comment: Comment) throws {
            let url = dir.appendingPathComponent("\(UUID().uuidString).gguf")
            try g.data().write(to: url)
            #expect(throws: MFPackError.self, comment) { _ = try MFPack.GGUFHeader(path: url.path).qwen35Config() }
        }
        var g = Self.model()
        g.magic = Array("GGUX".utf8)
        try rejects(g, "magic")
        g = Self.model()
        g.offsetOverride = [1: 0]
        try rejects(g, "overlap")
        g = Self.model()
        g.offsetOverride = [1: 3]
        try rejects(g, "misaligned")
        g = Self.model()
        g.offsetOverride = [1: 1 << 40]
        try rejects(g, "past the end")
        g = Self.model()
        g.tensors[2].shape = [255, 8]
        try rejects(g, "partial block")
        g = Self.model()
        g.tensors.append(g.tensors[0])
        try rejects(g, "duplicate tensor")
        g = Self.model()
        g.kv.removeAll { $0.0 == "general.architecture" }
        g.string("general.architecture", "llama")
        try rejects(g, "architecture")
        g = Self.model()
        g.kv.removeAll { $0.0 == "qwen35.ssm.inner_size" }
        try rejects(g, "missing key")
        g = Self.model()
        g.kv.append(("general.alignment", 10, GGUF.le(UInt64(32))))
        try rejects(g, "u64 alignment")
        // Truncated header.
        let url = dir.appendingPathComponent("short.gguf")
        try Self.model().data().prefix(200).write(to: url)
        #expect(throws: MFPackError.self) { _ = try MFPack.GGUFHeader(path: url.path) }
    }
}
