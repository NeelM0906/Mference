import Foundation
import Testing
@testable import MferenceRepackCore
@testable import Mference

/// The resident tensor contract checked at `Model.load`, against the
/// production header (described, not written) and the toy install.
@Suite struct Inkling2ResidentContractTests {

    /// The production header's resident tensors are exactly the contract's,
    /// with the shapes, formats and sizes the planner records.
    @Test func productionHeaderMatchesTheContract() throws {
        let fixture = Inkling2Shape.production.fixture()
        guard case .complete(let header) = try GGUF.parseHeader(
            path: "prod", fileSize: fixture.describedFileSize(), prefix: fixture.header().bytes) else {
            Issue.record("production header did not parse"); return
        }
        let arch = try ArchInfo.fromGGUFInkling2(metadata: header.metadata, tensors: header.tensors, path: "prod")
        let plan = try Inkling2GGUFPlanner.plan(header: header, shardPath: "prod", arch: arch,
                                                outputDir: "/nonexistent", matchedModelID: nil)
        let expected = Inkling2ResidentContract.expectedTensors(config: .inkling2Mini)
        #expect(expected.count == 696)
        #expect(Set(expected.map(\.name)) == Set(plan.resident.entries.map(\.name)))
        try Inkling2ResidentContract.validate(index: Self.index(plan.resident.entries), config: .inkling2Mini)
    }

    static func index(_ entries: [ResidentEntry]) -> ResidentIndex {
        var map: [String: ResidentIndexEntry] = [:]
        for e in entries {
            let s = e.logicalShape4
            map[e.name] = ResidentIndexEntry(name: e.name, dtype: e.dtype, fileOffset: e.fileOffset,
                                             sizeBytes: e.sizeBytes, shape: (s[0], s[1], s[2], s[3]),
                                             scaleOffset: 0, scaleSize: 0, biasOffset: 0, biasSize: 0)
        }
        return ResidentIndex(header: ResidentIndexHeader(indexSize: 0, residentSize: 0,
                                                         entryCount: UInt64(map.count)),
                             entries: map)
    }

    @Test func mismatchedIndexIsRefusedByName() throws {
        let shape = Inkling2Shape()
        let fixture = shape.fixture()
        let data = fixture.encode().data
        guard case .complete(let header) = try GGUF.parseHeader(path: "toy", fileSize: UInt64(data.count),
                                                                prefix: data) else {
            Issue.record("toy header did not parse"); return
        }
        let arch = try ArchInfo.fromGGUFInkling2(metadata: header.metadata, tensors: header.tensors, path: "toy")
        let plan = try Inkling2GGUFPlanner.plan(header: header, shardPath: "toy", arch: arch,
                                                outputDir: "/nonexistent", matchedModelID: nil)
        let good = Self.index(plan.resident.entries)
        try Inkling2ResidentContract.validate(index: good, config: shape.archConfig)

        func refused(_ edit: (inout [String: ResidentIndexEntry]) -> Void) -> ModelError? {
            var entries = good.entries
            edit(&entries)
            do {
                try Inkling2ResidentContract.validate(
                    index: ResidentIndex(header: good.header, entries: entries), config: shape.archConfig)
                return nil
            } catch { return error as? ModelError }
        }
        func with(_ name: String, dtype: UInt8? = nil, size: UInt64? = nil,
                  shape: (UInt32, UInt32, UInt32, UInt32)? = nil) -> (inout [String: ResidentIndexEntry]) -> Void {
            { entries in
                let e = entries[name]!
                entries[name] = ResidentIndexEntry(name: name, dtype: dtype ?? e.dtype, fileOffset: e.fileOffset,
                                                   sizeBytes: size ?? e.sizeBytes, shape: shape ?? e.shape,
                                                   scaleOffset: 0, scaleSize: 0, biasOffset: 0, biasSize: 0)
            }
        }
        #expect(refused { $0.removeValue(forKey: "blk.1.attn_rel.weight") }
                == .tensorNotFound(name: "blk.1.attn_rel.weight"))
        // The router must stay F32 (the shipped and the only planned format).
        #expect(refused(with("blk.0.ffn_gate_inp.weight", dtype: GGMLBlockFormat.bf16.residentDtype)) != nil)
        // A matrix whose bytes do not match its shape and format.
        #expect(refused(with("blk.0.attn_q.weight", size: 4)) != nil)
        // A transposed matrix ([64, 128] stored as [128, 64]).
        #expect(refused(with("blk.0.attn_k.weight", shape: (128, 64, 0, 0))) != nil)
        // A sliding layer's bias table at the global extent's shape is fine
        // only when the two agree; here the window (16) is the extent (16).
        #expect(refused(with("blk.0.attn_rel_bias.weight", shape: (8, 4, 0, 0))) != nil)
        // A tensor the family does not carry.
        #expect(refused { $0["blk.0.ffn_up_shexp.weight"] = $0["blk.0.ffn_norm.weight"] } != nil)
        // The NVFP4 layer's per-expert factor is required; the Q8_0 layer has none.
        #expect(refused { $0.removeValue(forKey: "blk.0.ffn_down_exps.scale") } != nil)
        #expect(good.entries["blk.1.ffn_down_exps.scale"] == nil)
        // One format per tensor, the one the kernels read in place: an F16
        // norm (its own size, so only the format check can catch it), a BF16
        // projection, a Q8_0 sink projection.
        #expect(refused(with("blk.0.attn_norm.weight", dtype: GGMLBlockFormat.f16.residentDtype, size: 128 * 2))
                != nil)
        #expect(refused(with("blk.0.attn_q.weight", dtype: GGMLBlockFormat.bf16.residentDtype, size: 128 * 128 * 2))
                != nil)
        #expect(refused(with("blk.0.attn_sink_proj.weight", dtype: GGMLBlockFormat.q8_0.residentDtype,
                             size: 2 * 4 * 34)) != nil)
        // An offset off the 256-byte boundary.
        #expect(refused { entries in
            let e = entries["blk.0.ffn_norm.weight"]!
            entries[e.name] = ResidentIndexEntry(name: e.name, dtype: e.dtype, fileOffset: e.fileOffset + 16,
                                                 sizeBytes: e.sizeBytes, shape: e.shape, scaleOffset: 0,
                                                 scaleSize: 0, biasOffset: 0, biasSize: 0)
        } != nil)
    }

    /// The contract against the real file's header, as K0 dumped it
    /// (`Fixtures/Inkling2/gguf_tensors.tsv`): every non-expert tensor by
    /// name, with the contract's shape (the GGUF's `ne` reversed), its one
    /// format, and its byte count.
    @Test func realHeaderDumpMatchesTheContract() throws {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<6 { root.deleteLastPathComponent() }
        let tsv = try String(contentsOf: root.appendingPathComponent(
            "Tests/Mference/Fixtures/Inkling2/gguf_tensors.tsv"), encoding: .utf8)
        var real: [String: (format: GGMLBlockFormat?, shape: [Int], bytes: UInt64)] = [:]
        for line in tsv.split(separator: "\n").dropFirst() {
            let f = line.split(separator: "\t", omittingEmptySubsequences: false).map(String.init)
            guard f.count >= 10, !f[1].hasSuffix("_exps.weight") else { continue }
            var ne = f[4...7].compactMap { Int($0) }
            while ne.count > 1, ne.last == 1 { ne.removeLast() }
            real[f[1]] = (GGMLBlockFormat(name: f[2]), Array(ne.reversed()), UInt64(f[9]) ?? 0)
        }
        let expected = Inkling2ResidentContract.expectedTensors(config: .inkling2Mini)
        #expect(real.count == 696)
        #expect(Set(real.keys) == Set(expected.map(\.name)))
        for t in expected {
            guard let r = real[t.name] else { continue }
            #expect(r.shape == t.shape, "\(t.name)")
            let format = try #require(r.format, "\(t.name)")
            #expect(t.formats == [format], "\(t.name)")
            #expect(UInt64(t.shape.dropLast().reduce(format.rowBytes(elements: t.shape.last!) ?? 0, *)) == r.bytes,
                    "\(t.name)")
        }
    }

    /// A sliding layer's bias table spans the window, a global layer's the
    /// relative extent (the toy and the real file have both equal).
    @Test func biasTableFollowsTheLayerKind() {
        var shape = Inkling2Shape()
        shape.window = 8
        let tensors = Inkling2ResidentContract.expectedTensors(config: shape.archConfig)
        let rows = Dictionary(uniqueKeysWithValues: tensors.map { ($0.name, $0.shape) })
        #expect(rows["blk.0.attn_rel_bias.weight"] == [8, shape.dRel])
        #expect(rows["blk.1.attn_rel_bias.weight"] == [shape.extent, shape.dRel])
    }

    @Test func residentRegionMustBeOneBuffer() throws {
        try Inkling2ResidentContract.validateOneBuffer(chunkCount: 1, residentSize: 1 << 30)
        #expect(throws: ModelError.self) {
            try Inkling2ResidentContract.validateOneBuffer(chunkCount: 2, residentSize: 1 << 34)
        }
    }
}
