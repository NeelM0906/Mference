import Foundation
import Metal
import Testing
@testable import MferenceRepackCore
@testable import Mference

/// The real Inkling 2 Mini install through `Model.load`: the baseline, the
/// expert layout and the resident contract all validate, and sampled tensors
/// and expert blobs are the GGUF's bytes. Env-gated (a model run: run it only
/// inside a lock hold): `MFERENCE_INKLING2_GTURBO` names the install and
/// `MFERENCE_INKLING2_GGUF` the pinned GGUF it was copied from.
@Suite(.serialized) struct Inkling2InstallRealModelTests {
    static let install = ProcessInfo.processInfo.environment["MFERENCE_INKLING2_GTURBO"]
    static let gguf = ProcessInfo.processInfo.environment["MFERENCE_INKLING2_GGUF"]

    @Test(.enabled(if: install != nil && gguf != nil))
    func realInstallLoadsAndHoldsTheGGUFBytes() throws {
        let directory = URL(fileURLWithPath: Self.install!)
        let ggufPath = Self.gguf!
        // The product funnel refuses the family by name until its runner
        // lands, then accepts it.
        if ManifestReader.familiesWithoutRunner[ModelFamily.inkling2Loop.rawValue] != nil {
            #expect(throws: ModelError.familyRunnerNotImplemented(
                family: "inkling2Loop", missingAxes: ManifestReader.inkling2RequiredAxes)) {
                _ = try ManifestReader.peekFamily(directoryURL: directory)
            }
        } else {
            #expect(try ManifestReader.peekFamily(directoryURL: directory) == .inkling2Loop)
        }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(directoryURL: directory, device: device, expecting: .inkling2Mini,
                                   integrityPolicy: .trustedReceiptWhenValid)
        #expect(model.integrityFallbackDetail == nil)
        #expect(model.modelID == "inkling-2-mini-ud-q4_k_xl")
        let layout = try model.inkling2ExpertLayout()
        #expect(layout.layers.prefix(20).allSatisfy { $0.format == .nvfp4 && $0.expertStride == 4_489_216 })
        #expect(layout.layers.suffix(4).allSatisfy { $0.format == .q8_0 && $0.expertStride == 8_470_528 })

        let fd = open(ggufPath, O_RDONLY)
        try #require(fd >= 0, "cannot open \(ggufPath)")
        defer { close(fd) }
        let size = (try FileManager.default.attributesOfItem(atPath: ggufPath)[.size] as! NSNumber).uint64Value
        let (header, _) = try LocalGGUFSource.readHeader(path: ggufPath, size: size)
        let byName = Dictionary(uniqueKeysWithValues: header.tensors.map { ($0.name, $0) })
        func ggufBytes(_ offset: UInt64, _ count: Int) throws -> [UInt8] {
            var bytes = [UInt8](repeating: 0, count: count)
            let read = bytes.withUnsafeMutableBytes { pread(fd, $0.baseAddress, count, off_t(offset)) }
            try #require(read == count, "short read at \(offset): \(read) of \(count)")
            return bytes
        }
        func viewBytes(_ view: TensorView, _ start: UInt64, _ count: Int) -> [UInt8] {
            Array(UnsafeRawBufferPointer(start: view.buffer.contents().advanced(by: Int(view.offset + start)),
                                         count: count))
        }
        // Resident tensors: first and last 4 KiB of a few, through the accessors.
        let samples: [(String, TensorView)] = [
            ("token_embd.weight", try model.inkling2Embedding),
            ("output.weight", try model.inkling2OutputHead),
            ("blk.11.attn_q.weight", try model.inkling2AttnQ(layer: 11)),
            ("blk.17.attn_rel_bias.weight", try model.inkling2AttnRelBias(layer: 17)),
            ("blk.6.loop_q_ctrl_down.weight", try model.inkling2LoopGateDown(layer: 6, .query)),
            ("blk.19.ffn_down_exps.scale", try model.inkling2ExpertDownScale(layer: 19)),
            ("blk.0.attn_sink_proj.weight", try model.inkling2AttnSinkProj(layer: 0)),
        ]
        for (name, view) in samples {
            let t = try #require(byName[name], "\(name)")
            #expect(view.length == t.sizeBytes, "\(name)")
            let n = Int(min(4096, t.sizeBytes))
            #expect(try viewBytes(view, 0, n) == ggufBytes(t.absoluteOffset, n), "\(name) head")
            #expect(try viewBytes(view, view.length - UInt64(n), n) == ggufBytes(t.absoluteOffset + t.sizeBytes - UInt64(n), n),
                    "\(name) tail")
        }
        // Expert blobs: gate-up head and down tail of experts in NVFP4 and Q8_0 layers.
        for (layer, expert) in [(0, 0), (7, 64), (19, 127), (20, 0), (23, 127)] {
            let blob = try model.routedExpert(layer: layer, expert: expert)
            let l = layout.layers[layer]
            let gateUp = try #require(byName["blk.\(layer).ffn_gate_up_exps.weight"])
            let down = try #require(byName["blk.\(layer).ffn_down_exps.weight"])
            #expect(try viewBytes(blob, l.gateUp.offset, 4096)
                    == ggufBytes(gateUp.absoluteOffset + UInt64(expert) * l.gateUp.size, 4096), "L\(layer) e\(expert) gate_up")
            #expect(try viewBytes(blob, l.down.offset + l.down.size - 4096, 4096)
                    == ggufBytes(down.absoluteOffset + UInt64(expert + 1) * l.down.size - 4096, 4096),
                    "L\(layer) e\(expert) down")
        }
    }
}
