import Foundation
import Metal
import Testing

@testable import Mference

/// Cross-implementation parity for the DFlash2 drafter against the z-lab
/// MLX reference run in FLOAT32 (`gen_dflash2_fixtures.py` +
/// `gen_dflash2_f32ref.py` dump the fixtures). The fp32 reference is the
/// comparison target because the bf16 reference disagrees with itself-in-
/// fp32 by the same magnitude as any correct fp16/fp32 port would (cosine
/// ~0.992-0.999 on these random fixtures); matching fp32 tightly is the
/// discriminating test. Gated on the real drafter checkpoint plus the
/// fixture directory under `scratch/`; draft quality only moves acceptance
/// length, never output bytes.
@Suite(.serialized) struct Qwen38DFlash2DrafterParityTests {

    private static var repoRoot: URL {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        return root
    }
    private static var drafterDir: URL {
        repoRoot.appendingPathComponent("scratch/dflash2-qwen38")
    }
    private static var fixtureDir: URL {
        repoRoot.appendingPathComponent("scratch/dflash2-fixtures")
    }
    private static var available: Bool {
        FileManager.default.fileExists(
            atPath: drafterDir.appendingPathComponent("model.safetensors").path)
            && FileManager.default.fileExists(
                atPath: fixtureDir.appendingPathComponent("f1_out.f32").path)
    }

    private static func floats(_ name: String) throws -> [Float] {
        let data = try Data(contentsOf: fixtureDir.appendingPathComponent(name))
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    private static func u32(_ name: String) throws -> [UInt32] {
        let data = try Data(contentsOf: fixtureDir.appendingPathComponent(name))
        return data.withUnsafeBytes { Array($0.bindMemory(to: UInt32.self)) }
    }

    private static func makeDrafter(_ context: MetalContext) throws -> Qwen38DFlash2Drafter {
        let cfg = try #require(ArchConfig.knownArchitectures[.qwen38])
        // Embedding / LM head are bypassed by the parity seams; a one-element
        // placeholder view keeps the initializer honest about not touching them.
        let dummy = context.device.makeBuffer(length: 64, options: .storageModeShared)!
        let view = TensorView(buffer: dummy, offset: 0, length: 64,
                              scaleOffset: 0, scaleLength: 0,
                              biasOffset: 0, biasLength: 0,
                              shape: (0, 0, 0, 0), dtype: 0)
        return try Qwen38DFlash2Drafter(context: context,
                                        directory: Self.drafterDir,
                                        embedding: view,
                                        lmHead: view,
                                        targetConfig: cfg,
                                        precision: .bf16)
    }

    private static func stageTaps(_ drafter: Qwen38DFlash2Drafter,
                                  rows: Int, values: [Float]) {
        let width = drafter.tapOrdinal.count * drafter.config.hidden_size
        precondition(values.count == rows * width)
        let dst = drafter.tapStaging.contents()
            .bindMemory(to: Float16.self, capacity: rows * width)
        for i in 0..<values.count { dst[i] = Float16(values[i]) }
        drafter.commitTapRows(rows)
    }

    private static func stageBlock(_ drafter: Qwen38DFlash2Drafter,
                                   rows: Int, values: [Float]) {
        let d = drafter.config.hidden_size
        precondition(values.count == rows * d)
        let dst = drafter.parityBlockHidden.contents()
            .bindMemory(to: Float.self, capacity: rows * d)
        for i in 0..<values.count { dst[i] = values[i] }
    }

    private static func compare(_ drafter: Qwen38DFlash2Drafter,
                                rows: Int, reference: [Float],
                                label: String) {
        let d = drafter.config.hidden_size
        let got = drafter.parityNormedOut.contents()
            .bindMemory(to: Float16.self, capacity: rows * d)
        for row in 0..<rows {
            var dot: Double = 0, na: Double = 0, nb: Double = 0
            var maxAbs: Double = 0
            for i in 0..<d {
                let a = Double(got[row * d + i])
                let b = Double(reference[row * d + i])
                dot += a * b
                na += a * a
                nb += b * b
                maxAbs = max(maxAbs, abs(a - b))
            }
            let cosine = dot / max(1e-12, (na * nb).squareRoot())
            #expect(cosine > 0.999,
                    "\(label) row \(row) cosine \(cosine)")
            #expect(maxAbs < 0.35,
                    "\(label) row \(row) max abs diff \(maxAbs)")
        }
    }

    @Test func coreForwardMatchesReferenceAcrossTwoRounds() throws {
        guard Self.available else { return }
        let context = try MetalContext()
        let drafter = try Self.makeDrafter(context)

        Self.stageTaps(drafter, rows: 3, values: try Self.floats("f1_ctx.f32"))
        Self.stageBlock(drafter, rows: 8, values: try Self.floats("f1_block.f32"))
        try drafter.runCoreForParity(blockTokens: 8)
        Self.compare(drafter, rows: 8, reference: try Self.floats("f1_out_fp32.f32"),
                     label: "round1")
        #expect(drafter.ctxKept == 3)
        #expect(drafter.ctxTotal == 3)

        Self.stageTaps(drafter, rows: 2, values: try Self.floats("f2_ctx.f32"))
        Self.stageBlock(drafter, rows: 8, values: try Self.floats("f2_block.f32"))
        try drafter.runCoreForParity(blockTokens: 8)
        Self.compare(drafter, rows: 8, reference: try Self.floats("f2_out_fp32.f32"),
                     label: "round2")
        #expect(drafter.ctxKept == 5)
        #expect(drafter.ctxTotal == 5)
    }

    @Test func selectorMatchesReferencePath() throws {
        guard Self.available else { return }
        let context = try MetalContext()
        let drafter = try Self.makeDrafter(context)

        let hidden = try Self.floats("sel_hidden.f32").map { Float16($0) }
        let logits = try Self.floats("sel_logits.f32").map { Float16($0) }
        let expectedPath = try Self.u32("sel_path_fp32.u32")
        let expectedCandidates = try Self.u32("sel_candidates_fp32.u32")
        let meta = try JSONSerialization.jsonObject(
            with: Data(contentsOf: Self.fixtureDir.appendingPathComponent("meta.json")))
            as! [String: Any]
        let anchor = Int32(meta["anchor"] as! Int)

        let result = try drafter.selectorParityRun(hidden: hidden,
                                                   logits: logits,
                                                   anchor: anchor, rows: 7)
        let vocab = try #require(ArchConfig.knownArchitectures[.qwen38]).vocabSize
        for position in 0..<7 {
            let expectedSet = Set(expectedCandidates[position * 16..<(position + 1) * 16])
            let gotSet = Set(result.candidates[position])
            // The 16th place can tie in the bf16-rounded fixture logits;
            // either side of an exact tie is a valid top-16. Any element in
            // the symmetric difference must sit exactly at the boundary value.
            let disputed = expectedSet.symmetricDifference(gotSet)
            if !disputed.isEmpty {
                let row = logits[position * vocab..<(position + 1) * vocab]
                let boundary = expectedSet.map { row[position * vocab + Int($0)] }.min()!
                for token in disputed {
                    #expect(row[position * vocab + Int(token)] == boundary,
                            "non-tied candidate mismatch at position \(position): \(token)")
                }
            }
        }
        #expect(result.path.map { UInt32($0) } == expectedPath)
    }
}
