import Foundation
import Metal
import Testing
@testable import Mference
@testable import MferenceRepackCore

/// The install leg of the reference-parity milestone: the toy checkpoint the
/// goldens were captured from becomes a real `.gturbo`, and the real runtime
/// loader opens it. Bytes flow reference checkpoint -> install -> loader.
///
/// Skips (rather than fails) when `scratch/qwen4exp-toy-ckpt-prodlayout/` is
/// absent: it is regenerable but not committed. See `Scripts/parity/README.md`.
@Suite struct FlashNextToyInstallTests {

    @Test func toyCheckpointInstallsAndLoadsThroughTheRealLoader() throws {
        try withKnownIssue("toy checkpoint not regenerated", isIntermittent: true) {
            try #require(FlashNextParity.checkpointIsPresent)
        } when: {
            !FlashNextParity.checkpointIsPresent
        }
        guard FlashNextParity.checkpointIsPresent else { return }
        let device = try #require(MTLCreateSystemDefaultDevice())

        let dir = try FlashNextParity.installToyCheckpoint()
        defer { try? FileManager.default.removeItem(at: dir) }

        // `Model.load(directoryURL:device:expecting:)` does not funnel through
        // `ManifestReader.peekFamily`, so the capability gate that refuses this
        // family at the auto-detect entry points is untouched: the reference
        // runner names its architecture instead of asking the manifest to.
        let model = try Model.load(directoryURL: dir,
                                   device: device,
                                   expecting: FlashNextParity.archConfig(),
                                   streamingMode: .resident)
        #expect(model.config.family == .qwen38flashnext)
        #expect(model.config.numLayers == 6)

        // The gate itself must still refuse the same directory by name.
        #expect(throws: ModelError.self) {
            _ = try ManifestReader.peekFamily(directoryURL: dir)
        }

        // Resident accessors resolve across all three tensor groups.
        #expect(try model.hcNorm(site: .attention, layer: 0).length == 256 * 2)
        #expect(try model.indexerQKProj(layer: 3).shape.0 == 24)
        #expect(try model.pleLayerMultipliers(layer: 1).count == 3)
        #expect(try model.pleNgramHeadVocabSizes(layer: 1)
            == [97, 101, 103, 107, 109, 113, 127, 131,
                137, 139, 149, 151, 157, 163, 167, 173])

        // The row pool addresses its own file and the head tables tile it.
        let pool = try model.openPleRowPool(layer: 1)
        #expect(pool.geometry.rows == 2176)
        #expect(pool.geometry.rowDim == 4)
        #expect(pool.geometry.ngramHeads == 16)
        #expect(try pool.readEmbedding(rows: Array(0..<16)).count == 64)

        // Routed experts resolve through the real streaming backend.
        let blob = try model.routedExpert(layer: 0, expert: 7)
        #expect(blob.length == model.packedExpertsLayout.expertStride)
    }

    /// The `(1 + w)` bake is a load-time transform for this family, and it must
    /// apply to the zero-centered norm set and to nothing else — in particular
    /// not to `linear_attn.norm`, which the reference initializes at ones.
    @Test func loadTimeNormBakeAppliesExactlyToTheZeroCenteredSet() throws {
        guard FlashNextParity.checkpointIsPresent else { return }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let dir = try FlashNextParity.installToyCheckpoint()
        defer { try? FileManager.default.removeItem(at: dir) }
        let model = try Model.load(directoryURL: dir, device: device,
                                   expecting: FlashNextParity.archConfig(),
                                   streamingMode: .resident)
        #expect(model.zeroCenteredNormPolicy == .bakeAtLoad)

        let source = try Safetensors(
            url: FlashNextParity.checkpointDirectory
                .appendingPathComponent("model.safetensors"))
        let trunk = FlashNextParity.trunk

        let baked = "\(trunk)layers.0.attn_hyper_connection.hc_norm.weight"
        let raw = try source.floats(baked)
        let loaded = FlashNextWeights.readBF16(try model.normWeight(name: baked))
        for i in 0..<raw.count {
            #expect(abs(loaded[i] - Quantization.bf16ToFloat(
                Quantization.bf16Bits(raw[i] + 1))) < 1e-9)
        }

        let gated = "\(trunk)layers.0.linear_attn.norm.weight"
        let gatedRaw = try source.floats(gated)
        let gatedLoaded = FlashNextWeights.readBF16(try model.normWeight(name: gated))
        #expect(gatedLoaded == gatedRaw)
    }
}
