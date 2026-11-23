import Foundation
import Testing
@testable import MferenceRepackCore
@testable import Mference

/// The `inkling2Loop` family contract beyond the baseline values (those are
/// in `ModelTypesTests`): the loop's visit order, the capability gate, and the
/// manifest round trip from the repacker's mirror to the runtime's validation.
@Suite struct Inkling2ContractTests {

    /// Every other baseline leaves the new axes at `.none`, so the manifests
    /// that predate them keep validating unchanged.
    @Test func otherFamiliesCarryNoInkling2Axes() {
        for (family, config) in ArchConfig.knownArchitectures where family != .inkling2Loop {
            #expect(config.inkling2 == .none, "\(family.rawValue)")
        }
    }

    /// Execution order and cache slots match the reference: prelude, three
    /// core passes, coda; pass `p > 0` uses slots `24 + 12 (p - 1) + (L - 6)`.
    @Test func layerVisitsFollowTheReferenceOrder() {
        let ink = ArchConfig.inkling2Mini.inkling2
        let visits = ink.layerVisits(numLayers: 24)
        #expect(ink.layerVisitCount(numLayers: 24) == 48)
        #expect(visits.count == 48)
        #expect(visits.map(\.order) == Array(0..<48))
        #expect(Set(visits.map(\.cacheSlot)) == Set(0..<48))
        #expect(visits[5] == .init(order: 5, layer: 5, pass: 0, cacheSlot: 5))
        #expect(visits[6] == .init(order: 6, layer: 6, pass: 0, cacheSlot: 6))
        #expect(visits[18] == .init(order: 18, layer: 6, pass: 1, cacheSlot: 24))
        #expect(visits[29] == .init(order: 29, layer: 17, pass: 1, cacheSlot: 35))
        #expect(visits[30] == .init(order: 30, layer: 6, pass: 2, cacheSlot: 36))
        #expect(visits[41] == .init(order: 41, layer: 17, pass: 2, cacheSlot: 47))
        #expect(visits[42] == .init(order: 42, layer: 18, pass: 0, cacheSlot: 18))
        #expect(visits[47] == .init(order: 47, layer: 23, pass: 0, cacheSlot: 23))
        // Each core layer runs three times, every other layer once.
        for layer in 0..<24 {
            let runs = visits.filter { $0.layer == layer }.count
            #expect(runs == (ink.layerIsLooped(layer) ? 3 : 1), "layer \(layer)")
        }
        // Eight global visits: layers 5 and 23 once, 11 and 17 three times.
        let mask = ArchConfig.inkling2Mini.fullAttentionLayerMask
        #expect(visits.filter { mask[$0.layer] == 1 }.count == 8)
        // No loop: the stored layers in order.
        #expect(Inkling2Config.none.layerVisits(numLayers: 3).map(\.layer) == [0, 1, 2])
        #expect(Inkling2Config.none.layerVisitCount(numLayers: 3) == 3)
        #expect(!Inkling2Config.none.layerIsLooped(0))
        // One pass is no loop; bounds past the stored layers fall back too.
        func config(start: Int, end: Int, passes: Int) -> Inkling2Config {
            Inkling2Config(loopCoreStart: start, loopCoreEnd: end, loopNumPasses: passes,
                           loopGateRank: 0, attnResBlockSize: 0, qkClipThreshold: 0,
                           relClipThreshold: 0, attentionSinkProjected: false,
                           expertGateSoftcap: 0, expertUpSoftcap: 0, routedExpertFormats: [])
        }
        let onePass = config(start: 1, end: 3, passes: 1)
        #expect(!onePass.layerIsLooped(1))
        #expect(onePass.layerVisitCount(numLayers: 4) == 4)
        #expect(onePass.layerVisits(numLayers: 4).map(\.layer) == [0, 1, 2, 3])
        let outOfRange = config(start: 2, end: 9, passes: 3)
        #expect(outOfRange.layerVisitCount(numLayers: 4) == 4)
        #expect(outOfRange.layerVisits(numLayers: 4).count == 4)
    }

    /// The family is installable but gated until its runner lands; the
    /// retired Inkling-Small name is still refused as retired.
    @Test func familyIsGatedByAxisName() {
        #expect(ModelFamily(rawValue: "inkling2Loop") == .inkling2Loop)
        #expect(ManifestReader.familiesWithoutRunner["inkling2Loop"]
                == ManifestReader.inkling2RequiredAxes)
        #expect(ManifestReader.capabilityRefusal(family: "inkling2Loop")
                == .familyRunnerNotImplemented(family: "inkling2Loop",
                                               missingAxes: ManifestReader.inkling2RequiredAxes))
        #expect(ManifestReader.retiredFamilies.contains("inklingSmall"))
        #expect(ModelFamily(rawValue: "inklingSmall") == nil)
        // The installer publishes the runtime's list.
        #expect(GTurboJSON.inkling2RequiredAxes == ManifestReader.inkling2RequiredAxes)
    }

    // MARK: - Manifest round trip

    private static func encodedManifest() throws -> [String: Any] {
        let plan = RepackPlan(
            arch: .inkling2Mini, baseMode: "gguf", baseGroupSize: 0, bitsOverrideCount: 0,
            resident: ResidentFilePlan(path: "weights.bin", entries: [], stringTable: [],
                                       stringTableOffsets: [], indexSize: 0, residentSize: 0),
            layers: [], matchedModelID: nil, excludedMultimodalTensorNames: [], flashHead: nil)
        var files: [(relativePath: String, info: GTurboJSON.FileEntry)] = [
            ("model_weights.bin", .init(size: 16_384, sha256: String(repeating: "0", count: 64))),
            ("packed_experts/layout.json", .init(size: 16, sha256: String(repeating: "0", count: 64))),
        ]
        for L in 0..<24 {
            files.append((String(format: "packed_experts/layer_%02d.bin", L),
                          .init(size: 16_384, sha256: String(repeating: "0", count: 64))))
        }
        let data = try GTurboJSON.encodeManifest(
            plan: plan, modelID: "inkling-2-mini-ud-q4_k_xl", sourceSnapshotHash: "sha256:0",
            files: files, expertsPerLayer: 128, numLayers: 24, expertStride: 16_384,
            bitWidths: .init(embedding: 0, attention: 0, router: 0, sharedExpert: 0, routedExpert: 0))
        return try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private static func write(_ manifest: [String: Any]) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("inkling2-contract-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
            .write(to: dir.appendingPathComponent("manifest.json"))
        return dir
    }

    /// What the repacker writes for this family validates against the
    /// runtime's baseline field by field, quant slots included.
    @Test func repackerManifestValidatesAgainstTheBaseline() throws {
        let manifest = try Self.encodedManifest()
        let dir = try Self.write(manifest)
        defer { try? FileManager.default.removeItem(at: dir) }
        let loaded = try ManifestReader.load(directoryURL: dir, expecting: .inkling2Mini)
        #expect(loaded.arch.family == "inkling2Loop")
        #expect(loaded.arch.loopNumPasses == 3)
        #expect(loaded.arch.routedExpertFormats == ArchConfig.inkling2Mini.inkling2.routedExpertFormats)
        #expect(loaded.arch.requiredAxes == ManifestReader.inkling2RequiredAxes)
        #expect(loaded.quant?.routedExpert == ManifestReader.ggufQuantSlots["routedExpert"])
        #expect(loaded.quant?.embedding == ManifestReader.ggufQuantSlots["embedding"])
        // The f32-widened values survive the JSON round trip exactly.
        #expect(loaded.arch.qkClipThreshold == ArchConfig.inkling2Mini.inkling2.qkClipThreshold)
        #expect(loaded.arch.relLogScalingAlpha == ArchConfig.inkling2Mini.relativePosition.logScalingAlpha)
    }

    /// A key the family sets away from its default must be written; dropping
    /// it is a named mismatch, not a silent default.
    @Test func droppedInkling2KeyIsRefused() throws {
        for key in ["loopNumPasses", "routedExpertFormats", "relExtent", "sconvKernelSize",
                    "attentionSinks", "qkNorm", "routerGateBias"] {
            var manifest = try Self.encodedManifest()
            var arch = try #require(manifest["arch"] as? [String: Any])
            arch.removeValue(forKey: key)
            manifest["arch"] = arch
            let dir = try Self.write(manifest)
            defer { try? FileManager.default.removeItem(at: dir) }
            #expect(throws: ModelError.archMismatch(field: key, expected: "present", actual: "missing")) {
                _ = try ManifestReader.load(directoryURL: dir, expecting: .inkling2Mini)
            }
        }
    }

    @Test func changedInkling2AxisIsRefused() throws {
        let changes: [(String, Any)] = [
            ("loopNumPasses", 2), ("loopCoreEnd", 12), ("qkClipThreshold", 22.627416997969522),
            ("routedExpertFormats", [String](repeating: "NVFP4", count: 24)),
            ("hiddenActivation", "silu"), ("slidingWindow", 512),
        ]
        for (key, value) in changes {
            var manifest = try Self.encodedManifest()
            var arch = try #require(manifest["arch"] as? [String: Any])
            arch[key] = value
            manifest["arch"] = arch
            let dir = try Self.write(manifest)
            defer { try? FileManager.default.removeItem(at: dir) }
            var thrown: Error?
            #expect(throws: ModelError.self) {
                do { _ = try ManifestReader.load(directoryURL: dir, expecting: .inkling2Mini) }
                catch { thrown = error; throw error }
            }
            guard case .archMismatch(let field, _, _)? = thrown as? ModelError else {
                Issue.record("\(key): expected archMismatch, got \(String(describing: thrown))")
                continue
            }
            #expect(field == key)
        }
    }

    @Test func otherQuantContractIsRefused() throws {
        var manifest = try Self.encodedManifest()
        var quant = try #require(manifest["quant"] as? [String: Any])
        quant["routedExpert"] = ["weightBits": 4, "scheme": "affine", "scaleType": "BF16",
                                 "biasType": "BF16", "groupSize": 64]
        manifest["quant"] = quant
        let dir = try Self.write(manifest)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(throws: ModelError.indexCorrupt(detail: "unsupported Inkling 2 quantization for routedExpert")) {
            _ = try ManifestReader.load(directoryURL: dir, expecting: .inkling2Mini)
        }
    }

    /// The new keys are written for this family only: another family's
    /// manifest from the same encoder carries none of them.
    @Test func otherFamiliesEmitNoInkling2Keys() throws {
        let qwen = ArchInfo(
            hiddenSize: 64, intermediateSize: 64, moeIntermediateSize: 64, numHeads: 2,
            numKVHeads: 1, numFullKVHeads: 1, headDim: 32, fullHeadDim: 32, vocabSize: 128,
            slidingWindow: 0, finalLogitSoftcap: 0, ropeTheta: 1, fullRopeTheta: 1,
            partialRotaryFactor: 1, numLayers: 1, numExperts: 2, topKExperts: 1,
            tieWordEmbeddings: false, attentionKEqV: false, fullAttentionLayerMask: [1],
            hiddenActivation: "silu", family: .qwen36, attnOutputGate: true, attentionScale: 1,
            embeddingScaledBySqrtHidden: false, routerScaled: false, ffnSandwichNorms: false,
            sharedExpertGated: true, ropeNeoxSubdim: true, linearNumKHeads: 0, linearNumVHeads: 0,
            linearKeyHeadDim: 0, linearValueHeadDim: 0, linearConvKernelSize: 0)
        let plan = RepackPlan(
            arch: qwen, baseMode: "affine", baseGroupSize: 64, bitsOverrideCount: 0,
            resident: ResidentFilePlan(path: "w.bin", entries: [], stringTable: [],
                                       stringTableOffsets: [], indexSize: 0, residentSize: 0),
            layers: [], matchedModelID: nil, excludedMultimodalTensorNames: [], flashHead: nil)
        let data = try GTurboJSON.encodeManifest(
            plan: plan, modelID: "toy", sourceSnapshotHash: "sha256:0", files: [],
            expertsPerLayer: 2, numLayers: 1, expertStride: 16_384,
            bitWidths: .init(embedding: 4, attention: 4, router: 8, sharedExpert: 4, routedExpert: 4))
        let manifest = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let arch = try #require(manifest["arch"] as? [String: Any])
        for key in ["loopCoreStart", "loopNumPasses", "routedExpertFormats", "relDRel",
                    "sconvKernelSize", "attentionSinks", "requiredAxes", "qkClipThreshold"] {
            #expect(arch[key] == nil, "\(key) leaked into a qwen36 manifest")
        }
    }

    // MARK: - GGML block formats

    @Test func ggmlBlockFormatsMatchGGMLCommon() {
        #expect(GGMLBlockFormat.q8_0.blockElements == 32 && GGMLBlockFormat.q8_0.blockBytes == 34)
        #expect(GGMLBlockFormat.nvfp4.blockElements == 64 && GGMLBlockFormat.nvfp4.blockBytes == 36)
        #expect(GGMLBlockFormat.q8_0.residentDtype == 0x48)
        #expect(GGMLBlockFormat.nvfp4.residentDtype == 0x68)
        #expect(GGMLBlockFormat.f32.residentDtype == 3 && GGMLBlockFormat.bf16.residentDtype == 1)
        for format in GGMLBlockFormat.allCases {
            #expect(GGMLBlockFormat(residentDtype: format.residentDtype) == format)
            #expect(GGMLBlockFormat(name: format.name) == format)
        }
        #expect(GGMLBlockFormat(residentDtype: 0) == nil)
        // Expert rows: 2304 and 1152 values.
        #expect(GGMLBlockFormat.nvfp4.rowBytes(elements: 2304) == 1296)
        #expect(GGMLBlockFormat.nvfp4.rowBytes(elements: 1152) == 648)
        #expect(GGMLBlockFormat.q8_0.rowBytes(elements: 2304) == 2448)
        #expect(GGMLBlockFormat.q8_0.rowBytes(elements: 1152) == 1224)
        #expect(GGMLBlockFormat.nvfp4.rowBytes(elements: 100) == nil)
    }
}
