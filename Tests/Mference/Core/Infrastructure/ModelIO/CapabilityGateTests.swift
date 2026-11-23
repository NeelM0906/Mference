import Foundation
import Testing
@testable import Mference

/// The capability gate and the retired-family refusal at
/// `ManifestReader.peekFamily`, the funnel every load path uses.
///
/// Invariants held for the *next* family port and for installs left on disk:
///   * the gate mechanism is intact: an injected table entry still produces the
///     named refusal (family plus the axes whose kernels are missing);
///   * a retired family is refused by name, not as corruption;
///   * a genuinely unknown `arch.family` is still reported as corruption.
@Suite struct CapabilityGateTests {

    /// A `manifest.json` naming `family` (nil omits the key), with the rest of
    /// `arch` fully shaped: a family that loads goes all the way through the
    /// strict `Manifest` decoder, so every required `ManifestArch` field has to
    /// be here or the fixture reports corruption instead of the family.
    /// `peekFamily` validates nothing beyond decodability; the values are a
    /// 48-layer hybrid shape and nothing checks them.
    private static func writeManifest(family: String?,
                                      modelID: String = "gate-fixture") throws -> URL {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mference-gate-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory,
                                                withIntermediateDirectories: true)
        var fullAttentionLayerMask = [Int](repeating: 0, count: 48)
        for i in stride(from: 3, to: 48, by: 4) { fullAttentionLayerMask[i] = 1 }
        let manifest: [String: Any] = [
            "magic": "GTURBO",
            "versionMajor": 1,
            "versionMinor": 0,
            "flags": ["streamingPresent": true,
                      "turboQuantKV": false,
                      "aneSharedExpert": false],
            "modelID": modelID,
            "sourceSnapshotHash": "sha256:0",
            "arch": [
                "hiddenSize": 2_560,
                "ffnIntermediate": 640,
                "moeIntermediateSize": 640,
                "numHeads": 24,
                "numKVHeads": 2,
                "numFullKVHeads": 2,
                "headDim": 256,
                "fullHeadDim": 256,
                "vocabSize": 248_320,
                "slidingWindow": 0,
                "finalLogitSoftcap": 0.0,
                "ropeTheta": 10_000_000.0,
                "fullRopeTheta": 10_000_000.0,
                "partialRotaryFactor": 0.25,
                "numLayers": 48,
                "numExperts": 512,
                "topKExperts": 10,
                "tieWordEmbeddings": false,
                "attentionKEqV": false,
                "hiddenActivation": "silu",
                "fullAttentionLayerMask": fullAttentionLayerMask,
            ].merging(family.map { ["family": $0] } ?? [:]) { _, new in new },
            "files": [:],
            "expertsPerLayer": 512,
            "numLayers": 48,
            "expertStride": 16_384,
        ]
        try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
            .write(to: directory.appendingPathComponent("manifest.json"))
        return directory
    }

    /// A kept family resolves at the funnel.
    @Test func peekFamilyResolvesAKeptFamily() throws {
        let directory = try Self.writeManifest(family: "qwen36")
        defer { try? FileManager.default.removeItem(at: directory) }
        #expect(try ManifestReader.peekFamily(directoryURL: directory) == .qwen36)
    }

    /// Every retired family, and a manifest without `arch.family` (the
    /// original Gemma 4 format), is refused by name. Their installs stay on
    /// disk, so the refusal has to say why rather than read as corruption.
    @Test func retiredFamiliesAreRefusedByName() throws {
        for raw in ManifestReader.retiredFamilies.sorted() {
            #expect(ModelFamily(rawValue: raw) == nil, "\(raw) is retired but still a case")
            let directory = try Self.writeManifest(family: raw)
            defer { try? FileManager.default.removeItem(at: directory) }
            #expect(throws: ModelError.retiredFamily(family: raw)) {
                _ = try ManifestReader.peekFamily(directoryURL: directory)
            }
        }
        let legacy = try Self.writeManifest(family: nil)
        defer { try? FileManager.default.removeItem(at: legacy) }
        #expect(throws: ModelError.retiredFamily(family: "gemma4")) {
            _ = try ManifestReader.peekFamily(directoryURL: legacy)
        }
        #expect(ModelError.retiredFamily(family: "flint").description
            .contains("RETIRED_FAMILIES.md"))
    }

    /// Swift-Qwen3.8 carries the kept `qwen38` family, so `peekFamily` must
    /// refuse it by checkpoint ID; otherwise it loads as base Qwen 3.8.
    @Test func retiredCheckpointOnAKeptFamilyIsRefusedById() throws {
        let swiftQwen = try Self.writeManifest(family: "qwen38", modelID: "swift-qwen3.8-27b-int4g64")
        defer { try? FileManager.default.removeItem(at: swiftQwen) }
        #expect(throws: ModelError.retiredCheckpoint(modelID: "swift-qwen3.8-27b-int4g64")) {
            _ = try ManifestReader.peekFamily(directoryURL: swiftQwen)
        }
        #expect(ModelError.retiredCheckpoint(modelID: "swift-qwen3.8-27b-int4g64").description
            .contains("Swift-Qwen3.8"))
        // The base checkpoint on the same family still resolves.
        let base = try Self.writeManifest(family: "qwen38", modelID: "qwen3.8-27b-4bit")
        defer { try? FileManager.default.removeItem(at: base) }
        #expect(try ManifestReader.peekFamily(directoryURL: base) == .qwen38)
    }

    /// The library classifier: retired checkpoint by ID, a family-less manifest
    /// under its modelID, a retired family by name, and nil for a kept install.
    @Test func retiredInstallClassification() throws {
        func classify(_ family: String?, _ modelID: String) throws -> ManifestReader.RetiredInstall? {
            let directory = try Self.writeManifest(family: family, modelID: modelID)
            defer { try? FileManager.default.removeItem(at: directory) }
            return ManifestReader.retiredInstall(
                manifestURL: directory.appendingPathComponent("manifest.json"))
        }
        #expect(try classify("qwen38", "swift-qwen3.8-27b-int4g64")?.identifier
                == "swift-qwen3.8-27b-int4g64")
        let legacy = try #require(try classify(nil, "gemma-4-26b-a4b-it-qat-q4_0-mlx-aligned"))
        #expect(legacy.identifier == "gemma-4-26b-a4b-it-qat-q4_0-mlx-aligned")
        #expect(legacy.detail.contains("no arch.family"))
        #expect(try classify("flint", "flint-40b")?.identifier == "flint")
        #expect(try classify("qwen36", "qwen3.6-35b-a3b") == nil)
        #expect(try classify("qwen38", "qwen3.8-27b-4bit") == nil)
        // A family-less manifest without a modelID is not an install of any
        // family: left to the caller's incomplete-install path.
        let bare = FileManager.default.temporaryDirectory
            .appendingPathComponent("gate-bare-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: bare) }
        try Data(#"{"arch":{}}"#.utf8).write(to: bare)
        #expect(ManifestReader.retiredInstall(manifestURL: bare) == nil)
    }

    /// An install without its tokenizer sidecar names what is missing rather
    /// than reporting a missing chat template.
    @Test func missingTokenizerSidecarIsNamed() async throws {
        let directory = try Self.writeManifest(family: "qwen36", modelID: "qwen3.6-35b-a3b")
        defer { try? FileManager.default.removeItem(at: directory) }
        do {
            _ = try await MFTokenizer.load(forModelDirectory: directory, environment: [:])
            Issue.record("loading without a tokenizer sidecar must throw")
        } catch let error as MFTokenizerError {
            guard case .missingTokenizerSidecar(let path) = error else {
                Issue.record("expected missingTokenizerSidecar, got \(error)")
                return
            }
            #expect(path.hasSuffix("/tokenizer"))
            #expect(error.description.contains("tokenizer.json"))
        }
    }

    /// The gate mechanism, exercised against an injected table because the
    /// shipped one is empty. This is the check that keeps the refusal path
    /// honest for the next family port: without it, a broken lookup would be
    /// invisible until someone gated a family and found it loading anyway.
    @Test func anInjectedEntryStillProducesTheNamedRefusal() throws {
        let axes = ["hyperConnectionsLowRank", "attentionIndexer",
                    "pleNgramEmbedding"]
        let table = ["a-future-family": axes]

        let refusal = try #require(
            ManifestReader.capabilityRefusal(family: "a-future-family", in: table))
        #expect(refusal == .familyRunnerNotImplemented(family: "a-future-family",
                                                       missingAxes: axes))
        // The message has to be actionable on its own: it is what the CLI and
        // the server both surface.
        let text = refusal.description
        #expect(text.contains("a-future-family"))
        #expect(text.contains("runner is not implemented"))
        for axis in axes { #expect(text.contains(axis)) }

        // A family absent from the table is not gated.
        #expect(ManifestReader.capabilityRefusal(family: "qwen36", in: table) == nil)
        #expect(ManifestReader.capabilityRefusal(family: "qwen36") == nil)
    }

    /// An unrecognized `arch.family` must not load. With the gate table empty
    /// this reaches the `ModelFamily(rawValue:)` lookup and must refuse there.
    @Test func aGenuinelyUnknownFamilyStillReportsCorruption() throws {
        let directory = try Self.writeManifest(family: "not-a-family")
        defer { try? FileManager.default.removeItem(at: directory) }

        var thrown: Error?
        #expect(throws: ModelError.self) {
            do { _ = try ManifestReader.peekFamily(directoryURL: directory) }
            catch { thrown = error; throw error }
        }
        let error = try #require(thrown as? ModelError)
        #expect(error == .indexCorrupt(detail: "unknown arch.family \"not-a-family\""))
    }

    /// The gate table is the authority, not the manifest, and it must not name
    /// a family the runtime already runs.
    ///
    /// "Has a runner" is not the same as "has a baseline". A gated family is
    /// expected to grow an `ArchConfig` baseline, a `ModelFamily` case and
    /// tensor accessors well before its kernels land — that is what the runner
    /// is written against. So the invariant checked here is behavioral: a
    /// family in the table must still be refused at the funnel, whatever
    /// machinery has been built for it.
    ///
    /// The table is empty today, so this loops zero times. It is kept rather
    /// than deleted because it is the end-to-end form of the invariant — the
    /// one that goes through a real `manifest.json` and `peekFamily` — and it
    /// starts asserting again the moment a new family is gated.
    /// `anInjectedEntryStillProducesTheNamedRefusal` covers the lookup itself
    /// in the meantime.
    @Test func gateTableCoversOnlyFamiliesWithoutARunner() throws {
        for (raw, axes) in ManifestReader.familiesWithoutRunner {
            #expect(!axes.isEmpty, "\(raw) must name the axes it is missing")
            let directory = try Self.writeManifest(family: raw)
            defer { try? FileManager.default.removeItem(at: directory) }
            var thrown: Error?
            #expect(throws: (any Error).self) {
                do { _ = try ManifestReader.peekFamily(directoryURL: directory) }
                catch { thrown = error; throw error }
            }
            #expect(thrown as? ModelError
                    == .familyRunnerNotImplemented(family: raw, missingAxes: axes),
                    "\(raw) is gated but peekFamily did not refuse it by name")
        }
    }

    /// The families the runtime does run must never appear in the table: an
    /// entry here disables a shipped family at every entry point.
    @Test func shippedFamiliesAreNotGated() {
        for family in ModelFamily.allCases where family != .inkling2Loop {
            #expect(ManifestReader.familiesWithoutRunner[family.rawValue] == nil,
                    "\(family.rawValue) has a runner; remove it from the gate table")
        }
    }

    /// Every family the runtime runs has its auto-detect baseline.
    @Test func everyFamilyHasABaseline() {
        for family in ModelFamily.allCases {
            #expect(ArchConfig.knownArchitectures[family] != nil, "\(family.rawValue)")
        }
    }
}
