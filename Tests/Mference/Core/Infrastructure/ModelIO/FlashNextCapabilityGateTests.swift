import Foundation
import Testing
@testable import Mference

/// The capability gate after the `qwen38flashnext` lift (2026-09-10, owner
/// decision): `FlashNextForwardRunner` executes this family, so
/// `ManifestReader.familiesWithoutRunner` no longer lists it and
/// `ManifestReader.peekFamily` — the funnel every load path uses — resolves it
/// like any shipped family.
///
/// Two invariants this suite still holds, because the gate has to work for the
/// *next* family port:
///   * the mechanism is intact — an injected table entry still produces the
///     named refusal (family plus the axes whose kernels are missing), rather
///     than "unknown arch.family" or a silent fall-through to another family's
///     runner;
///   * a genuinely unknown `arch.family` is still reported as corruption, which
///     the lift must not have turned into a load.
@Suite struct FlashNextCapabilityGateTests {

    /// A `manifest.json` naming `family`, with the rest of `arch` shaped like
    /// the real Qwen3.8-Flash-Next install.
    ///
    /// The full shape matters now that the gate is lifted: a gated family never
    /// reached the strict `Manifest` decoder, because `peekFamily` refused it
    /// off a minimal `arch.family`-only decode first. A family that loads goes
    /// all the way through, so every required `ManifestArch` field has to be
    /// here or the test fixture reports corruption instead of the family.
    /// `peekFamily` itself validates nothing beyond decodability — these values
    /// are the real baseline's so the fixture stays readable, not because
    /// anything checks them.
    private static func writeManifest(family: String) throws -> URL {
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
            "modelID": "qwen3.8-flash-next-int4g64",
            "sourceSnapshotHash": "sha256:0",
            "arch": [
                "family": family,
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
                "requiredAxes": ["hyperConnectionsLowRank",
                                 "attentionIndexer",
                                 "pleNgramEmbedding"],
            ],
            "files": [:],
            "expertsPerLayer": 512,
            "numLayers": 48,
            "expertStride": 16_384,
        ]
        try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
            .write(to: directory.appendingPathComponent("manifest.json"))
        return directory
    }

    /// The lift itself: the funnel resolves `qwen38flashnext` instead of
    /// refusing it, and it does so on a manifest that still publishes
    /// `arch.requiredAxes` — the install's advisory list does not get to gate a
    /// family the runtime can run.
    @Test func peekFamilyResolvesFlashNextNowThatItsRunnerLanded() throws {
        let directory = try Self.writeManifest(family: "qwen38flashnext")
        defer { try? FileManager.default.removeItem(at: directory) }

        #expect(try ManifestReader.peekFamily(directoryURL: directory)
                == .qwen38flashnext)
        #expect(ManifestReader.familiesWithoutRunner["qwen38flashnext"] == nil)
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

        // A family absent from the table is not gated — the same lookup that
        // now lets `qwen38flashnext` through.
        #expect(ManifestReader.capabilityRefusal(family: "qwen38flashnext",
                                                 in: table) == nil)
        #expect(ManifestReader.capabilityRefusal(family: "qwen38flashnext") == nil)
    }

    /// The lift must not have turned an unrecognized `arch.family` into a load.
    /// With the gate table empty this now reaches the `ModelFamily(rawValue:)`
    /// lookup — the intended path — and must still refuse there.
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
        let shipped: [ModelFamily] = [
            .gemma4, .qwen36, .qwen38, .deepseekV4Flash, .inklingSmall, .maple,
            .qwen38flashnext, .minicpm5, .glm53Flash,
        ]
        for family in shipped {
            #expect(ManifestReader.familiesWithoutRunner[family.rawValue] == nil,
                    "\(family.rawValue) has a runner; remove it from the gate table")
        }
    }

    /// Both halves of a loadable family, together: the auto-detect baseline
    /// `Model.load` resolves, and a funnel that no longer refuses before
    /// reaching it. `peekFamily` runs the gate check *first*, so with the entry
    /// gone the baseline is what decides — and it has to still be there.
    @Test func aBaselinePlusTheLiftedGateMakesTheFamilyLoadable() throws {
        #expect(ArchConfig.knownArchitectures[.qwen38flashnext] != nil,
                "the baseline FlashNextForwardRunner runs against went missing")
        let directory = try Self.writeManifest(family: "qwen38flashnext")
        defer { try? FileManager.default.removeItem(at: directory) }
        #expect(try ManifestReader.peekFamily(directoryURL: directory)
                == .qwen38flashnext)
    }
}
