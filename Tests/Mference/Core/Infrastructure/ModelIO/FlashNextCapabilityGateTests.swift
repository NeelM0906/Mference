import Foundation
import Testing
@testable import Mference

/// `MferenceRepack` can install `qwen38flashnext` today; no runner executes it.
/// Every load path funnels through `ManifestReader.peekFamily`, so the refusal
/// must happen there, name the family and name the axes whose kernels are
/// missing — never "unknown arch.family", and never a silent fall-through to
/// another family's runner.
@Suite struct FlashNextCapabilityGateTests {

    private static func writeManifest(family: String) throws -> URL {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mference-gate-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory,
                                                withIntermediateDirectories: true)
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
                "numLayers": 48,
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

    @Test func peekFamilyRefusesFlashNextByNameWithItsMissingAxes() throws {
        let directory = try Self.writeManifest(family: "qwen38flashnext")
        defer { try? FileManager.default.removeItem(at: directory) }

        var thrown: Error?
        #expect(throws: (any Error).self) {
            do { _ = try ManifestReader.peekFamily(directoryURL: directory) }
            catch { thrown = error; throw error }
        }
        let error = try #require(thrown as? ModelError)
        #expect(error == .familyRunnerNotImplemented(
            family: "qwen38flashnext",
            missingAxes: ["hyperConnectionsLowRank",
                          "attentionIndexer",
                          "pleNgramEmbedding"]))
        // The message has to be actionable on its own: it is what the CLI, the
        // server and the Mac app all surface.
        let text = error.description
        #expect(text.contains("qwen38flashnext"))
        #expect(text.contains("runner is not implemented"))
        #expect(text.contains("hyperConnectionsLowRank"))
        #expect(text.contains("attentionIndexer"))
        #expect(text.contains("pleNgramEmbedding"))
    }

    @Test func aGenuinelyUnknownFamilyStillReportsCorruption() throws {
        let directory = try Self.writeManifest(family: "not-a-family")
        defer { try? FileManager.default.removeItem(at: directory) }
        #expect(throws: ModelError.self) {
            _ = try ManifestReader.peekFamily(directoryURL: directory)
        }
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
        ]
        for family in shipped {
            #expect(ManifestReader.familiesWithoutRunner[family.rawValue] == nil,
                    "\(family.rawValue) has a runner; remove it from the gate table")
        }
    }

    /// The regression the runtime skeleton has to not cause: `qwen38flashnext`
    /// now has a compiled baseline (so its manifest can be validated and toy
    /// fixtures built), and adding it must not have made the family loadable.
    /// `Model.load`'s auto-detect path resolves the baseline only *after*
    /// `peekFamily`, so the refusal still comes first.
    @Test func aBaselineDoesNotMakeTheGatedFamilyLoadable() throws {
        #expect(ArchConfig.knownArchitectures[.qwen38flashnext] != nil,
                "the baseline the runner is being built against went missing")
        let directory = try Self.writeManifest(family: "qwen38flashnext")
        defer { try? FileManager.default.removeItem(at: directory) }
        #expect(throws: ModelError.familyRunnerNotImplemented(
            family: "qwen38flashnext",
            missingAxes: ["hyperConnectionsLowRank",
                          "attentionIndexer",
                          "pleNgramEmbedding"])) {
            _ = try ManifestReader.peekFamily(directoryURL: directory)
        }
    }
}
