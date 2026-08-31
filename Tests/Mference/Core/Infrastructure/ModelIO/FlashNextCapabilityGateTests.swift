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
    @Test func gateTableCoversOnlyFamiliesWithoutARunner() {
        for (raw, axes) in ManifestReader.familiesWithoutRunner {
            #expect(!axes.isEmpty, "\(raw) must name the axes it is missing")
            if let family = ModelFamily(rawValue: raw) {
                #expect(ArchConfig.knownArchitectures[family] == nil,
                        "\(raw) has a baseline; remove it from the gate table")
            }
        }
    }
}
