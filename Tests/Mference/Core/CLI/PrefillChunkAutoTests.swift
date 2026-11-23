import Foundation
import Testing
@testable import Mference
@testable import MferenceCLICore

/// The CLI's prefill chunk: one-shot `--prefill-chunk auto` (the path the
/// byte-identity runs take) and `--chat`'s. Every family but Inkling 2
/// resolves exactly what it did before the family chunk; Inkling 2 takes its
/// own, 2,048 (at most, one-shot). A fixed size is used as given.
@Suite struct PrefillChunkAutoTests {
    static let lengths = [16, 512, 2_048, 4_096, 5_000, 20_000]

    /// The rule before the family cap, as `Run.swift` wrote it at trunk
    /// 98a092c1: the smallest allowed chunk covering the prompt, else 4,096.
    static func before(_ promptTokens: Int) -> Int {
        RuntimeConfiguration.allowedPrefillChunkTokens.first(where: { $0 >= promptTokens })
            ?? RuntimeConfiguration.allowedPrefillChunkTokens.last!
    }

    @Test func existingFamiliesResolveExactlyAsBefore() {
        let expected = [32, 512, 2_048, 4_096, 4_096, 4_096]
        #expect(Self.lengths.map(Self.before) == expected)
        for family in ModelFamily.allCases where family != .inkling2Loop {
            #expect(RuntimeConfiguration.familyPrefillChunkTokens(for: family) == nil, "\(family)")
            let resolved = Self.lengths.map {
                RuntimeConfiguration.autoPrefillChunkTokens(promptTokens: $0, family: family)
            }
            #expect(resolved == expected, "\(family)")
        }
        // A family the CLI could not read before the load: as before.
        #expect(Self.lengths.map { RuntimeConfiguration.autoPrefillChunkTokens(promptTokens: $0, family: nil) }
                == expected)
        // Every prompt length up to past the largest chunk, not just the six.
        for n in 1...4_200 {
            for family in ModelFamily.allCases where family != .inkling2Loop {
                if RuntimeConfiguration.autoPrefillChunkTokens(promptTokens: n, family: family) != Self.before(n) {
                    Issue.record("\(family) at \(n) tokens")
                }
            }
        }
    }

    @Test func inkling2StopsAtTwoThousandFortyEight() {
        #expect(RuntimeConfiguration.familyPrefillChunkTokens(for: .inkling2Loop) == 2_048)
        let resolved = Self.lengths.map {
            RuntimeConfiguration.autoPrefillChunkTokens(promptTokens: $0, family: .inkling2Loop)
        }
        #expect(resolved == [32, 512, 2_048, 2_048, 2_048, 2_048])
        for n in 1...4_200 {
            if RuntimeConfiguration.autoPrefillChunkTokens(promptTokens: n, family: .inkling2Loop)
                != min(Self.before(n), 2_048) {
                Issue.record("inkling2Loop at \(n) tokens")
            }
        }
    }

    /// `--chat` has no prompt at load, so `auto` takes the server's default
    /// for the family (without the server's environment override): every
    /// family but Inkling 2 as before (Qwen 3.6 2,048 from 16 GiB, 128 below
    /// and for the others), Inkling 2 2,048 on every host.
    @Test func chatTakesTheFamilyDefault() {
        let gib = UInt64(1) << 30
        for memory in [8 * gib, 16 * gib - 1, 16 * gib, 24 * gib, 32 * gib, 48 * gib, 128 * gib] {
            for family in ModelFamily.allCases {
                let expected = switch family {
                case .inkling2Loop: 2_048
                case .qwen36: memory >= 16 * gib ? 2_048 : 128
                case .qwen38, .maple, .minicpm5: 128
                }
                #expect(PrefillChunkChoice.auto.chatChunkTokens(for: family, physicalMemoryBytes: memory) == expected,
                        "\(family) at \(memory >> 20) MiB")
                #expect(PrefillChunkChoice.fixed(4_096).chatChunkTokens(for: family, physicalMemoryBytes: memory)
                        == 4_096)
            }
        }
    }

    /// Through `PrefillChunkChoice` with the family read from each install's
    /// manifest, as `run` resolves it; an explicit size stays as given.
    @Test func throughTheManifest() throws {
        for family in ModelFamily.allCases {
            let directory = try Self.writeManifest(family: family)
            defer { try? FileManager.default.removeItem(at: directory) }
            #expect(try ManifestReader.peekFamily(directoryURL: directory) == family)
            let resolved = Self.lengths.map {
                PrefillChunkChoice.auto.rawChunkTokens(promptTokens: $0, modelDirectory: directory)
            }
            let expected = family == .inkling2Loop
                ? [32, 512, 2_048, 2_048, 2_048, 2_048] : [32, 512, 2_048, 4_096, 4_096, 4_096]
            #expect(resolved == expected, "\(family)")
            for size in RuntimeConfiguration.allowedPrefillChunkTokens {
                #expect(PrefillChunkChoice.fixed(size).rawChunkTokens(promptTokens: 4_096, modelDirectory: directory)
                        == size, "\(family) --prefill-chunk \(size)")
            }
        }
        // No readable manifest: bounded as before; the load reports the error.
        let missing = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mference-chunk-missing-\(UUID().uuidString)")
        #expect(Self.lengths.map { PrefillChunkChoice.auto.rawChunkTokens(promptTokens: $0, modelDirectory: missing) }
                == [32, 512, 2_048, 4_096, 4_096, 4_096])
    }

    /// A manifest `peekFamily` decodes, naming `family`. It validates nothing
    /// beyond decodability, so one placeholder shape serves every family.
    static func writeManifest(family: ModelFamily) throws -> URL {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mference-chunk-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let manifest: [String: Any] = [
            "magic": "GTURBO", "versionMajor": 1, "versionMinor": 0,
            "flags": ["streamingPresent": true, "turboQuantKV": false, "aneSharedExpert": false],
            "modelID": "chunk-fixture-\(family.rawValue)",
            "sourceSnapshotHash": "sha256:0",
            "arch": [
                "family": family.rawValue,
                "hiddenSize": 2_048, "ffnIntermediate": 512, "moeIntermediateSize": 512,
                "numHeads": 16, "numKVHeads": 2, "numFullKVHeads": 2, "headDim": 128, "fullHeadDim": 128,
                "vocabSize": 1_024, "slidingWindow": 0, "finalLogitSoftcap": 0.0,
                "ropeTheta": 10_000.0, "fullRopeTheta": 10_000.0, "partialRotaryFactor": 1.0,
                "numLayers": 4, "numExperts": 8, "topKExperts": 2, "tieWordEmbeddings": false,
                "attentionKEqV": false, "hiddenActivation": "silu", "fullAttentionLayerMask": [1, 1, 1, 1],
            ] as [String: Any],
            "files": [:] as [String: Any],
            "expertsPerLayer": 8, "numLayers": 4, "expertStride": 16_384,
        ]
        try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
            .write(to: directory.appendingPathComponent("manifest.json"))
        return directory
    }
}
