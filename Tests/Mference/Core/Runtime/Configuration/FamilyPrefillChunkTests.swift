import Foundation
import Testing
@testable import Mference

/// The server's default prefill chunk (`--library`, `--model`, the app through
/// the engine, and the CLI's `--chat` all take it): every family but Inkling 2
/// resolves exactly what it did before the family chunk, pinned over host
/// sizes and the environment override; Inkling 2 takes its own, 2,048, on
/// every host. Then the chunks a prompt of each length prefills in.
@Suite struct FamilyPrefillChunkTests {
    static let gib = UInt64(1) << 30
    static let memories: [UInt64] = [8 * gib, 16 * gib - 1, 16 * gib, 24 * gib, 32 * gib, 36 * gib, 48 * gib,
                                     128 * gib]
    static let lengths = [16, 512, 2_048, 4_096, 5_000, 20_000]

    /// The default before the family chunk, as `defaultServerPrefillChunkTokens`
    /// wrote it at trunk 98a092c1 (no environment override).
    static func before(_ family: ModelFamily, _ memory: UInt64) -> Int {
        guard memory >= 16 * gib else { return 128 }
        return family == .qwen36 ? 2_048 : 128
    }

    @Test func existingFamiliesKeepTheirDefault() {
        for family in ModelFamily.allCases where family != .inkling2Loop {
            for memory in Self.memories {
                #expect(RuntimeConfiguration.defaultServerPrefillChunkTokens(
                    for: family, physicalMemoryBytes: memory, environment: [:]) == Self.before(family, memory),
                        "\(family) at \(memory >> 20) MiB")
            }
        }
    }

    /// A family's own chunk must be an allowed size: the server default
    /// returns it as is, and `RuntimeConfiguration` refuses any other.
    @Test func everyFamilyChunkIsAnAllowedSize() {
        for family in ModelFamily.allCases {
            if let chunk = RuntimeConfiguration.familyPrefillChunkTokens(for: family) {
                #expect(RuntimeConfiguration.allowedPrefillChunkTokens.contains(chunk), "\(family) \(chunk)")
            }
        }
    }

    @Test func inkling2TakesItsOwnChunkOnEveryHost() {
        for memory in Self.memories {
            #expect(RuntimeConfiguration.defaultServerPrefillChunkTokens(
                for: .inkling2Loop, physicalMemoryBytes: memory, environment: [:]) == 2_048, "\(memory >> 20) MiB")
        }
    }

    /// `MFERENCE_SERVER_PREFILL_CHUNK` still wins for every family, Inkling 2
    /// included, with any allowed size (4,096 too); a value it does not allow
    /// leaves the default.
    @Test func theEnvironmentOverrideStillWins() {
        for family in ModelFamily.allCases {
            for size in RuntimeConfiguration.allowedPrefillChunkTokens {
                #expect(RuntimeConfiguration.defaultServerPrefillChunkTokens(
                    for: family, physicalMemoryBytes: 32 * Self.gib,
                    environment: ["MFERENCE_SERVER_PREFILL_CHUNK": String(size)]) == size, "\(family) \(size)")
            }
            let fallback = family == .inkling2Loop ? 2_048 : Self.before(family, 32 * Self.gib)
            for bad in ["1000", "auto", "", "8192"] {
                #expect(RuntimeConfiguration.defaultServerPrefillChunkTokens(
                    for: family, physicalMemoryBytes: 32 * Self.gib,
                    environment: ["MFERENCE_SERVER_PREFILL_CHUNK": bad]) == fallback, "\(family) \(bad)")
            }
        }
    }

    /// A prompt prefills in chunks of the resolved size and a last, shorter
    /// one (every family but Inkling 2, through `PrefillChunkPlanner.spans`),
    /// or, for Inkling 2, in the same number of chunks of near-equal size
    /// (`Inkling2ForwardRunner.prefillSpans`). A prompt shorter than the chunk
    /// is one chunk of its own length, never padded to the chunk. Pinned for
    /// each family's server default on a 32 GB Mac.
    @Test func promptsPrefillInChunksOfTheResolvedSize() {
        for family in ModelFamily.allCases where family != .inkling2Loop {
            let chunk = RuntimeConfiguration.defaultServerPrefillChunkTokens(
                for: family, physicalMemoryBytes: 32 * Self.gib, environment: [:])
            for n in Self.lengths {
                let spans = PrefillChunkPlanner.spans(tokenCount: n, startPosition: 0, chunkTokens: chunk)
                let expected = Array(repeating: chunk, count: n / chunk) + (n % chunk == 0 ? [] : [n % chunk])
                #expect(spans.map(\.tokenCount) == expected, "\(family) \(n)")
                #expect(spans.map(\.startPosition) == spans.map(\.tokenOffset))
                #expect(spans.last?.completedCount == n)
            }
        }
        let chunk = RuntimeConfiguration.defaultServerPrefillChunkTokens(
            for: .inkling2Loop, physicalMemoryBytes: 32 * Self.gib, environment: [:])
        let inkling = Self.lengths.map {
            Inkling2ForwardRunner.prefillSpans(tokenCount: $0, startPosition: 0, chunkTokens: chunk).map(\.tokenCount)
        }
        #expect(inkling == [[16], [512], [2_048], [2_048, 2_048], [1_667, 1_667, 1_666],
                            Array(repeating: 2_000, count: 10)])
    }
}
