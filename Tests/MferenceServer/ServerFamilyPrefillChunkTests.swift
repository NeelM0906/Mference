import Foundation
import Testing
import Mference
@testable import MferenceServerCore

/// The server's prefill chunk per family on both paths: `--model`
/// (`ServerModelSession.runtimeConfiguration`) and `--library`, whose load
/// options carry no chunk unless `--prefill-chunk` gives one (the engine then
/// resolves the same default; `NativePrefillChunkTests`). Every family but
/// Inkling 2 as before; Inkling 2 2,048 on every host; the flag, then the
/// environment, still win.
@Suite("Server prefill chunk per family") struct ServerFamilyPrefillChunkTests {
    static let gib = UInt64(1) << 30
    static let memories: [UInt64] = [8 * gib, 16 * gib - 1, 16 * gib, 24 * gib, 32 * gib, 48 * gib, 128 * gib]

    static func before(_ family: ModelFamily, _ memory: UInt64) -> Int {
        guard memory >= 16 * gib else { return 128 }
        return family == .qwen36 ? 2_048 : 128
    }

    @Test func modelModeResolvesTheFamilyDefault() {
        for family in ModelFamily.allCases {
            for memory in Self.memories {
                let runtime = ServerModelSession.runtimeConfiguration(
                    family: family, expertCacheSlots: 16, physicalMemoryBytes: memory, environment: [:])
                let expected = family == .inkling2Loop ? 2_048 : Self.before(family, memory)
                #expect(runtime.prefillChunkTokens == expected, "\(family) at \(memory >> 30) GiB")
            }
            let flagged = ServerModelSession.runtimeConfiguration(
                family: family, expertCacheSlots: 16, prefillChunkTokens: 4_096,
                physicalMemoryBytes: 32 * Self.gib, environment: ["MFERENCE_SERVER_PREFILL_CHUNK": "512"])
            #expect(flagged.prefillChunkTokens == 4_096, "\(family): the flag wins")
            let environment = ServerModelSession.runtimeConfiguration(
                family: family, expertCacheSlots: 16,
                physicalMemoryBytes: 32 * Self.gib, environment: ["MFERENCE_SERVER_PREFILL_CHUNK": "512"])
            #expect(environment.prefillChunkTokens == 512, "\(family): the environment wins over the default")
        }
    }

    @Test func libraryModePassesOnlyAGivenChunk() throws {
        #expect(try ServerArguments.parse(["--library"]).engineLoadOptions(environment: [:]).prefillChunkTokens
                == nil)
        #expect(try ServerArguments.parse(["--library", "--prefill-chunk", "auto"])
                    .engineLoadOptions(environment: [:]).prefillChunkTokens == nil)
        #expect(try ServerArguments.parse(["--library", "--prefill-chunk", "4096"])
                    .engineLoadOptions(environment: [:]).prefillChunkTokens == 4_096)
    }
}
