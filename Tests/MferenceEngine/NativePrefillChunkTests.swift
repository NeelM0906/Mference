import Foundation
import Testing
import Mference
@testable import MferenceEngine

/// The prefill chunk a native model loads with through the engine, the path
/// `MferenceServer --library` and the app take. The app passes no load
/// options (`EngineService.live()` builds its `EngineHost` without them), so
/// the chunk is the server default for the family: every family but Inkling 2
/// as before, Inkling 2 2,048. An explicit `prefillChunkTokens` stays as given.
@Suite struct NativePrefillChunkTests {
    static let gib = UInt64(1) << 30
    static let memories: [UInt64] = [8 * gib, 16 * gib - 1, 16 * gib, 24 * gib, 32 * gib, 48 * gib, 128 * gib]

    static func before(_ family: ModelFamily, _ memory: UInt64) -> Int {
        guard memory >= 16 * gib else { return 128 }
        return family == .qwen36 ? 2_048 : 128
    }

    @Test func theAppsHostPassesNoChunk() {
        let host = EngineHost(catalog: EngineCatalog(models: []), backends: [NativeBackend()])
        #expect(host.loadOptions.prefillChunkTokens == nil)
        #expect(EngineLoadOptions().prefillChunkTokens == nil)
    }

    @Test func nativeLoadsResolveTheFamilyDefault() {
        for family in ModelFamily.allCases {
            for memory in Self.memories {
                let runtime = NativeBackend.runtimeConfiguration(
                    family: family, expertCacheSlots: 16, prefillChunkTokens: EngineLoadOptions().prefillChunkTokens,
                    physicalMemoryBytes: memory, environment: [:])
                let expected = family == .inkling2Loop ? 2_048 : Self.before(family, memory)
                #expect(runtime.prefillChunkTokens == expected, "\(family) at \(memory >> 30) GiB")
                #expect(runtime.prefillConfig.chunkTokens == expected, "\(family) at \(memory >> 30) GiB")
            }
            for size in RuntimeConfiguration.allowedPrefillChunkTokens {
                let runtime = NativeBackend.runtimeConfiguration(
                    family: family, expertCacheSlots: 16, prefillChunkTokens: size,
                    physicalMemoryBytes: 32 * Self.gib, environment: [:])
                #expect(runtime.prefillChunkTokens == size, "\(family) explicit \(size)")
            }
        }
    }
}
