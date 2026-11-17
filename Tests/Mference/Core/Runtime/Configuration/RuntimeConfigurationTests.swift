import Testing
@testable import Mference

@Suite struct RuntimeConfigurationTests {
    @Test func productionDefaultsAreStable() {
        let runtime = RuntimeConfiguration.production
        #expect(runtime.fp16RingEnabled)
        #expect(runtime.expertCacheSlots == 16)
        #expect(runtime.expertCachePolicy == .lfu)
        #expect(runtime.rdadvisePolicy == .off)
        #expect(!runtime.rdadviseEnabled)
        #expect(runtime.prefillPolicy == .chunked)
        #expect(runtime.prefillChunkTokens == 128)
        #expect(runtime.prefillAttentionPath == .fullTensorOps2DPreferred)
        #expect(runtime.headPath == .fusedRows)
        #expect(!runtime.useMapleFlashHead)
    }

    @Test func automaticExpertCacheSlotsAreQwenSpecific() {
        let eightGiB = UInt64(8) * 1024 * 1024 * 1024
        let twentyFourGiB = UInt64(24) * 1024 * 1024 * 1024
        #expect(RuntimeConfiguration.defaultExpertCacheSlots(
            for: .qwen36, physicalMemoryBytes: twentyFourGiB) == 96)
        let sixteenGiB = UInt64(16) * 1024 * 1024 * 1024
        #expect(RuntimeConfiguration.defaultExpertCacheSlots(
            for: .qwen36, physicalMemoryBytes: sixteenGiB) == 32)
        #expect(RuntimeConfiguration.defaultExpertCacheSlots(
            for: .qwen36, physicalMemoryBytes: eightGiB) == 16)
        #expect(RuntimeConfiguration.defaultExpertCacheSlots(
            for: .gemma4, physicalMemoryBytes: twentyFourGiB) == 16)
        #expect(RuntimeConfiguration.defaultExpertCacheSlots(
            for: .deepseekV4Flash, physicalMemoryBytes: twentyFourGiB) == 16)
        #expect(RuntimeConfiguration.defaultExpertCacheSlots(
            for: .inklingSmall, physicalMemoryBytes: twentyFourGiB) == 16)
    }

    @Test func serverPrefillChunkGrowsForGemmaOnSixteenGiBHosts() {
        let gib = UInt64(1) << 30
        #expect(RuntimeConfiguration.defaultServerPrefillChunkTokens(
            for: .gemma4, physicalMemoryBytes: 16 * gib, environment: [:]) == 2048)
        #expect(RuntimeConfiguration.defaultServerPrefillChunkTokens(
            for: .gemma4, physicalMemoryBytes: 24 * gib, environment: [:]) == 2048)
        #expect(RuntimeConfiguration.defaultServerPrefillChunkTokens(
            for: .gemma4, physicalMemoryBytes: 8 * gib, environment: [:]) == 128)
    }

    @Test func serverPrefillChunkGrowsForQwen36OnSixteenGiBHosts() {
        let gib = UInt64(1) << 30
        #expect(RuntimeConfiguration.defaultServerPrefillChunkTokens(
            for: .qwen36, physicalMemoryBytes: 16 * gib, environment: [:]) == 2048)
        #expect(RuntimeConfiguration.defaultServerPrefillChunkTokens(
            for: .qwen36, physicalMemoryBytes: 24 * gib, environment: [:]) == 2048)
        #expect(RuntimeConfiguration.defaultServerPrefillChunkTokens(
            for: .qwen36, physicalMemoryBytes: 8 * gib, environment: [:]) == 128)
    }

    @Test func serverPrefillChunkStaysEstablishedForUnmeasuredFamilies() {
        let gib = UInt64(1) << 30
        #expect(RuntimeConfiguration.defaultServerPrefillChunkTokens(
            for: .inklingSmall, physicalMemoryBytes: 24 * gib, environment: [:]) == 128)
        #expect(RuntimeConfiguration.defaultServerPrefillChunkTokens(
            for: .qwen38flashnext, physicalMemoryBytes: 24 * gib, environment: [:]) == 128)
    }

    /// The Unsloth GGUF Flash-Next install's server chunk follows RAM: 4,096
    /// at 32 GiB and up, 1,024 from 16 GiB, 128 below; vendor Flash-Next is
    /// unmeasured and keeps 128; the env override still wins.
    @Test func serverPrefillChunkForUnslothFlashNextFollowsRAM() {
        let gib = UInt64(1) << 30
        func chunk(_ family: ModelFamily, _ ram: UInt64, _ env: [String: String] = [:]) -> Int {
            RuntimeConfiguration.defaultServerPrefillChunkTokens(for: family, physicalMemoryBytes: ram, environment: env)
        }
        #expect(chunk(.qwen38flashnextudiq3xxs, 64 * gib) == 4096)
        #expect(chunk(.qwen38flashnextudiq3xxs, 32 * gib) == 4096)
        #expect(chunk(.qwen38flashnextudiq3xxs, 32 * gib - 1) == 1024)
        #expect(chunk(.qwen38flashnextudiq3xxs, 24 * gib) == 1024)
        #expect(chunk(.qwen38flashnextudiq3xxs, 16 * gib) == 1024)
        #expect(chunk(.qwen38flashnextudiq3xxs, 16 * gib - 1) == 128)
        #expect(chunk(.qwen38flashnextudiq3xxs, 8 * gib) == 128)
        #expect(chunk(.qwen38flashnext, 32 * gib) == 128)
        #expect(chunk(.qwen38flashnext, 64 * gib) == 128)
        #expect(chunk(.qwen38flashnextudiq3xxs, 32 * gib, ["MFERENCE_SERVER_PREFILL_CHUNK": "128"]) == 128)
        #expect(chunk(.qwen38flashnextudiq3xxs, 24 * gib, ["MFERENCE_SERVER_PREFILL_CHUNK": "2048"]) == 2048)
    }

    /// An override that is not an allowed size is ignored, so the Unsloth
    /// install falls back to its RAM tier rather than to 128.
    @Test func serverPrefillChunkInvalidOverrideFallsBackToUnslothTiers() {
        let gib = UInt64(1) << 30
        func chunk(_ ram: UInt64, _ value: String) -> Int {
            RuntimeConfiguration.defaultServerPrefillChunkTokens(
                for: .qwen38flashnextudiq3xxs, physicalMemoryBytes: ram,
                environment: ["MFERENCE_SERVER_PREFILL_CHUNK": value])
        }
        #expect(chunk(32 * gib, "1000") == 4096)
        #expect(chunk(32 * gib, "abc") == 4096)
        #expect(chunk(24 * gib, "1000") == 1024)
        #expect(chunk(8 * gib, "8192") == 128)
    }

    @Test func serverPrefillChunkOverrideAcceptsOnlyAllowedSizes() {
        let gib = UInt64(1) << 30
        #expect(RuntimeConfiguration.defaultServerPrefillChunkTokens(
            for: .gemma4, physicalMemoryBytes: 16 * gib,
            environment: ["MFERENCE_SERVER_PREFILL_CHUNK": "128"]) == 128)
        #expect(RuntimeConfiguration.defaultServerPrefillChunkTokens(
            for: .qwen36, physicalMemoryBytes: 16 * gib,
            environment: ["MFERENCE_SERVER_PREFILL_CHUNK": "512"]) == 512)
        #expect(RuntimeConfiguration.defaultServerPrefillChunkTokens(
            for: .gemma4, physicalMemoryBytes: 16 * gib,
            environment: ["MFERENCE_SERVER_PREFILL_CHUNK": "1000"]) == 2048)
    }

    @Test func shadowPrefetchBudgetIsUnsetUnlessRequested() {
        #expect(RuntimeConfiguration.production.shadowPrefetchBudget == nil)
        #expect(RuntimeConfiguration(shadowPrefetchBudget: 0).shadowPrefetchBudget == 0)
        #expect(RuntimeConfiguration(shadowPrefetchBudget: 4).shadowPrefetchBudget == 4)
    }

    @Test func retainedControlsReachTypedRuntime() {
        let runtime = RuntimeConfiguration(
            expertCacheSlots: 32,
            expertCachePolicy: .lru,
            rdadvisePolicy: .adaptive,
            prefillEnabled: false,
            prefillChunkTokens: 64,
            prefillAttentionPath: .causalTiled,
            forceLogitsHead: true,
            useMapleFlashHead: true)
        #expect(runtime.expertCacheSlots == 32)
        #expect(runtime.modelExpertCachePolicy == .lru)
        #expect(runtime.rdadviseEnabled)
        #expect(runtime.prefillConfig == .off)
        #expect(runtime.prefillAttentionPath == .causalTiled)
        #expect(runtime.headPath == .logits)
        #expect(runtime.useMapleFlashHead)
    }

    @Test(arguments: [32, 64, 128, 256, 512, 1024, 2048, 4096])
    func productionPrefillSupportsPublicChunkSizes(_ chunkTokens: Int) {
        let runtime = RuntimeConfiguration(prefillChunkTokens: chunkTokens)
        #expect(runtime.prefillConfig.mode == .chunked)
        #expect(runtime.prefillConfig.chunkTokens == chunkTokens)
    }
}
