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
