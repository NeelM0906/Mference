import Testing
import Mference
@testable import MferenceServerCore

@Suite struct ServerRuntimeConfigurationTests {
    private static let sixteenGiB = UInt64(16) << 30

    @Test func gemmaSessionsPrefillLongPromptsInLargerChunks() {
        let runtime = ServerModelSession.runtimeConfiguration(
            family: .gemma4, expertCacheSlots: 16,
            physicalMemoryBytes: Self.sixteenGiB, environment: [:])
        #expect(runtime.prefillChunkTokens == 1024)
        #expect(runtime.expertCacheSlots == 16)
        #expect(runtime.headPath == .logits)
    }

    @Test func qwen36SessionsPrefillLongPromptsInLargerChunks() {
        let runtime = ServerModelSession.runtimeConfiguration(
            family: .qwen36, expertCacheSlots: 32,
            physicalMemoryBytes: Self.sixteenGiB, environment: [:])
        #expect(runtime.prefillChunkTokens == 2048)
        #expect(runtime.expertCacheSlots == 32)
    }

    @Test func otherFamiliesKeepTheEstablishedChunk() {
        let runtime = ServerModelSession.runtimeConfiguration(
            family: .inklingSmall, expertCacheSlots: 16,
            physicalMemoryBytes: Self.sixteenGiB, environment: [:])
        #expect(runtime.prefillChunkTokens == 128)
        #expect(runtime.expertCacheSlots == 16)
    }

    @Test func operatorsCanOverrideTheChunk() {
        let runtime = ServerModelSession.runtimeConfiguration(
            family: .gemma4, expertCacheSlots: 16, physicalMemoryBytes: Self.sixteenGiB,
            environment: ["MFERENCE_SERVER_PREFILL_CHUNK": "128"])
        #expect(runtime.prefillChunkTokens == 128)
    }
}
