import Testing
import Mference
@testable import MferenceServerCore

@Suite("Server KV reserve") struct ServerKVReserveArgumentTests {
    private static let sixteenGiB = UInt64(16) << 30

    @Test func kvGrowsUnlessReserveIsGiven() throws {
        #expect(try !ServerArguments.parse(["--model", "m.gturbo"]).reserveFullKV)
        #expect(try ServerArguments.parse(["--model", "m.gturbo", "--kv-reserve"]).reserveFullKV)
        #expect(try ServerArguments.parse(["--library", "--kv-reserve", "--port", "9000"]).reserveFullKV)
        #expect(ServerArguments.usage.contains("--kv-reserve"))
    }

    @Test func reserveTakesNoValue() {
        #expect(throws: ServerArgumentError.self) {
            _ = try ServerArguments.parse(["--model", "m.gturbo", "--kv-reserve", "4096"])
        }
    }

    @Test func theChoiceReachesTheRuntime() {
        let growing = ServerModelSession.runtimeConfiguration(
            family: .gemma4, expertCacheSlots: 16, reserveFullKV: false,
            physicalMemoryBytes: Self.sixteenGiB, environment: [:])
        #expect(growing.kvGrowthTokens == RuntimeConfiguration.defaultKVGrowthTokens)
        #expect(RuntimeConfiguration.defaultKVGrowthTokens == 16_384)
        let reserved = ServerModelSession.runtimeConfiguration(
            family: .gemma4, expertCacheSlots: 16, reserveFullKV: true,
            physicalMemoryBytes: Self.sixteenGiB, environment: [:])
        #expect(reserved.kvGrowthTokens == nil)
    }
}
