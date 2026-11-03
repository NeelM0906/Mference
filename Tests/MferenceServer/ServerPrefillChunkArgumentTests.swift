import Testing
import Mference
@testable import MferenceServerCore

@Suite("Server prefill chunk") struct ServerPrefillChunkArgumentTests {
    private static let sixteenGiB = UInt64(16) << 30

    @Test func theChunkIsAutoUnlessGiven() throws {
        #expect(try ServerArguments.parse(["--model", "m.gturbo"]).prefillChunk == nil)
        #expect(try ServerArguments.parse(["--library", "--prefill-chunk", "auto"]).prefillChunk == nil)
    }

    @Test func theChunkCanBeLowered() throws {
        #expect(try ServerArguments.parse(["--model", "m.gturbo", "--prefill-chunk", "1024"]).prefillChunk == 1024)
        #expect(try ServerArguments.parse(["--library", "--prefill-chunk", "128"]).prefillChunk == 128)
    }

    @Test(arguments: ["1000", "0", "8192", "large", ""])
    func unlistedSizesAreRejected(value: String) {
        #expect(throws: ServerArgumentError.self) {
            _ = try ServerArguments.parse(["--model", "m.gturbo", "--prefill-chunk", value])
        }
    }

    @Test func theFlagWinsOverTheEnvironmentAndTheFamilyDefault() {
        let flagged = ServerModelSession.runtimeConfiguration(
            family: .gemma4, expertCacheSlots: 16, prefillChunkTokens: 1024,
            physicalMemoryBytes: Self.sixteenGiB,
            environment: ["MFERENCE_SERVER_PREFILL_CHUNK": "512"])
        #expect(flagged.prefillChunkTokens == 1024)
        let environment = ServerModelSession.runtimeConfiguration(
            family: .gemma4, expertCacheSlots: 16,
            physicalMemoryBytes: Self.sixteenGiB,
            environment: ["MFERENCE_SERVER_PREFILL_CHUNK": "512"])
        #expect(environment.prefillChunkTokens == 512)
        let automatic = ServerModelSession.runtimeConfiguration(
            family: .gemma4, expertCacheSlots: 16,
            physicalMemoryBytes: Self.sixteenGiB, environment: [:])
        #expect(automatic.prefillChunkTokens == 2048)
    }
}
