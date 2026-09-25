import Testing
import Mference
@testable import MferenceServerCore

@Suite("Server shadow prefetch budget") struct ServerShadowBudgetArgumentTests {
    @Test func theBudgetIsUnsetUnlessGiven() throws {
        #expect(try ServerArguments.parse(["--model", "m.gturbo"]).shadowBudget == nil)
        #expect(try ServerArguments.parse(["--library"]).shadowBudget == nil)
    }

    @Test func theBudgetCanBeSetOrTurnedOff() throws {
        #expect(try ServerArguments.parse(["--model", "m.gturbo", "--shadow-budget", "4"]).shadowBudget == 4)
        #expect(try ServerArguments.parse(["--library", "--shadow-budget", "0"]).shadowBudget == 0)
    }

    @Test(arguments: ["9", "-1", "two", ""])
    func valuesOutsideZeroToEightAreRejected(value: String) {
        #expect(throws: ServerArgumentError.self) {
            _ = try ServerArguments.parse(["--model", "m.gturbo", "--shadow-budget", value])
        }
    }

    @Test func theBudgetReachesTheRuntimeConfiguration() {
        let sixteenGiB = UInt64(16) << 30
        #expect(ServerModelSession.runtimeConfiguration(
            family: .qwen36, expertCacheSlots: 32, shadowPrefetchBudget: 3,
            physicalMemoryBytes: sixteenGiB, environment: [:]).shadowPrefetchBudget == 3)
        #expect(ServerModelSession.runtimeConfiguration(
            family: .qwen36, expertCacheSlots: 32,
            physicalMemoryBytes: sixteenGiB, environment: [:]).shadowPrefetchBudget == nil)
    }
}
