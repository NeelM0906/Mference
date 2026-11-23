import Testing
import Mference
@testable import MferenceServerCore

@Suite("Server prompt cache arguments") struct ServerPromptCacheArgumentTests {
    private static let gibibyte = UInt64(1) << 30

    @Test func multiPrefixWithAnAutomaticBudgetIsTheDefault() throws {
        let arguments = try ServerArguments.parse(["--model", "m.gturbo"])
        #expect(arguments.promptCacheMode == .multiPrefix)
        #expect(arguments.promptCacheBytes == .auto)
        #expect(ServerArguments.usage.contains("--prompt-cache-bytes"))
        #expect(ServerArguments.usage.contains("multi-prefix"))
    }

    @Test func everyModeAndBudgetParses() throws {
        for mode in ["off", "single-prefix", "multi-prefix"] {
            let arguments = try ServerArguments.parse(["--model", "m.gturbo", "--prompt-cache-mode", mode])
            #expect(arguments.promptCacheMode.rawValue == mode)
        }
        #expect(try ServerArguments.parse(["--library", "--prompt-cache-bytes", "0"]).promptCacheBytes
            == .bytes(0))
        #expect(try ServerArguments.parse(["--model", "m.gturbo", "--prompt-cache-bytes", "1073741824"])
            .promptCacheBytes == .bytes(1 << 30))
        #expect(try ServerArguments.parse(["--model", "m.gturbo", "--prompt-cache-bytes", "auto"])
            .promptCacheBytes == .auto)
    }

    @Test func invalidValuesAreRefused() {
        for value in ["-1", "lots", "", "1.5"] {
            #expect(throws: ServerArgumentError.self) {
                _ = try ServerArguments.parse(["--model", "m.gturbo", "--prompt-cache-bytes", value])
            }
        }
        #expect(throws: ServerArgumentError.self) {
            _ = try ServerArguments.parse(["--model", "m.gturbo", "--prompt-cache-mode", "many"])
        }
    }

    @Test func runnersWithoutPrefixSnapshotsServeMultiPrefixAsSinglePrefix() {
        #expect(ServerModelSession.effectivePromptCacheMode(.multiPrefix, runnerSnapshotsPrefixes: true, family: nil)
            == .multiPrefix)
        #expect(ServerModelSession.effectivePromptCacheMode(.multiPrefix, runnerSnapshotsPrefixes: false, family: nil)
            == .singlePrefix)
        for mode in [ServerPromptCacheMode.off, .singlePrefix] {
            for snapshots in [false, true] {
                #expect(ServerModelSession.effectivePromptCacheMode(mode, runnerSnapshotsPrefixes: snapshots, family: nil)
                    == mode)
            }
        }
    }

    @Test func theAutomaticBudgetFollowsPhysicalMemory() {
        #expect(ServerPromptCacheBytes.auto.resolved(physicalMemoryBytes: 8 * Self.gibibyte) == 0)
        #expect(ServerPromptCacheBytes.auto.resolved(physicalMemoryBytes: 16 * Self.gibibyte) == 1 << 30)
        #expect(ServerPromptCacheBytes.auto.resolved(physicalMemoryBytes: 24 * Self.gibibyte) == 3 << 30)
        #expect(ServerPromptCacheBytes.auto.resolved(physicalMemoryBytes: 32 * Self.gibibyte) == 4 << 30)
        #expect(ServerPromptCacheBytes.auto.resolved(physicalMemoryBytes: 256 * Self.gibibyte) == 4 << 30)
        #expect(ServerPromptCacheBytes.bytes(12345).resolved(physicalMemoryBytes: 8 * Self.gibibyte) == 12345)
    }

}
