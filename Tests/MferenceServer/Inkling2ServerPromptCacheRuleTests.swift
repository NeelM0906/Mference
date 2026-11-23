import Foundation
import Testing
import Mference
@testable import MferenceServerCore

/// Inkling 2's runner snapshots its prefix state, but the server serves its
/// `multi-prefix` as `single-prefix` unless `MFERENCE_INKLING2_MULTI_PREFIX=1`,
/// and caps its automatic retained-state budget at 1 GiB, while the default
/// and the budget are measured. Every other family keeps its mode and budget.
@Suite("Inkling 2 prompt cache rule (server)") struct Inkling2ServerPromptCacheRuleTests {
    private static let gibibyte = UInt64(1) << 30
    private static let modes: [ServerPromptCacheMode] = [.off, .singlePrefix, .multiPrefix]
    private static let hosts: [UInt64] = [8, 12, 16, 20, 24, 32, 36, 48, 64, 96, 128, 192, 256].map { $0 * gibibyte }

    /// The rule before Inkling 2 could snapshot.
    private static func before(_ requested: ServerPromptCacheMode, _ snapshots: Bool) -> ServerPromptCacheMode {
        requested == .multiPrefix && !snapshots ? .singlePrefix : requested
    }

    @Test func otherFamiliesKeepTheirModeAndBudget() {
        for family in ModelFamily.allCases where family != .inkling2Loop {
            for mode in Self.modes {
                for snapshots in [false, true] {
                    for toggle in [false, true] {
                        #expect(ServerModelSession.effectivePromptCacheMode(
                            mode, runnerSnapshotsPrefixes: snapshots, family: family, inkling2MultiPrefix: toggle)
                            == Self.before(mode, snapshots), "\(family) \(mode) \(snapshots) \(toggle)")
                    }
                }
            }
            for host in Self.hosts {
                #expect(ServerPromptCacheBytes.auto.resolved(family: family, physicalMemoryBytes: host)
                    == ServerPromptCacheBytes.auto.resolved(physicalMemoryBytes: host), "\(family) \(host)")
            }
            #expect(ServerPromptCacheBytes.bytes(12345).resolved(family: family) == 12345)
        }
        // No family: the old rule and the old budget.
        for mode in Self.modes {
            for snapshots in [false, true] {
                #expect(ServerModelSession.effectivePromptCacheMode(mode, runnerSnapshotsPrefixes: snapshots, family: nil)
                    == Self.before(mode, snapshots))
            }
        }
        #expect(ServerPromptCacheBytes.auto.resolved(physicalMemoryBytes: 32 * Self.gibibyte) == 4 << 30)
        #expect(ServerPromptCacheBytes.auto.resolved(physicalMemoryBytes: 16 * Self.gibibyte) == 1 << 30)
        #expect(ServerPromptCacheBytes.auto.resolved(physicalMemoryBytes: 8 * Self.gibibyte) == 0)
    }

    @Test func inklingServesMultiPrefixOnlyWithTheToggle() {
        let family = ModelFamily.inkling2Loop
        #expect(ServerModelSession.effectivePromptCacheMode(.multiPrefix, runnerSnapshotsPrefixes: true, family: family)
            == .singlePrefix)
        #expect(ServerModelSession.effectivePromptCacheMode(.multiPrefix, runnerSnapshotsPrefixes: true, family: family,
                                                       inkling2MultiPrefix: true) == .multiPrefix)
        #expect(ServerModelSession.effectivePromptCacheMode(.multiPrefix, runnerSnapshotsPrefixes: false, family: family,
                                                       inkling2MultiPrefix: true) == .singlePrefix)
        for mode in [ServerPromptCacheMode.off, .singlePrefix] {
            for toggle in [false, true] {
                #expect(ServerModelSession.effectivePromptCacheMode(mode, runnerSnapshotsPrefixes: true, family: family,
                                                               inkling2MultiPrefix: toggle) == mode)
            }
        }
        #expect(!Inkling2PrefixSnapshots.multiPrefixEnabled(environment: [:]))
        #expect(!Inkling2PrefixSnapshots.multiPrefixEnabled(environment: ["MFERENCE_INKLING2_MULTI_PREFIX": "0"]))
        #expect(Inkling2PrefixSnapshots.multiPrefixEnabled(environment: ["MFERENCE_INKLING2_MULTI_PREFIX": "1"]))
    }

    @Test func inklingsAutomaticBudgetIsCappedAtOneGibibyte() {
        let family = ModelFamily.inkling2Loop
        for host in Self.hosts {
            let general = ServerPromptCacheBytes.auto.resolved(physicalMemoryBytes: host)
            #expect(ServerPromptCacheBytes.auto.resolved(family: family, physicalMemoryBytes: host)
                == min(general, 1 << 30), "\(host)")
        }
        #expect(ServerPromptCacheBytes.auto.resolved(family: family, physicalMemoryBytes: 32 * Self.gibibyte) == 1 << 30)
        #expect(ServerPromptCacheBytes.auto.resolved(family: family, physicalMemoryBytes: 8 * Self.gibibyte) == 0)
        #expect(ServerPromptCacheBytes.bytes(3 << 30).resolved(family: family) == 3 << 30)
    }
}
