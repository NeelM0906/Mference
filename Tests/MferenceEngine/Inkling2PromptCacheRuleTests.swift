import Foundation
import Testing
import Mference
@testable import MferenceEngine

/// Inkling 2's runner snapshots its prefix state, but the engine serves its
/// `multi-prefix` as `single-prefix` unless `MFERENCE_INKLING2_MULTI_PREFIX=1`,
/// and caps its automatic retained-state budget at 1 GiB, while the default
/// and the budget are measured. Every other family keeps its mode and budget.
@Suite("Inkling 2 prompt cache rule (engine)") struct Inkling2PromptCacheRuleTests {
    private static let gibibyte = UInt64(1) << 30
    private static let modes: [EnginePromptCacheMode] = [.off, .singlePrefix, .multiPrefix]
    private static let hosts: [UInt64] = [8, 12, 16, 20, 24, 32, 36, 48, 64, 96, 128, 192, 256].map { $0 * gibibyte }

    /// The rule before Inkling 2 could snapshot.
    private static func before(_ requested: EnginePromptCacheMode, _ snapshots: Bool) -> EnginePromptCacheMode {
        requested == .multiPrefix && !snapshots ? .singlePrefix : requested
    }

    @Test func otherFamiliesKeepTheirModeAndBudget() {
        for family in ModelFamily.allCases where family != .inkling2Loop {
            for mode in Self.modes {
                for snapshots in [false, true] {
                    for toggle in [false, true] {
                        #expect(NativeBackend.effectivePromptCacheMode(
                            mode, runnerSnapshotsPrefixes: snapshots, family: family, inkling2MultiPrefix: toggle)
                            == Self.before(mode, snapshots), "\(family) \(mode) \(snapshots) \(toggle)")
                    }
                }
            }
            for host in Self.hosts {
                #expect(EnginePromptCacheBytes.auto.resolved(family: family, physicalMemoryBytes: host)
                    == EnginePromptCacheBytes.auto.resolved(physicalMemoryBytes: host), "\(family) \(host)")
            }
            #expect(EnginePromptCacheBytes.bytes(12345).resolved(family: family) == 12345)
        }
        // No family: the old rule and the old budget.
        for mode in Self.modes {
            for snapshots in [false, true] {
                #expect(NativeBackend.effectivePromptCacheMode(mode, runnerSnapshotsPrefixes: snapshots, family: nil)
                    == Self.before(mode, snapshots))
            }
        }
        #expect(EnginePromptCacheBytes.auto.resolved(physicalMemoryBytes: 32 * Self.gibibyte) == 4 << 30)
        #expect(EnginePromptCacheBytes.auto.resolved(physicalMemoryBytes: 16 * Self.gibibyte) == 1 << 30)
        #expect(EnginePromptCacheBytes.auto.resolved(physicalMemoryBytes: 8 * Self.gibibyte) == 0)
    }

    @Test func inklingServesMultiPrefixOnlyWithTheToggle() {
        let family = ModelFamily.inkling2Loop
        #expect(NativeBackend.effectivePromptCacheMode(.multiPrefix, runnerSnapshotsPrefixes: true, family: family)
            == .singlePrefix)
        #expect(NativeBackend.effectivePromptCacheMode(.multiPrefix, runnerSnapshotsPrefixes: true, family: family,
                                                       inkling2MultiPrefix: true) == .multiPrefix)
        #expect(NativeBackend.effectivePromptCacheMode(.multiPrefix, runnerSnapshotsPrefixes: false, family: family,
                                                       inkling2MultiPrefix: true) == .singlePrefix)
        for mode in [EnginePromptCacheMode.off, .singlePrefix] {
            for toggle in [false, true] {
                #expect(NativeBackend.effectivePromptCacheMode(mode, runnerSnapshotsPrefixes: true, family: family,
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
            let general = EnginePromptCacheBytes.auto.resolved(physicalMemoryBytes: host)
            #expect(EnginePromptCacheBytes.auto.resolved(family: family, physicalMemoryBytes: host)
                == min(general, 1 << 30), "\(host)")
        }
        #expect(EnginePromptCacheBytes.auto.resolved(family: family, physicalMemoryBytes: 32 * Self.gibibyte) == 1 << 30)
        #expect(EnginePromptCacheBytes.auto.resolved(family: family, physicalMemoryBytes: 8 * Self.gibibyte) == 0)
        #expect(EnginePromptCacheBytes.bytes(3 << 30).resolved(family: family) == 3 << 30)
    }
}
