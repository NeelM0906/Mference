import Foundation
import Testing
@testable import Mference

/// The auto profile's top rung: pick `.resident` only when the routed-expert
/// pool, core weights, and a fixed headroom all fit physical memory, and only
/// for the Qwen family in Phase 1. Everything else falls back to the existing
/// slot-count rule.
@Suite struct ResidencyAutoProfileTests {

    static let gib = UInt64(1) << 30
    /// Approximate Qwen 3.6 sizes: 18.1 GB expert pool, 1.45 GB core.
    static let qwenPool = UInt64(18_100_000_000)
    static let qwenCore = UInt64(1_450_000_000)

    @Test("24 GiB host with Qwen picks resident")
    func qwenOn24GiBPicksResident() {
        let mode = RuntimeConfiguration.defaultExpertStreamingMode(
            for: .qwen36,
            physicalMemoryBytes: 24 * Self.gib,
            expertPoolBytes: Self.qwenPool,
            coreWeightsBytes: Self.qwenCore)
        guard case .resident = mode else {
            Issue.record("expected resident, got \(mode)")
            return
        }
    }

    @Test("16 GiB host with Qwen falls back to 32 slots")
    func qwenOn16GiBFallsBackToSlots() {
        let mode = RuntimeConfiguration.defaultExpertStreamingMode(
            for: .qwen36,
            physicalMemoryBytes: 16 * Self.gib,
            expertPoolBytes: Self.qwenPool,
            coreWeightsBytes: Self.qwenCore)
        guard case .pread(let slots) = mode, slots == 32 else {
            Issue.record("expected 32-slot fallback, got \(mode)")
            return
        }
    }

    @Test("8 GiB host with Qwen falls back to 16 slots")
    func qwenOn8GiBFallsBackToSmallSlots() {
        let mode = RuntimeConfiguration.defaultExpertStreamingMode(
            for: .qwen36,
            physicalMemoryBytes: 8 * Self.gib,
            expertPoolBytes: Self.qwenPool,
            coreWeightsBytes: Self.qwenCore)
        guard case .pread(let slots) = mode, slots == 16 else {
            Issue.record("expected 16-slot fallback, got \(mode)")
            return
        }
    }

    @Test("Non-Qwen families keep the slot default even when memory fits")
    func nonQwenFamiliesStayOnSlots() {
        let mode = RuntimeConfiguration.defaultExpertStreamingMode(
            for: .gemma4,
            physicalMemoryBytes: 128 * Self.gib,
            expertPoolBytes: UInt64(12_900_000_000),
            coreWeightsBytes: UInt64(1_360_000_000))
        guard case .pread(let slots) = mode, slots == 16 else {
            Issue.record("expected 16-slot Gemma default, got \(mode)")
            return
        }
    }
}
