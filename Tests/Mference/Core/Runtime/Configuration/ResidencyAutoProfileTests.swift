import Foundation
import Testing
@testable import Mference

/// The auto profile keeps the measured slot rule: the 2026-08-07 community
/// A/B on the 24 GB M5 showed `.resident` losing every case (long prompts by
/// 56%) because the page cache already holds the whole Qwen pool at 32
/// slots. `resident`, 96, and 128 stay explicit flags.
@Suite struct ResidencyAutoProfileTests {

    static let gib = UInt64(1) << 30
    /// Approximate Qwen 3.6 sizes: 18.1 GB expert pool, 1.45 GB core.
    static let qwenPool = UInt64(18_100_000_000)
    static let qwenCore = UInt64(1_450_000_000)

    @Test("24 GiB host with Qwen picks the measured 96-slot default")
    func qwenOn24GiBKeepsSlots() {
        let mode = RuntimeConfiguration.defaultExpertStreamingMode(
            for: .qwen36,
            physicalMemoryBytes: 24 * Self.gib,
            expertPoolBytes: Self.qwenPool,
            coreWeightsBytes: Self.qwenCore)
        guard case .pread(let slots) = mode, slots == 96 else {
            Issue.record("expected 96-slot default, got \(mode)")
            return
        }
    }

    @Test("8 GiB host with Qwen keeps 16 slots")
    func qwenOn8GiBKeepsSmallSlots() {
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

    @Test("Non-Qwen families keep the slot default")
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
