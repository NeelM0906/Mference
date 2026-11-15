import Metal
import Testing
@testable import Mference

/// `GPUKeepWarm.policy` turns the decode filler off on M1-series GPUs only.
/// Families follow Apple's Metal Feature Set Tables ("Metal GPUs"): M1-series
/// is Apple7, M2-series Apple8, M3- and M4-series Apple9, M5-series Apple10.
@Suite struct GPUKeepWarmPolicyTests {
    /// A device that reports every Apple family from Apple1 up to `highest`.
    private static func appleFamilies(upTo highest: Int) -> (MTLGPUFamily) -> Bool {
        { family in (1001...highest).contains(family.rawValue) }
    }

    private static let filler = GPUKeepWarm.Policy.spin(iterations: GPUKeepWarm.spinIterations)

    @Test func m1SeriesDisablesTheFiller() {
        #expect(GPUKeepWarm.policy(supportsFamily: Self.appleFamilies(upTo: 1007)) == .disabled)
    }

    @Test(arguments: [1008, 1009, 1010, 1011])
    func laterApplesKeepTheFiller(highestFamily: Int) {
        #expect(GPUKeepWarm.policy(supportsFamily: Self.appleFamilies(upTo: highestFamily))
                == Self.filler)
    }

    @Test func fillerKeepsItsMeasuredLength() {
        #expect(GPUKeepWarm.spinIterations == 100_000)
    }

    /// The check needs Apple7 *and* the absence of Apple8, so it does not rely
    /// on families being cumulative, and a device that reports no Apple family
    /// at all keeps today's behaviour.
    @Test func onlyApple7WithoutApple8Disables() {
        #expect(GPUKeepWarm.policy(supportsFamily: { $0 == .apple8 }) == Self.filler)
        #expect(GPUKeepWarm.policy(supportsFamily: { _ in false }) == Self.filler)
        #expect(GPUKeepWarm.policy(supportsFamily: { $0 == .apple7 }) == .disabled)
    }

    /// Cross-checks the family rule against this host's real GPU when its name
    /// identifies the chip generation; other hosts (for example a
    /// paravirtual CI GPU) assert nothing.
    @Test func hostDeviceMatchesItsChipGeneration() throws {
        let device = try #require(MetalContext.makeSystemDefaultDevice())
        let policy = GPUKeepWarm.policy(supportsFamily: { device.supportsFamily($0) })
        let name = device.name
        // "Apple M1", "Apple M1 Max", ... but not a future "Apple M10".
        func isGeneration(_ chip: String) -> Bool { name == chip || name.hasPrefix(chip + " ") }
        if isGeneration("Apple M1") {
            #expect(policy == .disabled, "\(name)")
        } else if ["Apple M2", "Apple M3", "Apple M4", "Apple M5"].contains(where: isGeneration) {
            #expect(policy == Self.filler, "\(name)")
        }
    }

    /// The process-wide filler follows the policy for its device: under
    /// `.disabled` no filler thread or queue exists.
    @Test func sharedFillerFollowsItsPolicy() throws {
        let context = try MetalContext()
        let keepWarm = try #require(context.keepWarm)
        #expect(keepWarm.policy
                == GPUKeepWarm.policy(supportsFamily: { context.device.supportsFamily($0) }))
        #expect(keepWarm.isSpinning == (keepWarm.policy != .disabled))
    }
}
