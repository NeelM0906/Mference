import Foundation
import Testing

/// Test tiers (engine program spec section 12, change 3).
///
/// Production-shape CPU-reference cases are slow in the debug test build and
/// add little over their small-shape siblings on every pull request, so they
/// run in the nightly workflow and in the coordinator's full-suite run on each
/// merged tip instead. They are not skipped there and nothing about what they
/// assert changes: set `MFERENCE_TEST_TIER=full` to run them anywhere.
enum TestTier {
    static let environmentKey = "MFERENCE_TEST_TIER"

    /// True when the full tier (T1b) is requested.
    static var isFull: Bool {
        ProcessInfo.processInfo.environment[environmentKey] == "full"
    }
}

extension Trait where Self == ConditionTrait {
    /// Runs the test only in the full tier (`MFERENCE_TEST_TIER=full`).
    static var fullTier: Self {
        .enabled(if: TestTier.isFull,
                 "production-shape case; runs nightly and with MFERENCE_TEST_TIER=full")
    }
}
