import Foundation
import Testing

/// Waits until `condition` holds, checking every few milliseconds, for up
/// to `timeout` (generous, so a loaded machine never fails a test that would
/// pass). Records an issue if it never holds. Tests wait for the specific
/// state they need this way, never for a count of yields.
@MainActor
func waitUntil(_ what: String = "the condition", timeout: Duration = .seconds(10),
               sourceLocation: SourceLocation = #_sourceLocation,
               _ condition: @MainActor () -> Bool) async {
    let deadline = ContinuousClock.now + timeout
    while !condition() {
        guard ContinuousClock.now < deadline else {
            Issue.record("Timed out waiting for \(what)", sourceLocation: sourceLocation)
            return
        }
        try? await Task.sleep(for: .milliseconds(2))
    }
}
