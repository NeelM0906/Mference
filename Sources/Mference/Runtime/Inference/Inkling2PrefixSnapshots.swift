import Foundation

/// How the engine and the server serve Inkling 2's prefix snapshots
/// (`Inkling2ForwardRunner+PrefixSnapshots`) while their default and budget
/// are measured.
public enum Inkling2PrefixSnapshots {
    /// Experiment toggle: `MFERENCE_INKLING2_MULTI_PREFIX=1` lets a
    /// `multi-prefix` request reach the runner's snapshots. Without it, both
    /// the engine and the server serve Inkling 2's `multi-prefix` as
    /// `single-prefix`, as before the runner could snapshot.
    public static let environmentKey = "MFERENCE_INKLING2_MULTI_PREFIX"

    public static func multiPrefixEnabled(
        environment: [String: String] = ProcessInfo.processInfo.environment) -> Bool {
        environment[environmentKey] == "1"
    }

    /// The `auto` retained-state budget's ceiling for Inkling 2: 1 GiB holds
    /// six turn ends of 1,023 tokens or more (171 MB each), or, for example,
    /// four of the live conversation plus one other conversation's turn end
    /// and its 4,096-token image (128 MiB). An explicit byte budget is kept.
    public static let autoBudgetCeiling = 1 << 30
}
