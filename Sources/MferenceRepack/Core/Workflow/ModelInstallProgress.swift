import Foundation

public enum ModelInstallProgress: Equatable, Sendable {
    case downloadingMetadata
    case planning(downloadBytes: UInt64, outputBytes: UInt64)
    case checkingDisk(DiskSpaceRequirement)
    case reservingOutput(bytes: UInt64)
    case copyingPayload(
        reusedBytes: UInt64,
        downloadedThisRunBytes: UInt64,
        totalBytes: UInt64
    )
    case hashingOutput(String)
    /// Rewriting a Gemma 4 QAT layer file without its `-8 * scale` biases.
    case compactingExperts(String)
    case finalizing

    /// Plain progress text, separate from the final machine-readable byte
    /// totals. Decimal GB matches checkpoint publishers' download sizes.
    public var statusLine: String {
        func gb(_ bytes: UInt64) -> String { String(format: "%.2f GB", Double(bytes) / 1_000_000_000) }
        switch self {
        case .downloadingMetadata:
            return "Reading pinned checkpoint metadata (resume also verifies saved ranges)."
        case let .planning(download, output):
            return "Plan: \(gb(download)) source payload; \(gb(output)) installed output."
        case .checkingDisk:
            return "Checking free disk space."
        case let .reservingOutput(bytes):
            return "Preparing \(gb(bytes)) of output files."
        case let .copyingPayload(reused, downloaded, total):
            let completed = Double(reused) + Double(downloaded)
            let percent = total == 0 ? 100 : min(100, completed / Double(total) * 100)
            return String(format: "Payload %.1f%%", percent)
                + ": \(gb(reused)) reused + \(gb(downloaded)) downloaded this run / \(gb(total)) total."
        case let .hashingOutput(path):
            return "Verifying output: \(path)"
        case let .compactingExperts(path):
            return "Storing experts without implied biases: \(path)"
        case .finalizing:
            return "Finalizing verified install."
        }
    }
}
