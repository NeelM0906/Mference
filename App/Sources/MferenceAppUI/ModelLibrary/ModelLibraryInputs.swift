import Foundation
import Observation

// The values U1's Model library, Onboarding and Settings > Models views
// read. U2's ModelLibrary / DownloadManager / MemoryFit provide them through
// `ModelLibraryModel` (adapter owned by U2); views never see engine types.
// Proposal recorded in $APP/notes/shell.md.

/// This Mac, for the "This Mac: ..." line and the Deep offer (L4).
public struct MachineSummary: Sendable, Hashable {
    /// "Apple M1 Max". Empty when unknown (the line then omits it).
    public var chipName: String
    public var physicalMemoryBytes: UInt64
    /// Free space on the volume that holds `~/llm-models`; nil when unknown.
    public var freeDiskBytes: Int64?

    public init(chipName: String, physicalMemoryBytes: UInt64, freeDiskBytes: Int64?) {
        self.chipName = chipName
        self.physicalMemoryBytes = physicalMemoryBytes
        self.freeDiskBytes = freeDiskBytes
    }

    /// More than 16 GB of memory: Deep may be offered, after approval (L4).
    public var canOfferDeep: Bool { physicalMemoryBytes > 16 * 1_073_741_824 }
}

/// Progress of a download, in bytes.
public struct TransferProgress: Sendable, Hashable {
    public var completedBytes: Int64
    public var totalBytes: Int64?

    public init(completedBytes: Int64, totalBytes: Int64?) {
        self.completedBytes = completedBytes
        self.totalBytes = totalBytes
    }

    public var fraction: Double? {
        guard let totalBytes, totalBytes > 0 else { return nil }
        return min(max(Double(completedBytes) / Double(totalBytes), 0), 1)
    }
}

/// Where a model stands on this Mac.
public enum ModelEntryState: Sendable, Hashable {
    /// Resident in memory now.
    case loaded
    /// Installed, not resident.
    case ready
    /// Being loaded; loads cannot be interrupted (spec 4.3.1).
    case loading
    case downloading(TransferProgress)
    case paused(TransferProgress)
    case notInstalled
    /// A load or download failed; the message is shown as is.
    case failed(String)

    public var isInstalled: Bool {
        switch self {
        case .loaded, .ready, .loading: true
        case .downloading, .paused, .notInstalled, .failed: false
        }
    }

    public var transfer: TransferProgress? {
        switch self {
        case .downloading(let progress), .paused(let progress): progress
        default: nil
        }
    }
}

/// Whether a model fits this Mac (MemoryFit).
public enum MemoryFitVerdict: Sendable, Hashable {
    case fits
    /// Offer it, download only after the person approves (Deep above 16 GB).
    case offerWithApproval
    /// Do not offer; the reason is shown.
    case tooLarge(String)
}

/// One model: a row of the library, a choice in onboarding.
public struct ModelEntry: Sendable, Hashable, Identifiable {
    /// The engine catalog id (`qwen3.5-4b`, `qwen3.8-27b`, ...).
    public var id: String
    /// "Quick", "Deep", "Qwen 3.6", "MiniCPM5", "Maple".
    public var productName: String
    /// "Qwen3.5 4B", "Qwen3.8 27B", "35B-A3B", "2B"; empty when the product name says it all.
    public var variantName: String
    /// One line on what it is for.
    public var summary: String
    /// "llama.cpp" or "Mference". Shown only under Details (L2).
    public var engineName: String
    /// "Unsloth UD-Q4_K_XL", "ISTA GSQ-RCO IQ3_S with MTP".
    public var formatDetail: String
    public var licence: String?
    /// Installed size, or the download size before install.
    public var sizeBytes: Int64?
    /// Where the files are; nil before install.
    public var location: String?
    /// In the "Recommended" group (Quick, Deep).
    public var isRecommended: Bool
    public var state: ModelEntryState
    public var fit: MemoryFitVerdict

    public init(id: String, productName: String, variantName: String, summary: String,
                engineName: String, formatDetail: String, licence: String? = nil,
                sizeBytes: Int64?, location: String? = nil, isRecommended: Bool,
                state: ModelEntryState, fit: MemoryFitVerdict) {
        self.id = id
        self.productName = productName
        self.variantName = variantName
        self.summary = summary
        self.engineName = engineName
        self.formatDetail = formatDetail
        self.licence = licence
        self.sizeBytes = sizeBytes
        self.location = location
        self.isRecommended = isRecommended
        self.state = state
        self.fit = fit
    }

    /// "Quick · Qwen3.5 4B", or "Maple" alone.
    public var title: String {
        variantName.isEmpty ? productName : "\(productName) · \(variantName)"
    }
}

/// The model library as the views see it. U2 provides the live one.
@MainActor
public protocol ModelLibraryModel: AnyObject, Observable {
    var machine: MachineSummary { get }
    var entries: [ModelEntry] { get }
    /// Used for new chats, mail drafts and scanning.
    var defaultModelID: String { get set }
    /// Idle unload after this many minutes (L13: 10).
    var idleUnloadMinutes: Int { get set }
    /// Starts a download. Called only after the person approved it in the UI.
    func download(_ id: String)
    func pause(_ id: String)
    func resume(_ id: String)
    func cancelDownload(_ id: String)
}

enum ModelFormat {
    static func bytes(_ value: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        formatter.allowedUnits = [.useGB, .useMB]
        return formatter.string(fromByteCount: value)
    }

    static func memory(_ value: UInt64) -> String {
        "\(Int((Double(value) / 1_073_741_824).rounded())) GB"
    }

    /// "This Mac: Apple M1 Max, 32 GB memory, 582 GB free."
    static func machineLine(_ machine: MachineSummary) -> String {
        var parts: [String] = []
        if !machine.chipName.isEmpty { parts.append(machine.chipName) }
        parts.append("\(memory(machine.physicalMemoryBytes)) memory")
        if let free = machine.freeDiskBytes { parts.append("\(bytes(free)) free") }
        return "This Mac: " + parts.joined(separator: ", ") + "."
    }

    /// "4.6 of 12.1 GB"
    static func transfer(_ progress: TransferProgress) -> String {
        guard let total = progress.totalBytes else { return bytes(progress.completedBytes) }
        return "\(bytes(progress.completedBytes)) of \(bytes(total))"
    }
}
