import Darwin
import Foundation

/// Whether a model fits this Mac's memory (spec 4.3, 4.3.2; decision L4).
///
/// The engine has no footprint API yet (asked in engine-asks.md), so the
/// estimate is the size on disk times the share that stays resident, plus
/// headroom for the KV cache and runtime, against three quarters of
/// physical memory (the rest is the system's and the app's). The margins are
/// provisional until checkpoint C4 records measured ones.
public struct MemoryFit: Sendable, Hashable {
    public enum Verdict: String, Sendable, Hashable {
        /// Runs comfortably next to the system and other apps.
        case fits
        /// Runs, but other apps may be pushed to swap; say so.
        case tight
        /// Do not offer or load it on this Mac.
        case tooLarge
    }

    public let physicalMemory: UInt64

    /// Deep is offered only on Macs with more than 16 GB (L4).
    public static let deepMinimumMemoryExclusive: UInt64 = 16 << 30

    public init(physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory) {
        self.physicalMemory = physicalMemory
    }

    /// Whether the app offers `model` on this Mac at all.
    public func isOffered(_ model: AppModel) -> Bool {
        notOfferedReason(model) == nil
    }

    /// Why it is not offered, for the library row; the engine applies the
    /// same rule at load (`EngineMemoryGate`).
    public func notOfferedReason(_ model: AppModel) -> String? {
        EngineMemoryGate.reason(model, physicalMemory: physicalMemory)
    }

    /// The share of the install that stays resident, and the headroom for
    /// KV cache and runtime. Qwen 3.6 streams its routed experts from the
    /// SSD, so most of its install is not resident.
    static func profile(_ model: AppModel) -> (residentShare: Double, headroom: UInt64) {
        switch model {
        case .quick: (1.0, 1_536 << 20)
        case .deep: (1.0, 2_560 << 20)
        case .qwen36: (0.3, 2_048 << 20)
        case .minicpm5: (1.0, 1_024 << 20)
        case .maple: (1.0, 1_536 << 20)
        case .apple: (0, 0)
        }
    }

    /// Estimated resident bytes for `model` with `sizeBytes` on disk.
    public func estimatedResidentBytes(_ model: AppModel, sizeBytes: UInt64) -> UInt64 {
        let profile = Self.profile(model)
        return UInt64(Double(sizeBytes) * profile.residentShare) + profile.headroom
    }

    public func verdict(for model: AppModel, sizeBytes: UInt64) -> Verdict {
        guard isOffered(model) else { return .tooLarge }
        let need = Double(estimatedResidentBytes(model, sizeBytes: sizeBytes))
        let memory = Double(physicalMemory)
        if need <= memory * 0.75 { return .fits }
        if need <= memory * 0.9 { return .tight }
        return .tooLarge
    }

    /// Memory as the Mac's spec sheet says it ("32 GB").
    public var memoryText: String {
        "\(Int((Double(physicalMemory) / Double(1 << 30)).rounded())) GB"
    }

    /// The chip ("Apple M1 Max"), from the kernel; nil when unknown.
    public static var chipName: String? {
        var size = 0
        guard sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buffer = [CChar](repeating: 0, count: size)
        guard sysctlbyname("machdep.cpu.brand_string", &buffer, &size, nil, 0) == 0 else { return nil }
        let name = String(decoding: buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
        return name.isEmpty ? nil : name
    }
}
