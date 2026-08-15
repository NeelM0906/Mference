import Foundation
import Mference
import MferenceRepackCore

public struct AppModelInstallDescriptor: Equatable, Sendable {
    public let family: ModelFamily
    public let displayName: String
    public let repoID: String
    /// Pinned commit; nil resolves HEAD at install time (trust-on-first-use,
    /// mirrors `SupportedModelSource`).
    public let revision: String?
    /// Pinned index SHA-256; nil skips the probe's checkpoint pin check.
    public let sourceIndexSHA256: String?
    public let approximateDownloadBytes: UInt64
    public let installedBytes: UInt64
    public let rangeStagingBytes: UInt64
    public let reserveBytes: UInt64

    public init(family: ModelFamily,
                displayName: String,
                repoID: String,
                revision: String?,
                sourceIndexSHA256: String?,
                approximateDownloadBytes: UInt64,
                installedBytes: UInt64,
                rangeStagingBytes: UInt64,
                reserveBytes: UInt64) {
        self.family = family
        self.displayName = displayName
        self.repoID = repoID
        self.revision = revision
        self.sourceIndexSHA256 = sourceIndexSHA256
        self.approximateDownloadBytes = approximateDownloadBytes
        self.installedBytes = installedBytes
        self.rangeStagingBytes = rangeStagingBytes
        self.reserveBytes = reserveBytes
    }

    public var requiredFreeBytes: UInt64 {
        installedBytes + rangeStagingBytes + reserveBytes
    }

    public static let `default` = AppModelInstallDescriptor(
        family: .gemma4,
        displayName: "Gemma 4 26B-A4B IT 4-bit",
        repoID: "mlx-community/gemma-4-26b-a4b-it-4bit",
        revision: "0d77464eeb233a2da68ebf9d7dc4edaac7db956d",
        sourceIndexSHA256: "bf198c9f5ea6462addca1966e5dd669c407537a876e82cf06db9084c5c850b13",
        approximateDownloadBytes: 14_620_479_420,
        installedBytes: 14_291_921_884,
        rangeStagingBytes: UInt64(RemoteChunkPolicy.defaultBytes),
        reserveBytes: 1_073_741_824)

    public static let qwen36 = AppModelInstallDescriptor(
        family: .qwen36,
        displayName: "Qwen3.6 35B-A3B 4-bit",
        repoID: "mlx-community/Qwen3.6-35B-A3B-4bit",
        revision: "38740b847e4cb78f352aba30aa41c76e08e6eb46",
        sourceIndexSHA256: "0b28df60e33753a14e816d3b31577ae2c93884c58430a4a6de6ae9ea483842ea",
        approximateDownloadBytes: 19_529_025_048,
        installedBytes: 19_546_491_213,
        rangeStagingBytes: UInt64(RemoteChunkPolicy.defaultBytes),
        reserveBytes: 1_073_741_824)

    /// Pinned after first-install verification (mirrors
    /// `SupportedModelSource.deepseekV4Flash`).
    public static let deepseekV4Flash = AppModelInstallDescriptor(
        family: .deepseekV4Flash,
        displayName: "DeepSeek-V4-Flash 284B-A13B 2-bit DQ",
        repoID: "mlx-community/DeepSeek-V4-Flash-2bit-DQ",
        revision: "722bf559b7de93575b2320973cf2002e05bfe6c9",
        sourceIndexSHA256:
            "d1c2d929ab0a35be32cf18026bb31d6f99dad58d6c93a5a2abbe43791f9d6c30",
        approximateDownloadBytes: 97_000_000_000,
        installedBytes: 97_500_000_000,
        rangeStagingBytes: UInt64(RemoteChunkPolicy.defaultBytes),
        reserveBytes: 2_147_483_648)

    /// Revision and index digest verified against the published repo; the byte
    /// counts are the repo's own totals and remain estimates for the installed
    /// `.gturbo` until a first install confirms them. Installing this family
    /// still fails in the repacker until `ArchInfo` learns
    /// `model_type: "inkling_mm_model"`. See `docs/INKLING_SMALL.md`.
    public static let inklingSmall = AppModelInstallDescriptor(
        family: .inklingSmall,
        displayName: "Inkling-Small 276B-A12B 4-bit",
        repoID: "pipenetwork/Inkling-Small-MLX-4bit",
        revision: "9d6e4720ab7002af25d6129c88ccea6cd9f19372",
        sourceIndexSHA256:
            "fe16aec3cef12438f1d0ff657f7e785781b61271528a66b3b7160fcf1aaca30c",
        approximateDownloadBytes: 148_441_426_867,
        installedBytes: 149_000_000_000,
        rangeStagingBytes: UInt64(RemoteChunkPolicy.defaultBytes),
        reserveBytes: 2_147_483_648)

    public static let maple = AppModelInstallDescriptor(
        family: .maple,
        displayName: "Maple Preview 2-bit MLX",
        repoID: "deepgrove/maple-preview-2bit-mlx",
        revision: "361db5da5e74ff6fcdd852d478e1f266ce11013a",
        sourceIndexSHA256:
            "56000110535c5023b43209a5c142035e12c1cde7b1118759cc9f86335d46ef95",
        approximateDownloadBytes: 5_330_000_000,
        installedBytes: 6_650_000_000,
        rangeStagingBytes: UInt64(RemoteChunkPolicy.defaultBytes),
        reserveBytes: 1_073_741_824)

    /// Pinned to the same revision/index digest as
    /// `SupportedModelSource.qwen38`; installed bytes measured after the
    /// first real install with the MTP tensors attached.
    public static let qwen38 = AppModelInstallDescriptor(
        family: .qwen38,
        displayName: "Qwen3.8 27B 4-bit",
        repoID: "mlx-community/Qwen3.8-27B-4bit",
        revision: "3e6447f082e89cc7f0bc6e5441afd38dfce760ff",
        sourceIndexSHA256:
            "13b840162b4cb35c66fef7df072f7dbb4717908204364f5e5d9f9655a2758fa8",
        approximateDownloadBytes: 15_200_000_000,
        installedBytes: 15_371_847_680,
        rangeStagingBytes: UInt64(RemoteChunkPolicy.defaultBytes),
        reserveBytes: 1_073_741_824)

    /// The shipped descriptor for a model family, if one exists.
    public static func descriptor(for family: ModelFamily) -> AppModelInstallDescriptor? {
        switch family {
        case .gemma4: return .default
        case .qwen36: return .qwen36
        case .qwen38: return .qwen38
        case .deepseekV4Flash: return .deepseekV4Flash
        case .inklingSmall: return .inklingSmall
        case .maple: return .maple
        }
    }

    /// Basename of the installed `.gturbo` directory for this descriptor.
    public var installDirectoryName: String {
        switch family {
        case .gemma4: return "gemma4.gturbo"
        case .qwen36: return "qwen36.gturbo"
        case .qwen38: return "qwen38.gturbo"
        case .deepseekV4Flash: return "deepseekv4flash.gturbo"
        case .inklingSmall: return "inklingsmall.gturbo"
        case .maple: return "maple.gturbo"
        }
    }

    /// The descriptor the app products select at launch. Defaults to Gemma 4.
    /// `MFERENCE_MODEL` in the environment wins; otherwise the persisted
    /// `Mference model` preference applies, so GUI launches without an
    /// environment can select another supported family.
    public static var selected: AppModelInstallDescriptor {
        let environmentValue = ProcessInfo.processInfo.environment["MFERENCE_MODEL"]
        let preferenceValue = UserDefaults(suiteName: "Mference")?
            .string(forKey: "model")
        switch environmentValue ?? preferenceValue {
        case "qwen36": return .qwen36
        case "qwen38": return .qwen38
        case "deepseekv4flash", "dsv4": return .deepseekV4Flash
        case "inklingsmall", "inkling": return .inklingSmall
        case "maple": return .maple
        default: return .default
        }
    }
}

public struct AppModelInstallRequirement: Equatable, Sendable {
    public let probePath: String
    public let requiredBytes: UInt64
    public let availableBytes: UInt64

    public init(probePath: String = "", requiredBytes: UInt64, availableBytes: UInt64) {
        self.probePath = probePath
        self.requiredBytes = requiredBytes
        self.availableBytes = availableBytes
    }

    public var canInstall: Bool { availableBytes >= requiredBytes }

    public var shortfallBytes: UInt64 {
        requiredBytes > availableBytes ? requiredBytes - availableBytes : 0
    }
}

public enum AppModelInstallReadiness: Equatable, Sendable {
    case checking
    case ready(AppModelInstallRequirement)
    case insufficientSpace(AppModelInstallRequirement)
    case failed(String)

    public var requirement: AppModelInstallRequirement? {
        switch self {
        case .ready(let requirement), .insufficientSpace(let requirement):
            return requirement
        case .checking, .failed:
            return nil
        }
    }
}
