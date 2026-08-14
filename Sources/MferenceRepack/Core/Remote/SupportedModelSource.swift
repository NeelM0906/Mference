import Foundation

/// A pinned upstream checkpoint the installer knows how to repack. Each value
/// fixes the repo, revision and index fingerprint so installs are exactly
/// reproducible. `revision`/`sourceIndexSHA256` may be nil for a source that
/// installs trust-on-first-use: the installer resolves HEAD's commit and
/// reports the computed index hash for pinning instead of failing on it.
public struct SupportedModelSource: Sendable, Equatable {
    /// CLI selector value (`--model <name>`).
    public let name: String
    public let displayName: String
    public let repoID: String
    /// Pinned commit; nil resolves HEAD's `X-Repo-Commit` at install time.
    public let revision: String?
    /// Pinned `model.safetensors.index.json` SHA-256; nil accepts any index
    /// and reports the computed hash for pinning.
    public let sourceIndexSHA256: String?
    /// Value recorded as `manifest.modelID` when the source fingerprint matches.
    public let modelID: String
    public let approximateDownloadBytes: UInt64
    public let installedBytes: UInt64
    public let reserveBytes: UInt64

    /// Both pins recorded; unpinned sources install trust-on-first-use.
    public var isPinned: Bool { revision != nil && sourceIndexSHA256 != nil }

    public func installOptions(outputDirectory: URL,
                               overwrite: Bool,
                               token: String?,
                               resume: Bool = false,
                               baseURL: URL? = nil,
                               dryRunSpaceCheck: Bool = false)
        -> RemoteStreamingRepackOptions {
        if let baseURL {
            return RemoteStreamingRepackOptions(
                repoID: repoID,
                revision: revision ?? "main",
                outputDir: outputDirectory.path,
                token: token,
                requireKnownSource: true,
                minFreeReserveBytes: reserveBytes,
                overwrite: overwrite,
                resume: resume,
                dryRunSpaceCheck: dryRunSpaceCheck,
                baseURL: baseURL)
        }
        return RemoteStreamingRepackOptions(
            repoID: repoID,
            revision: revision ?? "main",
            outputDir: outputDirectory.path,
            token: token,
            requireKnownSource: true,
            minFreeReserveBytes: reserveBytes,
            overwrite: overwrite,
            resume: resume,
            dryRunSpaceCheck: dryRunSpaceCheck)
    }

    public static let gemma4 = SupportedModelSource(
        name: "gemma4",
        displayName: "Gemma 4 26B-A4B IT 4-bit",
        repoID: "mlx-community/gemma-4-26b-a4b-it-4bit",
        revision: "0d77464eeb233a2da68ebf9d7dc4edaac7db956d",
        sourceIndexSHA256:
            "bf198c9f5ea6462addca1966e5dd669c407537a876e82cf06db9084c5c850b13",
        modelID: "mlx-community/gemma-4-26b-a4b-it-4bit",
        approximateDownloadBytes: 14_620_479_420,
        installedBytes: 14_291_921_884,
        reserveBytes: 1_073_741_824)

    /// Download estimate covers the `language_model.*` tensors plus tokenizer
    /// and metadata sidecars; the vision tower is never fetched. Installed
    /// bytes add the resident index and per-expert 16 KB page rounding
    /// (the 1,769,472-byte expert blob is already page-aligned) plus
    /// layout/manifest sidecars.
    public static let qwen36 = SupportedModelSource(
        name: "qwen36",
        displayName: "Qwen3.6 35B-A3B 4-bit",
        repoID: "mlx-community/Qwen3.6-35B-A3B-4bit",
        revision: "38740b847e4cb78f352aba30aa41c76e08e6eb46",
        sourceIndexSHA256:
            "0b28df60e33753a14e816d3b31577ae2c93884c58430a4a6de6ae9ea483842ea",
        modelID: "qwen3.6-35b-a3b-4bit",
        approximateDownloadBytes: 19_529_025_048,
        installedBytes: 19_546_491_213,
        reserveBytes: 1_073_741_824)

    /// Text stack of the multimodal Qwen3.8 checkpoint; the vision tower is
    /// excluded by the planner, so the download estimate is the repo total
    /// (16.05 GB) minus the ~0.9 GB tower. Installed bytes add the resident
    /// index plus layout/manifest sidecars — dense, so there are no
    /// packed-expert blobs or page rounding.
    public static let qwen38 = SupportedModelSource(
        name: "qwen38",
        displayName: "Qwen3.8 27B 4-bit",
        repoID: "mlx-community/Qwen3.8-27B-4bit",
        revision: "3e6447f082e89cc7f0bc6e5441afd38dfce760ff",
        sourceIndexSHA256:
            "13b840162b4cb35c66fef7df072f7dbb4717908204364f5e5d9f9655a2758fa8",
        modelID: "qwen3.8-27b-4bit",
        approximateDownloadBytes: 15_200_000_000,
        installedBytes: 15_400_000_000,
        reserveBytes: 1_073_741_824)

    /// Revision and index hash are not yet pinned (the upload has not been
    /// fingerprinted); the installer resolves HEAD and prints the computed
    /// index SHA-256 to record here. Byte estimates follow
    /// docs/DEEPSEEK_V4_FLASH.md (~91 GB installed) with headroom for the
    /// resident file and page rounding.
    public static let deepseekV4Flash = SupportedModelSource(
        name: "deepseekv4flash",
        displayName: "DeepSeek-V4-Flash 284B-A13B 2-bit DQ",
        repoID: "mlx-community/DeepSeek-V4-Flash-2bit-DQ",
        revision: "722bf559b7de93575b2320973cf2002e05bfe6c9",
        sourceIndexSHA256:
            "d1c2d929ab0a35be32cf18026bb31d6f99dad58d6c93a5a2abbe43791f9d6c30",
        modelID: "deepseek-v4-flash-2bit-dq",
        approximateDownloadBytes: 97_000_000_000,
        installedBytes: 97_500_000_000,
        reserveBytes: 2_147_483_648)

    /// Revision and index digest verified against the published repo. The
    /// download estimate is the repo's own total (148.4 GB); the vision and
    /// audio towers are excluded by the planner but they are only 18 tensors,
    /// so the saving is immaterial. Installed bytes carry headroom for the
    /// resident index and per-expert page rounding. See docs/INKLING_SMALL.md.
    public static let inklingSmall = SupportedModelSource(
        name: "inklingsmall",
        displayName: "Inkling-Small 276B-A12B 4-bit",
        repoID: "pipenetwork/Inkling-Small-MLX-4bit",
        revision: "9d6e4720ab7002af25d6129c88ccea6cd9f19372",
        sourceIndexSHA256:
            "fe16aec3cef12438f1d0ff657f7e785781b61271528a66b3b7160fcf1aaca30c",
        modelID: "inkling-small-4bit",
        approximateDownloadBytes: 148_441_426_867,
        installedBytes: 149_000_000_000,
        reserveBytes: 2_147_483_648)

    public static let maple = SupportedModelSource(
        name: "maple",
        displayName: "Maple Preview 2-bit MLX",
        repoID: "deepgrove/maple-preview-2bit-mlx",
        revision: "361db5da5e74ff6fcdd852d478e1f266ce11013a",
        sourceIndexSHA256:
            "56000110535c5023b43209a5c142035e12c1cde7b1118759cc9f86335d46ef95",
        modelID: "maple-preview-2bit-mlx",
        approximateDownloadBytes: 5_330_000_000,
        installedBytes: 6_650_000_000,
        reserveBytes: 1_073_741_824)

    /// Default source when no `--model` selector is given.
    public static let `default` = gemma4

    public static let all: [SupportedModelSource] = [
        gemma4, qwen36, qwen38, deepseekV4Flash, inklingSmall, maple,
    ]

    public static func named(_ name: String) -> SupportedModelSource? {
        all.first { $0.name == name }
    }
}
