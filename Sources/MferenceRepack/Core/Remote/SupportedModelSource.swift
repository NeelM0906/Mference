import Foundation

/// A pinned upstream checkpoint the installer knows how to repack. Each value
/// fixes the repo, revision and index fingerprint so installs are exactly
/// reproducible. `revision`/`sourceIndexSHA256` may be nil for a source that
/// installs trust-on-first-use: the installer resolves HEAD's commit and
/// reports the computed index hash for pinning instead of failing on it.
public struct SupportedModelSource: Sendable, Equatable {
    /// How the installer consumes the source repo.
    public enum Kind: String, Sendable, Equatable {
        /// A pre-quantized MLX community conversion: the installer only
        /// re-lays-out already-packed INT4/INT2 tensors.
        case preQuantized
        /// The model vendor's own BF16 upload: the installer quantizes to INT4
        /// affine group-64 in flight (Workstream 2). Pinning and SHA discipline
        /// are identical — only the byte transform differs.
        case originalRepoQuantize
    }

    /// CLI selector value (`--model <name>`).
    public let name: String
    public let displayName: String
    public let repoID: String
    public let kind: Kind
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

    public init(name: String,
                displayName: String,
                repoID: String,
                revision: String?,
                sourceIndexSHA256: String?,
                modelID: String,
                approximateDownloadBytes: UInt64,
                installedBytes: UInt64,
                reserveBytes: UInt64,
                kind: Kind = .preQuantized) {
        self.name = name
        self.displayName = displayName
        self.repoID = repoID
        self.kind = kind
        self.revision = revision
        self.sourceIndexSHA256 = sourceIndexSHA256
        self.modelID = modelID
        self.approximateDownloadBytes = approximateDownloadBytes
        self.installedBytes = installedBytes
        self.reserveBytes = reserveBytes
    }

    /// Both pins recorded; unpinned sources install trust-on-first-use.
    public var isPinned: Bool { revision != nil && sourceIndexSHA256 != nil }

    public func installOptions(outputDirectory: URL,
                               overwrite: Bool,
                               token: String?,
                               resume: Bool = false,
                               baseURL: URL? = nil,
                               dryRunSpaceCheck: Bool = false,
                               sidecarPolicy: SidecarPolicy = .default)
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
                baseURL: baseURL,
                sidecarPolicy: sidecarPolicy)
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
            dryRunSpaceCheck: dryRunSpaceCheck,
            sidecarPolicy: sidecarPolicy)
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

    /// The **same checkpoint as `qwen36`**, installed from the vendor's own
    /// BF16 upload through our quantizer instead of copying mlx-community's
    /// pre-quantized conversion. It exists so the two installs can be compared:
    /// this is the control pair the W2.1b quantizer-quality gate is measured on
    /// (docs/QUANTIZER_QUALITY.md), and the reason Qwen 3.6 was chosen is that
    /// the runtime already has a runner for it, so the comparison can be made
    /// at the model level and not just on bytes.
    ///
    /// Both pins recorded. Download bytes are the index's declared total
    /// (71.9 GB across 26 shards); the planner actually reads 71.0 GB of that,
    /// the difference being the dropped vision tower. Installed bytes are the
    /// dry-run's own figure for the larger of the two sidecar policies:
    /// 19,973,468,544 with the `mtp.*` draft group carried (19,498,342,656 with
    /// `--skip-mtp`), rounded up for the receipt and lock files. Every
    /// two-dimensional projection becomes INT4 affine group-64 — 0.5625 bytes
    /// per weight against BF16's 2.0 — while norms, 1-D vectors and the conv
    /// kernels ride through as BF16.
    ///
    /// `modelID` deliberately differs from `qwen36`'s: both entries carry a
    /// pinned index hash, and `SourceFingerprint.knownFingerprints` is keyed by
    /// model ID, so a shared ID would collide. The distinct ID also keeps the
    /// two installs of one checkpoint tellable apart in a manifest.
    public static let qwen36Original = SupportedModelSource(
        name: "qwen36original",
        displayName: "Qwen3.6 35B-A3B (quantized at install)",
        repoID: "Qwen/Qwen3.6-35B-A3B",
        revision: "995ad96eacd98c81ed38be0c5b274b04031597b0",
        sourceIndexSHA256:
            "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83",
        modelID: "qwen3.6-35b-a3b-int4g64",
        approximateDownloadBytes: 71_903_645_408,
        installedBytes: 20_000_000_000,
        reserveBytes: 2_147_483_648,
        kind: .originalRepoQuantize)

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

    /// The first `originalRepoQuantize` entry: no faithful pre-quantized MLX
    /// conversion of Qwen3.8-Flash-Next exists (checked 2026-08-31, see
    /// docs/families/QWEN38_FLASH_NEXT.md), so the installer reads the vendor's
    /// 131 BF16 shards and quantizes to INT4 group-64 in flight.
    ///
    /// Both pins recorded. Download bytes are the index's declared total
    /// (360.0 GB). Installed bytes are ~175 GB: routed experts ~68 GB (INT4
    /// g64), the PLE n-gram table ~102.8 GB (**BF16** — its rows are 160 wide,
    /// which the group size does not divide, so they cannot be quantized), the
    /// resident core ~2.5 GB, the MTP draft pool ~1.4 GB, plus page rounding
    /// and the PLE pool's 0.4% per-block slack. The Day-0 dossier's ~101 GB
    /// figure assumed an INT4 n-gram table and is superseded.
    ///
    /// The runtime has no runner for this family yet: installing it succeeds,
    /// loading it fails by name (see `ManifestReader.peekFamily`).
    public static let qwen38FlashNext = SupportedModelSource(
        name: "qwen38flashnext",
        displayName: "Qwen3.8-Flash-Next 180B-A3.5B (quantized at install)",
        repoID: "Qwen/Qwen3.8-Flash-Next",
        revision: "de4b8e4d43b917e7706784d8bb445c9af86a3540",
        sourceIndexSHA256:
            "99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de",
        modelID: "qwen3.8-flash-next-int4g64",
        approximateDownloadBytes: 359_999_963_128,
        installedBytes: 180_000_000_000,
        reserveBytes: 8_589_934_592,
        kind: .originalRepoQuantize)

    /// Default source when no `--model` selector is given.
    public static let `default` = gemma4

    public static let all: [SupportedModelSource] = [
        gemma4, qwen36, qwen36Original, qwen38, deepseekV4Flash, inklingSmall,
        maple, qwen38FlashNext,
    ]

    public static func named(_ name: String) -> SupportedModelSource? {
        all.first { $0.name == name }
    }
}
