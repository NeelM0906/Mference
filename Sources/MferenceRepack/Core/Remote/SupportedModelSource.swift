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
        let options: RemoteStreamingRepackOptions
        if let baseURL {
            options = RemoteStreamingRepackOptions(
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
        } else {
            options = RemoteStreamingRepackOptions(
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
        return options
    }

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

    /// MiniCPM5-2B from the vendor's own BF16 upload, quantized to INT4
    /// affine group-64 in flight — the project's first plain-llama family and
    /// the second original-repo family with a runner, so it re-runs the W2.1b
    /// quantizer gate against its own control (`minicpm5mlx`).
    ///
    /// Both pins recorded (2026-09-10). Download bytes are the single shard
    /// (5,033,557,096 B; the index's `total_size` is 5,033,512,960 of tensor
    /// data plus the 44,128-byte header). Installed bytes are the INT4 g64
    /// resident set — 1,415,925,760 B, to the byte the control's own
    /// `total_size` — plus the page-rounded index, ~10 MB of tokenizer
    /// sidecars, manifest and receipt. See docs/families/MINICPM5.md.
    public static let minicpm5 = SupportedModelSource(
        name: "minicpm5",
        displayName: "MiniCPM5-2B (quantized at install)",
        repoID: "openbmb/MiniCPM5-2B",
        revision: "cd199ce3ee67549c42ef7372f809f2c63599a3e9",
        sourceIndexSHA256:
            "6d839cd76e8395de548a0e6cc310386f66d1ecbb2c75d198a8dfd3d70892b756",
        modelID: "minicpm5-2b-int4g64",
        approximateDownloadBytes: 5_033_557_096,
        installedBytes: 1_430_000_000,
        reserveBytes: 1_073_741_824,
        kind: .originalRepoQuantize)

    /// The **same checkpoint as `minicpm5`**, from the vendor's own MLX INT4
    /// group-64 affine conversion through the pre-quantized path. It exists
    /// as the trusted control the W2.1b quantizer-quality gate measures
    /// `minicpm5` against (docs/QUANTIZER_QUALITY.md); its `modelID` differs
    /// because the fingerprint table is keyed by it. Download bytes are the
    /// single `model.safetensors` (1,416,035,216 B).
    public static let minicpm5MLX = SupportedModelSource(
        name: "minicpm5mlx",
        displayName: "MiniCPM5-2B MLX 4-bit (W2.1b control)",
        repoID: "openbmb/MiniCPM5-2B-MLX",
        revision: "35ac38ee7bdb0bf7fa748d0700eeb6d6675760a3",
        sourceIndexSHA256:
            "ccf202e0a06fe3c7eb8f354cfb29412a5e64956ad895413d4d9267ae4b3a6045",
        modelID: "minicpm5-2b-mlx-4bit",
        approximateDownloadBytes: 1_416_035_216,
        installedBytes: 1_430_000_000,
        reserveBytes: 1_073_741_824)

    /// Default source when no `--model` selector is given.
    public static let `default` = qwen36

    public static let all: [SupportedModelSource] = [
        qwen36, qwen36Original, qwen38, maple, minicpm5, minicpm5MLX,
    ]

    public static func named(_ name: String) -> SupportedModelSource? {
        all.first { $0.name == name }
    }
}
