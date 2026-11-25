import Foundation

/// Where a downloadable product model comes from: one pinned file with its
/// size and SHA-256 (the Hugging Face LFS object id), as recorded in
/// THIRD_PARTY_NOTICES.md. Only the two GGUF product models download from
/// the app; native installs come from `MferenceRepack`.
public struct DownloadSource: Sendable, Hashable {
    public let model: AppModel
    public let repository: String
    public let revision: String
    public let fileName: String
    /// The name the file is saved under; the repository's `fileName` unless
    /// that name is too generic to share a folder (Quick's projector).
    public let localFileName: String
    public let sizeBytes: UInt64
    /// Lowercase hex.
    public let sha256: String
    public let licence: String
    /// The file's URL; `resolve` URLs redirect to Hugging Face's CDN.
    public let url: URL

    public init(model: AppModel, repository: String, revision: String, fileName: String, sizeBytes: UInt64,
                sha256: String, licence: String, url: URL? = nil, localFileName: String? = nil) {
        self.model = model
        self.repository = repository
        self.revision = revision
        self.fileName = fileName
        self.localFileName = localFileName ?? fileName
        self.sizeBytes = sizeBytes
        self.sha256 = sha256.lowercased()
        self.licence = licence
        self.url = url ?? URL(string: "https://huggingface.co/\(repository)/resolve/\(revision)/\(fileName)")!
    }

    /// Quick: unsloth Qwen3.5-4B UD-Q4_K_XL (L1).
    public static let quick = DownloadSource(
        model: .quick, repository: "unsloth/Qwen3.5-4B-GGUF",
        revision: "e87f176479d0855a907a41277aca2f8ee7a09523",
        fileName: "Qwen3.5-4B-UD-Q4_K_XL.gguf", sizeBytes: 2_912_109_728,
        sha256: "b252c5610a42ca82d20fe2a12813e9d069eed89292907e26c783eeb0bc961bc7",
        licence: "Apache 2.0")

    /// Deep: ISTA-DASLab Qwen3.8-27B GSQ-RCO IQ3_S with the MTP head (L1).
    public static let deep = DownloadSource(
        model: .deep, repository: "ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF",
        revision: "d562806dbafae37109975e970aae91b43e73b440",
        fileName: "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf", sizeBytes: 12_120_016_960,
        sha256: "58fd826723939933dc86f45b7fe04545cbc2de1c70f6fe2cdd3858c87a98c12f",
        licence: "Apache 2.0")

    public static let product: [AppModel: DownloadSource] = [.quick: quick, .deep: deep]

    /// Quick's image understanding: the vision projector of the same
    /// revision (`mmproj-F16.gguf`), saved under a name of its own so it can
    /// sit beside other models' projectors; the engine pairs it with Quick.
    public static let quickImageUnderstanding = DownloadSource(
        model: .quick, repository: "unsloth/Qwen3.5-4B-GGUF",
        revision: "e87f176479d0855a907a41277aca2f8ee7a09523",
        fileName: "mmproj-F16.gguf", sizeBytes: 672_423_616,
        sha256: "cd88edcf8d031894960bb0c9c5b9b7e1fea6ebee02b9f7ce925a00d12891f864",
        licence: "Apache 2.0", localFileName: "Qwen3.5-4B-mmproj-F16.gguf")

    /// Deep's image understanding: the vision projector of the same revision.
    public static let deepImageUnderstanding = DownloadSource(
        model: .deep, repository: "ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF",
        revision: "d562806dbafae37109975e970aae91b43e73b440",
        fileName: "mmproj-Qwen3.8-27B-BF16.gguf", sizeBytes: 931_146_528,
        sha256: "13cb7bebccbd04afc8f4090cb949ecf8937cdf7377c5799b1a0c594e7c0d3e16",
        licence: "Apache 2.0")

    /// The optional image-understanding add-ons, by the model they extend.
    /// They download through their own `DownloadManager` into the same folder
    /// as the models.
    public static let imageUnderstanding: [AppModel: DownloadSource] = [
        .quick: quickImageUnderstanding, .deep: deepImageUnderstanding,
    ]

    /// Hosts a model download may reach (spec 4.4 (b)): Hugging Face and the
    /// CDN its `resolve` URLs redirect to. A host matches itself and its
    /// subdomains.
    public static let allowedHosts = ["huggingface.co", "hf.co"]

    /// Where the app puts downloads: `~/llm-models/gguf`, a directory the
    /// engine's discovery already scans.
    public static var defaultDestination: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appending(path: "llm-models/gguf", directoryHint: .isDirectory)
    }

    static func isAllowed(host: String?, allowedHosts: [String]) -> Bool {
        guard let host = host?.lowercased() else { return false }
        return allowedHosts.contains { host == $0 || host.hasSuffix("." + $0) }
    }
}

/// The person's go-ahead for one download (L4: Deep downloads only after
/// they approve). The UI makes one only from the confirmation dialog that
/// names the model and its size.
public struct DownloadApproval: Sendable, Hashable {
    public let source: DownloadSource
    public let approvedAt: Date

    public init(confirming source: DownloadSource, at date: Date = .now) {
        self.source = source
        self.approvedAt = date
    }
}
