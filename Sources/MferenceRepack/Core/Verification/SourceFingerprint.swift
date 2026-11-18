import Foundation

/// Snapshot fingerprints pinned by the project. Adding a new entry means the
/// importer has been validated against a fresh upload of the source.
public enum SourceFingerprint {
    public static let knownFingerprints: [String: String] = Dictionary(
        uniqueKeysWithValues: SupportedModelSource.all.compactMap { source in
            source.sourceIndexSHA256.map { (source.modelID, $0) }
        })

    /// Returns the recognised model ID for a given index.json SHA-256, or nil.
    public static func modelID(forIndexSha256 sha256Hex: String) -> String? {
        for (id, sha) in knownFingerprints where sha == sha256Hex { return id }
        return nil
    }

    /// Model ID for a supported source that ships without a pinned index
    /// hash (trust-on-first-use). Pinned sources never resolve through here,
    /// so a hash mismatch against a pinned repo still hard-fails.
    public static func trustOnFirstUseModelID(forRepoID repoID: String) -> String? {
        SupportedModelSource.all.first {
            $0.repoID == repoID && $0.sourceIndexSHA256 == nil
        }?.modelID
    }
}
