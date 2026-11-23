import Foundation
import Mference

/// API model identifier for a native family, derived without loading anything.
/// The same mapping `MferenceServer` has always used, so existing clients keep
/// their identifiers; checkpoint variants keep their own identity.
public enum EngineNativeModelID {
    public static func modelID(for family: ModelFamily, checkpointID: String? = nil) -> String {
        return switch family {
        case .qwen36: "qwen3.6-35b-a3b"
        case .qwen38: "qwen3.8-27b-4bit"
        case .maple: "maple-preview-2bit-mlx"
        case .minicpm5: "minicpm5-2b-int4g64"
        case .inkling2Loop: "inkling-2-mini-ud-q4_k_xl"
        }
    }
}

/// What a candidate directory turned out to be.
public enum EngineInstallProbeResult: Equatable, Sendable {
    /// No `manifest.json`: an ordinary directory, not a partial install.
    case notAModelDirectory
    /// A manifest is present but the install is not usable, with the reason.
    case partial(String)
    /// A complete install of a family no runner can execute.
    case notRunnable(family: String, detail: String)
    case complete(family: ModelFamily)
}

/// Completeness probe for one `.gturbo` directory: manifest, family, arch
/// baseline, `packed_experts/layout.json`, and a receipt bound to the
/// manifest. Metadata only; no weight file is opened.
///
/// The same checks as `MferenceServer`'s `ServerLibraryProbe`, which the
/// engine cannot import (the server depends on the engine). Like it, it does
/// not compare `sourceSnapshotHash` against a pinned descriptor, so a locally
/// requantized install is served too.
public enum EngineInstallProbe {
    /// `gatedFamilies` overrides the runtime's capability gate table, for tests.
    public static func probe(directory: URL,
                             gatedFamilies: [String: [String]]? = nil) -> EngineInstallProbeResult {
        let fileManager = FileManager.default
        let directory = directory.standardizedFileURL
        let name = directory.lastPathComponent
        // `MferenceRepack` stages into `<name>.partial` and holds
        // `<name>.install.lock` for the duration.
        if name.hasSuffix(".partial") {
            return .partial("in-progress install staging directory")
        }
        let lock = directory.deletingLastPathComponent()
            .appendingPathComponent(name + ".install.lock")
        if installLockIsHeld(at: lock) {
            return .partial("\(name).install.lock is held by a running install")
        }
        let manifestURL = directory.appendingPathComponent("manifest.json")
        guard fileManager.fileExists(atPath: manifestURL.path) else {
            return .notAModelDirectory
        }
        if let retired = ManifestReader.retiredInstall(manifestURL: manifestURL) {
            return .notRunnable(family: retired.identifier, detail: retired.detail)
        }
        if let raw = rawFamily(manifestURL: manifestURL),
           let missingAxes = gatedFamilies.map({ $0[raw] })
               ?? ManifestReader.missingRunnerAxes(forRawFamily: raw) {
            return .notRunnable(
                family: raw,
                detail: "no runner for \(raw); missing axes "
                    + missingAxes.joined(separator: ", "))
        }
        let family: ModelFamily
        do {
            family = try ManifestReader.peekFamily(directoryURL: directory)
        } catch {
            return .partial("\(error)")
        }
        guard let baseline = ArchConfig.knownArchitectures[family] else {
            return .partial("unknown model family \(family.rawValue)")
        }
        do {
            _ = try ManifestReader.load(directoryURL: directory, expecting: baseline)
            let layout = directory.appendingPathComponent("packed_experts/layout.json")
            guard fileManager.fileExists(atPath: layout.path) else {
                return .partial("packed_experts/layout.json is missing")
            }
            let receipt = try VerifiedInstallReceiptReader.load(directoryURL: directory)
            let manifestHash = try Sha256Verifier.hashFile(at: manifestURL, chunkBytes: 65_536)
            try VerifiedInstallReceiptReader.validateManifestBinding(
                receipt,
                directoryURL: directory,
                manifestSha256: manifestHash)
        } catch {
            return .partial("\(error)")
        }
        return .complete(family: family)
    }

    /// `arch.family` as the manifest spells it, or nil when unreadable.
    private static func rawFamily(manifestURL: URL) -> String? {
        struct FamilyPeek: Decodable {
            struct Arch: Decodable { let family: String? }
            let arch: Arch
        }
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: manifestURL.path),
              let size = attributes[.size] as? NSNumber,
              size.uint64Value <= ManifestReader.defaultMaxBytes,
              let data = try? Data(contentsOf: manifestURL) else { return nil }
        return (try? JSONDecoder().decode(FamilyPeek.self, from: data))?.arch.family
    }

    /// Whether a repacker holds the sibling install lock. Every completed
    /// install leaves a stale zero-byte lock file behind; only the advisory
    /// lock itself says an install is running.
    static func installLockIsHeld(at lock: URL) -> Bool {
        let descriptor = open(lock.path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
        guard descriptor >= 0 else { return false }
        defer { close(descriptor) }
        if flock(descriptor, LOCK_EX | LOCK_NB) == 0 {
            _ = flock(descriptor, LOCK_UN)
            return false
        }
        return errno == EWOULDBLOCK
    }
}
