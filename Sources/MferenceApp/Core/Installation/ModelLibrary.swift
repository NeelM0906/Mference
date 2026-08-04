import Foundation
import Mference

public struct ModelCatalogEntry: Identifiable, Equatable, Sendable {
    public let descriptor: AppModelInstallDescriptor
    /// Directory of a complete, verified install of this family, if one was
    /// found in any library root.
    public let installedURL: URL?

    public var id: String { descriptor.family.rawValue }
    public var isInstalled: Bool { installedURL != nil }

    public init(descriptor: AppModelInstallDescriptor, installedURL: URL?) {
        self.descriptor = descriptor
        self.installedURL = installedURL
    }
}

/// One dedicated, auto-detected place for models. The library scans its
/// roots for `.gturbo` directories, identifies each by the family its own
/// manifest declares, and reports every shipped model as installed or
/// downloadable — no folder picking required.
public enum ModelLibrary {
    /// Optional override for the primary library root
    /// (`defaults write <app> Mference.libraryRoot <path>`).
    public static let rootStorageKey = "Mference.libraryRoot"

    public static let shippedDescriptors: [AppModelInstallDescriptor] = [
        .default, .qwen36, .deepseekV4Flash, .inklingSmall,
    ]

    /// Where new downloads land: the first library root.
    public static func downloadRootURL(
        rememberedDirectory: URL? = nil,
        userDefaults: UserDefaults = .standard
    ) -> URL {
        candidateRoots(
            rememberedDirectory: rememberedDirectory,
            userDefaults: userDefaults)[0]
    }

    public static func defaultInstallURL(
        for descriptor: AppModelInstallDescriptor,
        rememberedDirectory: URL? = nil,
        userDefaults: UserDefaults = .standard
    ) -> URL {
        downloadRootURL(
            rememberedDirectory: rememberedDirectory,
            userDefaults: userDefaults)
            .appendingPathComponent(descriptor.installDirectoryName, isDirectory: true)
    }

    /// Scans the library roots and returns one entry per shipped model.
    public static func catalog(
        rememberedDirectory: URL? = nil,
        userDefaults: UserDefaults = .standard
    ) -> [ModelCatalogEntry] {
        catalog(
            roots: candidateRoots(
                rememberedDirectory: rememberedDirectory,
                userDefaults: userDefaults),
            listSubdirectories: gturboSubdirectories(of:),
            probe: { directory in
                guard let family = try? ManifestReader.peekFamily(directoryURL: directory) else {
                    return nil
                }
                return (family, AppModelInstallationProbe.status(at: directory))
            })
    }

    /// Roots searched in priority order: the explicit override, the package
    /// checkout's `scratch/` when running from a development tree, the
    /// per-user Application Support library, and the parent of the last
    /// remembered model directory (so existing installs are adopted).
    static func candidateRoots(
        rememberedDirectory: URL?,
        userDefaults: UserDefaults = .standard,
        fileManager: FileManager = .default
    ) -> [URL] {
        var roots: [URL] = []
        if let override = userDefaults.string(forKey: rootStorageKey), !override.isEmpty {
            roots.append(URL(fileURLWithPath: override, isDirectory: true))
        }
        let executableStart = Bundle.main.executableURL?.deletingLastPathComponent()
        let currentDirectory = URL(
            fileURLWithPath: fileManager.currentDirectoryPath,
            isDirectory: true)
        for start in [executableStart, currentDirectory].compactMap({ $0 }) {
            if let packageRoot = AppModelLocation.packageRoot(
                startingAt: start,
                fileExists: fileManager.fileExists(atPath:)) {
                roots.append(packageRoot.appendingPathComponent("scratch", isDirectory: true))
                break
            }
        }
        let applicationSupport = (try? fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: false)) ?? fileManager.homeDirectoryForCurrentUser
        roots.append(applicationSupport.appendingPathComponent("Mference", isDirectory: true))
        if let rememberedDirectory {
            roots.append(rememberedDirectory.deletingLastPathComponent())
        }

        var seen = Set<String>()
        return roots.compactMap { root in
            let standardized = root.standardizedFileURL
            guard seen.insert(standardized.path).inserted else { return nil }
            return standardized
        }
    }

    /// Testable core: earlier roots win when a family is installed in more
    /// than one place; incomplete or unrecognized directories are skipped.
    static func catalog(
        roots: [URL],
        listSubdirectories: (URL) -> [URL],
        probe: (URL) -> (family: ModelFamily, status: AppModelInstallationStatus)?
    ) -> [ModelCatalogEntry] {
        var installed: [ModelFamily: URL] = [:]
        for root in roots {
            for directory in listSubdirectories(root) {
                guard let result = probe(directory),
                      result.status == .complete,
                      installed[result.family] == nil else { continue }
                installed[result.family] = directory
            }
        }
        return shippedDescriptors.map { descriptor in
            ModelCatalogEntry(
                descriptor: descriptor,
                installedURL: installed[descriptor.family])
        }
    }

    private static func gturboSubdirectories(of root: URL) -> [URL] {
        let contents = (try? FileManager.default.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles])) ?? []
        return contents
            .filter { $0.pathExtension == "gturbo" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }
}
