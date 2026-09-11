import Foundation
import Mference

/// API model identifier for a family, derived without loading anything.
///
/// Library mode has to name every install in `GET /v1/models` before any of
/// them is loaded, so the mapping cannot come from a live `ServerModelSession`.
/// This is deliberately a second copy of `ServerModelSession.defaultModelID`'s
/// switch, and deliberately a `switch` rather than a dictionary: adding a
/// `ModelFamily` case fails the build here until it is given an identifier, so
/// the two copies cannot drift silently.
public enum ServerFamilyModelID {
    public static func modelID(for family: ModelFamily) -> String {
        switch family {
        case .gemma4: "gemma-4-26b-a4b-it"
        case .qwen36: "qwen3.6-35b-a3b"
        case .qwen38: "qwen3.8-27b-4bit"
        case .deepseekV4Flash: "deepseek-v4-flash-2bit-dq"
        case .inklingSmall: "inkling-small-4bit"
        case .maple: "maple-preview-2bit-mlx"
        case .qwen38flashnext: "qwen3.8-flash-next-int4g64"
        case .minicpm5: "minicpm5-2b-int4g64"
        case .glm53Flash: "glm-5.3-flash-mlx-mixed-4-8bit"
        }
    }

    /// Identifier for a raw `arch.family` string, or nil when the runtime does
    /// not know the family at all. Used for the installs `peekFamily` refuses
    /// by name, which never reach a `ModelFamily` value through it.
    public static func modelID(forRawFamily raw: String) -> String? {
        ModelFamily(rawValue: raw).map(modelID(for:))
    }
}

/// What a directory turned out to be when the library looked at it.
public enum ServerLibraryProbeResult: Equatable, Sendable {
    /// No `manifest.json`: an ordinary directory, not a partial install.
    case notAModelDirectory
    /// A manifest is present but the install is not usable, with the reason.
    case partial(String)
    /// A complete install of a family no runner can execute.
    case notRunnable(family: String, detail: String)
    case complete(family: ModelFamily)
}

/// Completeness probe for one candidate directory.
///
/// Checks manifest, family, arch baseline, `packed_experts/layout.json`, and a
/// receipt bound to the manifest, with one deliberate omission: it does not
/// compare `sourceSnapshotHash` against a pinned descriptor. The server has
/// always taken an explicit `--model` path and never pinned a checkpoint, and a
/// locally requantized repack (`qwen36-ourquant.gturbo`) carries a different
/// snapshot hash while being exactly the thing an operator wants to serve.
///
/// Self-contained on purpose: this is the only install probe left, and it reads
/// nothing but files the installer writes.
public enum ServerLibraryProbe {
    /// `gatedFamilies` overrides the runtime's capability gate table; tests use
    /// it to exercise the not-runnable branch now that the table ships empty.
    public static func probe(directory: URL,
                             gatedFamilies: [String: [String]]? = nil) -> ServerLibraryProbeResult {
        let fileManager = FileManager.default
        let directory = directory.standardizedFileURL
        let name = directory.lastPathComponent
        // `MferenceRepack` stages into a `<name>.partial` sibling directory and
        // holds a `<name>.install.lock` sibling file for the duration, so both
        // mark an install that is still being written.
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

        // A gated family has to be told apart from a broken install, and
        // `peekFamily` reports both by throwing. Asking the runner table first
        // keeps the two apart without matching on error text, and the moment a
        // family's gate lifts its entry disappears and the install simply falls
        // through to `peekFamily` and gets listed.
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
            let manifestHash = try Sha256Verifier.hashFile(at: manifestURL,
                                                           chunkBytes: 65_536)
            try VerifiedInstallReceiptReader.validateManifestBinding(
                receipt,
                directoryURL: directory,
                manifestSha256: manifestHash)
        } catch {
            return .partial("\(error)")
        }
        return .complete(family: family)
    }

    /// `arch.family` as the manifest spells it, or nil when the file cannot be
    /// read that far. Size-capped the same way `ManifestReader` caps metadata.
    private static func rawFamily(manifestURL: URL) -> String? {
        struct FamilyPeek: Decodable {
            struct Arch: Decodable { let family: String? }
            let arch: Arch
        }
        guard let attributes = try? FileManager.default
            .attributesOfItem(atPath: manifestURL.path),
              let size = attributes[.size] as? NSNumber,
              size.uint64Value <= ManifestReader.defaultMaxBytes,
              let data = try? Data(contentsOf: manifestURL) else { return nil }
        return (try? JSONDecoder().decode(FamilyPeek.self, from: data))?.arch.family
    }

    /// Whether a repacker currently holds the sibling install lock.
    ///
    /// `InstallLock` takes `flock(LOCK_EX | LOCK_NB)` on the file and leaves
    /// the file behind when it is done, so every completed install has a
    /// stale zero-byte lock file next to it. Only the advisory lock itself
    /// says an install is running; the file's existence says nothing.
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

/// One completed, runnable install the library can serve.
public struct ServerLibraryEntry: Equatable, Sendable {
    /// Identifier clients send in `model`. Equals `familyModelID` unless two
    /// installs share a family.
    public let modelID: String
    /// The family's identifier, before disambiguation.
    public let familyModelID: String
    /// Directory name with a trailing `.gturbo` removed.
    public let basename: String
    public let directory: URL
    public let family: ModelFamily

    public init(modelID: String,
                familyModelID: String,
                basename: String,
                directory: URL,
                family: ModelFamily) {
        self.modelID = modelID
        self.familyModelID = familyModelID
        self.basename = basename
        self.directory = directory
        self.family = family
    }
}

/// A candidate directory the library declined to advertise, and why. Reported
/// on stderr at startup so a partial or gated install is visible rather than
/// silently absent from the model picker.
public struct ServerLibrarySkip: Equatable, Sendable {
    public let directory: URL
    public let reason: String

    public init(directory: URL, reason: String) {
        self.directory = directory
        self.reason = reason
    }
}

/// The set of installs the server advertises, fixed at startup.
public struct ServerLibraryIndex: Equatable, Sendable {
    /// Sorted by `modelID`, so `/v1/models` order is stable across restarts.
    public let entries: [ServerLibraryEntry]
    public let skipped: [ServerLibrarySkip]

    public init(entries: [ServerLibraryEntry], skipped: [ServerLibrarySkip] = []) {
        self.entries = entries.sorted { $0.modelID < $1.modelID }
        self.skipped = skipped
    }

    public func entry(for modelID: String) -> ServerLibraryEntry? {
        entries.first { $0.modelID == modelID }
    }

    public var modelList: OpenAIModelList {
        OpenAIModelList(
            object: "list",
            data: entries.map {
                .init(id: $0.modelID, object: "model", created: 0, ownedBy: "mference")
            })
    }

    /// Writes what discovery found and what it declined to stderr in the
    /// `ServerLog` format. `ServerLog` is internal to the core, so the
    /// executable reports through here.
    public func logDiscovery() {
        for skip in skipped {
            ServerLog.librarySkipped(directory: skip.directory.path, reason: skip.reason)
        }
        ServerLog.libraryReady(models: entries.map(\.modelID))
    }
}

/// Library roots and the walk over them.
public enum ServerLibraryDiscovery {
    /// `defaults write Mference libraryRoot <path>`.
    public static let libraryRootDefaultsKey = "Mference.libraryRoot"
    /// Environment equivalent, for a shell that cannot reach user defaults.
    public static let libraryRootEnvironmentKey = "MFERENCE_LIBRARY_ROOT"

    /// The default library roots, in order: the `Mference.libraryRoot` default
    /// if set, the package checkout's `scratch/`, then
    /// `~/Library/Application Support/Mference` — the three places
    /// `MferenceRepack` is pointed at in practice.
    public static func defaultRoots(
        userDefaults: UserDefaults = .standard,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        executableURL: URL? = Bundle.main.executableURL,
        currentDirectoryURL: URL = URL(
            fileURLWithPath: FileManager.default.currentDirectoryPath,
            isDirectory: true),
        applicationSupportURL: URL? = try? FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: false),
        fileExists: (String) -> Bool = FileManager.default.fileExists(atPath:)
    ) -> [URL] {
        var roots: [URL] = []
        let configured = environment[libraryRootEnvironmentKey]
            ?? userDefaults.string(forKey: libraryRootDefaultsKey)
        if let configured, !configured.isEmpty {
            roots.append(URL(fileURLWithPath: configured, isDirectory: true)
                .standardizedFileURL)
        }
        let packageStart = executableURL?.deletingLastPathComponent()
            ?? currentDirectoryURL
        if let root = packageRoot(startingAt: packageStart, fileExists: fileExists)
            ?? packageRoot(startingAt: currentDirectoryURL, fileExists: fileExists) {
            roots.append(root.appendingPathComponent("scratch", isDirectory: true)
                .standardizedFileURL)
        }
        if let applicationSupportURL {
            roots.append(applicationSupportURL
                .appendingPathComponent("Mference", isDirectory: true)
                .standardizedFileURL)
        }
        return roots
    }

    /// Walks `roots` and returns the installs worth advertising.
    ///
    /// Each root contributes itself plus its immediate children, so a root may
    /// be either a directory of `.gturbo` installs (what `scratch/` is) or a
    /// single install. `explicitModelDirectory` — a `--model` passed alongside
    /// `--library` — is probed too, so preloading a directory outside every root
    /// still advertises it.
    ///
    /// Identifiers: an install alone in its family gets the family identifier.
    /// When two installs share a family, every install in that family gets
    /// `<family id>@<directory basename minus .gturbo>`; if that still collides
    /// (the same basename under two roots), the second and later get `#2`,
    /// `#3`, … in path order. The whole assignment depends only on the set of
    /// directories found, never on discovery order.
    public static func discover(
        roots: [URL],
        explicitModelDirectory: URL? = nil,
        childDirectories: (URL) -> [URL] = ServerLibraryDiscovery.childDirectories(of:),
        probe: (URL) -> ServerLibraryProbeResult = { ServerLibraryProbe.probe(directory: $0) }
    ) -> ServerLibraryIndex {
        var candidates: [URL] = []
        var seen = Set<String>()
        func consider(_ url: URL) {
            let standardized = url.standardizedFileURL
            guard seen.insert(standardized.path).inserted else { return }
            candidates.append(standardized)
        }
        if let explicitModelDirectory { consider(explicitModelDirectory) }
        for root in roots {
            consider(root)
            for child in childDirectories(root.standardizedFileURL) { consider(child) }
        }

        var found: [(basename: String, directory: URL, family: ModelFamily)] = []
        var skipped: [ServerLibrarySkip] = []
        for candidate in candidates {
            switch probe(candidate) {
            case .notAModelDirectory:
                continue
            case .partial(let reason):
                skipped.append(ServerLibrarySkip(directory: candidate,
                                                 reason: "incomplete install: \(reason)"))
            case .notRunnable(let family, let detail):
                let identifier = ServerFamilyModelID.modelID(forRawFamily: family)
                    ?? family
                skipped.append(ServerLibrarySkip(
                    directory: candidate,
                    reason: "not runnable (\(identifier)): \(detail)"))
            case .complete(let family):
                found.append((basename: strippedBasename(candidate),
                              directory: candidate,
                              family: family))
            }
        }

        return ServerLibraryIndex(entries: assignIdentifiers(found), skipped: skipped)
    }

    private static func assignIdentifiers(
        _ found: [(basename: String, directory: URL, family: ModelFamily)]
    ) -> [ServerLibraryEntry] {
        let sorted = found.sorted {
            let left = (ServerFamilyModelID.modelID(for: $0.family), $0.basename,
                        $0.directory.path)
            let right = (ServerFamilyModelID.modelID(for: $1.family), $1.basename,
                         $1.directory.path)
            return left < right
        }
        var familyCounts: [String: Int] = [:]
        for item in sorted {
            familyCounts[ServerFamilyModelID.modelID(for: item.family), default: 0] += 1
        }
        var used: [String: Int] = [:]
        var entries: [ServerLibraryEntry] = []
        for item in sorted {
            let familyModelID = ServerFamilyModelID.modelID(for: item.family)
            var modelID = familyModelID
            if familyCounts[familyModelID, default: 0] > 1 {
                modelID = "\(familyModelID)@\(item.basename)"
            }
            let occurrence = used[modelID, default: 0]
            used[modelID] = occurrence + 1
            if occurrence > 0 { modelID = "\(modelID)#\(occurrence + 1)" }
            entries.append(ServerLibraryEntry(modelID: modelID,
                                              familyModelID: familyModelID,
                                              basename: item.basename,
                                              directory: item.directory,
                                              family: item.family))
        }
        return entries
    }

    /// Immediate subdirectories of `root`, or none when it is not readable.
    /// Discovery is metadata-only: it reads manifests and receipts and never
    /// opens a weight file.
    public static func childDirectories(of root: URL) -> [URL] {
        guard let contents = try? FileManager.default.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]) else { return [] }
        return contents.filter {
            (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
        }.sorted { $0.path < $1.path }
    }

    private static func strippedBasename(_ directory: URL) -> String {
        let name = directory.lastPathComponent
        guard name.hasSuffix(".gturbo") else { return name }
        return String(name.dropLast(".gturbo".count))
    }

    /// Nearest ancestor that is this package's checkout: `Package.swift` beside
    /// the two source trees the server is itself built from.
    ///
    /// The marker is deliberately not `Sources/MferenceApp`. The browser UI is
    /// the frontend now and the Mac app is gone, so a checkout has to be
    /// recognized — and its `scratch/` still scanned — with no app sources on
    /// disk. Two markers rather than one keep an unrelated Swift package that
    /// happens to be an ancestor from being taken for this checkout.
    private static func packageRoot(startingAt start: URL,
                                    fileExists: (String) -> Bool) -> URL? {
        var candidatePath = start.standardizedFileURL.path
        while true {
            let candidate = URL(fileURLWithPath: candidatePath, isDirectory: true)
            let package = candidate.appendingPathComponent("Package.swift").path
            let runtimeSources = candidate.appendingPathComponent(
                "Sources/Mference", isDirectory: true).path
            let serverSources = candidate.appendingPathComponent(
                "Sources/MferenceServer", isDirectory: true).path
            if fileExists(package), fileExists(runtimeSources), fileExists(serverSources) {
                return candidate
            }
            let parentPath = (candidatePath as NSString).deletingLastPathComponent
            if parentPath.isEmpty || parentPath == candidatePath { return nil }
            candidatePath = parentPath
        }
    }
}
