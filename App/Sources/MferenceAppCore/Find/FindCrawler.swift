import CoreServices
import Foundation

/// Which files Find reads (canvas boards 4 and 5): hidden folders, apps,
/// caches, keys and build output are never read; code projects only when
/// Settings › Find › Code is on.
public struct FindCrawlRules: Sendable, Hashable {
    /// Read code in git projects (tracked files only).
    public var includeCode: Bool
    /// Read text in screenshots and scans.
    public var readImages: Bool
    /// Learn photos, video and audio by what they show and sound like.
    public var readMedia: Bool

    public init(includeCode: Bool = false, readImages: Bool = true, readMedia: Bool = false) {
        self.includeCode = includeCode
        self.readImages = readImages
        self.readMedia = readMedia
    }

    /// Folder names never entered.
    public static let skippedFolders: Set<String> = [
        "node_modules", "Library", "DerivedData", "Pods", ".build", "build", "target", "dist", "venv", ".venv",
        "__pycache__", "Caches", "cache", ".cache", ".Trash", "Photos Library.photoslibrary", "site-packages",
    ]

    /// Extensions never read (keys, certificates, password stores).
    public static let secretExtensions: Set<String> = [
        "key", "pem", "p12", "pfx", "cer", "crt", "keychain", "keychain-db", "kdbx", "gpg", "asc", "ovpn",
    ]

    /// The Settings list of skipped things, as shown.
    public static let skippedSummary = ["hidden folders", "node_modules", ".git", "Library", "*.key, *.pem",
                                        "files over 200 MB"]

    public func skipsFolder(named name: String) -> Bool {
        name.hasPrefix(".") || Self.skippedFolders.contains(name) || name.hasSuffix(".app")
            || name.hasSuffix(".photoslibrary") || name.hasSuffix(".xcarchive")
    }

    /// The kind Find reads `url` as, or nil to leave it out.
    public func kind(of url: URL) -> FindKind? {
        let name = url.lastPathComponent
        guard !name.hasPrefix("."), !name.hasPrefix("~$") else { return nil }
        let ext = url.pathExtension.lowercased()
        // Keynote decks are .key too; keys and certificates are skipped by
        // name only when they are not Keynote.
        if Self.secretExtensions.contains(ext), ext != "key" { return nil }
        guard let kind = FindKind.of(extension: ext) else { return nil }
        if kind == .code, !includeCode { return nil }
        return kind
    }
}

/// One file found on disk.
public struct FindCrawlEntry: Sendable, Hashable {
    public var url: URL
    public var kind: FindKind
    public var size: Int64
    public var modified: Date
    public var inode: Int64
    public var device: Int64
}

/// Walks folders (metadata only; nothing is read here).
public enum FindCrawler {
    static let keys: [URLResourceKey] = [
        .isDirectoryKey, .isRegularFileKey, .isPackageKey, .fileSizeKey, .contentModificationDateKey,
        .isSymbolicLinkKey,
    ]

    /// Every readable file under `root`.
    public static func crawl(root: URL, rules: FindCrawlRules,
                             isCancelled: @Sendable () -> Bool = { false }) -> [FindCrawlEntry] {
        var entries: [FindCrawlEntry] = []
        crawl(directory: root, rules: rules, into: &entries, isCancelled: isCancelled)
        return entries
    }

    private static func crawl(directory: URL, rules: FindCrawlRules, into entries: inout [FindCrawlEntry],
                              isCancelled: @Sendable () -> Bool) {
        let manager = FileManager.default
        // A git project: only its tracked code, and only with Code on.
        if manager.fileExists(atPath: directory.appending(path: ".git").path) {
            guard rules.includeCode else { return }
            let tracked = GitFiles.tracked(in: directory)
            for url in tracked ?? [] where !isCancelled() {
                if let entry = entry(for: url, rules: rules) { entries.append(entry) }
            }
            if tracked != nil { return }
        }
        guard let children = try? manager.contentsOfDirectory(at: directory, includingPropertiesForKeys: keys,
                                                               options: [.skipsHiddenFiles]) else { return }
        for child in children {
            if isCancelled() { return }
            guard let values = try? child.resourceValues(forKeys: Set(keys)), values.isSymbolicLink != true else {
                continue
            }
            if values.isDirectory == true, values.isPackage != true {
                if !rules.skipsFolder(named: child.lastPathComponent) {
                    crawl(directory: child, rules: rules, into: &entries, isCancelled: isCancelled)
                }
            } else if let entry = entry(for: child, values: values, rules: rules) {
                entries.append(entry)
            }
        }
    }

    /// The entry for one file, or nil when Find does not read it.
    public static func entry(for url: URL, rules: FindCrawlRules) -> FindCrawlEntry? {
        guard let values = try? url.resourceValues(forKeys: Set(keys)), values.isSymbolicLink != true else {
            return nil
        }
        if values.isDirectory == true, values.isPackage != true { return nil }
        return entry(for: url, values: values, rules: rules)
    }

    private static func entry(for url: URL, values: URLResourceValues, rules: FindCrawlRules) -> FindCrawlEntry? {
        guard let kind = rules.kind(of: url) else { return nil }
        // The inode and device tell a moved file from a new one.
        var info = stat()
        let found = lstat(url.path, &info) == 0
        let inode = found ? Int64(bitPattern: UInt64(info.st_ino)) : 0
        let device = found ? Int64(info.st_dev) : 0
        return FindCrawlEntry(url: url.standardizedFileURL, kind: kind, size: Int64(values.fileSize ?? 0),
                              modified: values.contentModificationDate ?? .distantPast, inode: inode, device: device)
    }

    /// Whether `url` is inside a folder Find skips, relative to `root`.
    public static func isInsideSkippedFolder(_ url: URL, root: URL, rules: FindCrawlRules) -> Bool {
        let rootComponents = root.standardizedFileURL.pathComponents
        let components = url.standardizedFileURL.pathComponents
        guard components.count > rootComponents.count, Array(components.prefix(rootComponents.count)) == rootComponents
        else { return true }
        var current = root.standardizedFileURL
        for name in components.dropFirst(rootComponents.count).dropLast() {
            if rules.skipsFolder(named: name) { return true }
            current.append(path: name)
            if !rules.includeCode, FileManager.default.fileExists(atPath: current.appending(path: ".git").path) {
                return true
            }
        }
        return false
    }
}

/// `git ls-files` for a project, when a developer git is installed (the
/// `/usr/bin/git` shim would ask to install the command line tools, so it
/// is never called).
enum GitFiles {
    static let candidates = [
        "/Library/Developer/CommandLineTools/usr/bin/git",
        "/Applications/Xcode.app/Contents/Developer/usr/bin/git",
        "/opt/homebrew/bin/git",
        "/usr/local/bin/git",
    ]

    static func tracked(in project: URL) -> [URL]? {
        guard let git = candidates.first(where: { FileManager.default.isExecutableFile(atPath: $0) }) else {
            return nil
        }
        let process = Process()
        process.executableURL = URL(filePath: git)
        process.arguments = ["-C", project.path, "ls-files", "-z"]
        let output = Pipe()
        process.standardOutput = output
        process.standardError = FileHandle.nullDevice
        do { try process.run() } catch { return nil }
        let data = output.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { return nil }
        return data.split(separator: 0).map { project.appending(path: String(decoding: $0, as: UTF8.self)) }
    }
}

/// Files the person opened recently, from Spotlight's own record
/// (`kMDItemLastUsedDate`), so they are read first.
public enum FindRecentFiles {
    public static func lastUsed(under roots: [URL], days: Int = 120) -> [String: Date] {
        guard !roots.isEmpty,
              let query = MDQueryCreate(kCFAllocatorDefault,
                                        "kMDItemLastUsedDate >= $time.today(-\(days))" as CFString, nil, nil)
        else { return [:] }
        MDQuerySetSearchScope(query, roots.map { $0.path } as CFArray, 0)
        guard MDQueryExecute(query, CFOptionFlags(kMDQuerySynchronous.rawValue)) else { return [:] }
        var dates: [String: Date] = [:]
        for index in 0..<MDQueryGetResultCount(query) {
            guard let raw = MDQueryGetResultAtIndex(query, index) else { continue }
            let item = Unmanaged<MDItem>.fromOpaque(raw).takeUnretainedValue()
            guard let path = MDItemCopyAttribute(item, kMDItemPath) as? String,
                  let date = MDItemCopyAttribute(item, kMDItemLastUsedDate) as? Date else { continue }
            dates[path] = date
        }
        return dates
    }
}

/// Watches folders with FSEvents (file-level events). The stream starts
/// from the last event id Find saw, so changes made while the app was closed
/// are replayed.
public final class FindFolderWatcher: @unchecked Sendable {
    public struct Batch: Sendable {
        public var paths: [String]
        /// FSEvents lost track (overflow, or a root changed): rescan.
        public var needsRescan: Bool
        public var lastEventID: UInt64
    }

    private var stream: FSEventStreamRef?
    private let queue = DispatchQueue(label: "ai.mference.find.watcher", qos: .utility)
    private let handler: @Sendable (Batch) -> Void

    public init(handler: @escaping @Sendable (Batch) -> Void) {
        self.handler = handler
    }

    deinit { stop() }

    /// Starts watching `roots` from `since` (nil: from now).
    public func start(roots: [URL], since: UInt64?) {
        stop()
        guard !roots.isEmpty else { return }
        var context = FSEventStreamContext(version: 0, info: Unmanaged.passUnretained(self).toOpaque(),
                                           retain: nil, release: nil, copyDescription: nil)
        let callback: FSEventStreamCallback = { _, info, count, paths, flags, ids in
            guard let info else { return }
            let watcher = Unmanaged<FindFolderWatcher>.fromOpaque(info).takeUnretainedValue()
            let array = Unmanaged<CFArray>.fromOpaque(paths).takeUnretainedValue() as? [String] ?? []
            var rescan = false
            var last: UInt64 = 0
            for index in 0..<count {
                let flag = Int(flags[index])
                if flag & (kFSEventStreamEventFlagMustScanSubDirs | kFSEventStreamEventFlagRootChanged
                    | kFSEventStreamEventFlagEventIdsWrapped | kFSEventStreamEventFlagKernelDropped
                    | kFSEventStreamEventFlagUserDropped) != 0 {
                    rescan = true
                }
                last = max(last, ids[index])
            }
            watcher.handler(Batch(paths: array, needsRescan: rescan, lastEventID: last))
        }
        let flags = UInt32(kFSEventStreamCreateFlagFileEvents | kFSEventStreamCreateFlagUseCFTypes
            | kFSEventStreamCreateFlagWatchRoot)
        guard let stream = FSEventStreamCreate(kCFAllocatorDefault, callback, &context,
                                               roots.map(\.path) as CFArray,
                                               since ?? FSEventStreamEventId(kFSEventStreamEventIdSinceNow),
                                               1.0, flags) else { return }
        FSEventStreamSetDispatchQueue(stream, queue)
        FSEventStreamStart(stream)
        self.stream = stream
    }

    public func stop() {
        guard let stream else { return }
        FSEventStreamStop(stream)
        FSEventStreamInvalidate(stream)
        FSEventStreamRelease(stream)
        self.stream = nil
    }

    /// The newest event id on this Mac, saved when no batch came yet.
    public static var currentEventID: UInt64 { FSEventsGetCurrentEventId() }
}
