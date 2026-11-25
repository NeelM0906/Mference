import Foundation
import Mference

/// Finds the models an `EngineCatalog` lists: `.gturbo` installs under the
/// library roots and `.gguf` files in the GGUF directories, filtered by the
/// product allowlist (spec D1) unless the scope is `.all`.
///
/// Discovery reads manifests, receipts, and GGUF headers only. It never opens
/// a weight file or a tensor.
public enum EngineCatalogDiscovery {
    /// `defaults write Mference libraryRoot <path>`.
    public static let libraryRootDefaultsKey = "Mference.libraryRoot"
    /// Environment equivalent, for a shell that cannot reach user defaults.
    public static let libraryRootEnvironmentKey = "MFERENCE_LIBRARY_ROOT"

    /// The product GGUF files, by exact file name (spec D1, D2).
    public static let productGGUFFiles: [String: EngineProductSlot] = [
        "Qwen3.5-4B-UD-Q4_K_XL.gguf": .quick,
        "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf": .deep,
    ]

    /// The vision projector files of the GGUF product models, by the names
    /// they are preferred under when a directory holds several that fit: the
    /// name the app downloads to first, then the repository's own.
    public static let productProjectorFiles: [EngineProductSlot: [String]] = [
        .quick: ["Qwen3.5-4B-mmproj-F16.gguf", "mmproj-F16.gguf"],
        .deep: ["mmproj-Qwen3.8-27B-BF16.gguf"],
    ]

    /// Whether a `.gguf` file name is a multimodal projector's, by
    /// llama.cpp's naming convention (`mmproj` anywhere in it). Such a file is
    /// never listed as a model; it is paired with one.
    public static func isProjectorFileName(_ name: String) -> Bool {
        name.lowercased().contains("mmproj")
    }

    /// The directory label `MferenceRepack` gives each native product family.
    /// When several installs of one product model are found, the one in the
    /// earliest root wins, and within a root the one with this label.
    static func installLabel(_ slot: EngineProductSlot) -> String? {
        switch slot {
        case .qwen36: "qwen36"
        case .minicpm5: "minicpm5"
        case .maple: "maple"
        case .inkling2Mini: "inkling2mini"
        case .quick, .deep: nil
        }
    }

    /// File-system access, injectable so tests run without installs.
    public struct FileSystem: Sendable {
        public var childDirectories: @Sendable (URL) -> [URL]
        public var ggufFiles: @Sendable (URL) -> [URL]
        public var probe: @Sendable (URL) -> EngineInstallProbeResult
        public var checkpointID: @Sendable (URL) -> String?
        public var readGGUF: @Sendable (URL) throws -> EngineGGUFMetadata
        /// Bytes on disk of a file, or of every regular file under a directory.
        public var size: @Sendable (URL) -> UInt64

        public init(childDirectories: @escaping @Sendable (URL) -> [URL],
                    ggufFiles: @escaping @Sendable (URL) -> [URL],
                    probe: @escaping @Sendable (URL) -> EngineInstallProbeResult,
                    checkpointID: @escaping @Sendable (URL) -> String?,
                    readGGUF: @escaping @Sendable (URL) throws -> EngineGGUFMetadata,
                    size: @escaping @Sendable (URL) -> UInt64) {
            self.childDirectories = childDirectories
            self.ggufFiles = ggufFiles
            self.probe = probe
            self.checkpointID = checkpointID
            self.readGGUF = readGGUF
            self.size = size
        }

        public static let live = FileSystem(
            childDirectories: { EngineCatalogDiscovery.childDirectories(of: $0) },
            ggufFiles: { EngineCatalogDiscovery.ggufFiles(in: $0) },
            probe: { EngineInstallProbe.probe(directory: $0) },
            checkpointID: { try? ManifestReader.peekModelID(directoryURL: $0) },
            readGGUF: { try EngineGGUFReader.read(url: $0) },
            size: { EngineCatalogDiscovery.bytesOnDisk($0) })
    }

    // MARK: Roots

    /// The library roots, in order: the `Mference.libraryRoot` default (or
    /// `MFERENCE_LIBRARY_ROOT`) if set, `~/llm-models`, the package checkout's
    /// `scratch/`, then `~/Library/Application Support/Mference`. The same
    /// roots `MferenceServer --library` has always scanned.
    public static func defaultRoots(
        userDefaults: UserDefaults = .standard,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        executableURL: URL? = Bundle.main.executableURL,
        currentDirectoryURL: URL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath,
                                       isDirectory: true),
        applicationSupportURL: URL? = try? FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: false),
        homeDirectoryURL: URL = FileManager.default.homeDirectoryForCurrentUser,
        fileExists: (String) -> Bool = FileManager.default.fileExists(atPath:)
    ) -> [URL] {
        var roots: [URL] = []
        let configured = environment[libraryRootEnvironmentKey]
            ?? userDefaults.string(forKey: libraryRootDefaultsKey)
        if let configured, !configured.isEmpty {
            roots.append(URL(fileURLWithPath: configured, isDirectory: true).standardizedFileURL)
        }
        roots.append(homeDirectoryURL.appendingPathComponent("llm-models", isDirectory: true)
            .standardizedFileURL)
        let packageStart = executableURL?.deletingLastPathComponent() ?? currentDirectoryURL
        if let root = packageRoot(startingAt: packageStart, fileExists: fileExists)
            ?? packageRoot(startingAt: currentDirectoryURL, fileExists: fileExists) {
            roots.append(root.appendingPathComponent("scratch", isDirectory: true).standardizedFileURL)
        }
        if let applicationSupportURL {
            roots.append(applicationSupportURL.appendingPathComponent("Mference", isDirectory: true)
                .standardizedFileURL)
        }
        return unique(roots)
    }

    /// Where GGUF files are looked for: each library root, then
    /// `~/llm-models/gguf`.
    public static func defaultGGUFDirectories(
        roots: [URL],
        homeDirectoryURL: URL = FileManager.default.homeDirectoryForCurrentUser
    ) -> [URL] {
        unique(roots + [homeDirectoryURL.appendingPathComponent("llm-models/gguf", isDirectory: true)
            .standardizedFileURL])
    }

    // MARK: Discovery

    /// Walks the roots and GGUF directories and builds the catalog.
    ///
    /// Native installs: each root contributes itself and its immediate
    /// children. A product install takes its product identifier; when several
    /// installs of one product model exist, the earliest root wins (then the
    /// one labelled as `MferenceRepack` labels it, then path order) and the
    /// rest are skipped with the winner named. With scope `.all`, every other
    /// runnable install is listed too, under the server's identifier rules:
    /// the family identifier, `@<basename>` when it is shared or taken, `#n`
    /// on a further collision.
    ///
    /// GGUF files: a product file is recognized by its exact name and must
    /// have a readable header with a context length; Deep must carry its MTP
    /// tensors. With scope `.all`, any other single-file GGUF is listed as
    /// its lowercased file name without the extension. Split GGUFs are
    /// skipped.
    ///
    /// `explicitModel` (a `--model` passed with `--library`) is considered
    /// first, ahead of every root.
    ///
    /// A native product model with a memory floor
    /// (`EngineProductSlot.minimumPhysicalMemory`) is skipped, in either
    /// scope, on a Mac with less `physicalMemory`, with the slot's reason
    /// under its identifier.
    public static func discover(roots: [URL],
                                ggufDirectories: [URL],
                                scope: EngineCatalogScope = .product,
                                explicitModel: URL? = nil,
                                fileSystem: FileSystem = .live,
                                physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory) -> EngineCatalog {
        var models: [EngineModelDescriptor] = []
        var skipped: [EngineCatalogSkip] = []
        let explicitIsGGUF = explicitModel?.pathExtension.lowercased() == "gguf"

        // Native installs.
        var candidates: [(url: URL, rank: Int)] = []
        var seen = Set<String>()
        func consider(_ url: URL, rank: Int) {
            let standardized = url.standardizedFileURL
            if seen.insert(standardized.path).inserted { candidates.append((standardized, rank)) }
        }
        if let explicitModel, !explicitIsGGUF { consider(explicitModel, rank: -1) }
        for (rank, root) in roots.enumerated() {
            consider(root, rank: rank)
            for child in fileSystem.childDirectories(root.standardizedFileURL) { consider(child, rank: rank) }
        }

        var found: [NativeInstall] = []
        for (url, rank) in candidates {
            switch fileSystem.probe(url) {
            case .notAModelDirectory:
                continue
            case .partial(let reason):
                skipped.append(EngineCatalogSkip(location: url, reason: "incomplete install: \(reason)"))
            case .notRunnable(let family, let detail):
                let checkpoint = fileSystem.checkpointID(url)
                let identifier = ModelFamily(rawValue: family).map {
                    EngineNativeModelID.modelID(for: $0, checkpointID: checkpoint)
                } ?? family
                skipped.append(EngineCatalogSkip(location: url,
                                                 reason: "not runnable (\(identifier)): \(detail)",
                                                 modelID: identifier))
            case .complete(let family):
                let checkpoint = fileSystem.checkpointID(url)
                found.append(NativeInstall(
                    directory: url, rank: rank, family: family, checkpointID: checkpoint,
                    identifier: EngineNativeModelID.modelID(for: family, checkpointID: checkpoint)))
            }
        }

        // Below a slot's memory floor, that is the reason a request gets, so
        // it goes ahead of any other skip under the same identifier.
        var floorSkips: [EngineCatalogSkip] = []
        for slot in EngineProductSlot.allCases where slot.engine == .mference {
            guard let reason = slot.memoryFloorReason(physicalMemory: physicalMemory) else { continue }
            for install in found where install.identifier == slot.modelID {
                floorSkips.append(EngineCatalogSkip(location: install.directory, reason: reason,
                                                    modelID: slot.modelID))
            }
            found.removeAll { $0.identifier == slot.modelID }
        }
        skipped = floorSkips + skipped

        var winners: [String: NativeInstall] = [:]
        for slot in EngineProductSlot.allCases where slot.engine == .mference {
            let contenders = found.filter { $0.identifier == slot.modelID }
            guard let winner = contenders.min(by: { left, right in
                let label = installLabel(slot)
                return (left.rank, left.basename == label ? 0 : 1, left.directory.path)
                    < (right.rank, right.basename == label ? 0 : 1, right.directory.path)
            }) else { continue }
            winners[slot.modelID] = winner
            models.append(nativeDescriptor(winner, id: slot.modelID, displayName: slot.displayName,
                                           product: slot, fileSystem: fileSystem, physicalMemory: physicalMemory))
        }
        let others = found.filter { winners[$0.identifier]?.directory != $0.directory }
        switch scope {
        case .product:
            for install in others {
                if let winner = winners[install.identifier] {
                    skipped.append(EngineCatalogSkip(
                        location: install.directory,
                        reason: "another install of \(install.identifier) is served from "
                            + winner.directory.path))
                } else {
                    skipped.append(EngineCatalogSkip(
                        location: install.directory,
                        reason: "\(install.identifier) is not in the product model list; "
                            + "the developer catalog scope lists it",
                        modelID: install.identifier))
                }
            }
        case .all:
            for (install, id) in assignIdentifiers(others, reserved: Set(winners.keys)) {
                models.append(nativeDescriptor(install, id: id, displayName: id,
                                               product: nil, fileSystem: fileSystem,
                                               physicalMemory: physicalMemory))
            }
        }

        // GGUF files.
        var files: [(url: URL, rank: Int)] = []
        var seenFiles = Set<String>()
        if let explicitModel, explicitIsGGUF {
            seenFiles.insert(explicitModel.standardizedFileURL.path)
            files.append((explicitModel.standardizedFileURL, -1))
        }
        for (rank, directory) in ggufDirectories.enumerated() {
            for file in fileSystem.ggufFiles(directory.standardizedFileURL)
            where seenFiles.insert(file.standardizedFileURL.path).inserted {
                files.append((file.standardizedFileURL, rank))
            }
        }
        // Product identifiers, native ones included, are reserved even before
        // their file or install is seen (or when it is skipped), so an
        // unrelated GGUF named like a product model can never take its
        // identifier: `Inkling-2-Mini-UD-Q4_K_XL.gguf` is the native slot's
        // source file, which llama.cpp does not run.
        let productIDs = Set(EngineProductSlot.allCases.map(\.modelID))
        var usedIDs = Set(models.map(\.id)).union(productIDs)
        var ggufWinners: [EngineProductSlot: URL] = [:]
        for (file, _) in files {
            let name = file.lastPathComponent
            // A projector is paired with the model beside it below.
            if isProjectorFileName(name), productGGUFFiles[name] == nil { continue }
            let slot = productGGUFFiles[name]
            let id = slot?.modelID ?? ggufModelID(fileName: name)
            if slot == nil, scope == .product {
                // Keep the identifier, so a request for it gets this reason
                // rather than "unknown model", unless it would shadow a
                // product model's own identifier.
                let reserved = productIDs.contains(id)
                skipped.append(EngineCatalogSkip(
                    location: file,
                    reason: "\(name) is not in the product model list; the developer catalog scope lists it",
                    modelID: reserved ? nil : id))
                continue
            }
            if isSplitShard(name) {
                skipped.append(EngineCatalogSkip(location: file,
                                                 reason: "split GGUF files are not supported", modelID: id))
                continue
            }
            if let slot, let winner = ggufWinners[slot] {
                skipped.append(EngineCatalogSkip(
                    location: file, reason: "another copy of \(id) is served from \(winner.path)"))
                continue
            }
            if slot == nil, usedIDs.contains(id) {
                skipped.append(EngineCatalogSkip(location: file,
                                                 reason: "another model already uses the identifier \(id)"))
                continue
            }
            let metadata: EngineGGUFMetadata
            do {
                metadata = try fileSystem.readGGUF(file)
            } catch {
                skipped.append(EngineCatalogSkip(location: file, reason: "unreadable GGUF: \(error)",
                                                 modelID: id))
                continue
            }
            guard let context = metadata.contextLength, context > 0 else {
                skipped.append(EngineCatalogSkip(location: file,
                                                 reason: "GGUF header has no context length", modelID: id))
                continue
            }
            if slot == .deep, !metadata.hasMTP {
                skipped.append(EngineCatalogSkip(
                    location: file, reason: "no MTP (nextn) tensors; Deep needs the -mtp file", modelID: id))
                continue
            }
            models.append(EngineModelDescriptor(
                id: id,
                displayName: slot?.displayName ?? metadata.name ?? name,
                engine: .llamaCpp,
                format: .gguf(metadata),
                location: file,
                sizeBytes: fileSystem.size(file),
                contextLength: context,
                product: slot))
            usedIDs.insert(id)
            if let slot { ggufWinners[slot] = file }
        }

        var projectors = ProjectorSearch(fileSystem: fileSystem,
                                         otherDirectories: ggufDirectories.map(\.standardizedFileURL))
        models = models.map { descriptor in
            guard case .gguf(let metadata) = descriptor.format else { return descriptor }
            return descriptor.pairing(projectors.projector(for: descriptor, metadata: metadata))
        }

        return EngineCatalog(models: models, skipped: skipped)
    }

    /// Discovery over the default roots and GGUF directories.
    public static func discoverDefault(scope: EngineCatalogScope = .product,
                                       explicitModel: URL? = nil,
                                       physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory)
        -> EngineCatalog {
        let roots = defaultRoots()
        return discover(roots: roots, ggufDirectories: defaultGGUFDirectories(roots: roots),
                        scope: scope, explicitModel: explicitModel, physicalMemory: physicalMemory)
    }

    // MARK: Helpers

    /// Finds the vision projector for a GGUF model among the `mmproj` files
    /// in its own directory: a projector with a vision encoder whose
    /// projection width is the model's embedding width. A product model
    /// prefers its `productProjectorFiles` names, then path order. When none
    /// is beside it, a product model also takes its own pinned projector from
    /// the other GGUF directories (the app downloads into
    /// `~/llm-models/gguf` while the model may sit in another library root);
    /// only those names, so another model's projector is never borrowed.
    /// Headers are read once per file.
    struct ProjectorSearch {
        let fileSystem: FileSystem
        let otherDirectories: [URL]
        private var candidates: [String: [URL]] = [:]
        private var headers: [String: EngineGGUFProjector?] = [:]

        init(fileSystem: FileSystem, otherDirectories: [URL] = []) {
            self.fileSystem = fileSystem
            self.otherDirectories = otherDirectories
        }

        mutating func projector(for model: EngineModelDescriptor,
                                metadata: EngineGGUFMetadata) -> EngineVisionProjector? {
            let directory = model.location.deletingLastPathComponent().standardizedFileURL
            let preferred = model.product.flatMap { productProjectorFiles[$0] } ?? []
            let beside = projectorFiles(in: directory).sorted { left, right in
                let leftRank = preferred.firstIndex(of: left.lastPathComponent) ?? preferred.count
                let rightRank = preferred.firstIndex(of: right.lastPathComponent) ?? preferred.count
                return (leftRank, left.path) < (rightRank, right.path)
            }
            if let found = firstFitting(beside, metadata: metadata) { return found }
            guard !preferred.isEmpty else { return nil }
            for other in otherDirectories where other.path != directory.path {
                let pinned = projectorFiles(in: other).filter { preferred.contains($0.lastPathComponent) }
                    .sorted { (preferred.firstIndex(of: $0.lastPathComponent) ?? 0)
                        < (preferred.firstIndex(of: $1.lastPathComponent) ?? 0) }
                if let found = firstFitting(pinned, metadata: metadata) { return found }
            }
            return nil
        }

        private mutating func projectorFiles(in directory: URL) -> [URL] {
            if let known = candidates[directory.path] { return known }
            let files = fileSystem.ggufFiles(directory)
                .filter { EngineCatalogDiscovery.isProjectorFileName($0.lastPathComponent) }
                .map(\.standardizedFileURL)
            candidates[directory.path] = files
            return files
        }

        private mutating func firstFitting(_ files: [URL], metadata: EngineGGUFMetadata) -> EngineVisionProjector? {
            for file in files {
                if headers[file.path] == nil {
                    headers[file.path] = .some((try? fileSystem.readGGUF(file))?.projector)
                }
                guard let header = headers[file.path] ?? nil, header.hasVisionEncoder,
                      let width = metadata.embeddingLength,
                      header.projectionDimension == width else { continue }
                return EngineVisionProjector(location: file, sizeBytes: fileSystem.size(file))
            }
            return nil
        }
    }
    struct NativeInstall {
        let directory: URL
        let rank: Int
        let family: ModelFamily
        let checkpointID: String?
        let identifier: String

        var basename: String {
            let name = directory.lastPathComponent
            return name.hasSuffix(".gturbo") ? String(name.dropLast(".gturbo".count)) : name
        }
    }

    /// A native install's descriptor. `hostContextCap` is the family's cap
    /// for this Mac's memory, if it has one (Inkling 2 under 48 GB).
    private static func nativeDescriptor(_ install: NativeInstall, id: String, displayName: String,
                                         product: EngineProductSlot?,
                                         fileSystem: FileSystem,
                                         physicalMemory: UInt64) -> EngineModelDescriptor {
        EngineModelDescriptor(
            id: id,
            displayName: displayName,
            engine: .mference,
            format: .gturbo(family: install.family, checkpointID: install.checkpointID),
            location: install.directory,
            sizeBytes: fileSystem.size(install.directory),
            contextLength: install.family.maximumContext,
            product: product,
            hostContextCap: install.family.memoryContextCap(physicalMemory: physicalMemory))
    }

    /// The server's rule: the family identifier when alone, else
    /// `<id>@<basename>`, then `#n` for a repeated basename. An identifier
    /// already taken by a product model counts as shared.
    private static func assignIdentifiers(_ installs: [NativeInstall],
                                          reserved: Set<String>) -> [(NativeInstall, String)] {
        let sorted = installs.sorted {
            ($0.identifier, $0.basename, $0.directory.path) < ($1.identifier, $1.basename, $1.directory.path)
        }
        var counts: [String: Int] = [:]
        for install in sorted { counts[install.identifier, default: 0] += 1 }
        var used: [String: Int] = [:]
        return sorted.map { install in
            var id = install.identifier
            if counts[id, default: 0] > 1 || reserved.contains(id) { id = "\(id)@\(install.basename)" }
            let occurrence = used[id, default: 0]
            used[id] = occurrence + 1
            if occurrence > 0 { id = "\(id)#\(occurrence + 1)" }
            return (install, id)
        }
    }

    /// `Foo-Bar.Q4.gguf` → `foo-bar.q4`.
    static func ggufModelID(fileName: String) -> String {
        let base = fileName.lowercased().hasSuffix(".gguf") ? String(fileName.dropLast(5)) : fileName
        return base.lowercased()
    }

    /// llama.cpp's split naming, `<name>-00001-of-00003.gguf`.
    static func isSplitShard(_ fileName: String) -> Bool {
        fileName.range(of: #"-\d{5}-of-\d{5}\.gguf$"#, options: [.regularExpression, .caseInsensitive]) != nil
    }

    private static func unique(_ urls: [URL]) -> [URL] {
        var seen = Set<String>()
        return urls.filter { seen.insert($0.path).inserted }
    }

    /// Immediate subdirectories of `root`, or none when it is not readable.
    public static func childDirectories(of root: URL) -> [URL] {
        guard let contents = try? FileManager.default.contentsOfDirectory(
            at: root, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles])
        else { return [] }
        return contents.filter {
            (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
        }.sorted { $0.path < $1.path }
    }

    /// `.gguf` files directly in `directory` that are, or link to, regular
    /// files, in path order. A symlink keeps its own path as the location.
    public static func ggufFiles(in directory: URL) -> [URL] {
        guard let contents = try? FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles])
        else { return [] }
        return contents.filter {
            $0.pathExtension.lowercased() == "gguf"
                && (try? $0.resolvingSymlinksInPath().resourceValues(forKeys: [.isRegularFileKey])
                    .isRegularFile) == true
        }.sorted { $0.path < $1.path }
    }

    /// Size of a regular file, or the sum over the regular files under a
    /// directory. Metadata only.
    public static func bytesOnDisk(_ url: URL) -> UInt64 {
        let keys: Set<URLResourceKey> = [.isRegularFileKey, .fileSizeKey]
        if let values = try? url.resolvingSymlinksInPath().resourceValues(forKeys: keys),
           values.isRegularFile == true {
            return UInt64(values.fileSize ?? 0)
        }
        guard let walker = FileManager.default.enumerator(at: url, includingPropertiesForKeys: Array(keys))
        else { return 0 }
        var total: UInt64 = 0
        for case let file as URL in walker {
            if let values = try? file.resourceValues(forKeys: keys), values.isRegularFile == true {
                total += UInt64(values.fileSize ?? 0)
            }
        }
        return total
    }

    /// Nearest ancestor that is this package's checkout: `Package.swift`
    /// beside `Sources/Mference` and `Sources/MferenceServer`.
    private static func packageRoot(startingAt start: URL, fileExists: (String) -> Bool) -> URL? {
        var candidatePath = start.standardizedFileURL.path
        while true {
            let candidate = URL(fileURLWithPath: candidatePath, isDirectory: true)
            if fileExists(candidate.appendingPathComponent("Package.swift").path),
               fileExists(candidate.appendingPathComponent("Sources/Mference", isDirectory: true).path),
               fileExists(candidate.appendingPathComponent("Sources/MferenceServer", isDirectory: true).path) {
                return candidate
            }
            let parentPath = (candidatePath as NSString).deletingLastPathComponent
            if parentPath.isEmpty || parentPath == candidatePath { return nil }
            candidatePath = parentPath
        }
    }
}
