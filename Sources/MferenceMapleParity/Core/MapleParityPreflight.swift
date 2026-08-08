import Foundation
import Mference

/// Observable conditions required before a real Maple trace is allowed to run.
/// Validation stays pure so every refusal can be tested without probing a host.
public struct MapleParityPreflightFacts: Equatable, Sendable {
    public let appleSilicon: Bool
    public let macOSMajor: Int
    public let macOSVersion: String
    public let swiftMajor: Int?
    public let swiftMinor: Int?
    public let swiftVersion: String?
    public let architecture: String
    public let modelDirectoryExists: Bool
    public let modelValidationError: String?
    public let freeBytes: UInt64?
    public let requiredFreeBytes: UInt64
    public let memoryFreePercent: Double?
    public let conflictingProcesses: [String]
    public let checkoutClean: Bool
    public let repositoryDirectory: URL?
    public let gitRevision: String?

    public init(appleSilicon: Bool,
                macOSMajor: Int,
                macOSVersion: String,
                swiftMajor: Int?,
                swiftMinor: Int?,
                swiftVersion: String?,
                architecture: String,
                modelDirectoryExists: Bool,
                modelValidationError: String?,
                freeBytes: UInt64?,
                requiredFreeBytes: UInt64,
                memoryFreePercent: Double?,
                conflictingProcesses: [String],
                checkoutClean: Bool,
                repositoryDirectory: URL?,
                gitRevision: String?) {
        self.appleSilicon = appleSilicon
        self.macOSMajor = macOSMajor
        self.macOSVersion = macOSVersion
        self.swiftMajor = swiftMajor
        self.swiftMinor = swiftMinor
        self.swiftVersion = swiftVersion
        self.architecture = architecture
        self.modelDirectoryExists = modelDirectoryExists
        self.modelValidationError = modelValidationError
        self.freeBytes = freeBytes
        self.requiredFreeBytes = requiredFreeBytes
        self.memoryFreePercent = memoryFreePercent
        self.conflictingProcesses = conflictingProcesses
        self.checkoutClean = checkoutClean
        self.repositoryDirectory = repositoryDirectory
        self.gitRevision = gitRevision
    }
}

public struct MapleParityPreflightReport: Equatable, Sendable {
    public let facts: MapleParityPreflightFacts
    public let failures: [String]

    public var isAccepted: Bool { failures.isEmpty }
}

public enum MapleParityPreflight {
    public static func validate(_ facts: MapleParityPreflightFacts) -> MapleParityPreflightReport {
        var failures: [String] = []
        if !facts.appleSilicon { failures.append("Apple Silicon is required") }
        if facts.macOSMajor < 15 { failures.append("macOS 15 or newer is required") }
        if !(facts.swiftMajor.map { major in
            facts.swiftMinor.map { minor in major > 6 || major == 6 && minor >= 1 } ?? false
        } ?? false) {
            failures.append("Swift 6.1 or newer is required")
        }
        if !facts.modelDirectoryExists { failures.append("Maple .gturbo directory is missing") }
        if let error = facts.modelValidationError { failures.append(error) }
        if facts.freeBytes == nil || facts.freeBytes! < facts.requiredFreeBytes {
            failures.append("insufficient free disk space for the Maple reserve")
        }
        if facts.memoryFreePercent == nil || facts.memoryFreePercent! < 10 {
            failures.append("memory_pressure -Q reports less than 10% free memory")
        }
        if !facts.conflictingProcesses.isEmpty { failures.append("another model or package-test process is running") }
        if facts.repositoryDirectory == nil { failures.append("unable to resolve the repository root") }
        if !facts.checkoutClean { failures.append("the checkout must be clean for parity export") }
        if let revision = facts.gitRevision,
           !isLowerHex(revision, count: 40) {
            failures.append("git revision is not a full lowercase commit hash")
        } else if facts.gitRevision == nil {
            failures.append("unable to determine the checkout revision")
        }
        return MapleParityPreflightReport(facts: facts, failures: failures)
    }

    public static func inspect(modelDirectory: URL, repositoryDirectory: URL) throws -> MapleParityPreflightFacts {
        let version = ProcessInfo.processInfo.operatingSystemVersion
        let swift = try command("/usr/bin/xcrun", ["swift", "--version"])
        let components = swiftVersion(swift.stdout)
        let architecture = try command("/usr/bin/uname", ["-m"]).stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        let resolvedRepository = try command("/usr/bin/git", ["-C", repositoryDirectory.path, "rev-parse", "--show-toplevel"])
            .stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        let root = URL(fileURLWithPath: resolvedRepository, isDirectory: true).standardizedFileURL
        let status = try command("/usr/bin/git", ["-C", root.path, "status", "--porcelain", "--untracked-files=all"])
        let revision = try command("/usr/bin/git", ["-C", root.path, "rev-parse", "HEAD"])
        let pressure = try command("/usr/bin/memory_pressure", ["-Q"])
        let conflicts = try command("/usr/bin/pgrep", ["-fl", processPattern], permitsFailure: true)
        var isDirectory: ObjCBool = false
        let exists = FileManager.default.fileExists(atPath: modelDirectory.path, isDirectory: &isDirectory)
        let attributes = try? FileManager.default.attributesOfFileSystem(forPath: modelDirectory.path)
        let free = (attributes?[.systemFreeSize] as? NSNumber)?.uint64Value
        let validationError: String?
        if exists && isDirectory.boolValue {
            do { try validateInstalledModel(at: modelDirectory); validationError = nil }
            catch { validationError = "invalid Maple install: \(error)" }
        } else {
            validationError = nil
        }
        let currentPID = ProcessInfo.processInfo.processIdentifier
        let processLines = conflicts.stdout.split(separator: "\n").map(String.init).filter {
            Int($0.split(maxSplits: 1, whereSeparator: \.isWhitespace).first ?? "") != Int(currentPID)
        }
        return MapleParityPreflightFacts(
            appleSilicon: architecture == "arm64",
            macOSMajor: version.majorVersion,
            macOSVersion: "\(version.majorVersion).\(version.minorVersion).\(version.patchVersion)",
            swiftMajor: components.major,
            swiftMinor: components.minor,
            swiftVersion: components.full,
            architecture: architecture,
            modelDirectoryExists: exists && isDirectory.boolValue,
            modelValidationError: validationError,
            freeBytes: free,
            requiredFreeBytes: max(UInt64(1 << 30), MapleParityPins.source.reserveBytes),
            memoryFreePercent: memoryFreePercent(pressure.stdout),
            conflictingProcesses: processLines,
            checkoutClean: status.stdout.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            repositoryDirectory: root,
            gitRevision: revision.stdout.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    /// Verifies metadata, the complete on-disk file set, and the receipt
    /// binding before the model loader touches resident weights.
    public static func validateInstalledModel(at directory: URL) throws {
        let standard = directory.standardizedFileURL
        let manifest = try ManifestReader.load(directoryURL: standard, expecting: .maplePreview)
        guard manifest.modelID == MapleParityPins.source.modelID,
              manifest.sourceSnapshotHash == "sha256:\(MapleParityPins.sourceIndexSHA256)" else {
            throw MapleParityError.invalid("manifest does not identify the pinned Maple checkpoint")
        }
        let manifestURL = standard.appendingPathComponent("manifest.json")
        let manifestSize = try fileSize(manifestURL)
        let manifestSHA = try Sha256Verifier.hashFile(at: manifestURL)
        let receipt = try VerifiedInstallReceiptReader.load(directoryURL: standard)
        try VerifiedInstallReceiptReader.validate(receipt, directoryURL: standard, manifest: manifest,
                                                  manifestSha256: manifestSHA, manifestSize: manifestSize)
        guard receipt.sourceRepoID == MapleParityPins.source.repoID,
              receipt.sourceRevision == MapleParityPins.modelRevision else {
            throw MapleParityError.invalid("receipt does not identify the pinned Maple source")
        }
        let expected = Set(manifest.files.keys).union(["manifest.json", VerifiedInstallReceiptReader.fileName])
        let actual = try regularFiles(in: standard)
        guard actual == expected else {
            throw MapleParityError.invalid("installed file set differs from the manifest and receipt")
        }
        for (relativePath, entry) in manifest.files {
            let url = standard.appendingPathComponent(relativePath)
            guard try fileSize(url) == entry.size else {
                throw MapleParityError.invalid("installed size differs for \(relativePath)")
            }
            if relativePath.hasPrefix("tokenizer/") {
                try Sha256Verifier.verifyFile(at: url, named: relativePath,
                                              expectedHex: entry.sha256)
            }
        }
    }

    private static let processPattern = "MferenceServer|MferenceMac|MferenceDecodeService|MferenceCLI|MferenceMapleParity|MferencePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm|maple_mlx_teacher_forcing"

    static func regularFiles(in root: URL) throws -> Set<String> {
        guard let enumerator = FileManager.default.enumerator(atPath: root.path) else {
            throw MapleParityError.io("unable to enumerate \(root.path)")
        }
        var files: Set<String> = []
        for case let relativePath as String in enumerator {
            let url = root.appendingPathComponent(relativePath)
            if try url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true {
                files.insert(relativePath)
            }
        }
        return files
    }

    private static func fileSize(_ url: URL) throws -> UInt64 {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        guard let value = attributes[.size] as? NSNumber else {
            throw MapleParityError.io("unable to read size for \(url.path)")
        }
        return value.uint64Value
    }

    private static func swiftVersion(_ output: String) -> (major: Int?, minor: Int?, full: String?) {
        guard let match = output.range(of: "Swift version [0-9]+\\.[0-9]+(?:\\.[0-9]+)?", options: .regularExpression) else {
            return (nil, nil, nil)
        }
        let text = String(output[match])
        let numbers = text.split(whereSeparator: { !$0.isNumber })
        guard numbers.count >= 2 else { return (nil, nil, nil) }
        return (Int(numbers[0]), Int(numbers[1]), text.replacingOccurrences(of: "Swift version ", with: ""))
    }

    private static func memoryFreePercent(_ output: String) -> Double? {
        guard let match = output.range(of: "[0-9]+(?:\\.[0-9]+)?(?=%)", options: .regularExpression) else {
            return nil
        }
        return Double(output[match])
    }

    static func isLowerHex(_ value: String, count: Int) -> Bool {
        value.count == count && value.allSatisfy { "0123456789abcdef".contains($0) }
    }

    private struct CommandResult {
        let stdout: String
    }

    private static func command(_ executable: String,
                                _ arguments: [String],
                                permitsFailure: Bool = false) throws -> CommandResult {
        let process = Process()
        let output = Pipe()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        process.standardOutput = output
        process.standardError = Pipe()
        do { try process.run() }
        catch { throw MapleParityError.io("unable to inspect \(executable): \(error.localizedDescription)") }
        process.waitUntilExit()
        let text = String(decoding: output.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        guard permitsFailure || process.terminationStatus == 0 else {
            throw MapleParityError.io("inspection command failed: \(executable) \(arguments.joined(separator: " "))")
        }
        return CommandResult(stdout: text)
    }
}
