import Foundation
import Testing

/// Every production Metal device must come from
/// `MetalContext.makeSystemDefaultDevice()`, which sets the AGX watchdog
/// default before the driver's one-time read. The audit fails closed: a walk
/// or read failure is a failure, not a silently skipped file.
@Suite struct MetalContextDeviceCreationTests {
    private static var sourcesDirectory: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources")
    }

    private static let sanctionedRelativePath = "Mference/Infrastructure/Metal/MetalContext.swift"

    @Test func onlyMetalContextCreatesTheSystemDevice() throws {
        let sources = Self.sourcesDirectory
        try #require(FileManager.default.fileExists(
            atPath: sources.appendingPathComponent(Self.sanctionedRelativePath).path),
                     "cannot locate Sources/ from \(#filePath)")

        var enumerationFailures: [String] = []
        var readFailures: [String] = []
        var offenders: [String] = []
        var audited = 0
        let walker = try #require(FileManager.default.enumerator(
            at: sources,
            includingPropertiesForKeys: nil,
            errorHandler: { url, error in
                enumerationFailures.append("\(url.path): \(error)")
                return false
            }))
        let sourcePrefix = sources.standardizedFileURL.path + "/"

        while let url = walker.nextObject() as? URL {
            guard url.pathExtension == "swift" else { continue }
            let path = url.standardizedFileURL.path
            guard path.hasPrefix(sourcePrefix) else {
                readFailures.append("cannot make \(path) relative to \(sources.path)")
                continue
            }
            let relativePath = String(path.dropFirst(sourcePrefix.count))
            guard relativePath != Self.sanctionedRelativePath else { continue }
            let text: String
            do {
                text = try String(contentsOf: url, encoding: .utf8)
            } catch {
                readFailures.append("\(relativePath): \(error)")
                continue
            }
            audited += 1
            for (index, line) in text.split(separator: "\n", omittingEmptySubsequences: false)
                .enumerated() where line.contains("MTLCreateSystemDefaultDevice(") {
                offenders.append("\(relativePath):\(index + 1)")
            }
        }

        #expect(audited > 100, "audited only \(audited) source files")
        #expect(enumerationFailures.isEmpty,
                "source enumeration failed: \(enumerationFailures.joined(separator: "; "))")
        #expect(readFailures.isEmpty,
                "source reads failed: \(readFailures.joined(separator: "; "))")
        #expect(offenders.isEmpty,
                "production device creation bypasses MetalContext: \(offenders.joined(separator: ", "))")
    }
}
