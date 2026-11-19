import Foundation
@testable import MferenceMac
import Testing

/// `Mference --verify-bundle` against fake resource bundles in a temp folder.
@Suite struct ShellBundleSelfCheckTests {
    /// Lays out `<root>/<name>.bundle/Contents/{Info.plist,Resources/<files>}`.
    static func makeBundle(_ name: String, files: [String], in root: URL) throws {
        let contents = root.appendingPathComponent("\(name).bundle/Contents")
        let resources = contents.appendingPathComponent("Resources")
        try FileManager.default.createDirectory(at: resources, withIntermediateDirectories: true)
        let info: [String: Any] = ["CFBundleIdentifier": "test.\(name)", "CFBundlePackageType": "BNDL"]
        try PropertyListSerialization.data(fromPropertyList: info, format: .xml, options: 0)
            .write(to: contents.appendingPathComponent("Info.plist"))
        for file in files {
            let url = resources.appendingPathComponent(file)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data("x".utf8).write(to: url)
        }
    }

    static func tempRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("selfcheck-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    @Test func completeBundlesPass() throws {
        let root = try Self.tempRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        try Self.makeBundle("Mference_Mference", files: ["Metal/core/a.metal"], in: root)
        try Self.makeBundle("Mference_LlamaCpp", files: ["kernels/add.metal", "ggml-metal-impl.h", "ggml-common.h"], in: root)
        #expect(BundleSelfCheck.run(resourceURL: root).isEmpty)
    }

    @Test func missingPiecesAreReported() throws {
        let root = try Self.tempRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        // A Metal folder without .metal files, a kernels folder without them,
        // no LlamaCpp headers.
        try Self.makeBundle("Mference_Mference", files: ["Metal/README"], in: root)
        try Self.makeBundle("Mference_LlamaCpp", files: ["kernels/README"], in: root)
        let failures = BundleSelfCheck.run(resourceURL: root)
        #expect(failures.contains { $0.contains("no .metal file under Metal") })
        #expect(failures.contains { $0.contains("no .metal file under kernels") })
        #expect(failures.contains { $0.contains("missing ggml-metal-impl.h") })
    }

    @Test func absentBundleIsReported() throws {
        let root = try Self.tempRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let failures = BundleSelfCheck.run(resourceURL: root)
        #expect(failures.count == 2)
        #expect(failures.allSatisfy { $0.contains("does not load") })
    }
}
