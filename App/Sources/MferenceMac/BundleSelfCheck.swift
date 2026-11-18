import Foundation

/// `Mference --verify-bundle`: resolves the SwiftPM resource bundles the same
/// way their generated `Bundle.module` accessors do in an app
/// (`Bundle.main.resourceURL/<name>.bundle`) and checks the files the engine
/// opens at load time are there. `Scripts/bundle.sh` runs it on the assembled
/// app; it never touches Metal or a model.
enum BundleSelfCheck {
    struct Requirement {
        let bundleName: String
        /// Paths inside the bundle's resources that must exist.
        let paths: [String]
        /// Directories inside the bundle that must each hold at least one file
        /// with this extension.
        let nonEmpty: [(directory: String, fileExtension: String)]
    }

    static let requirements: [Requirement] = [
        Requirement(bundleName: "Mference_Mference", paths: ["Metal", "Gemma4"],
                    nonEmpty: [("Metal", "metal")]),
        Requirement(bundleName: "Mference_LlamaCpp",
                    paths: ["kernels", "ggml-metal-impl.h", "ggml-common.h"],
                    nonEmpty: [("kernels", "metal")]),
    ]

    /// Returns the failures (empty when every bundle loads).
    static func run(resourceURL: URL? = Bundle.main.resourceURL,
                    requirements: [Requirement] = requirements) -> [String] {
        guard let resourceURL else { return ["Bundle.main.resourceURL is nil"] }
        var failures: [String] = []
        let files = FileManager.default
        for requirement in requirements {
            let url = resourceURL.appendingPathComponent(requirement.bundleName + ".bundle")
            guard let bundle = Bundle(url: url), let root = bundle.resourceURL else {
                failures.append("\(requirement.bundleName).bundle does not load from \(url.path)")
                continue
            }
            for path in requirement.paths where !files.fileExists(atPath: root.appendingPathComponent(path).path) {
                failures.append("\(requirement.bundleName).bundle is missing \(path)")
            }
            for (directory, fileExtension) in requirement.nonEmpty {
                let found = files.enumerator(at: root.appendingPathComponent(directory),
                                             includingPropertiesForKeys: nil)?
                    .contains { ($0 as? URL)?.pathExtension == fileExtension } ?? false
                if !found {
                    failures.append("\(requirement.bundleName).bundle has no .\(fileExtension) file under \(directory)")
                }
            }
        }
        return failures
    }
}
