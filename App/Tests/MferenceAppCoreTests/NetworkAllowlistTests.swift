import Foundation
@testable import MferenceAppCore
import Testing

/// G-A8 for speech (Notes spec 3, 7): speech models download only from the
/// pinned Hugging Face files, and no app source calls a FluidAudio API that
/// downloads on its own. Reads source files and constants only.
@Suite struct NetworkAllowlistTests {
    static let sources = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Sources")

    @Test func speechDownloadsStayOnHuggingFace() {
        for id in SpeechPackageID.allCases {
            let package = SpeechModelPins.package(id)
            #expect(package.totalBytes > 0)
            for source in package.sources {
                #expect(source.revision.count == 40, "pinned to a full commit")
                for file in source.files {
                    let url = source.url(for: file)
                    #expect(DownloadSource.isAllowed(host: url.host, allowedHosts: SpeechModelPins.allowedHosts))
                    #expect(url.path.contains("/resolve/\(source.revision)/"))
                    #expect(file.sha256.count == 64 && file.size > 0)
                    #expect(!file.installPath.contains(".."))
                }
            }
        }
        #expect(!DownloadSource.isAllowed(host: "huggingface.co.evil.example", allowedHosts: SpeechModelPins.allowedHosts))
    }

    @Test func pinnedSizesMatchTheMeasuredDownload() {
        #expect(SpeechModelPins.speech.totalBytes == 653_388_064)
        #expect(SpeechModelPins.speakerLabels.totalBytes == 66_627_357, "LS-EEND's 100 ms export, no AMI/* files")
        let installed = SpeechModelPins.speech.sources.flatMap(\.files).map(\.installPath)
        #expect(installed.contains { $0.contains("/JointDecisionv3.mlmodelc/") }, "the .v3 loader's joint name")
        #expect(!installed.contains { $0.contains("/JointDecision.mlmodelc/") })
        let labels = SpeechModelPins.speakerLabels.sources.flatMap(\.files).map(\.repoPath)
        #expect(!labels.contains { ($0 as NSString).lastPathComponent == ".DS_Store" })
        #expect(!labels.contains { $0.hasPrefix("AMI/") }, "the older LS-EEND export is not pinned")
    }

    /// FluidAudio's `load(from:)`, `downloadAndLoad` and `ModelHub` fetch
    /// models from the network; the app loads only verified local folders.
    @Test func noSourceCallsAFluidAudioDownload() throws {
        let forbidden = ["downloadAndLoad", "AsrModels.load(", "ModelHub.", "DownloadUtils.", ".download(to:"]
        let files = FileManager.default.enumerator(at: Self.sources, includingPropertiesForKeys: nil)?
            .compactMap { $0 as? URL }.filter { $0.pathExtension == "swift" } ?? []
        #expect(!files.isEmpty)
        for file in files {
            let text = try String(contentsOf: file, encoding: .utf8)
            guard text.contains("import FluidAudio") else { continue }
            #expect(file.path.contains("/MferenceAppCore/Speech/"), "only Speech imports FluidAudio: \(file.lastPathComponent)")
            for symbol in forbidden {
                #expect(!text.contains(symbol), "\(file.lastPathComponent) calls \(symbol)")
            }
        }
    }
}
