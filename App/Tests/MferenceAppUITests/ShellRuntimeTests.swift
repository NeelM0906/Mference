import Foundation
import MferenceAppCore
@testable import MferenceAppUI
@testable import MferenceMac
import Testing

/// The app builds exactly one engine (one composition root): two engine
/// hosts could each hold a model and break the one-model rule.
@Suite struct ShellRuntimeTests {
    @Test func chatAndScreensShareTheOneEngine() {
        let services = AppServices.preview()
        let runtime = AppRuntime(services: services, web: WebAnswers(settings: services.settings, runtime: nil, legacyStore: InMemorySettingsStore()))
        #expect(runtime.services === services)
        #expect((runtime.chat.service as AnyObject) === (services.engine as AnyObject))
        #expect(runtime.screenModels.library != nil && runtime.screenModels.memory != nil
                && runtime.screenModels.settings != nil)
    }

    /// The app target builds its engine only through `AppServices.live()`.
    @Test func appSourcesCreateNoOtherEngine() throws {
        let macSources = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/MferenceMac")
        let files = try FileManager.default.contentsOfDirectory(at: macSources, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "swift" }
        let text = try files.map { try String(contentsOf: $0, encoding: .utf8) }.joined(separator: "\n")
        #expect(!text.contains("EngineService.live("), "the app must not build a second EngineService")
        #expect(!text.contains("EngineHost("), "the app must not build an engine host of its own")
        // One `AppServices.live(presentPermissionReason:)` call (written
        // `.live(` as AppRuntime's default argument).
        #expect(text.components(separatedBy: "live(presentPermissionReason:").count - 1 == 1)
    }
}
