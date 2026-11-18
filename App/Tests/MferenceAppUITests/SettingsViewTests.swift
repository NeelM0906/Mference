import Foundation
import MferenceAppCore
@testable import MferenceAppUI
import SwiftUI
import Testing

@Suite struct SettingsViewTests {
    @Test func switchesReadAndWriteTheirSetting() {
        let settings = FixtureSettings()
        #expect(SettingSwitch.webSearchEnabled.get(settings) == false, "web search is off by default (L7)")
        #expect(SettingSwitch.remindersSyncEnabled.get(settings) == false, "Reminders sync is off by default (L9)")
        SettingSwitch.webSearchEnabled.set(settings, true)
        SettingSwitch.offlineMode.set(settings, true)
        #expect(settings.webSearchEnabled && settings.offlineMode)
    }

    @Test func paneTitlesFollowTheMockup() {
        #expect(SettingsView.Pane.allCases.map(\.title)
                == ["Models", "Memory", "Privacy", "Connections", "Dictation", "Shortcuts", "iPhone"])
    }

    @Test(arguments: [ColorScheme.light, .dark])
    func snapshot(colorScheme: ColorScheme) throws {
        let screenModels = ScreenModels(library: FixtureModelLibrary(), memory: FixtureMemory(), settings: FixtureSettings())
        let view = SettingsView(library: screenModels.library, memory: screenModels.memory, settings: screenModels.settings)
        try Snapshot.render(ShellBoard(selection: .settings, screenModels: screenModels) { view },
                            name: "08-settings", canvas: CGSize(width: 1280, height: 1520), colorScheme: colorScheme)
    }

    @Test func rendersWithoutInputs() {
        let view = Snapshot.framed(SettingsView(), framing: .mockup)
        #expect(ImageRenderer(content: view).cgImage != nil)
    }
}
