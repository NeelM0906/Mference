import Foundation
import MferenceAppCore
@testable import MferenceAppUI
import SwiftUI
import Testing

@Suite struct SettingsViewTests {
    /// Each switch reads and writes its own setting. (The shipped defaults
    /// belong to U2's AppSettings and are tested there.)
    @Test func switchesReadAndWriteTheirSetting() {
        let settings = FixtureSettings()
        let all: [SettingSwitch] = [.webSearchEnabled, .offlineMode, .blockRemoteMailContent,
                                    .remindersSyncEnabled, .useAppleModel]
        for setting in all {
            let before = setting.get(settings)
            setting.set(settings, !before)
            #expect(setting.get(settings) == !before)
        }
        #expect(settings.webSearchEnabled && settings.offlineMode && settings.remindersSyncEnabled)
    }

    @Test func chatOpensSettingsWhereItPoints() {
        let shell = ShellModel(selection: .chat, defaults: nil)
        shell.openSettings(.memory)
        #expect(shell.selection == .settings && shell.settingsPane == .memory && !shell.settingsShowsLibrary)
        shell.toggleSettings()
        #expect(shell.selection == .chat)
        shell.openSettings(.models, showingLibrary: true)
        #expect(shell.settingsShowsLibrary)
        shell.toggleSettings()
        shell.toggleSettings()
        #expect(shell.settingsPane == .models && !shell.settingsShowsLibrary, "the gear opens Settings at the top")
    }

    @Test func paneTitlesFollowTheMockup() {
        #expect(SettingsView.Pane.allCases.map(\.title)
                == ["Models", "Memory", "Privacy", "Connections", "Dictation", "Shortcuts", "iPhone"])
    }

    @Test func memoryAddEditDelete() throws {
        let memory = FixtureMemory()
        let count = memory.memories.count
        SettingsView.save(.new, text: "  Likes window seats  ", in: memory)
        #expect(memory.memories.count == count + 1)
        #expect(memory.memories.first?.text == "Likes window seats")
        SettingsView.save(.new, text: "   ", in: memory)
        #expect(memory.memories.count == count + 1, "blank text adds nothing")
        let first = try #require(memory.memories.first)
        SettingsView.save(.existing(first), text: "Likes aisle seats", in: memory)
        #expect(memory.memories.first?.text == "Likes aisle seats")
        // Delete goes through the confirmation dialog's action.
        SettingsView.confirmDelete(first, in: memory)
        #expect(memory.memories.count == count)
        #expect(!memory.memories.contains { $0.id == first.id })
    }

    @Test func offlineModeTurnsWebSearchAndDownloadsOff() {
        let settings = FixtureSettings()
        settings.webSearchEnabled = true
        #expect(SettingsView.webSearchShown(settings))
        #expect(SettingsView.downloadsAllowed(settings))
        settings.offlineMode = true
        #expect(!SettingsView.webSearchShown(settings), "shown off while offline")
        #expect(!SettingsView.downloadsAllowed(settings))
        #expect(settings.webSearchEnabled, "the stored choice comes back after offline mode")
        #expect(!SettingsView.webSearchShown(nil) && SettingsView.downloadsAllowed(nil))
    }

    @Test func connectionStatesRender() {
        #expect(SettingsView.connectionDisplay(.connected) == .connected)
        #expect(SettingsView.connectionDisplay(.notConnected) == .connectButton)
        #expect(SettingsView.connectionDisplay(.needsSystemSettings) == .openSystemSettings)
        #expect(SettingsView.connectionDisplay(nil) == .notConnected, "unwired reads Not connected, never Later")
    }

    @Test func modelRowsSayWhyTheyFailed() throws {
        let library = FixtureModelLibrary(deep: .downloadFailed("Not enough free space"))
        let deep = try #require(library.entries.first { $0.id == AppModel.deep.modelID })
        #expect(SettingsView.modelRowSubtitle(deep).hasSuffix("\nDownload failed: Not enough free space"))
    }

    @Test func sizesStayOnOneLine() throws {
        let deep = try #require(FixtureModelLibrary().entries.first { $0.id == AppModel.deep.modelID })
        #expect(SettingsView.modelRowSubtitle(deep).contains("12.1\u{00A0}GB"))
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
