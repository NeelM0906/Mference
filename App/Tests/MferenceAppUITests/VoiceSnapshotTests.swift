import Foundation
import MferenceAppCore
@testable import MferenceAppUI
import MferenceDesign
import SwiftUI
import Testing

/// G-A3 renders for the voice boards (N3), light and dark, on
/// `FakeSpeechService` fixtures. Names match `$P/mockups/png`.
@MainActor
@Suite(.serialized) struct VoiceSnapshotTests {
    private func voice(_ speech: FakeSpeechService, broker: FakePermissionBroker = FakePermissionBroker(),
                       offline: Bool = false) -> VoiceModels {
        VoiceModels(speech: speech, permissions: broker,
                    preferences: SpeechPreferences(store: InMemorySettingsStore()), isOffline: { offline })
    }

    @Test(arguments: [ColorScheme.light, .dark])
    func setupCard(colorScheme: ColorScheme) throws {
        let models = voice(FakeSpeechService.notInstalled())
        try Snapshot.render(NotesBoard { SpeechSetupCard(model: models.setup) },
                            name: "notes-07-setup", colorScheme: colorScheme)
    }

    @Test func setupCardStates() throws {
        let states: [(String, SpeechModelState, Bool)] = [
            ("downloading", .downloading(progress: 0.38), false),
            ("paused", .paused(progress: 0.38), false),
            ("preparing", .preparing, false),
            ("failed", .failed(.downloadFailed("timeout")), false),
            ("offline", .notInstalled, true),
        ]
        for (name, state, offline) in states {
            let models = voice(FakeSpeechService(status: SpeechStatus(speech: state)), offline: offline)
            try Snapshot.render(NotesBoard { SpeechSetupCard(model: models.setup) },
                                name: "notes-07-setup-\(name)")
        }
    }

    @Test(arguments: [ColorScheme.light, .dark])
    func settingsSpeech(colorScheme: ColorScheme) async throws {
        let speech = FakeSpeechService()
        speech.setRecordingsSize(1_200_000_000)
        let models = voice(speech, broker: FakePermissionBroker(states: [.microphone: .granted, .systemAudio: .granted]))
        await models.settings.refresh()
        let screenModels = ScreenModels(library: FixtureModelLibrary(), memory: FixtureMemory(),
                                        settings: FixtureSettings())
        let view = SettingsView(library: screenModels.library, memory: screenModels.memory,
                                settings: screenModels.settings, pane: .speech)
        try Snapshot.render(ShellBoard(selection: .settings, screenModels: screenModels) { view }
            .environment(models),
                            name: "notes-08-settings-speech", colorScheme: colorScheme)
    }

    @Test func settingsSpeechNotInstalledAndDenied() async throws {
        let models = voice(FakeSpeechService(status: SpeechStatus(speech: .notInstalled, speakerLabels: .notInstalled)),
                           broker: FakePermissionBroker(states: [.microphone: .denied, .systemAudio: .denied]))
        await models.settings.refresh()
        let screenModels = ScreenModels(settings: FixtureSettings())
        let view = SettingsView(settings: screenModels.settings, pane: .speech)
        try Snapshot.render(ShellBoard(selection: .settings, screenModels: screenModels) { view }
            .environment(models),
                            name: "notes-08-settings-speech-not-installed")
    }
}

/// The Notes window as the voice boards draw it behind a voice surface: the
/// chrome with Notes selected and its actions, then the sidebar and list as
/// grey placeholder bars, and `content` centered in the editor area
/// (`Setup.dc.html`'s `.side`, `.list`, `.ed`). N1's real panes replace the
/// placeholders in the app.
struct NotesBoard<Content: View>: View {
    let content: Content
    /// Grey bars in the sidebar and list (boards 3, 6, 7); off draws them empty.
    var placeholders = true
    let shell = ShellModel(selection: .notes, defaults: nil)

    init(placeholders: Bool = true, @ViewBuilder content: () -> Content) {
        self.placeholders = placeholders
        self.content = content()
    }

    var body: some View {
        VStack(spacing: 0) {
            ShellChromeBar(model: shell, items: [
                ShellToolbarItem(id: "notes.transcribe", view: AnyView(
                    Button {} label: { Label("Transcribe file", systemImage: "doc.text") }.buttonStyle(.mfSecondary))),
                ShellToolbarItem(id: "notes.record", view: AnyView(
                    Button {} label: { Label("Record meeting", systemImage: "record.circle") }.buttonStyle(.mfSecondary))),
                ShellToolbarItem(id: "notes.new", view: AnyView(
                    Button {} label: { Image(systemName: "square.and.pencil") }.buttonStyle(.mfSecondary))),
            ])
            HStack(spacing: 0) {
                bars(placeholders ? [0.62, 0.5, 0.48] : [], width: 210, fill: MFColor.sidebar)
                Rectangle().fill(MFColor.chromeDivider).frame(width: MFLayout.hairline)
                bars(placeholders ? [0.8, 0.6, 0.7, 0.5] : [], width: 300, fill: MFColor.surface)
                Rectangle().fill(MFColor.chromeDivider).frame(width: MFLayout.hairline)
                content
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(MFColor.surface)
            }
        }
        .mfWindowBackground()
    }

    private func bars(_ widths: [CGFloat], width: CGFloat, fill: Color) -> some View {
        VStack(alignment: .leading, spacing: MFSpacing.m) {
            ForEach(Array(widths.enumerated()), id: \.offset) { _, fraction in
                Capsule().fill(MFColor.fill).frame(width: (width - 28) * fraction, height: 12)
            }
            Spacer()
        }
        .padding(14)
        .frame(width: width)
        .frame(maxHeight: .infinity)
        .background(fill)
    }
}
