import Foundation
import MferenceAppCore
@testable import MferenceAppUI
import MferenceDesign
import SwiftUI
import Testing

/// A minimal note for UI tests: appends phrases at the end.
@MainActor
final class AppendingTarget: DictationInsertionTarget {
    let noteID = UUID()
    var text: String
    private var pending: Set<UUID> = []

    init(text: String = "") { self.text = text }

    func beginPhrase(_ phrase: UUID) { pending.insert(phrase) }
    func finishPhrase(_ phrase: UUID, text addition: String) {
        guard pending.remove(phrase) != nil else { return }
        let words = addition.trimmingCharacters(in: .whitespaces)
        guard let first = words.first else { return }
        text += (text.isEmpty ? "" : " ") + (text.isEmpty || text.hasSuffix(".") ? first.uppercased() + words.dropFirst() : words)
    }
    func dropAllPhrases() { pending = [] }
    func dropPhrase(_ phrase: UUID) { pending.remove(phrase) }
}

@MainActor
@Suite struct VoiceDictationTests {
    private func voice(_ speech: FakeSpeechService, broker: FakePermissionBroker = FakePermissionBroker()) -> VoiceModels {
        VoiceModels(speech: speech, permissions: broker,
                    preferences: SpeechPreferences(store: InMemorySettingsStore()), isOffline: { false })
    }

    /// Waits for `condition` (see `waitUntil`), never for a count of yields.
    private func eventually(sourceLocation: SourceLocation = #_sourceLocation,
                            _ condition: @MainActor () -> Bool) async throws {
        await waitUntil(timeout: .seconds(10), sourceLocation: sourceLocation, condition)
    }

    @Test func firstDictationGoesThroughSetupThenStarts() async throws {
        let speech = FakeSpeechService.notInstalled()
        speech.setDictationScript(.saying("hello"))
        let models = voice(speech)
        let note = AppendingTarget(text: "Note.")
        models.toggleDictation(in: note)
        #expect(models.setup.pendingAction == .dictate, "the setup card shows first")
        #expect(!models.dictation.isActive)
        models.setup.download()
        await models.setup.waitForInstall()
        try await eventually { models.dictation.phase == .listening }
        await models.dictation.stop()
        #expect(note.text == "Note. Hello")
    }

    @Test func shortcutWithoutAFocusedNoteOnlyStops() async throws {
        let speech = FakeSpeechService(stepped: true)
        let models = voice(speech)
        models.toggleDictation()
        #expect(!models.dictation.isActive && speech.calls.isEmpty, "nothing to dictate into")
        let note = AppendingTarget()
        models.focusedDictationTarget = note
        models.toggleDictation()
        try await eventually { models.dictation.isDictating(into: note.noteID) }
        models.focusedDictationTarget = nil
        models.toggleDictation()
        try await eventually { !models.dictation.isActive }
    }

    @Test func toggleInTheSameNoteStops() async throws {
        let models = voice(FakeSpeechService(stepped: true))
        let note = AppendingTarget()
        models.toggleDictation(in: note)
        try await eventually { models.dictation.isDictating(into: note.noteID) }
        models.toggleDictation(in: note)
        try await eventually { !models.dictation.isActive }
    }

    @Test func barCopy() {
        #expect(DictationBar.title(.listening, error: nil) == "Listening")
        #expect(DictationBar.detail(.listening, error: nil) == "Text lands after each pause")
        #expect(DictationBar.title(.idle, error: .permissionDenied(.microphone)) == "Dictation stopped")
        #expect(DictationBar.detail(.idle, error: .busy(.meeting)) == "A meeting is recording. Stop it first.")
        #expect(DictationBar.detail(.listening, error: nil, cleanUpChanged: true)
                == "Clean up applies from your next dictation")
        #expect(VoiceFormat.languageChip(.automatic, detected: "en") == "English · auto")
        #expect(VoiceFormat.languageChip(.automatic, detected: nil) == "Automatic")
        #expect(VoiceFormat.languageChip(.fixed("de"), detected: "en") == "German")
        #expect(VoiceFormat.languages.count == 25)
        #expect(LevelMeter.height(level: 0, share: 1) == DictationMetrics.meterRest)
        #expect(LevelMeter.height(level: 2, share: 1) == DictationMetrics.meterPeak, "levels are clamped")
    }

    @Test(arguments: [ColorScheme.light, .dark])
    func dictationBoard(colorScheme: ColorScheme) async throws {
        let speech = FakeSpeechService(dictation: .saying("ask about the sensitivity on the lower right",
                                                          "and whether the appointment can move to the morning"),
                                       stepped: true)
        let models = voice(speech)
        await models.dictation.start(in: AppendingTarget())
        // Level, the first phrase (English detected), then the second phrase started.
        for _ in 0..<4 {
            let before = models.dictation.handledEventCount
            speech.advance()
            await waitUntil("the dictation event") { models.dictation.handledEventCount > before }
        }
        #expect(models.dictation.phase == .listening && models.dictation.level > 0)
        #expect(models.dictation.detectedLanguage == "en" && models.dictation.pendingPhrases.count == 1)
        try Snapshot.render(NotesBoard(placeholders: false) {
            DictationBoardEditor()
                .overlay(alignment: .bottom) { DictationBar(controller: models.dictation) }
        }, name: "notes-02-dictation", colorScheme: colorScheme)
        await models.dictation.stop()
    }

    @Test func dictationBarError() async throws {
        let models = voice(FakeSpeechService(), broker: FakePermissionBroker(answers: [.microphone: .denied]))
        await models.dictation.start(in: AppendingTarget())
        try Snapshot.render(NotesBoard(placeholders: false) {
            DictationBoardEditor()
                .overlay(alignment: .bottom) { DictationBar(controller: models.dictation) }
        }, name: "notes-02-dictation-denied")
    }
}

/// A stand-in for N1's editor on board 2 (toolbar with "Dictating" pressed,
/// the note text, the pending phrase tinted): only for the render.
struct DictationBoardEditor: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: MFSpacing.m) {
                Text("Aa").font(MFFont.calloutSemibold)
                Image(systemName: "checklist")
                Image(systemName: "paperclip")
                Rectangle().fill(MFColor.hairline).frame(width: 1, height: 18)
                Label("Dictating", systemImage: "mic")
                    .font(MFFont.calloutEmphasis)
                    .foregroundStyle(MFColor.accentStrong)
                    .padding(.horizontal, MFSpacing.m)
                    .frame(height: 30)
                    .background(RoundedRectangle(cornerRadius: MFRadius.m).fill(MFColor.accentFill))
                Spacer()
                Image(systemName: "square.and.arrow.up")
            }
            .font(MFFont.callout)
            .foregroundStyle(MFColor.secondary)
            .padding(.horizontal, MFSpacing.comfortable)
            .frame(height: 48)
            .overlay(alignment: .bottom) { Rectangle().fill(MFColor.chromeDivider).frame(height: 1) }
            VStack(alignment: .leading, spacing: MFSpacing.m) {
                Text("19 November 2026 at 12:14")
                    .font(MFFont.caption).foregroundStyle(MFColor.tertiary)
                    .frame(maxWidth: .infinity)
                Text("Dentist questions").font(MFFont.largeTitle).foregroundStyle(MFColor.ink)
                Text("Questions for Dr. Hale on Thursday.")
                VStack(alignment: .leading, spacing: MFSpacing.xs) {
                    Text("•  Does the night guard need replacing? It has cracked on the left side.")
                    Text("•  Is this year's cleaning covered by the new plan?")
                    Text("•  Ask about the sensitivity on the lower right \(Text("and whether the appointment can move to the morning").foregroundStyle(MFColor.secondary))")
                }
            }
            .font(MFFont.body)
            .foregroundStyle(MFColor.ink)
            .padding(.horizontal, 56)
            .padding(.top, 22)
            Spacer()
        }
        .background(MFColor.surface)
    }
}

/// Leaving Notes mid-dictation (review of #218): the shell waits for the
/// last phrase, then switches.
@MainActor
@Suite struct VoiceLeaveNotesTests {
    @Test func leavingNotesLandsTheLastPhraseFirst() async throws {
        let speech = FakeSpeechService(dictation: .saying("one more thing"), stepped: true)
        let voice = VoiceModels(speech: speech, permissions: FakePermissionBroker(),
                                preferences: SpeechPreferences(store: InMemorySettingsStore()), isOffline: { false })
        let shell = ShellModel(selection: .notes, defaults: nil)
        shell.finishBeforeLeaving = { VoiceModels.finishBeforeLeaving($0, voice: voice) }
        let note = AppendingTarget(text: "Agenda.")
        await voice.dictation.start(in: note)
        speech.advance(); speech.advance() // level, phrase started
        await waitUntil("level and phrase handled") { voice.dictation.handledEventCount >= 2 }
        shell.show(.chat)
        #expect(shell.selection == .notes, "Notes stays until the phrase is in")
        await shell.waitForPendingSwitch()
        #expect(shell.selection == .chat)
        #expect(note.text == "Agenda. One more thing")
        #expect(!voice.dictation.isActive)
    }

    @Test func otherSwitchesAreImmediate() {
        let voice = VoiceModels(speech: FakeSpeechService(), permissions: FakePermissionBroker(),
                                preferences: SpeechPreferences(store: InMemorySettingsStore()), isOffline: { false })
        let shell = ShellModel(selection: .notes, defaults: nil)
        shell.finishBeforeLeaving = { VoiceModels.finishBeforeLeaving($0, voice: voice) }
        shell.show(.chat)
        #expect(shell.selection == .chat, "nothing to finish: the switch is immediate")
        #expect(VoiceModels.finishBeforeLeaving(.chat, voice: voice) == nil)
    }
}
