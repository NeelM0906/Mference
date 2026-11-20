import AppKit
import MferenceAppCore
@testable import MferenceAppUI
import SwiftUI
import Testing

/// Notes wired to the voice team's models (N3): Dictate, ⇧⌘D's focused
/// editor, Esc, note switches, Open note, the toolbar. FakeSpeechService.
@MainActor
@Suite(.serialized) struct NotesVoiceWiringTests {
    func setUp() throws -> (NotesModel, VoiceModels, NoteEditorTests.Harness) {
        let speech = FakeSpeechService(dictation: .saying("then the hotel."))
        let model = try NotesFixtures.model(speech: speech)
        let voice = VoiceModels(speech: speech,
                                permissions: FakePermissionBroker(states: [.microphone: .granted]),
                                preferences: SpeechPreferences(store: InMemorySettingsStore()), isOffline: { false })
        model.connect(voice)
        let h = NoteEditorTests.Harness()
        model.editor.attach(h.view)
        return (model, voice, h)
    }

    @Test func theFocusedEditorIsWhereShiftCommandDDictates() throws {
        let (model, voice, h) = try setUp()
        h.window.makeFirstResponder(nil)
        h.window.makeFirstResponder(h.view)
        #expect(voice.focusedDictationTarget === model.editor)
        #expect(voice.focusedDictationNoteID == model.selectedNoteID)
        h.window.makeFirstResponder(nil)
        #expect(voice.focusedDictationTarget == nil)
    }

    @Test func dictateFromTheButtonLandsInTheNoteAndEscStops() async throws {
        let (model, voice, h) = try setUp()
        let note = try #require(model.selectedNote)
        h.view.setSelectedRange(NSRange(location: (h.view.string as NSString).length, length: 0))
        model.voice.toggleDictation?(model.editor)
        #expect(await eventually { voice.dictation.isDictating(into: note.id) || note.hasDictation })
        #expect(await eventually { note.hasDictation })
        #expect(h.view.document.plainText.contains("then the hotel."))
        // Esc in the editor stops dictation when one runs; otherwise it is the
        // text view's own.
        if voice.dictation.isActive {
            #expect(model.editor.onCancel?() == true)
            #expect(await eventually { !voice.dictation.isActive })
        }
        #expect(model.editor.onCancel?() == false)
    }

    @Test func switchingNotesStopsDictation() async throws {
        let speech = FakeSpeechService(dictation: .saying("one", "two"), stepped: true)
        let model = try NotesFixtures.model(speech: speech)
        let voice = VoiceModels(speech: speech, permissions: FakePermissionBroker(states: [.microphone: .granted]),
                                preferences: SpeechPreferences(store: InMemorySettingsStore()), isOffline: { false })
        model.connect(voice)
        let h = NoteEditorTests.Harness()
        model.editor.attach(h.view)
        let first = try #require(model.selectedNote)
        model.voice.toggleDictation?(model.editor)
        #expect(await eventually { voice.dictation.isDictating(into: first.id) })
        model.selectAdjacent(1)
        #expect(await eventually { !voice.dictation.isActive })
    }

    @Test func openNoteShowsTheNoteWhereverTheListWas() throws {
        let (model, _, _) = try setUp()
        let dentist = try #require(model.notes.first { model.title(of: $0) == "Dentist questions" })
        let work = try #require(model.folders.first { $0.name == "Work" })
        model.scope = .folder(work.id)
        model.query = "nothing"
        model.commitQuery()
        model.open(noteID: dentist.id)
        #expect(model.selectedNoteID == dentist.id)
        #expect(model.scope == .all)
        #expect(model.appliedQuery.isEmpty)
    }

    @Test func theShellBarHostsTheVoiceToolbar() throws {
        let (model, _, _) = try setUp()
        #expect(model.voice.toolbar != nil)
        let root = RootView(model: ShellModel(selection: .notes, defaults: nil), chrome: .inline)
            .environment(ScreenModels(notes: model))
        #expect(ImageRenderer(content: root.frame(width: 1200, height: 748)).cgImage != nil)
    }
}
