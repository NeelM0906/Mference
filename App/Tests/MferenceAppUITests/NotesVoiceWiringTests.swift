import AppKit
import MferenceAppCore
@testable import MferenceAppUI
import SwiftUI
import Testing

/// Notes wired to the voice team's models (N3): Dictate, ⇧⌘D's focused
/// editor, Esc, note switches, Open note, the toolbar. FakeSpeechService.
@MainActor
@Suite(.serialized) struct NotesVoiceWiringTests {
    /// A small store (two notes, one folder): these tests need little data.
    static func smallModel(speech: FakeSpeechService) throws -> NotesModel {
        let store = NoteStore(container: try AppModelContainer.inMemory(), attachments: .temporary(), speech: speech)
        let work = try store.createFolder(named: "Work")
        let dentist = try store.createNote(now: NotesFixtures.date(11, 16))
        try store.save(.titled("Dentist questions", body: [.paragraph(runs: [TextRun("Ask about the guard.")])]),
                       to: dentist, now: NotesFixtures.date(11, 16))
        let lisbon = try store.createNote(in: work, now: NotesFixtures.date(11, 19))
        try store.save(.titled("Lisbon trip, March"), to: lisbon, now: NotesFixtures.date(11, 19))
        let model = NotesModel(store: store, speech: speech, now: { NotesFixtures.now })
        model.select(lisbon)
        return model
    }

    func setUp() throws -> (NotesModel, VoiceModels, NoteEditorTests.Harness) {
        let speech = FakeSpeechService(dictation: .saying("then the hotel."))
        let model = try Self.smallModel(speech: speech)
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

    // These tests start no dictation session: dictation end to end is
    // covered by NoteDictationTargetTests (the editor) and N3's controller
    // tests; running sessions here made N3's timing-sensitive setup test flaky.

    @Test func dictateGoesThroughTheVoiceModels() throws {
        let speech = FakeSpeechService.notInstalled()
        let model = try Self.smallModel(speech: speech)
        let voice = VoiceModels(speech: speech, permissions: FakePermissionBroker(),
                                preferences: SpeechPreferences(store: InMemorySettingsStore()), isOffline: { false })
        model.connect(voice)
        // Not installed yet: the Dictate button brings up the setup card.
        model.voice.toggleDictation?(model.editor)
        #expect(voice.setup.pendingAction == .dictate)
        #expect(model.voice.isDictating?(try #require(model.selectedNoteID)) == false)
        // Esc without dictation is the text view's own.
        #expect(model.editor.onCancel?() == false)
    }

    @Test func switchingNotesStopsDictationIntoTheNoteLeft() throws {
        let (model, _, _) = try setUp()
        let first = try #require(model.selectedNoteID)
        var stopped = 0
        model.voice.isDictating = { $0 == first }
        model.voice.stopDictation = { stopped += 1 }
        model.selectAdjacent(1)
        #expect(model.selectedNoteID != first)
        #expect(stopped == 1)
        // Moving on from a note not being dictated into stops nothing.
        model.selectAdjacent(-1)
        #expect(stopped == 1)
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
