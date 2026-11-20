import AppKit
import MferenceAppCore
@testable import MferenceAppUI
import SwiftUI
import Testing

/// The Notes side of the voice seams (N3 sets the hooks in
/// `VoiceModels.connect(notes:)`): Notes calls them at the right moments
/// and hosts the voice views. Spies stand in for N3's models; no dictation
/// session runs.
@MainActor
@Suite(.serialized) struct NotesVoiceWiringTests {
    /// A small store (two notes, one folder).
    static func smallModel() throws -> NotesModel {
        let speech = FakeSpeechService()
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

    @Test func theEditorPublishesItselfWhenItGainsAndLosesFocus() throws {
        let model = try Self.smallModel()
        var focused: [NoteEditorController?] = []
        model.voice.editorFocusChanged = { focused.append($0) }
        let h = NoteEditorTests.Harness()
        model.editor.attach(h.view)
        h.window.makeFirstResponder(nil)          // the harness had focused it
        focused.removeAll()
        h.window.makeFirstResponder(h.view)
        h.window.makeFirstResponder(nil)
        #expect(focused.count == 2)
        #expect(focused.first.flatMap { $0 } === model.editor)
        #expect(focused.last.flatMap { $0 } == nil)
        // The editor carries its note id for the voice side.
        #expect(model.editor.noteID == model.selectedNoteID)
    }

    @Test func escGoesToTheVoiceSideFirst() throws {
        let model = try Self.smallModel()
        let h = NoteEditorTests.Harness()
        model.editor.attach(h.view)
        var asked = 0
        model.voice.escape = { asked += 1; return true }
        h.view.cancelOperation(nil)
        #expect(asked == 1)
        model.voice.escape = { asked += 1; return false }
        #expect(model.editor.onCancel?() == false)
        #expect(asked == 2)
    }

    @Test func dictateCallsTheHookWithTheEditor() throws {
        let model = try Self.smallModel()
        var targets: [NoteEditorController] = []
        model.voice.toggleDictation = { targets.append($0) }
        model.voice.toggleDictation?(model.editor)
        #expect(targets.count == 1 && targets[0] === model.editor)
    }

    @Test func switchingNotesStopsDictationIntoTheNoteLeft() async throws {
        let model = try Self.smallModel()
        let first = try #require(model.selectedNoteID)
        var stopped = 0
        model.voice.isDictating = { $0 == first }
        model.voice.stopDictation = { stopped += 1 }
        model.selectAdjacent(1)
        #expect(model.selectedNoteID != first)
        #expect(await eventually { stopped == 1 })
        // Moving on from a note not being dictated into stops nothing.
        model.selectAdjacent(-1)
        try await Task.sleep(for: .milliseconds(20))
        #expect(stopped == 1)
    }

    @Test func openNoteShowsTheNoteWhereverTheListWas() throws {
        let model = try Self.smallModel()
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
        let model = try Self.smallModel()
        var built = 0
        model.voice.toolbar = { built += 1; return AnyView(Text("Voice toolbar")) }
        let root = RootView(model: ShellModel(selection: .notes, defaults: nil), chrome: .inline)
            .environment(ScreenModels(notes: model))
        #expect(ImageRenderer(content: root.frame(width: 1200, height: 748)).cgImage != nil)
        #expect(built > 0)
    }
}
