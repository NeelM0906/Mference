import AppKit
import Foundation
import MferenceAppCore
@testable import MferenceAppUI
import Observation
import SwiftData
import SwiftUI
import Synchronization
import Testing

/// The voice screens wired to the real owners (N4's MeetingSession and
/// FileTranscriptionQueue, N1's NotesModel) over FakeSpeechService.
@MainActor
@Suite(.serialized) struct VoiceWiringTests {
    /// Waits for `condition` (see `waitUntil`), never for a count of yields.
    private func eventually(sourceLocation: SourceLocation = #_sourceLocation,
                            _ condition: @MainActor () -> Bool) async throws {
        await waitUntil(timeout: .seconds(10), sourceLocation: sourceLocation, condition)
    }

    @Test func liveVoiceModelsDriveTheServicesMeetingSession() async throws {
        let speech = FakeSpeechService()
        let services = AppServices.preview(speech: speech)
        let voice = VoiceModels.live(services: services)
        defer { voice.setup.stop() }
        voice.connect(meetings: services.meetings)
        #expect(voice.preferences === services.speechPreferences, "one SpeechPreferences")
        #expect(voice.dictation === services.dictation)
        #expect(await voice.recordMeetingModel.startRecording())
        #expect(services.meetings.isActive && voice.isMeetingActive)
        try await eventually { voice.canMarkMoment }
        await voice.stopLiveSessions()
        #expect(!services.meetings.isActive, "quit's stop reaches the session")
    }

    @Test func theImportSheetShowsTheRealQueue() async throws {
        let speech = FakeSpeechService()
        let services = AppServices.preview(speech: speech)
        let voice = VoiceModels.live(services: services)
        defer { voice.setup.stop() }
        voice.connect(files: services.files)
        speech.setFileResult(.success(TranscriptResult(
            segments: [TranscriptSegment(track: .file, start: 0, end: 2, speakerID: "s1", text: "Hi", isFinal: true)],
            speakers: [SpeakerInfo(id: "s1", name: "Speaker 1", colorIndex: 1)], duration: 120,
            languageCode: "en")), forFileNamed: "talk.m4a")
        let file = FileManager.default.temporaryDirectory.appending(path: "talk.m4a")
        FileManager.default.createFile(atPath: file.path, contents: Data([0]))
        defer { try? FileManager.default.removeItem(at: file) }
        #expect(voice.transcribe([file]))
        let queue = try #require(voice.fileQueue)
        try await eventually { queue.rows.first?.phase == .done }
        let row = try #require(queue.rows.first)
        #expect(row.fileName == "talk.m4a" && row.noteID == row.id)
        #expect(FileQueueAdapter.phase(.failed("x")) == .failed("x"))
    }

    @Test func everyNotesVoiceHookIsSet() async throws {
        let services = AppServices.preview(speech: FakeSpeechService(stepped: true))
        let voice = VoiceModels.live(services: services)
        defer { voice.setup.stop() }
        let notes = NotesModel(store: services.notes, speech: services.speech)
        voice.connect(notes: notes)
        let hooks = notes.voice
        #expect(hooks.transcribeFile != nil && hooks.recordMeeting != nil && hooks.toolbar != nil
                && hooks.toggleDictation != nil && hooks.isDictating != nil && hooks.stopDictation != nil
                && hooks.editorFocusChanged != nil && hooks.escape != nil && hooks.transcribe != nil,
                "N1 defines and calls them; N3 sets every one")
        hooks.transcribeFile?()
        #expect(voice.showsImportSheet)
        hooks.recordMeeting?()
        #expect(voice.showsRecordPopover)

        // The editor's focus feeds ⇧⌘D; Dictate and "Dictating" follow the dictation.
        let note = try services.notes.createNote(title: "Dentist questions")
        notes.open(noteID: note.id)
        #expect(notes.selectedNoteID == note.id)
        hooks.editorFocusChanged?(notes.editor)
        #expect(voice.canToggleDictation && voice.focusedDictationNoteID == note.id)
        hooks.toggleDictation?(notes.editor)
        await waitUntil("dictation into the note") { voice.dictation.isDictating(into: note.id) }
        #expect(hooks.isDictating?(note.id) == true)
        #expect(hooks.escape?() == true, "Esc stops a dictation")
        await waitUntil("dictation to stop") { !voice.dictation.isActive }
        #expect(hooks.escape?() == false, "otherwise Esc is the editor's")
        hooks.editorFocusChanged?(nil)
        #expect(!voice.canToggleDictation)
        #expect(hooks.transcribe?([URL(fileURLWithPath: "/tmp/slides.pdf")]) == false)
    }

    @Test func openNoteShowsTheNoteInNotes() throws {
        let services = AppServices.preview(speech: FakeSpeechService())
        let voice = VoiceModels.live(services: services)
        defer { voice.setup.stop() }
        let notes = NotesModel(store: services.notes, speech: services.speech)
        voice.connect(notes: notes)
        let note = try services.notes.createNote(kind: .meeting, title: "Design review")
        voice.openNote?(note.id)
        #expect(notes.selectedNoteID == note.id, "the timer chip and Open note select it")
    }

    // MARK: The shell's wait before leaving Notes

    @Test func aStalledStopCannotFreezeTheSections() async {
        let shell = ShellModel(selection: .notes, defaults: nil)
        shell.leaveTimeout = .milliseconds(100)
        shell.finishBeforeLeaving = { _ in { try? await Task.sleep(for: .seconds(30)) } }
        shell.show(.chat)
        await shell.waitForPendingSwitch()
        #expect(shell.selection == .chat, "the switch happens after the timeout")
    }

    @Test func backToNotesDuringTheWaitStaysInNotes() async {
        let shell = ShellModel(selection: .notes, defaults: nil)
        shell.finishBeforeLeaving = { _ in { try? await Task.sleep(for: .milliseconds(200)) } }
        shell.show(.chat)
        shell.show(.notes) // changed their mind
        try? await Task.sleep(for: .milliseconds(400))
        #expect(shell.selection == .notes)
    }

    // MARK: ⇧⌘D and Esc

    @Test func shortcutStopsTheRunningDictationEvenWithAnotherNoteShown() async throws {
        let speech = FakeSpeechService(stepped: true)
        let voice = VoiceModels(speech: speech, permissions: FakePermissionBroker(),
                                preferences: SpeechPreferences(store: InMemorySettingsStore()), isOffline: { false })
        let noteA = AppendingTarget()
        let noteB = AppendingTarget()
        await voice.dictation.start(in: noteA)
        voice.focusedDictationTarget = noteB // the reused editor now shows B
        voice.toggleDictation()
        try await eventually { !voice.dictation.isActive }
        #expect(speech.calls.filter { if case .startDictation = $0 { true } else { false } }.count == 1,
                "⇧⌘D stopped A instead of starting B")
    }

    @Test func escInTheEditorStopsDictation() async throws {
        let voice = VoiceModels(speech: FakeSpeechService(stepped: true), permissions: FakePermissionBroker(),
                                preferences: SpeechPreferences(store: InMemorySettingsStore()), isOffline: { false })
        #expect(!voice.cancelDictation(), "not dictating: Esc is the editor's")
        await voice.dictation.start(in: AppendingTarget())
        #expect(voice.cancelDictation())
        try await eventually { !voice.dictation.isActive }
    }

    @Test func theDictateMenuItemReEnablesWhenANoteGetsFocus() throws {
        let voice = VoiceModels(speech: FakeSpeechService(), permissions: FakePermissionBroker(),
                                preferences: SpeechPreferences(store: InMemorySettingsStore()), isOffline: { false })
        let shell = ShellModel(selection: .notes, defaults: nil)
        // The items render (the app's own view code, not a stand-in).
        #expect(ImageRenderer(content: VStack { VoiceMenuItems(voice: voice, shell: shell) }).cgImage != nil)
        #expect(!voice.canToggleDictation)
        let changed = Mutex(false)
        withObservationTracking {
            _ = VoiceMenuItems(voice: voice, shell: shell).body
        } onChange: {
            changed.withLock { $0 = true }
        }
        let note = AppendingTarget() // the focused editor (held by its view in the app)
        voice.focusedDictationTarget = note
        #expect(changed.withLock { $0 }, "the menu items' body re-evaluates when a note gets focus")
        #expect(voice.canToggleDictation, "Dictate is enabled")
        voice.focusedDictationTarget = nil
        #expect(!voice.canToggleDictation, "and disabled again when focus leaves the note")
    }
}
