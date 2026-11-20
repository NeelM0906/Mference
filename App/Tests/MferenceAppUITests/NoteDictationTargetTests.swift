import AppKit
import MferenceAppCore
@testable import MferenceAppUI
import SwiftData
import Testing

/// The dictation insertion API N3's controller drives, end to end over
/// `FakeSpeechService` (no model, no microphone), and the meeting hooks.
@MainActor
@Suite(.serialized) struct NoteDictationTargetTests {
    /// What N3's DictationController does with each event (its one consumer).
    @discardableResult
    func apply(_ event: DictationEvent, to target: any DictationInsertionTarget) -> Bool {
        switch event {
        case .phraseStarted(let id): target.beginPhrase(id)
        case .phraseFinished(let id, let text, _): target.finishPhrase(id, text: text)
        case .phraseDropped(let id): target.dropPhrase(id)
        case .failed: target.dropAllPhrases()
        case .ended:
            target.dropAllPhrases()
            return false
        case .level: break
        }
        return true
    }

    func editor() throws -> (NoteEditorController, NoteStore, Note, NoteEditorTests.Harness) {
        let store = NoteStore(container: try AppModelContainer.inMemory(), attachments: .temporary(),
                              speech: FakeSpeechService())
        let note = try store.createNote()
        try store.save(NoteDocument(blocks: [.heading(level: .title, runs: [TextRun("Dentist")]),
                                             .paragraph(runs: [TextRun("Ask about the guard.")])]), to: note)
        let controller = NoteEditorController(store: store, autosaveDelay: .milliseconds(10))
        let h = NoteEditorTests.Harness()
        controller.attach(h.view)
        controller.open(note)
        return (controller, store, note, h)
    }

    @Test func aDictationSessionLandsEachPhraseAtItsCaretWhileThePersonTypes() async throws {
        let (controller, store, note, h) = try editor()
        let target: any DictationInsertionTarget = controller
        #expect(target.noteID == note.id)

        // The fake plays the whole session into the stream; it is consumed
        // in order, as N3's controller does, with the person acting between
        // events.
        let speech = FakeSpeechService(dictation: .saying("and the cost", "book it."))
        let session = try await speech.startDictation(DictationOptions())
        await session.stop()
        var started = 0
        for await event in session.events {
            guard apply(event, to: target) else { break }
            if case .phraseStarted = event {
                started += 1
                if started == 1 {
                    // Caret after "guard." when the first phrase started; the
                    // person clicks to the start of the line and types.
                    h.view.setSelectedRange(NSRange(location: 8, length: 0))
                    h.type("Q: ")
                }
            }
        }
        #expect(h.view.document.blocks[1] == .paragraph(runs: [TextRun("Q: book it. Ask about the guard. And the cost")]))
        #expect(h.view.pendingPhraseIDs.isEmpty)
        #expect(note.hasDictation)
        #expect(await eventually { controller.saveState == .saved })
        #expect(store.document(of: note).blocks[1] == .paragraph(runs: [TextRun("Q: book it. Ask about the guard. And the cost")]))
    }

    @Test func droppedAndEndedSessionsLeaveNoMarkers() throws {
        let (controller, _, note, h) = try editor()
        let (a, b) = (UUID(), UUID())
        apply(.phraseStarted(a), to: controller)
        apply(.phraseStarted(b), to: controller)
        apply(.phraseDropped(a), to: controller)
        #expect(h.view.pendingPhraseIDs == [b])
        #expect(apply(.ended, to: controller) == false)
        #expect(h.view.pendingPhraseIDs.isEmpty)
        #expect(!note.hasDictation)
        // Switching notes also forgets pending phrases.
        apply(.phraseStarted(UUID()), to: controller)
        controller.open(nil)
        #expect(h.view.pendingPhraseIDs.isEmpty)
    }

    @Test func meetingHooksInsertMarkersAndTimeChips() throws {
        let (controller, store, note, h) = try editor()
        let recording = NoteRecording()
        store.context.insert(recording)
        note.recording = recording
        let segment = TranscriptSegment(track: .microphone, start: 61, end: 64, speakerID: SpeakerInfo.youID,
                                        text: "Budget first", isFinal: true)
        recording.transcriptJSON = try JSONEncoder().encode([segment])
        let target: any MeetingNoteTarget = controller
        h.view.setSelectedRange(NSRange(location: 0, length: 0))
        target.insertMarker(time: 90)
        target.insertTranscriptAnchor(segmentID: segment.id)
        let blocks = h.view.document.blocks
        #expect(blocks.contains(.marker(time: 90)))
        #expect(blocks.contains(.transcriptAnchor(segmentID: segment.id)))
        let chip = try #require(h.view.textStorage?.attribute(.attachment, at: (h.view.string as NSString).range(of: "\u{FFFC}", options: .backwards).location, effectiveRange: nil) as? NoteBlockAttachment)
        #expect(chip.time == 61)
        target.reveal(segmentID: segment.id)
        #expect(h.view.selectedRange().location == (h.view.string as NSString).range(of: "\u{FFFC}", options: .backwards).location)
    }
}
