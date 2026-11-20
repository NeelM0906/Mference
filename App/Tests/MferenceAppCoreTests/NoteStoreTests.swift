import Foundation
@testable import MferenceAppCore
import SwiftData
import Testing

/// NoteStore: CRUD, folders, pin, Recently deleted and purge (Notes spec 4.1,
/// 9). In-memory store, temporary attachment folder, `FakeSpeechService`.
@MainActor
@Suite struct NoteStoreTests {
    let speech = FakeSpeechService()
    let attachments = NoteAttachments.temporary()
    let store: NoteStore
    let t0 = Date(timeIntervalSince1970: 1_800_000_000)

    init() throws {
        store = NoteStore(container: try AppModelContainer.inMemory(), attachments: attachments, speech: speech)
    }

    @Test func createSavesAtOnceWithAnEmptyTitleLine() throws {
        let note = try store.createNote(now: t0)
        let other = ModelContext(store.container)
        let saved = try other.fetch(FetchDescriptor<Note>())
        #expect(saved.map(\.id) == [note.id])
        #expect(store.document(of: note) == .empty)
        #expect(store.displayTitle(of: note) == "New note")
        #expect(note.kind == .note)
    }

    @Test func savingABodyMovesTheTitleSearchTextAndEditTime() throws {
        let note = try store.createNote(now: t0)
        let body = NoteDocument(blocks: [
            .heading(level: .title, runs: [TextRun("Lisbon trip, March")]),
            .paragraph(runs: [TextRun("Hotel in "), TextRun("Alfama", bold: true)]),
        ])
        let before = store.revision
        try store.save(body, to: note, now: t0.addingTimeInterval(60))
        #expect(note.title == "Lisbon trip, March")
        #expect(note.updatedAt == t0.addingTimeInterval(60))
        #expect(note.searchText.contains("alfama"))
        #expect(store.document(of: note) == body)
        #expect(store.revision > before)

        // The same body again saves nothing and keeps the edit time.
        let revision = store.revision
        try store.save(body, to: note, now: t0.addingTimeInterval(600))
        #expect(note.updatedAt == t0.addingTimeInterval(60))
        #expect(store.revision == revision)
    }

    @Test func untitledNotesShowTheirFirstLine() throws {
        let note = try store.createNote(now: t0)
        try store.save(NoteDocument(blocks: [.heading(level: .title, runs: []),
                                             .paragraph(runs: [TextRun("Oat milk, eggs")])]), to: note)
        #expect(note.title.isEmpty)
        #expect(store.displayTitle(of: note) == "Oat milk, eggs")
    }

    @Test func setTitleKeepsTheTitleLineInStep() throws {
        let note = try store.createNote(kind: .meeting, now: t0)
        try store.setTitle("Design review: Notes", of: note)
        #expect(note.title == "Design review: Notes")
        #expect(store.document(of: note).blocks.first == .heading(level: .title, runs: [TextRun("Design review: Notes")]))
    }

    @Test func pinAndMoveKeepTheEditTime() throws {
        let note = try store.createNote(now: t0)
        let folder = try store.createFolder(named: "Personal")
        try store.setPinned(true, note)
        try store.move(note, to: folder)
        #expect(note.isPinned)
        #expect(note.folder?.id == folder.id)
        #expect(note.updatedAt == t0)
        #expect(try store.notes(in: .folder(folder.id)).map(\.id) == [note.id])
        try store.move(note, to: nil)
        #expect(try store.notes(in: .folder(folder.id)).isEmpty)
    }

    @Test func scopesAndCounts() throws {
        let work = try store.createFolder(named: "Work")
        try store.createNote(in: work, now: t0)
        try store.createNote(kind: .meeting, in: work, now: t0)
        try store.createNote(kind: .file, now: t0)
        let gone = try store.createNote(now: t0)
        try store.moveToTrash(gone, now: t0)

        let counts = try store.counts()
        #expect(counts == NoteCounts(all: 3, meetings: 1, audioTranscripts: 1, recentlyDeleted: 1,
                                     folders: [work.id: 2]))
        #expect(try store.count(in: .meetings) == 1)
        #expect(try store.count(in: .audioTranscripts) == 1)
        #expect(try store.notes(in: .recentlyDeleted).map(\.id) == [gone.id])
    }

    @Test func notesComeNewestEditFirst() throws {
        let a = try store.createNote(now: t0)
        let b = try store.createNote(now: t0.addingTimeInterval(10))
        try store.save(.titled("A"), to: a, now: t0.addingTimeInterval(20))
        #expect(try store.notes(in: .all).map(\.id) == [a.id, b.id])
    }

    @Test func duplicateCopiesTextAndAttachmentsNotTheRecording() throws {
        let note = try store.createNote(kind: .meeting, in: try store.createFolder(named: "Work"), now: t0)
        let image = try attachments.write(Data([1, 2, 3]), fileExtension: "png", into: note.id)
        try store.save(NoteDocument(blocks: [.heading(level: .title, runs: [TextRun("Standup")]),
                                             .image(attachmentID: image)]), to: note)
        let copy = try store.duplicate(note, now: t0.addingTimeInterval(5))
        #expect(copy.id != note.id)
        #expect(copy.title == "Standup")
        #expect(copy.kind == .note)
        #expect(copy.recording == nil)
        #expect(copy.folder?.id == note.folder?.id)
        #expect(store.document(of: copy) == store.document(of: note))
        #expect(try Data(contentsOf: attachments.url(of: image, in: copy.id)) == Data([1, 2, 3]))
    }

    // MARK: Recently deleted

    @Test func trashAndRestore() throws {
        let note = try store.createNote(now: t0)
        try store.setPinned(true, note)
        try store.moveToTrash(note, now: t0)
        #expect(try store.notes(in: .all).isEmpty)
        #expect(!note.isPinned)
        #expect(store.purgeDate(of: note) == t0.addingTimeInterval(NoteStore.trashRetention))
        try store.restore(note)
        #expect(try store.notes(in: .all).map(\.id) == [note.id])
        #expect(try store.notes(in: .recentlyDeleted).isEmpty)
    }

    @Test func purgeAfterThirtyDaysDeletesAttachmentsAndRecordings() async throws {
        let old = try store.createNote(kind: .meeting, now: t0)
        let recent = try store.createNote(now: t0)
        let kept = try store.createNote(now: t0)
        let oldID = old.id
        _ = try attachments.write(Data([9]), fileExtension: "jpg", into: oldID)
        try store.moveToTrash(old, now: t0)
        try store.moveToTrash(recent, now: t0.addingTimeInterval(20 * 86_400))

        let purged = try await store.purgeExpired(now: t0.addingTimeInterval(30 * 86_400 + 1))
        #expect(purged == 1)
        #expect(try store.note(id: oldID) == nil)
        #expect(try store.note(id: recent.id) != nil)
        #expect(try store.note(id: kept.id) != nil)
        #expect(attachments.list(for: oldID).isEmpty)
        #expect(speech.calls.contains(.deleteRecordings(oldID)))
        #expect(!speech.calls.contains(.deleteRecordings(recent.id)))
    }

    @Test func emptyTrashDeletesEverythingInIt() async throws {
        let a = try store.createNote(now: t0)
        let b = try store.createNote(now: t0)
        let (aID, bID) = (a.id, b.id)
        try store.moveToTrash(a)
        try store.moveToTrash(b)
        try await store.emptyTrash()
        #expect(try store.notes(in: .recentlyDeleted).isEmpty)
        #expect(Set(speech.calls.filter { if case .deleteRecordings = $0 { true } else { false } })
                == [.deleteRecordings(aID), .deleteRecordings(bID)])
    }

    @Test func aDeletedNoteRefusesEdits() async throws {
        let note = try store.createNote(now: t0)
        try await store.deletePermanently(note)
        #expect(throws: NoteStoreError.gone) { try store.save(.titled("x"), to: note) }
    }

    // MARK: Folders

    @Test func foldersCreateRenameReorderAndDelete() throws {
        let personal = try store.createFolder(named: "Personal")
        let work = try store.createFolder(named: "  Work ")
        let ideas = try store.createFolder()
        #expect(try store.folders().map(\.name) == ["Personal", "Work", "New folder"])
        #expect(try store.createFolder().name == "New folder 2")

        try store.rename(ideas, to: "Ideas")
        try store.rename(ideas, to: "   ")
        #expect(ideas.name == "Ideas")

        // Drag Ideas to the top.
        try store.moveFolders(fromOffsets: IndexSet(integer: 2), toOffset: 0)
        #expect(try store.folders().map(\.name) == ["Ideas", "Personal", "Work", "New folder 2"])

        // Deleting a folder leaves its notes in All notes.
        let note = try store.createNote(in: work, now: t0)
        try store.delete(work)
        #expect(note.folder == nil)
        #expect(try store.notes(in: .all).map(\.id) == [note.id])
        #expect(try store.folders().map(\.id).contains(personal.id))
        #expect(try !store.folders().map(\.name).contains("Work"))
    }

    // MARK: Search

    @Test func searchMatchesEveryWordAcrossTitleBodyAndTranscript() throws {
        let trip = try store.createNote(now: t0)
        try store.save(NoteDocument(blocks: [.heading(level: .title, runs: [TextRun("Lisbon trip")]),
                                             .check(runs: [TextRun("Café near the Sé")], done: false, indent: 0)]),
                       to: trip)
        let meeting = try store.createNote(kind: .meeting, title: "Design review", now: t0)
        let recording = NoteRecording(startedAt: t0)
        store.context.insert(recording)
        meeting.recording = recording
        recording.transcriptJSON = try JSONEncoder().encode([
            TranscriptSegment(track: .microphone, start: 3, end: 5, speakerID: SpeakerInfo.youID,
                              text: "Ship the Notes beta on Friday", isFinal: true),
        ])
        try store.transcriptDidChange(in: meeting)

        #expect(try store.notes(in: .all, matching: "cafe SE").map(\.id) == [trip.id])
        #expect(try store.notes(in: .all, matching: "LISBON café").map(\.id) == [trip.id])
        #expect(try store.notes(in: .all, matching: "beta friday").map(\.id) == [meeting.id])
        #expect(try store.notes(in: .all, matching: "lisbon friday").isEmpty)
        #expect(try store.notes(in: .all, matching: "  ").count == 2)
    }
}
