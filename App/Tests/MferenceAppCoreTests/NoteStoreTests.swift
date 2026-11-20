import Foundation
@testable import MferenceAppCore
import SwiftData
import Synchronization
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
        store.calendar = Self.utc
    }

    static let utc: Calendar = {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "UTC")!
        return calendar
    }()

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
        #expect(store.purgeDate(of: note) == t0.addingTimeInterval(30 * 86_400))
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
        await #expect(throws: NoteStoreError.notInTrash) { try await store.deletePermanently(note) }
        try store.moveToTrash(note)
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

    // MARK: Bodies that must not be overwritten (review #212 HIGH 1, MEDIUM 4)

    @Test func aBodyFromANewerBuildIsReadOnly() throws {
        let note = try store.createNote(now: t0)
        let newer = Data(#"{"v":2,"blocks":[{"type":"paragraph","runs":[{"t":"From the future"}]}]}"#.utf8)
        note.bodyJSON = newer
        try store.context.save()
        #expect(store.document(of: note).readOnlyReason == .newerVersion)
        #expect(throws: NoteStoreError.readOnly(.newerVersion)) { try store.save(.titled("Overwrite"), to: note) }
        #expect(throws: NoteStoreError.readOnly(.newerVersion)) { try store.setTitle("New", of: note) }
        #expect(note.bodyJSON == newer)
    }

    @Test func aDamagedBodyIsNeverOverwritten() throws {
        let note = try store.createNote(now: t0)
        let damaged = Data("{\"v\":1,\"blocks\":[{\"type\"".utf8)
        note.bodyJSON = damaged
        try store.context.save()
        #expect(store.document(of: note).readOnlyReason == .damaged)
        #expect(throws: NoteStoreError.readOnly(.damaged)) { try store.save(.titled("Overwrite"), to: note) }
        #expect(note.bodyJSON == damaged)
    }

    @Test func unknownAndBrokenBlocksSurviveASave() throws {
        let note = try store.createNote(now: t0)
        note.bodyJSON = Data(#"{"v":1,"blocks":[{"type":"heading","level":1,"runs":[{"t":"Trip"}]},{"type":"video","attachment":"clip.mov","runs":[{"t":"Clip"}]},{"type":"image"},{"type":"paragraph","runs":[{"t":"End"}]}]}"#.utf8)
        try store.context.save()
        var document = store.document(of: note)
        #expect(document.readOnlyReason == nil)
        #expect(document.blocks.count == 4)
        guard case .unknown(let video) = document.blocks[1], case .unknown = document.blocks[2] else {
            Issue.record("unknown blocks were not kept")
            return
        }
        #expect(video.runText == "Clip")
        document.blocks.append(.paragraph(runs: [TextRun("Added")]))
        try store.save(document, to: note)
        let saved = store.document(of: note)
        #expect(note.searchText.contains("clip"))
        #expect(saved.blocks.count == 5)
        #expect(saved.blocks[1] == .unknown(video))
        let json = try #require(String(data: note.bodyJSON ?? Data(), encoding: .utf8))
        #expect(json.contains(#""attachment":"clip.mov""#))
    }

    @Test func aBodyThatCannotBeEncodedIsNotSaved() throws {
        let note = try store.createNote(now: t0)
        try store.save(.titled("Kept"), to: note)
        let before = note.bodyJSON
        #expect(throws: (any Error).self) {
            try store.save(NoteDocument(blocks: [.marker(time: .nan)]), to: note)
        }
        #expect(note.bodyJSON == before)
        #expect(note.title == "Kept")
    }

    // MARK: Deleting for good (review #212 HIGH 2, MEDIUM 3, MEDIUM 5)

    @Test func whenRecordingsCannotBeDeletedTheNoteStaysAndOthersGo() async throws {
        let failing = ControlledSpeech()
        let store = NoteStore(container: try AppModelContainer.inMemory(), attachments: attachments, speech: failing)
        let stuck = try store.createNote(now: t0)
        let fine = try store.createNote(now: t0)
        let (stuckID, fineID) = (stuck.id, fine.id)
        _ = try attachments.write(Data([1]), fileExtension: "png", into: stuckID)
        failing.failDeletion(for: stuckID)
        try store.moveToTrash(stuck, now: t0)
        try store.moveToTrash(fine, now: t0)
        await #expect(throws: NoteStoreError.someNotDeleted(1)) { try await store.emptyTrash() }
        #expect(try store.note(id: stuckID) != nil)
        #expect(attachments.list(for: stuckID).count == 1)
        #expect(try store.note(id: fineID) == nil)
    }

    @Test func whenAttachmentsCannotBeDeletedTheNoteStays() async throws {
        let note = try store.createNote(now: t0)
        let id = note.id
        _ = try attachments.write(Data([1]), fileExtension: "png", into: id)
        try store.moveToTrash(note, now: t0)
        let folder = attachments.directory(for: id).path(percentEncoded: false)
        try FileManager.default.setAttributes([.posixPermissions: 0o500], ofItemAtPath: folder)
        defer { try? FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: folder) }
        await #expect(throws: (any Error).self) { try await store.deletePermanently(note) }
        #expect(try store.note(id: id) != nil)
    }

    @Test func aNoteCannotBeRestoredWhileItIsBeingDeletedAndRestoredNotesSurviveAPurge() async throws {
        let gated = ControlledSpeech()
        gated.holdDeletions()
        let store = NoteStore(container: try AppModelContainer.inMemory(), attachments: attachments, speech: gated)
        store.calendar = Self.utc
        // The purge takes newest edits first: `first`, then `second`.
        let first = try store.createNote(now: t0)
        let second = try store.createNote(now: t0.addingTimeInterval(-60))
        let (firstID, secondID) = (first.id, second.id)
        _ = try attachments.write(Data([2]), fileExtension: "png", into: secondID)
        try store.moveToTrash(first, now: t0)
        try store.moveToTrash(second, now: t0)

        let purge = Task { try await store.purgeExpired(now: t0.addingTimeInterval(31 * 86_400)) }
        // The first note's recordings are being deleted: it cannot come back.
        #expect(await eventuallyAsync { gated.waiting.contains(firstID) })
        #expect(store.isBeingDeleted(first))
        #expect(throws: NoteStoreError.beingDeleted) { try store.restore(first) }
        // The second, not yet reached, is restored and must survive.
        try store.restore(second)
        gated.releaseDeletions()
        #expect(try await purge.value == 1)
        #expect(try store.note(id: firstID) == nil)
        #expect(try store.note(id: secondID)?.deletedAt == nil)
        #expect(attachments.list(for: secondID).count == 1)
        #expect(!gated.deleted.contains(secondID))
    }

    @Test func purgesRunOneAtATime() async throws {
        for _ in 0..<3 { try store.moveToTrash(try store.createNote(now: t0), now: t0) }
        let later = t0.addingTimeInterval(40 * 86_400)
        async let a = store.purgeExpired(now: later)
        async let b = store.purgeExpired(now: later)
        let counts = try await [a, b]
        #expect(counts.reduce(0, +) == 3)
        #expect(try store.notes(in: .recentlyDeleted).isEmpty)
    }

    @Test func thePurgeBoundaryIsThirtyCalendarDays() async throws {
        let note = try store.createNote(now: t0)
        try store.moveToTrash(note, now: t0)
        #expect(try await store.purgeExpired(now: t0.addingTimeInterval(30 * 86_400 - 1)) == 0)
        #expect(try await store.purgeExpired(now: t0.addingTimeInterval(30 * 86_400)) == 1)
    }

    @Test func purgeCountsDaysAcrossADaylightSavingChange() async throws {
        var lisbon = Calendar(identifier: .gregorian)
        lisbon.timeZone = TimeZone(identifier: "Europe/Lisbon")!
        store.calendar = lisbon
        // Deleted at noon on 10 October 2026; clocks go back on 25 October.
        let deleted = lisbon.date(from: DateComponents(year: 2026, month: 10, day: 10, hour: 12))!
        let note = try store.createNote(now: deleted)
        try store.moveToTrash(note, now: deleted)
        let due = lisbon.date(from: DateComponents(year: 2026, month: 11, day: 9, hour: 12))!
        #expect(store.purgeDate(of: note) == due)
        // 30 × 24 hours later is an hour early: not yet.
        #expect(try await store.purgeExpired(now: deleted.addingTimeInterval(30 * 86_400)) == 0)
        #expect(try await store.purgeExpired(now: due) == 1)
    }
}

/// Polls an async condition for up to ten seconds.
@MainActor
func eventuallyAsync(_ condition: () async -> Bool) async -> Bool {
    let deadline = ContinuousClock.now + .seconds(10)
    while ContinuousClock.now < deadline {
        if await condition() { return true }
        try? await Task.sleep(for: .milliseconds(5))
    }
    return false
}

/// A speech service whose recording deletion can fail or wait, over
/// `FakeSpeechService` for everything else.
final class ControlledSpeech: SpeechServicing {
    private let base = FakeSpeechService()
    private struct State {
        var failing: Set<UUID> = []
        var hold = false
        var waiting: Set<UUID> = []
        var deleted: [UUID] = []
        var releases: [CheckedContinuation<Void, Never>] = []
    }
    private let state = Mutex(State())

    func failDeletion(for id: UUID) { state.withLock { _ = $0.failing.insert(id) } }
    func holdDeletions() { state.withLock { $0.hold = true } }
    func releaseDeletions() {
        let releases = state.withLock { s -> [CheckedContinuation<Void, Never>] in
            s.hold = false
            defer { s.releases = [] }
            return s.releases
        }
        for release in releases { release.resume() }
    }
    var waiting: Set<UUID> { state.withLock { $0.waiting } }
    var deleted: [UUID] { state.withLock { $0.deleted } }

    func deleteRecordings(for noteID: UUID) async throws {
        if state.withLock({ $0.failing.contains(noteID) }) { throw SpeechError.offline }
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            let held = state.withLock { s -> Bool in
                guard s.hold else { return false }
                s.waiting.insert(noteID)
                s.releases.append(continuation)
                return true
            }
            if !held { continuation.resume() }
        }
        state.withLock { s in
            s.waiting.remove(noteID)
            s.deleted.append(noteID)
        }
    }

    var status: SpeechStatus { base.status }
    func statusUpdates() -> AsyncStream<SpeechStatus> { base.statusUpdates() }
    func downloadSize(of package: SpeechPackageID) -> Int64 { base.downloadSize(of: package) }
    func install(_ package: SpeechPackageID) async throws { try await base.install(package) }
    func pauseInstall(_ package: SpeechPackageID) async { await base.pauseInstall(package) }
    func cancelInstall(_ package: SpeechPackageID) async { await base.cancelInstall(package) }
    func remove(_ package: SpeechPackageID) async throws { try await base.remove(package) }
    func startDictation(_ options: DictationOptions) async throws -> DictationSession { try await base.startDictation(options) }
    func startMeeting(_ options: MeetingOptions) async throws -> MeetingRecording { try await base.startMeeting(options) }
    func transcribeFile(_ url: URL, options: FileTranscriptionOptions) async throws -> FileJob {
        try await base.transcribeFile(url, options: options)
    }
    func cancel(_ job: FileJob.ID) async { await base.cancel(job) }
    func inputDevices() async -> [AudioInputDevice] { await base.inputDevices() }
    func runningAudioApps() async -> [RunningAudioApp] { await base.runningAudioApps() }
    func recordingsSize() async -> Int64 { await base.recordingsSize() }
    func recordingsDirectory(for noteID: UUID) -> URL { base.recordingsDirectory(for: noteID) }
    func recoverInterruptedRecordings() async -> [InterruptedRecording] { await base.recoverInterruptedRecordings() }
    func shutdown() async { await base.shutdown() }
}
