import Foundation
import SwiftData
import Testing
@testable import MferenceAppCore

/// `FileTranscriptionQueue` on `FakeSpeechService` file results: notes made
/// per file, states, transcript writing, the optional summary, cancel,
/// failures, and launch recovery. Model-free; no file is read.
@MainActor
@Suite struct FileTranscriptionQueueTests {
    @MainActor struct Harness {
        let container: ModelContainer
        let writer: InMemoryMeetingNoteWriter
        let queue: FileTranscriptionQueue
        let speech: FakeSpeechService
        var context: ModelContext { container.mainContext }

        func note(_ id: UUID) -> Note? {
            try? context.fetch(FetchDescriptor<Note>(predicate: #Predicate { $0.id == id })).first
        }
    }

    func harness(_ speech: FakeSpeechService) throws -> Harness {
        let container = try AppModelContainer.inMemory()
        let writer = InMemoryMeetingNoteWriter(context: container.mainContext)
        return Harness(container: container, writer: writer,
                       queue: FileTranscriptionQueue(speech: speech, writer: writer),
                       speech: speech)
    }

    static let transcript = TranscriptResult(
        segments: [TranscriptSegment(track: .file, start: 0, end: 4, speakerID: SpeakerInfo.othersID,
                                     text: "Welcome to week three.", isFinal: true)],
        speakers: [.others],
        tracks: [RecordedTrack(fileName: "lecture-week-3.mp3", kind: .file, startOffset: 0, duration: 4_680)],
        duration: 4_680, languageCode: "en", audioDirectory: URL(filePath: "/tmp/rec"))

    @Test func eachFileBecomesANoteWithItsTranscript() async throws {
        let speech = FakeSpeechService(stepped: true)
        speech.setFileResult(.success(Self.transcript), forFileNamed: "lecture-week-3.mp3")
        let h = try harness(speech)
        var summarized: [UUID] = []
        h.queue.afterTranscript = { summarized.append($0) }
        let ids = await h.queue.add([URL(filePath: "/Users/me/lecture-week-3.mp3")], options: FileImportOptions())
        let id = try #require(ids.first)
        #expect(h.note(id)?.kind == .file)
        #expect(h.note(id)?.title == "lecture-week-3")
        #expect(h.note(id)?.recording?.sourceFileName == "lecture-week-3.mp3")
        #expect(h.note(id)?.recording?.transcriptionState == .queued)
        #expect(h.queue.rows.map(\.phase) == [.queued])
        #expect(h.queue.isBusy)

        speech.advance() // queued
        speech.advance() // started
        for _ in 0..<50 { await Task.yield() }
        #expect(h.note(id)?.recording?.transcriptionState == .running)
        #expect(h.queue.rows.first?.duration == 60)
        speech.advance() // progress
        for _ in 0..<50 { await Task.yield() }
        #expect(h.queue.rows.first?.fraction == 0.5 && h.queue.rows.first?.languageCode == "en")

        speech.advance() // finished
        await h.queue.waitUntilIdle()
        let record = try #require(h.note(id)?.recording)
        #expect(record.transcriptionState == .done)
        #expect(record.segments.map(\.text) == ["Welcome to week three."])
        #expect(record.duration == 4_680 && record.languageCode == "en" && record.audioKept)
        #expect(h.queue.rows.first?.phase == .done)
        #expect(h.queue.rows.first?.speakerCount == 1)
        #expect(!h.queue.isBusy)
        #expect(summarized.isEmpty)
        #expect(h.writer.searchRefreshes[id] == 1)
        h.queue.clearFinished()
        #expect(h.queue.rows.isEmpty)
    }

    @Test func aFileThatAskedForASummaryGetsOne() async throws {
        let speech = FakeSpeechService()
        speech.setFileResult(.success(Self.transcript), forFileNamed: "call.m4a")
        let h = try harness(speech)
        var summarized: [UUID] = []
        h.queue.afterTranscript = { summarized.append($0) }
        let ids = await h.queue.add([URL(filePath: "/tmp/call.m4a")], options: FileImportOptions(summarize: true))
        await h.queue.waitUntilIdle()
        #expect(summarized == ids)
        #expect(h.note(ids[0])?.recording?.summaryState == .pending)
    }

    @Test func aFailedFileKeepsItsNoteAndSaysWhy() async throws {
        let speech = FakeSpeechService()
        speech.setFileResult(.failure(.unsupportedFile("no audio")), forFileNamed: "slides.pdf")
        let h = try harness(speech)
        let ids = await h.queue.add([URL(filePath: "/tmp/slides.pdf")], options: FileImportOptions(summarize: true))
        await h.queue.waitUntilIdle()
        #expect(h.note(ids[0])?.recording?.transcriptionState == .failed)
        #expect(h.note(ids[0])?.recording?.summaryState == RecordingSummaryState.none)
        #expect(h.queue.rows.first?.phase == .failed(SpeechError.unsupportedFile("").message))
    }

    @Test func aRefusedStartFailsTheRow() async throws {
        let speech = FakeSpeechService.notInstalled()
        let h = try harness(speech)
        let ids = await h.queue.add([URL(filePath: "/tmp/a.m4a")], options: FileImportOptions())
        #expect(h.queue.rows.first?.phase == .failed(SpeechError.notInstalled(.speech).message))
        #expect(h.note(ids[0])?.recording?.transcriptionState == .failed)
    }

    @Test func cancellingRemovesTheRowAndItsNote() async throws {
        let speech = FakeSpeechService(stepped: true)
        let h = try harness(speech)
        let ids = await h.queue.add([URL(filePath: "/tmp/a.m4a"), URL(filePath: "/tmp/b.m4a")],
                                    options: FileImportOptions())
        await h.queue.cancel(ids[0])
        for _ in 0..<50 { await Task.yield() }
        #expect(h.queue.rows.map(\.noteID) == [ids[1]])
        #expect(h.note(ids[0]) == nil)
        #expect(h.note(ids[1]) != nil)
        #expect(speech.calls.contains { if case .cancel = $0 { true } else { false } })
    }

    @Test func cancellingThroughTheNotesStoreDeletesTheNoteAndItsRecordings() async throws {
        let speech = FakeSpeechService(stepped: true)
        let container = try AppModelContainer.inMemory()
        let notes = NoteStore(container: container, attachments: .temporary(), speech: speech)
        let queue = FileTranscriptionQueue(speech: speech, writer: notes)
        let ids = await queue.add([URL(filePath: "/tmp/lecture.m4a")], options: FileImportOptions(summarize: true))
        let id = try #require(ids.first)
        speech.advance() // queued
        speech.advance() // started
        for _ in 0..<50 { await Task.yield() }
        await queue.cancel(id)
        for _ in 0..<500 where (try? notes.note(id: id)) != nil { await Task.yield() }
        #expect(try notes.note(id: id) == nil)
        #expect(speech.calls.contains(.deleteRecordings(id)))
        #expect(queue.rows.isEmpty)
        // Nothing is left for the next launch to re-queue.
        #expect(await queue.recoverInterruptedJobs().isEmpty)
    }

    @Test func cancellingWhileTheJobIsBeingCreatedCancelsIt() async throws {
        let fake = FakeSpeechService(stepped: true)
        let speech = GatedFileSpeech(fake)
        let container = try AppModelContainer.inMemory()
        let notes = NoteStore(container: container, attachments: .temporary(), speech: speech)
        let queue = FileTranscriptionQueue(speech: speech, writer: notes)
        let adding = Task { await queue.add([URL(filePath: "/tmp/slow.m4a")], options: FileImportOptions()) }
        for _ in 0..<500 where queue.rows.isEmpty || !speech.isWaiting { await Task.yield() }
        let id = try #require(queue.rows.first?.noteID)
        await queue.cancel(id)
        speech.open()
        _ = await adding.value
        for _ in 0..<500 where (try? notes.note(id: id)) != nil { await Task.yield() }
        #expect(fake.calls.contains { if case .cancel = $0 { true } else { false } })
        #expect(try notes.note(id: id) == nil)
        #expect(queue.rows.isEmpty)
    }

    @Test func filesWaitWhileAMeetingRecords() async throws {
        let speech = FakeSpeechService(stepped: true)
        let h = try harness(speech)
        let meeting = try await speech.startMeeting(MeetingOptions(noteID: UUID()))
        _ = await h.queue.add([URL(filePath: "/tmp/a.m4a")], options: FileImportOptions())
        speech.advance() // the meeting's .started
        speech.advance() // the file's .queued
        speech.advance() // the file's .paused
        for _ in 0..<50 { await Task.yield() }
        #expect(h.queue.rows.first?.phase == .paused)
        await meeting.stop()
    }

    @Test func launchRecoveryLeavesAFileAlreadyQueued() async throws {
        let speech = FakeSpeechService(stepped: true)
        let h = try harness(speech)
        let ids = await h.queue.add([URL(filePath: "/tmp/early.m4a")], options: FileImportOptions())
        #expect(await h.queue.recoverInterruptedJobs().isEmpty)
        #expect(speech.calls.filter { if case .transcribeFile = $0 { true } else { false } }.count == 1)
        #expect(h.queue.rows.map(\.noteID) == ids)
    }

    @Test func launchRequeuesFilesWithACopyAndFailsTheRest() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "n4-queue-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let speech = FakeSpeechService(recordingsRoot: root)
        let h = try harness(speech)
        func interrupted(_ fileName: String, state: TranscriptionState, copy: Bool) throws -> UUID {
            let id = UUID()
            try h.writer.createNote(id: id, kind: .file, title: fileName, createdAt: .now)
            let record = NoteRecording(sourceFileName: fileName)
            record.transcriptionState = state
            h.context.insert(record)
            h.note(id)?.recording = record
            try h.context.save()
            if copy {
                let folder = speech.recordingsDirectory(for: id)
                try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
                try Data([0]).write(to: folder.appending(path: fileName))
            }
            return id
        }
        let kept = try interrupted("kept.m4a", state: .running, copy: true)
        let gone = try interrupted("gone.m4a", state: .queued, copy: false)
        let done = try interrupted("done.m4a", state: .done, copy: false)

        let requeued = await h.queue.recoverInterruptedJobs()
        #expect(requeued == [kept])
        await h.queue.waitUntilIdle()
        #expect(h.note(kept)?.recording?.transcriptionState == .done)
        #expect(h.note(gone)?.recording?.transcriptionState == .failed)
        #expect(h.queue.rows.first { $0.noteID == gone }?.phase == .failed(MeetingCopy.fileGone))
        #expect(h.note(done)?.recording?.transcriptionState == .done)
        #expect(await h.queue.recoverInterruptedJobs().isEmpty)
    }
}

/// `FakeSpeechService`, except that `transcribeFile` waits until the test
/// opens the gate: a job that is still being created.
final class GatedFileSpeech: SpeechServicing, @unchecked Sendable {
    private let base: FakeSpeechService
    private let lock = NSLock()
    private var waiter: CheckedContinuation<Void, Never>?
    private var opened = false

    init(_ base: FakeSpeechService) { self.base = base }

    var isWaiting: Bool { lock.withLock { waiter != nil } }

    func open() {
        let waiting = lock.withLock { () -> CheckedContinuation<Void, Never>? in
            opened = true
            defer { waiter = nil }
            return waiter
        }
        waiting?.resume()
    }

    func transcribeFile(_ url: URL, options: FileTranscriptionOptions) async throws -> FileJob {
        await withCheckedContinuation { continuation in
            let ready = lock.withLock { () -> Bool in
                if opened { return true }
                waiter = continuation
                return false
            }
            if ready { continuation.resume() }
        }
        return try await base.transcribeFile(url, options: options)
    }

    var status: SpeechStatus { base.status }
    func statusUpdates() -> AsyncStream<SpeechStatus> { base.statusUpdates() }
    func downloadSize(of package: SpeechPackageID) -> Int64 { base.downloadSize(of: package) }
    func install(_ package: SpeechPackageID) async throws { try await base.install(package) }
    func pauseInstall(_ package: SpeechPackageID) async { await base.pauseInstall(package) }
    func cancelInstall(_ package: SpeechPackageID) async { await base.cancelInstall(package) }
    func remove(_ package: SpeechPackageID) async throws { try await base.remove(package) }
    func startDictation(_ options: DictationOptions) async throws -> DictationSession {
        try await base.startDictation(options)
    }
    func startMeeting(_ options: MeetingOptions) async throws -> MeetingRecording {
        try await base.startMeeting(options)
    }
    func cancel(_ job: FileJob.ID) async { await base.cancel(job) }
    func inputDevices() async -> [AudioInputDevice] { await base.inputDevices() }
    func runningAudioApps() async -> [RunningAudioApp] { await base.runningAudioApps() }
    func recordingsSize() async -> Int64 { await base.recordingsSize() }
    func deleteRecordings(for noteID: UUID) async throws { try await base.deleteRecordings(for: noteID) }
    func recordingsDirectory(for noteID: UUID) -> URL { base.recordingsDirectory(for: noteID) }
    func recoverInterruptedRecordings() async -> [InterruptedRecording] { await base.recoverInterruptedRecordings() }
    func shutdown() async { await base.shutdown() }
}
