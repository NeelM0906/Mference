import Foundation
import SwiftData
import Testing
@testable import MferenceAppCore

/// `MeetingSession` and `TranscriptModel` on `FakeSpeechService` meeting
/// scripts and an in-memory store: segment saving, the final pass, renames,
/// pause and marks, call audio denied, and launch recovery. Model-free.
@MainActor
@Suite struct MeetingSessionTests {
    static let start = Date(timeIntervalSince1970: 1_795_082_520) // 2026-11-19 10:02 UTC
    static let utc = TimeZone(identifier: "UTC")!

    struct Harness {
        let container: ModelContainer
        let writer: InMemoryMeetingNoteWriter
        let session: MeetingSession
        let speech: FakeSpeechService

        var context: ModelContext { container.mainContext }

        func recording(_ noteID: UUID) -> NoteRecording? {
            try? context.fetch(FetchDescriptor<Note>(predicate: #Predicate { $0.id == noteID })).first?.recording
        }

        func note(_ noteID: UUID) -> Note? {
            try? context.fetch(FetchDescriptor<Note>(predicate: #Predicate { $0.id == noteID })).first
        }
    }

    func harness(_ speech: FakeSpeechService, endedAt: Date = start.addingTimeInterval(600)) throws -> Harness {
        let container = try AppModelContainer.inMemory()
        let writer = InMemoryMeetingNoteWriter(context: container.mainContext)
        let session = MeetingSession(speech: speech, context: container.mainContext, writer: writer,
                                     now: { endedAt }, locale: Locale(identifier: "en_GB"), timeZone: Self.utc)
        return Harness(container: container, writer: writer, session: session, speech: speech)
    }

    /// Lets the session's consumer task catch up with the fake's events.
    func settle(until condition: () -> Bool = { false }) async {
        for _ in 0..<200 {
            if condition() { return }
            await Task.yield()
        }
    }

    static func segment(_ text: String, _ start: TimeInterval, _ end: TimeInterval, speaker: String = "you",
                        track: TrackKind = .microphone, final: Bool = true, id: UUID = UUID()) -> TranscriptSegment {
        TranscriptSegment(id: id, track: track, start: start, end: end, speakerID: speaker, text: text, isFinal: final)
    }

    // MARK: Live saving

    @Test func finalSegmentsAreSavedAsTheyArrive() async throws {
        let first = UUID()
        let script = FakeSpeechService.MeetingScript(live: [
            .segment(Self.segment("Hello", 0, 1, final: false, id: first)),
            .segment(Self.segment("Hello there.", 0, 1.5, id: first)),
            .segment(Self.segment("Hi, can you hear me?", 2, 4, speaker: "spk1", track: .call)),
        ], result: TranscriptResult(segments: [], speakers: [.you], duration: 4))
        let speech = FakeSpeechService(meeting: script, stepped: true)
        let h = try harness(speech)
        let noteID = try await h.session.start(MeetingStartOptions())
        #expect(h.session.phase == .recording)
        #expect(h.note(noteID)?.kind == .meeting)
        let startedAt = try #require(h.recording(noteID)?.startedAt)
        #expect(h.note(noteID)?.title == MeetingCopy.defaultTitle(startedAt, locale: Locale(identifier: "en_GB"),
                                                                  timeZone: Self.utc))
        #expect(h.recording(noteID)?.transcriptionState == .live)

        speech.advance() // started
        await settle { h.session.callAudioAvailable }
        #expect(h.session.callAudioAvailable)

        speech.advance() // the phrase in progress: shown, not saved
        await settle { h.session.transcript?.segments.count == 1 }
        #expect(h.session.transcript?.segments.first?.isFinal == false)
        #expect(h.recording(noteID)?.segments.isEmpty == true)

        speech.advance() // the same phrase, final: saved
        await settle { h.recording(noteID)?.segments.count == 1 }
        #expect(h.recording(noteID)?.segments.map(\.text) == ["Hello there."])
        #expect(h.session.transcript?.segments.count == 1)

        speech.advance() // a call segment: saved after the first
        await settle { h.recording(noteID)?.segments.count == 2 }
        #expect(h.recording(noteID)?.segments.map(\.speakerID) == ["you", "spk1"])
        #expect(h.recording(noteID)?.speakers.map(\.id).contains("spk1") == true)
        #expect(h.recording(noteID)?.endedAt == nil)
    }

    @Test func segmentsStayInTimeOrderAcrossTracks() {
        let model = TranscriptModel()
        model.upsert(Self.segment("b", 5, 6))
        model.upsert(Self.segment("a", 2, 3, speaker: "spk1", track: .call))
        model.upsert(Self.segment("c", 7, 8))
        #expect(model.segments.map(\.text) == ["a", "b", "c"])
        #expect(model.speaker(for: model.segments[1]).name == "You")
    }

    @Test func anEchoRemovedAfterSavingIsRemovedFromTheStore() async throws {
        let echo = UUID()
        let script = FakeSpeechService.MeetingScript(live: [
            .segment(Self.segment("Can you hear me", 1, 2, id: echo)),
            .segment(Self.segment("Can you hear me?", 1, 2, speaker: "spk1", track: .call)),
            .segmentRemoved(echo),
        ], result: TranscriptResult(segments: [], speakers: [.you], duration: 2))
        let speech = FakeSpeechService(meeting: script, stepped: true)
        let h = try harness(speech)
        let noteID = try await h.session.start(MeetingStartOptions())
        while speech.advance() {}
        await settle { h.recording(noteID)?.segments.count == 1 }
        #expect(h.recording(noteID)?.segments.map(\.track) == [.call])
        #expect(h.session.transcript?.segments.count == 1)
    }

    // MARK: Ending

    @Test func endingSavesTheFinalTranscriptAndCarriesRenamesByMapping() async throws {
        let mine = UUID(), theirs = UUID()
        let live: [MeetingEvent] = [
            .segment(Self.segment("Morning.", 0, 1, id: mine)),
            .speakersUpdated([.you, SpeakerInfo(id: "live-2", name: "Speaker 2", colorIndex: 2)]),
            .segment(Self.segment("Morning, all.", 1, 3, speaker: "live-2", track: .call, id: theirs)),
        ]
        let result = TranscriptResult(
            segments: [Self.segment("Morning.", 0, 1, id: mine),
                       Self.segment("Morning, all.", 1, 3, speaker: "final-A", track: .call, id: theirs)],
            speakers: [.you, SpeakerInfo(id: "final-A", name: "Speaker 2", colorIndex: 2)],
            tracks: [RecordedTrack(fileName: "mic-1.m4a", kind: .microphone, startOffset: 0, duration: 3),
                     RecordedTrack(fileName: "call-1.m4a", kind: .call, startOffset: 0, duration: 3)],
            duration: 3, languageCode: "en", audioDirectory: URL(filePath: "/tmp/rec"),
            speakerMapping: ["live-2": "final-A"])
        let speech = FakeSpeechService(meeting: FakeSpeechService.MeetingScript(live: live, result: result),
                                       stepped: true)
        let h = try harness(speech)
        var after: (UUID, Bool)?
        h.session.afterMeeting = { after = ($0, $1) }
        let noteID = try await h.session.start(MeetingStartOptions())
        while speech.advance() {}
        await settle { h.session.transcript?.segments.count == 2 }
        h.session.transcript?.rename(speakerID: "live-2", to: "Priya")
        #expect(h.recording(noteID)?.speakers.first { $0.id == "live-2" }?.name == "Priya")

        await h.session.stop()
        await h.session.waitUntilFinished()
        #expect(h.session.phase == .idle)
        #expect(h.session.noteID == nil)
        #expect(h.session.lastEndedNoteID == noteID)
        let record = try #require(h.recording(noteID))
        #expect(record.endedAt == Self.start.addingTimeInterval(600))
        #expect(record.duration == 3)
        #expect(record.tracks.count == 2)
        #expect(record.audioKept)
        #expect(record.languageCode == "en")
        #expect(record.transcriptionState == .done)
        #expect(record.summaryState == .pending)
        #expect(record.segments.map(\.speakerID) == ["you", "final-A"])
        #expect(record.speakers.first { $0.id == "final-A" }?.name == "Priya")
        #expect(after?.0 == noteID && after?.1 == true)
        #expect(h.writer.searchRefreshes[noteID] ?? 0 >= 1)
        // The finished note's transcript is the same model, renames included.
        #expect(h.session.transcript(for: noteID)?.speaker(id: "final-A").name == "Priya")
    }

    @Test func noSummaryWhenSwitchedOff() async throws {
        let speech = FakeSpeechService(stepped: true)
        let h = try harness(speech)
        var after: Bool?
        h.session.afterMeeting = { after = $1 }
        _ = try await h.session.start(MeetingStartOptions(summaryWhenStopped: false))
        await h.session.stop()
        await h.session.waitUntilFinished()
        #expect(h.session.phase == .idle)
        #expect(h.session.lastEndedNoteID.flatMap { h.recording($0) }?.summaryState == RecordingSummaryState.none)
        #expect(after == false)
    }

    @Test func theDefaultTitleIsTimeStamped() {
        #expect(MeetingCopy.defaultTitle(Self.start, locale: Locale(identifier: "en_GB"), timeZone: Self.utc)
            == "Meeting 19 Nov, 10:02")
    }

    @Test func notKeepingAudioIsRecorded() async throws {
        let result = TranscriptResult(segments: [], speakers: [.you],
                                      tracks: [RecordedTrack(fileName: "mic-1.m4a", kind: .microphone,
                                                             startOffset: 0, duration: 5)],
                                      duration: 5, audioDirectory: nil)
        let speech = FakeSpeechService(meeting: FakeSpeechService.MeetingScript(result: result))
        let h = try harness(speech)
        let noteID = try await h.session.start(MeetingStartOptions(keepAudio: false))
        await h.session.stop()
        await h.session.waitUntilFinished()
        #expect(h.recording(noteID)?.audioKept == false)
    }

    // MARK: Renames

    @Test func renameChangesTheSpeakerEverywhere() {
        let model = TranscriptModel(segments: [Self.segment("a", 0, 1, speaker: "spk1", track: .call),
                                               Self.segment("b", 1, 2, speaker: "spk1", track: .call)],
                                    speakers: [SpeakerInfo(id: "spk1", name: "Speaker 2", colorIndex: 2)])
        var saved = 0
        model.onChange = { _ in saved += 1 }
        model.rename(speakerID: "spk1", to: "  Sam ")
        #expect(model.segments.allSatisfy { model.speaker(for: $0).name == "Sam" })
        model.rename(speakerID: "spk1", to: "   ")
        #expect(model.speaker(id: "spk1").name == "Sam")
        #expect(saved == 1)
    }

    @Test func renamesCarryOverTheFinalPassByOverlap() {
        let a = UUID(), b = UUID(), c = UUID()
        let model = TranscriptModel(
            segments: [Self.segment("one", 0, 4, speaker: "L1", track: .call, id: a),
                       Self.segment("two", 4, 5, speaker: "L1", track: .call, id: b),
                       Self.segment("three", 5, 9, speaker: "L2", track: .call, id: c)],
            speakers: [SpeakerInfo(id: "L1", name: "Speaker 2", colorIndex: 2),
                       SpeakerInfo(id: "L2", name: "Speaker 3", colorIndex: 3)])
        model.rename(speakerID: "L1", to: "Priya")
        model.rename(speakerID: "L2", to: "Sam")
        // The final pass split L1's segments; most of its time went to F2. No mapping given.
        model.applyFinal(TranscriptResult(
            segments: [Self.segment("one", 0, 4, speaker: "F2", track: .call, id: a),
                       Self.segment("two", 4, 5, speaker: "F3", track: .call, id: b),
                       Self.segment("three", 5, 9, speaker: "F3", track: .call, id: c)],
            speakers: [SpeakerInfo(id: "F2", name: "Speaker 2", colorIndex: 2),
                       SpeakerInfo(id: "F3", name: "Speaker 3", colorIndex: 3)],
            duration: 9))
        #expect(model.speaker(id: "F2").name == "Priya")
        #expect(model.speaker(id: "F3").name == "Sam")
    }

    @Test func renamesCarryOverByTimeWhenSegmentIDsDiffer() {
        let carried = TranscriptModel.carriedRenames(
            ["L1": "Priya"],
            liveSegments: [Self.segment("x", 10, 20, speaker: "L1", track: .call)],
            finalSegments: [Self.segment("x", 9, 12, speaker: "F1", track: .call),
                            Self.segment("y", 12, 21, speaker: "F2", track: .call)],
            mapping: [:])
        #expect(carried == ["F2": "Priya"])
    }

    // MARK: Pause, marks, gaps, call audio

    @Test func pauseResumeAndMarks() async throws {
        let speech = FakeSpeechService()
        let h = try harness(speech)
        let noteID = try await h.session.start(MeetingStartOptions())
        await h.session.pause()
        await settle { h.session.phase == .paused }
        #expect(h.session.phase == .paused)
        speech.setMeetingTime(754)
        #expect(await h.session.markMoment() == 754)
        await h.session.resume()
        await settle { h.session.phase == .recording }
        #expect(h.session.phase == .recording)
        #expect(h.session.marks == [754])
        #expect(h.writer.body(of: noteID).markers == [754])
        #expect(speech.calls.contains(.pause) && speech.calls.contains(.resume))
    }

    @Test func gapsAreSaved() async throws {
        let gap = RecordingGap(start: Self.start.addingTimeInterval(60), end: Self.start.addingTimeInterval(400),
                               at: 60, reason: .sleep)
        let speech = FakeSpeechService(meeting: FakeSpeechService.MeetingScript(
            live: [.gap(gap)], result: TranscriptResult(segments: [], speakers: [.you], gaps: [gap], duration: 80)),
                                       stepped: true)
        let h = try harness(speech)
        let noteID = try await h.session.start(MeetingStartOptions())
        while speech.advance() {}
        await settle { h.recording(noteID)?.gaps.count == 1 }
        #expect(h.recording(noteID)?.gaps == [gap])
        await h.session.stop()
        await h.session.waitUntilFinished()
        #expect(h.recording(noteID)?.gaps == [gap])
    }

    @Test func deniedCallAudioRecordsTheMicrophoneOnlyAndSaysSo() async throws {
        let speech = FakeSpeechService(meeting: FakeSpeechService.MeetingScript(
            callAudioAvailable: false, result: TranscriptResult(segments: [], speakers: [.you], duration: 0)),
                                       stepped: true)
        let h = try harness(speech)
        let noteID = try await h.session.start(MeetingStartOptions())
        while speech.advance() {}
        await settle { !h.writer.body(of: noteID).paragraphs.isEmpty }
        #expect(!h.session.callAudioAvailable)
        #expect(h.writer.body(of: noteID).paragraphs == [MeetingCopy.micOnly])
    }

    // MARK: Refusals

    @Test func aRefusedStartLeavesNoNote() async throws {
        let speech = FakeSpeechService()
        speech.failNextStart(with: .lowDiskSpace(availableBytes: 10))
        let h = try harness(speech)
        await #expect(throws: SpeechError.lowDiskSpace(availableBytes: 10)) {
            try await h.session.start(MeetingStartOptions())
        }
        #expect(h.session.phase == .idle)
        #expect(h.session.lastError == SpeechError.lowDiskSpace(availableBytes: 10).message)
        #expect(try h.context.fetch(FetchDescriptor<Note>()).isEmpty)
    }

    @Test func aSecondStartIsRefused() async throws {
        let speech = FakeSpeechService(stepped: true)
        let h = try harness(speech)
        _ = try await h.session.start(MeetingStartOptions())
        await #expect(throws: SpeechError.busy(.meeting)) { try await h.session.start(MeetingStartOptions()) }
        #expect(try h.context.fetch(FetchDescriptor<Note>()).count == 1)
    }

    @Test func aFailedNoteStopsTheRecording() async throws {
        let speech = FakeSpeechService()
        let h = try harness(speech)
        h.writer.failNextWrite = CocoaError(.fileWriteUnknown)
        await #expect(throws: CocoaError.self) { try await h.session.start(MeetingStartOptions()) }
        #expect(h.session.phase == .idle)
        #expect(speech.status.liveSession == nil)
        #expect(speech.calls.contains(.stopMeeting))
    }

    // MARK: Launch recovery

    @Test func launchClosesInterruptedMeetingsWithWhatWasSaved() async throws {
        let speech = FakeSpeechService()
        let h = try harness(speech)
        func openMeeting(kind: NoteKind = .meeting, segments: [TranscriptSegment]) throws -> UUID {
            let id = UUID()
            try h.writer.createNote(id: id, kind: kind, title: "Meeting", createdAt: Self.start)
            let record = NoteRecording(startedAt: Self.start)
            record.transcriptionState = kind == .meeting ? .live : .running
            record.segments = segments
            h.context.insert(record)
            h.note(id)?.recording = record
            try h.context.save()
            return id
        }
        let reported = try openMeeting(segments: [Self.segment("Saved.", 0, 30)])
        let silent = try openMeeting(segments: [Self.segment("Also saved.", 0, 120),
                                                Self.segment("in progress", 120, 125, final: false)])
        let file = try openMeeting(kind: .file, segments: [])
        let track = RecordedTrack(fileName: "mic-1.m4a", kind: .microphone, startOffset: 0, duration: 40)
        speech.setInterruptedRecordings([InterruptedRecording(noteID: reported, tracks: [track],
                                                              endedAt: Self.start.addingTimeInterval(40))])

        let closed = await h.session.recoverInterruptedMeetings()
        #expect(Set(closed) == [reported, silent])
        let first = try #require(h.recording(reported))
        #expect(first.endedAt == Self.start.addingTimeInterval(40))
        #expect(first.duration == 40)
        #expect(first.tracks == [track])
        #expect(first.transcriptionState == .done)
        #expect(h.writer.body(of: reported).paragraphs == ["Recording ended unexpectedly at 10:02"])
        let second = try #require(h.recording(silent))
        #expect(second.endedAt == Self.start.addingTimeInterval(120))
        #expect(second.segments.map(\.text) == ["Also saved."])
        #expect(second.audioKept == false)
        #expect(h.writer.body(of: silent).paragraphs == ["Recording ended unexpectedly at 10:04"])
        #expect(h.recording(file)?.endedAt == nil)
        // Once per launch.
        #expect(await h.session.recoverInterruptedMeetings().isEmpty)
        #expect(speech.calls.filter { $0 == .recoverInterruptedRecordings }.count == 1)
    }
}
