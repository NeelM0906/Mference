import Foundation
import SwiftData
import Testing
@testable import MferenceAppCore

/// `MeetingSession` and `TranscriptModel` on `FakeSpeechService` meeting
/// scripts and an in-memory store: segment saving, the final pass, renames,
/// pause and marks, call audio denied, and launch recovery. Model-free.
@MainActor
@Suite struct MeetingSessionTests {
    nonisolated static let start = Date(timeIntervalSince1970: 1_795_082_520) // 2026-11-19 10:02 UTC
    nonisolated static let utc = TimeZone(identifier: "UTC")!

    @MainActor struct Harness {
        let container: ModelContainer
        let writer: InMemoryMeetingNoteWriter
        let session: MeetingSession
        let speech: any SpeechServicing

        var context: ModelContext { container.mainContext }

        func recording(_ noteID: UUID) -> NoteRecording? {
            try? context.fetch(FetchDescriptor<Note>(predicate: #Predicate { $0.id == noteID })).first?.recording
        }

        func note(_ noteID: UUID) -> Note? {
            try? context.fetch(FetchDescriptor<Note>(predicate: #Predicate { $0.id == noteID })).first
        }
    }

    func harness(_ speech: any SpeechServicing, endedAt: Date = start.addingTimeInterval(600)) throws -> Harness {
        let container = try AppModelContainer.inMemory()
        let writer = InMemoryMeetingNoteWriter(context: container.mainContext)
        let session = MeetingSession(speech: speech, writer: writer,
                                     now: { endedAt }, locale: Locale(identifier: "en_GB"), timeZone: Self.utc)
        return Harness(container: container, writer: writer, session: session, speech: speech)
    }

    /// Lets the session's consumer task catch up with the fake's events.
    func settle(until condition: () -> Bool, sourceLocation: SourceLocation = #_sourceLocation) async {
        for _ in 0..<500 {
            if condition() { return }
            await Task.yield()
        }
        Issue.record("The condition never became true.", sourceLocation: sourceLocation)
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

    @Test func anInPlaceUpdateThatMovesASegmentReorders() {
        let first = UUID()
        let model = TranscriptModel(segments: [Self.segment("a", 0, 1, final: false, id: first),
                                               Self.segment("b", 5, 6), Self.segment("c", 10, 11)])
        model.upsert(Self.segment("a, later", 7, 8, id: first))
        #expect(model.segments.map(\.text) == ["b", "a, later", "c"])
    }

    @Test func theGapLabelNamesTheStretch() {
        let gap = RecordingGap(start: Self.start.addingTimeInterval(180), end: Self.start.addingTimeInterval(3_480),
                               at: 180, reason: .sleep)
        #expect(MeetingCopy.gapLabel(gap, locale: Locale(identifier: "en_GB"), timeZone: Self.utc)
            == "Gap: 10:05–11:00 (Mac asleep)")
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
        await settle { h.session.transcript?.segments.map(\.track) == [.call] }
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
            == "Meeting 19 Nov at 10:02")
    }

    @Test(arguments: [(true, true, true), (true, false, false), (false, true, false)])
    func audioIsKeptOnlyWhenAskedAndWritten(keep: Bool, written: Bool, kept: Bool) async throws {
        let result = TranscriptResult(segments: [], speakers: [.you],
                                      tracks: [RecordedTrack(fileName: "mic-1.m4a", kind: .microphone,
                                                             startOffset: 0, duration: 5)],
                                      duration: 5, audioDirectory: written ? URL(filePath: "/tmp/rec") : nil)
        let speech = FakeSpeechService(meeting: FakeSpeechService.MeetingScript(result: result))
        let h = try harness(speech)
        let noteID = try await h.session.start(MeetingStartOptions(keepAudio: keep))
        await h.session.stop()
        await h.session.waitUntilFinished()
        #expect(h.recording(noteID)?.audioKept == kept)
        #expect((h.session.audio(for: noteID) != nil) == kept)
        if kept {
            let audio = try #require(h.session.audio(for: noteID))
            #expect(audio.duration == 5)
            #expect(audio.url(of: audio.tracks[0]) == speech.recordingsDirectory(for: noteID).appending(path: "mic-1.m4a"))
        }
    }

    @Test func aSecondStopIsHarmless() async throws {
        let speech = FakeSpeechService()
        let h = try harness(speech)
        var ended = 0
        h.session.afterMeeting = { _, _ in ended += 1 }
        let noteID = try await h.session.start(MeetingStartOptions())
        await h.session.stop()
        await h.session.stop()
        await h.session.waitUntilFinished()
        await h.session.stop()
        #expect(ended == 1)
        #expect(h.session.phase == .idle)
        #expect(h.recording(noteID)?.transcriptionState == .done)
    }

    // MARK: Streams that end badly

    @Test func aStreamEndingWithoutAResultClosesTheNoteAndRecoveryAttachesItsAudio() async throws {
        let speech = ManualMeetingSpeech()
        let h = try harness(speech)
        let noteID = try await h.session.start(MeetingStartOptions())
        speech.send(.started(callAudio: true))
        speech.send(.segment(Self.segment("Before the crash.", 0, 2)))
        speech.send(.failed(.failed("tap died")))
        speech.finishStream()
        await h.session.waitUntilFinished()
        #expect(h.session.phase == .idle)
        #expect(h.session.lastError == SpeechError.failed("").message)
        let record = try #require(h.recording(noteID))
        #expect(record.endedAt == Self.start.addingTimeInterval(600))
        #expect(record.transcriptionState == .done)
        #expect(record.segments.map(\.text) == ["Before the crash."])
        #expect(h.writer.body(of: noteID).paragraphs == ["Recording ended unexpectedly at 10:12"])
        #expect(h.session.audio(for: noteID) == nil)

        // Next launch: the speech service reports the files it closed.
        let track = RecordedTrack(fileName: "mic-1.m4a", kind: .microphone, startOffset: 0, duration: 2)
        speech.interrupted = [InterruptedRecording(noteID: noteID, tracks: [track], endedAt: Self.start)]
        let relaunched = MeetingSession(speech: speech, writer: h.writer, now: { Self.start },
                                        locale: Locale(identifier: "en_GB"), timeZone: Self.utc)
        #expect(await relaunched.recoverInterruptedMeetings().isEmpty)
        #expect(h.recording(noteID)?.tracks == [track])
        #expect(h.recording(noteID)?.endedAt == Self.start.addingTimeInterval(600))
        #expect(relaunched.audio(for: noteID)?.tracks == [track])
        #expect(h.writer.body(of: noteID).paragraphs.count == 1)
    }

    @Test func eventsAfterTheEndNeverReachTheNextMeeting() async throws {
        let speech = ManualMeetingSpeech()
        let h = try harness(speech)
        let first = try await h.session.start(MeetingStartOptions(summaryWhenStopped: false))
        let firstStream = try #require(speech.continuation)
        firstStream.yield(.segment(Self.segment("First meeting.", 0, 1)))
        firstStream.yield(.ended(TranscriptResult(segments: [Self.segment("First meeting.", 0, 1)],
                                                  speakers: [.you], duration: 1)))
        await h.session.waitUntilFinished()
        let second = try await h.session.start(MeetingStartOptions(summaryWhenStopped: false))
        // A stray event on the first meeting's stream after its end.
        firstStream.yield(.segment(Self.segment("Late.", 2, 3)))
        speech.send(.segment(Self.segment("Second meeting.", 0, 1)))
        await settle { h.recording(second)?.segments.count == 1 }
        #expect(h.recording(first)?.segments.map(\.text) == ["First meeting."])
        #expect(h.recording(second)?.segments.map(\.text) == ["Second meeting."])
        #expect(h.session.transcript?.segments.map(\.text) == ["Second meeting."])
    }

    @Test func aFinalResultWithoutSegmentsKeepsTheLiveTranscript() async throws {
        let speech = ManualMeetingSpeech()
        let h = try harness(speech)
        let noteID = try await h.session.start(MeetingStartOptions(summaryWhenStopped: false))
        speech.send(.segment(Self.segment("Kept.", 0, 1)))
        speech.send(.ended(TranscriptResult(segments: [], speakers: [.you], duration: 1)))
        await h.session.waitUntilFinished()
        #expect(h.recording(noteID)?.segments.map(\.text) == ["Kept."])
    }

    @Test func callAudioFailingMidCallIsSaidOnce() async throws {
        let speech = ManualMeetingSpeech()
        let h = try harness(speech)
        let noteID = try await h.session.start(MeetingStartOptions())
        speech.send(.started(callAudio: true))
        speech.send(.callAudioUnavailable(.failed("tap")))
        speech.send(.callAudioUnavailable(.failed("tap")))
        await settle { !h.session.callAudioAvailable && !h.writer.body(of: noteID).paragraphs.isEmpty }
        for _ in 0..<50 { await Task.yield() }
        #expect(h.writer.body(of: noteID).paragraphs
            == ["Call audio stopped at 10:12, so only your microphone was recorded after that."])
    }

    // MARK: Options from Settings

    @Test func startOptionsComeFromTheSharedPreferences() {
        let preferences = SpeechPreferences(store: InMemorySettingsStore())
        preferences.microphoneID = "usb"
        preferences.callAudio = .app(bundleID: "us.zoom.xos", name: "zoom.us")
        preferences.keepAudio = .deleteAfterTranscribing
        preferences.language = .fixed("de")
        var options = MeetingStartOptions(preferences: preferences)
        #expect(options.microphoneID == "usb" && options.keepAudio == false && options.language == .fixed("de"))
        #expect(options.callAudio == .app(bundleID: "us.zoom.xos", name: "zoom.us"))
        #expect(options.summaryWhenStopped)
        preferences.summaryModel = .off
        options = MeetingStartOptions(preferences: preferences)
        #expect(!options.summaryWhenStopped)
        let files = FileImportOptions(preferences: preferences)
        #expect(!files.summarize && !files.keepAudio && files.language == .fixed("de"))
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
        #expect(speech.calls.contains { if case .deleteRecordings = $0 { true } else { false } })
        #expect(try h.context.fetch(FetchDescriptor<Note>()).isEmpty)
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

    @Test func recoveryLeavesTheLiveMeetingAlone() async throws {
        let speech = FakeSpeechService(stepped: true)
        let h = try harness(speech)
        let noteID = try await h.session.start(MeetingStartOptions())
        speech.setInterruptedRecordings([InterruptedRecording(noteID: noteID, tracks: [], endedAt: Self.start)])
        #expect(await h.session.recoverInterruptedMeetings().isEmpty)
        #expect(h.recording(noteID)?.endedAt == nil)
        #expect(h.recording(noteID)?.transcriptionState == .live)
    }

    @Test func recoveryNeverEndsBeforeTheLastGap() async throws {
        let speech = FakeSpeechService()
        let h = try harness(speech)
        let id = UUID()
        try h.writer.createNote(id: id, kind: .meeting, title: "Meeting", createdAt: Self.start)
        let record = NoteRecording(startedAt: Self.start)
        record.transcriptionState = .live
        record.segments = [Self.segment("Before sleep.", 0, 300), Self.segment("After wake.", 300, 420)]
        record.gaps = [RecordingGap(start: Self.start.addingTimeInterval(300),
                                    end: Self.start.addingTimeInterval(3_600), at: 300, reason: .sleep)]
        h.context.insert(record)
        h.note(id)?.recording = record
        try h.context.save()
        #expect(await h.session.recoverInterruptedMeetings() == [id])
        // Recording time says 10:09; the Mac slept until 11:02, so it ended no earlier.
        #expect(h.recording(id)?.endedAt == Self.start.addingTimeInterval(3_600))
        #expect(h.writer.body(of: id).paragraphs == ["Recording ended unexpectedly at 11:02"])
    }
}

/// A speech service whose meeting stream the test drives by hand: events,
/// a stream that ends without `.ended`, and launch reports.
final class ManualMeetingSpeech: SpeechServicing, @unchecked Sendable {
    private let lock = NSLock()
    private var current: AsyncStream<MeetingEvent>.Continuation?
    private var reported: [InterruptedRecording] = []

    var continuation: AsyncStream<MeetingEvent>.Continuation? { lock.withLock { current } }
    var interrupted: [InterruptedRecording] {
        get { lock.withLock { reported } }
        set { lock.withLock { reported = newValue } }
    }

    func send(_ event: MeetingEvent) { continuation?.yield(event) }
    func finishStream() { continuation?.finish() }

    var status: SpeechStatus { SpeechStatus(speech: .ready, speakerLabels: .ready) }
    func statusUpdates() -> AsyncStream<SpeechStatus> {
        let (stream, continuation) = AsyncStream.makeStream(of: SpeechStatus.self)
        continuation.yield(status)
        return stream
    }
    func downloadSize(of package: SpeechPackageID) -> Int64 { 0 }
    func install(_ package: SpeechPackageID) async throws {}
    func pauseInstall(_ package: SpeechPackageID) async {}
    func cancelInstall(_ package: SpeechPackageID) async {}
    func remove(_ package: SpeechPackageID) async throws {}
    func startDictation(_ options: DictationOptions) async throws -> DictationSession {
        throw SpeechError.busy(.meeting)
    }
    func startMeeting(_ options: MeetingOptions) async throws -> MeetingRecording {
        let (events, continuation) = AsyncStream.makeStream(of: MeetingEvent.self)
        lock.withLock { current = continuation }
        return MeetingRecording(noteID: options.noteID, startedAt: MeetingSessionTests.start, events: events,
                                controls: .init(pause: {}, resume: {}, markMoment: { 0 },
                                                stop: { continuation.finish() }))
    }
    func transcribeFile(_ url: URL, options: FileTranscriptionOptions) async throws -> FileJob {
        throw SpeechError.notInstalled(.speech)
    }
    func cancel(_ job: FileJob.ID) async {}
    func inputDevices() async -> [AudioInputDevice] { [] }
    func runningAudioApps() async -> [RunningAudioApp] { [] }
    func recordingsSize() async -> Int64 { 0 }
    func deleteRecordings(for noteID: UUID) async throws {}
    func recordingsDirectory(for noteID: UUID) -> URL {
        URL(filePath: "/tmp/manual-recordings").appending(path: noteID.uuidString)
    }
    func recoverInterruptedRecordings() async -> [InterruptedRecording] { interrupted }
    func shutdown() async {}
}
