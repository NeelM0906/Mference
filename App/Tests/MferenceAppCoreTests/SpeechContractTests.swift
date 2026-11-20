import Foundation
import MferenceAppCore
import Testing

/// The `SpeechServicing` contract as `FakeSpeechService` plays it: the rules
/// every team builds against (one live session, ready before use, scripted
/// phrases and segments). Model-free: no audio, no TCC, no network.
@Suite struct SpeechContractTests {
    @Test func dictationPlaysPhrasesInOrderAndEnds() async throws {
        let speech = FakeSpeechService(dictation: .saying("Hello there.", "Second phrase."))
        let session = try await speech.startDictation(DictationOptions())
        #expect(speech.status.liveSession?.kind == .dictation)
        await session.stop()
        var texts: [String] = []
        var started = 0
        var ended = false
        for await event in session.events {
            switch event {
            case .phraseStarted: started += 1
            case .phraseFinished(_, let text, _): texts.append(text)
            case .ended: ended = true
            default: break
            }
        }
        #expect(started == 2 && texts == ["Hello there.", "Second phrase."] && ended)
        #expect(speech.status.liveSession == nil)
    }

    @Test func aDroppedPhraseIsReported() async throws {
        let speech = FakeSpeechService(dictation: FakeSpeechService.DictationScript(phrases: [(nil, nil)]))
        let session = try await speech.startDictation(DictationOptions())
        await session.stop()
        var dropped = false
        for await event in session.events { if case .phraseDropped = event { dropped = true } }
        #expect(dropped)
    }

    @Test func oneLiveSessionAtATime() async throws {
        let speech = FakeSpeechService(stepped: true)
        let meeting = try await speech.startMeeting(MeetingOptions(noteID: UUID()))
        await #expect(throws: SpeechError.busy(.meeting)) {
            _ = try await speech.startDictation(DictationOptions())
        }
        await meeting.stop()
        _ = try await speech.startDictation(DictationOptions())
    }

    @Test func nothingStartsBeforeTheModelIsReady() async throws {
        let speech = FakeSpeechService.notInstalled()
        await #expect(throws: SpeechError.notInstalled(.speech)) {
            _ = try await speech.startDictation(DictationOptions())
        }
        try await speech.install(.speech)
        #expect(speech.status.speech == .ready)
        _ = try await speech.startDictation(DictationOptions())
    }

    @Test func installFailureIsShownAsNeedsAttention() async {
        let speech = FakeSpeechService.notInstalled()
        speech.failNextInstall(with: .verificationFailed(file: "Encoder.mlmodelc/weights/weight.bin"))
        await #expect(throws: SpeechError.self) { try await speech.install(.speech) }
        #expect(speech.status.speech == .failed(.verificationFailed(file: "Encoder.mlmodelc/weights/weight.bin")))
    }

    @Test func statusUpdatesStartWithTheCurrentStatus() async throws {
        let speech = FakeSpeechService.notInstalled()
        var updates = speech.statusUpdates().makeAsyncIterator()
        #expect(await updates.next()?.speech == .notInstalled)
        try await speech.install(.speech)
        var seen: [SpeechModelState] = []
        while let next = await updates.next() {
            seen.append(next.speech)
            if next.speech == .ready { break }
        }
        #expect(seen.contains(.preparing) && seen.last == .ready)
    }

    @Test func deniedCallAudioKeepsTheMeetingOnTheMicrophone() async throws {
        let speech = FakeSpeechService(meeting: FakeSpeechService.MeetingScript(
            callAudioAvailable: false,
            result: TranscriptResult(segments: [], speakers: [.you], duration: 12)))
        let meeting = try await speech.startMeeting(MeetingOptions(noteID: UUID(), callAudio: .allApps))
        await meeting.stop()
        var events: [MeetingEvent] = []
        for await event in meeting.events { events.append(event) }
        #expect(events.first == .started(callAudio: false))
        #expect(events.contains(.callAudioUnavailable(.permissionDenied(.systemAudio))))
        guard case .ended(let result) = events.last else { Issue.record("no end"); return }
        #expect(result.duration == 12)
    }

    @Test func theFakeSendsTheNoCallAudioHintAfterStart() async throws {
        let script = FakeSpeechService.MeetingScript(noCallAudioHeard: true,
                                                     result: TranscriptResult(segments: [], speakers: [.you], duration: 40))
        let speech = FakeSpeechService(meeting: script)
        let meeting = try await speech.startMeeting(MeetingOptions(noteID: UUID(), callAudio: .allApps))
        await meeting.stop()
        var events: [MeetingEvent] = []
        for await event in meeting.events { events.append(event) }
        #expect(Array(events.prefix(2)) == [.started(callAudio: true), .noCallAudioHeard])
        #expect(events.filter { $0 == .noCallAudioHeard }.count == 1)
        // No call audio, no hint: off, or unavailable.
        for (options, available) in [(MeetingOptions(noteID: UUID(), callAudio: .off), true),
                                     (MeetingOptions(noteID: UUID(), callAudio: .allApps), false)] {
            var unheard = script
            unheard.callAudioAvailable = available
            speech.setMeetingScript(unheard)
            let other = try await speech.startMeeting(options)
            await other.stop()
            var seen: [MeetingEvent] = []
            for await event in other.events { seen.append(event) }
            #expect(!seen.contains(.noCallAudioHeard))
        }
        #expect(MeetingEvent.noCallAudioHeardMessage
                == "No call audio yet. If the other side is talking, check Call audio in System Settings.")
    }

    @Test func theFakeReportsWhetherCallAudioWasHeard() {
        let speech = FakeSpeechService()
        #expect(!speech.hasHeardCallAudio)
        speech.setHasHeardCallAudio(true)
        #expect(speech.hasHeardCallAudio)
    }

    @Test func fileJobsReportProgressAndTheirResult() async throws {
        let speech = FakeSpeechService()
        let segment = TranscriptSegment(track: .file, start: 0, end: 2, speakerID: SpeakerInfo.othersID,
                                        text: "Welcome.", isFinal: true, languageCode: "en")
        speech.setFileResult(.success(TranscriptResult(segments: [segment], speakers: [.others], duration: 2)),
                             forFileNamed: "memo.m4a")
        let job = try await speech.transcribeFile(URL(fileURLWithPath: "/tmp/memo.m4a"),
                                                  options: FileTranscriptionOptions(noteID: UUID()))
        var finished: TranscriptResult?
        for await event in job.events { if case .finished(let result) = event { finished = result } }
        #expect(finished?.segments == [segment])
        #expect(speech.status.fileQueue.first?.phase == .done)
    }

    @Test func transcriptValuesRoundTripAsJSON() throws {
        let segments = [TranscriptSegment(track: .call, start: 1.5, end: 3, speakerID: "spk2", text: "Hi",
                                          isFinal: true, languageCode: "de")]
        let decoded = try JSONDecoder().decode([TranscriptSegment].self, from: JSONEncoder().encode(segments))
        #expect(decoded == segments)
        let speakers = [SpeakerInfo.you, SpeakerInfo(id: "spk2", name: "Ana", colorIndex: 2)]
        #expect(try JSONDecoder().decode([SpeakerInfo].self, from: JSONEncoder().encode(speakers)) == speakers)
        let language = SpeechLanguage.fixed("fr")
        #expect(try JSONDecoder().decode(SpeechLanguage.self, from: JSONEncoder().encode(language)) == language)
    }

    @Test func twentyFiveLanguagesWithNames() {
        #expect(SpeechLanguage.supportedCodes.count == 25)
        #expect(Set(SpeechLanguage.supportedCodes).count == 25)
        #expect(SpeechLanguage.displayName(for: "de", locale: Locale(identifier: "en")) == "German")
    }

    @Test func everyErrorHasAMessage() {
        let errors: [SpeechError] = [.notInstalled(.speech), .notInstalled(.speakerLabels), .busy(.dictation),
                                     .busy(.meeting), .permissionDenied(.microphone), .permissionDenied(.systemAudio),
                                     .lowDiskSpace(availableBytes: 1), .offline, .downloadFailed("x"),
                                     .verificationFailed(file: "x"), .unsupportedFile("x"), .failed("x")]
        for error in errors { #expect(!error.message.isEmpty) }
        #expect(SpeechError.offline.message == "Turn off Offline mode in Settings to download.")
    }

    @Test func theUnwiredServiceNeverStartsAnything() async {
        let speech = UnavailableSpeechService()
        #expect(speech.status.speech == .notInstalled)
        await #expect(throws: SpeechError.notInstalled(.speech)) {
            _ = try await speech.startMeeting(MeetingOptions(noteID: UUID()))
        }
        await #expect(throws: SpeechError.self) { try await speech.install(.speech) }
    }

    @Test func stoppingTwiceOrLateNeverReleasesANewerSession() async throws {
        let speech = FakeSpeechService()
        let first = try await speech.startDictation(DictationOptions())
        await first.stop()
        let meeting = try await speech.startMeeting(MeetingOptions(noteID: UUID()))
        await first.stop()
        #expect(speech.status.liveSession?.kind == .meeting, "a late stop must not free the meeting's microphone")
        await #expect(throws: SpeechError.busy(.meeting)) { _ = try await speech.startDictation(DictationOptions()) }
        await meeting.stop()
        await meeting.stop()
        #expect(speech.status.liveSession == nil)
    }

    @Test func cancelEndsTheJobStream() async throws {
        let speech = FakeSpeechService(stepped: true)
        let job = try await speech.transcribeFile(URL(fileURLWithPath: "/tmp/a.m4a"),
                                                  options: FileTranscriptionOptions(noteID: UUID()))
        await speech.cancel(job.id)
        var events: [FileJobEvent] = []
        for await event in job.events { events.append(event) }
        #expect(events == [.cancelled])
        #expect(!speech.advance(), "a cancelled job plays nothing later")
        #expect(speech.status.fileQueue.isEmpty)
    }

    @Test func shutdownEndsLiveSessionsAndJobs() async throws {
        let speech = FakeSpeechService(stepped: true)
        let meeting = try await speech.startMeeting(MeetingOptions(noteID: UUID()))
        let job = try await speech.transcribeFile(URL(fileURLWithPath: "/tmp/b.m4a"),
                                                  options: FileTranscriptionOptions(noteID: UUID()))
        await speech.shutdown()
        var meetingEnded = false
        for await event in meeting.events { if case .ended = event { meetingEnded = true } }
        var jobEvents: [FileJobEvent] = []
        for await event in job.events { jobEvents.append(event) }
        #expect(meetingEnded && jobEvents == [.cancelled])
        #expect(speech.status.liveSession == nil)
    }

    @Test func filesWaitWhileALiveSessionRuns() async throws {
        let speech = FakeSpeechService()
        let meeting = try await speech.startMeeting(MeetingOptions(noteID: UUID()))
        let job = try await speech.transcribeFile(URL(fileURLWithPath: "/tmp/c.m4a"),
                                                  options: FileTranscriptionOptions(noteID: UUID()))
        var events: [FileJobEvent] = []
        for await event in job.events { events.append(event) }
        #expect(events.contains(.paused))
        await meeting.stop()
    }

    @Test func liveStatusCarriesTheNoteAndPausedTime() async throws {
        let speech = FakeSpeechService()
        let noteID = UUID()
        let meeting = try await speech.startMeeting(MeetingOptions(noteID: noteID))
        let live = try #require(speech.status.liveSession)
        #expect(live.kind == .meeting && live.noteID == noteID && !live.isPaused)
        await meeting.pause()
        let paused = try #require(speech.status.liveSession)
        #expect(paused.isPaused)
        #expect(paused.recordedTime(at: .now.addingTimeInterval(60)) == paused.recordedTime, "paused time stands still")
        let running = LiveSessionStatus(kind: .meeting, noteID: noteID, startedAt: .now, recordedTime: 10,
                                        updatedAt: Date(timeIntervalSince1970: 0))
        #expect(running.recordedTime(at: Date(timeIntervalSince1970: 5)) == 15)
        await meeting.stop()
    }

    @Test func interruptedRecordingsAreReportedOnce() async {
        let speech = FakeSpeechService()
        let lost = InterruptedRecording(noteID: UUID(), tracks: [], endedAt: .now)
        speech.setInterruptedRecordings([lost])
        #expect(await speech.recoverInterruptedRecordings() == [lost])
        #expect(await speech.recoverInterruptedRecordings().isEmpty)
    }
}
