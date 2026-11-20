import Foundation
import SwiftData
import Testing
@testable import MferenceAppCore

/// The meeting owners as `AppServices` holds them: one of each, sharing the
/// store, the summary hooked to the end of a meeting, and launch recovery
/// run once. Model-free (`AppServices.preview`).
@MainActor
@Suite struct MeetingsWiringTests {
    @Test func appServicesHoldsTheMeetingOwners() async throws {
        let speech = FakeSpeechService()
        let services = AppServices.preview(speech: speech)
        let noteID = try await services.meetings.start(MeetingStartOptions(summaryWhenStopped: true))
        #expect(services.speech.status.liveSession?.noteID == noteID)
        await services.meetings.stop()
        await services.meetings.waitUntilFinished()
        // The meeting's note is in the app's store, and its summary was asked for.
        let notes = try services.modelContainer.mainContext.fetch(FetchDescriptor<Note>())
        #expect(notes.map(\.id) == [noteID])
        #expect(services.summaries.status(for: noteID) != .none)
        await services.summaries.waitUntilIdle()
    }

    @Test func aMeetingEndedAtQuitIsSummarisedAtTheNextLaunch() async throws {
        let speech = FakeSpeechService()
        let segment = TranscriptSegment(track: .microphone, start: 0, end: 2, speakerID: SpeakerInfo.youID,
                                        text: "Ship it on Friday.", isFinal: true)
        speech.setMeetingScript(FakeSpeechService.MeetingScript(
            live: [.segment(segment)], result: TranscriptResult(segments: [segment], speakers: [.you], duration: 2)))
        let services = AppServices.preview(speech: speech)
        let noteID = try await services.meetings.start(MeetingStartOptions(summaryWhenStopped: true))
        await services.shutDownVoice()
        #expect(services.summaries.status(for: noteID) != .queued)
        #expect(try services.notes.note(id: noteID)?.recording?.summaryState == .pending)
        // Next launch (same store): recovery, then the pending summary is queued.
        await services.startLaunchRecovery().value
        #expect(services.summaries.status(for: noteID) != .none)
        await services.summaries.waitUntilIdle()
    }

    @Test func aTranscriptShownWhileTheFileWasQueuedFillsInWhenItFinishes() async throws {
        let speech = FakeSpeechService(stepped: true)
        let segment = TranscriptSegment(track: .file, start: 0, end: 4, speakerID: SpeakerInfo.othersID,
                                        text: "Welcome to week three.", isFinal: true)
        speech.setFileResult(.success(TranscriptResult(segments: [segment], speakers: [.others], duration: 4)),
                             forFileNamed: "lecture.m4a")
        let services = AppServices.preview(speech: speech)
        let id = try #require(await services.files.add([URL(filePath: "/tmp/lecture.m4a")],
                                                       options: FileImportOptions()).first)
        // The note is opened while the file is still queued: an empty transcript.
        let shown = try #require(services.meetings.transcript(for: id))
        #expect(shown.segments.isEmpty)
        while speech.advance() {}
        await services.files.waitUntilIdle()
        // The same model, now with the transcript.
        #expect(services.meetings.transcript(for: id) === shown)
        #expect(shown.segments.map(\.text) == ["Welcome to week three."])
        #expect(shown.speaker(for: shown.segments[0]).name == "Others")
    }

    @Test func launchRecoveryRunsOnce() async throws {
        let speech = FakeSpeechService()
        let services = AppServices.preview(speech: speech)
        let first = services.startLaunchRecovery()
        let second = services.startLaunchRecovery()
        #expect(first == second)
        await first.value
        #expect(speech.calls.filter { $0 == .recoverInterruptedRecordings }.count == 1)
        #expect(services.launchRecovery == first)
    }
}
