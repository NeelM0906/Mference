import Foundation
import MferenceAppCore
@testable import MferenceAppUI
import MferenceEngine
import Testing

/// Quit with the real services (N4's MeetingSession, MeetingSummaries and
/// FileTranscriptionQueue) over fakes: fast, saved, nothing loads Quick.
@MainActor
@Suite(.serialized) struct VoiceQuitTests {
    /// The stored note, or nil.
    func stored(_ services: AppServices, _ id: UUID) -> Note? { (try? services.notes.note(id: id)) ?? nil }

    static let script = FakeSpeechService.MeetingScript(live: [
        .segment(TranscriptSegment(track: .microphone, start: 1, end: 3, speakerID: SpeakerInfo.youID,
                                   text: "Let's keep audio by default.", isFinal: true, languageCode: "en")),
    ], result: TranscriptResult(segments: [], speakers: [.you], duration: 600))

    @Test func quitWithAMeetingStartsNoSummaryAndLeavesItForTheNextLaunch() async throws {
        let speech = FakeSpeechService(meeting: Self.script)
        let journal = FakeEngineJournal()
        let services = AppServices.preview(journal: journal, speech: speech)
        let voice = VoiceModels.live(services: services)
        defer { voice.setup.stop() }
        voice.connect(meetings: services.meetings)
        #expect(await voice.recordMeetingModel.startRecording())
        let noteID = try #require(services.meetings.noteID)
        await waitUntil("the live segment") { services.meetings.transcript?.segments.isEmpty == false }

        await voice.shutDownForQuit()

        #expect(!services.meetings.isActive, "the meeting ended and was saved")
        #expect(speech.calls.contains(.shutdown) && !speech.calls.contains(.stopMeeting),
                "quit used the quick shutdown, never a normal stop")
        try? await Task.sleep(for: .milliseconds(200)) // anything queued would have started by now
        #expect(journal.entries.isEmpty, "nothing loads Quick or generates during quit")
        #expect(services.summaries.status(for: noteID) != .writing(nil) && services.summaries.status(for: noteID) != .queued)
        let record = try #require(stored(services, noteID)?.recording)
        #expect(record.endedAt != nil, "the recording is closed with what was saved")
    }

    @Test func quitKeepsQueuedFileNotes() async throws {
        let speech = FakeSpeechService(stepped: true)
        let services = AppServices.preview(speech: speech)
        let voice = VoiceModels.live(services: services)
        defer { voice.setup.stop() }
        voice.connect(files: services.files)
        let file = FileManager.default.temporaryDirectory.appending(path: "quit-\(UUID().uuidString).m4a")
        FileManager.default.createFile(atPath: file.path, contents: Data([0]))
        defer { try? FileManager.default.removeItem(at: file) }
        let ids = await services.files.add([file], options: FileImportOptions())
        let noteID = try #require(ids.first)
        await voice.shutDownForQuit()
        #expect(stored(services, noteID) != nil, "a queued file's note survives quit (re-queued at launch)")
    }

    @Test func theMenuBarsStopRunsTheFullFinish() async throws {
        let speech = FakeSpeechService(meeting: Self.script)
        let services = AppServices.preview(speech: speech)
        let voice = VoiceModels.live(services: services)
        defer { voice.setup.stop() }
        voice.connect(meetings: services.meetings)
        #expect(await voice.recordMeetingModel.startRecording())
        let noteID = try #require(services.meetings.noteID)
        await voice.recordingControls?.stopRecording()
        await services.meetings.waitUntilFinished()
        #expect(speech.calls.contains(.stopMeeting), "a normal stop")
        #expect(speech.calls.contains(.shutdown) == false)
        await waitUntil("the summary to be asked for") {
            stored(services, noteID)?.recording?.summaryState != RecordingSummaryState.none
        }
    }
}
