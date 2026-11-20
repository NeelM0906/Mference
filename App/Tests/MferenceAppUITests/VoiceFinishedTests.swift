import Foundation
import MferenceAppCore
@testable import MferenceAppUI
import MferenceDesign
import SwiftUI
import Testing

/// A playback double: a clock the test moves.
@MainActor
final class FakePlayback: AudioPlayback {
    var loaded: [RecordedTrack] = []
    var length: TimeInterval = 2530
    var failLoad = false
    var playing = false
    var rate: Float = 1
    var currentTime: TimeInterval = 0
    var onTick: (@MainActor () -> Void)?

    func load(tracks: [RecordedTrack], directory: URL) async throws -> TimeInterval {
        if failLoad { throw CocoaError(.fileReadCorruptFile) }
        loaded = tracks
        return length
    }
    func play() { playing = true }
    func pause() { playing = false }
    func seek(to time: TimeInterval) { currentTime = time }

    /// Moves the clock and ticks the model.
    func advance(to time: TimeInterval) {
        currentTime = time
        onTick?()
    }
}

/// The finished meeting (board 5): player, chips, summary header, To-dos
/// button, transcript.
@MainActor
@Suite(.serialized) struct VoiceFinishedTests {
    static let speakers = [SpeakerInfo.you, SpeakerInfo(id: "spk2", name: "Priya", colorIndex: 1),
                           SpeakerInfo(id: "spk3", name: "Sam", colorIndex: 2)]

    static func segment(_ text: String, _ start: TimeInterval, _ speaker: String) -> TranscriptSegment {
        TranscriptSegment(track: speaker == "you" ? .microphone : .call, start: start, end: start + 12,
                          speakerID: speaker, text: text, isFinal: true, languageCode: "en")
    }

    static func boardTranscript() -> TranscriptModel {
        TranscriptModel(segments: [
            segment("For the list column, do we group by day like the mockup, or by folder?", 712, "spk2"),
            segment("By day. Folders live in the sidebar, and the list is always newest first.", 725, "you"),
            segment("And meetings get the waveform mark, so you can spot them in the list.", 739, "spk3"),
            segment("Ship it.", 2400, "you"),
        ], speakers: speakers)
    }

    /// 19 November 2026, 10:02 UTC.
    static var boardDate: Date {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "UTC")!
        return calendar.date(from: DateComponents(year: 2026, month: 11, day: 19, hour: 10, minute: 2))!
    }

    static let tracks = [RecordedTrack(fileName: "mic-1.m4a", kind: .microphone, startOffset: 0, duration: 2530),
                         RecordedTrack(fileName: "call-1.m4a", kind: .call, startOffset: 0, duration: 2530)]

    @Test func playerPlaysFromATranscriptTimeAndHighlightsTheRow() async {
        let playback = FakePlayback()
        let player = RecordingPlayerModel(tracks: Self.tracks, directory: URL(fileURLWithPath: "/tmp/x"),
                                          duration: 2530, playback: playback)
        await player.load()
        #expect(player.hasAudio == true && playback.loaded == Self.tracks)
        let transcript = Self.boardTranscript()
        player.play(from: 725)
        #expect(player.isPlaying && playback.playing && playback.currentTime == 725)
        #expect(player.playingSegment(in: transcript.segments) == transcript.segments[1].id)
        playback.advance(to: 740)
        #expect(player.playingSegment(in: transcript.segments) == transcript.segments[2].id)
        player.togglePlay()
        #expect(!player.isPlaying && !playback.playing)
    }

    @Test func speedCyclesAndTheEndStops() async {
        let playback = FakePlayback()
        let player = RecordingPlayerModel(tracks: Self.tracks, directory: URL(fileURLWithPath: "/tmp/x"),
                                          duration: 100, playback: playback)
        await player.load()
        #expect(player.duration == 2530, "the loaded length wins")
        player.nextSpeed()
        #expect(player.speed == 1.5 && playback.rate == 1.5 && RecordingPlayerModel.speedLabel(1.5) == "1.5×")
        player.nextSpeed()
        player.nextSpeed()
        #expect(player.speed == 1 && RecordingPlayerModel.speedLabel(1) == "1×")
        player.togglePlay()
        playback.advance(to: 2530)
        #expect(!player.isPlaying && player.currentTime == 2530)
        player.togglePlay()
        #expect(player.isPlaying && playback.currentTime == 0, "play at the end starts over")
    }

    @Test func noAudioKeptMeansNoPlayer() async {
        let deleted = RecordingPlayerModel(tracks: Self.tracks, directory: nil, duration: 60, playback: FakePlayback())
        await deleted.load()
        #expect(deleted.hasAudio == false)
        deleted.play(from: 10)
        #expect(!deleted.isPlaying)
        let broken = FakePlayback()
        broken.failLoad = true
        let failed = RecordingPlayerModel(tracks: Self.tracks, directory: URL(fileURLWithPath: "/tmp"), duration: 60,
                                          playback: broken)
        await failed.load()
        #expect(failed.hasAudio == false && failed.loadError == "The recording could not be opened.")
    }

    @Test func scrubberBandsAndChips() {
        let transcript = Self.boardTranscript()
        let bands = RecordingPlayerBar.bands(transcript.finalSegments, speakers: transcript, duration: 2530)
        #expect(bands.map(\.colorIndex) == [1, 0, 2, 0])
        #expect(abs(bands[0].start - 712.0 / 2530) < 0.0001)
        #expect(abs(bands[0].length - 13.0 / 2530) < 0.0001, "a colour runs until the next speaker starts")
        #expect(abs(bands[3].start + bands[3].length - 1) < 0.0001, "the last runs to the end")
        #expect(RecordingPlayerBar.time(at: 50, width: 100, duration: 2530) == 1265)
        #expect(RecordingPlayerBar.fraction(9999, of: 100) == 1)
        #expect(RecordingChips.speakersInOrder(transcript).map(\.name) == ["You", "Priya", "Sam"])
        let date = Self.boardDate
        #expect(RecordingChips.dateText(date, locale: Locale(identifier: "en_GB"), timeZone: TimeZone(identifier: "UTC")!)
                == "19 Nov · 10:02")
        #expect(TranscriptSection.showAllTitle(duration: 2530) == "Show all (42 min)")
        #expect(AddToTodosButton.title(count: 4, added: false) == "Add 4 to To-dos")
        #expect(AddToTodosButton.title(count: 4, added: true) == "Added to To-dos")
    }

    @Test(arguments: [ColorScheme.light, .dark])
    func meetingDoneBoard(colorScheme: ColorScheme) async throws {
        let playback = FakePlayback()
        let player = RecordingPlayerModel(tracks: Self.tracks, directory: URL(fileURLWithPath: "/tmp/x"),
                                          duration: 2530, playback: playback)
        await player.load()
        player.seek(to: 725)
        let transcript = Self.boardTranscript()
        let date = Self.boardDate
        let board = NotesBoard(placeholders: false) {
            VStack(spacing: 0) {
                RecordingPlayerBar(player: player, transcript: transcript)
                VStack(alignment: .leading, spacing: MFSpacing.l) {
                    Text("Design review: Notes").font(MFFont.largeTitle).foregroundStyle(MFColor.ink)
                    RecordingChips(startedAt: date, duration: 2530, transcript: transcript,
                                   locale: Locale(identifier: "en_GB"), timeZone: TimeZone(identifier: "UTC")!)
                    SummaryHeader(state: .written("Written by Quick on this Mac"), onRedo: {}, onRetry: {})
                    VStack(alignment: .leading, spacing: MFSpacing.xs) {
                        Text("•  Notes uses three panes: folders, a list grouped by day, and the editor.")
                        Text("•  Dictation inserts text after each pause; meetings show a live transcript beside your notes.")
                        Text("•  If the lid closes, what was transcribed stays and the gap is marked.")
                    }
                    .font(MFFont.body).foregroundStyle(MFColor.ink)
                    HStack {
                        Text("Action items").font(MFFont.headline).foregroundStyle(MFColor.ink)
                        Spacer()
                        AddToTodosButton(count: 4, added: false) {}
                    }
                    VStack(alignment: .leading, spacing: MFSpacing.s) {
                        ForEach(["Draw the empty state for Meetings · Sam", "Check call-audio permission copy · You",
                                 "Decide default for keeping recordings · Priya",
                                 "Test a 90 minute recording on battery · You"], id: \.self) { item in
                            Label(item, systemImage: "circle").font(MFFont.body).foregroundStyle(MFColor.ink)
                        }
                    }
                    TranscriptSection(transcript: transcript, player: player)
                    Spacer()
                }
                .padding(.horizontal, 56)
                .padding(.top, MFSpacing.xl)
            }
            .background(MFColor.surface)
        }
        try Snapshot.render(board, name: "notes-05-meeting-done", colorScheme: colorScheme)
    }
}
