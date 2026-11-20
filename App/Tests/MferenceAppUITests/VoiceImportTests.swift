import Foundation
import MferenceAppCore
@testable import MferenceAppUI
import MferenceDesign
import Observation
import SwiftUI
import Testing
import UniformTypeIdentifiers

/// A file-queue double for the import sheet.
@MainActor
@Observable
final class FakeFileQueue: FileQueueModel {
    var rows: [FileQueueRow] = []
    var added: [(urls: [URL], language: SpeechLanguage, speakerLabels: Bool, summary: Bool)] = []
    var cancelled: [UUID] = []

    func add(_ urls: [URL], language: SpeechLanguage, speakerLabels: Bool, summary: Bool) async {
        added.append((urls, language, speakerLabels, summary))
        rows += urls.map { FileQueueRow(id: UUID(), noteID: UUID(), fileName: $0.lastPathComponent) }
    }

    func cancel(_ id: UUID) { cancelled.append(id) }
}

/// "Transcribe audio files" (board 6).
@MainActor
@Suite(.serialized) struct VoiceImportTests {
    private func voice(_ speech: FakeSpeechService) -> VoiceModels {
        VoiceModels(speech: speech, permissions: FakePermissionBroker(),
                    preferences: SpeechPreferences(store: InMemorySettingsStore()), isOffline: { false })
    }

    static let boardRows = [
        FileQueueRow(id: UUID(), noteID: UUID(), fileName: "lecture-week-3.mp3", duration: 4680, languageCode: "es",
                     fraction: 0.64, remaining: 70, phase: .running),
        FileQueueRow(id: UUID(), noteID: UUID(), fileName: "podcast-ep12.m4a", duration: 2820, phase: .queued),
        FileQueueRow(id: UUID(), noteID: UUID(), fileName: "interview-dana.m4a", duration: 2280, languageCode: "en",
                     speakerCount: 2, fraction: 1, phase: .done),
    ]

    @Test func rowTextFollowsTheBoard() {
        let rows = Self.boardRows
        #expect(FileQueueRowView.detail(rows[0]) == "1 h 18 min · Spanish, detected")
        #expect(FileQueueRowView.state(rows[0]) == "64% · about 1 min left", "a real time left, no brackets")
        #expect(FileQueueRowView.detail(rows[1]) == "47 min · waiting")
        #expect(FileQueueRowView.state(rows[1]) == "Queued")
        #expect(FileQueueRowView.detail(rows[2]) == "38 min · 2 speakers · English")
        #expect(FileQueueRowView.state(rows[2]) == "Done")
        #expect(FileQueueRowView.timeLeft(30) == "under a minute left")
        #expect(FileQueueRowView.timeLeft(3900) == "about 1 h 5 min left")
        var paused = rows[1]
        paused.phase = .paused
        #expect(FileQueueRowView.detail(paused) == "47 min · waits while dictation or a meeting runs")
        var failed = rows[1]
        failed.phase = .failed("This file has no audio Mference can read.")
        #expect(FileQueueRowView.state(failed) == "Failed" && FileQueueRowView.stateIsFailure(failed))
    }

    @Test func transcribeFilesGoesThroughSetupThenShowsTheSheet() async {
        let speech = FakeSpeechService.notInstalled()
        let models = voice(speech)
        models.transcribeFiles()
        #expect(models.setup.pendingAction == .transcribeFiles && !models.showsImportSheet)
        models.setup.download()
        await models.setup.waitForInstall()
        #expect(models.showsImportSheet)
    }

    @Test func droppedFilesQueueWithTheRememberedOptions() async throws {
        let models = voice(FakeSpeechService())
        let queue = FakeFileQueue()
        models.fileQueue = queue
        models.preferences.language = .fixed("es")
        models.preferences.fileSummary = false
        models.transcribe([URL(fileURLWithPath: "/tmp/lecture.mp3"), URL(fileURLWithPath: "/tmp/talk.mov")])
        for _ in 0..<100 where queue.added.isEmpty { await Task.yield() }
        let added = try #require(queue.added.first)
        #expect(added.urls.map(\.lastPathComponent) == ["lecture.mp3", "talk.mov"])
        #expect(added.language == .fixed("es") && added.speakerLabels && !added.summary)
        #expect(models.showsImportSheet, "the sheet shows the queue")
        models.transcribe([])
        #expect(queue.added.count == 1, "nothing dropped, nothing queued")
    }

    @Test func acceptedTypesCoverAudioAndVideo() {
        for ext in ["m4a", "mp3", "wav", "aiff", "caf", "flac", "mp4", "mov"] {
            let type = UTType(filenameExtension: ext)
            #expect(type.map { t in TranscribeFilesSheet.acceptedTypes.contains { t.conforms(to: $0) } } == true, "\(ext)")
        }
    }

    @Test(arguments: [ColorScheme.light, .dark])
    func importBoard(colorScheme: ColorScheme) throws {
        let queue = FakeFileQueue()
        queue.rows = Self.boardRows
        let preferences = SpeechPreferences(store: InMemorySettingsStore())
        try Snapshot.render(NotesBoard { Color.clear }
            .overlay { Rectangle().fill(MFColor.ink.opacity(0.1)).padding(.top, 57) }
            .overlay {
                TranscribeFilesSheet(queue: queue, preferences: preferences, onOpenNote: { _ in }, onClose: {})
                    .clipShape(RoundedRectangle(cornerRadius: MFRadius.xl))
                    .overlay(RoundedRectangle(cornerRadius: MFRadius.xl).strokeBorder(MFColor.hairlineStrong))
                    .shadow(color: MFColor.ink.opacity(0.2), radius: 30, y: 20)
                    .padding(.top, 40)
            }, name: "notes-06-import", colorScheme: colorScheme)
    }

    @Test func importSheetUnwired() throws {
        try Snapshot.render(TranscribeFilesSheet(queue: nil, preferences: SpeechPreferences(store: InMemorySettingsStore()),
                                                 onOpenNote: { _ in }, onClose: {}),
                            name: "notes-06-import-unwired", framing: .fill)
    }
}
