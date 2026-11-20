import MferenceAppCore
import MferenceDesign
import SwiftUI

/// The finished meeting or transcribed file around its note (board 5):
/// the player, the date / length / speaker chips, the summary's state with
/// Redo or Retry, Add to To-dos above the note, and the transcript below
/// it. N3's views over N4's MeetingSession and MeetingSummaries.
struct NoteRecordingHeader: View {
    let model: NotesModel
    let note: Note
    let transcript: TranscriptModel
    let player: RecordingPlayerModel

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            if player.hasAudio != false {
                RecordingPlayerBar(player: player, transcript: transcript)
            }
            VStack(alignment: .leading, spacing: MFSpacing.s) {
                if let recording = note.recording {
                    RecordingChips(startedAt: recording.startedAt, duration: recording.duration, transcript: transcript,
                                   locale: model.locale, timeZone: model.calendar.timeZone)
                }
                if let summaries = model.summaries {
                    HStack(alignment: .firstTextBaseline) {
                        if let state = Self.summaryState(summaries.status(for: note.id)) {
                            SummaryHeader(state: state,
                                          onRedo: { summaries.requestRedo(note.id) },
                                          onRetry: { summaries.retry(note.id) })
                        }
                        Spacer(minLength: MFSpacing.s)
                        let count = summaries.uncheckedActionItemCount(for: note.id)
                        let added = summaries.todosAdded(for: note.id)
                        if count > 0 || added {
                            AddToTodosButton(count: count, added: added) {
                                _ = summaries.addActionItemsToTodos(for: note.id)
                            }
                        }
                    }
                }
            }
            .padding(.horizontal, NoteEditorMetrics.horizontalInset)
            .padding(.top, MFSpacing.m)
        }
    }

    /// What the summary heading says; nil when no summary was asked for.
    static func summaryState(_ status: MeetingSummaries.Status) -> SummaryHeader.State? {
        switch status {
        case .none: nil
        case .written: .written(MeetingSummaries.writtenLabel)
        case .waitingForChat: .working(MeetingSummaries.waitingLabel)
        case .queued, .writing: .working("Writing the summary on this Mac…")
        case .failed(let message): .failed(message)
        }
    }
}

/// The transcript under the note (board 5), with its own scroll.
struct NoteRecordingFooter: View {
    let transcript: TranscriptModel
    let player: RecordingPlayerModel

    var body: some View {
        ScrollView {
            TranscriptSection(transcript: transcript, player: player)
                .padding(.horizontal, NoteEditorMetrics.horizontalInset)
                .padding(.vertical, MFSpacing.m)
        }
        .frame(maxHeight: NotesMetrics.transcriptMaxHeight)
        .background(MFColor.surface)
    }
}
