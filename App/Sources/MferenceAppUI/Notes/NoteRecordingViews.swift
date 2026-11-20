import MferenceAppCore
import MferenceDesign
import SwiftUI

/// The finished meeting or transcribed file around its note (board 5):
/// the player, the date / length / speaker chips, one slim line for the
/// summary ("Summary written by Quick on this Mac · Redo", its progress, or
/// its failure with Retry) with Add to To-dos above the note, and the
/// transcript below it. The note's own Summary heading is the only
/// "Summary" label. N3's views over N4's MeetingSession and MeetingSummaries.
struct NoteRecordingHeader: View {
    let model: NotesModel
    let note: Note
    let transcript: TranscriptModel
    let player: RecordingPlayerModel

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Only once the audio is known to be there (no flash while it loads).
            if player.hasAudio == true {
                RecordingPlayerBar(player: player, transcript: transcript)
                Divider().overlay(MFColor.chromeDivider)
            }
            VStack(alignment: .leading, spacing: MFSpacing.s) {
                if let recording = note.recording {
                    RecordingChips(startedAt: recording.startedAt, duration: recording.duration, transcript: transcript,
                                   locale: model.locale, timeZone: model.calendar.timeZone)
                }
                if let summaries = model.summaries {
                    summaryLine(summaries)
                }
            }
            .padding(.horizontal, NoteEditorMetrics.horizontalInset)
            .padding(.top, MFSpacing.m)
        }
    }

    @ViewBuilder
    private func summaryLine(_ summaries: MeetingSummaries) -> some View {
        let actions = model.showsSummaryActions(for: note)
        let state = Self.summaryState(summaries.status(for: note.id))
        let count = summaries.uncheckedActionItemCount(for: note.id)
        let added = summaries.todosAdded(for: note.id)
        let showsTodos = actions && (count > 0 || added)
        if state != nil || showsTodos {
            HStack(alignment: .firstTextBaseline, spacing: MFSpacing.compact) {
                switch state {
                case .written(let label):
                    Text(Self.writtenLine(label))
                        .font(MFFont.callout)
                        .foregroundStyle(MFColor.tertiary)
                    if actions {
                        Text("·").font(MFFont.callout).foregroundStyle(MFColor.tertiary).accessibilityHidden(true)
                        Button("Redo") { summaries.requestRedo(note.id) }
                            .buttonStyle(.plain)
                            .font(MFFont.calloutEmphasis)
                            .foregroundStyle(MFColor.secondary)
                            .help("Write the summary and action items again")
                    }
                case .working(let label):
                    ProgressView().controlSize(.small)
                    Text(label).font(MFFont.callout).foregroundStyle(MFColor.secondary)
                case .failed(let message):
                    Text(message).font(MFFont.callout).foregroundStyle(MFColor.danger)
                    if actions {
                        Button("Retry") { summaries.retry(note.id) }
                            .buttonStyle(.mfSecondary)
                    }
                case nil:
                    EmptyView()
                }
                Spacer(minLength: MFSpacing.s)
                if showsTodos {
                    AddToTodosButton(count: count, added: added) { model.addActionItemsToTodos(for: note) }
                }
            }
        }
    }

    /// What the summary line says; nil when no summary was asked for.
    static func summaryState(_ status: MeetingSummaries.Status) -> SummaryHeader.State? {
        switch status {
        case .none: nil
        case .written: .written(MeetingSummaries.writtenLabel)
        case .waitingForChat: .working(MeetingSummaries.waitingLabel)
        case .queued, .writing: .working("Writing the summary on this Mac…")
        case .failed(let message): .failed(message)
        }
    }

    /// "Written by Quick on this Mac" → "Summary written by Quick on this Mac".
    static func writtenLine(_ label: String) -> String {
        guard let first = label.first else { return "Summary" }
        return "Summary " + first.lowercased() + label.dropFirst()
    }
}

/// The transcript under the note (board 5), with its own scroll.
struct NoteRecordingFooter: View {
    let transcript: TranscriptModel
    let player: RecordingPlayerModel

    /// The lines' own height, measured: the footer is as tall as its lines
    /// up to the cap, then it scrolls ("Show all").
    @State private var contentHeight: CGFloat = 0

    var body: some View {
        ScrollView {
            TranscriptSection(transcript: transcript, player: player)
                .padding(.horizontal, NoteEditorMetrics.horizontalInset)
                .padding(.vertical, MFSpacing.m)
                .onGeometryChange(for: CGFloat.self, of: \.size.height) { contentHeight = $0 }
        }
        .scrollBounceBehavior(.basedOnSize)
        .frame(height: min(contentHeight, NotesMetrics.transcriptMaxHeight))
        .background(MFColor.surface)
    }
}
