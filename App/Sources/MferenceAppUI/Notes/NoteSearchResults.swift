import MferenceAppCore
import MferenceDesign
import SwiftUI

/// ⌘K's Notes section: a row per hit with its kind, title and the line
/// that matched (a transcript line with its time).
struct NoteSearchResults: View {
    let hits: [NoteSearchHit]
    let open: (NoteSearchHit) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.xxs) {
            Text("NOTES")
                .font(MFFont.sectionHeader)
                .tracking(MFFont.sectionHeaderTracking)
                .foregroundStyle(MFColor.tertiary)
                .padding(.horizontal, MFSpacing.s)
                .padding(.top, MFSpacing.s)
                .accessibilityAddTraits(.isHeader)
            ForEach(hits) { hit in
                Button { open(hit) } label: { row(hit) }
                    .buttonStyle(.plain)
                    .accessibilityLabel(accessibilityText(hit))
            }
        }
    }

    private func row(_ hit: NoteSearchHit) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: MFSpacing.s) {
            Image(systemName: icon(hit))
                .frame(width: NotesMetrics.kindIcon)
                .foregroundStyle(MFColor.secondary)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: MFSpacing.xxs) {
                Text(hit.title)
                    .font(MFFont.bodyEmphasis)
                    .foregroundStyle(MFColor.ink)
                    .lineLimit(1)
                if !hit.snippet.isEmpty {
                    HStack(spacing: MFSpacing.xs) {
                        if case .transcript(_, let time) = hit.match {
                            Text(NoteExport.clock(time))
                                .font(MFFont.preview.weight(.medium))
                                .foregroundStyle(MFColor.accent)
                                .monospacedDigit()
                        }
                        Text(hit.snippet)
                            .font(MFFont.preview)
                            .foregroundStyle(MFColor.secondary)
                            .lineLimit(1)
                    }
                }
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, MFSpacing.s)
        .padding(.vertical, MFSpacing.snug)
        .contentShape(Rectangle())
    }

    private func icon(_ hit: NoteSearchHit) -> String {
        switch hit.match {
        case .transcript: "waveform"
        default:
            switch hit.kind {
            case .meeting: "waveform"
            case .file: "doc"
            case .note: "note.text"
            }
        }
    }

    private func accessibilityText(_ hit: NoteSearchHit) -> String {
        if case .transcript(_, let time) = hit.match {
            return "\(hit.title), at \(NoteExport.clock(time)): \(hit.snippet)"
        }
        return hit.snippet.isEmpty ? hit.title : "\(hit.title): \(hit.snippet)"
    }
}
