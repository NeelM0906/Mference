import MferenceAppCore
import MferenceDesign
import SwiftUI

/// The ⌘K panel: finds and opens sections (with their shortcuts), and
/// notes (title, text and transcript lines; Notes spec 6.6).
struct ShellSearchPanel: View {
    @Bindable var model: ShellModel
    @Environment(ScreenModels.self) private var screens: ScreenModels?
    @State private var query = ""
    /// Notes results, computed once per pause in typing (not per view pass).
    @State private var noteHits: [NoteSearchHit] = []
    @FocusState private var focused: Bool

    private var matches: [AppSection] {
        let all = AppSection.pills + [.settings]
        let trimmed = query.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return all }
        return all.filter { $0.title.localizedCaseInsensitiveContains(trimmed) }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.s) {
            TextField("Go to…", text: $query)
                .textFieldStyle(.plain)
                .font(MFFont.body)
                .focused($focused)
                .onSubmit { if let first = matches.first { model.show(first) } }
                .padding(.horizontal, MFSpacing.m)
                .padding(.vertical, MFSpacing.s)
                .background(RoundedRectangle(cornerRadius: MFRadius.m).fill(MFColor.fill))
                .accessibilityLabel("Search sections")
            if matches.isEmpty, noteHits.isEmpty {
                Text("No section matches “\(query)”.")
                    .font(MFFont.callout)
                    .foregroundStyle(MFColor.secondary)
                    .padding(MFSpacing.s)
            }
            ForEach(matches) { section in
                Button {
                    model.show(section)
                } label: {
                    HStack(spacing: MFSpacing.s) {
                        Image(systemName: section.systemImage)
                            .frame(width: ShellMetrics.searchRowIconWidth)
                            .accessibilityHidden(true)
                        Text(section.title)
                        if section.isComingLater {
                            MFBadge(.later)
                        }
                        Spacer()
                        MFKeycap("⌘\(String(ShellModel.shortcut(for: section).character))")
                    }
                    .font(MFFont.body)
                    .foregroundStyle(MFColor.ink)
                    .padding(.horizontal, MFSpacing.s)
                    .padding(.vertical, MFSpacing.snug)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
            if !noteHits.isEmpty, let notes = screens?.notes {
                NoteSearchResults(hits: noteHits) { hit in
                    model.show(.notes)
                    notes.open(hit)
                }
            }
        }
        .padding(MFSpacing.m)
        .frame(width: ShellMetrics.searchPanelWidth)
        .onAppear { focused = true }
        // Notes: titles first, then text, then transcript lines (Notes spec 6.6).
        .task(id: query) {
            guard let notes = screens?.notes, !NoteSearch.terms(query).isEmpty else {
                noteHits = []
                return
            }
            try? await Task.sleep(for: NotesModel.searchPause)
            guard !Task.isCancelled else { return }
            noteHits = notes.searchHits(query)
        }
    }
}
