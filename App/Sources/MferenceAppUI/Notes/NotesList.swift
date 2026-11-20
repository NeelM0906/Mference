import MferenceAppCore
import MferenceDesign
import SwiftUI

/// The search field and the grouped note list (Pinned, Today, Yesterday,
/// Previous 7 days, Previous 30 days, months), with the empty states.
struct NotesList: View {
    @Bindable var model: NotesModel

    var body: some View {
        VStack(spacing: 0) {
            searchField
            let sections = model.sections
            if sections.isEmpty {
                emptyState
            } else {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 0) {
                        if model.scope == .recentlyDeleted { trashHeader }
                        ForEach(sections) { section in
                            if !section.title.isEmpty { header(section.title) }
                            ForEach(section.items) { note in
                                NoteRow(model: model, note: note, isLast: note.id == sections.last?.items.last?.id)
                            }
                        }
                    }
                    .padding(.bottom, MFSpacing.m)
                }
                .scrollIndicators(.automatic)
                // Keyboard: ↑ / ↓ move the selection, ⌫ deletes (G-A7).
                .focusable()
                .focusEffectDisabled()
                .onMoveCommand { direction in
                    switch direction {
                    case .up: model.selectAdjacent(-1)
                    case .down: model.selectAdjacent(1)
                    default: break
                    }
                }
                .onDeleteCommand { model.deleteSelected() }
                .accessibilityHint("Use the up and down arrows to choose a note")
            }
        }
        .background(MFColor.surface)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Notes")
    }

    private var searchField: some View {
        HStack(spacing: MFSpacing.s) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: NotesMetrics.searchIcon))
                .foregroundStyle(MFColor.tertiary)
                .accessibilityHidden(true)
            TextField("Search notes and transcripts", text: $model.query)
                .onSubmit { model.commitQuery() }
                .textFieldStyle(.plain)
                .font(MFFont.callout)
                .foregroundStyle(MFColor.ink)
                .accessibilityLabel("Search notes and transcripts")
            if !model.query.isEmpty {
                Button {
                    model.query = ""
                    model.commitQuery()
                } label: {
                    Image(systemName: "xmark.circle.fill").foregroundStyle(MFColor.tertiary)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Clear search")
            }
        }
        .padding(.horizontal, MFSpacing.s + MFSpacing.xxs)
        .frame(height: NotesMetrics.searchHeight)
        .background(MFColor.fill, in: RoundedRectangle(cornerRadius: MFRadius.m))
        .padding(.top, MFSpacing.m)
        .padding(.horizontal, MFSpacing.m)
        .padding(.bottom, MFSpacing.s - MFSpacing.xxs)
    }

    private func header(_ title: String) -> some View {
        Text(title.uppercased())
            .font(MFFont.sectionHeader)
            .tracking(MFFont.sectionHeaderTracking)
            .foregroundStyle(MFColor.tertiary)
            .padding(.horizontal, MFSpacing.xl - MFSpacing.xxs)
            .padding(.top, MFSpacing.m)
            .padding(.bottom, MFSpacing.xs)
            .accessibilityAddTraits(.isHeader)
    }

    private var trashHeader: some View {
        HStack {
            Text("Notes stay here for 30 days, then go for good with their audio.")
                .font(MFFont.caption)
                .foregroundStyle(MFColor.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: MFSpacing.s)
            Button("Empty…") { model.requestEmptyTrash() }
                .buttonStyle(.mfSecondary)
                .accessibilityLabel("Empty Recently deleted")
        }
        .padding(.horizontal, MFSpacing.l)
        .padding(.vertical, MFSpacing.s)
    }

    @ViewBuilder
    private var emptyState: some View {
        if model.isSearching {
            NotesEmptyState(systemImage: "magnifyingglass", title: "No results",
                            message: "Nothing matches “\(model.appliedQuery.trimmingCharacters(in: .whitespaces))”. Search looks at titles, text and transcripts.")
        } else {
            switch model.scope {
            case .all:
                NotesEmptyState(systemImage: "note.text", title: "No notes yet",
                                message: "Start a note, dictate one, or transcribe a recording.") {
                    NotesStartActions(model: model)
                }
            case .meetings:
                NotesEmptyState(systemImage: "waveform", title: "No meetings yet",
                                message: "Record a meeting to get a live transcript beside your notes.") {
                    Button("Record meeting") { model.voice.recordMeeting?() }
                        .buttonStyle(.mfSecondary)
                        .disabled(model.voice.recordMeeting == nil)
                }
            case .audioTranscripts:
                NotesEmptyState(systemImage: "text.document", title: "No audio transcripts yet",
                                message: "Transcribe an audio or video file; it becomes a note with its audio attached.") {
                    Button("Transcribe file") { model.voice.transcribeFile?() }
                        .buttonStyle(.mfSecondary)
                        .disabled(model.voice.transcribeFile == nil)
                }
            case .folder:
                NotesEmptyState(systemImage: "folder", title: "This folder is empty",
                                message: "Drag notes here, or start a new one.") {
                    Button("New note") { model.newNote() }.buttonStyle(.mfSecondary)
                }
            case .recentlyDeleted:
                NotesEmptyState(systemImage: "trash", title: "Nothing in Recently deleted",
                                message: "Deleted notes stay here for 30 days.")
            }
        }
    }
}

/// New note, Dictate and Transcribe file, for the "no notes yet" state.
struct NotesStartActions: View {
    let model: NotesModel

    var body: some View {
        VStack(spacing: MFSpacing.s) {
            Button("New note") { model.newNote() }
                .buttonStyle(.mfPrimary)
            HStack(spacing: MFSpacing.s) {
                Button("Dictate") {
                    model.newNote()
                    model.voice.toggleDictation?(model.editor)
                }
                .buttonStyle(.mfSecondary)
                .disabled(model.voice.toggleDictation == nil)
                Button("Transcribe file") { model.voice.transcribeFile?() }
                    .buttonStyle(.mfSecondary)
                    .disabled(model.voice.transcribeFile == nil)
            }
        }
        .padding(.top, MFSpacing.xs)
    }
}

/// One note in the list: kind mark, title, time and preview.
struct NoteRow: View {
    let model: NotesModel
    let note: Note
    let isLast: Bool
    @State private var isHovered = false

    private var isSelected: Bool { model.selectedNoteID == note.id }

    var body: some View {
        Button { model.select(note) } label: {
            VStack(alignment: .leading, spacing: NotesMetrics.rowLineGap) {
                HStack(spacing: MFSpacing.s - MFSpacing.xxs) {
                    if let mark = kindMark {
                        Image(systemName: mark)
                            .font(.system(size: NotesMetrics.kindGlyph, weight: .medium))
                            .foregroundStyle(MFColor.secondary)
                            .frame(width: NotesMetrics.kindIcon)
                            .accessibilityHidden(true)
                    }
                    Text(model.title(of: note))
                        .font(MFFont.rowTitle)
                        .foregroundStyle(MFColor.ink)
                        .lineLimit(1)
                }
                HStack(spacing: NotesMetrics.timeGap) {
                    Text(model.timeLabel(of: note))
                        .font(MFFont.preview.weight(.medium))
                        .foregroundStyle(MFColor.secondary)
                    Text(model.preview(of: note))
                        .font(MFFont.preview)
                        .foregroundStyle(MFColor.tertiary)
                        .lineLimit(1)
                        .truncationMode(.tail)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.vertical, NotesMetrics.rowVerticalPadding)
            .padding(.horizontal, MFSpacing.m + MFSpacing.xxs)
            .background(isSelected ? MFColor.fill : (isHovered ? MFColor.fillSubtle : .clear),
                        in: RoundedRectangle(cornerRadius: MFRadius.m))
            .overlay(alignment: .bottom) {
                // Every row has a hairline below it except the selected one
                // and the very last (as on the board).
                if !isSelected, !isLast {
                    Rectangle().fill(MFColor.hairlineFaint).frame(height: MFLayout.hairline)
                        .padding(.horizontal, MFSpacing.xxs)
                }
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .padding(.horizontal, MFSpacing.s)
        .onHover { isHovered = $0 }
        .draggable(NotesDragItem.note(note.id))
        .contextMenu { menu }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(accessibilityText)
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }

    private var kindMark: String? {
        switch note.kind {
        case .meeting: "waveform"
        case .file: "doc"
        case .note: note.hasDictation ? "mic" : nil
        }
    }

    private var accessibilityText: String {
        var parts = [model.title(of: note)]
        switch note.kind {
        case .meeting: parts.append("meeting")
        case .file: parts.append("audio transcript")
        case .note: if note.hasDictation { parts.append("dictated") }
        }
        if note.isPinned { parts.append("pinned") }
        parts.append(model.timeLabel(of: note))
        parts.append(model.preview(of: note))
        return parts.joined(separator: ", ")
    }

    @ViewBuilder
    private var menu: some View {
        if note.deletedAt != nil {
            Button("Restore") { model.restore(note) }
            Button("Delete Forever…", role: .destructive) { model.requestDeleteForever(note) }
        } else {
            Button(note.isPinned ? "Unpin" : "Pin") { model.togglePin(note) }
            Menu("Move to") {
                Button("No folder") { model.move(note, to: nil) }
                    .disabled(note.folder == nil)
                ForEach(model.folders) { folder in
                    Button(folder.name) { model.move(note, to: folder) }
                        .disabled(note.folder?.id == folder.id)
                }
            }
            Button("Duplicate") { model.duplicate(note) }
            Menu("Export") {
                ForEach(NoteExportFormat.allCases, id: \.self) { format in
                    Button(format.label) { model.export(note, as: format) }
                }
            }
            Divider()
            Button("Delete", role: .destructive) { model.delete(note) }
        }
    }
}
