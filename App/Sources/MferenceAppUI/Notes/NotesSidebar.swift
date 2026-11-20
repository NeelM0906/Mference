import MferenceAppCore
import MferenceDesign
import SwiftUI
import UniformTypeIdentifiers

/// What a drag in Notes carries: a note (onto a folder) or a folder (to
/// reorder). Plain text, so nothing outside the app can misread it.
enum NotesDragItem {
    static let notePrefix = "mference-note:"
    static let folderPrefix = "mference-folder:"

    static func note(_ id: UUID) -> String { notePrefix + id.uuidString }
    static func folder(_ id: UUID) -> String { folderPrefix + id.uuidString }

    static func ids(_ items: [String], prefix: String) -> [UUID] {
        items.compactMap { $0.hasPrefix(prefix) ? UUID(uuidString: String($0.dropFirst(prefix.count))) : nil }
    }
}

/// Library rows, folders, Recently deleted and the speech status card.
struct NotesSidebar: View {
    @Bindable var model: NotesModel

    var body: some View {
        let counts = model.counts
        VStack(alignment: .leading, spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: MFSpacing.xxs) {
                    header("Library")
                    row("All notes", systemImage: "doc", scope: .all, count: counts.all)
                    row("Meetings", systemImage: "waveform", scope: .meetings, count: counts.meetings)
                    row("Audio transcripts", systemImage: "text.document", scope: .audioTranscripts,
                        count: counts.audioTranscripts)
                    header("Folders")
                    ForEach(model.folders) { folder in
                        folderRow(folder, count: counts.folders[folder.id] ?? 0)
                    }
                    Button { model.newFolder() } label: {
                        rowLabel("New folder", systemImage: "plus", count: nil, isSelected: false, tint: MFColor.tertiary)
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("New folder")
                }
            }
            .scrollIndicators(.never)
            Spacer(minLength: MFSpacing.s)
            row("Recently deleted", systemImage: "trash", scope: .recentlyDeleted, count: counts.recentlyDeleted)
            SpeechStatusCard(status: model.speechStatus.speech)
                .padding(.top, MFSpacing.s + MFSpacing.xxs)
                .padding(.horizontal, MFSpacing.xs)
                .padding(.bottom, MFSpacing.xs)
        }
        .padding(.vertical, MFSpacing.m)
        .padding(.horizontal, MFSpacing.s + MFSpacing.xxs)
        .background(MFColor.sidebar)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Notes sidebar")
    }

    private func header(_ title: String) -> some View {
        Text(title.uppercased())
            .font(MFFont.sectionHeader)
            .tracking(MFFont.sectionHeaderTracking)
            .foregroundStyle(MFColor.tertiary)
            .padding(.horizontal, MFSpacing.s + MFSpacing.xxs)
            .padding(.top, MFSpacing.m)
            .padding(.bottom, MFSpacing.s - MFSpacing.xxs)
            .accessibilityAddTraits(.isHeader)
    }

    private func row(_ title: String, systemImage: String, scope: NoteScope, count: Int) -> some View {
        Button { model.scope = scope } label: {
            rowLabel(title, systemImage: systemImage, count: count, isSelected: model.scope == scope)
        }
        .buttonStyle(.plain)
        .accessibilityLabel("\(title), \(count)")
        .accessibilityAddTraits(model.scope == scope ? .isSelected : [])
    }

    @ViewBuilder
    private func folderRow(_ folder: NoteFolder, count: Int) -> some View {
        let isSelected = model.scope == .folder(folder.id)
        Group {
            if model.renamingFolderID == folder.id {
                FolderNameField(name: folder.name) { model.rename(folder, to: $0) }
                    .padding(.horizontal, MFSpacing.s + MFSpacing.xxs)
                    .frame(height: NotesMetrics.sidebarRowHeight)
            } else {
                Button { model.scope = .folder(folder.id) } label: {
                    rowLabel(folder.name, systemImage: "folder", count: count, isSelected: isSelected)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("\(folder.name), \(count)")
                .accessibilityAddTraits(isSelected ? .isSelected : [])
            }
        }
        .contextMenu {
            Button("Rename") { model.renamingFolderID = folder.id }
            Button("Move Up") { model.moveFolder(folder, by: -1) }
                .disabled(model.folders.first?.id == folder.id)
            Button("Move Down") { model.moveFolder(folder, by: 1) }
                .disabled(model.folders.last?.id == folder.id)
            Divider()
            Button("Delete folder", role: .destructive) { model.delete(folder) }
        }
        .accessibilityAction(named: "Move up") { model.moveFolder(folder, by: -1) }
        .accessibilityAction(named: "Move down") { model.moveFolder(folder, by: 1) }
        .draggable(NotesDragItem.folder(folder.id))
        .dropDestination(for: String.self) { items, _ in
            let notes = NotesDragItem.ids(items, prefix: NotesDragItem.notePrefix)
            if !notes.isEmpty { return model.moveNotes(notes, to: folder) }
            guard let moving = NotesDragItem.ids(items, prefix: NotesDragItem.folderPrefix).first else { return false }
            return model.dropFolder(moving, onto: folder.id)
        }
    }

    private func rowLabel(_ title: String, systemImage: String, count: Int?, isSelected: Bool,
                          tint: Color = MFColor.ink) -> some View {
        HStack(spacing: NotesMetrics.sidebarIconGap) {
            Image(systemName: systemImage)
                .font(.system(size: NotesMetrics.sidebarGlyph, weight: .regular))
                .foregroundStyle(MFColor.secondary)
                .frame(width: NotesMetrics.sidebarIcon)
                .accessibilityHidden(true)
            Text(title)
                .font(isSelected ? MFFont.calloutEmphasis : MFFont.callout)
                .foregroundStyle(tint)
                .lineLimit(1)
            Spacer(minLength: 0)
            if let count {
                Text("\(count)")
                    .font(MFFont.caption)
                    .foregroundStyle(MFColor.tertiary)
                    .monospacedDigit()
            }
        }
        .padding(.horizontal, MFSpacing.s + MFSpacing.xxs)
        .frame(height: NotesMetrics.sidebarRowHeight)
        .background(isSelected ? MFColor.sidebarSelected : .clear, in: RoundedRectangle(cornerRadius: MFRadius.s))
        .contentShape(Rectangle())
    }
}

/// Renames a folder in place; Return saves, Escape keeps the old name.
private struct FolderNameField: View {
    @State var name: String
    let onCommit: (String) -> Void
    @FocusState private var focused: Bool

    var body: some View {
        TextField("Folder name", text: $name)
            .textFieldStyle(.plain)
            .font(MFFont.callout)
            .focused($focused)
            .onSubmit { onCommit(name) }
            .onExitCommand { onCommit("") }
            .onAppear { focused = true }
            // Clicking away keeps what was typed.
            .onChange(of: focused) { _, isFocused in if !isFocused { onCommit(name) } }
            .accessibilityLabel("Folder name")
    }
}

/// "Fluid Speech Pro · Ready · On this Mac" (or Not installed, Downloading
/// n%, Preparing, Needs attention).
struct SpeechStatusCard: View {
    let status: SpeechModelState

    var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.s - MFSpacing.xxs) {
            Text("Fluid Speech Pro")
                .font(MFFont.caption.weight(.semibold))
                .foregroundStyle(MFColor.ink)
            HStack(spacing: MFSpacing.s - MFSpacing.xxs) {
                MFChip(stateTitle, tone: tone)
                if status == .ready { MFChip("On this Mac") }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.vertical, MFSpacing.s + MFSpacing.xxs)
        .padding(.horizontal, MFSpacing.m)
        .background(MFColor.surface, in: RoundedRectangle(cornerRadius: MFRadius.m))
        .overlay(RoundedRectangle(cornerRadius: MFRadius.m).strokeBorder(MFColor.hairline, lineWidth: MFLayout.hairline))
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Fluid Speech Pro, \(stateTitle)")
    }

    var stateTitle: String {
        switch status {
        case .ready: "Ready"
        case .notInstalled: "Not installed"
        case .downloading(let progress): "Downloading \(Int((progress * 100).rounded()))%"
        case .paused(let progress): "Paused at \(Int((progress * 100).rounded()))%"
        case .preparing: "Preparing…"
        case .failed: "Needs attention"
        }
    }

    private var tone: MFTone {
        switch status {
        case .ready: .success
        case .downloading, .preparing: .accent
        case .failed, .paused: .preview
        case .notInstalled: .neutral
        }
    }
}
