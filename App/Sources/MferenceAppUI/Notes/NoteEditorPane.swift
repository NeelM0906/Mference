import AppKit
import MferenceAppCore
import MferenceDesign
import SwiftUI
import UniformTypeIdentifiers

/// The editor column: formatting toolbar, the note, and the footer
/// ("Saved · Personal · 86 words").
struct NoteEditorPane: View {
    @Bindable var model: NotesModel
    @Environment(VoiceModels.self) private var voice: VoiceModels?
    /// The finished recording's player and transcript for the open note.
    @State private var recording: RecordingViewState?

    struct RecordingViewState {
        var noteID: UUID
        var transcript: TranscriptModel
        var player: RecordingPlayerModel
    }
    @State private var importingAttachment = false

    var body: some View {
        content
            // First use of a voice feature: the setup card takes the editor
            // area until the model is ready or the person goes back (board 7).
            .overlay {
                if let voice, voice.setup.pendingAction != nil {
                    SpeechSetupCard(model: voice.setup)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .background(MFColor.surface)
                        // The hidden note must not take typing or Esc.
                        .onAppear { model.editor.resignFocus() }
                }
            }
    }

    @ViewBuilder
    private var content: some View {
        if let note = model.selectedNote {
            VStack(spacing: 0) {
                toolbar(note)
                Divider().overlay(MFColor.chromeDivider)
                // A newer or damaged body explains itself in the editor.
                if note.deletedAt != nil { deletedBanner(note) }
                if let recording, recording.noteID == note.id {
                    NoteRecordingHeader(model: model, note: note, transcript: recording.transcript,
                                        player: recording.player)
                }
                NoteEditorView(controller: model.editor,
                               dateLine: Self.dateLine(note.updatedAt, locale: model.locale, calendar: model.calendar),
                               bottomInset: showsDictationBar ? NoteEditorMetrics.dictationBarInset : 0)
                    .accessibilityLabel("Note text")
                    // Dictation's bar sits over the bottom of the note (board 2).
                    .overlay(alignment: .bottom) {
                        if let voice, showsDictationBar {
                            DictationBar(controller: voice.dictation)
                                .padding(.horizontal, MFSpacing.l)
                                .padding(.bottom, MFSpacing.m)
                        }
                    }
                if let recording, recording.noteID == note.id {
                    Divider().overlay(MFColor.chromeDivider)
                    NoteRecordingFooter(transcript: recording.transcript, player: recording.player)
                }
                Divider().overlay(MFColor.chromeDivider)
                footer(note)
            }
            .background(MFColor.surface)
            .task(id: note.id) { await loadRecording(for: note) }
            .onDisappear { recording?.player.pause() }
            .fileImporter(isPresented: $importingAttachment, allowedContentTypes: [.image, .audio],
                          allowsMultipleSelection: true) { result in
                guard case .success(let urls) = result else { return }
                for url in urls {
                    let scoped = url.startAccessingSecurityScopedResource()
                    defer { if scoped { url.stopAccessingSecurityScopedResource() } }
                    try? model.editor.attachFile(url)
                }
            }
        } else {
            NotesEmptyState(systemImage: "square.and.pencil", title: "No note selected",
                            message: "Choose a note, or start a new one with ⌘N.")
                .background(MFColor.surface)
        }
    }

    private var showsDictationBar: Bool {
        guard let voice else { return false }
        return voice.dictation.isActive || voice.dictation.error != nil
    }

    /// A finished meeting or file note gets its player and transcript; a
    /// live meeting, a typed note or a build without meetings gets none.
    private func loadRecording(for note: Note) async {
        recording?.player.pause()
        guard let meetings = model.meetings, note.recording != nil, model.liveMeetingNoteID != note.id,
              let transcript = meetings.transcript(for: note.id) else {
            recording = nil
            model.editor.onTimeChipClick = nil
            return
        }
        let audio = meetings.audio(for: note.id)
        let player = RecordingPlayerModel(tracks: audio?.tracks ?? [], directory: audio?.directory,
                                          duration: audio?.duration ?? note.recording?.duration ?? 0)
        recording = RecordingViewState(noteID: note.id, transcript: transcript, player: player)
        // A time chip in the note plays from there.
        model.editor.onTimeChipClick = { [weak player] time in player?.play(from: time) }
        await player.load()
    }

    /// "19 November 2026 at 11:40".
    static func dateLine(_ date: Date, locale: Locale = .current, calendar: Calendar = .current) -> String {
        var day = Date.FormatStyle().day().month(.wide).year()
        var time = Date.FormatStyle(date: .omitted, time: .shortened)
        day.locale = locale
        day.calendar = calendar
        day.timeZone = calendar.timeZone
        time.locale = locale
        time.calendar = calendar
        time.timeZone = calendar.timeZone
        return "\(date.formatted(day)) at \(date.formatted(time))"
    }

    // MARK: Toolbar

    private func toolbar(_ note: Note) -> some View {
        let editable = note.deletedAt == nil
        return HStack(spacing: MFSpacing.xxs) {
            Menu {
                ForEach([NoteBlockKind.title, .heading, .subheading, .body], id: \.self) { kind in
                    Button { model.editor.apply(kind) } label: {
                        if model.editor.currentKind == kind { Label(kind.label, systemImage: "checkmark") } else { Text(kind.label) }
                    }
                }
                Divider()
                // The text view handles these keys itself; the menu shows them.
                Button("Bold  ⌘B") { model.editor.toggleBold() }
                Button("Italic  ⌘I") { model.editor.toggleItalic() }
                Button("Underline  ⌘U") { model.editor.toggleUnderline() }
                Button("Strikethrough  ⇧⌘X") { model.editor.toggleStrikethrough() }
                Divider()
                Button("Bulleted list") { model.editor.apply(.bullet) }
                Button("Numbered list") { model.editor.apply(.numbered) }
            } label: {
                Text("Aa").font(MFFont.calloutSemibold)
            }
            .menuStyle(.button)
            .menuIndicator(.hidden)
            .buttonStyle(.plain)
            .frame(width: NotesMetrics.iconButton, height: NotesMetrics.iconButton)
            .accessibilityLabel("Text style")
            .help("Text style")

            iconButton("checklist", label: "Checklist", help: "Checklist (⇧⌘L)") { model.editor.toggleChecklist() }
            iconButton("paperclip", label: "Attach", help: "Attach an image or audio file") { importingAttachment = true }

            Rectangle().fill(MFColor.hairline)
                .frame(width: MFLayout.hairline, height: NotesMetrics.separatorHeight)
                .padding(.horizontal, MFSpacing.s - MFSpacing.xxs)
                .accessibilityHidden(true)

            Button {
                model.voice.toggleDictation?(model.editor)
            } label: {
                HStack(spacing: MFSpacing.s - MFSpacing.xxs) {
                    let dictating = model.voice.isDictating?(note.id) == true
                    Image(systemName: dictating ? "mic.fill" : "mic")
                    Text(dictating ? "Dictating" : "Dictate")
                    MFKeycap("⇧⌘D")
                }
            }
            .buttonStyle(.mfSecondary)
            // ⇧⌘D is N3's menu command; the keycap only shows it.
            .disabled(model.voice.toggleDictation == nil)
            .accessibilityLabel("Dictate")
            .accessibilityHint("Shift Command D")

            Spacer(minLength: MFSpacing.s)

            if model.liveMeetingNoteID == note.id, model.showsPanesDuringMeeting {
                // Back to the meeting layout (board 4).
                Button {
                    model.showsPanesDuringMeeting = false
                } label: {
                    Label("Meeting", systemImage: "waveform")
                }
                .buttonStyle(.mfSecondary)
                .help("Show the live meeting")
            }

            iconButton(note.isPinned ? "pin.fill" : "pin", label: note.isPinned ? "Unpin" : "Pin",
                       help: note.isPinned ? "Unpin" : "Pin") { model.togglePin(note) }
            Menu {
                // The Markdown is made only when something is shared.
                ShareLink("Share…", item: NoteShareItem(editor: model.editor),
                          preview: SharePreview(model.title(of: note)))
                Divider()
                Section("Export as") {
                    ForEach(NoteExportFormat.allCases, id: \.self) { format in
                        Button(format.label) { model.export(note, as: format) }
                    }
                }
            } label: {
                Image(systemName: "square.and.arrow.up")
                    .font(.system(size: NotesMetrics.toolbarGlyph))
                    .foregroundStyle(MFColor.secondary)
                    .frame(width: NotesMetrics.iconButton, height: NotesMetrics.iconButton)
            }
            .menuStyle(.button)
            .menuIndicator(.hidden)
            .buttonStyle(.plain)
            .frame(width: NotesMetrics.iconButton, height: NotesMetrics.iconButton)
            .accessibilityLabel("Share or export")
            .help("Share or export")
            iconButton("trash", label: "Delete", help: "Delete") { model.delete(note) }
        }
        .disabled(!editable)
        .padding(.horizontal, MFSpacing.m + MFSpacing.xxs)
        .frame(height: NotesMetrics.editorToolbarHeight)
    }

    private func iconButton(_ systemImage: String, label: String, help: String,
                            action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.system(size: NotesMetrics.toolbarGlyph))
                .foregroundStyle(MFColor.secondary)
                .frame(width: NotesMetrics.iconButton, height: NotesMetrics.iconButton)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(label)
        .help(help)
    }

    private func deletedBanner(_ note: Note) -> some View {
        HStack(spacing: MFSpacing.s) {
            Text("This note is in Recently deleted.")
                .font(MFFont.callout)
                .foregroundStyle(MFColor.secondary)
            Spacer()
            Button("Restore") { model.restore(note) }.buttonStyle(.mfSecondary)
        }
        .padding(.horizontal, MFSpacing.l)
        .padding(.vertical, MFSpacing.s)
        .background(MFColor.fillSubtle)
    }

    // MARK: Footer

    private func footer(_ note: Note) -> some View {
        HStack(spacing: MFSpacing.s + MFSpacing.xxs) {
            Text(saveLabel)
            if let folder = note.folder {
                Text("·").accessibilityHidden(true)
                Text(folder.name)
            }
            Text("·").accessibilityHidden(true)
            Text(model.editor.wordCount == 1 ? "1 word" : "\(model.editor.wordCount) words")
            Spacer()
        }
        .font(MFFont.caption)
        .foregroundStyle(MFColor.tertiary)
        .padding(.horizontal, MFSpacing.l + MFSpacing.xxs)
        .frame(height: NotesMetrics.footerHeight)
        .accessibilityElement(children: .combine)
    }

    private var saveLabel: String {
        switch model.editor.saveState {
        case .saved: "Saved"
        case .editing: "Editing"
        case .failed: "Not saved"
        }
    }
}

/// The note's text for the share sheet, made only when something is shared
/// (not on every keystroke).
nonisolated struct NoteShareItem: Transferable {
    let editor: NoteEditorController

    static var transferRepresentation: some TransferRepresentation {
        ProxyRepresentation { item in item.text() }
    }

    /// The editor lives on the main thread; the share sheet may ask from
    /// another one.
    private func text() -> String {
        if Thread.isMainThread { return MainActor.assumeIsolated { editor.shareText } }
        return DispatchQueue.main.sync { MainActor.assumeIsolated { editor.shareText } }
    }
}
