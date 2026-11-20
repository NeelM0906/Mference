import MferenceAppCore
import MferenceDesign
import SwiftUI

/// Notes-only sizes measured from the board (`Main.dc.html`, CSS px = pt).
enum NotesMetrics {
    /// `.side` and `.list` widths.
    static let sidebarWidth: CGFloat = 210
    static let listWidth: CGFloat = 300
    /// `.srow` height and icon size.
    static let sidebarRowHeight: CGFloat = 32
    static let sidebarIcon: CGFloat = 15
    static let sidebarGlyph: CGFloat = 13
    /// `.srow` gap between icon and title.
    static let sidebarIconGap: CGFloat = 9
    /// The list's kind marks (`.nt svg`).
    static let kindIcon: CGFloat = 13
    static let kindGlyph: CGFloat = 12
    /// `.nrow`: padding 9, title-to-preview 3, time-to-preview 5.
    static let rowVerticalPadding: CGFloat = 9
    static let rowLineGap: CGFloat = 3
    static let timeGap: CGFloat = 5
    static let searchIcon: CGFloat = 13
    /// `.search` height.
    static let searchHeight: CGFloat = 32
    /// `.etb` and `.foot` heights.
    static let editorToolbarHeight: CGFloat = 48
    static let footerHeight: CGFloat = 34
    /// `.btn.icon` side, and its glyphs.
    static let iconButton: CGFloat = 30
    static let toolbarIcon: CGFloat = 16
    static let toolbarGlyph: CGFloat = 14
    /// `.sep` between the formatting buttons and Dictate.
    static let separatorHeight: CGFloat = 18
    /// `.rec` dot on Record meeting.
    static let recordDot: CGFloat = 9
    /// Empty states are at most this wide.
    static let emptyStateWidth: CGFloat = 320
    static let emptyStateIcon: CGFloat = 24
}

/// The Notes section (Notes spec 4.1, board 1): sidebar, note list and
/// editor in three panes under the shell chrome, which gets Transcribe file,
/// Record meeting and New note (⌘N).
public struct NotesRootView: View {
    @Environment(ScreenModels.self) private var screens: ScreenModels?
    private let injected: NotesModel?

    /// The shell's slot: the model comes from `ScreenModels.notes`.
    public init() { injected = nil }

    /// Snapshots, previews and tests.
    public init(model: NotesModel) { injected = model }

    public var body: some View {
        if let model = injected ?? screens?.notes {
            NotesScreen(model: model)
        } else {
            NotesEmptyState(systemImage: "note.text", title: "Notes",
                            message: "Notes is starting. Try again in a moment.")
        }
    }
}

struct NotesScreen: View {
    @Bindable var model: NotesModel

    var body: some View {
        HStack(spacing: 0) {
            NotesSidebar(model: model)
                .frame(width: NotesMetrics.sidebarWidth)
            Divider().overlay(MFColor.chromeDivider)
            NotesList(model: model)
                .frame(width: NotesMetrics.listWidth)
            Divider().overlay(MFColor.chromeDivider)
            NoteEditorPane(model: model)
                .frame(maxWidth: .infinity)
        }
        .background(MFColor.surface)
        .task {
            await model.purgeExpired()
            model.selectFirstIfNeeded()
        }
        .shellToolbar(id: "notes.actions") { NotesToolbarActions(model: model) }
        .confirmationDialog(model.pendingDeletion?.title ?? "",
                            isPresented: Binding(get: { model.pendingDeletion != nil },
                                                 set: { if !$0 { model.cancelDeletion() } }),
                            titleVisibility: .visible) {
            Button("Delete Forever", role: .destructive) { Task { await model.confirmDeletion() } }
            Button("Cancel", role: .cancel) { model.cancelDeletion() }
        } message: {
            Text(model.pendingDeletion?.message ?? "")
        }
        .alert("Notes", isPresented: Binding(get: { model.errorMessage != nil },
                                             set: { if !$0 { model.dismissError() } })) {
            Button("OK") { model.dismissError() }
        } message: {
            Text(model.errorMessage ?? "")
        }
    }
}

/// Transcribe file, Record meeting and New note in the shell's top bar.
struct NotesToolbarActions: View {
    let model: NotesModel

    var body: some View {
        HStack(spacing: MFSpacing.s) {
            if let toolbar = model.voice.toolbar {
                // N3's toolbar: Transcribe file and Record meeting with its popover.
                toolbar()
            } else {
                ownVoiceButtons
            }
            newNoteButton
        }
    }

    /// Until N3's toolbar is wired: the same two buttons, disabled without
    /// their actions.
    @ViewBuilder
    private var ownVoiceButtons: some View {
        Button {
            model.voice.transcribeFile?()
        } label: {
            Label("Transcribe file", systemImage: "text.document")
        }
        .buttonStyle(.mfSecondary)
        .disabled(model.voice.transcribeFile == nil)
        .help("Turn an audio or video file into a note")

        Button {
            model.voice.recordMeeting?()
        } label: {
            HStack(spacing: MFSpacing.s - MFSpacing.xxs) {
                Circle().fill(MFColor.danger)
                    .frame(width: NotesMetrics.recordDot, height: NotesMetrics.recordDot)
                    .accessibilityHidden(true)
                Text("Record meeting")
            }
        }
        .buttonStyle(.mfSecondary)
        .disabled(model.voice.recordMeeting == nil)
        .help("Record a meeting with a live transcript")
    }

    private var newNoteButton: some View {
        Button {
            model.newNote()
        } label: {
            Image(systemName: "square.and.pencil")
                .frame(width: NotesMetrics.iconButton, height: NotesMetrics.iconButton)
        }
        .buttonStyle(.plain)
        .background(MFColor.surface, in: RoundedRectangle(cornerRadius: MFRadius.m))
        .overlay(RoundedRectangle(cornerRadius: MFRadius.m).strokeBorder(MFColor.hairline, lineWidth: MFLayout.hairline))
        .keyboardShortcut("n", modifiers: .command)
        .accessibilityLabel("New note")
        .help("New note (⌘N)")
    }
}

/// A centered message with an optional row of actions.
struct NotesEmptyState<Actions: View>: View {
    let systemImage: String
    let title: String
    let message: String
    let actions: Actions

    init(systemImage: String, title: String, message: String, @ViewBuilder actions: () -> Actions) {
        self.systemImage = systemImage
        self.title = title
        self.message = message
        self.actions = actions()
    }

    var body: some View {
        VStack(spacing: MFSpacing.m) {
            Image(systemName: systemImage)
                .font(.system(size: NotesMetrics.emptyStateIcon, weight: .regular))
                .foregroundStyle(MFColor.tertiary)
                .accessibilityHidden(true)
            Text(title)
                .font(MFFont.headline)
                .foregroundStyle(MFColor.ink)
            Text(message)
                .font(MFFont.callout)
                .foregroundStyle(MFColor.secondary)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
            actions
        }
        .frame(maxWidth: NotesMetrics.emptyStateWidth)
        .padding(MFSpacing.xl)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .accessibilityElement(children: .contain)
    }
}

extension NotesEmptyState where Actions == EmptyView {
    init(systemImage: String, title: String, message: String) {
        self.init(systemImage: systemImage, title: title, message: message) { EmptyView() }
    }
}
