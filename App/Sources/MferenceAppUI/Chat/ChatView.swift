import MferenceAppCore
import MferenceDesign
import SwiftUI

/// Where the chat screen hands off to the rest of the app. The shell wires
/// these; the defaults do nothing (previews, snapshots).
public struct ChatActions {
    /// "Review" on the Memory card: Settings > Memory.
    public var openMemorySettings: @MainActor () -> Void
    /// "Manage models" in the picker: the model library.
    public var openModelLibrary: @MainActor () -> Void
    /// Suggestion chips that belong to another screen ("Scan a receipt").
    public var openSection: @MainActor (AppSection) -> Void

    public init(openMemorySettings: @escaping @MainActor () -> Void = {},
                openModelLibrary: @escaping @MainActor () -> Void = {},
                openSection: @escaping @MainActor (AppSection) -> Void = { _ in }) {
        self.openMemorySettings = openMemorySettings
        self.openModelLibrary = openModelLibrary
        self.openSection = openSection
    }
}

/// Initial UI state, for snapshots of the mockup's states (02b picker open,
/// 02c steps expanded).
public struct ChatPresentation: Sendable, Hashable {
    public var pickerOpen: Bool
    public var stepsExpanded: Bool

    public init(pickerOpen: Bool = false, stepsExpanded: Bool = false) {
        self.pickerOpen = pickerOpen
        self.stepsExpanded = stepsExpanded
    }
}

/// The Chat section (mockup 02, 02b, 02c): conversation sidebar, title bar,
/// messages, composer with the model picker, suggestion chips.
public struct ChatView: View {
    @Bindable var model: ChatViewModel
    let actions: ChatActions
    let presentation: ChatPresentation

    public init(model: ChatViewModel, actions: ChatActions = ChatActions(),
                presentation: ChatPresentation = ChatPresentation()) {
        self.model = model
        self.actions = actions
        self.presentation = presentation
    }

    public var body: some View {
        HStack(spacing: 0) {
            ChatSidebar(model: model, actions: actions)
                .frame(width: MFLayout.sidebarWidth)
            Rectangle()
                .fill(MFColor.hairlineFaint)
                .frame(width: ChatMetrics.hairlineWidth)
                .accessibilityHidden(true)
            ChatDetail(model: model, actions: actions, presentation: presentation)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .background(MFColor.surface)
        .task { model.start() }
    }
}

/// Title bar, messages and composer.
struct ChatDetail: View {
    @Bindable var model: ChatViewModel
    let actions: ChatActions
    @State private var pickerOpen: Bool

    init(model: ChatViewModel, actions: ChatActions, presentation: ChatPresentation) {
        self.model = model
        self.actions = actions
        self.presentation = presentation
        _pickerOpen = State(initialValue: presentation.pickerOpen)
    }

    let presentation: ChatPresentation

    var body: some View {
        VStack(spacing: 0) {
            VStack(spacing: 0) {
                titleBar
                Rectangle()
                    .fill(MFColor.hairlineFaint)
                    .frame(height: ChatMetrics.hairlineWidth)
                    .accessibilityHidden(true)
                MessageList(model: model, stepsExpanded: presentation.stepsExpanded)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            .overlay {
                if pickerOpen {
                    // Clicking anywhere outside the picker closes it.
                    Color.clear
                        .contentShape(Rectangle())
                        .onTapGesture { pickerOpen = false }
                        .accessibilityHidden(true)
                }
            }
            VStack(alignment: .leading, spacing: MFSpacing.m) {
                // A zero-height anchor on the composer's top edge: the picker
                // hangs from it upward, over the messages (mockup 02b).
                Color.clear
                    .frame(height: 0)
                    .overlay(alignment: .bottomLeading) {
                        if pickerOpen {
                            ModelPicker(model: model, isOpen: $pickerOpen, actions: actions)
                                .padding(.bottom, MFSpacing.s)
                                .transition(.opacity)
                        }
                    }
                    .padding(.bottom, -MFSpacing.m)
                if model.historyTrimmed {
                    LongChatNotice(model: model)
                }
                Composer(model: model, pickerOpen: $pickerOpen)
                SuggestionChips(model: model, actions: actions)
            }
            .frame(maxWidth: ChatMetrics.columnWidth)
            .padding(.horizontal, MFSpacing.xl)
            .padding(.bottom, MFSpacing.l)
            .zIndex(1)
        }
    }

    private var titleBar: some View {
        HStack(spacing: MFSpacing.s) {
            Text(model.title)
                .font(MFFont.headline)
                .foregroundStyle(MFColor.ink)
                .lineLimit(1)
                .accessibilityAddTraits(.isHeader)
            privateToggle
            Spacer(minLength: 0)
        }
        .padding(.horizontal, MFSpacing.xl)
        .padding(.vertical, MFSpacing.m)
    }

    /// Private on or off for this chat (private chats never feed memory).
    private var privateToggle: some View {
        Button {
            model.setPrivate(!model.isPrivate)
        } label: {
            ChatChip(systemImage: model.isPrivate ? "lock" : "lock.open",
                     title: model.isPrivate ? "Private" : "Not private",
                     foreground: model.isPrivate ? MFColor.secondary : MFColor.tertiary)
                .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .help(model.isPrivate ? "Private: never used for memory. Click to make it a regular chat."
                              : "Click to make this chat private: it is never used for memory.")
        .accessibilityLabel("Private chat")
        .accessibilityValue(model.isPrivate ? "On" : "Off")
        .accessibilityAddTraits(.isToggle)
    }
}

/// Shown above the composer when Apple's model left out the oldest messages
/// of a long chat to fit its context.
struct LongChatNotice: View {
    @Bindable var model: ChatViewModel

    var body: some View {
        HStack(spacing: MFSpacing.compact) {
            Text("This chat is longer than Apple’s model can read at once, so it sees your latest messages only.")
                .font(MFFont.preview)
                .foregroundStyle(MFColor.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: MFSpacing.s)
            if let target = model.longChatModel {
                Button("Switch to \(target.name)") { model.selectModel(target) }
                    .buttonStyle(.plain)
                    .font(MFFont.preview.weight(.medium))
                    .foregroundStyle(MFColor.accent)
            }
        }
        .padding(.horizontal, MFSpacing.comfortable)
        .padding(.vertical, MFSpacing.compact)
        .background(MFColor.approvalFill, in: RoundedRectangle(cornerRadius: MFRadius.m))
        .overlay(RoundedRectangle(cornerRadius: MFRadius.m).strokeBorder(MFColor.approvalBorder))
        .accessibilityElement(children: .contain)
    }
}

/// A small static capsule ("Private", "Sample conversation").
struct ChatChip: View {
    var systemImage: String?
    var title: String
    var foreground: Color = MFColor.secondary
    var background: Color = MFColor.fill

    var body: some View {
        HStack(spacing: MFSpacing.xs) {
            if let systemImage {
                Image(systemName: systemImage).imageScale(.small)
            }
            Text(title)
        }
        .font(MFFont.chip)
        .foregroundStyle(foreground)
        .padding(.horizontal, MFSpacing.s)
        .padding(.vertical, MFSpacing.xs)
        .background(background, in: Capsule())
    }
}
