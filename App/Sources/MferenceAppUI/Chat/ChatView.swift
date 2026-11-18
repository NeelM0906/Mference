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
            if model.isPrivate {
                ChatChip(systemImage: "lock", title: "Private")
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, MFSpacing.xl)
        .padding(.vertical, MFSpacing.m)
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
        .font(MFFont.calloutEmphasis)
        .foregroundStyle(foreground)
        .padding(.horizontal, MFSpacing.s)
        .padding(.vertical, MFSpacing.xs)
        .background(background, in: Capsule())
    }
}
