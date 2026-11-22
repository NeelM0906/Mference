import MferenceAppCore
import MferenceDesign
import SwiftUI

/// The app window: chrome (brand, section pills, "On this Mac", search,
/// appearance, Settings) over the selected section's screen, on the frosted
/// window background.
///
/// Each section routes to one view (`SectionContent`): Chat → `ChatView`
/// (U3), Find → `FindView`, Mail / Calendar / Scan → `MailView` / `CalendarView` / `ScanView`
/// (U4), Settings → `SettingsView` (U1), Notes → `NotesRootView` (N1), Errands → `ComingLaterView`
/// (Phase C).
public struct RootView: View {
    /// Where the chrome is drawn.
    public enum ChromePlacement: Sendable {
        /// In the window's toolbar, next to the system window buttons (the app).
        case toolbar
        /// As a bar at the top of the content (snapshots and previews).
        case inline
    }

    @Bindable var model: ShellModel
    let chrome: ChromePlacement
    @Environment(ScreenModels.self) private var screenModels: ScreenModels?
    /// The visible screen's `shellToolbar` items, for the window toolbar.
    @State private var toolbarItems: [ShellToolbarItem] = []

    public init(model: ShellModel, chrome: ChromePlacement = .toolbar) {
        self.model = model
        self.chrome = chrome
    }

    /// First run shows until it is completed, once a model library is wired.
    private var showsOnboarding: Bool {
        model.needsOnboarding(hasLibrary: screenModels?.library != nil)
    }

    public var body: some View {
        if showsOnboarding, let library = screenModels?.library {
            // The first-run window keeps its close and minimise buttons; the
            // left panel runs under them, as in the mockup.
            OnboardingView(library: library) { model.completeOnboarding() }
                .ignoresSafeArea(edges: .top)
                .frame(minWidth: MFLayout.minimumWindow.width, minHeight: MFLayout.minimumWindow.height)
                .toolbar(removing: .title)
                .toolbarBackgroundVisibility(.hidden, for: .windowToolbar)
                .preferredColorScheme(model.appearance.colorScheme)
        } else {
            shell
        }
    }

    @ViewBuilder
    private var shell: some View {
        switch chrome {
        case .inline:
            // The chrome reads the screen's toolbar items in the same pass
            // (works in ImageRenderer snapshots).
            SectionContent(section: model.selection)
                .environment(model)
                .padding(.top, ShellMetrics.chromeHeight)
                .overlayPreferenceValue(ShellToolbarKey.self, alignment: .top) { items in
                    ShellChromeBar(model: model, items: items)
                }
            .mfWindowBackground()
            .preferredColorScheme(model.appearance.colorScheme)
        case .toolbar:
            SectionContent(section: model.selection)
                .environment(model)
                .onPreferenceChange(ShellToolbarKey.self) { items in
                    Task { @MainActor in toolbarItems = items }
                }
                // The mockup's border under the top bar; the shell draws it,
                // not each screen.
                .overlay(alignment: .top) {
                    Rectangle().fill(MFColor.chromeDivider).frame(height: MFLayout.hairline)
                }
                .frame(minWidth: MFLayout.minimumWindow.width, minHeight: MFLayout.minimumWindow.height)
                .toolbar {
                    ToolbarItem(placement: .navigation) {
                        HStack(spacing: MFSpacing.m) {
                            ShellBrand()
                            ShellPillBar(model: model)
                        }
                    }
                    .sharedBackgroundVisibility(.hidden)
                    ToolbarSpacer(.flexible)
                    ToolbarItem(placement: .primaryAction) {
                        ShellTrailingControls(model: model, items: toolbarItems)
                            .frame(maxWidth: .infinity, alignment: .trailing)
                            .padding(.trailing, MFSpacing.xs)
                    }
                    .sharedBackgroundVisibility(.hidden)
                }
                .toolbar(removing: .title)
                .toolbarBackgroundVisibility(.hidden, for: .windowToolbar)
                .containerBackground(for: .window) { MFWindowBackground() }
                .preferredColorScheme(model.appearance.colorScheme)
        }
    }
}

/// The screen for one section.
struct SectionContent: View {
    let section: AppSection
    @Environment(ScreenModels.self) private var screenModels: ScreenModels?
    @Environment(ShellModel.self) private var shell: ShellModel?
    /// The app's one chat model (U3), kept for the app's lifetime so switching
    /// pills does not drop a running answer. Absent in snapshots.
    @Environment(ChatViewModel.self) private var chat: ChatViewModel?

    var body: some View {
        Group {
            switch section {
            case .chat:
                if let chat {
                    ChatView(model: chat, actions: chatActions)
                } else {
                    SectionPlaceholderView(section: .chat)
                }
            case .find: FindView(openChat: openChatAbout, askAcross: askAcross)
            case .mail: MailView()
            case .calendar: CalendarView(continueInChat: weekToChat)
            case .scan: ScanView()
            case .settings:
                SettingsView(library: screenModels?.library, memory: screenModels?.memory,
                             settings: screenModels?.settings, pane: shell?.settingsPane ?? .models,
                             showsLibrary: shell?.settingsShowsLibrary ?? false)
            case .notes: NotesRootView()
            case .errands: ComingLaterView(section: section)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .id(section)
    }

    /// Find's "Ask about this": a new chat about the file, with Files on.
    private func openChatAbout(_ file: URL) {
        chat?.startChat(about: file)
        shell?.show(.chat)
    }

    /// Nil without a chat model (snapshots), which hides the button.
    private var weekToChat: ((WeekAskModel) -> Void)? {
        guard chat != nil else { return nil }
        return { ask in continueWeekInChat(ask) }
    }

    /// Calendar's "Continue in Chat": a Private chat holding the week's
    /// questions and answers.
    private func continueWeekInChat(_ ask: WeekAskModel) {
        guard chat?.continueChat(title: ask.chatTitle, context: ask.chatContext, turns: ask.chatTurns) == true else {
            ask.chatUnavailable()
            return
        }
        shell?.show(.chat)
    }

    /// Find's "Ask in Chat": a new chat with Files on, asking the search.
    private func askAcross(_ question: String) {
        chat?.startChat(question: question)
        shell?.show(.chat)
    }

    /// Where chat hands off: Settings (Memory, the model library) and other
    /// sections (suggestion chips such as "Scan a receipt").
    private var chatActions: ChatActions {
        ChatActions(openMemorySettings: { shell?.openSettings(.memory) },
                    openModelLibrary: { shell?.openSettings(.models, showingLibrary: true) },
                    openSection: { shell?.show($0) })
    }
}

/// Menu commands: a Go menu with a shortcut per section (⌘1-⌘7), Settings
/// on ⌘, and search on ⌘K. They drive the same `ShellModel` as the pills.
public struct ShellCommands: Commands {
    let model: ShellModel

    public init(model: ShellModel) { self.model = model }

    public var body: some Commands {
        CommandGroup(replacing: .appSettings) {
            Button("Settings…") { model.show(.settings) }
                .keyboardShortcut(ShellModel.shortcut(for: .settings), modifiers: .command)
        }
        CommandMenu("Go") {
            ForEach(AppSection.pills) { section in
                Button(section.title) { model.show(section) }
                    .keyboardShortcut(ShellModel.shortcut(for: section), modifiers: .command)
            }
            Divider()
            Button("Search…") { model.isSearchPresented = true }
                .keyboardShortcut("k", modifiers: .command)
        }
    }
}
