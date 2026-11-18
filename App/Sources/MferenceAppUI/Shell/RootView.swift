import MferenceAppCore
import MferenceDesign
import SwiftUI

/// The app window: chrome (brand, section pills, "On this Mac", search,
/// appearance, Settings) over the selected section's screen, on the frosted
/// window background.
///
/// Each section routes to one view (`SectionContent`): Chat → `ChatView`
/// (U3), Mail / Calendar / Scan → `MailView` / `CalendarView` / `ScanView`
/// (U4), Settings → `SettingsView` (U1), Notes and Errands → `ComingLaterView`
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
            OnboardingView(library: library) { model.completeOnboarding() }
                .frame(minWidth: MFLayout.minimumWindow.width, minHeight: MFLayout.minimumWindow.height)
                .preferredColorScheme(model.appearance.colorScheme)
        } else {
            shell
        }
    }

    @ViewBuilder
    private var shell: some View {
        switch chrome {
        case .inline:
            VStack(spacing: 0) {
                ShellChromeBar(model: model)
                SectionContent(section: model.selection)
            }
            .mfWindowBackground()
            .preferredColorScheme(model.appearance.colorScheme)
        case .toolbar:
            SectionContent(section: model.selection)
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
                        ShellTrailingControls(model: model)
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

    var body: some View {
        Group {
            switch section {
            case .chat: ChatView()
            case .mail: MailView()
            case .calendar: CalendarView()
            case .scan: ScanView()
            case .settings: SettingsView()
            case .notes, .errands: ComingLaterView(section: section)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .id(section)
    }
}

/// Menu commands: a Go menu with a shortcut per section (⌘1-⌘6), Settings
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
