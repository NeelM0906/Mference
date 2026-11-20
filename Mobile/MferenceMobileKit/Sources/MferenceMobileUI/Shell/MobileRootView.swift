import MferenceDesign
import SwiftUI
import UIKit

/// The app's root. Picks the layout from the window size (`ShellLayout`):
/// tabs on iPhone and Duo closed, tabs in one centred column on a turned Duo,
/// two panes split at the fold on Duo open. All of them read the same
/// `MobileAppModel` and `ShellState`, held here, so folding, unfolding or
/// turning keeps everything where it was.
public struct MobileRootView: View {
    @State private var model: MobileAppModel
    @State private var shell = ShellState()
    /// The full window size, measured under the safe area, to place the fold.
    @State private var windowSize: CGSize = .zero

    public init(model: MobileAppModel) {
        _model = State(initialValue: model)
    }

    public var body: some View {
        GeometryReader { geometry in
            let contentMinX = geometry.frame(in: .global).minX
            let full = windowSize
            let layout = ShellLayout(windowSize: full)
            Group {
                if full.width <= 0 {
                    // Until the window is measured: deciding from the safe-area
                    // size could pick the wrong shell and rebuild it a frame later.
                    Color.clear
                } else if model.isFirstRunPresented {
                    FirstRunView(model: model)
                } else if layout == .twoPane {
                    TwoPaneShell(model: model, shell: shell,
                                 fold: FoldGeometry(width: geometry.size.width,
                                                    foldX: full.width / 2 - contentMinX))
                } else {
                    // One branch for tabs and the turned column, so moving
                    // between them keeps CompactShell's identity and every
                    // screen's state (navigation, drafts, open sheets).
                    CompactShell(model: model, columnWidth: layout.columnWidth)
                }
            }
            .sheet(isPresented: $shell.isAddingEvent) {
                NewEventSheet(model: model.calendar).phoneSheetDetents([.medium])
            }
            .environment(\.phoneSheetsBelowFold, layout.sheetsBelowFold)
            .environment(shell)
        }
        .background(
            GeometryReader { full in
                Color.clear
                    .onAppear { windowSize = full.size }
                    .onChange(of: full.size) { windowSize = full.size }
            }
            .ignoresSafeArea())
        .background(MFColor.canvas.ignoresSafeArea())
        .tint(MFColor.ink)
    }

    /// Registers Geist and sets the navigation bar titles in it. Call once at
    /// launch, before the first view renders.
    public static func configureAppearance() {
        MFFontRegistry.registerBundledFonts()
        func geist(_ name: String, _ size: CGFloat, _ style: UIFont.TextStyle) -> UIFont? {
            UIFont(name: name, size: size).map { UIFontMetrics(forTextStyle: style).scaledFont(for: $0) }
        }
        let bar = UINavigationBar.appearance()
        if let large = geist("Geist-SemiBold", 30, .largeTitle) {
            bar.largeTitleTextAttributes = [.font: large]
        }
        if let inline = geist("Geist-SemiBold", 17, .headline) {
            bar.titleTextAttributes = [.font: inline]
        }
    }
}

/// iPhone, Duo closed and a turned Duo: the system's floating tab bar,
/// selected tab in ink. On a turned Duo each screen sits in one centred
/// column `columnWidth` wide; the tab bar keeps the full width.
struct CompactShell: View {
    @Bindable var model: MobileAppModel
    /// nil: screens take the full width.
    var columnWidth: CGFloat?

    var body: some View {
        TabView(selection: $model.selection) {
            Tab(MobileSection.chat.title, systemImage: MobileSection.chat.systemImage, value: MobileSection.chat) {
                column { ChatScreen(model: model) }
            }
            Tab(MobileSection.calendar.title, systemImage: MobileSection.calendar.systemImage, value: MobileSection.calendar) {
                column { CalendarScreen(model: model.calendar) }
            }
            Tab(MobileSection.scan.title, systemImage: MobileSection.scan.systemImage, value: MobileSection.scan) {
                column { ScanScreen(model: model.scan) }
            }
            Tab(MobileSection.settings.title, systemImage: MobileSection.settings.systemImage, value: MobileSection.settings) {
                column { SettingsScreen(model: model) }
            }
        }
        .tabBarMinimizeBehavior(.onScrollDown)
    }

    /// The same structure with or without a column, so turning the Duo
    /// keeps each screen's identity (scroll position, navigation).
    private func column<Screen: View>(@ViewBuilder _ screen: () -> Screen) -> some View {
        screen()
            .frame(maxWidth: columnWidth ?? .infinity)
            .frame(maxWidth: .infinity)
            .background(MFColor.canvas.ignoresSafeArea())
    }
}

/// Duo open: where you are on the left of the fold, what you are looking at
/// on the right. Nothing sits within the keep-out around the fold.
struct TwoPaneShell: View {
    @Bindable var model: MobileAppModel
    @Bindable var shell: ShellState
    let fold: FoldGeometry

    private var keepOut: CGFloat { MFPhoneMetrics.foldKeepOut }

    var body: some View {
        HStack(spacing: 0) {
            leading
                .padding(.leading, MFPhoneMetrics.screenMargin)
                .padding(.trailing, keepOut + MFSpacing.xs)
                .frame(width: max(0, fold.foldX))
                .frame(maxHeight: .infinity, alignment: .top)
                .background((model.selection == .scan ? ScanCameraView.darkGround : MFColor.sidebar)
                    .ignoresSafeArea())
            trailing
                .padding(.leading, keepOut)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
                .background(MFColor.canvas.ignoresSafeArea())
        }
    }

    @ViewBuilder private var leading: some View {
        VStack(alignment: .leading, spacing: MFSpacing.m) {
            SectionPillBar(selection: $model.selection, onDark: model.selection == .scan)
                .padding(.top, MFSpacing.s)
            switch model.selection {
            case .chat:
                ScrollView {
                    ChatsListContent(model: model, style: .sidebar)
                        .padding(.bottom, MFSpacing.l)
                }
                .scrollIndicators(.hidden)
            case .calendar:
                ScrollView {
                    VStack(alignment: .leading, spacing: MFSpacing.l) {
                        MonthGridView(model: model.calendar)
                        TodoListView(model: model.calendar)
                    }
                    .padding(.bottom, MFSpacing.l)
                }
                .scrollIndicators(.hidden)
            case .scan:
                ScanCameraView(model: model.scan)
                    .clipShape(RoundedRectangle(cornerRadius: MFRadius.l))
                    .padding(.bottom, MFSpacing.s)
            case .settings:
                VStack(alignment: .leading, spacing: 2) {
                    Text("Settings").font(MFPhoneFont.title).foregroundStyle(MFColor.ink)
                        .padding(.horizontal, MFSpacing.xs)
                        .padding(.bottom, MFSpacing.s)
                    ForEach(SettingsGroup.allCases) { group in
                        Button {
                            shell.settingsGroup = group
                        } label: {
                            Text(group.rawValue)
                                .font(group == shell.settingsGroup ? MFPhoneFont.button : MFPhoneFont.callout)
                                .foregroundStyle(MFColor.ink)
                                .padding(.horizontal, MFSpacing.m)
                                .frame(maxWidth: .infinity, minHeight: MFPhoneMetrics.touchTarget, alignment: .leading)
                                .background(RoundedRectangle(cornerRadius: MFRadius.m)
                                    .fill(group == shell.settingsGroup ? MFColor.sidebarSelected : .clear))
                                .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .accessibilityAddTraits(group == shell.settingsGroup ? .isSelected : [])
                    }
                    Spacer()
                    Label("No account and no usage reporting.", systemImage: "lock")
                        .font(MFPhoneFont.caption)
                        .foregroundStyle(MFColor.secondary)
                        .padding(.horizontal, MFSpacing.xs)
                        .padding(.bottom, MFSpacing.m)
                }
            }
        }
    }

    @ViewBuilder private var trailing: some View {
        switch model.selection {
        case .chat:
            VStack(spacing: 0) {
                HStack(spacing: MFSpacing.s) {
                    Text(model.chatTitle)
                        .font(MFPhoneFont.headline)
                        .foregroundStyle(MFColor.ink)
                        .lineLimit(1)
                    Spacer()
                    PhoneChip(title: "On this iPhone", systemImage: "lock", tone: .success)
                }
                .frame(minHeight: MFPhoneMetrics.touchTarget)
                .padding(.horizontal, MFPhoneMetrics.paneMargin)
                .padding(.top, MFSpacing.s)
                ChatConversationView(model: model, margin: MFPhoneMetrics.paneMargin)
            }
        case .calendar:
            VStack(spacing: 0) {
                CalendarHeader(model: model.calendar,
                               title: model.calendar.selectedDay.formatted(.dateTime.weekday(.wide).month(.abbreviated).day()),
                               isAddingEvent: $shell.isAddingEvent, titleFont: MFPhoneFont.title)
                    .padding(.horizontal, MFPhoneMetrics.paneMargin)
                    .padding(.top, MFSpacing.s)
                DayTimeline(model: model.calendar, margin: MFPhoneMetrics.paneMargin)
                    .safeAreaInset(edge: .bottom) {
                        if model.calendar.suggestion != nil {
                            SuggestionBanner(model: model.calendar)
                                .padding(.horizontal, MFSpacing.m)
                                .padding(.bottom, MFSpacing.s)
                        }
                    }
            }
        case .scan:
            ScrollView {
                if model.scan.scanPhase == .viewfinder {
                    Text("Point the camera at a receipt, a document, a card or a whiteboard. The fields appear here.")
                        .font(MFPhoneFont.callout)
                        .foregroundStyle(MFColor.secondary)
                        .padding(.top, MFSpacing.xxl)
                } else {
                    ScanFieldsView(model: model.scan, compact: false)
                        .padding(.top, MFSpacing.m)
                }
            }
            .padding(.horizontal, MFPhoneMetrics.paneMargin)
        case .settings:
            ScrollView {
                VStack(alignment: .leading, spacing: MFSpacing.m) {
                    Text(shell.settingsGroup.rawValue).font(MFPhoneFont.title).foregroundStyle(MFColor.ink)
                    if shell.settingsGroup == .models {
                        Text("This iPhone: [model], [8 GB] memory, [96 GB] free. One model is loaded at a time.")
                            .font(MFPhoneFont.preview).foregroundStyle(MFColor.secondary)
                    }
                    SettingsGroupContent(group: shell.settingsGroup, model: model)
                }
                .padding(.horizontal, MFPhoneMetrics.paneMargin)
                .padding(.vertical, MFSpacing.m)
            }
        }
    }
}
