import MferenceDesign
import SwiftUI
import UIKit

/// The app's root. Picks the layout from the window width: tabs on iPhone
/// and Duo closed, two panes split at the fold on Duo open. Both read the
/// same `MobileAppModel`, so folding keeps everything where it was.
public struct MobileRootView: View {
    @State private var model: MobileAppModel
    /// The full window width, measured under the safe area, to place the fold.
    @State private var windowWidth: CGFloat = 0

    public init(model: MobileAppModel) {
        _model = State(initialValue: model)
    }

    public var body: some View {
        GeometryReader { geometry in
            let contentMinX = geometry.frame(in: .global).minX
            let fullWidth = windowWidth > 0 ? windowWidth : geometry.size.width
            let fold = FoldGeometry(width: geometry.size.width, foldX: fullWidth / 2 - contentMinX)
            Group {
                if model.isFirstRunPresented {
                    FirstRunView(model: model)
                } else if FoldGeometry(width: fullWidth).mode == .twoPane {
                    TwoPaneShell(model: model, fold: fold)
                } else {
                    CompactShell(model: model)
                }
            }
        }
        .background(
            GeometryReader { full in
                Color.clear
                    .onAppear { windowWidth = full.size.width }
                    .onChange(of: full.size.width) { windowWidth = full.size.width }
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

/// iPhone and Duo closed: the system's floating tab bar, selected tab in ink.
struct CompactShell: View {
    @Bindable var model: MobileAppModel

    var body: some View {
        TabView(selection: $model.selection) {
            Tab(MobileSection.chat.title, systemImage: MobileSection.chat.systemImage, value: MobileSection.chat) {
                ChatScreen(model: model)
            }
            Tab(MobileSection.calendar.title, systemImage: MobileSection.calendar.systemImage, value: MobileSection.calendar) {
                CalendarScreen(model: model)
            }
            Tab(MobileSection.scan.title, systemImage: MobileSection.scan.systemImage, value: MobileSection.scan) {
                ScanScreen(model: model)
            }
            Tab(MobileSection.settings.title, systemImage: MobileSection.settings.systemImage, value: MobileSection.settings) {
                SettingsScreen(model: model)
            }
        }
        .tabBarMinimizeBehavior(.onScrollDown)
    }
}

/// Duo open: where you are on the left of the fold, what you are looking at
/// on the right. Nothing sits within the keep-out around the fold.
struct TwoPaneShell: View {
    @Bindable var model: MobileAppModel
    let fold: FoldGeometry
    @State private var settingsGroup: SettingsGroup = .models
    @State private var isAddingEvent = false

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
        .sheet(isPresented: $isAddingEvent) {
            NewEventSheet(model: model).presentationDetents([.medium])
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
                        MonthGridView(model: model)
                        TodoListView(model: model)
                    }
                    .padding(.bottom, MFSpacing.l)
                }
                .scrollIndicators(.hidden)
            case .scan:
                ScanCameraView(model: model)
                    .clipShape(RoundedRectangle(cornerRadius: MFRadius.l))
                    .padding(.bottom, MFSpacing.s)
            case .settings:
                VStack(alignment: .leading, spacing: 2) {
                    Text("Settings").font(MFPhoneFont.title).foregroundStyle(MFColor.ink)
                        .padding(.horizontal, MFSpacing.xs)
                        .padding(.bottom, MFSpacing.s)
                    ForEach(SettingsGroup.allCases) { group in
                        Button {
                            settingsGroup = group
                        } label: {
                            Text(group.rawValue)
                                .font(group == settingsGroup ? MFPhoneFont.button : MFPhoneFont.callout)
                                .foregroundStyle(MFColor.ink)
                                .padding(.horizontal, MFSpacing.m)
                                .frame(maxWidth: .infinity, minHeight: MFPhoneMetrics.touchTarget, alignment: .leading)
                                .background(RoundedRectangle(cornerRadius: MFRadius.m)
                                    .fill(group == settingsGroup ? MFColor.sidebarSelected : .clear))
                                .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .accessibilityAddTraits(group == settingsGroup ? .isSelected : [])
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
                CalendarHeader(model: model,
                               title: model.selectedDay.formatted(.dateTime.weekday(.wide).month(.abbreviated).day()),
                               isAddingEvent: $isAddingEvent, titleFont: MFPhoneFont.title)
                    .padding(.horizontal, MFPhoneMetrics.paneMargin)
                    .padding(.top, MFSpacing.s)
                DayTimeline(model: model, margin: MFPhoneMetrics.paneMargin)
                    .safeAreaInset(edge: .bottom) {
                        if model.suggestion != nil {
                            SuggestionBanner(model: model)
                                .padding(.horizontal, MFSpacing.m)
                                .padding(.bottom, MFSpacing.s)
                        }
                    }
            }
        case .scan:
            ScrollView {
                if model.scanPhase == .viewfinder {
                    Text("Point the camera at a receipt, a document, a card or a whiteboard. The fields appear here.")
                        .font(MFPhoneFont.callout)
                        .foregroundStyle(MFColor.secondary)
                        .padding(.top, MFSpacing.xxl)
                } else {
                    ScanFieldsView(model: model, compact: false)
                        .padding(.top, MFSpacing.m)
                }
            }
            .padding(.horizontal, MFPhoneMetrics.paneMargin)
        case .settings:
            ScrollView {
                VStack(alignment: .leading, spacing: MFSpacing.m) {
                    Text(settingsGroup.rawValue).font(MFPhoneFont.title).foregroundStyle(MFColor.ink)
                    if settingsGroup == .models {
                        Text("This iPhone: [model], [8 GB] memory, [96 GB] free. One model is loaded at a time.")
                            .font(MFPhoneFont.preview).foregroundStyle(MFColor.secondary)
                    }
                    SettingsGroupContent(group: settingsGroup, model: model)
                }
                .padding(.horizontal, MFPhoneMetrics.paneMargin)
                .padding(.vertical, MFSpacing.m)
            }
        }
    }
}
