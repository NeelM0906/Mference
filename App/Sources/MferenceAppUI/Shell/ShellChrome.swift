import MferenceAppCore
import MferenceDesign
import SwiftUI

/// The app mark and name at the left of the chrome.
struct ShellBrand: View {
    var body: some View {
        HStack(spacing: MFSpacing.s) {
            AppMark(side: ShellMetrics.markSide, radius: ShellMetrics.markRadius, letterFont: MFFont.calloutSemibold)
            Text("Mference")
                .font(MFFont.brand)
                .foregroundStyle(MFColor.ink)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Mference")
    }
}

/// The section pills (Chat, Mail, Calendar, Notes, Errands, Scan).
struct ShellPillBar: View {
    @Bindable var model: ShellModel
    /// Icons only, except the open section, when the window is too narrow
    /// for every name (`ShellMetrics.usesCompactPills`). Names stay in the
    /// tooltips and the accessibility labels.
    var compact = false

    var body: some View {
        HStack(spacing: MFSpacing.xxs) {
            ForEach(AppSection.pills) { section in
                MFPill(section.title, systemImage: section.systemImage,
                       isSelected: model.selection == section,
                       showsTitle: !compact || model.selection == section) {
                    model.show(section)
                }
                .help("\(section.title) (⌘\(AppSection.pills.firstIndex(of: section).map { $0 + 1 } ?? 0))")
            }
        }
        .padding(ShellMetrics.pillTrackPadding)
        .background(Capsule().fill(MFColor.fill))
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Sections")
    }
}

/// "On this Mac", search, appearance and Settings at the right of the chrome.
struct ShellTrailingControls: View {
    @Bindable var model: ShellModel
    /// Actions the visible screen contributes (`shellToolbar`).
    var items: [ShellToolbarItem] = []
    /// A hidden copy that only measures the bar's width: no popover, and a
    /// blank of `measuredItemsWidth` in place of the screen's actions, whose
    /// views may present sheets of their own and must exist once.
    var isMeasuring = false
    var measuredItemsWidth: CGFloat = 0
    /// Reports the screen actions' width while they show (the real bar).
    var onItemsWidth: ((CGFloat) -> Void)?
    /// Team N3's voice models: the red recording timer chip (D7).
    @Environment(VoiceModels.self) private var voice: VoiceModels?

    /// A screen with its own actions (Scan, Calendar, Mail) gets the bar the
    /// mockup draws for it: its actions and the Settings gear. "On this
    /// Mac", search and appearance step back (⌘K still opens search).
    static func showsGeneralControls(screenItems: [ShellToolbarItem]) -> Bool { screenItems.isEmpty }

    var body: some View {
        if isMeasuring {
            controls
        } else {
            // On the whole group, so ⌘K works even when the search button steps back.
            controls.popover(isPresented: $model.isSearchPresented, arrowEdge: .bottom) {
                ShellSearchPanel(model: model)
            }
        }
    }

    private var controls: some View {
        HStack(spacing: MFSpacing.m) {
            // While a meeting records, on every section: opens the meeting note.
            if let voice, let live = voice.liveMeeting {
                RecordingTimerChip(live: live) {
                    model.show(.notes)
                    voice.openLiveMeeting()
                }
            }
            if isMeasuring {
                if !items.isEmpty {
                    Color.clear.frame(width: measuredItemsWidth, height: 1)
                }
            } else {
                // At their natural width, so a squeezed toolbar never reports
                // a smaller width that would keep the names and hide the
                // overflow.
                ShellToolbarItems(items: items)
                    .fixedSize(horizontal: true, vertical: false)
                    .onGeometryChange(for: CGFloat.self) { $0.size.width } action: { onItemsWidth?($0) }
            }
            if Self.showsGeneralControls(screenItems: items) {
                MFChip("On this Mac", systemImage: "lock", tone: .success)
                    .help("Runs on this Mac. Nothing leaves it except web searches you turn on and model downloads.")

                Button {
                    model.isSearchPresented.toggle()
                } label: {
                    HStack(spacing: MFSpacing.snug) {
                        Image(systemName: "magnifyingglass")
                            .accessibilityHidden(true)
                        MFKeycap("⌘K")
                    }
                }
                .buttonStyle(.mfSecondary)
                .accessibilityLabel("Search everything")

                Button {
                    model.setAppearance(model.appearance.next)
                } label: {
                    Image(systemName: model.appearance.systemImage)
                        .frame(width: ShellMetrics.iconButtonSide, height: ShellMetrics.iconButtonSide)
                }
                .buttonStyle(MFPillButtonStyle())
                .accessibilityLabel("Appearance: \(model.appearance.title)")
                .help("Appearance: \(model.appearance.title)")
            }

            Button {
                model.toggleSettings()
            } label: {
                HStack(spacing: MFSpacing.snug) {
                    Image(systemName: AppSection.settings.systemImage)
                        .accessibilityHidden(true)
                    if model.selection == .settings {
                        Text(AppSection.settings.title)
                    }
                }
            }
            .buttonStyle(MFPillButtonStyle(isSelected: model.selection == .settings))
            .accessibilityLabel(model.selection == .settings ? "Close Settings" : "Settings")
            .help(model.selection == .settings ? "Close Settings" : "Settings (⌘,)")
        }
    }
}

/// The window toolbar's full width with every pill named, laid out hidden
/// behind the content and never shown or clicked, so the shell knows when
/// the names must step back (`ShellMetrics.usesCompactPills`).
struct ShellToolbarMeasure: View {
    @Bindable var model: ShellModel
    let items: [ShellToolbarItem]
    /// The screen actions' last measured width (`ShellMetrics`' estimate
    /// until they have shown once).
    let itemsWidth: CGFloat
    let onWidth: (CGFloat) -> Void

    var body: some View {
        HStack(spacing: MFSpacing.m) {
            ShellBrand()
            ShellPillBar(model: model)
            ShellTrailingControls(model: model, items: items, isMeasuring: true, measuredItemsWidth: itemsWidth)
        }
        .fixedSize()
        .onGeometryChange(for: CGFloat.self) { $0.size.width } action: { onWidth($0) }
        .hidden()
        .disabled(true)
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }
}

/// The whole chrome as one inline bar, as drawn in the mockup. Snapshots
/// render this; the app window puts the same pieces in its toolbar.
struct ShellChromeBar: View {
    @Bindable var model: ShellModel
    var items: [ShellToolbarItem] = []
    /// Leading room for the window buttons (traffic lights) in snapshots.
    var showsWindowButtons = true

    var body: some View {
        HStack(spacing: MFSpacing.m) {
            if showsWindowButtons {
                ShellWindowButtonsPlaceholder()
            }
            ShellBrand()
            ShellPillBar(model: model)
            Spacer(minLength: MFSpacing.l)
            ShellTrailingControls(model: model, items: items)
        }
        .padding(.horizontal, MFSpacing.comfortable)
        .frame(height: ShellMetrics.chromeHeight)
        .overlay(alignment: .bottom) {
            Rectangle().fill(MFColor.chromeDivider).frame(height: MFLayout.hairline)
        }
    }
}

/// Snapshot stand-in for the system's close/minimise/zoom buttons, so a
/// rendered board lines up with the mockup. The real window draws its own.
struct ShellWindowButtonsPlaceholder: View {
    var body: some View {
        HStack(spacing: ShellMetrics.windowButtonSpacing) {
            Circle().fill(Color(nsColor: .systemRed))
            Circle().fill(Color(nsColor: .systemYellow))
            Circle().fill(Color(nsColor: .systemGreen))
        }
        .frame(width: ShellMetrics.windowButtonsSize.width, height: ShellMetrics.windowButtonsSize.height)
        .padding(.trailing, MFSpacing.snug)
        .accessibilityHidden(true)
    }
}
