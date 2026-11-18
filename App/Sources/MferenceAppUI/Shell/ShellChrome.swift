import MferenceAppCore
import MferenceDesign
import SwiftUI

/// The app mark and name at the left of the chrome.
struct ShellBrand: View {
    var body: some View {
        HStack(spacing: MFSpacing.s) {
            Text("M")
                .font(MFFont.calloutSemibold)
                .foregroundStyle(MFColor.secondary)
                .frame(width: ShellMetrics.markSide, height: ShellMetrics.markSide)
                .background(
                    RoundedRectangle(cornerRadius: ShellMetrics.markRadius)
                        .fill(LinearGradient(colors: [MFColor.surface, MFColor.hairlineStrong],
                                             startPoint: .topLeading, endPoint: .bottomTrailing)))
                .overlay(RoundedRectangle(cornerRadius: ShellMetrics.markRadius).strokeBorder(MFColor.hairlineStrong))
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

    var body: some View {
        HStack(spacing: MFSpacing.xxs) {
            ForEach(AppSection.pills) { section in
                MFPill(section.title, systemImage: section.systemImage,
                       isSelected: model.selection == section) {
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

    var body: some View {
        HStack(spacing: MFSpacing.m) {
            ShellToolbarItems(items: items)
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
            .popover(isPresented: $model.isSearchPresented, arrowEdge: .bottom) {
                ShellSearchPanel(model: model)
            }

            Button {
                model.setAppearance(model.appearance.next)
            } label: {
                Image(systemName: model.appearance.systemImage)
                    .frame(width: ShellMetrics.iconButtonSide, height: ShellMetrics.iconButtonSide)
            }
            .buttonStyle(MFPillButtonStyle())
            .accessibilityLabel("Appearance: \(model.appearance.title)")
            .help("Appearance: \(model.appearance.title)")

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
