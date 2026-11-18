import MferenceAppCore
import MferenceDesign
import SwiftUI

/// The app mark and name at the left of the chrome.
struct ShellBrand: View {
    var body: some View {
        HStack(spacing: MFSpacing.s) {
            Text("M")
                .font(MFFont.calloutEmphasis)
                .fontWeight(.bold)
                .foregroundStyle(MFColor.secondary)
                .frame(width: ShellMetrics.markSide, height: ShellMetrics.markSide)
                .background(
                    RoundedRectangle(cornerRadius: ShellMetrics.markRadius)
                        .fill(LinearGradient(colors: [MFColor.surface, MFColor.hairlineStrong],
                                             startPoint: .topLeading, endPoint: .bottomTrailing)))
                .overlay(RoundedRectangle(cornerRadius: ShellMetrics.markRadius).strokeBorder(MFColor.hairlineStrong))
            Text("Mference")
                .font(MFFont.bodyEmphasis)
                .fontWeight(.semibold)
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

    var body: some View {
        HStack(spacing: MFSpacing.m) {
            MFChip("On this Mac", systemImage: "lock", tone: .success)
                .help("Everything runs on this Mac. No account, nothing sent anywhere.")

            Button {
                model.isSearchPresented.toggle()
            } label: {
                HStack(spacing: MFSpacing.s - 2) {
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
                model.show(model.selection == .settings ? .chat : .settings)
            } label: {
                HStack(spacing: MFSpacing.s - 2) {
                    Image(systemName: AppSection.settings.systemImage)
                        .accessibilityHidden(true)
                    if model.selection == .settings {
                        Text(AppSection.settings.title)
                    }
                }
            }
            .buttonStyle(MFPillButtonStyle(isSelected: model.selection == .settings))
            .accessibilityLabel("Settings")
            .accessibilityAddTraits(model.selection == .settings ? .isSelected : [])
            .help("Settings (⌘,)")
        }
    }
}

/// The whole chrome as one inline bar, as drawn in the mockup. Snapshots
/// render this; the app window puts the same pieces in its toolbar.
struct ShellChromeBar: View {
    @Bindable var model: ShellModel
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
            ShellTrailingControls(model: model)
        }
        .padding(.horizontal, MFSpacing.m + 2)
        .padding(.vertical, MFSpacing.m - 2)
        .overlay(alignment: .bottom) {
            Rectangle().fill(MFColor.hairline).frame(height: 1)
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
        .padding(.trailing, MFSpacing.xs + 2)
        .accessibilityHidden(true)
    }
}
