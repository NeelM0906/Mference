import SwiftUI

/// Every component on one board, for previews and the light/dark snapshot
/// check (`DesignComponentTests`). Not shown in the app.
public struct MFComponentGallery: View {
    @State private var on = true
    @State private var off = false

    public init() {}

    public var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.l) {
            HStack(spacing: MFSpacing.xxs) {
                MFPill("Chat", systemImage: "bubble.left", isSelected: true) {}
                MFPill("Mail", systemImage: "envelope", isSelected: false) {}
                MFPill("Calendar", systemImage: "calendar", isSelected: false) {}
            }
            .padding(MFMetrics.pillTrackPadding)
            .background(Capsule().fill(MFColor.fill))
            HStack(spacing: MFSpacing.s) {
                MFChip("On this Mac", systemImage: "lock", tone: .success)
                MFBadge(.loaded)
                MFBadge(.ready)
                MFBadge(.preview)
                MFBadge(.later)
                MFBadge(.installed)
                MFBadge(.custom("Built in", .neutral))
                MFProgressBadge("Downloading", fraction: 0.62)
            }
            HStack(spacing: MFSpacing.s) {
                Button("Details") {}.buttonStyle(.mfSecondary)
                Button("Connect") {}.buttonStyle(.mfPrimary)
                Button("Track the price") {}.buttonStyle(.mfQuiet)
                Toggle("On", isOn: $on).toggleStyle(.mf)
                Toggle("Off", isOn: $off).toggleStyle(.mf)
                MFKeycap("⌘K")
                MFCitationMark(2)
                MFSourceChip(number: 1, title: "example.org") {}
            }
            HStack(spacing: MFSpacing.s) {
                Text("Disabled").font(MFFont.caption).foregroundStyle(MFColor.tertiary)
                Button("Details") {}.buttonStyle(.mfSecondary)
                Button("Connect") {}.buttonStyle(.mfPrimary)
                Button("Track the price") {}.buttonStyle(.mfQuiet)
                MFPill("Mail", systemImage: "envelope", isSelected: false) {}
                Toggle("On", isOn: .constant(true)).toggleStyle(.mf)
                Toggle("Off", isOn: .constant(false)).toggleStyle(.mf)
            }
            .disabled(true)
            MFSectionHeader("Today")
            MFCard {
                MFListRow("Quick", subtitle: "Default for chat, mail, web answers and scanning.") {
                    MFBadge(.custom("Installed · default", .success))
                    Button("Details") {}.buttonStyle(.mfSecondary)
                }
                MFListRow("Offline mode", subtitle: "Turns off web search, mail sync and errands.") {
                    Toggle("Offline mode", isOn: $off).toggleStyle(.mf)
                }
                VStack(alignment: .leading) {
                    MFProgressBar(0.62, label: "Downloading")
                }
                .padding(MFSpacing.l)
            }
            MFApprovalCard(eyebrow: "Approve to book", title: "A table for four, Friday 7:30 PM",
                           detail: "Free to book. Nothing is sent until you approve.") {
                Button("Approve") {}.buttonStyle(.mfPrimary)
                Button("Not now") {}.buttonStyle(.mfSecondary)
            }
        }
        .padding(MFSpacing.xl)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .mfWindowBackground()
    }
}
