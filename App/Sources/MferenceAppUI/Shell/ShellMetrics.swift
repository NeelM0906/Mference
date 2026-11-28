import CoreGraphics

/// Shell sizes from the mockup chrome (Main.dc.html) that are not shared
/// tokens.
enum ShellMetrics {
    /// The app mark square and its corner radius.
    static let markSide: CGFloat = 24
    static let markRadius: CGFloat = 7
    /// The inline chrome bar (snapshots): 10 + 36 + 10, plus its hairline.
    static let chromeHeight: CGFloat = 57
    /// Padding inside the grey pill track.
    static let pillTrackPadding: CGFloat = 3
    /// Snapshot stand-in for the traffic lights.
    static let windowButtonsSize = CGSize(width: 50, height: 12)
    static let windowButtonSpacing: CGFloat = 7
    /// Icon-only toolbar buttons.
    static let iconButtonSide: CGFloat = 16
    /// The ⌘K panel.
    static let searchPanelWidth: CGFloat = 320
    static let searchRowIconWidth: CGFloat = 18
    /// The Coming later column.
    static let comingLaterWidth: CGFloat = 520
    /// Window width the toolbar takes beyond its items: the window buttons,
    /// the toolbar's edge insets and the gap between the left and right
    /// groups.
    static let toolbarChromeAllowance: CGFloat = 160
    /// A screen's actions before they have shown once (Notes' three, the
    /// widest, are about 340).
    static let unmeasuredScreenActionsWidth: CGFloat = 360

    /// Section pills show icons only (the open one keeps its name) when the
    /// bar with every name would not fit the window. Otherwise macOS folds
    /// the right-hand controls (screen actions, Settings) into its » menu,
    /// which cannot show them. Zero widths (not measured yet) keep the names.
    static func usesCompactPills(windowWidth: CGFloat, fullBarWidth: CGFloat) -> Bool {
        guard windowWidth > 0, fullBarWidth > 0 else { return false }
        return windowWidth < fullBarWidth + toolbarChromeAllowance
    }
}
