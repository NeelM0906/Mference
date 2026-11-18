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
}
