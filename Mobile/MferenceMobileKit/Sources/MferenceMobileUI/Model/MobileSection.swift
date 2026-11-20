import CoreGraphics
import MferenceDesign

/// The phone's sections (locked: Chat, Calendar, Scan, Settings). Mail, Notes
/// and Errands stay off the phone until they run there.
public enum MobileSection: String, CaseIterable, Identifiable, Sendable {
    case chat, calendar, scan, settings

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .chat: "Chat"
        case .calendar: "Calendar"
        case .scan: "Scan"
        case .settings: "Settings"
        }
    }

    /// The same SF Symbols as the Mac's `AppSection`.
    public var systemImage: String {
        switch self {
        case .chat: "bubble.left"
        case .calendar: "calendar"
        case .scan: "viewfinder"
        case .settings: "gearshape"
        }
    }
}

/// How the window is laid out: one column (iPhone, Duo closed) or two panes
/// split at the fold (Duo open).
public enum MobileLayoutMode: Equatable, Sendable {
    case compact, twoPane
}

/// Where the panes go for a window of a given width. On the Duo's inner
/// screen (951 pt wide) the fold is the vertical midline; nothing sits within
/// `MFPhoneMetrics.foldKeepOut` of it.
public struct FoldGeometry: Equatable, Sendable {
    public let width: CGFloat
    /// The fold's x position in this view's coordinates.
    public let foldX: CGFloat

    /// - Parameters:
    ///   - width: the content width.
    ///   - foldX: where the fold is; defaults to the midline (the fold of a
    ///     full-screen Duo window whose safe area is symmetric).
    public init(width: CGFloat, foldX: CGFloat? = nil) {
        self.width = width
        self.foldX = foldX ?? width / 2
    }

    public var mode: MobileLayoutMode {
        width >= MFPhoneMetrics.twoPaneMinWidth ? .twoPane : .compact
    }

    /// The leading pane runs from 0 to the keep-out before the fold.
    public var leadingPaneWidth: CGFloat { max(0, foldX - MFPhoneMetrics.foldKeepOut) }

    /// The trailing pane starts after the keep-out.
    public var trailingPaneX: CGFloat { foldX + MFPhoneMetrics.foldKeepOut }
    public var trailingPaneWidth: CGFloat { max(0, width - trailingPaneX) }
}
