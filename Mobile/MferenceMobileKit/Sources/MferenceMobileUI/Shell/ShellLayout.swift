import CoreGraphics
import MferenceDesign
import Observation

/// Layout sizes the shell keeps for itself (decision D4).
enum ShellMetrics {
    /// A turned Duo's one column is at most this wide.
    static let turnedColumnMaxWidth: CGFloat = 560
    /// Portrait windows at least this wide are a turned Duo (669 pt). Every
    /// iPhone (at most 440) and the closed Duo (466) are narrower.
    static let turnedMinWidth: CGFloat = 600
}

/// How the shell lays out a window, decided from the full window size.
enum ShellLayout: Equatable {
    /// iPhone and Duo closed: tabs, full width.
    case tabs
    /// Duo open, turned (669 × 951): tabs, one centred column, and the fold
    /// running across the window.
    case turned(TurnedFold)
    /// Duo open (951 × 669): two panes split at the fold.
    case twoPane

    init(windowSize size: CGSize) {
        if FoldGeometry(width: size.width).mode == .twoPane {
            self = .twoPane
        } else if size.width >= ShellMetrics.turnedMinWidth, size.height > size.width {
            self = .turned(TurnedFold(windowSize: size))
        } else {
            self = .tabs
        }
    }

    /// The turned column's width, or nil for full-width tabs and two panes.
    var columnWidth: CGFloat? {
        if case .turned(let turned) = self { turned.columnWidth } else { nil }
    }

    /// Whether sheets must stay below a fold that runs across the window.
    var sheetsBelowFold: Bool {
        if case .turned = self { true } else { false }
    }
}

/// A turned Duo: the fold is a horizontal line across the middle of the
/// window. Content is one column; sheets and the composer stay below the
/// fold line.
struct TurnedFold: Equatable {
    /// The column's width: at most `ShellMetrics.turnedColumnMaxWidth`, and
    /// never closer than the screen margin to either edge.
    let columnWidth: CGFloat
    /// The fold's y position in window coordinates (the midline).
    let foldY: CGFloat

    init(windowSize size: CGSize) {
        columnWidth = min(ShellMetrics.turnedColumnMaxWidth,
                          max(0, size.width - 2 * MFPhoneMetrics.screenMargin))
        foldY = size.height / 2
    }
}

/// Shell state that has to outlive a change of layout (folding, unfolding
/// or turning the Duo), so it lives above the shells, next to the app model.
/// The root puts it in the environment (`@Environment(ShellState.self)`) for
/// screens whose sheets must survive turning the Duo.
@Observable
final class ShellState {
    /// The Settings group shown in the right pane on Duo open.
    var settingsGroup: SettingsGroup = .models
    /// The new-event sheet the shell presents (Duo open's calendar header).
    /// Presented from the root, so turning the Duo keeps it open.
    var isAddingEvent = false
}
