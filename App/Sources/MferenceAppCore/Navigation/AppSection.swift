import Foundation

// SHARED CONTRACT: the top pill bar (spec section 6, mockup chrome). U1 owns
// the shell that renders it; each team owns its section's screen.

/// A top-level screen in the pill bar, in mockup order, plus Settings.
public enum AppSection: String, Sendable, Hashable, Codable, CaseIterable, Identifiable {
    case chat, mail, calendar, notes, errands, scan, settings

    public var id: String { rawValue }

    /// The sections shown as pills (Settings is the gear at the right).
    public static let pills: [AppSection] = [.chat, .mail, .calendar, .notes, .errands, .scan]

    public var title: String {
        switch self {
        case .chat: "Chat"
        case .mail: "Mail"
        case .calendar: "Calendar"
        case .notes: "Notes"
        case .errands: "Errands"
        case .scan: "Scan"
        case .settings: "Settings"
        }
    }

    /// SF Symbol for the pill.
    public var systemImage: String {
        switch self {
        case .chat: "bubble.left"
        case .mail: "envelope"
        case .calendar: "calendar"
        case .notes: "note.text"
        case .errands: "globe"
        case .scan: "viewfinder"
        case .settings: "gearshape"
        }
    }

    /// Sections shown with a "Coming later" screen (P7). Notes left this
    /// list with the Notes program (spec v1.0, D10).
    public var isComingLater: Bool { self == .errands }
}
