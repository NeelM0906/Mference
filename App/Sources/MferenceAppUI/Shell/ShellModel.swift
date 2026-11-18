import MferenceAppCore
import Observation
import SwiftUI

/// The shell's state: which section is showing, the search panel, and the
/// appearance choice. The app creates one and shares it with the menu
/// commands, so keyboard shortcuts and the pill bar drive the same state.
@Observable
public final class ShellModel {
    /// Light, dark, or follow the system.
    public enum Appearance: String, CaseIterable, Sendable {
        case system, light, dark

        public var colorScheme: ColorScheme? {
            switch self {
            case .system: nil
            case .light: .light
            case .dark: .dark
            }
        }

        public var next: Appearance {
            switch self {
            case .system: .light
            case .light: .dark
            case .dark: .system
            }
        }

        public var title: String {
            switch self {
            case .system: "Match the system"
            case .light: "Light"
            case .dark: "Dark"
            }
        }

        public var systemImage: String {
            switch self {
            case .system: "circle.lefthalf.filled"
            case .light: "sun.max"
            case .dark: "moon"
            }
        }
    }

    public var selection: AppSection
    public var isSearchPresented = false
    public private(set) var appearance: Appearance

    @ObservationIgnored private let defaults: UserDefaults?
    static let appearanceKey = "shell.appearance"

    /// - Parameter defaults: where the appearance choice persists; nil keeps it in memory (tests).
    public init(selection: AppSection = .chat, defaults: UserDefaults? = .standard) {
        self.selection = selection
        self.defaults = defaults
        appearance = defaults?.string(forKey: Self.appearanceKey).flatMap(Appearance.init(rawValue:)) ?? .system
    }

    /// Sets and remembers the appearance.
    public func setAppearance(_ appearance: Appearance) {
        self.appearance = appearance
        defaults?.set(appearance.rawValue, forKey: Self.appearanceKey)
    }

    public func show(_ section: AppSection) {
        selection = section
        isSearchPresented = false
    }

    /// The keyboard shortcut for a section: ⌘1-⌘6 for the pills, ⌘, for Settings.
    public static func shortcut(for section: AppSection) -> KeyEquivalent {
        if let index = AppSection.pills.firstIndex(of: section) {
            return KeyEquivalent(Character(String(index + 1)))
        }
        return ","
    }
}
