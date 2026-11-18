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
    /// First run finished (or skipped because no model library is wired yet).
    public private(set) var hasCompletedOnboarding: Bool

    @ObservationIgnored private let defaults: UserDefaults?
    static let appearanceKey = "shell.appearance"
    static let onboardingKey = "shell.onboardingCompleted"

    /// - Parameter defaults: where the appearance choice persists; nil keeps it in memory (tests).
    public init(selection: AppSection = .chat, defaults: UserDefaults? = .standard) {
        self.selection = selection
        self.defaults = defaults
        appearance = defaults?.string(forKey: Self.appearanceKey).flatMap(Appearance.init(rawValue:)) ?? .system
        hasCompletedOnboarding = defaults?.bool(forKey: Self.onboardingKey) ?? false
    }

    /// First run shows until it is completed, and only once a model library
    /// is wired (it has nothing to install without one).
    public func needsOnboarding(hasLibrary: Bool) -> Bool {
        hasLibrary && !hasCompletedOnboarding
    }

    /// Marks first run as done and remembers it.
    public func completeOnboarding() {
        hasCompletedOnboarding = true
        defaults?.set(true, forKey: Self.onboardingKey)
    }

    /// Sets and remembers the appearance.
    public func setAppearance(_ appearance: Appearance) {
        self.appearance = appearance
        defaults?.set(appearance.rawValue, forKey: Self.appearanceKey)
    }

    /// Where Settings opens: the section to scroll to, and whether the model
    /// library shows. Reset to Models whenever Settings opens another way.
    public private(set) var settingsPane: SettingsView.Pane = .models
    public private(set) var settingsShowsLibrary = false

    /// Opens Settings at a section (chat's Memory "Review") or on the model
    /// library (the picker's "Manage models").
    public func openSettings(_ pane: SettingsView.Pane, showingLibrary: Bool = false) {
        settingsPane = pane
        settingsShowsLibrary = showingLibrary
        show(.settings)
    }

    /// The section to return to when Settings closes.
    public private(set) var sectionBeforeSettings: AppSection = .chat

    public func show(_ section: AppSection) {
        if section == .settings, selection != .settings {
            sectionBeforeSettings = selection
        }
        selection = section
        isSearchPresented = false
    }

    /// The Settings gear: opens Settings, or closes it back to the section
    /// that was showing before.
    public func toggleSettings() {
        if selection != .settings {
            settingsPane = .models
            settingsShowsLibrary = false
        }
        show(selection == .settings ? sectionBeforeSettings : .settings)
    }

    /// The keyboard shortcut for a section: ⌘1-⌘6 for the pills, ⌘, for Settings.
    public static func shortcut(for section: AppSection) -> KeyEquivalent {
        if let index = AppSection.pills.firstIndex(of: section) {
            return KeyEquivalent(Character(String(index + 1)))
        }
        return ","
    }
}
