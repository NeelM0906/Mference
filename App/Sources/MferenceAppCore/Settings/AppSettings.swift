import Foundation
import Observation
import Synchronization

/// Where settings are kept. `UserDefaults` in the app; an in-memory store in
/// tests and previews, so they never touch the person's preferences.
public protocol SettingsStore: AnyObject {
    func object(forKey key: String) -> Any?
    func set(_ value: Any?, forKey key: String)
}

extension UserDefaults: SettingsStore {}

/// A settings store that lives only as long as the object.
public final class InMemorySettingsStore: SettingsStore, Sendable {
    private let values: Mutex<[String: any Sendable]>

    public init(_ values: [String: any Sendable] = [:]) {
        self.values = Mutex(values)
    }

    public func object(forKey key: String) -> Any? {
        let value: (any Sendable)? = values.withLock { $0[key] }
        return value
    }

    public func set(_ value: Any?, forKey key: String) {
        values.withLock { stored in
            switch value {
            case let value as Bool: stored[key] = value
            case let value as Int: stored[key] = value
            case let value as String: stored[key] = value
            case let value as Double: stored[key] = value
            default: stored[key] = nil
            }
        }
    }
}

/// The app's settings (spec sections 2 and 4.4), each saved as soon as it
/// changes. Defaults follow the locked decisions: idle unload after 10
/// minutes (L13), web search off (L7), Reminders sync off (L9), Quick as the
/// model for new chats (L1).
@MainActor
@Observable
public final class AppSettings {
    /// The `UserDefaults` keys, under the app's own domain.
    public enum Key {
        public static let idleUnloadMinutes = "idleUnloadMinutes"
        public static let webSearchEnabled = "webSearchEnabled"
        public static let offlineMode = "offlineMode"
        public static let remindersSyncEnabled = "remindersSyncEnabled"
        public static let defaultModel = "defaultModel"
        public static let thinkingByDefault = "thinkingByDefault"
        public static let blockRemoteMailContent = "blockRemoteMailContent"
        public static let hasCompletedFirstRun = "hasCompletedFirstRun"
        public static let useAppleModel = "useAppleModel"
    }

    /// The choices Settings offers for idle unload; 0 is "Never".
    public static let idleUnloadChoices = [5, 10, 15, 30, 60, 0]
    public static let defaultIdleUnloadMinutes = 10

    @ObservationIgnored private let store: any SettingsStore
    /// Called with the new delay whenever `idleUnloadMinutes` changes; the
    /// composition root points it at the engine's `IdleUnloadControl`.
    @ObservationIgnored public var onIdleUnloadChange: ((Duration?) -> Void)?
    /// Called with the new value whenever `offlineMode` changes; the
    /// composition root pauses running downloads when it turns on.
    @ObservationIgnored public var onOfflineModeChange: ((Bool) -> Void)?

    /// Minutes after the last token before the model is released; 0 never.
    public var idleUnloadMinutes: Int {
        didSet {
            if idleUnloadMinutes < 0 { idleUnloadMinutes = 0; return }
            store.set(idleUnloadMinutes, forKey: Key.idleUnloadMinutes)
            onIdleUnloadChange?(idleUnloadDelay)
        }
    }

    /// Web search through the bundled SearXNG (L7: off until turned on).
    public var webSearchEnabled: Bool {
        didSet { store.set(webSearchEnabled, forKey: Key.webSearchEnabled) }
    }

    /// Turns off web search, mail sync, errands and downloads.
    public var offlineMode: Bool {
        didSet {
            store.set(offlineMode, forKey: Key.offlineMode)
            if offlineMode != oldValue { onOfflineModeChange?(offlineMode) }
        }
    }

    /// To-dos sync to Apple Reminders (L9: only after the person turns it on).
    public var remindersSyncEnabled: Bool {
        didSet { store.set(remindersSyncEnabled, forKey: Key.remindersSyncEnabled) }
    }

    /// The model a new chat starts with.
    public var defaultModel: AppModel {
        didSet { store.set(defaultModel.rawValue, forKey: Key.defaultModel) }
    }

    /// Whether a new chat reasons before answering, on models that can.
    public var thinkingByDefault: Bool {
        didSet { store.set(thinkingByDefault, forKey: Key.thinkingByDefault) }
    }

    /// Settings > Privacy: remote images and trackers removed from mail.
    public var blockRemoteMailContent: Bool {
        didSet { store.set(blockRemoteMailContent, forKey: Key.blockRemoteMailContent) }
    }

    /// Apple's on-device model does mail sorting, titles, short summaries and
    /// scan fields when Apple Intelligence is on; Quick does them otherwise
    /// (L12).
    public var useAppleModel: Bool {
        didSet { store.set(useAppleModel, forKey: Key.useAppleModel) }
    }

    /// First run finished (or skipped); the onboarding is not shown again.
    public var hasCompletedFirstRun: Bool {
        didSet { store.set(hasCompletedFirstRun, forKey: Key.hasCompletedFirstRun) }
    }

    public init(store: any SettingsStore = UserDefaults.standard) {
        self.store = store
        idleUnloadMinutes = max(0, store.object(forKey: Key.idleUnloadMinutes) as? Int
            ?? Self.defaultIdleUnloadMinutes)
        webSearchEnabled = store.object(forKey: Key.webSearchEnabled) as? Bool ?? false
        offlineMode = store.object(forKey: Key.offlineMode) as? Bool ?? false
        remindersSyncEnabled = store.object(forKey: Key.remindersSyncEnabled) as? Bool ?? false
        defaultModel = (store.object(forKey: Key.defaultModel) as? String).flatMap(AppModel.init(rawValue:))
            ?? .quick
        thinkingByDefault = store.object(forKey: Key.thinkingByDefault) as? Bool ?? true
        blockRemoteMailContent = store.object(forKey: Key.blockRemoteMailContent) as? Bool ?? true
        hasCompletedFirstRun = store.object(forKey: Key.hasCompletedFirstRun) as? Bool ?? false
        useAppleModel = store.object(forKey: Key.useAppleModel) as? Bool ?? true
    }

    /// The idle delay for the engine; nil never unloads.
    public var idleUnloadDelay: Duration? {
        idleUnloadMinutes > 0 ? .seconds(idleUnloadMinutes * 60) : nil
    }

    /// Web search actually runs: switched on and not offline.
    public var isWebSearchActive: Bool { webSearchEnabled && !offlineMode }

    /// Anything that reaches the network for the person (downloads, mail
    /// sync, errands) checks this first.
    public var allowsNetwork: Bool { !offlineMode }

    /// Thinking for a new request on `model`: nil when the model has no
    /// thinking switch, so its own default applies.
    public func thinking(for model: AppModel) -> Bool? {
        model.supportsThinking ? thinkingByDefault : nil
    }
}
