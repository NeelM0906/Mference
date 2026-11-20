import Foundation
import MferenceAppCore
import Observation

/// The voice screens' long-lived models (team N3): the setup card, Settings
/// › Speech, the remembered voice choices and dictation. The app builds one at launch
/// from `AppServices` and injects it with `.environment(voice)`; snapshots
/// and tests build one over `FakeSpeechService`. A screen without it shows
/// its honest "not available" state.
@MainActor
@Observable
public final class VoiceModels {
    public let preferences: SpeechPreferences
    public let setup: SpeechSetupModel
    public let settings: SpeechSettingsModel
    /// `AppServices.dictation`: the one consumer of dictation events.
    public let dictation: DictationController
    /// The note editor that has focus, set by the editor (team N1): ⇧⌘D
    /// dictates into it. Weak, so a closed editor never receives text.
    @ObservationIgnored public weak var focusedDictationTarget: (any DictationInsertionTarget)?

    public init(speech: any SpeechServicing, permissions: any PermissionBroker, preferences: SpeechPreferences,
                dictation: DictationController? = nil,
                isOffline: @escaping @MainActor () -> Bool) {
        self.preferences = preferences
        self.dictation = dictation ?? DictationController(speech: speech, permissions: permissions,
                                                          preferences: preferences)
        setup = SpeechSetupModel(speech: speech, isOffline: isOffline)
        settings = SpeechSettingsModel(setup: setup, preferences: preferences, speech: speech,
                                       permissions: permissions)
    }

    /// The app's: over `services.speech` and `services.dictation`, Offline
    /// mode from `services.settings`.
    public static func live(services: AppServices) -> VoiceModels {
        let settings = services.settings
        let voice = VoiceModels(speech: services.speech, permissions: services.permissions,
                                preferences: services.speechPreferences, dictation: services.dictation,
                                isOffline: { settings.offlineMode })
        voice.setup.start()
        return voice
    }

    // MARK: Dictation (spec 4.2)

    /// The Dictate button and ⇧⌘D: stops dictating into `target`, or starts
    /// there (through the setup card on first use).
    public func toggleDictation(in target: any DictationInsertionTarget) {
        if dictation.isDictating(into: target.noteID) {
            Task { await dictation.stop() }
            return
        }
        setup.require(.dictate) { [dictation, weak target] in
            guard let target else { return }
            Task { await dictation.start(in: target) }
        }
    }

    /// ⇧⌘D from the menu: the focused editor, or stop when there is none.
    public func toggleDictation() {
        if let target = focusedDictationTarget {
            toggleDictation(in: target)
        } else if dictation.isActive {
            Task { await dictation.stop() }
        }
    }

    /// The shell's hook: leaving Notes while dictating stops dictation first,
    /// so its last phrase lands in its note before the editor goes away.
    public static func finishBeforeLeaving(_ section: AppSection,
                                           voice: VoiceModels) -> (@MainActor () async -> Void)? {
        guard section == .notes, voice.dictation.isActive else { return nil }
        return { await voice.stopDictation() }
    }

    /// Switching notes or leaving Notes: the last phrase lands in the note
    /// it belongs to, then dictation stops (N1 calls this).
    public func stopDictation() async {
        await dictation.stop()
    }
}
