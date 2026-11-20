import Foundation
import MferenceAppCore
import Observation

/// The voice screens' long-lived models (team N3): the setup card, Settings
/// › Speech and the remembered voice choices. The app builds one at launch
/// from `AppServices` and injects it with `.environment(voice)`; snapshots
/// and tests build one over `FakeSpeechService`. A screen without it shows
/// its honest "not available" state.
@MainActor
@Observable
public final class VoiceModels {
    public let preferences: SpeechPreferences
    public let setup: SpeechSetupModel
    public let settings: SpeechSettingsModel

    public init(speech: any SpeechServicing, permissions: any PermissionBroker, preferences: SpeechPreferences,
                isOffline: @escaping @MainActor () -> Bool) {
        self.preferences = preferences
        setup = SpeechSetupModel(speech: speech, isOffline: isOffline)
        settings = SpeechSettingsModel(setup: setup, preferences: preferences, speech: speech,
                                       permissions: permissions)
    }

    /// The app's: over `services.speech`, Offline mode from `services.settings`.
    public static func live(services: AppServices,
                            preferences: SpeechPreferences = SpeechPreferences()) -> VoiceModels {
        let settings = services.settings
        let voice = VoiceModels(speech: services.speech, permissions: services.permissions,
                                preferences: preferences, isOffline: { settings.offlineMode })
        voice.setup.start()
        return voice
    }
}
