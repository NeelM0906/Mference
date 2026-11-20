import Foundation
import Observation

/// The voice choices the person makes once and the app remembers (Notes
/// spec 4.2, 4.3, 4.8): Settings › Speech shows and edits them, the
/// dictation bar's chips and the record popover start from them and write
/// back what the person picks.
@MainActor
@Observable
public final class SpeechPreferences {
    /// What "Summaries" in Settings › Speech › Meetings writes with (D5).
    public enum SummaryModel: String, Sendable, Hashable, CaseIterable {
        case quick
        case off
    }

    /// Settings › Speech › Recordings › Keep audio (D4).
    public enum KeepAudio: String, Sendable, Hashable, CaseIterable {
        case keep
        case deleteAfterTranscribing
    }

    public enum Key {
        public static let language = "speech.language"
        public static let cleanUpFillers = "speech.cleanUpFillers"
        public static let microphoneID = "speech.microphoneID"
        public static let callAudio = "speech.callAudio"
        public static let speakerLabels = "speech.speakerLabels"
        public static let keepAudio = "speech.keepAudio"
        public static let summaryModel = "speech.summaryModel"
        public static let summaryWhenYouStop = "speech.summaryWhenYouStop"
        public static let fileSpeakerLabels = "speech.fileSpeakerLabels"
        public static let fileSummary = "speech.fileSummary"
    }

    /// Automatic, or one of the 25 languages pinned.
    public var language: SpeechLanguage {
        didSet { store.set(Self.encode(language), forKey: Key.language) }
    }
    /// Remove "um", "uh" and repeated starts from dictation (spec 6.4). Default on.
    public var cleanUpFillers: Bool {
        didSet { store.set(cleanUpFillers, forKey: Key.cleanUpFillers) }
    }
    /// The microphone for meetings and dictation; nil is the system default.
    public var microphoneID: String? {
        didSet { store.set(microphoneID, forKey: Key.microphoneID) }
    }
    /// Whose audio a meeting records besides the microphone. Default All apps.
    public var callAudio: CallAudioSource {
        didSet { store.set(Self.encode(callAudio), forKey: Key.callAudio) }
    }
    /// Speaker 2, 3, … in meetings (D3). Default on.
    public var speakerLabels: Bool {
        didSet { store.set(speakerLabels, forKey: Key.speakerLabels) }
    }
    /// Keep meeting audio after transcribing (D4). Default keep.
    public var keepAudio: KeepAudio {
        didSet { store.set(keepAudio.rawValue, forKey: Key.keepAudio) }
    }
    /// The model that writes meeting summaries; Off turns them off.
    public var summaryModel: SummaryModel {
        didSet { store.set(summaryModel.rawValue, forKey: Key.summaryModel) }
    }
    /// The popover's "Summary when you stop" (D5). Default on.
    public var summaryWhenYouStop: Bool {
        didSet { store.set(summaryWhenYouStop, forKey: Key.summaryWhenYouStop) }
    }
    /// The import sheet's Speaker labels and Write a summary switches.
    public var fileSpeakerLabels: Bool {
        didSet { store.set(fileSpeakerLabels, forKey: Key.fileSpeakerLabels) }
    }
    public var fileSummary: Bool {
        didSet { store.set(fileSummary, forKey: Key.fileSummary) }
    }

    @ObservationIgnored private let store: any SettingsStore

    public init(store: any SettingsStore = UserDefaults.standard) {
        self.store = store
        language = (store.object(forKey: Key.language) as? String).flatMap(Self.decodeLanguage) ?? .automatic
        cleanUpFillers = store.object(forKey: Key.cleanUpFillers) as? Bool ?? true
        microphoneID = store.object(forKey: Key.microphoneID) as? String
        callAudio = (store.object(forKey: Key.callAudio) as? String).flatMap(Self.decodeCallAudio) ?? .allApps
        speakerLabels = store.object(forKey: Key.speakerLabels) as? Bool ?? true
        keepAudio = (store.object(forKey: Key.keepAudio) as? String).flatMap(KeepAudio.init(rawValue:)) ?? .keep
        summaryModel = (store.object(forKey: Key.summaryModel) as? String).flatMap(SummaryModel.init(rawValue:)) ?? .quick
        summaryWhenYouStop = store.object(forKey: Key.summaryWhenYouStop) as? Bool ?? true
        fileSpeakerLabels = store.object(forKey: Key.fileSpeakerLabels) as? Bool ?? true
        fileSummary = store.object(forKey: Key.fileSummary) as? Bool ?? true
    }

    /// Whether a meeting writes its summary on stop: the popover's switch,
    /// unless Settings turned summaries off.
    public var writesMeetingSummary: Bool { summaryModel != .off && summaryWhenYouStop }

    /// The options a dictation starts with.
    public var dictationOptions: DictationOptions {
        DictationOptions(language: language, cleanUpFillers: cleanUpFillers, microphoneID: microphoneID)
    }

    /// The options a meeting in `noteID` starts with.
    public func meetingOptions(noteID: UUID) -> MeetingOptions {
        MeetingOptions(noteID: noteID, microphoneID: microphoneID, callAudio: callAudio,
                       speakerLabels: speakerLabels, keepAudio: keepAudio == .keep, language: language)
    }

    // MARK: Stored forms

    static func encode(_ language: SpeechLanguage) -> String {
        switch language {
        case .automatic: "auto"
        case .fixed(let code): code
        }
    }

    static func decodeLanguage(_ raw: String) -> SpeechLanguage? {
        if raw == "auto" { return .automatic }
        return SpeechLanguage.supportedCodes.contains(raw) ? .fixed(raw) : nil
    }

    static func encode(_ source: CallAudioSource) -> String {
        switch source {
        case .off: "off"
        case .allApps: "all"
        case .app(let bundleID, let name): "app\u{1F}\(bundleID)\u{1F}\(name)"
        }
    }

    static func decodeCallAudio(_ raw: String) -> CallAudioSource? {
        switch raw {
        case "off": return .off
        case "all": return .allApps
        default:
            let parts = raw.split(separator: "\u{1F}", omittingEmptySubsequences: false).map(String.init)
            guard parts.count == 3, parts[0] == "app", !parts[1].isEmpty else { return nil }
            return .app(bundleID: parts[1], name: parts[2])
        }
    }
}
