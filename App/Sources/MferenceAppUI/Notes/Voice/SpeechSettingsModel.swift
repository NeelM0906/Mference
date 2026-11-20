import Foundation
import MferenceAppCore
import Observation

/// What Settings › Speech (Notes spec 4.8, board 8) reads and changes: the
/// model packages (through the shared `SpeechSetupModel`, so the card and
/// Settings show one download), the remembered voice choices, devices and
/// running apps, the two permissions and the space recordings use.
@MainActor
@Observable
public final class SpeechSettingsModel {
    public let setup: SpeechSetupModel
    public let preferences: SpeechPreferences
    public private(set) var devices: [AudioInputDevice] = []
    public private(set) var apps: [RunningAudioApp] = []
    public private(set) var microphonePermission: PermissionState = .notDetermined
    public private(set) var callAudioPermission: PermissionState = .notDetermined
    public private(set) var recordingsBytes: Int64 = 0
    /// Set while Remove runs, and when it failed.
    public private(set) var isRemoving = false
    public private(set) var removeError: String?

    @ObservationIgnored private let speech: any SpeechServicing
    @ObservationIgnored private let permissions: any PermissionBroker
    @ObservationIgnored private var labelsTask: Task<Void, Never>?

    public init(setup: SpeechSetupModel, preferences: SpeechPreferences, speech: any SpeechServicing,
                permissions: any PermissionBroker) {
        self.setup = setup
        self.preferences = preferences
        self.speech = speech
        self.permissions = permissions
    }

    /// Re-reads devices, apps, permission states and the recordings size.
    /// Never prompts (L6).
    public func refresh() async {
        devices = await speech.inputDevices()
        apps = await speech.runningAudioApps()
        microphonePermission = await permissions.state(of: .microphone)
        callAudioPermission = await permissions.state(of: .systemAudio)
        recordingsBytes = await speech.recordingsSize()
    }

    // MARK: Model card

    public var speechState: SpeechModelState { setup.status.speech }
    public var speakerLabelsState: SpeechModelState { setup.status.speakerLabels }
    public var speechBytes: Int64 { setup.downloadBytes }
    public var speakerLabelsBytes: Int64 { speech.downloadSize(of: .speakerLabels) }

    /// Deletes Fluid Speech Pro (after the view's confirm); voice features
    /// go back to setup.
    public func removeSpeech() async {
        isRemoving = true
        removeError = nil
        do {
            try await speech.remove(.speech)
        } catch {
            removeError = (error as? SpeechError)?.message ?? "Fluid Speech Pro could not be removed. Try again."
        }
        isRemoving = false
    }

    public func downloadSpeakerLabels() {
        guard labelsTask == nil, !setup.offline else { return }
        labelsTask = Task { [weak self, speech] in
            try? await speech.install(.speakerLabels)
            self?.labelsTask = nil
        }
    }

    public func pauseSpeakerLabels() {
        Task { [speech] in await speech.pauseInstall(.speakerLabels) }
    }

    public func cancelSpeakerLabels() {
        Task { [speech] in await speech.cancelInstall(.speakerLabels) }
    }

    /// Waits for a speaker-labels download (tests).
    func waitForSpeakerLabels() async { await labelsTask?.value }

    // MARK: Meetings card

    /// The microphone field's text: the chosen device, else the default one.
    public var microphoneName: String {
        Self.microphoneName(id: preferences.microphoneID, devices: devices)
    }

    static func microphoneName(id: String?, devices: [AudioInputDevice]) -> String {
        if let id, let device = devices.first(where: { $0.id == id }) { return device.name }
        return devices.first(where: \.isDefault)?.name ?? devices.first?.name ?? "Default microphone"
    }

    /// The call-audio field's text.
    public var callAudioName: String { Self.callAudioName(preferences.callAudio) }

    static func callAudioName(_ source: CallAudioSource) -> String {
        switch source {
        case .off: "Off"
        case .allApps: "All apps"
        case .app(_, let name): name
        }
    }

    /// "System audio recording · allowed", and so on (board 8).
    public var callAudioLine: String { Self.callAudioLine(callAudioPermission) }

    static func callAudioLine(_ state: PermissionState) -> String {
        switch state {
        case .granted: "System audio recording · allowed"
        case .notDetermined: "System audio recording · asked at your first meeting"
        case .denied, .needsSystemSettings: "System audio recording · off. Meetings record your microphone only."
        }
    }

    public var callAudioNeedsSystemSettings: Bool {
        callAudioPermission == .denied || callAudioPermission == .needsSystemSettings
    }

    public var microphoneNeedsSystemSettings: Bool {
        microphonePermission == .denied || microphonePermission == .needsSystemSettings
    }

    public func openSystemSettings(for permission: AppPermission) {
        Task { [permissions] in await permissions.openSystemSettings(for: permission) }
    }

    /// The summaries field's text.
    static func summaryName(_ model: SpeechPreferences.SummaryModel) -> String {
        switch model {
        case .quick: "Quick · Qwen3.5 4B"
        case .off: "Off"
        }
    }

    // MARK: Recordings card

    static func keepAudioName(_ keep: SpeechPreferences.KeepAudio) -> String {
        switch keep {
        case .keep: "Keep"
        case .deleteAfterTranscribing: "Delete after transcribing"
        }
    }

    /// "1.2 GB", or "No recordings yet".
    public var spaceUsedLine: String { Self.spaceUsedLine(bytes: recordingsBytes) }

    static func spaceUsedLine(bytes: Int64) -> String {
        bytes > 0 ? "\(VoiceFormat.size(bytes)) on this Mac" : "No recordings yet"
    }

    /// The folder that holds every note's recordings, for Show in Finder.
    public var recordingsRoot: URL {
        speech.recordingsDirectory(for: UUID()).deletingLastPathComponent()
    }
}
