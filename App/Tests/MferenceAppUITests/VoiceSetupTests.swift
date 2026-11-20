import Foundation
import MferenceAppCore
@testable import MferenceAppUI
import Testing

/// The setup card (board 7) and Settings › Speech (board 8) on
/// `FakeSpeechService` and `FakePermissionBroker`: no model, no network, no TCC.
@MainActor
@Suite struct VoiceSetupTests {
    final class Offline: @unchecked Sendable { var value = false }

    private func voice(_ speech: FakeSpeechService, broker: FakePermissionBroker = FakePermissionBroker(),
                       offline: Offline = Offline()) -> VoiceModels {
        VoiceModels(speech: speech, permissions: broker,
                    preferences: SpeechPreferences(store: InMemorySettingsStore()),
                    isOffline: { offline.value })
    }

    @Test func readyRunsTheActionAtOnce() {
        let models = voice(FakeSpeechService())
        var ran = 0
        #expect(models.setup.require(.dictate) { ran += 1 })
        #expect(ran == 1 && models.setup.pendingAction == nil)
    }

    @Test func setupThenDownloadThenTheActionContinues() async {
        let speech = FakeSpeechService.notInstalled()
        let models = voice(speech)
        var ran: [SpeechSetupModel.Action] = []
        #expect(!models.setup.require(.recordMeeting) { ran.append(.recordMeeting) })
        #expect(models.setup.pendingAction == .recordMeeting, "the card shows")
        #expect(models.setup.display == .download)
        models.setup.download()
        await models.setup.waitForInstall()
        #expect(speech.calls.contains(.install(.speech)))
        #expect(ran == [.recordMeeting], "the original action continues")
        #expect(models.setup.pendingAction == nil && models.setup.display == .ready)
    }

    @Test func offlineModeBlocksTheDownload() async {
        let offline = Offline()
        offline.value = true
        let speech = FakeSpeechService.notInstalled()
        let models = voice(speech, offline: offline)
        models.setup.require(.dictate) {}
        models.setup.download()
        await models.setup.waitForInstall()
        #expect(!speech.calls.contains(.install(.speech)))
        #expect(models.setup.offline && models.setup.pendingAction == .dictate)
    }

    @Test func aFailedDownloadSaysWhyAndKeepsTheActionWaiting() async {
        let speech = FakeSpeechService.notInstalled()
        speech.failNextInstall(with: .verificationFailed(file: "Encoder.mlmodelc/weights/weight.bin"))
        let models = voice(speech)
        models.setup.start()
        var ran = false
        models.setup.require(.transcribeFiles) { ran = true }
        models.setup.download()
        await models.setup.waitForInstall()
        #expect(!ran)
        #expect(models.setup.display == .failed("A downloaded file didn't match. Download again."))
        // Try again works.
        models.setup.download()
        await models.setup.waitForInstall()
        #expect(ran)
        models.setup.stop()
    }

    @Test func pauseResumeAndCancelReachTheService() async throws {
        let speech = FakeSpeechService.notInstalled()
        speech.setStatus(SpeechStatus(speech: .downloading(progress: 0.42)))
        let models = voice(speech)
        models.setup.start()
        try await eventually { models.setup.display == .downloading(fraction: 0.42) }
        models.setup.pause()
        try await eventually { models.setup.display == .paused(fraction: 0.42) }
        models.setup.cancel()
        try await eventually { models.setup.display == .download }
        #expect(speech.calls == [.pauseInstall(.speech), .cancelInstall(.speech)])
        models.setup.stop()
    }

    @Test func installedFromSettingsWhileTheCardWaitedContinues() async throws {
        let speech = FakeSpeechService.notInstalled()
        let models = voice(speech)
        models.setup.start()
        var ran = false
        models.setup.require(.dictate) { ran = true }
        try await speech.install(.speech)
        try await eventually { ran }
        #expect(models.setup.pendingAction == nil)
        models.setup.stop()
    }

    @Test func dismissDropsTheWaitingAction() async {
        let speech = FakeSpeechService.notInstalled()
        let models = voice(speech)
        var ran = false
        models.setup.require(.dictate) { ran = true }
        models.setup.dismiss()
        #expect(models.setup.pendingAction == nil)
        models.setup.download()
        await models.setup.waitForInstall()
        #expect(!ran, "a dismissed action does not run later")
    }

    @Test func cardCopyUsesTheMeasuredSize() {
        // 653,388,064 bytes (Pro plus the voice-activity model) in whole megabytes.
        #expect(SpeechSetupCard.modelLine(bytes: SpeechModelPins.speech.totalBytes)
                == "653 MB · runs on the Neural Engine, so chat and the model you use there keep running")
        #expect(SpeechSetupCard.approvalMessage(bytes: 653_388_064).contains("(653 MB)"))
        #expect(VoiceFormat.size(66_591_913) == "67 MB")
        #expect(VoiceFormat.size(1_234_000_000) == "1.2 GB")
        #expect(!SpeechSetupCard.modelLine(bytes: 1).contains("["), "bracketed values never ship")
    }

    // MARK: Settings › Speech

    @Test func settingsReadsWithoutPrompting() async {
        let speech = FakeSpeechService()
        speech.setRecordingsSize(1_200_000_000)
        let broker = FakePermissionBroker(states: [.systemAudio: .granted])
        let models = voice(speech, broker: broker)
        await models.settings.refresh()
        #expect(models.settings.devices.map(\.name) == ["MacBook Pro Microphone"])
        #expect(models.settings.apps.map(\.name) == ["zoom.us"])
        #expect(models.settings.callAudioLine == "System audio recording · allowed")
        #expect(models.settings.spaceUsedLine == "1.2 GB on this Mac")
        #expect(!broker.calls.contains { if case .request = $0 { true } else { false } }, "never prompts (L6)")
    }

    @Test func callAudioLineAndSystemSettingsButton() async {
        #expect(SpeechSettingsModel.callAudioLine(.notDetermined) == "System audio recording · asked at your first meeting")
        #expect(SpeechSettingsModel.callAudioLine(.denied).hasPrefix("System audio recording · off"))
        let broker = FakePermissionBroker(states: [.systemAudio: .denied, .microphone: .needsSystemSettings])
        let models = voice(FakeSpeechService(), broker: broker)
        await models.settings.refresh()
        #expect(models.settings.callAudioNeedsSystemSettings && models.settings.microphoneNeedsSystemSettings)
        models.settings.openSystemSettings(for: .systemAudio)
        try? await eventually { broker.calls.contains(.openSystemSettings(.systemAudio)) }
        #expect(broker.calls.contains(.openSystemSettings(.systemAudio)))
    }

    @Test func fieldNamesFollowTheChoices() {
        let devices = [AudioInputDevice(id: "a", name: "Studio Mic", isDefault: false),
                       AudioInputDevice(id: "b", name: "MacBook Pro Microphone", isDefault: true)]
        #expect(SpeechSettingsModel.microphoneName(id: nil, devices: devices) == "MacBook Pro Microphone")
        #expect(SpeechSettingsModel.microphoneName(id: "a", devices: devices) == "Studio Mic")
        #expect(SpeechSettingsModel.microphoneName(id: "gone", devices: devices) == "MacBook Pro Microphone")
        #expect(SpeechSettingsModel.microphoneName(id: nil, devices: []) == "Default microphone")
        #expect(SpeechSettingsModel.callAudioName(.allApps) == "All apps")
        #expect(SpeechSettingsModel.callAudioName(.app(bundleID: "us.zoom.xos", name: "Zoom")) == "Zoom")
        #expect(SpeechSettingsModel.summaryName(.quick) == "Quick · Qwen3.5 4B")
        #expect(SpeechSettingsModel.keepAudioName(.deleteAfterTranscribing) == "Delete after transcribing")
        #expect(SpeechSettingsModel.spaceUsedLine(bytes: 0) == "No recordings yet")
        #expect(SpeechSettingsView.speechLine(.ready, bytes: 653_388_064) == "653 MB · Neural Engine · CC BY 4.0")
        #expect(SpeechSettingsView.credit.hasPrefix("Fluid Speech Pro by FluidVoice"), "the licence credit (spec 3)")
    }

    @Test func removeTurnsVoiceFeaturesBackToSetup() async throws {
        let speech = FakeSpeechService()
        let models = voice(speech)
        models.setup.start()
        await models.settings.removeSpeech()
        #expect(speech.calls.contains(.remove(.speech)))
        try await eventually { models.setup.display == .download }
        var ran = false
        #expect(!models.setup.require(.dictate) { ran = true }, "voice actions show the setup card again")
        #expect(!ran)
        models.setup.stop()
    }

    @Test func speakerLabelsDownloadRespectsOfflineMode() async {
        let offline = Offline()
        offline.value = true
        let speech = FakeSpeechService(status: SpeechStatus(speech: .ready, speakerLabels: .notInstalled))
        let models = voice(speech, offline: offline)
        models.settings.downloadSpeakerLabels()
        await models.settings.waitForSpeakerLabels()
        #expect(!speech.calls.contains(.install(.speakerLabels)))
        offline.value = false
        models.settings.downloadSpeakerLabels()
        await models.settings.waitForSpeakerLabels()
        #expect(speech.calls.contains(.install(.speakerLabels)))
        #expect(speech.status.speakerLabels == .ready)
    }

    /// Waits (briefly) for a condition that a background task makes true.
    private func eventually(_ condition: @MainActor () -> Bool) async throws {
        for _ in 0..<200 {
            if condition() { return }
            try await Task.sleep(for: .milliseconds(5))
        }
        Issue.record("condition never became true")
    }
}
