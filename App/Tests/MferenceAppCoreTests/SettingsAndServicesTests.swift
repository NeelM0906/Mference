import AVFoundation
import EventKit
import Foundation
import Mference
@testable import MferenceAppCore
import MferenceEngine
import SwiftData
import Synchronization
import Testing

@Suite struct SettingsTests {
    @Test @MainActor func defaultsFollowTheLockedDecisions() {
        let settings = AppSettings(store: InMemorySettingsStore())
        #expect(settings.idleUnloadMinutes == 10)
        #expect(settings.idleUnloadDelay == .seconds(600))
        #expect(settings.webSearchEnabled == false)
        #expect(settings.offlineMode == false)
        #expect(settings.remindersSyncEnabled == false)
        #expect(settings.defaultModel == .quick)
        #expect(settings.thinkingByDefault == true)
        #expect(settings.blockRemoteMailContent == true)
        #expect(settings.hasCompletedFirstRun == false)
    }

    @Test @MainActor func changesPersistToTheStore() {
        let store = InMemorySettingsStore()
        let settings = AppSettings(store: store)
        settings.idleUnloadMinutes = 30
        settings.webSearchEnabled = true
        settings.offlineMode = true
        settings.remindersSyncEnabled = true
        settings.defaultModel = .deep
        settings.thinkingByDefault = false
        settings.blockRemoteMailContent = false
        settings.hasCompletedFirstRun = true

        let reopened = AppSettings(store: store)
        #expect(reopened.idleUnloadMinutes == 30)
        #expect(reopened.webSearchEnabled)
        #expect(reopened.offlineMode)
        #expect(reopened.remindersSyncEnabled)
        #expect(reopened.defaultModel == .deep)
        #expect(reopened.thinkingByDefault == false)
        #expect(reopened.blockRemoteMailContent == false)
        #expect(reopened.hasCompletedFirstRun)
    }

    @Test @MainActor func unknownStoredValuesFallBackToDefaults() {
        let store = InMemorySettingsStore([AppSettings.Key.defaultModel: "gpt", AppSettings.Key.idleUnloadMinutes: -4])
        let settings = AppSettings(store: store)
        #expect(settings.defaultModel == .quick)
        #expect(settings.idleUnloadMinutes == 0)
    }

    @Test @MainActor func offlineModeTurnsWebSearchOff() {
        let settings = AppSettings(store: InMemorySettingsStore())
        settings.webSearchEnabled = true
        #expect(settings.isWebSearchActive)
        settings.offlineMode = true
        #expect(settings.isWebSearchActive == false)
        #expect(settings.allowsNetwork == false)
        // The person's choice is kept for when they go back online.
        #expect(settings.webSearchEnabled)
    }

    @Test @MainActor func offlineChangesAreReportedOnce() {
        let settings = AppSettings(store: InMemorySettingsStore())
        var seen: [Bool] = []
        settings.onOfflineModeChange = { seen.append($0) }
        settings.offlineMode = true
        settings.offlineMode = true
        settings.offlineMode = false
        #expect(seen == [true, false])
    }

    @Test @MainActor func neverMeansNoIdleDelay() {
        let settings = AppSettings(store: InMemorySettingsStore())
        settings.idleUnloadMinutes = 0
        #expect(settings.idleUnloadDelay == nil)
        #expect(AppSettings.idleUnloadChoices.contains(0))
        #expect(AppSettings.idleUnloadChoices.contains(AppSettings.defaultIdleUnloadMinutes))
    }

    @Test @MainActor func thinkingAppliesOnlyToModelsThatThink() {
        let settings = AppSettings(store: InMemorySettingsStore())
        #expect(settings.thinking(for: .quick) == true)
        #expect(settings.thinking(for: .maple) == nil)
        settings.thinkingByDefault = false
        #expect(settings.thinking(for: .deep) == false)
    }
}

@Suite struct IdleUnloadControlTests {
    @Test func changingTheDelayRestartsARunningWait() async throws {
        let sleeper = TestSleeper()
        let control = IdleUnloadControl(delay: .seconds(600), sleep: sleeper.sleep)
        let wait = Task { try await control.waitForIdleDelay() }
        while sleeper.calls == 0 { await Task.yield() }
        control.setDelay(.seconds(60))
        // The first sleep is abandoned and a new one starts.
        while sleeper.calls < 2 { await Task.yield() }
        sleeper.fire()
        try await wait.value
    }

    @Test func cancellingTheWaitThrows() async {
        let control = IdleUnloadControl(delay: nil)
        let wait = Task { try await control.waitForIdleDelay() }
        wait.cancel()
        await #expect(throws: CancellationError.self) { try await wait.value }
    }

    @Test func settingTheSameDelayDoesNotRestart() async throws {
        let sleeper = TestSleeper()
        let control = IdleUnloadControl(delay: .seconds(600), sleep: sleeper.sleep)
        let wait = Task { try await control.waitForIdleDelay() }
        while sleeper.calls == 0 { await Task.yield() }
        control.setDelay(.seconds(600))
        for _ in 0..<50 { await Task.yield() }
        #expect(sleeper.calls == 1)
        sleeper.fire()
        try await wait.value
    }
}

@Suite struct PermissionsTests {
    @Test func eventKitStatesMap() {
        #expect(LivePermissionBroker.state(EKAuthorizationStatus.notDetermined) == .notDetermined)
        #expect(LivePermissionBroker.state(EKAuthorizationStatus.fullAccess) == .granted)
        #expect(LivePermissionBroker.state(EKAuthorizationStatus.writeOnly) == .needsSystemSettings)
        #expect(LivePermissionBroker.state(EKAuthorizationStatus.denied) == .needsSystemSettings)
        #expect(LivePermissionBroker.state(EKAuthorizationStatus.restricted) == .denied)
    }

    @Test func captureStatesMap() {
        #expect(LivePermissionBroker.state(AVAuthorizationStatus.notDetermined) == .notDetermined)
        #expect(LivePermissionBroker.state(AVAuthorizationStatus.authorized) == .granted)
        #expect(LivePermissionBroker.state(AVAuthorizationStatus.denied) == .needsSystemSettings)
        #expect(LivePermissionBroker.state(AVAuthorizationStatus.restricted) == .denied)
    }

    @Test func automationStatesMap() {
        #expect(LivePermissionBroker.automationState(OSStatus(noErr)) == .granted)
        #expect(LivePermissionBroker.automationState(OSStatus(errAEEventWouldRequireUserConsent)) == .notDetermined)
        #expect(LivePermissionBroker.automationState(OSStatus(procNotFound)) == .notDetermined)
        #expect(LivePermissionBroker.automationState(OSStatus(errAEEventNotPermitted)) == .needsSystemSettings)
    }

    @Test func everyPermissionHasASettingsPaneAndAReason() {
        for permission in AppPermission.allCases {
            let url = LivePermissionBroker.settingsURL(for: permission)
            #expect(url.scheme == "x-apple.systempreferences")
            #expect(url.query?.hasPrefix("Privacy_") == true)
            #expect(!permission.reason.isEmpty)
        }
    }

    @Test func fullDiskAccessProbeReadsAFileOnlyItAllows() throws {
        // A home without the TCC database reads as "not granted", never a prompt.
        let home = FileManager.default.temporaryDirectory.appending(path: "fda-\(UUID().uuidString)")
        #expect(LivePermissionBroker.fullDiskAccessState(homeDirectory: home) == .needsSystemSettings)
        let tcc = home.appending(path: "Library/Application Support/com.apple.TCC")
        try FileManager.default.createDirectory(at: tcc, withIntermediateDirectories: true)
        try Data().write(to: tcc.appending(path: "TCC.db"))
        defer { try? FileManager.default.removeItem(at: home) }
        #expect(LivePermissionBroker.fullDiskAccessState(homeDirectory: home) == .granted)
    }

    /// Records the order of reason, prompt and settings-pane calls.
    final class CallLog: Sendable {
        private let log = Mutex<[String]>([])
        private let states = Mutex<[AppPermission: PermissionState]>([:])
        var calls: [String] { log.withLock { $0 } }
        func record(_ call: String) { log.withLock { $0.append(call) } }
        func set(_ state: PermissionState, _ permission: AppPermission) { states.withLock { $0[permission] = state } }
        func state(_ permission: AppPermission) -> PermissionState { states.withLock { $0[permission] ?? .notDetermined } }

        func broker(goOn: Bool, grant: PermissionState = .granted) -> LivePermissionBroker {
            LivePermissionBroker(
                presentReason: { permission in
                    self.record("reason:\(permission.rawValue)")
                    return goOn
                },
                system: .init(
                    state: { self.state($0) },
                    prompt: { permission in
                        self.record("prompt:\(permission.rawValue)")
                        self.set(grant, permission)
                    },
                    openURL: { url in self.record("open:\(url.query ?? "")") }))
        }
    }

    @Test func reasonComesBeforeThePrompt() async {
        let log = CallLog()
        let broker = log.broker(goOn: true)
        #expect(await broker.request(.calendar) == .granted)
        #expect(log.calls == ["reason:calendar", "prompt:calendar"])
    }

    @Test func declinedReasonMeansNoPrompt() async {
        let log = CallLog()
        let broker = log.broker(goOn: false)
        #expect(await broker.request(.camera) == .notDetermined)
        #expect(log.calls == ["reason:camera"])
    }

    @Test func decidedPermissionsAreNotAskedAgain() async {
        let log = CallLog()
        log.set(.needsSystemSettings, .calendar)
        let broker = log.broker(goOn: true)
        #expect(await broker.request(.calendar) == .needsSystemSettings)
        #expect(log.calls.isEmpty)
        #expect(await broker.state(of: .reminders) == .notDetermined)
        #expect(log.calls.isEmpty, "reading never shows a reason or a prompt")
    }

    @Test func fullDiskAccessShowsTheReasonThenTheSettingsPane() async {
        let log = CallLog()
        log.set(.needsSystemSettings, .fullDiskAccess)
        let broker = log.broker(goOn: true)
        #expect(await broker.request(.fullDiskAccess) == .needsSystemSettings)
        #expect(log.calls == ["reason:fullDiskAccess", "open:Privacy_AllFiles"])
    }

    @Test func fakeBrokerAnswersOnlyUndeterminedRequests() async {
        let broker = FakePermissionBroker(states: [.camera: .needsSystemSettings], answers: [.calendar: .denied])
        #expect(await broker.state(of: .calendar) == .notDetermined)
        #expect(await broker.request(.calendar) == .denied)
        #expect(await broker.request(.camera) == .needsSystemSettings)
        #expect(await broker.request(.reminders) == .granted)
        await broker.openSystemSettings(for: .camera)
        #expect(broker.calls.contains(.openSystemSettings(.camera)))
    }
}

@Suite struct AppServicesTests {
    @Test @MainActor func previewWiresFakesAndAnInMemoryStore() async throws {
        let services = AppServices.preview(installed: [.quick, .minicpm5])
        let rows = await services.engine.models()
        #expect(rows.count == AppModel.allCases.count)
        #expect(rows.first { $0.model == .quick }?.availability == .ready)
        #expect(rows.first { $0.model == .quick }?.sizeBytes ?? 0 > 0)
        #expect(services.storeError == nil)
        let context = services.modelContainer.mainContext
        context.insert(Conversation(title: "Preview"))
        try context.save()
        #expect(try context.fetchCount(FetchDescriptor<Conversation>()) == 1)
        #expect(services.settings.webSearchEnabled == false)
        _ = try await collect(services.engine.generate(model: .quick, turns: hello, options: GenerationOptions()))
    }

    @Test @MainActor func idleSettingDrivesTheEngineControl() async throws {
        let sleeper = TestSleeper()
        let control = IdleUnloadControl(delay: nil, sleep: sleeper.sleep)
        let settings = AppSettings(store: InMemorySettingsStore([AppSettings.Key.idleUnloadMinutes: 0]))
        let journal = FakeEngineJournal()
        let services = AppServices.preview(journal: journal, settings: settings, idleUnload: control)
        #expect(control.delay == nil)
        try await services.engine.load(.quick)
        settings.idleUnloadMinutes = 5
        #expect(control.delay == .seconds(300))
        while sleeper.calls == 0 { await Task.yield() }
        sleeper.fire()
        let engine = try #require(services.engine as? EngineService)
        #expect(await waitForStatus(engine) { $0.phase == .empty } != nil)
        #expect(journal.entries.contains(.unloaded(AppModel.quick.modelID)))
    }
}
