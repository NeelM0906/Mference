import Foundation
import MferenceAppCore
import Observation

/// First use (Notes spec 4.7, board 7): any voice action before Fluid Speech
/// Pro is installed shows the setup card; the download runs from there
/// (progress, pause, resume, cancel, verify), then "Preparing…" while the
/// first load compiles for the Neural Engine, then the action the person
/// started continues.
///
/// One model for the app (held by `VoiceModels`), so the card, the sidebar's
/// speech status and Settings › Speech show the same download.
@MainActor
@Observable
public final class SpeechSetupModel {
    /// The voice feature the person tried before the model was installed.
    public enum Action: Sendable, Hashable {
        case dictate, recordMeeting, transcribeFiles
    }

    /// What the card shows.
    public enum Display: Equatable {
        /// Download (disabled with the Offline line when offline).
        case download
        case downloading(fraction: Double)
        case paused(fraction: Double)
        case preparing
        case ready
        /// The failure's line and Try again.
        case failed(String)
    }

    public private(set) var status: SpeechStatus
    /// The card is showing (a voice action is waiting for the model).
    public private(set) var pendingAction: Action?
    /// Bytes the Speech package downloads, from its pins.
    public let downloadBytes: Int64

    @ObservationIgnored private let speech: any SpeechServicing
    @ObservationIgnored private let isOffline: @MainActor () -> Bool
    @ObservationIgnored private var continuation: (@MainActor () -> Void)?
    @ObservationIgnored private var statusTask: Task<Void, Never>?
    @ObservationIgnored private var installTask: Task<Void, Never>?

    public init(speech: any SpeechServicing, isOffline: @escaping @MainActor () -> Bool) {
        self.speech = speech
        self.isOffline = isOffline
        status = speech.status
        downloadBytes = speech.downloadSize(of: .speech)
    }

    /// Follows the service's status for the app's lifetime.
    public func start() {
        guard statusTask == nil else { return }
        let updates = speech.statusUpdates()
        statusTask = Task { [weak self] in
            for await status in updates {
                guard let self else { return }
                self.apply(status)
            }
        }
    }

    public func stop() {
        statusTask?.cancel()
        statusTask = nil
    }

    public var offline: Bool { isOffline() }
    public var modelState: SpeechModelState { status.speech }

    public var display: Display {
        Self.display(for: status.speech)
    }

    static func display(for state: SpeechModelState) -> Display {
        switch state {
        case .notInstalled: .download
        case .downloading(let progress): .downloading(fraction: min(max(progress, 0), 1))
        case .paused(let progress): .paused(fraction: min(max(progress, 0), 1))
        case .preparing: .preparing
        case .ready: .ready
        case .failed(let error): .failed(error.message)
        }
    }

    /// Runs `action` now when the model is ready; otherwise shows the setup
    /// card and runs it once the model is ready. Returns whether it ran now.
    @discardableResult
    public func require(_ kind: Action, then action: @escaping @MainActor () -> Void) -> Bool {
        if status.speech.isReady {
            action()
            return true
        }
        pendingAction = kind
        continuation = action
        return false
    }

    /// Closes the card without running the waiting action (the person left
    /// Notes or picked another note); a running download keeps going.
    public func dismiss() {
        pendingAction = nil
        continuation = nil
    }

    /// Download, or resume a paused download. Approval happens in the view.
    public func download() {
        guard installTask == nil, !offline else { return }
        installTask = Task { [weak self, speech] in
            do {
                try await speech.install(.speech)
                self?.installFinished()
            } catch {
                // Paused, cancelled or failed: the status says which.
                self?.installTask = nil
                self?.apply(speech.status)
            }
        }
    }

    public func pause() {
        Task { [speech] in await speech.pauseInstall(.speech) }
    }

    public func cancel() {
        Task { [speech] in await speech.cancelInstall(.speech) }
    }

    /// Waits for the current download (tests).
    func waitForInstall() async {
        await installTask?.value
    }

    private func installFinished() {
        installTask = nil
        apply(speech.status)
        guard status.speech.isReady, let action = continuation else { return }
        continuation = nil
        pendingAction = nil
        action()
    }

    private func apply(_ status: SpeechStatus) {
        self.status = status
        // Installed elsewhere (Settings) while the card waited.
        if status.speech.isReady, installTask == nil, let action = continuation {
            continuation = nil
            pendingAction = nil
            action()
        }
    }
}
