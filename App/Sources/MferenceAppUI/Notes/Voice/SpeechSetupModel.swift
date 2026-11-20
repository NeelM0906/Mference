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
    /// Bytes the Speech package downloads, from its pins (0 where no
    /// service is built: the card then leaves the size out).
    public let downloadBytes: Int64
    /// Why the last download stopped, when the service threw without
    /// changing its status (Offline mode mid-call, low disk, no service).
    /// Shown as a failure until the next download or status change.
    public private(set) var installError: SpeechError?

    @ObservationIgnored private let speech: any SpeechServicing
    @ObservationIgnored private let isOffline: @MainActor () -> Bool
    @ObservationIgnored private var continuation: (@MainActor () -> Void)?
    @ObservationIgnored private var statusTask: Task<Void, Never>?
    @ObservationIgnored private var installTask: Task<Void, Never>?
    /// Bumped by each download, so a paused download's late end never
    /// clears the task of the Resume that followed it.
    @ObservationIgnored private var installGeneration: UInt64 = 0
    /// Resume pressed while the paused download is still winding down: the
    /// service would join that install and end paused (its contract), so the
    /// download starts again once it has ended.
    @ObservationIgnored private var resumeAfterWindDown = false

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
        Self.display(for: status.speech, installError: installError)
    }

    static func display(for state: SpeechModelState, installError: SpeechError?) -> Display {
        switch state {
        case .notInstalled, .failed:
            if let installError { return .failed(installError.message) }
        case .downloading, .paused, .preparing, .ready:
            break
        }
        return display(for: state)
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
    /// A Resume while the paused download is still winding down starts the
    /// install again (the service joins or resumes it).
    public func download() {
        guard !offline else { return }
        // The service's own status: the cached copy can lag a pause by a beat.
        if installTask != nil {
            if case .paused = speech.status.speech { resumeAfterWindDown = true }
            return
        }
        installGeneration &+= 1
        let generation = installGeneration
        installError = nil
        installTask = Task { [weak self, speech] in
            do {
                try await speech.install(.speech)
                self?.installEnded(generation, error: nil, finished: true)
            } catch is CancellationError {
                // Paused or cancelled: the status says which.
                self?.installEnded(generation, error: nil, finished: false)
            } catch {
                self?.installEnded(generation, error: (error as? SpeechError) ?? .failed(String(describing: error)),
                                   finished: false)
            }
        }
    }

    public func pause() {
        resumeAfterWindDown = false
        Task { [speech] in await speech.pauseInstall(.speech) }
    }

    public func cancel() {
        Task { [speech] in await speech.cancelInstall(.speech) }
    }

    /// Waits for the current download (tests).
    func waitForInstall() async {
        await installTask?.value
    }

    private func installEnded(_ generation: UInt64, error: SpeechError?, finished: Bool) {
        guard generation == installGeneration else { return }
        installTask = nil
        installError = error
        if resumeAfterWindDown {
            resumeAfterWindDown = false
            if !finished, error == nil, case .paused = speech.status.speech {
                download()
                return
            }
        }
        apply(speech.status)
        guard finished, status.speech.isReady, let action = continuation else { return }
        continuation = nil
        pendingAction = nil
        action()
    }

    private func apply(_ status: SpeechStatus) {
        // A download that started or finished since replaces the old error.
        if status.speech != self.status.speech, installTask == nil, status.speech != .notInstalled {
            installError = nil
        }
        self.status = status
        // Installed elsewhere (Settings) while the card waited.
        if status.speech.isReady, installTask == nil, let action = continuation {
            continuation = nil
            pendingAction = nil
            action()
        }
    }
}
