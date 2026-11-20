import AppKit
import MferenceAppCore
import MferenceAppUI

/// The app's long-lived objects, built once at launch and kept for the app's
/// lifetime: U2's composition root (`AppServices.live`), U3's web answers,
/// the one chat model (switching pills must not drop a running answer) and
/// the screen models U1's Settings, model library and first run read.
@MainActor
final class AppRuntime {
    let services: AppServices
    let web: WebAnswers
    let chat: ChatViewModel
    let screenModels: ScreenModels
    /// Team N3's voice models: the speech setup card and Settings › Speech.
    let voice: VoiceModels

    /// Quit steps, run in order: web answers (SearXNG) stop before the engine
    /// shuts down (U3's notes).
    private let shutdownSteps: [@MainActor () async -> Void]

    /// One composition root for the whole app: every engine use (chat, model
    /// library, Scan, Calendar, Mail) goes through `services.engine`, so only
    /// one engine host can hold a model. Chat goes through
    /// `services.chatEngine`, the same engine plus Apple's on-device model. Nothing here loads or warms a model;
    /// that happens only when the person picks a model or opens a chat.
    init(services: AppServices = .live(presentPermissionReason: PermissionReasonAlert.present),
         web: WebAnswers? = nil) {
        self.services = services
        // Web search follows Settings > Privacy (services.settings).
        let web = web ?? WebAnswers(settings: services.settings)
        self.web = web
        chat = ChatViewModel(service: services.chatEngine,
                             store: ConversationStore(container: services.modelContainer),
                             tools: web,
                             defaultModel: { [settings = services.settings] in settings.defaultModel })
        screenModels = ScreenModels.live(services: services)
        voice = VoiceModels.live(services: services)
        let engine = services.engine
        let speech = services.speech
        let voice = voice
        // Spec 6.1: a recording stops cleanly (the person confirmed quitting)
        // and dictation lands its last phrase, then speech, web, the engine.
        shutdownSteps = [
            {
                await voice.stopLiveSessions()
                voice.setup.stop()
            },
            { await speech.shutdown() },
            { await web.shutdown() },
            { await engine.shutdown() },
        ]
    }

    /// Runs every quit step in order.
    func shutdown() async {
        for step in shutdownSteps {
            await step()
        }
    }
}

/// Shows a permission's one-line reason before macOS asks (L6). Returns
/// whether the person chose to continue to the system prompt.
enum PermissionReasonAlert {
    static let present: LivePermissionBroker.ReasonPresenter = { permission in
        await MainActor.run {
            let alert = NSAlert()
            alert.messageText = title(for: permission)
            alert.informativeText = permission.reason
            alert.addButton(withTitle: "Continue")
            alert.addButton(withTitle: "Not now")
            return alert.runModal() == .alertFirstButtonReturn
        }
    }

    static func title(for permission: AppPermission) -> String {
        switch permission {
        case .calendar: "Mference would like to use Calendar"
        case .reminders: "Mference would like to use Reminders"
        case .camera: "Mference would like to use the camera"
        case .fullDiskAccess: "Mference needs Full Disk Access to read Mail"
        case .mailAutomation: "Mference would like to send through Mail"
        case .microphone: "Mference would like to use the microphone"
        case .systemAudio: "Mference would like to hear call audio"
        }
    }
}

/// Quit waits for the shutdown sequence (`applicationShouldTerminate`
/// returns `.terminateLater` and replies when it is done), but never longer
/// than `shutdownDeadline`: a stalled step must not keep the app from
/// quitting.
final class AppDelegate: NSObject, NSApplicationDelegate {
    let runtime = AppRuntime()
    static let shutdownDeadline: Duration = .seconds(10)
    private var terminating = false
    private var replied = false

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        // A meeting is recording: ask first (spec 6.1, D7).
        if !terminating, runtime.voice.isMeetingActive,
           !QuitConfirmation.confirm(finishing: runtime.voice.isMeetingFinishing) {
            return .terminateCancel
        }
        if !terminating {
            terminating = true
            Task {
                await runtime.shutdown()
                reply(to: sender)
            }
            Task {
                try? await Task.sleep(for: Self.shutdownDeadline)
                if !replied {
                    FileHandle.standardError.write(Data("quit: shutdown passed its deadline; quitting anyway\n".utf8))
                }
                reply(to: sender)
            }
        }
        return .terminateLater
    }

    private func reply(to sender: NSApplication) {
        guard !replied else { return }
        replied = true
        sender.reply(toApplicationShouldTerminate: true)
    }
}

/// The quit confirmation while a meeting records or is still finishing.
enum QuitConfirmation {
    struct Copy: Equatable {
        let title: String
        let message: String
        let quit: String
        let stay: String
    }

    static func copy(finishing: Bool) -> Copy {
        finishing
            ? Copy(title: "A meeting is finishing",
                   message: "Mference is transcribing the last phrases and labelling speakers. Quit waits until the transcript is saved.",
                   quit: "Finish and Quit", stay: "Keep Open")
            : Copy(title: "A meeting is recording",
                   message: "Quitting stops the recording. Everything transcribed so far is saved in the meeting note.",
                   quit: "Stop and Quit", stay: "Keep Recording")
    }

    /// Shows the alert; true when the person chose to quit.
    @MainActor
    static func confirm(finishing: Bool) -> Bool {
        let copy = copy(finishing: finishing)
        let alert = NSAlert()
        alert.messageText = copy.title
        alert.informativeText = copy.message
        alert.addButton(withTitle: copy.quit)
        alert.addButton(withTitle: copy.stay)
        return alert.runModal() == .alertFirstButtonReturn
    }
}
