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
    /// The menu-bar extra's view of the engine (chat's engine, so Apple's
    /// model is listed while Settings allows it).
    let menuBar: MenuBarModel
    private var menuBarTask: Task<Void, Never>?

    /// Quit steps, run in order: web answers (SearXNG) and Find stop before
    /// the engine shuts down (U3's notes).
    private let shutdownSteps: [@MainActor () async -> Void]

    /// One composition root for the whole app: every engine use (chat, model
    /// library, Scan, Calendar, Mail) goes through `services.engine`, so only
    /// one engine host can hold a model. Chat goes through
    /// `services.chatEngine`, the same engine plus Apple's on-device model. Nothing here loads or warms a model;
    /// that happens only when the person picks a model or opens a chat.
    init(services: AppServices = .live(presentPermissionReason: PermissionReasonAlert.present,
                                       findHelper: Bundle.main.executableURL),
         web: WebAnswers? = nil) {
        self.services = services
        // Web search follows Settings > Privacy (services.settings).
        let web = web ?? WebAnswers(settings: services.settings)
        self.web = web
        chat = ChatViewModel(service: services.chatEngine,
                             store: ConversationStore(container: services.modelContainer),
                             tools: web,
                             files: services.find,
                             defaultModel: { [settings = services.settings] in settings.defaultModel })
        screenModels = ScreenModels.live(services: services)
        let menuBar = MenuBarModel(engine: services.chatEngine)
        self.menuBar = menuBar
        let engine = services.engine
        let find = services.find
        shutdownSteps = [
            { await web.shutdown() },
            // Find frees its search model before the engine shuts down.
            { await find?.shutdown() },
            { await engine.shutdown() },
        ]
        // Find picks up where it left off (nothing runs while it is off).
        Task { await find?.start() }
        menuBarTask = Task { await menuBar.run() }
    }

    /// Runs every quit step in order.
    func shutdown() async {
        menuBarTask?.cancel()
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
        case .photos: "Mference would like to read your Photos library"
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
