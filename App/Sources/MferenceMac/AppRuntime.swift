import AppKit
import MferenceAppCore
import SwiftData

/// The app's long-lived objects, built once at launch and kept for the app's
/// lifetime: the engine and the one chat model (U3's notes: switching pills
/// must not drop a running answer).
///
/// When U2's `AppServices.live()` lands (#102), the engine and store come
/// from it; when U3's `WebAnswers` lands (#112), it becomes the chat's
/// `tools` and its shutdown is added before the engine's (see `init`).
@MainActor
final class AppRuntime {
    let engine: EngineService
    let chat: ChatViewModel
    /// Set when the on-disk store could not be opened (nothing is saved).
    let storeError: String?

    /// Quit steps, run in order. Web answers (SearXNG) must stop before the
    /// engine shuts down.
    private var shutdownSteps: [@MainActor () async -> Void] = []

    init() {
        engine = EngineService.live()
        var storeError: String?
        let container: ModelContainer
        do {
            container = try AppModelContainer.make()
        } catch {
            // Run on an in-memory store this session; the UI can say nothing is saved.
            storeError = String(describing: error)
            do {
                container = try AppModelContainer.inMemory()
            } catch {
                fatalError("The in-memory store did not open (a schema bug the tests catch): \(error)")
            }
        }
        self.storeError = storeError
        chat = ChatViewModel(service: engine, store: ConversationStore(container: container), tools: nil)
        let engine = engine
        shutdownSteps.append { await engine.shutdown() }
    }

    /// Runs every quit step in order.
    func shutdown() async {
        for step in shutdownSteps {
            await step()
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
