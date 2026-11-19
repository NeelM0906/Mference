import Foundation
import MferenceAppCore
import Testing

/// Web search follows Settings > Privacy (`AppSettings.webSearchEnabled`), the
/// one consent switch. No model, no network: a missing SearXNG runtime and the
/// scripted engine.
@MainActor
@Suite struct WebSettingsTests {
    /// A runtime directory that does not exist: SearXNG never actually starts.
    static func web(_ settings: AppSettings, legacy: InMemorySettingsStore = InMemorySettingsStore()) -> WebAnswers {
        let root = FileManager.default.temporaryDirectory.appending(path: "no-searxng-\(UUID())")
        return WebAnswers(settings: settings, runtime: SearXNGRuntime(root: root), legacyStore: legacy)
    }

    static func settings(webOn: Bool, offline: Bool = false) -> AppSettings {
        let settings = AppSettings(store: InMemorySettingsStore())
        settings.webSearchEnabled = webOn
        settings.offlineMode = offline
        return settings
    }

    @Test func settingsOnLetsTheChatSearch() async throws {
        let settings = Self.settings(webOn: true)
        let web = Self.web(settings)
        #expect(!web.needsWebConsent)
        let toolbox = try #require(web.makeToolbox(webEnabled: true))
        #expect(toolbox.definitions.map(\.name) == ["web_search", "open_page"])
        #expect(web.makeToolbox(webEnabled: false) == nil, "a chat with its own toggle off gets no tools")

        let service = ScriptedEngineService(replies: [.init(content: "Answer.")])
        let chat = ChatViewModel(service: service, store: ConversationStore(container: try AppModelContainer.inMemory()),
                                 tools: web)
        chat.setWebEnabled(true)
        #expect(!chat.webConsentRequested)
        #expect(chat.webEnabled)
        chat.draft = "Q"
        chat.send()
        await chat.waitUntilIdle()
        #expect(service.requests.first?.options.tools.map(\.name) == ["web_search", "open_page"])
    }

    @Test func settingsOffOffersNoToolsAndStartsNothing() async throws {
        let settings = Self.settings(webOn: false)
        let web = Self.web(settings)
        #expect(web.needsWebConsent)
        #expect(web.makeToolbox(webEnabled: true) == nil)
        web.webUsageChanged(enabled: true)
        try await Task.sleep(for: .milliseconds(200))
        let process = try #require(web.process)
        #expect(await process.state == .stopped, "SearXNG must not start while web search is off")

        // The chat's own toggle asks for consent, and agreeing turns the
        // Settings switch on.
        let service = ScriptedEngineService(replies: [.init(content: "Answer.")])
        let chat = ChatViewModel(service: service, store: ConversationStore(container: try AppModelContainer.inMemory()),
                                 tools: web)
        chat.setWebEnabled(true)
        #expect(chat.webConsentRequested)
        #expect(!chat.webEnabled)
        chat.declineWebConsent()
        chat.draft = "Q"
        chat.send()
        await chat.waitUntilIdle()
        #expect(service.requests.first?.options.tools.isEmpty == true)

        chat.setWebEnabled(true)
        chat.confirmWebConsent()
        #expect(settings.webSearchEnabled, "consent in the chat is the Settings switch")
        #expect(web.makeToolbox(webEnabled: true) != nil)
    }

    @Test func offlineWithWebOnOffersNoToolAndStartsNothing() async throws {
        let settings = Self.settings(webOn: true, offline: true)
        let web = Self.web(settings)
        #expect(web.makeToolbox(webEnabled: true) == nil)
        #expect(web.webUnavailableReason == "Offline mode is on")
        web.webUsageChanged(enabled: true)
        try await Task.sleep(for: .milliseconds(200))
        let process = try #require(web.process)
        #expect(await process.state == .stopped, "SearXNG must not start in offline mode")

        // The chat's Web toggle shows the reason and stays off, with no
        // consent prompt; an answer gets no tools.
        let service = ScriptedEngineService(replies: [.init(content: "Answer.")])
        let chat = ChatViewModel(service: service, store: ConversationStore(container: try AppModelContainer.inMemory()),
                                 tools: web)
        #expect(chat.webBlockedReason == "Offline mode is on")
        chat.setWebEnabled(true)
        #expect(!chat.webEnabled)
        #expect(!chat.webConsentRequested)
        chat.draft = "Q"
        chat.send()
        await chat.waitUntilIdle()
        #expect(service.requests.first?.options.tools.isEmpty == true)

        // Back online, the same chat can turn web on without asking again.
        settings.offlineMode = false
        #expect(chat.webBlockedReason == nil)
        chat.setWebEnabled(true)
        #expect(chat.webEnabled)
        #expect(web.makeToolbox(webEnabled: true) != nil)
    }

    @Test func goingOfflineStopsSearXNG() async throws {
        let settings = Self.settings(webOn: true)
        let web = Self.web(settings)
        let process = try #require(web.process)
        settings.offlineMode = true
        let off = WebError.searchUnavailable("web search is off for now.")
        var refused = false
        for _ in 0..<200 where !refused {
            do { _ = try await process.ensureRunning() } catch let error as WebError where error == off {
                refused = true
            } catch {}
            if !refused { try await Task.sleep(for: .milliseconds(10)) }
        }
        #expect(refused, "SearXNG still startable after offline mode turned on")
    }

    @Test func turningTheSwitchOffStopsSearXNG() async throws {
        let settings = Self.settings(webOn: true)
        let web = Self.web(settings)
        let process = try #require(web.process)
        settings.webSearchEnabled = false
        // The stop runs on a task; once it has, nothing may start SearXNG.
        let off = WebError.searchUnavailable("web search is off for now.")
        var refused = false
        for _ in 0..<200 where !refused {
            do { _ = try await process.ensureRunning() } catch let error as WebError where error == off {
                refused = true
            } catch {}
            if !refused { try await Task.sleep(for: .milliseconds(10)) }
        }
        #expect(refused, "SearXNG still startable after Settings turned web search off")
    }

    /// Codex P1 on #140: an answer already running keeps its toolbox, so the
    /// toolbox re-reads Settings before every search or page fetch.
    @Test func aRunningAnswerLosesWebWhenSettingsTurnOff() async throws {
        for turnOff in ["switch", "offline"] {
            let settings = Self.settings(webOn: true)
            let web = Self.web(settings)
            let toolbox = try #require(web.makeToolbox(webEnabled: true))
            if turnOff == "switch" { settings.webSearchEnabled = false } else { settings.offlineMode = true }
            for (name, arguments) in [("web_search", ["query": "q"]), ("open_page", ["url": "https://a.example/"])] {
                let call = ToolCall(id: name, name: name, arguments: .object(arguments.mapValues(ToolJSON.string)),
                                    argumentsJSON: "{}")
                let outcome = await toolbox.run(call)
                #expect(outcome.steps.first?.kind == .failed, "\(turnOff): \(name) still ran")
                #expect(outcome.content.contains("turned off"), "\(turnOff): \(outcome.content)")
            }
        }
    }

    /// Codex P2 on #140: off (or offline) and then on again, a chat whose Web
    /// toggle stayed on can search again without toggling it.
    @Test func searchResumesWhenSettingsTurnBackOn() async throws {
        for turnOff in ["switch", "offline"] {
            let settings = Self.settings(webOn: true)
            let web = Self.web(settings)
            let process = try #require(web.process)
            if turnOff == "switch" { settings.webSearchEnabled = false } else { settings.offlineMode = true }
            // Wait until the stop has happened.
            let off = WebError.searchUnavailable("web search is off for now.")
            var stopped = false
            for _ in 0..<200 where !stopped {
                do { _ = try await process.ensureRunning() } catch let error as WebError where error == off {
                    stopped = true
                } catch {}
                if !stopped { try await Task.sleep(for: .milliseconds(10)) }
            }
            #expect(stopped)
            if turnOff == "switch" { settings.webSearchEnabled = true } else { settings.offlineMode = false }
            let toolbox = try #require(web.makeToolbox(webEnabled: true))
            let call = ToolCall(id: "s", name: "web_search", arguments: .object(["query": .string("q")]), argumentsJSON: "{}")
            let outcome = await toolbox.run(call)
            // The runtime here does not exist, so the start fails, but it is
            // attempted: not refused as "off".
            #expect(!outcome.content.contains("off for now"), "\(turnOff): \(outcome.content)")
            #expect(!outcome.content.contains("turned off"), "\(turnOff): \(outcome.content)")
        }
    }

    @Test func legacySwitchMigratesOnceThenGoes() {
        let legacy = InMemorySettingsStore([WebAnswers.legacyEnabledKey: true])
        let settings = Self.settings(webOn: false)
        _ = Self.web(settings, legacy: legacy)
        #expect(settings.webSearchEnabled)
        #expect(legacy.object(forKey: WebAnswers.legacyEnabledKey) == nil)

        let declined = InMemorySettingsStore([WebAnswers.legacyEnabledKey: false])
        let other = Self.settings(webOn: false)
        _ = Self.web(other, legacy: declined)
        #expect(!other.webSearchEnabled)
        #expect(declined.object(forKey: WebAnswers.legacyEnabledKey) == nil)
    }
}
