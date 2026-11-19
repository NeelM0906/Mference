import Foundation
import Synchronization

/// The versioned prompt and tool schemas for web answers
/// (`Web/Resources/web-answers.v<N>.json`, spec 4.5.3), so the evaluation
/// program can load exactly what the app sends.
public struct WebAnswersSpec: Sendable {
    public static let currentVersion = 2

    public let version: Int
    public let systemPromptTemplate: String
    public let tools: [ToolDefinition]

    public static func load(version: Int = currentVersion) throws -> WebAnswersSpec {
        guard let url = Bundle.module.url(forResource: "web-answers.v\(version)", withExtension: "json") else {
            throw CocoaError(.fileNoSuchFile)
        }
        return try parse(Data(contentsOf: url))
    }

    public static func parse(_ data: Data) throws -> WebAnswersSpec {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let version = root["version"] as? Int,
              let prompt = root["systemPrompt"] as? String,
              let tools = root["tools"] as? [[String: Any]] else {
            throw CocoaError(.fileReadCorruptFile)
        }
        let definitions = try tools.map { tool -> ToolDefinition in
            guard let name = tool["name"] as? String, let description = tool["description"] as? String,
                  let parameters = tool["parameters"] else { throw CocoaError(.fileReadCorruptFile) }
            return ToolDefinition(name: name, description: description, parameters: ToolJSON(jsonObject: parameters))
        }
        return WebAnswersSpec(version: version, systemPromptTemplate: prompt, tools: definitions)
    }

    public func systemPrompt(now: Date) -> String {
        let date = now.formatted(.dateTime.weekday(.wide).day().month(.wide).year())
        return systemPromptTemplate.replacingOccurrences(of: "{{date}}", with: date)
    }
}

extension ToolJSON {
    /// From a `JSONSerialization` object.
    public init(jsonObject value: Any) {
        switch value {
        case let dictionary as [String: Any]: self = .object(dictionary.mapValues(ToolJSON.init(jsonObject:)))
        case let array as [Any]: self = .array(array.map(ToolJSON.init(jsonObject:)))
        case let string as String: self = .string(string)
        case let number as NSNumber:
            if CFGetTypeID(number) == CFBooleanGetTypeID() { self = .bool(number.boolValue) }
            else if CFNumberIsFloatType(number) { self = .number(number.doubleValue) }
            else { self = .integer(number.int64Value) }
        default: self = .null
        }
    }

    /// The string value of `key` in an object.
    public subscript(string key: String) -> String? {
        guard case .object(let object) = self, case .string(let value)? = object[key] else { return nil }
        return value
    }
}

/// Runs `web_search` and `open_page` for one answer. Sources get numbers in
/// the order they first appear, so "[2]" means the same page all answer long.
public final class WebToolbox: ChatToolbox {
    public let spec: WebAnswersSpec
    private let searchBase: (@Sendable () async throws -> URL)?
    private let http: WebHTTP
    private let client: SearXNGClient
    private let reader: PageReader
    private let state = Mutex(State())
    /// Pages read per answer.
    public static let maxPages = 4
    /// Longest query sent to SearXNG.
    public static let maxQueryLength = 200
    /// Markers around everything taken from the web (prompt v2: untrusted),
    /// with a random code per tool call so a page cannot guess the closing
    /// marker. Web text inside is also neutralized (no "<<<" or ">>>").
    static func markers() -> (start: String, end: String) {
        let code = (0..<4).map { _ in String(format: "%02x", UInt8.random(in: .min ... .max)) }.joined()
        return ("<<<WEB CONTENT \(code)>>>", "<<<END WEB CONTENT \(code)>>>")
    }

    private struct State {
        var numbers: [String: Int] = [:]
        var citations: [Int: ChatCitation] = [:]
        var pagesRead = 0
    }

    /// `searchBase` returns the running SearXNG's base URL (starting it if
    /// needed); nil when this build has no SearXNG, and the tools say so.
    public init(spec: WebAnswersSpec, http: WebHTTP, searchBase: (@Sendable () async throws -> URL)?) {
        self.spec = spec
        self.searchBase = searchBase
        self.http = http
        self.client = SearXNGClient(http: http)
        self.reader = PageReader(http: http)
    }

    public func systemPrompt(now: Date) -> String? { spec.systemPrompt(now: now) }

    public var definitions: [ToolDefinition] { spec.tools }

    public func run(_ call: ToolCall) async -> ToolOutcome {
        switch call.name {
        case "web_search":
            guard let raw = call.arguments[string: "query"]?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !raw.isEmpty else {
                return failure("web_search needs a non-empty \"query\".")
            }
            let query = String(raw.split(whereSeparator: \.isNewline).joined(separator: " ").prefix(Self.maxQueryLength))
            return await search(query)
        case "open_page":
            guard let string = call.arguments[string: "url"], let url = URL(string: string) else {
                return failure("open_page needs the \"url\" of a search result.")
            }
            return await open(url)
        default:
            return failure("There is no tool named \(call.name).")
        }
    }

    // MARK: Tools

    private func search(_ query: String) async -> ToolOutcome {
        guard let searchBase else {
            return failure(WebError.searchUnavailable("it is not included in this build of Mference.").message)
        }
        do {
            let base = try await searchBase()
            http.allowlist.setSearchBase(base)
            let results = try await client.search(query, base: base)
            guard !results.isEmpty else {
                return ToolOutcome(content: "No results for “\(query)”.", steps: [.search(query, results: 0)])
            }
            let markers = Self.markers()
            var lines = ["Results for “\(query)”:", markers.start]
            var cited: [ChatCitation] = []
            for result in results {
                http.allowlist.allowPage(result.url)
                let citation = register(result.url, title: result.title, snippet: result.content)
                cited.append(citation)
                lines.append("")
                lines.append("[\(citation.number)] \(Self.neutralize(result.title))")
                lines.append(result.url.absoluteString)
                if let snippet = result.content { lines.append(Self.neutralize(String(snippet.prefix(300)))) }
            }
            lines.append(markers.end)
            return ToolOutcome(content: lines.joined(separator: "\n"),
                               steps: [.search(query, results: results.count)], citations: cited)
        } catch {
            return failure(Self.message(error))
        }
    }

    private func open(_ url: URL) async -> ToolOutcome {
        guard http.allowlist.isAllowedPage(url) else { return failure(WebError.notAllowed(url).message) }
        let allowed = state.withLock { state -> Bool in
            guard state.pagesRead < Self.maxPages else { return false }
            state.pagesRead += 1
            return true
        }
        guard allowed else {
            return failure("Already read \(Self.maxPages) pages for this answer. Answer with what you have.")
        }
        do {
            let page = try await reader.read(url)
            let citation = register(url, title: page.title, snippet: nil)
            let markers = Self.markers()
            var content = "Page [\(citation.number)]: \(url.absoluteString)\n\(markers.start)\n"
            content += "\(Self.neutralize(page.title ?? citation.title))\n\n\(Self.neutralize(page.text))\n\(markers.end)"
            if page.isTruncated { content += "\n(The page continues; this is the first part.)" }
            return ToolOutcome(content: content, steps: [.readPage(url, title: page.title)], citations: [citation])
        } catch {
            return failure(Self.message(error))
        }
    }

    // MARK: Helpers

    private func register(_ url: URL, title: String?, snippet: String?) -> ChatCitation {
        let key = WebAllowlist.key(url) ?? url.absoluteString
        return state.withLock { state in
            if let number = state.numbers[key], var citation = state.citations[number] {
                if citation.snippet == nil { citation.snippet = snippet }
                if let title, !title.isEmpty, citation.title == citation.url.host() { citation.title = title }
                state.citations[number] = citation
                return citation
            }
            let number = state.numbers.count + 1
            state.numbers[key] = number
            let citation = ChatCitation(number: number, url: url, title: title ?? url.host() ?? url.absoluteString,
                                        snippet: snippet)
            state.citations[number] = citation
            return citation
        }
    }

    /// Page text cannot close the untrusted-content block early.
    public static func neutralize(_ text: String) -> String {
        text.replacingOccurrences(of: "<<<", with: "‹‹‹").replacingOccurrences(of: ">>>", with: "›››")
    }

    private func failure(_ message: String) -> ToolOutcome {
        ToolOutcome(content: message, steps: [ChatStep(kind: .failed, detail: message)])
    }

    static func message(_ error: any Error) -> String {
        if let error = error as? WebError { return error.message }
        if error is CancellationError { return "Stopped." }
        return "The web request failed: \(error.localizedDescription)"
    }
}

/// Gives the chat its web tools and runs SearXNG only while web search is on.
/// One per app; `shutdown()` on quit.
///
/// The master switch is `AppSettings.webSearchEnabled` (Settings > Privacy,
/// the person's consent, L7), together with offline mode
/// (`AppSettings.isWebSearchActive`). A chat's own Web toggle works only while
/// that is on; with it off, turning a chat's Web on asks for consent, and
/// agreeing turns the master switch on.
@MainActor
public final class WebAnswers: ChatToolboxProvider {
    /// The pre-AppSettings key, read once and removed (see `migrateLegacyConsent`).
    public static let legacyEnabledKey = "web.searchEnabled"

    public let settings: AppSettings
    public let process: SearXNGProcess?
    private let spec: WebAnswersSpec?
    private let protocolClasses: [AnyClass]?

    /// Lives as long as the app (one `WebAnswers` per app).
    private var terminationObserver: (any NSObjectProtocol)?

    /// `legacyStore` is where an older build kept its own web switch
    /// (`UserDefaults.standard` in the app).
    public init(settings: AppSettings,
                runtime: SearXNGRuntime? = SearXNGRuntime.locate(),
                protocolClasses: [AnyClass]? = nil,
                legacyStore: any SettingsStore = UserDefaults.standard) {
        self.settings = settings
        let process = runtime.map { SearXNGProcess(runtime: $0) }
        self.process = process
        self.spec = try? WebAnswersSpec.load()
        self.protocolClasses = protocolClasses
        Self.migrateLegacyConsent(from: legacyStore, into: settings)
        // Belt and braces for quit: the shell awaits `shutdown()`; if the app
        // terminates without it, SearXNG still gets SIGTERM synchronously.
        // (`NSApplication.willTerminateNotification`, by name: Core does not
        // import AppKit.)
        terminationObserver = NotificationCenter.default.addObserver(
            forName: Notification.Name("NSApplicationWillTerminateNotification"), object: nil, queue: nil) { _ in
            process?.terminateNow()
        }
        followMasterSwitch()
    }

    /// An earlier build stored its own consent under `web.searchEnabled`. If
    /// the person had turned it on, carry that over once; then drop the key.
    static func migrateLegacyConsent(from store: any SettingsStore, into settings: AppSettings) {
        guard let legacy = store.object(forKey: legacyEnabledKey) as? Bool else { return }
        if legacy, !settings.webSearchEnabled { settings.webSearchEnabled = true }
        store.set(nil, forKey: legacyEnabledKey)
    }

    /// Stops SearXNG whenever web search stops being active (switched off in
    /// Settings, or offline mode turned on).
    private func followMasterSwitch() {
        let active = withObservationTracking {
            settings.isWebSearchActive
        } onChange: { [weak self] in
            Task { @MainActor in self?.followMasterSwitch() }
        }
        if !active, let process { Task { await process.stop() } }
    }

    public var needsWebConsent: Bool { !settings.webSearchEnabled }

    public func grantWebConsent() {
        settings.webSearchEnabled = true
    }

    public func makeToolbox(webEnabled: Bool) -> (any ChatToolbox)? {
        guard webEnabled, settings.isWebSearchActive, let spec else { return nil }
        let http = WebHTTP(allowlist: WebAllowlist(), protocolClasses: protocolClasses)
        var searchBase: (@Sendable () async throws -> URL)?
        if let process {
            searchBase = { try await process.ensureRunning() }
        }
        return WebToolbox(spec: spec, http: http, searchBase: searchBase)
    }

    public func webUsageChanged(enabled: Bool) {
        guard let process else { return }
        if enabled, settings.isWebSearchActive {
            Task { await process.warmUp() }
        } else if !enabled {
            Task { await process.stop() }
        }
    }

    /// Stops SearXNG; call on quit.
    public func shutdown() async {
        await process?.stop()
    }
}
