import Foundation
import Synchronization

/// The versioned prompt and tool schemas for web answers
/// (`Web/Resources/web-answers.v<N>.json`, spec 4.5.3), so the evaluation
/// program can load exactly what the app sends.
public struct WebAnswersSpec: Sendable {
    public static let currentVersion = 1

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
            guard let query = call.arguments[string: "query"]?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !query.isEmpty else {
                return failure("web_search needs a non-empty \"query\".")
            }
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
            var lines = ["Results for “\(query)”:"]
            var cited: [ChatCitation] = []
            for result in results {
                http.allowlist.allowPage(result.url)
                let citation = register(result.url, title: result.title, snippet: result.content)
                cited.append(citation)
                lines.append("")
                lines.append("[\(citation.number)] \(result.title)")
                lines.append(result.url.absoluteString)
                if let snippet = result.content { lines.append(String(snippet.prefix(300))) }
            }
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
            var content = "[\(citation.number)] \(page.title ?? citation.title)\n\(url.absoluteString)\n\n\(page.text)"
            if page.isTruncated { content += "\n\n(The page continues; this is the first part.)" }
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

    private func failure(_ message: String) -> ToolOutcome {
        ToolOutcome(content: message, steps: [ChatStep(kind: .failed, detail: message)])
    }

    static func message(_ error: any Error) -> String {
        if let error = error as? WebError { return error.message }
        if error is CancellationError { return "Stopped." }
        return "The web request failed: \(error.localizedDescription)"
    }
}

/// The person's opt-in for web search (L7: off until turned on). Settings >
/// Privacy shows the same switch.
public struct WebSearchSettings: Sendable {
    public static let enabledKey = "web.searchEnabled"
    private let suiteName: String?

    /// `suiteName` nil: the app's standard defaults.
    public init(suiteName: String? = nil) {
        self.suiteName = suiteName
    }

    private var defaults: UserDefaults { suiteName.flatMap(UserDefaults.init(suiteName:)) ?? .standard }

    public var isEnabled: Bool {
        get { defaults.bool(forKey: Self.enabledKey) }
        nonmutating set { defaults.set(newValue, forKey: Self.enabledKey) }
    }
}

/// Gives the chat its web tools and runs SearXNG only while some chat has web
/// on. One per app; `shutdown()` on quit.
@MainActor
public final class WebAnswers: ChatToolboxProvider {
    public let settings: WebSearchSettings
    public let process: SearXNGProcess?
    private let spec: WebAnswersSpec?
    private let protocolClasses: [AnyClass]?

    public init(settings: WebSearchSettings = WebSearchSettings(),
                runtime: SearXNGRuntime? = SearXNGRuntime.locate(),
                protocolClasses: [AnyClass]? = nil) {
        self.settings = settings
        self.process = runtime.map { SearXNGProcess(runtime: $0) }
        self.spec = try? WebAnswersSpec.load()
        self.protocolClasses = protocolClasses
    }

    public var needsWebConsent: Bool { !settings.isEnabled }

    public func grantWebConsent() {
        settings.isEnabled = true
    }

    public func makeToolbox(webEnabled: Bool) -> (any ChatToolbox)? {
        guard webEnabled, settings.isEnabled, let spec else { return nil }
        let http = WebHTTP(allowlist: WebAllowlist(), protocolClasses: protocolClasses)
        var searchBase: (@Sendable () async throws -> URL)?
        if let process {
            searchBase = { try await process.ensureRunning() }
        }
        return WebToolbox(spec: spec, http: http, searchBase: searchBase)
    }

    public func webUsageChanged(enabled: Bool) {
        guard let process else { return }
        if enabled, settings.isEnabled {
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
