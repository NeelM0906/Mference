import Foundation
import Synchronization

/// Every web request the app makes goes through `WebHTTP`, which enforces the
/// network allowlist (spec 4.4, gate G-A8): the loopback SearXNG process, and
/// pages SearXNG returned for the answer being written. Anything else is
/// refused before a socket opens.
public final class WebAllowlist: Sendable {
    private struct State {
        var searchBase: URL?
        var pages: Set<String> = []
    }

    private let state = Mutex(State())

    public init() {}

    /// The running SearXNG (always `127.0.0.1:<port>`).
    public func setSearchBase(_ url: URL?) {
        state.withLock { $0.searchBase = url }
    }

    /// Allows `url` (a search result) for `open_page`.
    public func allowPage(_ url: URL) {
        guard let key = Self.key(url) else { return }
        state.withLock { _ = $0.pages.insert(key) }
    }

    public func isAllowedPage(_ url: URL) -> Bool {
        guard let key = Self.key(url) else { return false }
        return state.withLock { $0.pages.contains(key) }
    }

    /// Whether a request to `url` may be sent at all.
    public func permits(_ url: URL) -> Bool {
        if let base = state.withLock({ $0.searchBase }), Self.sameOrigin(url, base) { return true }
        return isAllowedPage(url)
    }

    /// A redirect is followed only within the host the page was allowed for
    /// (http to https, a trailing path change); the new URL is then allowed.
    public func permitsRedirect(from original: URL, to target: URL) -> Bool {
        guard isAllowedPage(original), Self.isWebURL(target),
              Self.bareHost(original) == Self.bareHost(target) else { return false }
        allowPage(target)
        return true
    }

    static func isWebURL(_ url: URL) -> Bool {
        guard let scheme = url.scheme?.lowercased(), scheme == "https" || scheme == "http",
              let host = url.host(), !host.isEmpty else { return false }
        return true
    }

    /// Pages are matched without fragment; scheme and host case-folded.
    static func key(_ url: URL) -> String? {
        guard isWebURL(url), var components = URLComponents(url: url, resolvingAgainstBaseURL: false) else { return nil }
        components.fragment = nil
        components.scheme = components.scheme?.lowercased()
        components.host = components.host?.lowercased()
        if components.path.isEmpty { components.path = "/" }
        return components.string
    }

    static func bareHost(_ url: URL) -> String? {
        guard var host = url.host()?.lowercased() else { return nil }
        if host.hasPrefix("www.") { host.removeFirst(4) }
        return host
    }

    static func sameOrigin(_ a: URL, _ b: URL) -> Bool {
        a.scheme?.lowercased() == b.scheme?.lowercased() && a.host()?.lowercased() == b.host()?.lowercased()
            && a.port == b.port
    }
}

/// Why a web request did not produce a page.
public enum WebError: Error, Sendable, Hashable {
    case notAllowed(URL)
    case badStatus(Int)
    case tooLarge
    case unsupportedContent(String)
    case searchUnavailable(String)
    case transport(String)

    /// A sentence for the tool message (the model reads it) and the steps.
    public var message: String {
        switch self {
        case .notAllowed(let url): "Not opened: \(url.host() ?? url.absoluteString) was not in the search results."
        case .badStatus(let code): "The page answered with HTTP \(code)."
        case .tooLarge: "The page is too large to read."
        case .unsupportedContent(let type): "The page is \(type), not a web page."
        case .searchUnavailable(let reason): "Web search is not available: \(reason)"
        case .transport(let reason): "The page could not be loaded: \(reason)"
        }
    }
}

/// The app's HTTP client for web answers: an ephemeral session (no cookies,
/// no cache, no credentials), a fixed user agent, bounded bodies, and the
/// allowlist checked before every request and every redirect.
public final class WebHTTP: Sendable {
    public let allowlist: WebAllowlist
    /// Largest body read, in bytes.
    public let maxBytes: Int
    private let session: URLSession

    /// `protocolClasses` lets tests stub the transport (G-A8 test).
    public init(allowlist: WebAllowlist, maxBytes: Int = 2_000_000, timeout: TimeInterval = 12,
                protocolClasses: [AnyClass]? = nil) {
        self.allowlist = allowlist
        self.maxBytes = maxBytes
        let configuration = URLSessionConfiguration.ephemeral
        configuration.httpCookieAcceptPolicy = .never
        configuration.httpShouldSetCookies = false
        configuration.urlCache = nil
        configuration.urlCredentialStorage = nil
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.timeoutIntervalForRequest = timeout
        configuration.timeoutIntervalForResource = timeout * 2
        configuration.httpAdditionalHeaders = [
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Safari/605.1.15",
            "Accept-Language": "en",
        ]
        if let protocolClasses { configuration.protocolClasses = protocolClasses }
        session = URLSession(configuration: configuration, delegate: RedirectGuard(allowlist: allowlist),
                             delegateQueue: nil)
    }

    deinit {
        session.invalidateAndCancel()
    }

    /// GETs `url` if the allowlist permits it; returns at most `maxBytes`.
    public func get(_ url: URL, accept: String = "*/*") async throws -> (data: Data, response: HTTPURLResponse) {
        guard allowlist.permits(url) else { throw WebError.notAllowed(url) }
        var request = URLRequest(url: url)
        request.setValue(accept, forHTTPHeaderField: "Accept")
        let bytes: URLSession.AsyncBytes
        let response: URLResponse
        do {
            (bytes, response) = try await session.bytes(for: request)
        } catch let error as URLError {
            throw WebError.transport(error.localizedDescription)
        }
        guard let http = response as? HTTPURLResponse else { throw WebError.transport("not HTTP") }
        guard (200..<300).contains(http.statusCode) else { throw WebError.badStatus(http.statusCode) }
        if http.expectedContentLength > Int64(maxBytes) * 4 { throw WebError.tooLarge }
        var data = Data()
        data.reserveCapacity(min(Int(max(http.expectedContentLength, 0)), maxBytes))
        for try await byte in bytes {
            data.append(byte)
            if data.count >= maxBytes { break }
        }
        return (data, http)
    }
}

/// Follows a redirect only when the allowlist permits it.
final class RedirectGuard: NSObject, URLSessionTaskDelegate, Sendable {
    let allowlist: WebAllowlist

    init(allowlist: WebAllowlist) {
        self.allowlist = allowlist
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest) async -> URLRequest? {
        guard let from = task.originalRequest?.url, let to = request.url else { return nil }
        if allowlist.permits(to), WebAllowlist.sameOrigin(from, to) { return request }
        return allowlist.permitsRedirect(from: from, to: to) ? request : nil
    }
}
