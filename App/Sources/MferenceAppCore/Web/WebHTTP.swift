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
        if isSearchBase(url) { return true }
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

    /// http(s), no credentials, a usual web port, and a host that passes
    /// `HostPolicy` (no LAN, loopback, link-local or local names).
    static func isWebURL(_ url: URL) -> Bool {
        guard let scheme = url.scheme?.lowercased(), scheme == "https" || scheme == "http",
              let host = url.host(), !host.isEmpty, url.user() == nil, url.password() == nil,
              HostPolicy.isAllowedHost(host) else { return false }
        if let port = url.port, ![80, 443, 8080, 8443].contains(port) { return false }
        return true
    }

    /// Whether `url` is the running SearXNG (the one loopback destination).
    func isSearchBase(_ url: URL) -> Bool {
        guard let base = state.withLock({ $0.searchBase }) else { return false }
        return Self.sameOrigin(url, base)
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
    /// Not a web page (the server's own type text is not passed on: it would
    /// reach the model outside the untrusted-content markers).
    case unsupportedContent
    case searchUnavailable(String)
    case transport(String)

    /// A sentence for the tool message (the model reads it) and the steps.
    public var message: String {
        switch self {
        case .notAllowed(let url): "Not opened: \(url.host() ?? url.absoluteString) was not in the search results."
        case .badStatus(let code): "The page answered with HTTP \(code)."
        case .tooLarge: "The page is too large to read."
        case .unsupportedContent: "The page is not a web page (a file or other content type)."
        case .searchUnavailable(let reason): "Web search is not available: \(reason)"
        case .transport(let reason): "The page could not be loaded: \(reason)"
        }
    }
}

/// The app's HTTP client for web answers: an ephemeral session (no cookies,
/// no cache, no credentials), a fixed user agent, bounded bodies, and the
/// allowlist checked before every request and every redirect. A page's host
/// is resolved first and every address must be public; after the fetch, the
/// address actually connected to is checked again (DNS rebinding), and the
/// body is discarded if it was not public.
public final class WebHTTP: Sendable {
    public let allowlist: WebAllowlist
    /// Largest body read, in bytes.
    public let maxBytes: Int
    let resolver: HostPolicy.Resolver
    /// Refuse a page whose connected address cannot be confirmed public.
    let requiresConnectedAddress: Bool
    private let session: URLSession

    /// `protocolClasses` and `resolver` let tests stub the transport and DNS
    /// (G-A8 test); the defaults are the system's. A stub transport reports no
    /// connected address, so tests over one pass
    /// `requiresConnectedAddress: false`; the app never does.
    public init(allowlist: WebAllowlist, maxBytes: Int = 2_000_000, timeout: TimeInterval = 12,
                protocolClasses: [AnyClass]? = nil, resolver: HostPolicy.Resolver? = nil,
                requiresConnectedAddress: Bool = true) {
        self.allowlist = allowlist
        self.maxBytes = maxBytes
        self.requiresConnectedAddress = requiresConnectedAddress
        let resolver = resolver ?? HostPolicy.systemResolver
        self.resolver = resolver
        let configuration = URLSessionConfiguration.ephemeral
        configuration.httpCookieAcceptPolicy = .never
        configuration.httpShouldSetCookies = false
        configuration.urlCache = nil
        configuration.urlCredentialStorage = nil
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.timeoutIntervalForRequest = timeout
        configuration.timeoutIntervalForResource = timeout * 2
        configuration.connectionProxyDictionary = [:]
        configuration.httpAdditionalHeaders = [
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Safari/605.1.15",
            "Accept-Language": "en",
        ]
        if let protocolClasses { configuration.protocolClasses = protocolClasses }
        session = URLSession(configuration: configuration,
                             delegate: RedirectGuard(allowlist: allowlist, resolver: resolver),
                             delegateQueue: nil)
    }

    deinit {
        session.invalidateAndCancel()
    }

    /// GETs `url` if the allowlist permits it; returns at most `maxBytes`.
    public func get(_ url: URL, accept: String = "*/*") async throws -> (data: Data, response: HTTPURLResponse) {
        guard allowlist.permits(url) else { throw WebError.notAllowed(url) }
        let isSearch = allowlist.isSearchBase(url)
        if !isSearch { try await Self.requirePublic(url, resolver: resolver) }
        var request = URLRequest(url: url)
        request.setValue(accept, forHTTPHeaderField: "Accept")
        let probe = ConnectionProbe()
        let bytes: URLSession.AsyncBytes
        let response: URLResponse
        do {
            (bytes, response) = try await session.bytes(for: request, delegate: probe)
        } catch let error as URLError {
            throw WebError.transport(error.localizedDescription)
        }
        // Read the body (or decide not to), then confirm where we connected
        // before any outcome, including an error status, reaches the model.
        var failure: (any Error)?
        var data = Data()
        if let http = response as? HTTPURLResponse {
            if !(200..<300).contains(http.statusCode) {
                failure = WebError.badStatus(http.statusCode)
            } else if http.expectedContentLength > Int64(maxBytes) * 4 {
                failure = WebError.tooLarge
            } else {
                data.reserveCapacity(min(Int(max(http.expectedContentLength, 0)), maxBytes))
                do {
                    for try await byte in bytes {
                        data.append(byte)
                        if data.count >= maxBytes { break }
                    }
                } catch let error as URLError {
                    failure = WebError.transport(error.localizedDescription)
                } catch {
                    failure = error
                }
            }
        } else {
            failure = WebError.transport("not HTTP")
        }
        if failure != nil || data.count >= maxBytes { bytes.task.cancel() }
        if !isSearch {
            let remote = await probe.remoteAddress()
            let confirmed = remote.flatMap(IPAddress.init)?.isPublic
            if confirmed == false || (confirmed == nil && requiresConnectedAddress) {
                throw WebError.notAllowed(url)
            }
        }
        if let failure { throw failure }
        guard let http = response as? HTTPURLResponse else { throw WebError.transport("not HTTP") }
        return (data, http)
    }

    static func requirePublic(_ url: URL, resolver: HostPolicy.Resolver) async throws {
        guard let host = url.host() else { throw WebError.notAllowed(url) }
        if let literal = IPAddress(host) {
            guard literal.isPublic else { throw WebError.notAllowed(url) }
            return
        }
        let addresses = try await resolver(host)
        guard !addresses.isEmpty, addresses.allSatisfy(\.isPublic) else { throw WebError.notAllowed(url) }
    }
}

/// Follows a redirect only when the allowlist permits it and the new host
/// resolves to public addresses only.
final class RedirectGuard: NSObject, URLSessionTaskDelegate, Sendable {
    let allowlist: WebAllowlist
    let resolver: HostPolicy.Resolver

    init(allowlist: WebAllowlist, resolver: @escaping HostPolicy.Resolver) {
        self.allowlist = allowlist
        self.resolver = resolver
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest) async -> URLRequest? {
        guard let from = task.originalRequest?.url, let to = request.url else { return nil }
        if allowlist.isSearchBase(from) {
            return allowlist.isSearchBase(to) ? request : nil
        }
        guard allowlist.permitsRedirect(from: from, to: to) else { return nil }
        guard (try? await WebHTTP.requirePublic(to, resolver: resolver)) != nil else { return nil }
        return request
    }
}

/// Records the remote address a task connected to (from its metrics).
final class ConnectionProbe: NSObject, URLSessionTaskDelegate, Sendable {
    private struct State {
        var address: String??
        var timedOut = false
        var waiters: [CheckedContinuation<String?, Never>] = []
    }

    private let state = Mutex(State())

    func urlSession(_ session: URLSession, task: URLSessionTask, didFinishCollecting metrics: URLSessionTaskMetrics) {
        let address = metrics.transactionMetrics.last(where: { $0.remoteAddress != nil })?.remoteAddress
        let waiters = state.withLock { state -> [CheckedContinuation<String?, Never>] in
            state.address = .some(address)
            defer { state.waiters.removeAll() }
            return state.waiters
        }
        for waiter in waiters { waiter.resume(returning: address) }
    }

    /// The connected address, waiting briefly for the metrics; nil when the
    /// transport reports none (a stub) or they do not arrive in time.
    func remoteAddress(timeout: Duration = .seconds(2)) async -> String? {
        await withTaskGroup(of: String?.self) { group in
            group.addTask {
                await withCheckedContinuation { continuation in
                    let ready = self.state.withLock { state -> String?? in
                        if let address = state.address { return .some(address) }
                        // Registering after the timeout: answer at once.
                        if state.timedOut { return .some(nil) }
                        state.waiters.append(continuation)
                        return nil
                    }
                    if let ready { continuation.resume(returning: ready) }
                }
            }
            group.addTask {
                try? await Task.sleep(for: timeout)
                return nil
            }
            let first = await group.next() ?? nil
            group.cancelAll()
            self.flush()
            return first
        }
    }

    /// Resumes anyone still waiting (after a timeout).
    private func flush() {
        let waiters = state.withLock { state -> [CheckedContinuation<String?, Never>] in
            state.timedOut = true
            defer { state.waiters.removeAll() }
            return state.waiters
        }
        for waiter in waiters { waiter.resume(returning: nil) }
    }
}
