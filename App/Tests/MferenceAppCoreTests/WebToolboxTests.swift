import Foundation
import MferenceAppCore
import Synchronization
import Testing

/// A transport stub: records every request and answers from a table. No
/// request reaches the network.
final class StubURLProtocol: URLProtocol, @unchecked Sendable {
    struct Reply: Sendable {
        var status: Int = 200
        var contentType = "text/html; charset=utf-8"
        var body = Data()
        var redirect: URL?
    }

    private static let state = Mutex<(requests: [URL], replies: [String: Reply])>(([], [:]))

    static func reset(_ replies: [String: Reply]) {
        state.withLock { $0 = ([], replies) }
    }

    static var requests: [URL] { state.withLock { $0.requests } }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let url = request.url else { return }
        let reply = Self.state.withLock { state -> Reply? in
            state.requests.append(url)
            var key = url.absoluteString
            if let query = url.query() { key = key.replacingOccurrences(of: "?" + query, with: "") }
            return state.replies[key] ?? state.replies[url.absoluteString]
        }
        let reply_ = reply ?? Reply(status: 404)
        if let target = reply_.redirect {
            let response = HTTPURLResponse(url: url, statusCode: 302, httpVersion: "HTTP/1.1",
                                           headerFields: ["Location": target.absoluteString])!
            client?.urlProtocol(self, wasRedirectedTo: URLRequest(url: target), redirectResponse: response)
            client?.urlProtocolDidFinishLoading(self)
            return
        }
        let response = HTTPURLResponse(url: url, statusCode: reply_.status, httpVersion: "HTTP/1.1",
                                       headerFields: ["Content-Type": reply_.contentType])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: reply_.body)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

/// Serialized: the stub transport's table is shared.
@Suite(.serialized) struct WebToolboxTests {
    static let base = URL(string: "http://127.0.0.1:43123")!

    static func searchJSON(_ results: [(String, String, String)]) -> Data {
        let items = results.map { ["url": $0.0, "title": $0.1, "content": $0.2] }
        return try! JSONSerialization.data(withJSONObject: ["query": "q", "results": items])
    }

    static func toolbox() throws -> WebToolbox {
        let spec = try WebAnswersSpec.load()
        let http = WebHTTP(allowlist: WebAllowlist(), protocolClasses: [StubURLProtocol.self])
        return WebToolbox(spec: spec, http: http, searchBase: { base })
    }

    static func call(_ name: String, _ arguments: [String: String]) -> ToolCall {
        let json = String(data: try! JSONSerialization.data(withJSONObject: arguments), encoding: .utf8)!
        return ToolCall(id: UUID().uuidString, name: name,
                        arguments: .object(arguments.mapValues(ToolJSON.string)), argumentsJSON: json)
    }

    /// G-A8: the only hosts contacted are loopback SearXNG and pages it
    /// returned; anything else is refused before a request is made.
    @Test func networkAllowlist() async throws {
        let page = "<html><head><title>Oakline Duo sale</title></head><body><nav>menu</nav><main><h1>Sale</h1><p>The Oakline Duo is $499 through Sunday.</p></main></body></html>"
        StubURLProtocol.reset([
            "\(Self.base.absoluteString)/search": .init(contentType: "application/json", body: Self.searchJSON([
                ("https://oakline.example/sale", "Oakline Duo sale", "Now $499"),
                ("https://stride-desks.example/e2", "Stride E2", "15 yr frame"),
            ])),
            "https://oakline.example/sale": .init(body: Data(page.utf8)),
            "https://stride-desks.example/e2": .init(redirect: URL(string: "https://tracker.example/e2")!),
        ])
        let toolbox = try Self.toolbox()

        let search = await toolbox.run(Self.call("web_search", ["query": "oakline duo sale"]))
        #expect(search.content.contains("[1] Oakline Duo sale"))
        #expect(search.content.contains("[2] Stride E2"))
        #expect(search.citations.map(\.number) == [1, 2])
        #expect(search.steps == [.search("oakline duo sale", results: 2)])

        let read = await toolbox.run(Self.call("open_page", ["url": "https://oakline.example/sale"]))
        #expect(read.content.hasPrefix("[1] Oakline Duo sale"))
        #expect(read.content.contains("The Oakline Duo is $499 through Sunday."))
        #expect(!read.content.contains("menu"))
        #expect(read.citations.map(\.number) == [1])

        let outside = await toolbox.run(Self.call("open_page", ["url": "https://evil.example/collect"]))
        #expect(outside.steps.first?.kind == .failed)

        let redirected = await toolbox.run(Self.call("open_page", ["url": "https://stride-desks.example/e2"]))
        #expect(redirected.steps.first?.kind == .failed)

        let hosts = Set(StubURLProtocol.requests.compactMap { $0.host() })
        #expect(hosts.isSubset(of: ["127.0.0.1", "oakline.example", "stride-desks.example"]), "contacted \(hosts)")
        #expect(!hosts.contains("evil.example"))
        #expect(!hosts.contains("tracker.example"))
        let searchRequest = try #require(StubURLProtocol.requests.first)
        #expect(searchRequest.query()?.contains("format=json") == true)
    }

    @Test func allowlistRules() {
        let allowlist = WebAllowlist()
        allowlist.setSearchBase(Self.base)
        #expect(allowlist.permits(URL(string: "http://127.0.0.1:43123/search?q=x")!))
        #expect(!allowlist.permits(URL(string: "http://127.0.0.1:9999/")!))
        #expect(!allowlist.permits(URL(string: "https://a.example/page")!))
        allowlist.allowPage(URL(string: "https://A.example/page#top")!)
        #expect(allowlist.permits(URL(string: "https://a.example/page")!))
        #expect(!allowlist.permits(URL(string: "https://a.example/other")!))
        #expect(allowlist.permitsRedirect(from: URL(string: "https://a.example/page")!,
                                          to: URL(string: "https://www.a.example/page/")!))
        #expect(!allowlist.permitsRedirect(from: URL(string: "https://a.example/page")!,
                                           to: URL(string: "https://b.example/page")!))
        allowlist.allowPage(URL(string: "file:///etc/passwd")!)
        #expect(!allowlist.permits(URL(string: "file:///etc/passwd")!))
    }

    @Test func pagesPerAnswerAreBounded() async throws {
        var replies: [String: StubURLProtocol.Reply] = [:]
        var results: [(String, String, String)] = []
        for index in 1...6 {
            let url = "https://site\(index).example/"
            results.append((url, "Site \(index)", ""))
            replies[url] = .init(body: Data("<p>page \(index)</p>".utf8))
        }
        replies["\(Self.base.absoluteString)/search"] = .init(contentType: "application/json", body: Self.searchJSON(results))
        StubURLProtocol.reset(replies)
        let toolbox = try Self.toolbox()
        _ = await toolbox.run(Self.call("web_search", ["query": "sites"]))
        var failed = 0
        for index in 1...6 {
            let outcome = await toolbox.run(Self.call("open_page", ["url": "https://site\(index).example/"]))
            if outcome.steps.first?.kind == .failed { failed += 1 }
        }
        #expect(failed == 6 - WebToolbox.maxPages)
    }

    @Test func missingRuntimeIsReported() async throws {
        let toolbox = WebToolbox(spec: try WebAnswersSpec.load(),
                                 http: WebHTTP(allowlist: WebAllowlist(), protocolClasses: [StubURLProtocol.self]),
                                 searchBase: nil)
        StubURLProtocol.reset([:])
        let outcome = await toolbox.run(Self.call("web_search", ["query": "x"]))
        #expect(outcome.content.contains("not included in this build"))
        #expect(StubURLProtocol.requests.isEmpty)
    }
}

@Suite struct WebParsingTests {
    @Test func specLoadsVersionedResource() throws {
        let spec = try WebAnswersSpec.load()
        #expect(spec.version == WebAnswersSpec.currentVersion)
        #expect(spec.tools.map(\.name) == ["web_search", "open_page"])
        guard case .object(let schema) = spec.tools[0].parameters,
              case .array(let required)? = schema["required"] else {
            Issue.record("web_search has no required list"); return
        }
        #expect(required == [.string("query")])
        let prompt = spec.systemPrompt(now: Date(timeIntervalSince1970: 1_800_000_000))
        #expect(!prompt.contains("{{date}}"))
        #expect(prompt.contains("2027"))
    }

    @Test func searchResultsAreDedupedAndWebOnly() throws {
        let data = WebToolboxTests.searchJSON([
            ("https://a.example/x", "A", "first"),
            ("https://a.example/x#frag", "A again", ""),
            ("javascript:alert(1)", "bad", ""),
            ("https://b.example/", "", "no title"),
        ])
        let results = try SearXNGClient.parse(data, limit: 8)
        #expect(results.map(\.url.absoluteString) == ["https://a.example/x", "https://b.example/"])
        #expect(results[1].title == "b.example")
        #expect(results[1].content == "no title")
    }

    @Test func htmlExtractionKeepsTheReadableText() {
        let html = """
        <html><head><title>Desk &amp; chair &#8211; review</title><style>p{color:red}</style>
        <script>var x = "<p>not text</p>";</script></head>
        <body><header>Site header</header><nav><a href="/">Home</a></nav>
        <article><h2>Warranty</h2><p>Frame: 15&nbsp;years.<br>Motor: 5 years.</p>
        <ul><li>One</li><li>Two</li></ul><table><tr><td>Price</td><td>$589</td></tr></table></article>
        <footer>Copyright</footer><!-- comment --></body></html>
        """
        let page = HTMLText.extract(html)
        #expect(page.title == "Desk & chair – review")
        #expect(page.text.contains("Warranty"))
        #expect(page.text.contains("Frame: 15 years."))
        #expect(page.text.contains("Motor: 5 years."))
        #expect(page.text.contains("• One"))
        #expect(page.text.contains("Price | $589"))
        for hidden in ["not text", "color:red", "Site header", "Home", "Copyright", "comment"] {
            #expect(!page.text.contains(hidden), "\(hidden) leaked into the text")
        }
    }

    @Test func toolJSONFromJSONObject() {
        let object: Any = ["a": 1, "b": 1.5, "c": true, "d": ["x", NSNull()]] as [String: Any]
        #expect(ToolJSON(jsonObject: object) == .object([
            "a": .integer(1), "b": .number(1.5), "c": .bool(true), "d": .array([.string("x"), .null]),
        ]))
    }
}

/// The live bundled runtime, only when `MFERENCE_SEARXNG_DIR` points at one
/// (prepare.sh output). It talks to 127.0.0.1 only: health check, no search.
@Suite struct SearXNGProcessTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["MFERENCE_SEARXNG_DIR"] != nil))
    func startsOnLoopbackAndStops() async throws {
        let runtime = try #require(SearXNGRuntime.locate())
        let pidFile = FileManager.default.temporaryDirectory.appending(path: "mference-test-\(UUID()).pid")
        let process = SearXNGProcess(runtime: runtime, pidFile: pidFile)
        let base = try await process.ensureRunning()
        #expect(base.host() == "127.0.0.1")
        #expect(FileManager.default.fileExists(atPath: pidFile.path))
        let again = try await process.ensureRunning()
        #expect(again == base)
        await process.stop()
        #expect(await process.state == .stopped)
        #expect(!FileManager.default.fileExists(atPath: pidFile.path))
    }
}
