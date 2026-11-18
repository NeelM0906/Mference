import Foundation

/// One search result from SearXNG's JSON output.
public struct WebSearchResult: Sendable, Hashable, Codable {
    public var url: URL
    public var title: String
    public var content: String?

    public init(url: URL, title: String, content: String? = nil) {
        self.url = url
        self.title = title
        self.content = content
    }
}

/// Queries the local SearXNG (`/search?format=json`).
public struct SearXNGClient: Sendable {
    public let http: WebHTTP

    public init(http: WebHTTP) {
        self.http = http
    }

    public func search(_ query: String, base: URL, limit: Int = 8) async throws -> [WebSearchResult] {
        var components = URLComponents(url: base.appending(path: "search"), resolvingAgainstBaseURL: false)!
        components.queryItems = [
            URLQueryItem(name: "q", value: query),
            URLQueryItem(name: "format", value: "json"),
            URLQueryItem(name: "safesearch", value: "0"),
            URLQueryItem(name: "language", value: "all"),
        ]
        guard let url = components.url else { throw WebError.searchUnavailable("bad query") }
        let (data, _) = try await http.get(url, accept: "application/json")
        return try Self.parse(data, limit: limit)
    }

    public static func parse(_ data: Data, limit: Int) throws -> [WebSearchResult] {
        struct Response: Decodable {
            struct Item: Decodable {
                var url: String?
                var title: String?
                var content: String?
            }
            var results: [Item]
        }
        let response: Response
        do { response = try JSONDecoder().decode(Response.self, from: data) } catch {
            throw WebError.searchUnavailable("SearXNG sent an unreadable answer.")
        }
        var seen: Set<String> = []
        var results: [WebSearchResult] = []
        for item in response.results {
            guard let string = item.url, let url = URL(string: string), WebAllowlist.isWebURL(url),
                  let key = WebAllowlist.key(url), seen.insert(key).inserted else { continue }
            let title = (item.title ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            let snippet = item.content?.trimmingCharacters(in: .whitespacesAndNewlines)
            results.append(WebSearchResult(url: url, title: title.isEmpty ? (url.host() ?? string) : title,
                                           content: snippet?.isEmpty == true ? nil : snippet))
            if results.count == limit { break }
        }
        return results
    }
}

/// A page reduced to readable text.
public struct ReadablePage: Sendable, Hashable {
    public var url: URL
    public var title: String?
    public var text: String
    /// The text was cut to the size limit.
    public var isTruncated: Bool
}

/// Fetches a page through `WebHTTP` (allowlist, bounded size) and extracts
/// its readable text. Extraction is done here rather than with WebKit or
/// `NSAttributedString` HTML import, because those load the page's images,
/// styles and scripts from other hosts, which the network allowlist forbids.
public struct PageReader: Sendable {
    public let http: WebHTTP
    /// Characters of text handed to the model per page.
    public let maxCharacters: Int

    public init(http: WebHTTP, maxCharacters: Int = 6_000) {
        self.http = http
        self.maxCharacters = maxCharacters
    }

    public func read(_ url: URL) async throws -> ReadablePage {
        let (data, response) = try await http.get(url, accept: "text/html,application/xhtml+xml,text/plain;q=0.8")
        let type = (response.value(forHTTPHeaderField: "Content-Type") ?? "text/html").lowercased()
        guard type.contains("html") || type.contains("text/plain") || type.contains("xml") else {
            throw WebError.unsupportedContent(type.split(separator: ";").first.map(String.init) ?? type)
        }
        let html = Self.decode(data, contentType: type)
        if type.contains("text/plain") {
            return Self.bound(ReadablePage(url: url, title: nil, text: Self.collapse(html), isTruncated: false),
                              to: maxCharacters)
        }
        let extracted = HTMLText.extract(html)
        return Self.bound(ReadablePage(url: url, title: extracted.title, text: extracted.text, isTruncated: false),
                          to: maxCharacters)
    }

    static func decode(_ data: Data, contentType: String) -> String {
        if contentType.contains("charset=iso-8859-1") || contentType.contains("charset=latin1") {
            return String(data: data, encoding: .isoLatin1) ?? String(decoding: data, as: UTF8.self)
        }
        return String(decoding: data, as: UTF8.self)
    }

    static func bound(_ page: ReadablePage, to limit: Int) -> ReadablePage {
        guard page.text.count > limit else { return page }
        var page = page
        page.text = String(page.text.prefix(limit))
        page.isTruncated = true
        return page
    }

    static func collapse(_ text: String) -> String {
        text.split(whereSeparator: \.isNewline)
            .map { $0.split(whereSeparator: \.isWhitespace).joined(separator: " ") }
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
    }
}

/// A small, forgiving HTML-to-text pass: drops scripts, styles and page
/// chrome (nav, header, footer, aside, forms), keeps block structure as
/// line breaks, decodes entities.
public enum HTMLText {
    public static func extract(_ html: String) -> (title: String?, text: String) {
        var source = html
        let title = firstMatch(#"<title[^>]*>(.*?)</title>"#, in: source).map { decodeEntities(stripTags($0)) }
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        // Prefer <main> or <article> when the page has one.
        if let main = firstMatch(#"<main[^>]*>(.*?)</main>"#, in: source)
            ?? firstMatch(#"<article[^>]*>(.*?)</article>"#, in: source),
           main.count > 500 {
            source = main
        }
        for tag in ["script", "style", "noscript", "svg", "nav", "header", "footer", "aside", "form", "template", "iframe"] {
            source = source.replacingOccurrences(of: "(?s)<\(tag)\\b[^>]*>.*?</\(tag)>", with: " ",
                                                 options: [.regularExpression, .caseInsensitive])
        }
        source = source.replacingOccurrences(of: "(?s)<!--.*?-->", with: " ", options: .regularExpression)
        source = source.replacingOccurrences(
            of: #"<(br|/p|/div|/li|/h[1-6]|/tr|/section|/article|/blockquote|/pre|/table|hr)\b[^>]*>"#,
            with: "\n", options: [.regularExpression, .caseInsensitive])
        source = source.replacingOccurrences(of: #"<li\b[^>]*>"#, with: "\n• ",
                                             options: [.regularExpression, .caseInsensitive])
        source = source.replacingOccurrences(of: #"</t[dh]>"#, with: " | ",
                                             options: [.regularExpression, .caseInsensitive])
        let text = PageReader.collapse(decodeEntities(stripTags(source)))
        return (title?.isEmpty == true ? nil : title, text)
    }

    static func firstMatch(_ pattern: String, in text: String) -> String? {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive, .dotMatchesLineSeparators]),
              let match = regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)),
              match.numberOfRanges > 1, let range = Range(match.range(at: 1), in: text) else { return nil }
        return String(text[range])
    }

    static func stripTags(_ text: String) -> String {
        text.replacingOccurrences(of: "<[^>]+>", with: " ", options: .regularExpression)
    }

    static let entities: [String: String] = [
        "amp": "&", "lt": "<", "gt": ">", "quot": "\"", "apos": "'", "nbsp": " ", "ndash": "–", "mdash": "—",
        "hellip": "…", "rsquo": "’", "lsquo": "‘", "rdquo": "”", "ldquo": "“", "copy": "©", "reg": "®",
        "trade": "™", "euro": "€", "pound": "£", "middot": "·", "times": "×",
    ]

    static func decodeEntities(_ text: String) -> String {
        guard text.contains("&") else { return text }
        var result = ""
        var index = text.startIndex
        while index < text.endIndex {
            if text[index] == "&", let semicolon = text[index...].prefix(10).firstIndex(of: ";") {
                let name = String(text[text.index(after: index)..<semicolon])
                var replacement: String?
                if name.hasPrefix("#x") || name.hasPrefix("#X") {
                    replacement = UInt32(name.dropFirst(2), radix: 16).flatMap(Unicode.Scalar.init).map { String(Character($0)) }
                } else if name.hasPrefix("#") {
                    replacement = UInt32(name.dropFirst()).flatMap(Unicode.Scalar.init).map { String(Character($0)) }
                } else {
                    replacement = entities[name.lowercased()]
                }
                if let replacement {
                    result += replacement
                    index = text.index(after: semicolon)
                    continue
                }
            }
            result.append(text[index])
            index = text.index(after: index)
        }
        return result
    }
}
