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
            throw WebError.unsupportedContent
        }
        let html = Self.decode(data, contentType: type)
        if type.contains("text/plain") {
            return Self.bound(ReadablePage(url: url, title: nil, text: Self.collapse(html), isTruncated: false),
                              to: maxCharacters)
        }
        let extracted = try HTMLText.extract(html, checkCancellation: true)
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

/// A small, forgiving HTML-to-text pass in one linear scan over the bytes:
/// drops scripts, styles and page chrome (nav, header, footer, aside, forms),
/// keeps block structure as line breaks, decodes entities, and prefers the
/// page's <main> or <article> when it has a substantial one. Input is capped
/// (`maxInputBytes`) and the scan checks for cancellation as it goes.
public enum HTMLText {
    /// HTML bytes examined per page (review finding 6).
    public static let maxInputBytes = 512 * 1024

    static let skipped: Set<String> = ["script", "style", "noscript", "svg", "nav", "header", "footer", "aside",
                                       "form", "template", "iframe", "select", "button"]
    static let blocks: Set<String> = ["br", "p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section",
                                      "article", "blockquote", "pre", "table", "hr", "ul", "ol", "dl", "dd", "dt",
                                      "main", "figure", "figcaption"]

    public static func extract(_ html: String) -> (title: String?, text: String) {
        (try? extract(html, checkCancellation: false)) ?? (nil, "")
    }

    public static func extract(_ html: String, checkCancellation: Bool) throws -> (title: String?, text: String) {
        let bytes = Array(html.utf8.prefix(maxInputBytes))
        let count = bytes.count
        var output: [UInt8] = []
        output.reserveCapacity(min(count, 256 * 1024))
        var title: [UInt8] = []
        var inTitle = false
        var mainStart: Int?
        var mainEnd: Int?
        var mainDepth = 0
        var i = 0
        var nextCheck = 0

        func lower(_ byte: UInt8) -> UInt8 { (byte >= 65 && byte <= 90) ? byte + 32 : byte }
        func isNameByte(_ byte: UInt8) -> Bool {
            (byte >= 97 && byte <= 122) || (byte >= 65 && byte <= 90) || (byte >= 48 && byte <= 57) || byte == 45
        }
        func append(_ text: String) { output.append(contentsOf: text.utf8) }

        // Every branch below advances `i` past what it examined, so the scan
        // is linear; cancellation is checked every 64 KB of progress.
        while i < count {
            if checkCancellation, i >= nextCheck {
                try Task.checkCancellation()
                nextCheck = i + 0x10000
            }
            let byte = bytes[i]
            guard byte == 60 /* < */ else {
                if inTitle { title.append(byte) } else { output.append(byte) }
                i += 1
                continue
            }
            // Comment: skip to "-->".
            if i + 3 < count, bytes[i + 1] == 33, bytes[i + 2] == 45, bytes[i + 3] == 45 {
                var j = i + 4
                while j + 2 < count, !(bytes[j] == 45 && bytes[j + 1] == 45 && bytes[j + 2] == 62) { j += 1 }
                i = min(j + 3, count)
                continue
            }
            // CDATA: skip to "]]>".
            if i + 8 < count, Array(bytes[(i + 1)...(i + 8)]) == Array("![CDATA[".utf8) {
                var j = i + 9
                while j + 2 < count, !(bytes[j] == 93 && bytes[j + 1] == 93 && bytes[j + 2] == 62) { j += 1 }
                i = min(j + 3, count)
                continue
            }
            // <!DOCTYPE ...>, other declarations and <?...?>: skip to ">".
            if i + 1 < count, bytes[i + 1] == 33 || bytes[i + 1] == 63 {
                var j = i + 2
                while j < count, bytes[j] != 62 { j += 1 }
                i = min(j + 1, count)
                continue
            }
            var j = i + 1
            let closing = j < count && bytes[j] == 47 /* / */
            if closing { j += 1 }
            var name: [UInt8] = []
            while j < count, isNameByte(bytes[j]), name.count < 16 {
                name.append(lower(bytes[j]))
                j += 1
            }
            guard !name.isEmpty else {
                // A bare "<" in text ("a < b"): keep it and move on, without
                // looking for a ">" (that made "<<<<..." quadratic).
                if inTitle { title.append(byte) } else { output.append(byte) }
                i += 1
                continue
            }
            // Find the end of the tag, skipping quoted attribute values.
            var quote: UInt8 = 0
            while j < count {
                let c = bytes[j]
                if quote != 0 { if c == quote { quote = 0 } }
                else if c == 34 || c == 39 { quote = c }
                else if c == 62 /* > */ { break }
                j += 1
            }
            let tagEnd = min(j + 1, count)
            let tag = String(decoding: name, as: UTF8.self)
            if !closing, skipped.contains(tag) {
                // Skip to the matching close tag (no nesting needed for these).
                let close = Array("</\(tag)".utf8)
                var k = tagEnd
                var found = count
                while k + close.count <= count {
                    if bytes[k] == 60 {
                        var matches = true
                        for (offset, expected) in close.enumerated() where lower(bytes[k + offset]) != expected {
                            matches = false
                            break
                        }
                        if matches { found = k; break }
                    }
                    k += 1
                }
                var after = found
                while after < count, bytes[after] != 62 { after += 1 }
                i = min(after + 1, count)
                output.append(32)
                continue
            }
            switch tag {
            case "title":
                inTitle = !closing
            case "main", "article":
                if !closing {
                    if mainStart == nil, mainDepth == 0 { mainStart = output.count }
                    mainDepth += 1
                } else if mainDepth > 0 {
                    mainDepth -= 1
                    if mainDepth == 0, mainEnd == nil { mainEnd = output.count }
                }
                append("\n")
            case "li":
                append(closing ? "\n" : "\n• ")
            case "td", "th":
                if closing { append(" | ") }
            default:
                if blocks.contains(tag) { append("\n") } else { output.append(32) }
            }
            i = tagEnd
        }

        var body = output
        if let start = mainStart {
            let end = mainEnd ?? output.count
            if end - start > 500 { body = Array(output[start..<end]) }
        }
        let text = PageReader.collapse(decodeEntities(String(decoding: body, as: UTF8.self)))
        let titleText = decodeEntities(String(decoding: title, as: UTF8.self))
            .split(whereSeparator: \.isWhitespace).joined(separator: " ")
        return (titleText.isEmpty ? nil : titleText, text)
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
