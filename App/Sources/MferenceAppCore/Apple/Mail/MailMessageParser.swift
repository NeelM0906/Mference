import Foundation

/// A mail body ready to show: plain text only, so nothing remote ever loads.
public struct MailBody: Sendable, Hashable {
    public var text: String
    /// Remote images that looked like tracking pixels and were removed.
    public var trackersRemoved: Int
    /// Other remote images not loaded.
    public var remoteImagesBlocked: Int
    public var headers: [String: String]

    public init(text: String, trackersRemoved: Int = 0, remoteImagesBlocked: Int = 0, headers: [String: String] = [:]) {
        self.text = text
        self.trackersRemoved = trackersRemoved
        self.remoteImagesBlocked = remoteImagesBlocked
        self.headers = headers
    }
}

/// Reads Apple Mail's `.emlx` files and the MIME message inside: headers,
/// quoted-printable and base64, charsets, multipart, HTML to text.
public enum MailMessageParser {
    /// Largest message read, so a huge attachment never lands in memory twice.
    public static let maximumBytes = 8 * 1024 * 1024

    /// `.emlx`: a byte count line, the RFC 822 message, then a plist.
    public static func parseEMLX(_ data: Data) -> MailBody? {
        guard let newline = data.firstIndex(of: 0x0A),
              let count = Int(String(decoding: data[data.startIndex..<newline], as: UTF8.self)
                .trimmingCharacters(in: .whitespaces)) else { return parseMessage(data) }
        let start = data.index(after: newline)
        let end = min(data.endIndex, start + max(0, count))
        return parseMessage(data[start..<end])
    }

    public static func parseMessage(_ data: Data) -> MailBody? {
        let part = MIMEPart(data: Data(data))
        guard !part.headers.isEmpty || !part.body.isEmpty else { return nil }
        var plain: String?
        var html: String?
        collect(part, plain: &plain, html: &html, depth: 0)
        if let plain, !plain.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            let images = html.map(MailHTML.remoteImages) ?? []
            return MailBody(text: normalize(plain),
                            trackersRemoved: images.filter(\.isTracker).count,
                            remoteImagesBlocked: images.filter { !$0.isTracker }.count,
                            headers: part.headers)
        }
        if let html {
            let images = MailHTML.remoteImages(in: html)
            return MailBody(text: normalize(MailHTML.text(from: html)),
                            trackersRemoved: images.filter(\.isTracker).count,
                            remoteImagesBlocked: images.filter { !$0.isTracker }.count,
                            headers: part.headers)
        }
        return MailBody(text: "", headers: part.headers)
    }

    private static func collect(_ part: MIMEPart, plain: inout String?, html: inout String?, depth: Int) {
        guard depth < 8 else { return }
        let type = part.contentType
        if type.hasPrefix("multipart/") {
            for child in part.children() { collect(child, plain: &plain, html: &html, depth: depth + 1) }
            return
        }
        if part.isAttachment { return }
        if type == "text/plain" || type.isEmpty, plain == nil { plain = part.decodedText }
        else if type == "text/html", html == nil { html = part.decodedText }
    }

    static func normalize(_ text: String) -> String {
        let unified = text.replacingOccurrences(of: "\r\n", with: "\n").replacingOccurrences(of: "\r", with: "\n")
        var lines: [String] = []
        var blank = 0
        for line in unified.split(separator: "\n", omittingEmptySubsequences: false) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty {
                blank += 1
                if blank > 1 { continue }
            } else {
                blank = 0
            }
            lines.append(trimmed.isEmpty ? "" : String(line.reversed().drop(while: \.isWhitespace).reversed()))
        }
        return lines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Decodes RFC 2047 encoded words in a header ("=?UTF-8?B?...?=").
    public static func decodeHeader(_ value: String) -> String {
        let pattern = /=\?([^?]+)\?([bBqQ])\?([^?]*)\?=/
        var result = ""
        var rest = Substring(value)
        var lastWasEncoded = false
        while let match = rest.firstMatch(of: pattern) {
            let between = rest[rest.startIndex..<match.range.lowerBound]
            if !(lastWasEncoded && between.allSatisfy(\.isWhitespace)) { result += between }
            let charset = String(match.output.1), encoding = match.output.2.uppercased(), text = String(match.output.3)
            let bytes: Data? = encoding == "B"
                ? Data(base64Encoded: text.padding(toLength: (text.count + 3) / 4 * 4, withPad: "=", startingAt: 0))
                : QuotedPrintable.decode(text.replacingOccurrences(of: "_", with: " "), header: true)
            if let bytes, let decoded = Charset.decode(bytes, charset: charset) {
                result += decoded
                lastWasEncoded = true
            } else {
                result += match.output.0
                lastWasEncoded = false
            }
            rest = rest[match.range.upperBound...]
        }
        return result + rest
    }
}

/// One MIME entity.
struct MIMEPart {
    var headers: [String: String]
    var body: Data

    init(data: Data) {
        let separator = Self.headerEnd(in: data)
        let headerData = separator.map { data[data.startIndex..<$0.lowerBound] } ?? data[...]
        body = separator.map { Data(data[$0.upperBound...]) } ?? Data()
        headers = Self.parseHeaders(String(decoding: headerData, as: UTF8.self))
    }

    static func headerEnd(in data: Data) -> Range<Data.Index>? {
        let crlf = data.range(of: Data("\r\n\r\n".utf8))
        let lf = data.range(of: Data("\n\n".utf8))
        switch (crlf, lf) {
        case let (c?, l?): return c.lowerBound < l.lowerBound ? c : l
        case let (c?, nil): return c
        case let (nil, l?): return l
        default: return nil
        }
    }

    static func parseHeaders(_ text: String) -> [String: String] {
        var headers: [String: String] = [:]
        var current: (String, String)?
        for rawLine in text.replacingOccurrences(of: "\r\n", with: "\n").split(separator: "\n",
                                                                               omittingEmptySubsequences: false) {
            if let first = rawLine.first, first == " " || first == "\t", current != nil {
                current?.1 += " " + rawLine.trimmingCharacters(in: .whitespaces)
                continue
            }
            if let (name, value) = current, headers[name] == nil { headers[name] = value }
            guard let colon = rawLine.firstIndex(of: ":") else { current = nil; continue }
            current = (rawLine[..<colon].lowercased().trimmingCharacters(in: .whitespaces),
                       rawLine[rawLine.index(after: colon)...].trimmingCharacters(in: .whitespaces))
        }
        if let (name, value) = current, headers[name] == nil { headers[name] = value }
        return headers
    }

    /// "text/plain", lowercased, without parameters.
    var contentType: String {
        (headers["content-type"] ?? "").split(separator: ";").first.map {
            $0.trimmingCharacters(in: .whitespaces).lowercased()
        } ?? ""
    }

    func parameter(_ name: String, of header: String) -> String? {
        guard let value = headers[header] else { return nil }
        for piece in value.split(separator: ";").dropFirst() {
            let pair = piece.split(separator: "=", maxSplits: 1)
            guard pair.count == 2, pair[0].trimmingCharacters(in: .whitespaces).lowercased() == name else { continue }
            return pair[1].trimmingCharacters(in: .whitespaces).trimmingCharacters(in: CharacterSet(charactersIn: "\""))
        }
        return nil
    }

    var isAttachment: Bool {
        (headers["content-disposition"] ?? "").lowercased().hasPrefix("attachment")
    }

    var decodedBody: Data {
        switch (headers["content-transfer-encoding"] ?? "").lowercased().trimmingCharacters(in: .whitespaces) {
        case "base64":
            let compact = String(decoding: body, as: UTF8.self).filter { !$0.isWhitespace }
            return Data(base64Encoded: compact, options: .ignoreUnknownCharacters) ?? body
        case "quoted-printable":
            return QuotedPrintable.decode(String(decoding: body, as: UTF8.self), header: false) ?? body
        default:
            return body
        }
    }

    var decodedText: String {
        Charset.decode(decodedBody, charset: parameter("charset", of: "content-type") ?? "utf-8")
            ?? String(decoding: decodedBody, as: UTF8.self)
    }

    func children() -> [MIMEPart] {
        guard let boundary = parameter("boundary", of: "content-type"), !boundary.isEmpty else { return [] }
        let delimiter = Data("--\(boundary)".utf8)
        var parts: [MIMEPart] = []
        var searchStart = body.startIndex
        var partStart: Data.Index?
        while let range = body.range(of: delimiter, in: searchStart..<body.endIndex) {
            if let start = partStart {
                var end = range.lowerBound
                // Drop the line break that belongs to the delimiter.
                if end > start, body[body.index(before: end)] == 0x0A { end = body.index(before: end) }
                if end > start, body[body.index(before: end)] == 0x0D { end = body.index(before: end) }
                parts.append(MIMEPart(data: Data(body[start..<end])))
            }
            var next = range.upperBound
            if body[next...].starts(with: Data("--".utf8)) { break }
            while next < body.endIndex, body[next] == 0x0D || body[next] == 0x0A || body[next] == 0x20 {
                next = body.index(after: next)
            }
            partStart = next
            searchStart = next
            if parts.count > 64 { break }
        }
        return parts
    }
}

enum QuotedPrintable {
    static func decode(_ text: String, header: Bool) -> Data? {
        var output = Data()
        let bytes = Array(text.utf8)
        var index = 0
        while index < bytes.count {
            let byte = bytes[index]
            if byte == UInt8(ascii: "=") {
                if index + 1 < bytes.count, bytes[index + 1] == 0x0A { index += 2; continue }
                if index + 2 < bytes.count, bytes[index + 1] == 0x0D, bytes[index + 2] == 0x0A { index += 3; continue }
                if index + 2 < bytes.count, let value = UInt8(String(decoding: bytes[(index + 1)...(index + 2)],
                                                                    as: UTF8.self), radix: 16) {
                    output.append(value)
                    index += 3
                    continue
                }
            }
            output.append(byte)
            index += 1
        }
        return output
    }
}

enum Charset {
    static func decode(_ data: Data, charset: String) -> String? {
        let name = charset.lowercased().trimmingCharacters(in: CharacterSet(charactersIn: "\" "))
        let encoding: String.Encoding
        switch name {
        case "utf-8", "utf8", "us-ascii", "ascii": encoding = .utf8
        case "iso-8859-1", "latin1", "latin-1": encoding = .isoLatin1
        case "windows-1252", "cp1252": encoding = .windowsCP1252
        case "utf-16": encoding = .utf16
        default:
            let cfEncoding = CFStringConvertIANACharSetNameToEncoding(name as CFString)
            guard cfEncoding != kCFStringEncodingInvalidId else { return String(data: data, encoding: .utf8) }
            encoding = String.Encoding(rawValue: CFStringConvertEncodingToNSStringEncoding(cfEncoding))
        }
        return String(data: data, encoding: encoding) ?? String(data: data, encoding: .isoLatin1)
    }
}

/// HTML mail, reduced to text. Remote images never load; tracking pixels are
/// counted for the "trackers removed" badge.
public enum MailHTML {
    public struct RemoteImage: Sendable, Hashable {
        public var source: String
        public var isTracker: Bool
    }

    static let trackerHints = ["track", "pixel", "open", "beacon", "/o/", "/e/", "wf/open", "mailtrack", "/t.gif",
                               "spacer", "1x1"]

    public static func remoteImages(in html: String) -> [RemoteImage] {
        html.matches(of: /(?is)<img\b[^>]*>/).compactMap { match in
            let tag = String(match.output)
            guard let source = attribute("src", in: tag),
                  source.lowercased().hasPrefix("http") || source.hasPrefix("//") else { return nil }
            let width = attribute("width", in: tag).flatMap { Int($0.filter(\.isNumber)) }
            let height = attribute("height", in: tag).flatMap { Int($0.filter(\.isNumber)) }
            let style = (attribute("style", in: tag) ?? "").lowercased().replacingOccurrences(of: " ", with: "")
            let tiny = (width.map { $0 <= 2 } ?? false) || (height.map { $0 <= 2 } ?? false)
                || style.contains("display:none") || style.contains("width:1px") || style.contains("height:1px")
            let hinted = trackerHints.contains { source.lowercased().contains($0) }
            return RemoteImage(source: source, isTracker: tiny || (hinted && width == nil && height == nil))
        }
    }

    static func attribute(_ name: String, in tag: String) -> String? {
        let lower = tag.lowercased()
        guard let range = lower.range(of: "\(name)=") ?? lower.range(of: "\(name) =") else { return nil }
        var rest = tag[range.upperBound...].drop(while: \.isWhitespace)
        if let quote = rest.first, quote == "\"" || quote == "'" {
            rest = rest.dropFirst()
            return String(rest.prefix { $0 != quote })
        }
        return String(rest.prefix { !$0.isWhitespace && $0 != ">" })
    }

    /// Text of an HTML body: scripts, styles and comments dropped, block
    /// elements become line breaks, entities decoded.
    public static func text(from html: String) -> String {
        var text = html
        text = text.replacing(/(?is)<(script|style|head|title)\b.*?<\/\1\s*>/, with: "")
        text = text.replacing(/(?s)<!--.*?-->/, with: "")
        text = text.replacing(/(?i)<br\s*\/?>/, with: "\n")
        text = text.replacing(/(?i)<\/(p|div|tr|li|h[1-6]|table|blockquote)\s*>/, with: "\n")
        text = text.replacing(/(?i)<li\b[^>]*>/, with: "• ")
        text = text.replacing(/(?s)<[^>]+>/, with: "")
        return decodeEntities(text)
    }

    static func decodeEntities(_ text: String) -> String {
        let named = ["&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": "\"", "&#39;": "'",
                     "&apos;": "'", "&rsquo;": "’", "&lsquo;": "‘", "&ldquo;": "“", "&rdquo;": "”",
                     "&hellip;": "…", "&mdash;": "—", "&ndash;": "–", "&euro;": "€", "&zwnj;": ""]
        var result = text
        for (entity, value) in named where entity != "&amp;" { result = result.replacingOccurrences(of: entity, with: value) }
        result = result.replacing(/&#(x?)([0-9A-Fa-f]{1,6});/) { match in
            let radix = match.output.1.isEmpty ? 10 : 16
            guard let code = UInt32(match.output.2, radix: radix), let scalar = Unicode.Scalar(code) else { return "" }
            return String(Character(scalar))
        }
        return result.replacingOccurrences(of: "&amp;", with: "&")
    }
}
