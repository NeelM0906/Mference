import Foundation
import Jinja

/// A tool call's arguments as Inkling 2's reference renderers see them.
///
/// The vendor template renders `args | dictsort` and each value with
/// `tojson(separators=(",", ":"))`. Transformers (Python's `json`) and
/// llama.cpp (ordered JSON) parse the arguments from the call's JSON text, so
/// a nested object keeps the order its text gave, and a number keeps its kind:
/// `1.0` stays a float and prints `1.0`, `1e3` prints `1000.0`. `JSONValue`
/// keeps neither: its objects are dictionaries, which reach the template with
/// sorted keys, and its decoder reads `1.0` as the integer 1. So the history of
/// a call whose arguments nest objects in another order, or hold such a float,
/// re-rendered differently from what the model wrote. This parses the text
/// itself, into the template's own value type.
///
/// Only the Inkling 2 render, the Inkling 2 tool-call parser and the
/// server's Inkling 2 history validation use it.
public enum Inkling2ToolArguments {
    /// `text` as the compact text `compactText` writes, when it is one JSON
    /// object a template value carries exactly; nil otherwise. The server
    /// keeps a history call's arguments in this form, so two texts that render
    /// the same compare equal (whitespace aside) and a text the render would
    /// not use is not kept, as the tool-call parser does for a model's call.
    public static func normalizedText(_ text: String) -> String? {
        templateValue(text).flatMap { try? compactText($0) }
    }

    /// The template value of `text` when it is one JSON object; nil when it is
    /// not, or holds what no `Jinja.Value` carries exactly: an integer outside
    /// `Int`, a number that is not finite, an unpaired surrogate escape, or
    /// nesting deeper than `JSONValue.maximumDepth`. Callers then fall back to
    /// the `JSONValue` path.
    static func templateValue(_ text: String) -> Jinja.Value? {
        var parser = Parser(Array(text.utf8))
        guard let value = parser.document(), case .object = value else { return nil }
        return value
    }

    /// The `args` object of a tool-call envelope `{"name":…,"args":{…}}`, in
    /// its text's order; nil as for `templateValue`.
    static func envelopeArguments(_ text: String) -> Jinja.Value? {
        var parser = Parser(Array(text.utf8))
        guard let value = parser.document(), case .object(let members) = value,
              let arguments = members[.string("args")], case .object = arguments else { return nil }
        return arguments
    }

    /// `value` as Python's `json.dumps(value, ensure_ascii=False,
    /// separators=(",", ":"))` writes it, with the serializer the template's
    /// `tojson` uses: the text returned to clients as a call's `arguments`, so
    /// that a history which sends it back renders as the model wrote it.
    static func compactText(_ value: Jinja.Value) throws -> String {
        try Jinja.JSON.Serializer.standard.dumps(value, options: .init(ensureASCII: false,
                                                                     separators: (item: ",", key: ":")))
    }

    /// A strict RFC 8259 parser over UTF-8 bytes. Objects keep their members'
    /// order; a repeated key keeps its first place and takes its last value,
    /// as a Python `dict` does. A number with a fraction or an exponent is a
    /// double; one without is an `Int`.
    private struct Parser {
        let bytes: [UInt8]
        var index = 0

        init(_ bytes: [UInt8]) { self.bytes = bytes }

        mutating func document() -> Jinja.Value? {
            skipWhitespace()
            guard let value = value(depth: 0) else { return nil }
            skipWhitespace()
            return index == bytes.count ? value : nil
        }

        private var peek: UInt8? { index < bytes.count ? bytes[index] : nil }

        private mutating func consume(_ byte: UInt8) -> Bool {
            guard peek == byte else { return false }
            index += 1
            return true
        }

        private mutating func skipWhitespace() {
            while let byte = peek, byte == 0x20 || byte == 0x09 || byte == 0x0A || byte == 0x0D { index += 1 }
        }

        private mutating func value(depth: Int) -> Jinja.Value? {
            switch peek {
            case UInt8(ascii: "{"): return object(depth: depth + 1)
            case UInt8(ascii: "["): return array(depth: depth + 1)
            case UInt8(ascii: "\""): return string().map(Jinja.Value.string)
            case UInt8(ascii: "t"): return literal("true", .boolean(true))
            case UInt8(ascii: "f"): return literal("false", .boolean(false))
            case UInt8(ascii: "n"): return literal("null", .null)
            case .some: return number()
            case nil: return nil
            }
        }

        private mutating func literal(_ word: String, _ value: Jinja.Value) -> Jinja.Value? {
            let utf8 = Array(word.utf8)
            guard index + utf8.count <= bytes.count, Array(bytes[index..<index + utf8.count]) == utf8 else {
                return nil
            }
            index += utf8.count
            return value
        }

        private mutating func object(depth: Int) -> Jinja.Value? {
            guard depth <= JSONValue.maximumDepth else { return nil }
            index += 1
            var members = OrderedDictionary<Jinja.ObjectKey, Jinja.Value>()
            skipWhitespace()
            if consume(UInt8(ascii: "}")) { return .object(members) }
            while true {
                skipWhitespace()
                guard peek == UInt8(ascii: "\""), let key = string() else { return nil }
                skipWhitespace()
                guard consume(UInt8(ascii: ":")) else { return nil }
                skipWhitespace()
                guard let member = value(depth: depth) else { return nil }
                members[.string(key)] = member
                skipWhitespace()
                if consume(UInt8(ascii: ",")) { continue }
                return consume(UInt8(ascii: "}")) ? .object(members) : nil
            }
        }

        private mutating func array(depth: Int) -> Jinja.Value? {
            guard depth <= JSONValue.maximumDepth else { return nil }
            index += 1
            var items: [Jinja.Value] = []
            skipWhitespace()
            if consume(UInt8(ascii: "]")) { return .array(items) }
            while true {
                skipWhitespace()
                guard let item = value(depth: depth) else { return nil }
                items.append(item)
                skipWhitespace()
                if consume(UInt8(ascii: ",")) { continue }
                return consume(UInt8(ascii: "]")) ? .array(items) : nil
            }
        }

        /// A string literal at `index` (its opening quote). Bytes other than
        /// escapes are copied: the input came from a `String`, so they are
        /// valid UTF-8, and splitting at ASCII quotes and backslashes keeps it so.
        private mutating func string() -> String? {
            index += 1
            var out: [UInt8] = []
            while let byte = peek {
                index += 1
                switch byte {
                case UInt8(ascii: "\""):
                    return String(decoding: out, as: UTF8.self)
                case UInt8(ascii: "\\"):
                    guard let escape = peek else { return nil }
                    index += 1
                    switch escape {
                    case UInt8(ascii: "\""), UInt8(ascii: "\\"), UInt8(ascii: "/"): out.append(escape)
                    case UInt8(ascii: "b"): out.append(0x08)
                    case UInt8(ascii: "f"): out.append(0x0C)
                    case UInt8(ascii: "n"): out.append(0x0A)
                    case UInt8(ascii: "r"): out.append(0x0D)
                    case UInt8(ascii: "t"): out.append(0x09)
                    case UInt8(ascii: "u"):
                        guard let scalar = unicodeEscape() else { return nil }
                        out.append(contentsOf: Array(String(Character(scalar)).utf8))
                    default:
                        return nil
                    }
                case 0x00..<0x20:
                    return nil
                default:
                    out.append(byte)
                }
            }
            return nil
        }

        /// The scalar of a `\uXXXX` escape whose `\u` was just read, joining a
        /// surrogate pair; nil for an unpaired surrogate.
        private mutating func unicodeEscape() -> Unicode.Scalar? {
            guard let high = hex4() else { return nil }
            switch high {
            case 0xD800...0xDBFF:
                guard consume(UInt8(ascii: "\\")), consume(UInt8(ascii: "u")),
                      let low = hex4(), (0xDC00...0xDFFF).contains(low) else { return nil }
                return Unicode.Scalar(0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00))
            case 0xDC00...0xDFFF:
                return nil
            default:
                return Unicode.Scalar(high)
            }
        }

        private mutating func hex4() -> UInt32? {
            guard index + 4 <= bytes.count,
                  let value = UInt32(String(decoding: bytes[index..<index + 4], as: UTF8.self), radix: 16) else {
                return nil
            }
            // `UInt32(_:radix:)` accepts a leading sign; JSON does not.
            guard bytes[index] != UInt8(ascii: "+"), bytes[index] != UInt8(ascii: "-") else { return nil }
            index += 4
            return value
        }

        private func isDigit(_ byte: UInt8?) -> Bool {
            guard let byte else { return false }
            return byte >= UInt8(ascii: "0") && byte <= UInt8(ascii: "9")
        }

        private mutating func digits() { while isDigit(peek) { index += 1 } }

        private mutating func number() -> Jinja.Value? {
            let start = index
            _ = consume(UInt8(ascii: "-"))
            guard isDigit(peek) else { return nil }
            if !consume(UInt8(ascii: "0")) { digits() }
            var integral = true
            if consume(UInt8(ascii: ".")) {
                integral = false
                guard isDigit(peek) else { return nil }
                digits()
            }
            if peek == UInt8(ascii: "e") || peek == UInt8(ascii: "E") {
                integral = false
                index += 1
                if !consume(UInt8(ascii: "+")) { _ = consume(UInt8(ascii: "-")) }
                guard isDigit(peek) else { return nil }
                digits()
            }
            let text = String(decoding: bytes[start..<index], as: UTF8.self)
            if integral { return Int(text).map(Jinja.Value.int) }
            guard let double = Double(text), double.isFinite else { return nil }
            return .double(double)
        }
    }
}
