import Foundation

public indirect enum JSONValue: Codable, Equatable, Sendable {
    case object([String: JSONValue])
    case array([JSONValue])
    case string(String)
    case integer(Int64)
    case unsignedInteger(UInt64)
    case decimal(Decimal)
    case number(Double)
    case bool(Bool)
    case null

    /// How deep a decoded tree may nest, counted from the root of the enclosing
    /// document. Decoding recurses once per level, and so does every consumer of
    /// the result — schema key validation, the Jinja
    /// bridge — so the ceiling has to leave stack for all of them rather than
    /// just for the decode. Request bodies are decoded on the sole event-loop
    /// thread, whose 512 KiB stack a debug build exhausts at roughly 220 levels;
    /// the JSON scanner does not help, because it accepts up to 512. A stack
    /// overflow traps in hardware and cannot be caught, so the process dies
    /// outright. Real tool schemas nest a handful of levels.
    public static let maximumDepth = 64

    public init(from decoder: Decoder) throws {
        guard decoder.codingPath.count <= Self.maximumDepth else {
            throw DecodingError.dataCorrupted(.init(
                codingPath: decoder.codingPath,
                debugDescription: "JSON nests deeper than \(Self.maximumDepth) levels"))
        }
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Int64.self) {
            self = .integer(value)
        } else if let value = try? container.decode(UInt64.self) {
            self = .unsignedInteger(value)
        } else if let value = try? container.decode(Decimal.self) {
            self = .decimal(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([JSONValue].self) {
            self = .array(value)
        } else {
            self = .object(try container.decode([String: JSONValue].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .object(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        case .string(let value): try container.encode(value)
        case .integer(let value): try container.encode(value)
        case .unsignedInteger(let value): try container.encode(value)
        case .decimal(let value): try container.encode(value)
        case .number(let value): try container.encode(value)
        case .bool(let value): try container.encode(value)
        case .null: try container.encodeNil()
        }
    }

    public func jinjaSendableValue() throws -> any Sendable {
        switch self {
        case .object(let value):
            return try value.mapValues { try $0.jinjaSendableValue() }
        case .array(let value):
            return try value.map { try $0.jinjaSendableValue() }
        case .string(let value):
            return value
        case .integer(let value):
            guard let value = Int(exactly: value) else {
                throw ToolCallParserError.malformed
            }
            return value
        case .unsignedInteger(let value):
            guard let value = Int(exactly: value) else {
                throw ToolCallParserError.malformed
            }
            return value
        case .decimal(let value):
            let text = NSDecimalNumber(decimal: value).stringValue
            guard let double = Double(text),
                  double.isFinite,
                  let roundTrip = Decimal(
                    string: String(double),
                    locale: Locale(identifier: "en_US_POSIX")),
                  roundTrip == value else {
                throw ToolCallParserError.malformed
            }
            return double
        case .number(let value):
            return value
        case .bool(let value):
            return value
        case .null:
            return Optional<String>.none as String?
        }
    }

    public var objectValue: [String: JSONValue]? {
        guard case .object(let value) = self else { return nil }
        return value
    }

    public func encoded(sortedKeys: Bool = true) throws -> String {
        let encoder = JSONEncoder()
        if sortedKeys { encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes] }
        return String(decoding: try encoder.encode(self), as: UTF8.self)
    }
}
