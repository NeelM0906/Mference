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
    /// the result — schema key validation, Gemma schema normalization, the Jinja
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

    /// Rewrites this JSON Schema fragment so every property the Gemma tool
    /// template renders carries a concrete scalar `type`. The template evaluates
    /// `value['type'] | upper` unconditionally for each property, so union
    /// (`anyOf`/`oneOf`/`allOf`), array-typed (`"type": ["string", "null"]`), or
    /// type-less properties would abort rendering. `anyOf`/`oneOf` collapse to
    /// the first branch resolving to a concrete non-null type, `allOf` merges
    /// its branches; both take on the parent's sibling keys (e.g. `description`)
    /// and run the result through normalization again. An array `type` takes its
    /// first non-null member; a type-less node defaults to `object` when it
    /// carries `properties`, otherwise `string`. Every nested `properties` value
    /// and `items` schema is normalized recursively. Non-object values are
    /// returned unchanged.
    ///
    /// The ChatML dialect renders `tool | tojson`, so it must not go through
    /// this: unions there are carried to the model intact.
    public func gemmaSchemaNormalized() -> JSONValue {
        guard case .object(var object) = self else { return self }

        if !object.hasScalarStringType,
           case .array(let members)? = object["type"],
           let concrete = members.firstNonNullTypeName {
            object["type"] = .string(concrete)
        }
        if !object.hasScalarStringType {
            for keyword in Self.unionKeywords {
                guard case .array(let branches)? = object[keyword] else { continue }
                return keyword == "allOf"
                    ? Self.mergedIntersection(parent: object, branches: branches)
                    : Self.collapsedUnion(parent: object, branches: branches)
            }
        }
        if !object.hasScalarStringType {
            object["type"] = object["properties"] != nil
                ? .string("object")
                : .string("string")
        }

        if case .object(let properties)? = object["properties"] {
            object["properties"] = .object(
                properties.mapValues { $0.gemmaSchemaNormalized() })
        }
        switch object["items"] {
        case .object?:
            object["items"] = object["items"]?.gemmaSchemaNormalized()
        case .array(let items)?:
            object["items"] = .array(items.map { $0.gemmaSchemaNormalized() })
        default:
            break
        }
        return .object(object)
    }

    private static let unionKeywords = ["anyOf", "oneOf", "allOf"]

    /// Picks the single branch of an `anyOf`/`oneOf` that the template renders.
    /// A `{"type":"null"}` branch is taken only when no other branch resolves:
    /// a NULL-typed parameter reads to the model as "this argument is null",
    /// which is the opposite of what a nullable parameter means.
    private static func collapsedUnion(parent: [String: JSONValue],
                                       branches: [JSONValue]) -> JSONValue {
        let resolved = branches
            .compactMap { $0.gemmaSchemaNormalized().objectValue }
            .filter(\.hasScalarStringType)
        let chosen = resolved.first { $0["type"] != .string("null") } ?? resolved.first
        return completed(chosen ?? [:], siblingsOf: parent)
    }

    /// `allOf` is an intersection, not a choice: every branch constrains the
    /// same value. Keeping only the first would hide the parameters the later
    /// branches declare, so branch keys are merged instead — earlier branches
    /// win a conflict, and `properties` maps are unioned.
    private static func mergedIntersection(parent: [String: JSONValue],
                                           branches: [JSONValue]) -> JSONValue {
        var merged: [String: JSONValue] = [:]
        for case .object(let branch) in branches {
            absorb(branch, into: &merged)
        }
        return completed(merged, siblingsOf: parent)
    }

    /// Finishes a collapsed union or a merged intersection: the parent's own
    /// keys (`description` and the like) join the result, which then goes back
    /// through normalization. That second pass is what normalizes the keys the
    /// parent contributed — nested `properties` and `items` above all — and what
    /// gives a result that still names no type the same shape default as any
    /// other type-less node. It terminates because every pass consumes the union
    /// keyword it collapsed, and a keyword can only reappear from one level
    /// further down a tree whose depth `maximumDepth` bounds.
    private static func completed(_ collapsed: [String: JSONValue],
                                  siblingsOf parent: [String: JSONValue]) -> JSONValue {
        var result = collapsed
        var siblings = parent
        for keyword in unionKeywords { siblings[keyword] = nil }
        absorb(siblings, into: &result)
        return JSONValue.object(result).gemmaSchemaNormalized()
    }

    /// Adds `addition`'s keys to `base` without displacing what is already
    /// there, unioning `properties` maps so no declared parameter is dropped.
    private static func absorb(_ addition: [String: JSONValue],
                               into base: inout [String: JSONValue]) {
        for (key, value) in addition {
            if key == "properties",
               case .object(let existing)? = base[key],
               case .object(let incoming) = value {
                base[key] = .object(existing.merging(incoming) { current, _ in current })
            } else if base[key] == nil {
                base[key] = value
            }
        }
    }

    public func encoded(sortedKeys: Bool = true) throws -> String {
        let encoder = JSONEncoder()
        if sortedKeys { encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes] }
        return String(decoding: try encoder.encode(self), as: UTF8.self)
    }

    public func gemmaToolArgumentBody() throws -> String {
        guard case .object(let value) = self else {
            throw ToolCallParserError.malformed
        }
        return try value.keys.sorted().map { key in
            guard GemmaToolCallParser.isRepresentableObjectKey(key) else {
                throw ToolCallParserError.malformed
            }
            return "\(key):\(try value[key]!.gemmaToolValue())"
        }.joined(separator: ",")
    }

    private func gemmaToolValue() throws -> String {
        switch self {
        case .object:
            return "{\(try gemmaToolArgumentBody())}"
        case .array(let value):
            return "[\(try value.map { try $0.gemmaToolValue() }.joined(separator: ","))]"
        case .string(let value):
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.withoutEscapingSlashes]
            return String(decoding: try encoder.encode(value), as: UTF8.self)
        case .integer(let value):
            return String(value)
        case .unsignedInteger(let value):
            return String(value)
        case .decimal(let value):
            return NSDecimalNumber(decimal: value).stringValue
        case .number(let value):
            guard value.isFinite else {
                throw ToolCallParserError.malformed
            }
            return String(value)
        case .bool(let value):
            return value ? "true" : "false"
        case .null:
            return "null"
        }
    }
}

private extension Dictionary where Key == String, Value == JSONValue {
    /// True when `type` is present as a single JSON string, which is the only
    /// shape the Gemma template can feed to its `| upper` filter.
    var hasScalarStringType: Bool {
        if case .string? = self["type"] { return true }
        return false
    }
}

private extension Array where Element == JSONValue {
    /// The first `type` member that names a concrete (non-`"null"`) type, used to
    /// flatten `"type": ["string", "null"]` into a single scalar type.
    var firstNonNullTypeName: String? {
        for case .string(let name) in self where name != "null" { return name }
        return nil
    }
}
