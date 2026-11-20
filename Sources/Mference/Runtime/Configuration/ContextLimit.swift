extension ModelFamily {
    /// The pinned checkpoint's native context: its `max_position_embeddings`.
    /// A longer context has positions the model was never trained to address.
    public var maximumContext: Int {
        switch self {
        case .qwen36, .qwen38: 262_144
        case .minicpm5: 131_072
        case .maple: 128_000
        }
    }

    /// The longest context any family accepts.
    public static var largestMaximumContext: Int {
        allCases.map(\.maximumContext).max() ?? 0
    }
}

/// A load asked for more context than the model supports.
public struct ContextLimitError: Error, Equatable, CustomStringConvertible {
    public let family: ModelFamily
    public let requested: Int
    public let maximum: Int

    public init(family: ModelFamily, requested: Int, maximum: Int) {
        self.family = family
        self.requested = requested
        self.maximum = maximum
    }

    public var description: String {
        "--max-context \(requested) is above \(family.rawValue)'s native context of \(maximum) tokens"
    }
}
