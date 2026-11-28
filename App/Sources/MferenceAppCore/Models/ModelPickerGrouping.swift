import Foundation

/// How the chat's model picker groups its rows: by size (the default) or by
/// model family. Recommended is a tag on Quick and Deep inside their group,
/// not a section of its own, so every model is listed once.
public enum ModelPickerGrouping: String, Sendable, Hashable, CaseIterable, Identifiable {
    case size
    case family

    /// The `UserDefaults` key the picker keeps the choice under.
    public static let storageKey = "modelPickerGrouping"

    public var id: String { rawValue }

    /// The switch's label ("By size").
    public var title: String {
        switch self {
        case .size: "By size"
        case .family: "By family"
        }
    }

    /// One picker section.
    public struct Section: Sendable, Hashable, Identifiable {
        public let title: String
        /// Small text after the title ("up to 4B"), or nil.
        public let hint: String?
        public let models: [AppModel]

        public var id: String { title }
    }

    /// The sections for `models`, in picker order. A model not in `models`
    /// is left out (Apple's row while Apple Intelligence is off), and an
    /// empty section is dropped.
    public func sections(for models: [AppModel]) -> [Section] {
        let offered = Set(models)
        let layout: [(String, String?, [AppModel])] = switch self {
        case .size:
            [("Small", "up to 4B", [.quick, .minicpm5, .minicpm5_1b, .apple]),
             ("Large", "27B dense", [.deep, .bonsai27b]),
             ("Mixture of experts", nil, [.qwen36, .inkling2Mini, .maple])]
        case .family:
            [("Qwen", nil, [.quick, .deep, .qwen36]),
             ("MiniCPM", nil, [.minicpm5, .minicpm5_1b]),
             ("Inkling", nil, [.inkling2Mini]),
             ("Bonsai", "built on Qwen3.8", [.bonsai27b]),
             ("Maple", nil, [.maple]),
             ("Apple", nil, [.apple])]
        }
        return layout.compactMap { title, hint, members in
            let shown = members.filter(offered.contains)
            return shown.isEmpty ? nil : Section(title: title, hint: hint, models: shown)
        }
    }
}
