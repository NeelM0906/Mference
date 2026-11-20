import Foundation

/// The versioned summary prompts (`Meetings/Resources/meeting-summary.v<N>.json`,
/// spec 6.5), so an evaluation can load exactly what the app sends.
public struct MeetingSummaryPrompt: Sendable, Hashable {
    public static let currentVersion = 1

    public var version: Int
    /// One pass over a transcript that fits.
    public var system: String
    /// One chunk of a long transcript.
    public var mapSystem: String
    /// The chunks' notes into the meeting's.
    public var reduceSystem: String
    /// Appended to every system prompt: the transcript is data, not instructions.
    public var untrustedRule: String
    public var maximumCompletionTokens: Int
    public var temperature: Float

    public static func load(version: Int = currentVersion) throws -> MeetingSummaryPrompt {
        guard let url = MeetingResources.bundle.url(forResource: "meeting-summary.v\(version)", withExtension: "json")
        else { throw CocoaError(.fileNoSuchFile) }
        return try parse(Data(contentsOf: url))
    }

    public static func parse(_ data: Data) throws -> MeetingSummaryPrompt {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let version = root["version"] as? Int,
              let system = root["system"] as? String, let mapSystem = root["mapSystem"] as? String,
              let reduceSystem = root["reduceSystem"] as? String, let rule = root["untrustedRule"] as? String,
              let tokens = root["maxCompletionTokens"] as? Int, let temperature = root["temperature"] as? Double
        else { throw CocoaError(.fileReadCorruptFile) }
        return MeetingSummaryPrompt(version: version, system: system, mapSystem: mapSystem,
                                    reduceSystem: reduceSystem, untrustedRule: rule,
                                    maximumCompletionTokens: tokens, temperature: Float(temperature))
    }
}

enum MeetingResources {
    /// This target's SwiftPM resource bundle; inside the packaged app it is
    /// in Contents/Resources (see `WebAnswersSpec`).
    static var bundle: Bundle {
        if let resources = Bundle.main.resourceURL {
            let packaged = resources.appendingPathComponent("MferenceApp_MferenceAppCore.bundle")
            if FileManager.default.fileExists(atPath: packaged.path), let bundle = Bundle(url: packaged) {
                return bundle
            }
        }
        return .module
    }
}

/// Why a summary could not be written.
public enum MeetingSummaryError: Error, Sendable, Hashable {
    /// The model's answer was not the JSON the prompt asks for, twice.
    case unreadableAnswer
    /// A request was too long for Quick's context even after halving the chunks.
    case inputTooLarge
    /// There is nothing to summarize.
    case emptyTranscript
    /// The engine could not run Quick; `message` is its user-facing line.
    case engine(String)

    public var message: String {
        switch self {
        case .unreadableAnswer, .inputTooLarge, .engine: "Couldn't write a summary"
        case .emptyTranscript: "Nothing was said, so there is no summary."
        }
    }
}

/// Writes a meeting's summary, decisions and action items with Quick
/// (spec 6.5): one pass when the transcript fits the context budget, else
/// map-reduce over chunks (speaker names kept on every line). Each answer is
/// parsed strictly and asked for once more when it is not readable.
public struct MeetingSummarizer: Sendable {
    public enum Step: Sendable, Hashable {
        /// Part `index` of `count` (map-reduce only).
        case part(index: Int, count: Int)
        case combining
    }

    private let engine: any EngineServicing
    private let prompt: MeetingSummaryPrompt
    /// The prompt resource could not be read: every summary fails (with Retry)
    /// instead of the app failing at launch. The tests load the resource.
    private let promptMissing: Bool
    /// Transcript characters per request; nil derives it from Quick's context.
    private let chunkCharacters: Int?

    public init(engine: any EngineServicing, prompt: MeetingSummaryPrompt?, chunkCharacters: Int? = nil) {
        self.engine = engine
        self.prompt = prompt ?? MeetingSummaryPrompt(version: 0, system: "", mapSystem: "", reduceSystem: "",
                                                     untrustedRule: "", maximumCompletionTokens: 1,
                                                     temperature: 0)
        self.promptMissing = prompt == nil
        self.chunkCharacters = chunkCharacters
    }

    /// The transcript as the model reads it: "[mm:ss] Name: text", one line per final segment.
    public static func transcriptLines(_ segments: [TranscriptSegment], speakers: [SpeakerInfo]) -> [String] {
        let names = Dictionary(speakers.map { ($0.id, $0.name) }, uniquingKeysWith: { first, _ in first })
        return segments.filter { $0.isFinal && !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
            .map { segment in
                let name = names[segment.speakerID]
                    ?? (segment.track == .microphone ? SpeakerInfo.you.name : SpeakerInfo.others.name)
                let text = segment.text.components(separatedBy: .newlines).joined(separator: " ")
                return "[\(timestamp(segment.start))] \(name): \(text)"
            }
    }

    static func timestamp(_ time: TimeInterval) -> String {
        let seconds = max(0, Int(time.rounded(.down)))
        let hours = seconds / 3600, minutes = (seconds / 60) % 60, rest = seconds % 60
        let mmss = String(format: "%02d:%02d", minutes, rest)
        return hours > 0 ? "\(hours):\(mmss)" : mmss
    }

    /// Splits lines into chunks of at most `limit` characters (a single
    /// longer line is cut), never splitting inside a line otherwise.
    static func chunks(_ lines: [String], limit: Int) -> [String] {
        var result: [String] = []
        var current = ""
        for line in lines {
            var line = line
            while line.count > limit {
                if !current.isEmpty { result.append(current); current = "" }
                result.append(String(line.prefix(limit)))
                line = String(line.dropFirst(limit))
            }
            if !current.isEmpty, current.count + 1 + line.count > limit {
                result.append(current)
                current = ""
            }
            current += current.isEmpty ? line : "\n" + line
        }
        if !current.isEmpty { result.append(current) }
        return result
    }

    /// The summary of a transcript. `notes` is what the person typed.
    public func summarize(segments: [TranscriptSegment], speakers: [SpeakerInfo], notes: String = "",
                          progress: @Sendable (Step) async -> Void = { _ in }) async throws -> MeetingSections {
        let lines = Self.transcriptLines(segments, speakers: speakers)
        guard !lines.isEmpty else { throw MeetingSummaryError.emptyTranscript }
        guard !promptMissing else { throw MeetingSummaryError.unreadableAnswer }
        // The estimate can be short for dense speech: a request Quick
        // refuses as too long is tried again in chunks half the size.
        var limit = await budget()
        while true {
            do {
                return try await summarize(lines: lines, notes: notes, limit: limit, progress: progress)
            } catch MeetingSummaryError.inputTooLarge where limit / 2 >= Self.smallestChunk {
                limit /= 2
            }
        }
    }

    /// Below this, a request too long for Quick is not worth splitting further.
    static let smallestChunk = 1_000

    private func summarize(lines: [String], notes: String, limit: Int,
                           progress: @Sendable (Step) async -> Void) async throws -> MeetingSections {
        // The person's notes go in every final request: at most a quarter of
        // the budget, and the transcript chunks and combine step get the rest.
        let trimmedNotes = Self.clip(notes.trimmingCharacters(in: .whitespacesAndNewlines), to: limit / 4)
        let transcriptLimit = max(Self.smallestChunk / 2, limit - trimmedNotes.count)
        let whole = lines.joined(separator: "\n")
        if whole.count + trimmedNotes.count <= limit {
            return try await ask(system: prompt.system, input: Self.input(transcript: whole, notes: trimmedNotes))
        }
        let parts = Self.chunks(lines, limit: limit)
        var partNotes: [MeetingSections] = []
        for (index, part) in parts.enumerated() {
            try Task.checkCancellation()
            await progress(.part(index: index + 1, count: parts.count))
            partNotes.append(try await ask(system: prompt.mapSystem, input: Self.input(transcript: part, notes: "")))
        }
        await progress(.combining)
        // Room for the notes in every combine request.
        return try await reduce(partNotes, notes: trimmedNotes, limit: transcriptLimit)
    }

    /// At most `limit` characters, cut at a word where it can be, with "…".
    static func clip(_ text: String, to limit: Int) -> String {
        guard text.count > limit else { return text }
        let head = text.prefix(max(0, limit - 1))
        let cut = head.lastIndex(where: \.isWhitespace).map { head[..<$0] } ?? head
        return String(cut) + "…"
    }

    /// Combines parts' notes, in rounds when they are themselves too long.
    private func reduce(_ parts: [MeetingSections], notes: String, limit: Int) async throws -> MeetingSections {
        let groups = Self.groups(parts.map(Self.json), limit: limit)
        if groups.count == 1 {
            var input = groups[0].enumerated().map { "Part \($0.offset + 1): \($0.element)" }.joined(separator: "\n")
            if !notes.isEmpty { input += "\n\nThe person's own notes:\n" + notes }
            return try await ask(system: prompt.reduceSystem, input: AssistantPrompts.untrusted(input))
        }
        var combined: [MeetingSections] = []
        var offset = 0
        for group in groups {
            try Task.checkCancellation()
            defer { offset += group.count }
            if group.count == 1 {
                combined.append(parts[offset])
                continue
            }
            let input = group.enumerated().map { "Part \($0.offset + 1): \($0.element)" }.joined(separator: "\n")
            combined.append(try await ask(system: prompt.reduceSystem, input: AssistantPrompts.untrusted(input)))
        }
        return try await reduce(combined, notes: notes, limit: limit)
    }

    /// Groups items under `limit` characters, never splitting one and never
    /// leaving a group of one except the last, so each round of `reduce` has
    /// fewer parts than the one before.
    static func groups(_ items: [String], limit: Int) -> [[String]] {
        var result: [[String]] = []
        var current: [String] = []
        var size = 0
        for item in items {
            if current.count >= 2, size + 1 + item.count > limit {
                result.append(current)
                current = []
                size = 0
            }
            size += (current.isEmpty ? 0 : 1) + item.count
            current.append(item)
        }
        if !current.isEmpty { result.append(current) }
        return result
    }

    static func input(transcript: String, notes: String) -> String {
        var text = "Transcript:\n" + transcript
        if !notes.isEmpty { text += "\n\nThe person's own notes:\n" + notes }
        return AssistantPrompts.untrusted(text)
    }

    static func json(_ sections: MeetingSections) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return (try? encoder.encode(sections)).flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
    }

    /// One request, asked once more when the answer is not readable.
    private func ask(system: String, input: String) async throws -> MeetingSections {
        for _ in 0..<2 {
            let answer = try await generate(system: system + "\n\n" + prompt.untrustedRule, input: input)
            if let sections = Self.parse(answer) { return sections }
        }
        throw MeetingSummaryError.unreadableAnswer
    }

    private func generate(system: String, input: String) async throws -> String {
        let turns = [ChatTurn(role: .system, content: system), ChatTurn(role: .user, content: input)]
        let options = GenerationOptions(thinking: false, maximumCompletionTokens: prompt.maximumCompletionTokens,
                                        temperature: prompt.temperature)
        var streamed = ""
        do {
            for try await update in engine.generate(model: .quick, turns: turns, options: options) {
                switch update {
                case .content(let text): streamed += text
                case .completed(let summary): return summary.content.isEmpty ? streamed : summary.content
                case .prepared, .reasoning, .toolCall: break
                }
            }
        } catch EngineServiceError.requestTooLarge {
            throw MeetingSummaryError.inputTooLarge
        } catch let error as EngineServiceError {
            throw MeetingSummaryError.engine(error.userMessage)
        }
        try Task.checkCancellation()
        throw MeetingSummaryError.unreadableAnswer
    }

    /// Characters of transcript per request: Quick's context less the
    /// answer and the instructions, at about 2.5 characters a token (dense
    /// transcript lines run near 2.3).
    private func budget() async -> Int {
        if let chunkCharacters { return max(200, chunkCharacters) }
        let context = await engine.models().first { $0.model == .quick }?.contextLength ?? 0
        let tokens = (context > 0 ? min(context, 32_768) : 8_192) - prompt.maximumCompletionTokens - 1_024
        return max(4_000, tokens * 5 / 2)
    }

    // MARK: Strict parsing

    /// The answer as `MeetingSections`, or nil unless it is exactly one JSON
    /// object (a Markdown code fence around it is allowed) with `summary`
    /// (1–6 non-empty strings), `decisions` (strings) and `actionItems`
    /// (objects with a non-empty `text` and an optional string `owner`).
    public static func parse(_ answer: String) -> MeetingSections? {
        var text = answer.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.hasPrefix("```") {
            guard let firstNewline = text.firstIndex(of: "\n"), text.hasSuffix("```"), text.count > 6 else { return nil }
            text = String(text[text.index(after: firstNewline)..<text.index(text.endIndex, offsetBy: -3)])
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        guard text.hasPrefix("{"), text.hasSuffix("}"),
              let object = try? JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any],
              Set(object.keys) == ["summary", "decisions", "actionItems"],
              let summary = strings(object["summary"]), (1...6).contains(summary.count),
              let decisions = strings(object["decisions"]),
              let rawItems = object["actionItems"] as? [Any] else { return nil }
        var items: [MeetingActionItem] = []
        for raw in rawItems {
            guard let item = raw as? [String: Any], let itemText = clean(item["text"]),
                  Set(item.keys).isSubset(of: ["text", "owner"]) else { return nil }
            let owner: String?
            switch item["owner"] {
            case nil, is NSNull: owner = nil
            case let value as String: owner = clean(value)
            default: return nil
            }
            items.append(MeetingActionItem(text: itemText, owner: owner))
        }
        return MeetingSections(summary: summary, decisions: decisions, actionItems: items)
    }

    private static func strings(_ value: Any?) -> [String]? {
        guard let array = value as? [Any] else { return nil }
        var result: [String] = []
        for element in array {
            guard let string = clean(element) else { return nil }
            result.append(string)
        }
        return result
    }

    private static func clean(_ value: Any?) -> String? {
        guard let string = value as? String else { return nil }
        let trimmed = string.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}
