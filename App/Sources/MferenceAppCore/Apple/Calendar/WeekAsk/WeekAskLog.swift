import Foundation
import Synchronization

/// One question, as Ask answered it: one line of the log.
public struct WeekAskLogEntry: Codable, Sendable, Hashable {
    /// ISO 8601, UTC.
    public var time: String
    /// "off" (mode A) or "warn" (mode B).
    public var review: String
    /// "Quick 4B", or "none" when no model could run.
    public var model: String
    public var question: String
    /// Earlier questions in the thread.
    public var historyCount: Int
    /// "lines" or "json": how the schedule was written (nil in older lines).
    public var format: String?
    /// The schedule the model read.
    public var schedule: String
    /// The model's reply, exactly as it came back.
    public var reply: String
    /// What the card showed.
    public var shown: String
    public var cited: [String]
    public var warnings: [String]
    /// "model", "noModel", "modelFailed".
    public var source: String
    public var wasCut: Bool
    public var latencyMilliseconds: Int
    public var promptTokens: Int?
    public var completionTokens: Int?
    public var error: String?

    public init(time: String, review: String, model: String, question: String, historyCount: Int,
                format: String? = nil, schedule: String,
                reply: String, shown: String, cited: [String], warnings: [String], source: String, wasCut: Bool,
                latencyMilliseconds: Int, promptTokens: Int?, completionTokens: Int?, error: String?) {
        self.time = time
        self.review = review
        self.model = model
        self.question = question
        self.historyCount = historyCount
        self.format = format
        self.schedule = schedule
        self.reply = reply
        self.shown = shown
        self.cited = cited
        self.warnings = warnings
        self.source = source
        self.wasCut = wasCut
        self.latencyMilliseconds = latencyMilliseconds
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.error = error
    }
}

/// Internal-testing log of how models answer Ask about your week: one JSON
/// line per question in `~/Library/Logs/Mference/week-ask.jsonl`, read with
/// `App/Scripts/week_ask_report.py`. Off unless turned on
/// (`defaults write ai.mference.app WeekAskLog -bool true`), because each
/// line holds calendar details; the file is readable by its owner only and
/// rolls over to `.1` at its size cap.
public final class WeekAskLog: Sendable {
    public let url: URL
    private let maxBytes: Int
    private let isEnabled: @Sendable () -> Bool
    private let lock = Mutex(())

    public init(url: URL, maxBytes: Int = 1_000_000, isEnabled: @escaping @Sendable () -> Bool) {
        self.url = url
        self.maxBytes = maxBytes
        self.isEnabled = isEnabled
    }

    public static let defaultsKey = "WeekAskLog"

    /// The app's: `~/Library/Logs/Mference/week-ask.jsonl`, on while the
    /// `WeekAskLog` default is true.
    public static func live() -> WeekAskLog {
        let logs = FileManager.default.homeDirectoryForCurrentUser.appending(path: "Library/Logs/Mference")
        return WeekAskLog(url: logs.appending(path: "week-ask.jsonl"),
                          isEnabled: { UserDefaults.standard.bool(forKey: defaultsKey) })
    }

    public var enabled: Bool { isEnabled() }

    /// Appends one line. A log that cannot be written is skipped: it never
    /// stops an answer.
    public func record(_ entry: WeekAskLogEntry) {
        guard isEnabled() else { return }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        guard var line = try? encoder.encode(entry) else { return }
        line.append(0x0A)
        lock.withLock { _ in
            let files = FileManager.default
            let directory = url.deletingLastPathComponent()
            try? files.createDirectory(at: directory, withIntermediateDirectories: true,
                                       attributes: [.posixPermissions: 0o700])
            if let size = (try? files.attributesOfItem(atPath: url.path))?[.size] as? NSNumber,
               size.intValue + line.count > maxBytes {
                let previous = url.path + ".1"
                try? files.removeItem(atPath: previous)
                try? files.moveItem(atPath: url.path, toPath: previous)
            }
            if !files.fileExists(atPath: url.path) {
                files.createFile(atPath: url.path, contents: nil, attributes: [.posixPermissions: 0o600])
            }
            guard let handle = try? FileHandle(forWritingTo: url) else { return }
            defer { try? handle.close() }
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: line)
        }
    }
}
