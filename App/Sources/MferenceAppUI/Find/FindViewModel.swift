import Foundation
import MferenceAppCore
import Observation

/// The search box, filters and results behind both the Find screen and the
/// quick panel. Word matches show on every keystroke; meaning matches merge
/// in when typing pauses (the query is embedded while typing).
@Observable
public final class FindViewModel {
    public var query = "" {
        didSet { if query != oldValue { queryChanged() } }
    }
    public var kind: FindKindFilter = .all {
        didSet { if kind != oldValue { searchNow() } }
    }
    public var modified: FindDateFilter = .anyTime {
        didSet { if modified != oldValue { searchNow() } }
    }
    public var folder: URL? {
        didSet { if folder != oldValue { searchNow() } }
    }
    public private(set) var results = FindResults.empty
    public private(set) var isSearching = false
    public var selection: URL?
    public var showsWeak = false

    @ObservationIgnored public let find: FindService
    @ObservationIgnored private var wordsTask: Task<Void, Never>?
    @ObservationIgnored private var meaningTask: Task<Void, Never>?
    /// Pause after the last keystroke before meaning search runs.
    @ObservationIgnored public var pause: Duration = .milliseconds(220)

    public init(find: FindService) {
        self.find = find
    }

    public var selected: FindResult? {
        results.results.first { $0.url == selection } ?? results.strong.first
    }

    /// Results shown: strong ones, plus the weak ones when unfolded.
    public var visible: [FindResult] {
        showsWeak ? results.results : results.strong
    }

    /// Counts per kind filter for the current results.
    public func count(_ filter: FindKindFilter) -> Int {
        results.results.filter { filter.includes($0.kind) }.count
    }

    private var request: FindQuery {
        FindQuery(query, kind: kind, modified: modified, folder: folder)
    }

    private func queryChanged() {
        let text = query.trimmingCharacters(in: .whitespacesAndNewlines)
        wordsTask?.cancel()
        meaningTask?.cancel()
        guard !text.isEmpty else {
            results = .empty
            selection = nil
            isSearching = false
            return
        }
        find.prepare(query)
        isSearching = true
        let request = self.request
        wordsTask = Task { [weak self, find] in
            let words = await find.searchWords(request)
            guard !Task.isCancelled, let self else { return }
            self.apply(words)
        }
        let pause = self.pause
        meaningTask = Task { [weak self, find] in
            try? await Task.sleep(for: pause)
            guard !Task.isCancelled else { return }
            let full = await find.search(request)
            guard !Task.isCancelled, let self else { return }
            self.apply(full)
            self.isSearching = false
        }
    }

    /// Runs the full search at once (a filter changed, or Return).
    public func searchNow() {
        let text = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        wordsTask?.cancel()
        meaningTask?.cancel()
        isSearching = true
        let request = self.request
        meaningTask = Task { [weak self, find] in
            let full = await find.search(request)
            guard !Task.isCancelled, let self else { return }
            self.apply(full)
            self.isSearching = false
        }
    }

    private func apply(_ results: FindResults) {
        self.results = results
        if let selection, results.results.contains(where: { $0.url == selection }) { return }
        selection = results.strong.first?.url
    }

    /// Moves the selection up or down the visible list.
    public func moveSelection(by offset: Int) {
        let list = visible
        guard !list.isEmpty else { return }
        let current = list.firstIndex { $0.url == selection } ?? -1
        let next = min(max(current + offset, 0), list.count - 1)
        selection = list[next].url
    }

    /// "48,212 files · 40 ms", or what is still being learned.
    public var summary: String {
        guard !results.query.isEmpty else { return "" }
        let count = results.strong.count
        let files = count == 1 ? "1 file" : "\(count) files"
        let time = "\(max(1, Int((results.seconds * 1000).rounded()))) ms"
        return results.wordsOnly ? "\(files) · word matches · \(time)" : "\(files) · \(time)"
    }
}

/// Relative dates for result rows ("Mar 2", "Nov 18, 2025").
enum FindDates {
    static func short(_ date: Date, now: Date = .now) -> String {
        let calendar = Calendar.current
        if calendar.isDate(date, equalTo: now, toGranularity: .year) {
            return date.formatted(.dateTime.month(.abbreviated).day())
        }
        return date.formatted(.dateTime.month(.abbreviated).day().year())
    }
}

extension FindKind {
    /// The tile label ("PDF", "DOCX").
    func tileLabel(for url: URL) -> String {
        let ext = url.pathExtension.uppercased()
        return ext.isEmpty ? rawValue.uppercased() : String(ext.prefix(5))
    }
}

extension FindIndexStatus {
    /// One line for the banner and Settings.
    var line: String? {
        switch phase {
        case .off, .upToDate: nil
        case .scanning: "Looking through your folders…"
        case .reading: "Reading files: \(readFiles.formatted()) of \(totalFiles.formatted()). Word search works on those already read."
        case .learning: "Learning what your files mean: \(learnedFiles.formatted()) of \(totalFiles.formatted()). Word matches show first; better matches appear as it finishes."
        case .paused: "\(pauseReason ?? "Paused"). \(remainingFiles.formatted()) files still to learn."
        }
    }
}
