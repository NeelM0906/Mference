import Foundation

/// Search over notes (Notes spec 6.6): title, body text and transcript text,
/// case- and diacritic-insensitive; every word of the query must match.
/// `Note.searchText` holds the folded text, rewritten on every save.
public enum NoteSearch {
    /// Folds case and diacritics the same way for the index and the query.
    public static func fold(_ text: String) -> String {
        text.folding(options: [.caseInsensitive, .diacriticInsensitive, .widthInsensitive], locale: nil)
    }

    /// The query's words, folded; empty when the query is blank.
    public static func terms(_ query: String) -> [String] {
        fold(query).split(whereSeparator: { $0.isWhitespace }).map(String.init)
    }

    /// The text a note is found by: its title, body and transcript, folded.
    public static func searchText(title: String, document: NoteDocument, transcript: [TranscriptSegment]) -> String {
        var parts: [String] = []
        if !title.isEmpty, title != document.title { parts.append(title) }
        parts.append(document.plainText)
        parts.append(contentsOf: transcript.map(\.text))
        return fold(parts.filter { !$0.isEmpty }.joined(separator: "\n"))
    }

    /// Whether folded `text` contains every term.
    public static func matches(_ text: String, terms: [String]) -> Bool {
        terms.allSatisfy { text.contains($0) }
    }

    /// Whether `text` (not yet folded) contains every term.
    public static func matches(unfolded text: String, terms: [String]) -> Bool {
        matches(fold(text), terms: terms)
    }

    /// Transcript segments that contain every term, in time order (⌘K's
    /// transcript hits, each with its time).
    public static func transcriptHits(_ segments: [TranscriptSegment], terms: [String]) -> [TranscriptSegment] {
        guard !terms.isEmpty else { return [] }
        return segments.filter { matches(unfolded: $0.text, terms: terms) }.sorted { $0.start < $1.start }
    }

    /// Decodes `NoteRecording.transcriptJSON`; unreadable data is no segments.
    public static func transcript(_ data: Data?) -> [TranscriptSegment] {
        guard let data, !data.isEmpty else { return [] }
        return (try? JSONDecoder().decode([TranscriptSegment].self, from: data)) ?? []
    }
}
