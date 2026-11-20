import Foundation

/// One ⌘K result from Notes (Notes spec 6.6).
public struct NoteSearchHit: Sendable, Hashable, Identifiable {
    public enum Match: Sendable, Hashable {
        /// The title matches every word.
        case title
        /// The body text matches (every word somewhere in title or body).
        case body
        /// A transcript line matches; choosing it opens the note at that line.
        case transcript(segmentID: UUID, time: TimeInterval)
    }

    public var noteID: UUID
    public var title: String
    public var kind: NoteKind
    public var match: Match
    /// The line that matched (body or transcript), for the row's second line.
    public var snippet: String

    public var id: String {
        switch match {
        case .title: "\(noteID)-title"
        case .body: "\(noteID)-body"
        case .transcript(let segment, _): "\(noteID)-\(segment)"
        }
    }

    public init(noteID: UUID, title: String, kind: NoteKind, match: Match, snippet: String) {
        self.noteID = noteID
        self.title = title
        self.kind = kind
        self.match = match
        self.snippet = snippet
    }
}

extension NoteStore {
    /// ⌘K hits for `query`: title matches first, then body matches, then
    /// transcript lines with their times. Notes in Recently deleted are
    /// left out. At most `limit` hits.
    public func searchHits(_ query: String, limit: Int = 12) throws -> [NoteSearchHit] {
        let terms = NoteSearch.terms(query)
        guard !terms.isEmpty else { return [] }
        var titles: [NoteSearchHit] = []
        var bodies: [NoteSearchHit] = []
        var transcripts: [NoteSearchHit] = []
        for note in try notes(in: .all, matching: query) {
            let title = displayTitle(of: note)
            let document = document(of: note)
            if NoteSearch.matches(unfolded: title, terms: terms) {
                titles.append(NoteSearchHit(noteID: note.id, title: title, kind: note.kind, match: .title,
                                            snippet: document.previewLine))
                continue
            }
            if NoteSearch.matches(unfolded: title + "\n" + document.plainText, terms: terms) {
                let line = document.lines.first { NoteSearch.matches(unfolded: $0, terms: terms) }
                    ?? document.lines.first { line in terms.contains { NoteSearch.fold(line).contains($0) } }
                    ?? document.previewLine
                bodies.append(NoteSearchHit(noteID: note.id, title: title, kind: note.kind, match: .body, snippet: line))
                continue
            }
            let segments = NoteSearch.transcript(note.recording?.transcriptJSON)
            for segment in NoteSearch.transcriptHits(segments, terms: terms).prefix(3) {
                transcripts.append(NoteSearchHit(noteID: note.id, title: title, kind: note.kind,
                                                 match: .transcript(segmentID: segment.id, time: segment.start),
                                                 snippet: segment.text))
            }
        }
        return Array((titles + bodies + transcripts).prefix(limit))
    }
}
