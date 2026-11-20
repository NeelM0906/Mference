import Foundation

/// Text helpers shared by indexing and search: words, chunks, the prose
/// gate and snippets.
public enum FindText {
    /// Characters per chunk and the overlap between neighbours (about 450
    /// and 50 tokens; DigUp's measured setting).
    public static let chunkCharacters = 1_800
    public static let chunkOverlap = 200
    /// Chunks kept per file at most.
    public static let maxChunksPerFile = 5_000

    /// Lowercased words (letters, digits, apostrophes and inner hyphens).
    public static func words(_ text: String) -> [String] {
        var words: [String] = []
        var current = ""
        for scalar in text.lowercased().unicodeScalars {
            if CharacterSet.alphanumerics.contains(scalar) || scalar == "'" || scalar == "-" {
                current.unicodeScalars.append(scalar)
            } else if !current.isEmpty {
                words.append(current.trimmingCharacters(in: CharacterSet(charactersIn: "'-")))
                current = ""
            }
        }
        if !current.isEmpty { words.append(current.trimmingCharacters(in: CharacterSet(charactersIn: "'-"))) }
        return words.filter { !$0.isEmpty }
    }

    /// The query's searchable words: no stop words, no duplicates, in order.
    public static func queryWords(_ text: String) -> [String] {
        var seen = Set<String>()
        return words(text).filter { !stopWords.contains($0) && seen.insert($0).inserted }
    }

    public static let stopWords: Set<String> = [
        "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "about", "at", "by", "from", "is",
        "are", "was", "were", "be", "it", "its", "this", "that", "these", "those", "my", "our", "your", "me", "we",
        "i", "you", "what", "which", "where", "when", "how", "who", "file", "files", "find", "show", "any", "all",
        "some", "there", "has", "have", "had", "do", "does", "did", "can", "could", "should", "would", "into",
        "than", "then", "so", "if", "not", "no", "as", "up", "out", "one",
    ]

    /// Splits text into overlapping chunks at paragraph, then sentence, then
    /// word boundaries.
    public static func chunks(_ text: String, size: Int = chunkCharacters, overlap: Int = chunkOverlap,
                              limit: Int = maxChunksPerFile) -> [String] {
        let cleaned = normalizeWhitespace(text)
        guard !cleaned.isEmpty else { return [] }
        guard cleaned.count > size else { return [cleaned] }
        var chunks: [String] = []
        var start = cleaned.startIndex
        while start < cleaned.endIndex, chunks.count < limit {
            let hardEnd = cleaned.index(start, offsetBy: size, limitedBy: cleaned.endIndex) ?? cleaned.endIndex
            var end = hardEnd
            if hardEnd < cleaned.endIndex {
                let window = cleaned[start..<hardEnd]
                let minimum = cleaned.index(start, offsetBy: size / 2)
                if let cut = window.range(of: "\n\n", options: .backwards), cut.upperBound > minimum {
                    end = cut.upperBound
                } else if let cut = lastSentenceEnd(in: window), cut > minimum {
                    end = cut
                } else if let cut = window.lastIndex(where: \.isWhitespace), cut > minimum {
                    end = cleaned.index(after: cut)
                }
            }
            let chunk = cleaned[start..<end].trimmingCharacters(in: .whitespacesAndNewlines)
            if !chunk.isEmpty { chunks.append(chunk) }
            guard end < cleaned.endIndex else { break }
            // Step back by the overlap, to the start of a word.
            var next = cleaned.index(end, offsetBy: -overlap, limitedBy: start) ?? end
            if next <= start { next = end }
            while next < end, !cleaned[next].isWhitespace, next > start,
                  !cleaned[cleaned.index(before: next)].isWhitespace {
                next = cleaned.index(after: next)
            }
            start = next
        }
        return chunks
    }

    private static func lastSentenceEnd(in window: Substring) -> String.Index? {
        var index = window.endIndex
        while index > window.startIndex {
            index = window.index(before: index)
            let character = window[index]
            if character == "." || character == "!" || character == "?" || character == "。" || character == "।" {
                let after = window.index(after: index)
                if after == window.endIndex || window[after].isWhitespace { return after }
            }
        }
        return nil
    }

    /// Collapses runs of spaces and tabs, keeps paragraph breaks.
    public static func normalizeWhitespace(_ text: String) -> String {
        var lines: [String] = []
        var blank = 0
        for line in text.replacingOccurrences(of: "\r\n", with: "\n").split(separator: "\n",
                                                                               omittingEmptySubsequences: false) {
            let trimmed = line.split(whereSeparator: { $0 == " " || $0 == "\t" || $0 == "\u{00A0}" })
                .joined(separator: " ")
            if trimmed.isEmpty {
                blank += 1
                if blank == 1, !lines.isEmpty { lines.append("") }
            } else {
                blank = 0
                lines.append(trimmed)
            }
        }
        return lines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Whether a chunk is prose worth a meaning vector: at least 20% letters
    /// and 4 words. Number dumps and stubs stay searchable by words only, so
    /// they cannot win generic queries (DigUp's prose gate).
    public static func isProse(_ text: String) -> Bool {
        let scalars = text.unicodeScalars
        guard !scalars.isEmpty else { return false }
        let letters = scalars.filter { CharacterSet.letters.contains($0) }.count
        return Double(letters) / Double(scalars.count) >= 0.2 && words(text).count >= 4
    }

    /// Picks the passage to show from a chunk: the sentence (or line) that
    /// shares most words with the query, with a little context either side.
    public static func snippet(of chunk: String, queryWords: [String],
                               maxHit: Int = 180, context: Int = 60) -> (before: String, hit: String, after: String) {
        let text = chunk.replacingOccurrences(of: "\n", with: " ")
        let sentences = sentenceRanges(text)
        guard !sentences.isEmpty else { return ("", String(text.prefix(maxHit)), "") }
        let query = Set(queryWords.map { stem($0) })
        var best = sentences[0]
        var bestScore = -1
        for range in sentences {
            let found = Set(words(String(text[range])).map { stem($0) })
            let score = query.intersection(found).count
            if score > bestScore {
                best = range
                bestScore = score
            }
        }
        var hit = String(text[best]).trimmingCharacters(in: .whitespaces)
        if hit.count > maxHit { hit = String(hit.prefix(maxHit)) + "…" }
        let beforeStart = text.index(best.lowerBound, offsetBy: -context, limitedBy: text.startIndex) ?? text.startIndex
        var before = String(text[beforeStart..<best.lowerBound])
        if beforeStart > text.startIndex { before = "…" + before.drop(while: { !$0.isWhitespace }) }
        let afterEnd = text.index(best.upperBound, offsetBy: context, limitedBy: text.endIndex) ?? text.endIndex
        var after = String(text[best.upperBound..<afterEnd])
        if afterEnd < text.endIndex { after = String(after.reversed().drop(while: { !$0.isWhitespace }).reversed()) + "…" }
        return (before.trimmingCharacters(in: .whitespaces).isEmpty ? "" : before,
                hit, after.trimmingCharacters(in: .whitespaces).isEmpty ? "" : after)
    }

    static func sentenceRanges(_ text: String) -> [Range<String.Index>] {
        var ranges: [Range<String.Index>] = []
        var start = text.startIndex
        var index = text.startIndex
        while index < text.endIndex {
            let character = text[index]
            let next = text.index(after: index)
            if character == "." || character == "!" || character == "?" || character == "。",
               next == text.endIndex || text[next].isWhitespace {
                ranges.append(start..<next)
                start = next
            }
            index = next
        }
        if start < text.endIndex { ranges.append(start..<text.endIndex) }
        return ranges.filter { !text[$0].trimmingCharacters(in: .whitespaces).isEmpty }
    }

    /// A crude English stem (plural and -ing/-ed), for matching query words
    /// against text the way FTS5's porter tokenizer does, near enough.
    public static func stem(_ word: String) -> String {
        for suffix in ["ing", "ed", "es", "s"] where word.count > suffix.count + 2 && word.hasSuffix(suffix) {
            return String(word.dropLast(suffix.count))
        }
        return word
    }
}
