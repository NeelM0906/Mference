import Foundation

/// Joins a dictated phrase into the text around the place it lands
/// (Notes spec 4.2): one space between phrases, never two; no leading space
/// at the start of a line; a capital letter after the end of a sentence or at
/// the start of a line. Pure, so dictation tests need no editor.
public enum PhraseJoiner {
    /// The text to insert for `phrase` between `preceding` and `following`
    /// (the text before and after the insertion point; only their ends
    /// matter). Empty when the phrase has no words.
    public static func text(for phrase: String, preceding: some StringProtocol,
                            following: some StringProtocol) -> String {
        var words = collapseSpaces(phrase.trimmingCharacters(in: .whitespacesAndNewlines))
        guard !words.isEmpty else { return "" }

        let before = preceding.last
        let atLineStart = before == nil || before!.isNewline
        let lastVisible = preceding.last { !$0.isWhitespace || $0.isNewline }
        // A sentence end may sit inside closing quotes or brackets ('." ', '.)').
        let lastMeaningful = preceding.last { !($0.isWhitespace && !$0.isNewline) && !closingQuotes.contains($0) }
        let afterSentence = lastMeaningful.map { sentenceEnds.contains($0) } ?? false

        if atLineStart || afterSentence || (lastVisible?.isNewline ?? false) {
            words = capitalizingFirstLetter(words)
        }
        let leading: String
        if atLineStart || before!.isWhitespace || isOpening(before!, in: preceding) || startsWithClosingPunctuation(words) {
            leading = ""
        } else {
            leading = " "
        }
        let trailing: String
        if let next = following.first, !next.isWhitespace, !closers.contains(next) {
            trailing = " "
        } else {
            trailing = ""
        }
        return leading + words + trailing
    }

    private static let sentenceEnds: Set<Character> = [".", "!", "?", "…", "。", "！", "？"]
    private static let closingQuotes: Set<Character> = ["\"", "'", "”", "’", "»", ")", "]"]
    private static let openers: Set<Character> = ["(", "[", "{", "\"", "'", "“", "‘", "«", "/", "-", "—"]
    private static let closers: Set<Character> = [".", ",", ";", ":", "!", "?", ")", "]", "}", "\"", "'", "”", "’", "»", "…"]

    /// An opening bracket or quote. A straight quote opens only after a
    /// space or at the start ("said \"" opens; "stop.\"" closes).
    private static func isOpening(_ character: Character, in preceding: some StringProtocol) -> Bool {
        guard openers.contains(character) else { return false }
        guard character == "\"" || character == "'" else { return true }
        let beforeQuote = preceding.dropLast().last
        return beforeQuote == nil || beforeQuote!.isWhitespace || openers.contains(beforeQuote!)
    }

    private static func startsWithClosingPunctuation(_ text: String) -> Bool {
        guard let first = text.first else { return false }
        return [",", ".", ";", ":", "!", "?", ")"].contains(first)
    }

    private static func collapseSpaces(_ text: String) -> String {
        text.split(whereSeparator: { $0 == " " || $0 == "\t" }).joined(separator: " ")
    }

    /// Capitalises a first word that starts with a lowercase letter and is
    /// all lowercase ("next" → "Next"); leaves "3 apples", "iPhone", "eBay".
    private static func capitalizingFirstLetter(_ text: String) -> String {
        guard let first = text.first, first.isLetter, first.isLowercase else { return text }
        let word = text.prefix { !$0.isWhitespace }
        guard !word.dropFirst().contains(where: \.isUppercase) else { return text }
        return first.uppercased() + text.dropFirst()
    }
}
