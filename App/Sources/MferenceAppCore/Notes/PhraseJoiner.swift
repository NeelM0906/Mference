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
        let afterSentence = lastVisible.map { sentenceEnds.contains($0) } ?? false

        if atLineStart || afterSentence || (lastVisible?.isNewline ?? false) {
            words = capitalizingFirstLetter(words)
        }
        let leading: String
        if atLineStart || before!.isWhitespace || openers.contains(before!) || startsWithClosingPunctuation(words) {
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
    private static let openers: Set<Character> = ["(", "[", "{", "\"", "'", "“", "‘", "«", "/", "-", "—"]
    private static let closers: Set<Character> = [".", ",", ";", ":", "!", "?", ")", "]", "}", "\"", "'", "”", "’", "»", "…"]

    private static func startsWithClosingPunctuation(_ text: String) -> Bool {
        guard let first = text.first else { return false }
        return [",", ".", ";", ":", "!", "?", ")"].contains(first)
    }

    private static func collapseSpaces(_ text: String) -> String {
        text.split(whereSeparator: { $0 == " " || $0 == "\t" }).joined(separator: " ")
    }

    private static func capitalizingFirstLetter(_ text: String) -> String {
        guard let index = text.firstIndex(where: \.isLetter) else { return text }
        return text.replacingCharacters(in: index...index, with: String(text[index]).uppercased())
    }
}
