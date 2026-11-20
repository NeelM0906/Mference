import Foundation

/// The text around a pending phrase marker at the moment its phrase
/// finishes: the line's text before the marker, and the character right
/// after it (nil at the end of the line).
struct TestSurroundings: Sendable, Hashable {
    /// The text from the start of the marker's line (paragraph, list item)
    /// up to the marker. Empty at the start of a line.
    var lineBefore: String
    var after: Character?

    init(lineBefore: String, after: Character?) {
        self.lineBefore = lineBefore
        self.after = after
    }
}

/// The test notes' stand-in for N1's PhraseJoiner (the editor joins phrases
/// in the app). Joins a dictated phrase to the text around it (Notes spec 4.2): a space
/// between phrases, never a double space, no leading space at the start of a
/// line, a capital letter where a sentence starts, and a space before text
/// that follows straight on.
enum TestPhraseJoiner {
    static func join(_ phrase: String, in surroundings: TestSurroundings) -> String {
        var text = phrase.split(whereSeparator: \.isWhitespace).joined(separator: " ")
        guard let first = text.first else { return "" }

        let before = surroundings.lineBefore
        let lastCharacter = before.last
        let lastWord = before.last { !$0.isWhitespace }
        let startsLine = lastWord == nil
        let opensSentence = startsLine || lastWord.map { sentenceEnds.contains($0) } == true

        if opensSentence {
            text = first.uppercased() + text.dropFirst()
        } else if startsWithPlainCapital(text) {
            // The model capitalises a phrase it heard on its own ("And
            // whether…"); mid-sentence it continues in lower case.
            text = first.lowercased() + text.dropFirst()
        }

        let needsSpaceBefore = !startsLine
            && !(lastCharacter?.isWhitespace ?? true)
            && !closing.contains(first)
            && !opening.contains(lastCharacter ?? " ")
        if needsSpaceBefore { text = " " + text }

        if let after = surroundings.after, !after.isWhitespace, !closing.contains(after) {
            text += " "
        }
        return text
    }

    private static let sentenceEnds: Set<Character> = [".", "?", "!", "…"]
    private static let closing: Set<Character> = [".", ",", ";", ":", "!", "?", "…", ")", "]", "}", "»", "”", "\"", "'"]
    private static let opening: Set<Character> = ["(", "[", "{", "«", "“"]

    /// The phrase starts with a common word the model capitalised only
    /// because it heard the phrase on its own ("And", "Whether"). Names,
    /// "I" and acronyms keep their capitals.
    private static func startsWithPlainCapital(_ text: String) -> Bool {
        let word = text.prefix { $0.isLetter }.lowercased()
        return openers.contains(word) && text.first?.isUppercase == true
    }

    /// Words the model capitalises only because a phrase started there.
    private static let openers: Set<String> = [
        "and", "but", "or", "so", "because", "then", "which", "that", "whether", "with", "without",
        "to", "for", "of", "in", "on", "at", "from", "if", "when", "while", "as", "also", "maybe",
        "the", "a", "an", "it", "this", "we", "you", "they", "he", "she", "is", "are", "was", "not",
    ]
}
