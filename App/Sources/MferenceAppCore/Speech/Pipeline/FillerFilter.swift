import Foundation

/// Removes standalone filler words ("um", "uh", "äh", "euh" …) and immediate
/// repeated words ("I I think") from dictated text (spec 6.4). The Core ML
/// build of Fluid Speech Pro has no clean mode yet, so this deterministic
/// pass stands in for it. Whole words only: "umbrella" and "hummus" are never
/// touched. Punctuation stays tidy: a filler's own comma goes with it, a
/// sentence end it carried moves to the word before, and a sentence that
/// started with a filler starts with a capital again.
///
/// Dictation only by default; meeting and file transcripts stay verbatim.
public enum FillerFilter {
    /// The languages with a filler list (spec 6.4).
    public static let languages: Set<String> = ["en", "fr", "de", "es", "it", "pt", "nl"]

    /// Fillers per language. Words that are fillers in one language and real
    /// words in another ("em" is Portuguese for "in", "er" German and Dutch
    /// for "he"/"there") are left out, since a language guessed from a short
    /// phrase can be wrong. Callers pass a language only when the person
    /// pinned it; otherwise the universal list applies.
    static let fillers: [String: Set<String>] = [
        "en": ["um", "umm", "uh", "uhh", "uhm", "erm", "hmm", "hm", "mm", "mhm"],
        "fr": ["euh", "euhh", "heu", "hum", "hmm", "bah"],
        "de": ["äh", "ähm", "öh", "öhm", "hm", "hmm", "mhm"],
        "es": ["eh", "ehm", "emm", "mmm", "hmm"],
        "it": ["eh", "ehm", "ehh", "uhm", "mmm", "hmm"],
        "pt": ["hum", "hã", "ãh", "ahn", "uhm", "hmm"],
        "nl": ["eh", "ehm", "uh", "uhm", "euh", "hmm"],
    ]

    /// Used when the language is not known: fillers in every listed language
    /// that are no word anywhere ("um" is left out: Portuguese "a", German
    /// "at"; it goes only when English is pinned).
    static let universal: Set<String> = ["umm", "uh", "uhh", "uhm", "erm", "ehm", "euh", "äh", "ähm", "hmm", "mhm"]

    /// Doubled words that are grammatical ("that that", "nous nous"): kept.
    static let allowedRepeats: [String: Set<String>] = [
        "en": ["that", "had"],
        "fr": ["nous", "vous"],
        "de": ["die", "der", "das"],
        "nl": ["die", "dat"],
        "es": [],
        "it": [],
        "pt": [],
    ]
    static let universalAllowedRepeats: Set<String> = ["that", "had", "nous", "vous", "die", "der", "das", "dat"]

    /// Cleans `text`, read as `languageCode` (BCP 47; nil or unlisted uses
    /// the universal list).
    public static func clean(_ text: String, languageCode: String?) -> String {
        let language = languageCode.map { String($0.prefix(2)).lowercased() }
        let fillerSet = language.flatMap { fillers[$0] } ?? universal
        let repeatsAllowed = language.flatMap { allowedRepeats[$0] } ?? universalAllowedRepeats

        var output: [Word] = []
        // Set when a removed word started a sentence: the next word takes its capital.
        var capitalizeNext = false
        for token in text.split(whereSeparator: \.isWhitespace) {
            var word = Word(String(token))
            if fillerSet.contains(word.key) {
                if word.core.first?.isUppercase == true { capitalizeNext = true }
                // The filler goes with its comma; a sentence end it carried
                // moves to the word before, and the next word starts a sentence.
                if word.endsSentence {
                    if var last = output.popLast() {
                        last.trailing = last.trailing.filter { !",;:".contains($0) }
                            + word.trailing.filter { ".?!…".contains($0) }
                        output.append(last)
                    }
                    capitalizeNext = true
                }
                continue
            }
            if let last = output.last, !word.key.isEmpty, last.key == word.key, last.trailing.isEmpty,
               word.leading.isEmpty, !repeatsAllowed.contains(word.key), !word.key.allSatisfy(\.isNumber) {
                // "I I think": drop the first copy, keep its capital.
                output.removeLast()
                if last.core.first?.isUppercase == true { capitalizeNext = true }
            }
            if capitalizeNext {
                word.capitalize()
                capitalizeNext = false
            }
            output.append(word)
        }
        return output.map(\.text).joined(separator: " ")
    }

    /// A whitespace-separated token: leading punctuation, the word, trailing punctuation.
    struct Word {
        var leading: String
        var core: String
        var trailing: String

        init(_ token: String) {
            let scalars = Array(token)
            var start = 0
            var end = scalars.count
            while start < end, Word.isPunctuation(scalars[start]) { start += 1 }
            while end > start, Word.isPunctuation(scalars[end - 1]) { end -= 1 }
            leading = String(scalars[0..<start])
            core = String(scalars[start..<end])
            trailing = String(scalars[end..<scalars.count])
        }

        static func isPunctuation(_ character: Character) -> Bool {
            character.isPunctuation || character.isSymbol
        }

        var text: String { leading + core + trailing }
        /// Case-folded for comparison.
        var key: String { core.lowercased() }
        var endsSentence: Bool { trailing.contains { ".?!…".contains($0) } }

        mutating func capitalize() {
            guard let first = core.first, first.isLowercase else { return }
            core = first.uppercased() + core.dropFirst()
        }
    }
}
