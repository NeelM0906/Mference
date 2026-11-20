import Foundation
import NaturalLanguage

/// Labels a phrase's language. Fluid Speech Pro detects the language while
/// decoding but the Core ML build does not report it, and a pinned language
/// cannot steer decoding (spike S1: the joint has no top-K outputs, so
/// FluidAudio is never given a `language:` hint). So a pinned language only
/// labels phrases, and automatic mode guesses from the text with the
/// system's language recogniser, limited to the 25 supported languages.
public enum PhraseLanguage {
    /// Shorter phrases are too short to guess from.
    static let minimumLetters = 12

    public static func code(for text: String, setting: SpeechLanguage) -> String? {
        if case .fixed(let code) = setting { return code }
        return guess(text)
    }

    public static func guess(_ text: String) -> String? {
        guard text.filter(\.isLetter).count >= minimumLetters else { return nil }
        let recognizer = NLLanguageRecognizer()
        recognizer.languageConstraints = SpeechLanguage.supportedCodes.map { NLLanguage(rawValue: $0) }
        recognizer.processString(text)
        guard let (language, confidence) = recognizer.languageHypotheses(withMaximum: 1).first,
              confidence >= 0.5 else { return nil }
        let code = String(language.rawValue.prefix(2))
        return SpeechLanguage.supportedCodes.contains(code) ? code : nil
    }
}
