import Foundation
import MferenceAppCore

/// Text the voice screens share: sizes, times and language names.
enum VoiceFormat {
    /// "653 MB" under a gigabyte (the setup card and Settings show whole
    /// megabytes, as the boards do), "1.2 GB" from a gigabyte up. Decimal units,
    /// like Finder.
    static func size(_ bytes: Int64) -> String {
        let megabytes = Double(bytes) / 1_000_000
        if megabytes < 999.5 {
            return "\(Int(megabytes.rounded())) MB"
        }
        return (Double(bytes) / 1_000_000_000).formatted(.number.precision(.fractionLength(1))) + " GB"
    }

    /// Meeting time as "mm:ss", or "h:mm:ss" from an hour.
    static func clock(_ time: TimeInterval) -> String {
        let total = max(0, Int(time.rounded(.down)))
        let hours = total / 3600, minutes = (total % 3600) / 60, seconds = total % 60
        if hours > 0 {
            return String(format: "%d:%02d:%02d", hours, minutes, seconds)
        }
        return String(format: "%02d:%02d", minutes, seconds)
    }

    /// The clock for VoiceOver: "12 minutes 48 seconds".
    static func spokenClock(_ time: TimeInterval) -> String {
        let total = max(0, Int(time.rounded(.down)))
        let components = DateComponents(hour: total / 3600, minute: (total % 3600) / 60, second: total % 60)
        let formatter = DateComponentsFormatter()
        formatter.unitsStyle = .full
        formatter.allowedUnits = total >= 3600 ? [.hour, .minute, .second] : [.minute, .second]
        formatter.zeroFormattingBehavior = .dropLeading
        return formatter.string(from: components) ?? clock(time)
    }

    /// A length in words: "42 min", "1 h 18 min", "under a minute".
    static func length(_ time: TimeInterval) -> String {
        let minutes = Int((time / 60).rounded())
        if minutes < 1 { return "under a minute" }
        if minutes < 60 { return "\(minutes) min" }
        let rest = minutes % 60
        return rest == 0 ? "\(minutes / 60) h" : "\(minutes / 60) h \(rest) min"
    }

    /// "English", from a BCP 47 code, in English (the app's copy language).
    static func languageName(_ code: String) -> String {
        SpeechLanguage.displayName(for: code, locale: Locale(identifier: "en"))
    }

    /// The language chip: "English · auto" while automatic (the language last
    /// detected), "German" when pinned, "Automatic" before anything is heard.
    static func languageChip(_ language: SpeechLanguage, detected: String?) -> String {
        switch language {
        case .automatic: detected.map { "\(languageName($0)) · auto" } ?? "Automatic"
        case .fixed(let code): languageName(code)
        }
    }

    /// The 25 languages by name, for pickers.
    static var languages: [(code: String, name: String)] {
        SpeechLanguage.supportedCodes.map { ($0, languageName($0)) }.sorted { $0.name < $1.name }
    }
}
