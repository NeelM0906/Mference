import Foundation

/// Without headphones the microphone hears the call. A microphone segment is
/// an echo when the call says nearly the same words over the same stretch
/// of time (spec 6.3: normalised token overlap >= 0.8), checked both ways:
///
/// - at least `threshold` (0.8) of the microphone segment's words appear in
///   the call words spoken while it ran, and
/// - at least `reverseThreshold` (0.6) of those call words appear in the
///   microphone segment (the echo is quieter, so its edges can be lost).
///
/// Call words are placed in time evenly across their segment (the Core ML
/// build reports no word times), and a call word counts as "while it ran"
/// within `timeTolerance` of the microphone segment. Microphone segments
/// under `minimumWords` are never dropped: a short reply ("Okay.") to a
/// long call phrase that happens to contain the same word stays.
public enum EchoDeduplicator {
    public static let threshold = 0.8
    public static let reverseThreshold = 0.6
    public static let minimumWords = 3
    public static let timeTolerance: TimeInterval = 0.5

    /// Whether `microphone` repeats what `call` said at the same time.
    public static func isEcho(_ microphone: TranscriptSegment, of call: [TranscriptSegment]) -> Bool {
        let mine = tokens(microphone.text)
        guard mine.count >= minimumWords else { return false }
        let from = microphone.start - timeTolerance
        let to = microphone.end + timeTolerance
        var theirs: [String: Int] = [:]
        var theirCount = 0
        for segment in call where segment.track == .call && segment.start < to && from < segment.end {
            let words = tokens(segment.text)
            guard !words.isEmpty else { continue }
            let step = max(segment.end - segment.start, 0) / Double(words.count)
            for (index, word) in words.enumerated() {
                let time = segment.start + (Double(index) + 0.5) * step
                guard time >= from, time <= to else { continue }
                theirs[word, default: 0] += 1
                theirCount += 1
            }
        }
        guard theirCount > 0 else { return false }
        var shared = 0
        for token in mine where (theirs[token] ?? 0) > 0 {
            shared += 1
            theirs[token]! -= 1
        }
        return Double(shared) / Double(mine.count) >= threshold
            && Double(shared) / Double(theirCount) >= reverseThreshold
    }

    /// Lowercased, diacritics folded, punctuation stripped.
    static func tokens(_ text: String) -> [String] {
        text.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: nil)
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }
    }
}
